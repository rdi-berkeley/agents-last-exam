"""Check conserved original music independently of replacement plugin choices."""

from collections import Counter, defaultdict, deque
import hashlib
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mido


SIMPLE_SETTERS = {1, 2, 7, 10, 11}
CONTROLLER_TICKS = 5


def midi_events(path):
    midi = mido.MidiFile(path)
    events = []
    tick = 0
    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        if not message.is_meta:
            fields = message.dict()
            fields.pop("time")
            events.append((round(tick * 1920 / midi.ticks_per_beat), fields))
    return events


def source_path(project, source):
    name = source.get("name")
    if Path(name).name != name:
        raise ValueError("External MIDI sources must be collected into the delivered project")
    matches = list(project.glob("interchange/*/midifiles/" + name))
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous MIDI source: {name}")
    return matches[0]


def note_events(events):
    active = defaultdict(deque)
    notes = Counter()
    for tick, message in events:
        kind = message["type"]
        if kind not in ("note_on", "note_off"):
            continue
        key = (message["channel"], message["note"])
        if kind == "note_on" and message["velocity"]:
            active[key].append((tick, message["velocity"]))
        elif active[key]:
            start, velocity = active[key].popleft()
            notes[(start, tick, *key, velocity, message["velocity"])] += 1
    if any(active.values()):
        raise ValueError("Unterminated MIDI notes")
    return notes


def control_curves(events):
    curves = defaultdict(list)
    ordered = []
    for tick, message in events:
        if message["type"] in ("note_on", "note_off"):
            continue
        if message["type"] == "control_change" and message["control"] in SIMPLE_SETTERS:
            key = (message["channel"], message["control"])
            points = curves[key]
            if not points or points[-1][1] != message["value"]:
                points.append((tick, message["value"]))
        else:
            ordered.append((tick, message))
    return curves, ordered


def initial_midi_setters(tree, route, project):
    playlist = tree.find(f'Playlists/Playlist[@id="{route.get("midi-playlist")}"]')
    if playlist is None:
        return {}
    sources = {source.get("id"): source for source in tree.findall("Sources/Source")}
    timeline = []
    try:
        for region_index, region in enumerate(playlist.findall("Region")):
            if region.get("muted") == "1":
                continue
            start = float(region.attrib["start-beats"]) * 1920
            end = start + float(region.attrib["length-beats"]) * 1920
            position = float(region.attrib["beat"]) * 1920
            events = midi_events(source_path(project, sources[region.get("source-0")]))
            for order, (tick, message) in enumerate(events):
                if start <= tick < end:
                    timeline.append((position + tick - start, region_index, order, message))
    except (KeyError, ValueError, OSError, EOFError):
        return {}
    first_notes = {}
    for tick, region_index, order, message in sorted(timeline, key=lambda entry: entry[:3]):
        if message["type"] == "note_on" and message["velocity"]:
            first_notes.setdefault(message["channel"], (tick, region_index, order))
    setters = defaultdict(list)
    for tick, region_index, order, message in timeline:
        if message["type"] != "control_change" or message["control"] not in SIMPLE_SETTERS:
            continue
        note = first_notes.get(message["channel"])
        if note is not None and (
            tick < note[0] or (tick == note[0] and region_index == note[1] and order < note[2])
        ):
            setters[(message["channel"], message["control"])].append(
                (tick, region_index, order, message["value"])
            )
    result = {}
    for key, points in setters.items():
        last_tick = max(point[0] for point in points)
        last = [point for point in points if point[0] == last_tick]
        if len({point[1] for point in last}) == 1:
            result[key] = max(last, key=lambda point: point[2])[3]
    return result


def ordered_controls_preserved(original_events, events):
    original = control_curves(original_events)[1]
    first_notes = {}
    for index, (tick, message) in enumerate(events):
        if message["type"] == "note_on" and message["velocity"]:
            first_notes.setdefault(message["channel"], (tick, index))
    consumed = 0
    maximum_offset = 0
    initializers = []
    for index, (tick, message) in enumerate(events):
        if message["type"] in ("note_on", "note_off") or (
            message["type"] == "control_change" and message["control"] in SIMPLE_SETTERS
        ):
            continue
        if (
            consumed < len(original)
            and message == original[consumed][1]
            and abs(tick - original[consumed][0]) <= CONTROLLER_TICKS
        ):
            maximum_offset = max(maximum_offset, abs(tick - original[consumed][0]))
            consumed += 1
            continue
        first_note = first_notes.get(message.get("channel"))
        patch_initializer = message["type"] == "program_change" or (
            message["type"] == "control_change" and message["control"] in (0, 32)
        )
        if not patch_initializer or first_note is None or (tick, index) >= first_note:
            return False, maximum_offset, initializers
        initializers.append((tick, message))
    return consumed == len(original), maximum_offset, initializers


def compare_project(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = ET.parse(baseline_path).getroot()
    try:
        candidate = ET.parse(candidate_path).getroot()
    except (OSError, ET.ParseError) as exc:
        return {"passed": False, "failures": [str(exc)], "notes": 0}
    failures = []
    observations = []
    instrument_initializers = []
    maximum_offset = 0
    totals = Counter()
    for tag in ("Tempo", "Meter"):
        original = [entry.attrib for entry in baseline.findall("TempoMap/" + tag)]
        actual = [entry.attrib for entry in candidate.findall("TempoMap/" + tag)]
        if original != actual:
            failures.append(f"Changed {tag} map")
    original_routes = {route.get("name"): route for route in baseline.findall("Routes/Route")}
    routes = {route.get("name"): route for route in candidate.findall("Routes/Route")}
    if original_routes.keys() != routes.keys():
        failures.append("Changed original track/bus inventory")
    original_sources = {source.get("id"): source for source in baseline.findall("Sources/Source")}
    sources = {source.get("id"): source for source in candidate.findall("Sources/Source")}
    for name, original in original_routes.items():
        actual = routes.get(name)
        if actual is None:
            continue
        for key in ("playback-channel-mode", "playback-channel-mask"):
            if original.get(key) != actual.get(key):
                failures.append(f"{name}: changed {key}")
        for direction in ("Input", "Output"):
            expression = f'IO[@direction="{direction}"]/Port/Connection'
            before = sorted(node.get("other") for node in original.findall(expression))
            after = sorted(node.get("other") for node in actual.findall(expression))
            original_info = original.find("PresentationInfo")
            actual_info = actual.find("PresentationInfo")
            if direction == "Output" and all(
                info is not None and "MasterOut" in info.get("flags", "").split(",")
                for info in (original_info, actual_info)
            ):
                before = [
                    target for target in before
                    if not re.fullmatch(r"system:playback_[1-9][0-9]*", target or "")
                ]
                after = [
                    target for target in after
                    if not re.fullmatch(r"system:playback_[1-9][0-9]*", target or "")
                ]
            if before != after:
                failures.append(f"{name}: changed {direction} routing")
        for node in original.findall("Controllable"):
            control = node.get("name")
            if control not in ("solo", "mute") and not control.startswith("midicc-"):
                continue
            target = actual.find(f'Controllable[@name="{control}"]')
            if target is None or float(node.get("value")) != float(target.get("value")):
                if target is not None and control.startswith("midicc-"):
                    try:
                        key = tuple(map(int, control.split("-")[1:]))
                        before = initial_midi_setters(baseline, original, baseline_path.parent)
                        after = initial_midi_setters(candidate, actual, candidate_path.parent)
                        if key in before and before[key] == after.get(key):
                            continue
                    except ValueError:
                        pass
                failures.append(f"{name}: changed {control} initial state")
        for node in original.findall(".//AutomationList"):
            if node.get("state") in (None, "Off"):
                continue
            targets = actual.findall(f'.//AutomationList[@id="{node.get("id")}"]')
            if len(targets) != 1 or ET.tostring(node).strip() != ET.tostring(targets[0]).strip():
                observations.append(
                    f"{name}: active automation needs equivalent-adapter verification"
                )
        playlist_id = original.get("midi-playlist")
        if playlist_id is None:
            continue
        original_playlist = baseline.find(f'Playlists/Playlist[@id="{playlist_id}"]')
        playlist = candidate.find(f'Playlists/Playlist[@id="{actual.get("midi-playlist")}"]')
        if playlist is None:
            failures.append(f"{name}: missing editable playlist")
            continue
        before = sorted(
            original_playlist.findall("Region"),
            key=lambda entry: (entry.get("position"), entry.get("name")),
        )
        after = sorted(
            playlist.findall("Region"), key=lambda entry: (entry.get("position"), entry.get("name"))
        )
        if len(before) != len(after):
            failures.append(f"{name}: changed region inventory")
            continue
        for source_region, region in zip(before, after, strict=True):
            totals["regions"] += 1
            for field in ("muted", "position", "length", "start", "start-beats", "length-beats"):
                expected_value, actual_value = source_region.get(field), region.get(field)
                same = expected_value == actual_value
                if field != "muted" and expected_value is not None and actual_value is not None:
                    same = float(expected_value) == float(actual_value)
                if not same:
                    failures.append(f"{name}: changed region {field}")
            zero_notes = source_region.find("Extra/OriginalCubaseZeroVelocityNotes")
            if zero_notes is not None:
                counterpart = region.find("Extra/OriginalCubaseZeroVelocityNotes")
                if counterpart is None or [entry.attrib for entry in zero_notes] != [
                    entry.attrib for entry in counterpart
                ]:
                    failures.append(f"{name}: lost silent original velocity metadata")
            try:
                original_path = source_path(
                    baseline_path.parent, original_sources[source_region.get("source-0")]
                )
                path = source_path(candidate_path.parent, sources[region.get("source-0")])
                original_events, events = midi_events(original_path), midi_events(path)
                original_notes, notes = note_events(original_events), note_events(events)
                totals["notes"] += notes.total()
                if region.get("muted") == "1":
                    totals["muted_notes"] += notes.total()
                if original_notes != notes:
                    failures.append(
                        f"{name}: notes differ (missing {(original_notes - notes).total()}, extra {(notes - original_notes).total()})"
                    )
                old_curves, _ = control_curves(original_events)
                curves, _ = control_curves(events)
                preserved, offset, initializers = ordered_controls_preserved(original_events, events)
                if not preserved:
                    failures.append(
                        f"{name}: changed ordered bank/program, switch or stateful messages"
                    )
                else:
                    maximum_offset = max(maximum_offset, offset)
                    if offset > CONTROLLER_TICKS:
                        failures.append(f"{name}: ordered message displaced {offset}/1920 beats")
                    if initializers:
                        instrument_initializers.append({"route": name, "events": initializers})
                if old_curves.keys() != curves.keys():
                    failures.append(f"{name}: changed controller inventory")
                for key, points in old_curves.items():
                    other = curves.get(key, [])
                    if [value for _, value in points] != [value for _, value in other]:
                        failures.append(f"{name}: CC{key} effective values changed")
                    else:
                        offset = max(
                            (abs(left[0] - right[0]) for left, right in zip(points, other)),
                            default=0,
                        )
                        maximum_offset = max(maximum_offset, offset)
                        if offset > CONTROLLER_TICKS:
                            failures.append(f"{name}: CC{key} change displaced {offset}/1920 beats")
            except (ValueError, OSError, KeyError, EOFError) as exc:
                failures.append(f"{name}: {exc}")
    unresolved = [
        processor.get("name")
        for processor in candidate.findall("Routes/Route/Processor")
        if processor.get("active") == "1"
        and processor.find("OriginalCubasePlugin") is not None
        and processor.get("name", "").startswith("Unavailable:")
        and not (
            processor.find("OriginalCubasePlugin/Equalizer") is not None
            and all(
                band.find("Enabled").get("value") == "false"
                for band in processor.findall("OriginalCubasePlugin/Equalizer/Band")
            )
        )
    ]
    return {
        "passed": not failures and not observations,
        "failures": failures,
        "unsupported": observations,
        **totals,
        "routes": len(routes),
        "maximum_controller_offset_ticks": maximum_offset,
        "instrument_initializers": instrument_initializers,
        "unresolved_active_plugins": unresolved,
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    }
