"""Check conserved original music independently of replacement plugin choices."""

from collections import Counter, defaultdict, deque
import hashlib
from pathlib import Path
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


def compare_project(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = ET.parse(baseline_path).getroot()
    try:
        candidate = ET.parse(candidate_path).getroot()
    except (OSError, ET.ParseError) as exc:
        return {"passed": False, "failures": [str(exc)], "notes": 0}
    failures = []
    observations = []
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
            if before != after:
                failures.append(f"{name}: changed {direction} routing")
        for node in original.findall("Controllable"):
            control = node.get("name")
            if control not in ("solo", "mute") and not control.startswith("midicc-"):
                continue
            target = actual.find(f'Controllable[@name="{control}"]')
            if target is None or float(node.get("value")) != float(target.get("value")):
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
                old_curves, old_ordered = control_curves(original_events)
                curves, ordered = control_curves(events)
                if [message for _, message in old_ordered] != [message for _, message in ordered]:
                    failures.append(
                        f"{name}: changed ordered bank/program, switch or stateful messages"
                    )
                else:
                    offset = max(
                        (abs(left[0] - right[0]) for left, right in zip(old_ordered, ordered)),
                        default=0,
                    )
                    maximum_offset = max(maximum_offset, offset)
                    if offset > CONTROLLER_TICKS:
                        failures.append(f"{name}: ordered message displaced {offset}/1920 beats")
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
        "unresolved_active_plugins": unresolved,
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    }
