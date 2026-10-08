import io
import math
import re
import zipfile
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field, replace
from fractions import Fraction
from xml.etree import ElementTree


@dataclass
class Note:
    pitch: float
    start: Fraction
    end: Fraction
    program: int | None
    percussion: bool
    staff: str
    voice: str
    articulations: frozenset[str] = frozenset()
    tremolo: int = 0
    ties: tuple = ()
    beams: int = 0
    arpeggio: tuple[str, str] | None = None
    grace_order: int | None = None
    trill_pitch: float | None = None
    pedals: tuple[tuple[Fraction, Fraction], ...] = ()


@dataclass
class Mark:
    kind: str
    value: str
    position: Fraction
    staff: str
    number: str = "1"
    note_end: Fraction | None = None
    note_pitch: float | None = None


@dataclass
class Part:
    name: str
    notes: list[Note] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)


@dataclass
class Score:
    parts: list[Part]
    tempos: list[tuple[Fraction, float]]
    title: str
    composer: str

    def seconds(self, position):
        elapsed = 0.0
        previous = Fraction(0)
        tempo = 120.0
        for instant, value in self.tempos:
            if instant > position:
                break
            elapsed += float(instant - previous) * 60 / tempo
            previous, tempo = instant, value
        return elapsed + float(position - previous) * 60 / tempo


def group_reference_parts(score, groups):
    used = set()
    grouped = []
    for indices in groups:
        if not indices or any(type(index) is not int or not 0 <= index < len(score.parts) for index in indices):
            raise ValueError("Invalid reference instrument grouping")
        if len(indices) != len(set(indices)) or used.intersection(indices):
            raise ValueError("Reference instrument groups overlap")
        used.update(indices)
        combined = Part(" + ".join(score.parts[index].name for index in indices))
        for index in indices:
            original = score.parts[index]
            combined.notes.extend(replace(note, staff=f"{index}/{note.staff}", arpeggio=(f"{index}/{note.arpeggio[0]}", note.arpeggio[1]) if note.arpeggio else None) for note in original.notes)
            combined.marks.extend(replace(mark, staff=f"{index}/{mark.staff}", number=mark.number if mark.kind == "tremolo" else f"{index}/{mark.number}") for mark in original.marks)
        combined.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
        combined.marks.sort(key=lambda mark: (mark.position, mark.value != "stop"))
        grouped.append((min(indices), combined))
    grouped.extend((index, part) for index, part in enumerate(score.parts) if index not in used)
    return replace(score, parts=[part for _, part in sorted(grouped)])


def _read_document(content):
    if len(content) > 32 * 1024 * 1024:
        raise ValueError("MusicXML exceeds 32 MiB")
    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            container = ElementTree.fromstring(archive.read("META-INF/container.xml"))
            entry = next(node for node in container.iter() if node.tag.rsplit("}", 1)[-1] == "rootfile")
            path = entry.get("full-path")
            if archive.getinfo(path).file_size > 32 * 1024 * 1024:
                raise ValueError("Uncompressed MusicXML exceeds 32 MiB")
            content = archive.read(path)
    document = ElementTree.fromstring(content)
    for node in document.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    if document.tag == "score-timewise":
        converted = ElementTree.Element("score-partwise", document.attrib)
        converted.extend(node for node in document if node.tag != "measure")
        for definition in document.findall("part-list/score-part"):
            part = ElementTree.SubElement(converted, "part", {"id": definition.get("id")})
            for source_measure in document.findall("measure"):
                entries = [entry for entry in source_measure.findall("part") if entry.get("id") == definition.get("id")]
                if len(entries) != 1:
                    raise ValueError("Timewise measure must contain each declared part exactly once")
                measure = ElementTree.SubElement(part, "measure", source_measure.attrib)
                measure.extend(entries[0])
        document = converted
    if document.tag != "score-partwise":
        raise ValueError("Expected MusicXML score-partwise or score-timewise")
    return document


def _measure_order(measures):
    stack = []
    repeats = {}
    endings = {}
    active_ending = set()
    for index, measure in enumerate(measures):
        for barline in measure.findall("barline"):
            ending = barline.find("ending")
            if ending is not None and ending.get("type") == "start":
                active_ending = set()
                for item in ending.get("number", "").split(","):
                    bounds = [int(value) for value in re.findall(r"\d+", item)]
                    if len(bounds) == 2:
                        active_ending.update(range(bounds[0], bounds[1] + 1))
                    else:
                        active_ending.update(bounds)
            repeat = barline.find("repeat")
            if repeat is not None and repeat.get("direction") == "forward":
                stack.append(index)
            elif repeat is not None:
                repeats[index] = (stack.pop() if stack else 0, int(repeat.get("times", "2")))
        endings[index] = active_ending.copy()
        if any(ending.get("type") in ("stop", "discontinue") for ending in measure.findall("barline/ending")):
            active_ending = set()
    iterations = defaultdict(lambda: 1)
    order = []
    index = 0
    last_repeat = None
    while index < len(measures):
        enclosing = [end for end, (start, _) in repeats.items() if start <= index <= end]
        governing = min(enclosing) if enclosing else last_repeat
        iteration = iterations[governing] if governing is not None else 1
        if not endings[index] or iteration in endings[index]:
            order.append((index, iteration))
        if len(order) > 4096:
            raise ValueError("Expanded score exceeds 4096 measures")
        if index in repeats:
            start, times = repeats[index]
            if not 1 <= times <= 64:
                raise ValueError("Invalid repeat count")
            if iterations[index] < times:
                iterations[index] += 1
                for end, (inner_start, _) in repeats.items():
                    if start <= inner_start and end < index:
                        iterations[end] = 1
                index = start
                continue
            last_repeat = index
        index += 1
    return order


def _direction(event, position, divisions, marks, tempos):
    instant = position + Fraction(event.findtext("offset", "0")) / divisions
    staff = event.findtext("staff", "1")
    for item in event.findall("direction-type/*"):
        if item.tag == "dynamics":
            for dynamic in item:
                value = dynamic.tag if dynamic.tag != "other-dynamics" else "".join(dynamic.itertext()).strip().lower()
                marks.append(Mark("dynamic", value, instant, staff))
        elif item.tag in ("wedge", "pedal", "octave-shift"):
            value = item.get("type", "")
            if value != "continue":
                marks.append(Mark(item.tag, value, instant, staff, item.get("number", "1")))
        elif item.tag == "words":
            value = " ".join("".join(item.itertext()).split()).lower()
            marks.append(Mark("words", value, instant, staff))
    sound = event.find("sound")
    if sound is not None and any(name in sound.attrib for name in ("dacapo", "dalsegno", "tocoda", "fine")):
        raise ValueError("Export D.C./D.S./coda/fine passages with playback navigation expanded")
    if sound is not None and sound.get("tempo") is not None:
        tempo = float(sound.get("tempo"))
        if not math.isfinite(tempo) or tempo <= 0:
            raise ValueError("Nonpositive or non-finite score tempo")
        sound_position = position
        offset = event.find("offset")
        if sound.find("offset") is not None:
            sound_position += Fraction(sound.findtext("offset")) / divisions
        elif offset is not None and offset.get("sound", "no") == "yes":
            sound_position += Fraction(offset.text) / divisions
        tempos.append((sound_position, tempo))
    elif event.find("direction-type/metronome/per-minute") is not None:
        metronome = event.find("direction-type/metronome")
        units = {"whole": 4, "half": 2, "quarter": 1, "eighth": .5, "16th": .25, "32nd": .125}
        dots = len(metronome.findall("beat-unit-dot"))
        tempo = float(metronome.findtext("per-minute")) * units[metronome.findtext("beat-unit")]
        tempo *= 2 - 2 ** -dots
        if not math.isfinite(tempo) or tempo <= 0:
            raise ValueError("Invalid metronome mark")
        tempos.append((instant, tempo))


def parse_musicxml(content, *, tempo_override=None):
    document = _read_document(content)
    definitions = {part.get("id"): part for part in document.findall("part-list/score-part")}
    parts = []
    score_tempos = {}
    for source_part in document.findall("part"):
        definition = definitions[source_part.get("id")]
        programs = {}
        percussion = {}
        for instrument in definition.findall("midi-instrument"):
            identity = instrument.get("id")
            if instrument.find("midi-program") is not None:
                programs[identity] = int(instrument.findtext("midi-program")) - 1
                if not 0 <= programs[identity] <= 127:
                    raise ValueError("MIDI program is outside 1..128")
            if instrument.find("midi-unpitched") is not None:
                percussion[identity] = int(instrument.findtext("midi-unpitched")) - 1
        default_instrument = next(iter(programs or percussion), None)
        divisions = 1
        transpositions = {None: 0}
        key_signatures = {None: {}}
        nominal_length = None
        measures = source_part.findall("measure")
        blocks = []
        pending_spans = {}
        span_count = 0
        for measure in measures:
            cursor = Fraction(0)
            chord_start = Fraction(0)
            ending = Fraction(0)
            notes, marks, tempos = [], [], []
            chords = defaultdict(list)
            chord_group = 0
            grace_orders = defaultdict(lambda: -1)
            accidentals = defaultdict(list)
            trills = []
            for event in measure:
                if event.tag == "attributes":
                    divisions = int(event.findtext("divisions", str(divisions)))
                    if divisions <= 0:
                        raise ValueError("Nonpositive MusicXML divisions")
                    signature = event.find("time")
                    if signature is not None and signature.find("senza-misura") is None:
                        beats = signature.findall("beats")
                        units = signature.findall("beat-type")
                        nominal_length = sum(Fraction(sum(map(int, beat.text.split("+"))) * 4, int(unit.text)) for beat, unit in zip(beats, units, strict=True))
                    for transpose in event.findall("transpose"):
                        if transpose.get("number") is None:
                            transpositions.clear()
                        transpositions[transpose.get("number")] = float(transpose.findtext("chromatic")) + 12 * int(transpose.findtext("octave-change", "0"))
                    for key in event.findall("key"):
                        alterations = {}
                        if key.find("fifths") is not None:
                            fifths = int(key.findtext("fifths"))
                            if not -7 <= fifths <= 7:
                                raise ValueError("Unsupported key signature")
                            alterations = dict.fromkeys(("FCGDAEB" if fifths >= 0 else "BEADGCF")[:abs(fifths)], 1 if fifths >= 0 else -1)
                        else:
                            alterations = {step.text: float(alter.text) for step, alter in zip(key.findall("key-step"), key.findall("key-alter"), strict=True)}
                        if key.get("number") is None:
                            key_signatures.clear()
                        key_signatures[key.get("number")] = alterations
                elif event.tag in ("backup", "forward"):
                    cursor += Fraction(event.findtext("duration")) / divisions * (1 if event.tag == "forward" else -1)
                    if cursor < 0:
                        raise ValueError("MusicXML backup crosses the measure start")
                    ending = max(ending, cursor)
                elif event.tag == "direction":
                    _direction(event, cursor, divisions, marks, tempos)
                elif event.tag == "sound" and event.get("tempo"):
                    proxy = ElementTree.Element("direction")
                    proxy.append(event)
                    _direction(proxy, cursor, divisions, marks, tempos)
                elif event.tag == "note":
                    duration = Fraction(event.findtext("duration", "0")) / divisions
                    if duration < 0:
                        raise ValueError("Negative note duration")
                    start = chord_start if event.find("chord") is not None else cursor
                    if event.find("chord") is None:
                        chord_group += 1
                        chord_start = cursor
                        cursor += duration
                    ending = max(ending, cursor)
                    if event.find("rest") is not None or event.find("cue") is not None:
                        continue
                    grace = event.find("grace") is not None
                    if duration == 0 and not grace:
                        raise ValueError("Sounding note has zero duration")
                    staff, voice = event.findtext("staff", "1"), event.findtext("voice", "1")
                    instrument = event.find("instrument")
                    identity = instrument.get("id") if instrument is not None else default_instrument
                    is_percussion = event.find("unpitched") is not None
                    if is_percussion:
                        if identity not in percussion:
                            raise ValueError("Unpitched note has no declared MIDI percussion instrument")
                        pitch = percussion[identity]
                        if pitch in (35, 36):
                            pitch = 35
                        elif pitch in (49, 57):
                            pitch = 49
                    else:
                        pitch_node = event.find("pitch")
                        pitch = 12 * (int(pitch_node.findtext("octave")) + 1)
                        pitch += {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[pitch_node.findtext("step")]
                        pitch += float(pitch_node.findtext("alter", "0")) + transpositions.get(staff, transpositions[None])
                    if not math.isfinite(pitch):
                        raise ValueError("Non-finite note pitch")
                    articulations = frozenset(item.tag for item in event.findall("notations/articulations/*"))
                    if event.find("notations/technical/harmonic") is not None:
                        articulations |= {"tech:harmonic"}
                    if event.find("notations/technical/pluck") is not None:
                        articulations |= {"tech:pizzicato"}
                    if grace:
                        articulations |= {"grace"}
                    tremolo = event.find("notations/ornaments/tremolo")
                    ties = event.findall("tie") or event.findall("notations/tied")
                    tie_types = tuple((tie.get("type"), tuple(int(number) for number in re.findall(r"\d+", tie.get("time-only", "")))) for tie in ties)
                    note_type = event.findtext("type", "quarter")
                    beams = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4, "128th": 5, "256th": 6, "512th": 7, "1024th": 8}.get(note_type, 0)
                    musical_note = Note(pitch, start, start + duration, None if is_percussion else programs.get(identity), is_percussion, staff, voice, articulations, int(tremolo.text or "0") if tremolo is not None else 0, tie_types, beams)
                    if grace:
                        if event.find("chord") is None:
                            grace_orders[start, staff, voice] += 1
                        musical_note.grace_order = grace_orders[start, staff, voice]
                    if not is_percussion:
                        step = pitch_node.findtext("step")
                        octave = int(pitch_node.findtext("octave"))
                        accidentals[staff, step, octave].append((start, float(pitch_node.findtext("alter", "0"))))
                        trill = event.find("notations/ornaments/trill-mark")
                        if trill is not None:
                            upper_step = "CDEFGAB"[("CDEFGAB".index(step) + 1) % 7]
                            upper_octave = octave + (step == "B")
                            signature = key_signatures.get(staff, key_signatures.get(None, {}))
                            upper = 12 * (upper_octave + 1) + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[upper_step]
                            upper += transpositions.get(staff, transpositions.get(None, 0))
                            explicit = [mark for mark in event.findall("notations/ornaments/accidental-mark") if mark.get("placement", "above") == "above"]
                            alterations = {"natural": 0, "sharp": 1, "flat": -1, "double-sharp": 2, "sharp-sharp": 2, "flat-flat": -2}
                            if explicit:
                                if len(explicit) != 1 or explicit[0].text not in alterations:
                                    raise ValueError("Unsupported trill accidental")
                                musical_note.trill_pitch = upper + alterations[explicit[0].text]
                            elif trill.get("trill-step") is not None:
                                musical_note.trill_pitch = pitch + {"half": 1, "whole": 2, "unison": 0}[trill.get("trill-step")]
                            else:
                                trills.append((musical_note, upper_step, upper_octave, upper, signature.get(upper_step, 0)))
                    arpeggio = event.find("notations/arpeggiate")
                    if arpeggio is not None:
                        direction = arpeggio.get("direction", "up")
                        if direction not in {"up", "down"}:
                            raise ValueError("Invalid arpeggio direction")
                        musical_note.arpeggio = arpeggio.get("number", "1"), direction
                    notes.append(musical_note)
                    chords[chord_group, staff, voice].append(musical_note)
                    for item in event.findall("notations/*"):
                        if item.tag in ("slur", "glissando", "slide"):
                            marks.append(Mark(item.tag, item.get("type", ""), start, staff, item.get("number", "1"), start + duration, pitch))
                    if tremolo is not None:
                        marks.append(Mark("tremolo", tremolo.get("type", "single"), start, staff, f"{voice}:{tremolo.text}", start + duration))
            for musical_note, step, octave, upper, alteration in trills:
                preceding = [(instant, value) for instant, value in accidentals[musical_note.staff, step, octave] if instant <= musical_note.start]
                if preceding:
                    alteration = max(preceding, key=lambda entry: entry[0])[1]
                musical_note.trill_pitch = upper + alteration
            for members in chords.values():
                combined = frozenset(mark for member in members for mark in member.articulations)
                tremolo = max(member.tremolo for member in members)
                for member in members:
                    member.articulations = combined
                    member.tremolo = tremolo
            length = ending
            if nominal_length is not None and measure.get("implicit") != "yes":
                if ending <= nominal_length or abs(ending - nominal_length) < Fraction(1, 60):
                    length = nominal_length
            for mark in marks:
                if mark.kind not in {"slur", "glissando", "slide"}:
                    continue
                identity = mark.kind, mark.number
                if mark.value == "continue":
                    if identity in pending_spans:
                        mark.number = pending_spans[identity][1]
                    continue
                if mark.value not in {"start", "stop"}:
                    raise ValueError(f"Invalid {mark.kind} endpoint")
                if identity not in pending_spans:
                    span_count += 1
                    pending_spans[identity] = mark.value, f"{mark.kind}/{span_count}"
                    mark.number = pending_spans[identity][1]
                else:
                    endpoint, number = pending_spans.pop(identity)
                    if endpoint == mark.value:
                        raise ValueError(f"Overlapping {mark.kind} identifiers in document order")
                    mark.number = number
            blocks.append((length, notes, marks, tempos))
        if pending_spans:
            raise ValueError("Unmatched notation span endpoint")
        part = Part(definition.findtext("part-name", source_part.get("id")))
        position = Fraction(0)
        active_ties = {}
        part_tempos = {}
        for measure_index, iteration in _measure_order(measures):
            length, notes, marks, tempos = blocks[measure_index]
            for original in notes:
                musical_note = replace(original, start=position + original.start, end=position + original.end)
                key = musical_note.pitch, musical_note.percussion, musical_note.staff, musical_note.voice
                ties = {kind for kind, passes in original.ties if not passes or iteration in passes}
                if "stop" in ties:
                    previous = active_ties.get(key)
                    if previous is None or abs(previous.end - musical_note.start) > Fraction(1, 60):
                        raise ValueError("Unmatched or discontinuous tied note")
                    previous.end = musical_note.end
                    previous.articulations |= musical_note.articulations
                    if "start" not in ties:
                        del active_ties[key]
                else:
                    part.notes.append(musical_note)
                    if "start" in ties:
                        active_ties[key] = musical_note
            part.marks.extend(replace(mark, position=position + mark.position, note_end=position + mark.note_end if mark.note_end is not None else None) for mark in marks)
            for instant, tempo in tempos:
                instant += position
                part_tempos[instant] = tempo
            position += length
        for instant, tempo in part_tempos.items():
            if tempo_override is None and instant in score_tempos and abs(score_tempos[instant] - tempo) > 1e-6:
                raise ValueError("Contradictory simultaneous tempo marks across parts")
            score_tempos[instant] = tempo
        if active_ties:
            raise ValueError("Unterminated tie")
        part.notes.sort(key=lambda musical_note: (musical_note.start, musical_note.pitch, musical_note.end, musical_note.staff, musical_note.voice))
        part.marks.sort(key=lambda mark: (mark.position, mark.value != "stop"))
        if part.notes:
            parts.append(part)
    if not parts:
        raise ValueError("Score has no sounding notes")
    if tempo_override is not None:
        if not tempo_override or any(instant < 0 or not math.isfinite(tempo) or tempo <= 0 for instant, tempo in tempo_override):
            raise ValueError("Invalid native playback tempo map")
        score_tempos = dict(tempo_override)
    title = document.findtext("work/work-title", document.findtext("movement-title", ""))
    composer = "; ".join(creator.text or "" for creator in document.findall("identification/creator") if creator.get("type") == "composer")
    return Score(parts, sorted(score_tempos.items()), title, composer)


def dynamic_values(part, staff, sample_times):
    levels = {"pppp": 0, "ppp": 1, "pp": 2, "p": 3, "mp": 4, "mf": 5, "f": 6, "ff": 7, "fff": 8, "ffff": 9, "n": -1}
    markings = sorted((mark for mark in part.marks if mark.staff == staff), key=lambda mark: mark.position)
    static = {mark.position: float(levels[mark.value]) for mark in markings if mark.kind == "dynamic" and mark.value in levels}
    for mark in markings:
        if mark.kind == "dynamic" and mark.value in {"fp", "sfp", "sfpp"}:
            static[mark.position] = 6.0
            static[mark.position + Fraction(1, 8)] = 2.0 if mark.value == "sfpp" else 3.0
    active = {}
    wedges = []
    for mark in markings:
        if mark.kind == "wedge":
            if mark.value in {"crescendo", "diminuendo"}:
                active[mark.number] = mark
            elif mark.value == "stop" and mark.number in active:
                beginning = active.pop(mark.number)
                if beginning.position < mark.position:
                    wedges.append((beginning.position, mark.position, 1 if beginning.value == "crescendo" else -1))
        elif mark.kind == "words" and re.fullmatch(r"(?:poco a poco )?(cresc(?:endo)?|dim(?:inuendo)?|decresc(?:endo)?)[.]?", mark.value):
            following = [instant for instant in static if instant > mark.position]
            if following:
                wedges.append((mark.position, min(following), 1 if "cresc" in mark.value and "decresc" not in mark.value else -1))
    if active:
        raise ValueError("Unterminated dynamic hairpin")
    segments = []
    for start, end, direction in sorted(wedges):
        preceding = [instant for instant in static if instant <= start]
        initial = static[max(preceding)] if preceding else 5.0
        targets = [instant for instant in static if abs(instant - end) <= Fraction(1, 60)]
        final = static[min(targets, key=lambda instant: abs(instant - end))] if targets else initial + direction
        if (final - initial) * direction <= 0:
            final = initial + direction
        segments.append((start, end, initial, final))
        static.setdefault(start, initial)
        static.setdefault(end, final)
    instants = sorted(static)
    values = []
    for instant in sample_times:
        index = bisect_right(instants, instant) - 1
        value = static[instants[index]] if index >= 0 else None
        for start, end, initial, final in segments:
            if start <= instant < end:
                value = initial + float((instant - start) / (end - start)) * (final - initial)
        values.append(value)
    return values
