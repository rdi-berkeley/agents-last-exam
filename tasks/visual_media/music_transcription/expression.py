from collections import defaultdict
from dataclasses import replace
import re
import unicodedata


TECHNIQUES = frozenset({"tech:muted", "tech:pizzicato", "tech:harmonic", "tech:flautando", "tech:spiccato"})


def pedal_spans(part):
    active = {}
    spans = []
    events = {(mark.position, mark.value, mark.number) for mark in part.marks if mark.kind == "pedal"}
    for position, value, number in sorted(events, key=lambda event: (event[0], event[1] != "stop")):
        if value in {"stop", "discontinue", "change"}:
            if number not in active:
                raise ValueError("Unmatched pedal release")
            beginning = active.pop(number)
            if position > beginning:
                spans.append((beginning, position))
        if value in {"start", "resume", "change"}:
            if number in active:
                raise ValueError("Overlapping pedal marks")
            active[number] = position
    if active:
        ending = max(note.end for note in part.notes)
        spans.extend((beginning, ending) for beginning in active.values() if beginning < ending)
    return spans


def expressive_notes(part, notes):
    active = {}
    slurs = []
    directions = defaultdict(list)
    for mark in sorted(part.marks, key=lambda mark: (mark.position, mark.kind == "slur" and mark.value == "stop")):
        if mark.kind == "slur":
            key = mark.staff, mark.number
            if mark.value == "start":
                if key in active:
                    raise ValueError("Overlapping slurs share an identity")
                active[key] = mark
            elif mark.value == "stop":
                if key not in active:
                    alternatives = [identity for identity in active if identity[1] == mark.number]
                    if len(alternatives) != 1:
                        raise ValueError("Unmatched slur endpoint")
                    key = alternatives[0]
                beginning = active.pop(key)
                slurs.append((beginning.position, mark.position, {beginning.staff, mark.staff}))
        elif mark.kind in {"words", "technique"}:
            value = unicodedata.normalize("NFKD", mark.value.lower())
            value = " ".join(re.sub(r"[^a-z\s]", "", value).split())
            directions[mark.staff].append((mark.position, value))
    if active:
        raise ValueError("Unterminated slur")
    pedals = pedal_spans(part)
    glissandi = [mark for mark in part.marks if mark.kind in {"glissando", "slide"} and mark.value in {"start", "stop"}]
    result = []
    for note in notes:
        effects = set()
        if any(start <= note.start <= end and note.staff in staves for start, end, staves in slurs):
            effects.add("legato")
        if any(start < note.end and note.start < end for start, end in pedals):
            effects.add("pedal")
        for mark in glissandi:
            if mark.position == note.start and mark.staff == note.staff and (mark.note_pitch is None or mark.note_pitch == note.pitch):
                effects.add(f"glissando:{mark.value}")
        for position, value in directions[note.staff]:
            if position > note.start:
                break
            if value in {"pizz", "pizzicato", "plucked"}:
                effects.add("tech:pizzicato")
            elif value in {"arco", "bowed"}:
                effects.discard("tech:pizzicato")
            elif value in {"con sord", "con sordino", "muted", "mute"}:
                effects.add("tech:muted")
            elif value in {"senza sord", "senza sordino", "unmuted", "open", "without mute"}:
                effects.discard("tech:muted")
            elif value in {"flautando", "flautato"}:
                effects.add("tech:flautando")
            elif value in {"spicc", "spiccato"}:
                effects.add("tech:spiccato")
                effects.discard("legato")
            elif value == "legato":
                effects.add("legato")
                effects.discard("tech:spiccato")
            elif value in {"non legato", "detache"}:
                effects.discard("legato")
            elif value in {"ord", "ordinario", "normale", "naturale"}:
                effects.difference_update({"tech:pizzicato", "tech:flautando", "tech:spiccato", "legato"})
            elif position == note.start and value in {"harm", "harmonic", "harmonics", "natural harmonic", "natural harmonics"}:
                effects.add("tech:harmonic")
        intervals = tuple((max(start, note.start), end) for start, end in pedals if start < note.end and note.start < end)
        result.append(replace(note, articulations=note.articulations | effects, pedals=intervals) if effects or intervals else note)
    return result
