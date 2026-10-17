from collections import defaultdict
from bisect import bisect_right
from dataclasses import replace
from fractions import Fraction
from statistics import median

from .notation import Note, Part, Score
from .expression import pedal_spans
from .notation_scoring import TIMING_TOLERANCE_SECONDS, assign_parts, match_notes, paired_tremolos


def ornament_window(notation, part, members):
    position = members[0].start
    voices = {(note.staff, note.voice) for note in members}
    neighbors = [note for note in part.notes if (note.staff, note.voice) in voices and "grace" not in note.articulations]
    previous = max((note.start for note in neighbors if note.start < position), default=Fraction(0))
    following = min([note.end for note in members if note.end > position] + [note.start for note in neighbors if note.start > position])
    return notation.seconds(previous), notation.seconds(following)


def reference_trill_timing(notation, tracks):
    timing = []
    for part in notation.parts:
        for note in part.notes:
            if note.trill_pitch is None:
                continue
            start, end = notation.seconds(note.start), notation.seconds(note.end)
            pitches = {note.pitch, note.trill_pitch}
            attacks = sorted(
                (event for track in tracks if not track.is_drum and track.program == note.program
                 for event in track.notes if event.pitch in pitches
                 and start - TIMING_TOLERANCE_SECONDS <= event.start < end - 1e-9),
                key=lambda event: event.start,
            )
            if len(attacks) < 3 or abs(attacks[0].start - start) > TIMING_TOLERANCE_SECONDS or any(
                right.start <= left.start or (right.pitch == left.pitch and len(pitches) > 1)
                for left, right in zip(attacks, attacks[1:])
            ):
                raise ValueError(f"Reference MIDI has no complete alternating trill for {part.name} at {start}")
            gaps = [right.start - left.start for left, right in zip(attacks, attacks[1:])]
            timing.append((note.pitch, note.trill_pitch, note.program, start, end, max(gaps), end - attacks[-1].start))
    return timing


def playback_position(notation, instant):
    position = Fraction(0)
    elapsed = 0.0
    tempo = 120.0
    for following, value in notation.tempos:
        boundary = elapsed + float(following - position) * 60 / tempo
        if boundary > instant:
            break
        position, elapsed, tempo = following, boundary, value
    return float(position) + (instant - elapsed) * tempo / 60


def tremolo_regions(part):
    regions = []
    paired_ids = set()
    for first, second, beams in paired_tremolos(part):
        paired_ids.update(id(note) for note in first + second)
        regions.append(([first, second], first[0].start, max(note.end for note in second), Fraction(1, 2 ** beams)))
    regions.extend(([[note]], note.start, note.end, Fraction(1, 2 ** (note.tremolo + note.beams))) for note in part.notes if note.tremolo and id(note) not in paired_ids)
    return regions


def playback_score(tracks):
    parts = []
    locations = {}
    for track_index, track in enumerate(tracks):
        notes = []
        for event_index, event in enumerate(track.notes):
            pitch = event.pitch
            if track.is_drum:
                if pitch in (35, 36):
                    pitch = 35
                elif pitch in (49, 57):
                    pitch = 49
            notes.append(Note(pitch, Fraction(str(event.start)), Fraction(str(event.end)), track.program, track.is_drum, "1", "1"))
            locations[id(notes[-1])] = track_index, event_index
        parts.append(Part(track.name, notes))
    return Score(parts, [(Fraction(0), 60.0)], "", ""), locations


def reference_tremolo_timing(notation, tracks):
    if not any(note.tremolo for part in notation.parts for note in part.notes):
        return []
    performance, _ = playback_score(tracks)
    routed, _ = assign_parts(performance, notation)
    timing = []
    for part, (_, fragments) in zip(notation.parts, routed, strict=True):
        for phases, beginning, ending, _ in tremolo_regions(part):
            start, end = notation.seconds(beginning), notation.seconds(ending)
            for members in phases:
                for note in members:
                    events = [event for _, event in fragments if event.pitch == note.pitch and event.percussion == note.percussion]
                    boundary = end
                    if any(following.start == ending and (following.pitch, following.percussion) == (note.pitch, note.percussion) for following in part.notes):
                        boundary = min((float(event.start) for event in events if abs(float(event.start) - end) <= TIMING_TOLERANCE_SECONDS), key=lambda instant: abs(instant - end), default=end)
                    strikes = sorted({playback_position(notation, float(event.start)) for event in events if start - TIMING_TOLERANCE_SECONDS <= float(event.start) < boundary - 1e-9})
                    if len(strikes) < 2:
                        continue
                    cadence = median(right - left for left, right in zip(strikes, strikes[1:]))
                    timing.append((note.pitch, note.percussion, note.program, start, end, cadence, (strikes[0] - float(beginning)) / cadence, strikes[-1] - strikes[0] + cadence, notation.seconds(beginning + Fraction(str(cadence))) - start))
    return timing


def compare_playback(notation, tracks, *, note_matches=None, trill_timing=None, tremolo_timing=None):
    performance, locations = playback_score(tracks)
    routing = replace(notation, parts=[replace(part, notes=part.notes + [replace(note, pitch=note.trill_pitch, trill_pitch=None) for note in part.notes if note.trill_pitch is not None]) for part in notation.parts])
    groups, extra_notes = assign_parts(performance, routing)
    details = []
    for part, (_, fragments) in zip(notation.parts, groups, strict=True):
        actual = [note for _, note in fragments]
        regions = tremolo_regions(part)
        tremolo_ids = {id(note) for phases, _, _, _ in regions for members in phases for note in members}
        arpeggios = defaultdict(list)
        for note in part.notes:
            if note.arpeggio:
                arpeggios[note.start, note.arpeggio[0]].append(note)
        arpeggios = [members for members in arpeggios.values() if len({note.pitch for note in members}) > 1]
        arpeggio_ids = {id(note) for members in arpeggios for note in members}
        grace_groups = defaultdict(list)
        for note in part.notes:
            if "grace" in note.articulations:
                grace_groups[note.start, note.staff, note.voice].append(note)
        consumed = set()
        covered = 0
        correct_programs = 0
        ornament_ids = {id(note) for note in part.notes if note.trill_pitch is not None or (note.start, note.staff, note.voice) in grace_groups}
        local_matches = {}

        def remember(source_note, indices):
            local_matches[id(source_note)] = indices
            if note_matches is not None:
                note_matches[id(source_note)] = [locations[id(actual[index])] for index in indices]

        ordinary = [note for note in part.notes if id(note) not in tremolo_ids | arpeggio_ids | ornament_ids]
        remaining = [index for index in range(len(actual)) if index not in consumed]
        anchors = [(remaining[agent_index], source_index) for agent_index, source_index in match_notes(performance, [actual[index] for index in remaining], notation, ordinary)]
        consumed.update(agent_index for agent_index, _ in anchors)
        represented = defaultdict(list)
        for agent_index, index in anchors:
            represented[ordinary[index].pitch, ordinary[index].percussion, ordinary[index].program, ordinary[index].start].append(agent_index)
        covered += sum((note.pitch, note.percussion, note.program, note.start) in represented for note in ordinary)
        correct_programs += sum(actual[agent_index].program == ordinary[source_index].program for agent_index, source_index in anchors)
        for note in ordinary:
            indices = represented.get((note.pitch, note.percussion, note.program, note.start), [])
            if indices:
                remember(note, indices)
        for (position, staff, voice), graces in grace_groups.items():
            principals = [note for note in part.notes if note.start == position and note.staff == staff and note.voice == voice and "grace" not in note.articulations]
            if not principals:
                continue
            beginning = notation.seconds(position)
            earliest, latest = ornament_window(notation, part, principals)
            selected = []
            previous = earliest
            previous_order = None
            for source_note in sorted(graces, key=lambda note: (note.grace_order or 0, note.pitch)):
                order = source_note.grace_order or 0
                eligible = [index for index, note in enumerate(actual) if index not in consumed and index not in selected and note.pitch == source_note.pitch and note.percussion == source_note.percussion and previous <= float(note.start) < latest and (previous_order is None or previous_order == order or float(note.start) > previous)]
                if not eligible:
                    break
                chosen = min(eligible, key=lambda index: actual[index].start)
                selected.append(chosen)
                previous = float(actual[chosen].start)
                previous_order = order
            if len(selected) != len(graces):
                continue
            principal_indices = []
            for source_note in principals:
                eligible = [index for index, note in enumerate(actual) if index not in consumed and index not in selected and index not in principal_indices and note.pitch == source_note.pitch and note.percussion == source_note.percussion and previous < float(note.start) < latest]
                if not eligible:
                    break
                principal_indices.append(min(eligible, key=lambda index: abs(float(actual[index].start) - beginning)))
            if len(principal_indices) != len(principals):
                continue
            principal_time = min(float(actual[index].start) for index in principal_indices)
            first_time = float(actual[selected[0]].start)
            if min(abs(first_time - beginning), abs(principal_time - beginning)) > TIMING_TOLERANCE_SECONDS:
                continue
            covered += len(graces) + len(principals)
            consumed.update(selected + principal_indices)
            ordered_graces = sorted(graces, key=lambda note: (note.grace_order or 0, note.pitch))
            correct_programs += sum(actual[index].program == note.program for index, note in zip(selected + principal_indices, ordered_graces + principals, strict=True))
            for index, note in zip(selected + principal_indices, ordered_graces + principals, strict=True):
                remember(note, [index])
        for source_note in part.notes:
            if source_note.trill_pitch is None:
                continue
            beginning = notation.seconds(source_note.start)
            ending = notation.seconds(source_note.end)
            pitches = {source_note.pitch, source_note.trill_pitch}
            selected = sorted((index for index, note in enumerate(actual) if index not in consumed and not note.percussion and note.pitch in pitches and beginning - TIMING_TOLERANCE_SECONDS <= float(note.start) < ending - 1e-9), key=lambda index: actual[index].start)
            if len(selected) < 3:
                continue
            starts = [float(actual[index].start) for index in selected]
            alternating = all(actual[left].pitch != actual[right].pitch or len(pitches) == 1 for left, right in zip(selected, selected[1:]))
            if trill_timing is None:
                complete = float(actual[selected[-1]].end) >= ending - TIMING_TOLERANCE_SECONDS and all(
                    float(actual[right].start - actual[left].end) <= TIMING_TOLERANCE_SECONDS
                    for left, right in zip(selected, selected[1:])
                )
            else:
                source_timing = [entry for entry in trill_timing if entry[:3] == (source_note.pitch, source_note.trill_pitch, source_note.program) and abs(entry[3] - beginning) <= TIMING_TOLERANCE_SECONDS and abs(entry[4] - ending) <= TIMING_TOLERANCE_SECONDS]
                complete = any(
                    ending - starts[-1] <= tail + TIMING_TOLERANCE_SECONDS
                    and all(right - left <= gap + TIMING_TOLERANCE_SECONDS for left, right in zip(starts, starts[1:]))
                    for _, _, _, _, _, gap, tail in source_timing
                )
            if abs(starts[0] - beginning) <= TIMING_TOLERANCE_SECONDS and complete and all(right > left for left, right in zip(starts, starts[1:])) and alternating:
                covered += 1
                consumed.update(selected)
                correct_programs += sum(actual[index].program == source_note.program for index in selected)
                remember(source_note, selected)
        for members in arpeggios:
            directions = {note.arpeggio[1] for note in members}
            if len(directions) != 1:
                raise ValueError("Contradictory arpeggio directions")
            pitches = sorted({note.pitch for note in members}, reverse=directions == {"down"})
            beginning = notation.seconds(members[0].start)
            earliest, latest = ornament_window(notation, part, members)
            selected = []
            previous = earliest
            for pitch in pitches:
                eligible = [index for index, note in enumerate(actual) if index not in consumed and index not in selected and note.pitch == pitch and not note.percussion and previous <= float(note.start) < latest]
                if not eligible:
                    break
                index = min(eligible, key=lambda candidate: actual[candidate].start)
                selected.append(index)
                previous = float(actual[index].start)
            complete = len(selected) == len(pitches)
            spread = float(actual[selected[-1]].start - actual[selected[0]].start) if complete else 0
            anchored = complete and min(abs(float(actual[index].start) - beginning) for index in (selected[0], selected[-1])) <= TIMING_TOLERANCE_SECONDS
            if anchored and spread > 0:
                covered += len(members)
                consumed.update(selected)
                correct_programs += sum(any(note.pitch == actual[index].pitch and note.program == actual[index].program for note in members) for index in selected)
                for note in members:
                    remember(note, [index for index in selected if actual[index].pitch == note.pitch])
        for phases, beginning, ending, step in regions:
            source_notes = [note for members in phases for note in members]
            interval = step * len(phases)
            start = notation.seconds(beginning)
            end = notation.seconds(ending)
            instants = [start]
            position = beginning
            while position < ending:
                position = min(position + interval, ending)
                instants.append(notation.seconds(position))
            intervals = [right - left for left, right in zip(instants, instants[1:])]
            identities = {(note.pitch, note.percussion) for note in source_notes}
            boundaries = dict.fromkeys(identities, end)
            for identity in identities:
                if any(note.start == ending and (note.pitch, note.percussion) == identity for note in part.notes):
                    boundaries[identity] = min((float(note.start) for note in actual if (note.pitch, note.percussion) == identity and abs(float(note.start) - end) <= TIMING_TOLERANCE_SECONDS), key=lambda instant: abs(instant - end), default=end)
            eligible = [index for index, note in enumerate(actual) if index not in consumed and (note.pitch, note.percussion) in identities and start - TIMING_TOLERANCE_SECONDS <= float(note.start) < boundaries[note.pitch, note.percussion] - 1e-9]
            valid = True
            for identity in identities:
                strikes = sorted(float(actual[index].start) for index in eligible if (actual[index].pitch, actual[index].percussion) == identity)
                if len(strikes) < 2:
                    valid = False
                    continue
                phase = next(index for index, members in enumerate(phases) if any((note.pitch, note.percussion) == identity for note in members))
                written_instants = []
                position = beginning + phase * step
                while position < ending:
                    written_instants.append(notation.seconds(position))
                    position += interval
                if len(strikes) == len(written_instants) and all(
                    abs(actual - expected) <= TIMING_TOLERANCE_SECONDS
                    for actual, expected in zip(strikes, written_instants, strict=True)
                ):
                    continue
                source_timing = [entry for entry in tremolo_timing or [] if entry[:2] == identity and entry[2] == next(note.program for note in source_notes if (note.pitch, note.percussion) == identity) and abs(entry[3] - start) <= TIMING_TOLERANCE_SECONDS and abs(entry[4] - end) <= TIMING_TOLERANCE_SECONDS]
                if source_timing:
                    positions = [playback_position(notation, instant) for instant in strikes]
                    cadence = median(right - left for left, right in zip(positions, positions[1:]))
                    if cadence <= 0 or any(round((instant - positions[0]) / cadence) != index for index, instant in enumerate(positions)):
                        valid = False
                        continue
                    complete = any(
                        notation.seconds(beginning + Fraction(str(cadence))) - start <= source_seconds + 1e-9
                        and abs(strikes[0] - notation.seconds(beginning + Fraction(str(source_phase * cadence)))) <= TIMING_TOLERANCE_SECONDS
                        and abs(notation.seconds(beginning + Fraction(str(positions[-1] - positions[0] + cadence))) - notation.seconds(beginning + Fraction(str(extent)))) <= TIMING_TOLERANCE_SECONDS
                        for _, _, _, _, _, _, source_phase, extent, source_seconds in source_timing
                    )
                    valid &= complete
                    continue
                if abs(strikes[0] - notation.seconds(beginning + phase * step)) > TIMING_TOLERANCE_SECONDS:
                    valid = False
                for left, right in zip(strikes, [*strikes[1:], end], strict=True):
                    interval_index = min(len(intervals) - 1, max(0, bisect_right(instants, left + TIMING_TOLERANCE_SECONDS) - 1))
                    if right - left > intervals[interval_index] + TIMING_TOLERANCE_SECONDS:
                        valid = False
            if valid:
                covered += len(source_notes)
                consumed.update(eligible)
                expected_programs = {(note.pitch, note.percussion): note.program for note in source_notes}
                correct_programs += sum(actual[index].program == expected_programs[actual[index].pitch, actual[index].percussion] for index in eligible)
                for note in source_notes:
                    remember(note, [index for index in eligible if (actual[index].pitch, actual[index].percussion) == (note.pitch, note.percussion)])
        glissandi = {}
        deducted = set()
        for mark in part.marks:
            if mark.kind not in {"glissando", "slide"}:
                continue
            key = mark.kind, mark.number
            if mark.value == "start":
                glissandi[key] = mark
            elif mark.value == "stop" and key in glissandi:
                beginning = glissandi.pop(key)
                first = [note for note in part.notes if note.start == beginning.position and note.staff == beginning.staff and (beginning.note_pitch is None or note.pitch == beginning.note_pitch)]
                last = [note for note in part.notes if note.start == mark.position and note.staff == mark.staff and (mark.note_pitch is None or note.pitch == mark.note_pitch)]
                if len(first) != 1 or len(last) != 1:
                    continue
                start = notation.seconds(beginning.position)
                end = notation.seconds(mark.position)
                lower, upper = sorted((first[0].pitch, last[0].pitch))
                selected = sorted((index for index, note in enumerate(actual) if index not in consumed and not note.percussion and start <= float(note.start) < end and lower <= note.pitch <= upper), key=lambda index: actual[index].start)
                direction = 1 if first[0].pitch < last[0].pitch else -1
                complete = any(lower < actual[index].pitch < upper for index in selected) and all((actual[right].pitch - actual[left].pitch) * direction >= 0 for left, right in zip(selected, selected[1:]))
                if complete:
                    consumed.update(selected)
                    correct_programs += sum(actual[index].program == first[0].program for index in selected)
                else:
                    for note in first + last:
                        identity = note.pitch, note.percussion, note.program, note.start
                        if represented.get(identity) and id(note) not in deducted:
                            covered -= 1
                            deducted.add(id(note))
                        if note_matches is not None:
                            note_matches.pop(id(note), None)
        pedals = pedal_spans(part)
        missing_pedal = 0
        for note in part.notes:
            required = [(max(beginning, note.start), ending) for beginning, ending in pedals if beginning < note.end and note.start < ending]
            if not required or id(note) not in local_matches or id(note) in deducted:
                continue
            valid = True
            for beginning, ending in required:
                start, end = notation.seconds(beginning), notation.seconds(ending)
                for index in local_matches[id(note)]:
                    track_index, event_index = locations[id(actual[index])]
                    event = tracks[track_index].notes[event_index]
                    if event.end >= end - TIMING_TOLERANCE_SECONDS:
                        continue
                    controls = sorted((control.time, control.value) for control in tracks[track_index].control_changes if control.number == 64)
                    times = [instant for instant, _ in controls]
                    midpoint = (start + end) / 2
                    for instant in (min(start + TIMING_TOLERANCE_SECONDS, midpoint), midpoint, max(end - TIMING_TOLERANCE_SECONDS, midpoint)):
                        control_index = bisect_right(times, instant) - 1
                        if control_index < 0 or controls[control_index][1] < 64:
                            valid = False
            if not valid:
                covered -= 1
                missing_pedal += 1
                if note_matches is not None:
                    note_matches.pop(id(note), None)
        details.append({
            "part": part.name,
            "notation_notes": len(part.notes),
            "covered_notation_notes": covered,
            "midi_notes": len(actual),
            "explained_midi_notes": len(consumed),
            "correct_program_notes": correct_programs,
            "missing_pedal_notes": missing_pedal,
        })
    expected = sum(entry["notation_notes"] for entry in details)
    observed = sum(entry["midi_notes"] for entry in details) + extra_notes
    explained = sum(entry["explained_midi_notes"] for entry in details)
    return {
        "notation_coverage": sum(entry["covered_notation_notes"] for entry in details) / expected if expected else 0.0,
        "midi_precision": explained / observed if observed else 0.0,
        "program_accuracy": sum(entry["correct_program_notes"] for entry in details) / explained if explained else 0.0,
        "unassigned_midi_notes": extra_notes,
        "parts": details,
    }
