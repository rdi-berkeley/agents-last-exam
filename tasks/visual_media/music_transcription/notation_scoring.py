from collections import defaultdict
from dataclasses import replace
from fractions import Fraction

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, spearmanr

from .notation import Mark, dynamic_values
from .expression import TECHNIQUES, expressive_notes


TIMING_TOLERANCE_SECONDS = 0.040


def paired_tremolos(part):
    paired = {}
    spans = []
    for mark in part.marks:
        if mark.kind != "tremolo" or mark.value not in {"start", "stop"}:
            continue
        key = mark.staff, mark.number
        if mark.value == "start":
            if key in paired:
                raise ValueError("Overlapping paired tremolo")
            paired[key] = mark
        elif key in paired:
            start = paired.pop(key)
            voice, beams = mark.number.split(":")
            first = [note for note in part.notes if note.start == start.position and note.staff == mark.staff and note.voice == voice]
            second = [note for note in part.notes if note.start == mark.position and note.staff == mark.staff and note.voice == voice]
            if not first or not second or start.position >= mark.position:
                raise ValueError("Invalid paired tremolo endpoints")
            spans.append((first, second, int(beams)))
        else:
            raise ValueError("Unmatched paired tremolo stop")
    if paired:
        raise ValueError("Unterminated paired tremolo")
    return spans


def expanded_notes(part):
    spans = paired_tremolos(part)
    consumed = set()
    result = []
    for first, second, beams in spans:
        consumed.update(id(note) for note in first + second)
        start = first[0].start
        end = max(note.end for note in second)
        step = Fraction(1, 2 ** beams)
        instant = start
        strike = 0
        while instant < end:
            for note in first if strike % 2 == 0 else second:
                result.append(replace(note, start=instant, end=min(instant + step, end), tremolo=0))
            instant += step
            strike += 1
    for note in part.notes:
        if id(note) in consumed:
            continue
        if note.tremolo:
            step = Fraction(1, 2 ** (note.tremolo + note.beams))
            instant = note.start
            while instant < note.end:
                result.append(replace(note, start=instant, end=min(instant + step, note.end), tremolo=0))
                instant += step
        else:
            result.append(note)
    if len(result) > 100000:
        raise ValueError("Expanded part exceeds 100,000 notes")
    return expressive_notes(part, sorted(result, key=lambda note: (note.start, note.pitch, note.end)))


def match_notes(agent_score, agent_notes, reference_score, reference_notes, *, rhythm=False, preference=None):
    if not agent_notes or not reference_notes:
        return []
    tolerance = TIMING_TOLERANCE_SECONDS
    agent_starts = np.array([agent_score.seconds(note.start) for note in agent_notes])
    reference_starts = np.array([reference_score.seconds(note.start) for note in reference_notes])
    errors = abs(reference_starts[:, None] - agent_starts)
    eligible = errors <= tolerance + 1e-9
    eligible &= np.array([note.percussion for note in reference_notes])[:, None] == np.array([note.percussion for note in agent_notes])
    if rhythm:
        duration_errors = abs(
            np.array([reference_score.seconds(note.end) - reference_score.seconds(note.start) for note in reference_notes])[:, None]
            - np.array([agent_score.seconds(note.end) - agent_score.seconds(note.start) for note in agent_notes])
        )
        errors = np.maximum(errors, duration_errors)
        eligible &= duration_errors <= tolerance + 1e-9
        duration_marks = {"staccato", "staccatissimo", "tenuto", "detached-legato", "grace", "legato", "glissando:start", "glissando:stop"}
        reference_articulations = [(note.articulations & duration_marks, note.arpeggio[1] if note.arpeggio else None, note.grace_order, note.trill_pitch - note.pitch if note.trill_pitch is not None else None) for note in reference_notes]
        agent_articulations = [(note.articulations & duration_marks, note.arpeggio[1] if note.arpeggio else None, note.grace_order, note.trill_pitch - note.pitch if note.trill_pitch is not None else None) for note in agent_notes]
        eligible &= np.array([[expected == actual for actual in agent_articulations] for expected in reference_articulations])
        if any(note.pedals for note in reference_notes + agent_notes):
            pedal_intervals = []
            for score, notes in ((reference_score, reference_notes), (agent_score, agent_notes)):
                intervals = []
                for note in notes:
                    boundaries = [(score.seconds(start), score.seconds(end)) for start, end in note.pedals]
                    intervals.append([(start, end) for start, end in boundaries if end - score.seconds(note.start) > tolerance + 1e-9])
                pedal_intervals.append(intervals)
            for reference_index, agent_index in zip(*np.nonzero(eligible), strict=True):
                expected = pedal_intervals[0][reference_index]
                actual = pedal_intervals[1][agent_index]
                eligible[reference_index, agent_index] = len(expected) == len(actual) and all(
                    abs(expected_start - actual_start) <= tolerance + 1e-9
                    and abs(expected_end - actual_end) <= tolerance + 1e-9
                    for (expected_start, expected_end), (actual_start, actual_end) in zip(expected, actual)
                )
    else:
        eligible &= np.array([note.pitch for note in reference_notes])[:, None] == np.array([note.pitch for note in agent_notes])
    quality = 1 / (1 + errors / tolerance)
    if preference is not None:
        if preference.shape != eligible.shape or not np.isfinite(preference).all() or np.any((preference < 0) | (preference > 1)):
            raise ValueError("Invalid note pairing preference")
        quality = (quality + preference) / 2
    weights = eligible * (1 + quality / (min(eligible.shape) + 1))
    rows, columns = linear_sum_assignment(weights, maximize=True)
    return [(int(column), int(row)) for row, column in zip(rows, columns, strict=True) if eligible[row, column]]


def assign_parts(agent_score, reference_score):
    references = [expanded_notes(part) for part in reference_score.parts]
    fragments = []
    for part_index, part in enumerate(agent_score.parts):
        voices = defaultdict(list)
        for note in expanded_notes(part):
            voices[note.staff, note.voice].append(note)
        fragments.extend((part_index, notes) for notes in voices.values())
    similarities = np.zeros((len(references), len(fragments)))
    for reference_index, notes in enumerate(references):
        for fragment_index, (_, candidate) in enumerate(fragments):
            matches = match_notes(agent_score, candidate, reference_score, notes)
            program_matches = sum(candidate[agent_index].program == notes[source_index].program for agent_index, source_index in matches)
            similarities[reference_index, fragment_index] = len(matches) + program_matches / (len(candidate) + len(notes) + 1)
            if not matches:
                rhythm_matches = match_notes(agent_score, candidate, reference_score, notes, rhythm=True)
                similarities[reference_index, fragment_index] = len(rhythm_matches) / (2 * (len(candidate) + len(notes) + 1))
    rows, columns = linear_sum_assignment(similarities, maximize=True)
    selected = defaultdict(list)
    assigned = set()
    for reference_index, fragment_index in zip(rows, columns, strict=True):
        if similarities[reference_index, fragment_index] > 0:
            selected[int(reference_index)].append(int(fragment_index))
            assigned.add(int(fragment_index))
    remaining = {}
    for reference_index, notes in enumerate(references):
        candidates = [note for fragment_index in selected[reference_index] for note in fragments[fragment_index][1]]
        matches = match_notes(agent_score, candidates, reference_score, notes)
        if not matches:
            matches = match_notes(agent_score, candidates, reference_score, notes, rhythm=True)
        matched = {source_index for _, source_index in matches}
        remaining[reference_index] = [note for source_index, note in enumerate(notes) if source_index not in matched]
    for fragment_index, (_, candidate) in enumerate(fragments):
        if fragment_index in assigned:
            continue
        alternatives = []
        for reference_index, notes in remaining.items():
            matches = match_notes(agent_score, candidate, reference_score, notes)
            pitch_count = len(matches)
            if not matches:
                matches = match_notes(agent_score, candidate, reference_score, notes, rhythm=True)
            program_matches = sum(candidate[agent_index].program == notes[source_index].program for agent_index, source_index in matches)
            alternatives.append((pitch_count, len(matches), program_matches, reference_index, matches))
        _, count, _, reference_index, matches = max(alternatives)
        if count:
            selected[reference_index].append(fragment_index)
            assigned.add(fragment_index)
            consumed = {source_index for _, source_index in matches}
            remaining[reference_index] = [note for source_index, note in enumerate(remaining[reference_index]) if source_index not in consumed]
    groups = []
    for reference_index, notes in enumerate(references):
        candidate = [(part_index, note) for fragment_index in selected[reference_index] for part_index, fragment in [fragments[fragment_index]] for note in fragment]
        groups.append((notes, candidate))
    extra_notes = sum(len(notes) for fragment_index, (_, notes) in enumerate(fragments) if fragment_index not in assigned)
    return groups, extra_notes


def dynamics_similarity(agent_score, candidate, reference_score, reference_part, reference_notes, matches):
    if not candidate or not matches:
        return 0.0
    fractions = (Fraction(0), Fraction(1, 2), Fraction(9, 10))
    source_samples = defaultdict(list)
    candidate_samples = defaultdict(list)
    for source_index, note in enumerate(reference_notes):
        source_samples[note.staff].extend((source_index, sample_index, note.start + (note.end - note.start) * fraction) for sample_index, fraction in enumerate(fractions))
    for agent_index, (part_index, actual_note) in enumerate(candidate):
        candidate_samples[part_index, actual_note.staff].extend((agent_index, sample_index, actual_note.start + (actual_note.end - actual_note.start) * fraction) for sample_index, fraction in enumerate(fractions))
    source_values = {}
    actual_values = {}
    for staff, samples in source_samples.items():
        values = dynamic_values(reference_part, staff, [entry[2] for entry in samples])
        source_values.update((entry[:2], value) for entry, value in zip(samples, values, strict=True))
    for (part_index, staff), samples in candidate_samples.items():
        values = dynamic_values(agent_score.parts[part_index], staff, [entry[2] for entry in samples])
        actual_values.update((entry[:2], value) for entry, value in zip(samples, values, strict=True))
    requested = sum(value is not None for value in source_values.values())
    source_rank = {}
    actual_rank = {}
    for values, ranked in ((source_values, source_rank), (actual_values, actual_rank)):
        present = [(key, value) for key, value in values.items() if value is not None]
        if present:
            ranked.update((key, rank / len(present)) for (key, _), rank in zip(present, rankdata([value for _, value in present]), strict=True))
    preference = np.zeros((len(reference_notes), len(candidate)))
    for sample_index in range(len(fractions)):
        expected_ranks = np.array([source_rank.get((index, sample_index), np.nan) for index in range(len(reference_notes))])
        actual_ranks = np.array([actual_rank.get((index, sample_index), np.nan) for index in range(len(candidate))])
        preference += np.nan_to_num(1 - abs(expected_ranks[:, None] - actual_ranks), nan=0) / len(fractions)
    matches = match_notes(agent_score, [note for _, note in candidate], reference_score, reference_notes, preference=np.clip(preference, 0, 1))
    pairs = [
        (source_values[source_index, sample_index], actual_values[agent_index, sample_index])
        for agent_index, source_index in matches
        for sample_index in range(len(fractions))
        if source_values[source_index, sample_index] is not None and actual_values[agent_index, sample_index] is not None
    ]
    expected = [pair[0] for pair in pairs]
    actual = [pair[1] for pair in pairs]
    covered = len(pairs)
    if requested == 0:
        return 2 * len(matches) / (len(candidate) + len(reference_notes))
    if not expected:
        return 0.0
    if max(expected) - min(expected) < 1e-9:
        similarity = 1.0 if max(actual) - min(actual) < 1e-9 else 0.0
    elif max(actual) - min(actual) < 1e-9:
        similarity = 0.0
    elif np.allclose(expected, actual, rtol=0, atol=0.02):
        similarity = 1.0
    else:
        correlation = float(spearmanr(expected, actual).statistic)
        similarity = max(0.0, correlation) if np.isfinite(correlation) else 0.0
    return similarity * covered / requested * min(1.0, 2 * len(reference_notes) / (len(candidate) + len(reference_notes)))


def collapse_explicit_trills(agent_score, reference_score):
    trills = [note for part in reference_score.parts for note in part.notes if note.trill_pitch is not None]
    if not trills:
        return agent_score
    normalized = []
    for part in agent_score.parts:
        notes = list(part.notes)
        for trill in trills:
            start, end = reference_score.seconds(trill.start), reference_score.seconds(trill.end)
            groups = defaultdict(list)
            for note in notes:
                if note.trill_pitch is None and not note.percussion and note.program == trill.program and note.pitch in {trill.pitch, trill.trill_pitch} and start - TIMING_TOLERANCE_SECONDS <= agent_score.seconds(note.start) < end - 1e-9:
                    groups[note.staff, note.voice].append(note)
            for members in groups.values():
                members.sort(key=lambda note: note.start)
                if len(members) < 3 or len({note.program for note in members}) != 1:
                    continue
                if abs(agent_score.seconds(members[0].start) - start) > TIMING_TOLERANCE_SECONDS or abs(agent_score.seconds(members[-1].end) - end) > TIMING_TOLERANCE_SECONDS:
                    continue
                if any(right.start <= left.start or abs(agent_score.seconds(right.start) - agent_score.seconds(left.end)) > TIMING_TOLERANCE_SECONDS or left.pitch == right.pitch for left, right in zip(members, members[1:])):
                    continue
                identities = {id(note) for note in members}
                notes = [note for note in notes if id(note) not in identities]
                notes.append(replace(members[0], pitch=trill.pitch, end=members[-1].end, trill_pitch=trill.trill_pitch))
        normalized.append(replace(part, notes=sorted(notes, key=lambda note: (note.start, note.pitch, note.end))))
    return replace(agent_score, parts=normalized)


def collapse_explicit_glissandi(agent_score, reference_score):
    spans = []
    for part in reference_score.parts:
        active = {}
        for mark in part.marks:
            if mark.kind not in {"glissando", "slide"}:
                continue
            if mark.value == "start":
                active[mark.kind, mark.number] = mark
            elif mark.value == "stop":
                beginning = active.pop((mark.kind, mark.number), None)
                if beginning is not None and beginning.note_pitch is not None and mark.note_pitch is not None:
                    programs = {note.program for note in part.notes if note.start == beginning.position and note.pitch == beginning.note_pitch and note.staff == beginning.staff}
                    if len(programs) == 1:
                        spans.append((beginning, mark, next(iter(programs))))
    if not spans:
        return agent_score
    normalized = []
    for part in agent_score.parts:
        notes, marks = list(part.notes), list(part.marks)
        for beginning, ending, program in spans:
            start, end = reference_score.seconds(beginning.position), reference_score.seconds(ending.position)
            lower, upper = sorted((beginning.note_pitch, ending.note_pitch))
            direction = 1 if beginning.note_pitch < ending.note_pitch else -1
            groups = defaultdict(list)
            for note in notes:
                if not note.percussion and note.program == program and lower <= note.pitch <= upper and start - TIMING_TOLERANCE_SECONDS <= agent_score.seconds(note.start) <= end + TIMING_TOLERANCE_SECONDS:
                    groups[note.voice, note.program].append(note)
            for members in groups.values():
                members.sort(key=lambda note: note.start)
                if len(members) < 3 or members[0].pitch != beginning.note_pitch or members[-1].pitch != ending.note_pitch:
                    continue
                if abs(agent_score.seconds(members[0].start) - start) > TIMING_TOLERANCE_SECONDS or abs(agent_score.seconds(members[-1].start) - end) > TIMING_TOLERANCE_SECONDS:
                    continue
                if any((right.pitch - left.pitch) * direction <= 0 or right.start <= left.start or abs(agent_score.seconds(right.start) - agent_score.seconds(left.end)) > TIMING_TOLERANCE_SECONDS for left, right in zip(members, members[1:])):
                    continue
                first, last = members[0], members[-1]
                if any(mark.kind in {"glissando", "slide"} and mark.position == first.start and mark.staff == first.staff for mark in marks):
                    continue
                consumed = {id(note) for note in members[:-1]}
                notes = [note for note in notes if id(note) not in consumed]
                notes.append(replace(first, end=last.start))
                number = f"expanded-glissando/{len(marks)}"
                marks.extend((Mark("glissando", "start", first.start, first.staff, number, last.start, first.pitch), Mark("glissando", "stop", last.start, last.staff, number, last.end, last.pitch)))
        normalized.append(replace(part, notes=sorted(notes, key=lambda note: (note.start, note.pitch, note.end)), marks=sorted(marks, key=lambda mark: mark.position)))
    return replace(agent_score, parts=normalized)


def normalize_ornaments(agent_score, reference_score):
    agent_score = collapse_explicit_trills(agent_score, reference_score)
    return collapse_explicit_glissandi(agent_score, reference_score)


def compare_notation(agent_score, reference_score):
    agent_score = normalize_ornaments(agent_score, reference_score)
    groups, extras = assign_parts(agent_score, reference_score)
    details = []
    for reference_part, (notes, candidate) in zip(reference_score.parts, groups, strict=True):
        agent_notes = [note for _, note in candidate]
        pitch_matches = match_notes(agent_score, agent_notes, reference_score, notes)
        rhythm_matches = match_notes(agent_score, agent_notes, reference_score, notes, rhythm=True)
        denominator = len(notes) + len(agent_notes)
        instrument_preference = np.array([
            [expected.program == actual.program and expected.articulations & TECHNIQUES == actual.articulations & TECHNIQUES for actual in agent_notes]
            for expected in notes
        ], dtype=float).reshape((len(notes), len(agent_notes)))
        instrument_pairs = match_notes(agent_score, agent_notes, reference_score, notes, preference=instrument_preference)
        program_matches = sum(
            notes[source_index].program == agent_notes[agent_index].program
            and notes[source_index].percussion == agent_notes[agent_index].percussion
            and notes[source_index].articulations & TECHNIQUES == agent_notes[agent_index].articulations & TECHNIQUES
            for agent_index, source_index in instrument_pairs
        )
        force_marks = {"accent", "strong-accent", "soft-accent", "stress", "unstress"}
        reference_forces = sum(len(note.articulations & force_marks) for note in notes)
        agent_forces = sum(len(note.articulations & force_marks) for note in agent_notes)
        force_matches = sum(len(notes[source_index].articulations & agent_notes[agent_index].articulations & force_marks) for agent_index, source_index in pitch_matches)
        force_score = 2 * force_matches / (reference_forces + agent_forces) if reference_forces + agent_forces else 1.0
        dynamics = dynamics_similarity(agent_score, candidate, reference_score, reference_part, notes, pitch_matches)
        expressive_dynamics = dynamics * (0.8 + 0.2 * force_score)
        details.append({
            "reference_part": reference_part.name,
            "candidate_parts": sorted({agent_score.parts[part_index].name for part_index, _ in candidate}),
            "reference_notes": len(notes),
            "candidate_notes": len(candidate),
            "pitch": 2 * len(pitch_matches) / denominator if denominator else 0.0,
            "rhythm": 2 * len(rhythm_matches) / denominator if denominator else 0.0,
            "dynamics": expressive_dynamics,
            "dynamic_profile": dynamics,
            "force_articulations": force_score,
            "instruments": 2 * program_matches / denominator if denominator else 0.0,
        })
    reference_count = sum(len(notes) for notes, _ in groups)
    extra_penalty = reference_count / (reference_count + extras)
    return {
        **{metric: float(np.mean([detail[metric] for detail in details])) * extra_penalty for metric in ("pitch", "rhythm", "dynamics", "instruments")},
        "extra_note_penalty": extra_penalty,
        "unassigned_extra_notes": extras,
        "parts": details,
    }
