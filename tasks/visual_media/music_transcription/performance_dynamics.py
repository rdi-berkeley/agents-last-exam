from bisect import bisect_right
from collections import defaultdict
from fractions import Fraction

import numpy as np
from scipy.stats import spearmanr

from .notation import Note, Part, Score
from .playback import compare_playback, reference_tremolo_timing, reference_trill_timing


def performance_score(tracks):
    parts = []
    loudness = {}
    for track in tracks:
        controls = {}
        for number in (7, 11):
            changes = sorted((event.time, event.value) for event in track.control_changes if event.number == number)
            controls[number] = ([entry[0] for entry in changes], [entry[1] for entry in changes])
        notes = []
        for event in track.notes:
            pitch = event.pitch
            if track.is_drum and pitch in (35, 36):
                pitch = 35
            elif track.is_drum and pitch in (49, 57):
                pitch = 49
            musical_note = Note(pitch, Fraction(str(event.start)), Fraction(str(event.end)), track.program, track.is_drum, "1", "1")
            value = float(event.velocity)
            for times, values in controls.values():
                index = bisect_right(times, event.start) - 1
                value *= values[index] / 127 if index >= 0 else 1.0
            loudness[id(musical_note)] = value
            notes.append(musical_note)
        if notes:
            parts.append(Part(track.name, notes))
    return Score(parts, [(Fraction(0), 60.0)], "", ""), loudness


def compare_performance_dynamics(notation, reference_tracks, candidate_tracks, part_indices, *, trill_timing=None, tremolo_timing=None):
    if trill_timing is None:
        trill_timing = reference_trill_timing(notation, reference_tracks)
    if tremolo_timing is None:
        tremolo_timing = reference_tremolo_timing(notation, reference_tracks)
    profiles = []
    for tracks in (reference_tracks, candidate_tracks):
        performance, loudness = performance_score(tracks)
        locations = [(track_index, event_index) for track_index, track in enumerate(tracks) for event_index in range(len(track.notes))]
        values = dict(zip(locations, [loudness[id(note)] for part in performance.parts for note in part.notes], strict=True))
        matches = {}
        compare_playback(notation, tracks, note_matches=matches, trill_timing=trill_timing, tremolo_timing=tremolo_timing)
        profile = {}
        for identity, matched in matches.items():
            attacks = defaultdict(list)
            for location in matched:
                attacks[tracks[location[0]].notes[location[1]].start].append(values[location])
            profile[identity] = float(np.mean([max(values) for values in attacks.values()]))
        profiles.append(profile)
    reference, candidate = profiles
    results = {}
    for part_index in part_indices:
        notes = notation.parts[part_index].notes
        expected = [id(note) for note in notes if id(note) in reference]
        actual = [id(note) for note in notes if id(note) in candidate]
        if not expected:
            raise ValueError(f"Reference MIDI has no performance for {notation.parts[part_index].name}")
        if len(expected) != len(notes):
            raise ValueError(f"Reference MIDI does not cover every written note of {notation.parts[part_index].name}")
        matches = [identity for identity in expected if identity in candidate]
        source_values = [reference[identity] for identity in matches]
        candidate_values = [candidate[identity] for identity in matches]
        if not matches:
            similarity = 0.0
        elif np.ptp(source_values) < 1e-9:
            similarity = float(np.ptp(candidate_values) < 1e-9)
        elif np.ptp(candidate_values) < 1e-9:
            similarity = 0.0
        else:
            correlation = float(spearmanr(source_values, candidate_values).statistic)
            similarity = max(0.0, correlation) if np.isfinite(correlation) else 0.0
        coverage = 2 * len(matches) / (len(expected) + len(actual))
        results[part_index] = {
            "score": similarity * coverage,
            "correlation": similarity,
            "matched_notes": len(matches),
            "reference_notes": len(expected),
            "candidate_notes": len(actual),
        }
    return results
