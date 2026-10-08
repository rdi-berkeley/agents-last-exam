from dataclasses import replace
from fractions import Fraction

import pretty_midi
import pytest

from tasks.visual_media.music_transcription.main import TrackInfo
from tasks.visual_media.music_transcription.notation import Mark, Note, Part, Score
from tasks.visual_media.music_transcription.notation_scoring import compare_notation
from tasks.visual_media.music_transcription.performance_dynamics import compare_performance_dynamics
from tasks.visual_media.music_transcription.playback import compare_playback, reference_trill_timing


def performance(pitches, starts, durations=None, program=46):
    durations = durations or [.05] * len(starts)
    notes = [pretty_midi.Note(80, pitch, start, start + duration) for pitch, start, duration in zip(pitches, starts, durations, strict=True)]
    return [TrackInfo("Harp", program, False, notes, [])]


def decorated_score(kind, *, end=3, following=None):
    principal = Note(60, Fraction(1), Fraction(end), 46, False, "1", "1")
    if kind == "arpeggio":
        notes = [replace(principal, pitch=pitch, arpeggio=("1", "up")) for pitch in (60, 64, 67)]
    else:
        notes = [replace(principal, pitch=pitch, end=principal.start, articulations=frozenset({"grace"}), grace_order=index) for index, pitch in enumerate((62, 64))]
        notes.append(principal)
    if following is not None:
        notes.append(replace(principal, pitch=72, start=Fraction(str(following)), end=Fraction(end + 1)))
    return Score([Part("Harp", notes)], [(Fraction(0), 60)], "", "")


@pytest.mark.parametrize("kind,pitches", [("arpeggio", (60, 64, 67)), ("grace", (62, 64, 60))])
@pytest.mark.parametrize("starts", [(1, 1.6, 2.4), (.1, .6, 1)])
def test_ornament_timing_uses_local_score_span_without_half_duration_or_seconds_cap(kind, pitches, starts):
    notation = decorated_score(kind)
    result = compare_playback(notation, performance(pitches, starts))
    assert result["notation_coverage"] == result["midi_precision"] == result["program_accuracy"] == 1


@pytest.mark.parametrize("kind,pitches", [("arpeggio", (60, 64, 67)), ("grace", (62, 64, 60))])
def test_short_ornament_can_exceed_half_of_principal_without_crossing_its_end(kind, pitches):
    notation = decorated_score(kind, end=Fraction(6, 5))
    assert compare_playback(notation, performance(pitches, (1, 1.07, 1.15)))["notation_coverage"] == 1
    assert compare_playback(notation, performance(pitches, (1, 1.07, 1.25)))["notation_coverage"] == 0


@pytest.mark.parametrize("kind,pitches", [("arpeggio", (60, 64, 67)), ("grace", (62, 64, 60))])
def test_ornament_cannot_spill_into_the_next_written_event(kind, pitches):
    notation = decorated_score(kind, following=1.8)
    result = compare_playback(notation, performance((*pitches, 72), (1, 1.6, 2.4, 1.8)))
    assert result["notation_coverage"] == .25
    assert result["midi_precision"] == .25


@pytest.mark.parametrize("kind,pitches", [("arpeggio", (60, 64, 67)), ("grace", (62, 64, 60))])
def test_ornament_cannot_anticipate_across_the_previous_written_onset(kind, pitches):
    notation = decorated_score(kind)
    notation.parts[0].notes.insert(0, Note(55, Fraction(1, 2), Fraction(1), 46, False, "1", "1"))
    result = compare_playback(notation, performance((55, *pitches), (.5, .1, .6, 1)))
    assert result["notation_coverage"] == .25


@pytest.mark.parametrize("kind,pitches", [("arpeggio", (60, 64, 67)), ("grace", (62, 64, 60))])
def test_ornaments_cannot_borrow_an_ordinary_note_attack(kind, pitches):
    notation = decorated_score(kind)
    notation.parts[0].notes.append(Note(pitches[1], Fraction(8, 5), Fraction(2), 46, False, "2", "2"))
    result = compare_playback(notation, performance(pitches, (1, 1.6, 2.4)))
    assert result["notation_coverage"] == .25
    complete = performance((*pitches, pitches[1]), (1, 1.6, 2.4, 1.6))
    assert compare_playback(notation, complete)["notation_coverage"] == 1


@pytest.mark.parametrize("error", [.039, .041])
def test_ornament_allowance_keeps_ordinary_onset_accuracy(error):
    notation = decorated_score("arpeggio", following=3)
    result = compare_playback(notation, performance((60, 64, 67, 72), (1, 1.6, 2.4, 3 + error)))
    assert result["notation_coverage"] == (1 if error < .04 else .75)
    altered = replace(notation, parts=[replace(notation.parts[0], notes=[*notation.parts[0].notes[:-1], replace(notation.parts[0].notes[-1], end=Fraction(str(4 + error)))])])
    assert compare_notation(altered, notation)["rhythm"] == (1 if error < .04 else .75)


def slow_trill():
    note = Note(60, Fraction(0), Fraction(3), 46, False, "1", "1", trill_pitch=62)
    notation = Score([Part("Harp", [note])], [(Fraction(0), 60)], "", "")
    source = performance((60, 62, 60, 62, 60), (0, .6, 1.2, 1.8, 2.4))
    return notation, source


@pytest.mark.parametrize("strikes", [5, 10])
def test_trill_uses_source_cadence_and_accepts_different_complete_rates(strikes):
    notation, source = slow_trill()
    timing = reference_trill_timing(notation, source)
    candidate = performance([60 if index % 2 == 0 else 62 for index in range(strikes)], [index * 3 / strikes for index in range(strikes)])
    result = compare_playback(notation, candidate, trill_timing=timing)
    assert result["notation_coverage"] == result["midi_precision"] == 1
    assert compare_performance_dynamics(notation, source, candidate, [0])[0]["score"] == 1


@pytest.mark.parametrize("case", ["held", "wrong_neighbor", "missing_middle", "truncated", "late", "simultaneous"])
def test_reference_anchored_trill_does_not_excuse_incomplete_or_incorrect_music(case):
    notation, source = slow_trill()
    timing = reference_trill_timing(notation, source)
    pitches, starts = [60, 62, 60, 62, 60], [0, .6, 1.2, 1.8, 2.4]
    if case == "held":
        pitches, starts = [60], [0]
    elif case == "wrong_neighbor":
        pitches[1] = 63
    elif case == "missing_middle":
        del pitches[1:3]
        del starts[1:3]
    elif case == "truncated":
        pitches, starts = pitches[:3], starts[:3]
    elif case == "late":
        starts = [start + .05 for start in starts]
    elif case == "simultaneous":
        starts = [0] * 5
    result = compare_playback(notation, performance(pitches, starts), trill_timing=timing)
    assert result["notation_coverage"] == 0


def test_trill_bounds_come_from_reference_not_candidate_releases():
    notation, source = slow_trill()
    timing = reference_trill_timing(notation, source)
    sparse = performance((60, 62, 60), (0, .1, 2.9), durations=(2, 2, 2))
    assert compare_playback(notation, sparse, trill_timing=timing)["notation_coverage"] == 0
    with pytest.raises(ValueError, match="Reference MIDI"):
        reference_trill_timing(notation, performance((60,), (0,)))


def test_reference_trill_excludes_next_note_despite_floating_point_boundary_rounding():
    notation, source = slow_trill()
    note = notation.parts[0].notes[0]
    notation.parts[0].notes = [replace(note, end=Fraction("3.0000000000000004")), replace(note, pitch=62, start=Fraction(3), end=Fraction(4), trill_pitch=None)]
    source[0].notes.append(pretty_midi.Note(80, 62, 3, 3.5))
    timing = reference_trill_timing(notation, source)
    result = compare_playback(notation, source, trill_timing=timing)
    assert result["notation_coverage"] == result["midi_precision"] == 1


def test_native_roll_can_group_adjacent_chord_pitches_at_one_attack():
    notation = decorated_score("arpeggio")
    pitches = (62, 65, 69, 73, 77, 81)
    notation.parts[0].notes = [replace(notation.parts[0].notes[0], pitch=pitch) for pitch in pitches]
    result = compare_playback(notation, performance(pitches, (1, 1, 1, 1, 1.05, 1.1)))
    assert result["notation_coverage"] == result["midi_precision"] == 1


@pytest.mark.parametrize("kind", ["trill", "glissando"])
@pytest.mark.parametrize("case", ["complete", "gap", "wrong_pitch", "reversed", "shortened"])
def test_slow_explicit_ornaments_keep_span_pitch_direction_and_continuity(kind, case):
    if kind == "trill":
        reference, _ = slow_trill()
        pitches = (60, 62, 60, 62, 60)
        step = Fraction(3, 5)
    else:
        notes = [Note(60, Fraction(0), Fraction(3), 46, False, "1", "1"), Note(72, Fraction(3), Fraction(4), 46, False, "1", "1")]
        marks = [Mark("glissando", "start", Fraction(0), "1", note_pitch=60), Mark("glissando", "stop", Fraction(3), "1", note_pitch=72)]
        reference = Score([Part("Harp", notes, marks)], [(Fraction(0), 60)], "", "")
        pitches = (60, 62, 64, 67, 69)
        step = Fraction(3, 5)
    notes = [replace(reference.parts[0].notes[0], pitch=pitch, start=index * step, end=(index + 1) * step, trill_pitch=None) for index, pitch in enumerate(pitches)]
    if kind == "glissando":
        notes.append(reference.parts[0].notes[-1])
    if case == "gap":
        notes[1] = replace(notes[1], end=notes[1].end - Fraction(1, 10))
    elif case == "wrong_pitch":
        notes[2] = replace(notes[2], pitch=80)
    elif case == "reversed":
        notes[1], notes[2] = replace(notes[1], pitch=notes[2].pitch), replace(notes[2], pitch=notes[1].pitch)
    elif case == "shortened":
        notes[-1 if kind == "trill" else -2] = replace(notes[-1 if kind == "trill" else -2], end=Fraction(14, 5))
    candidate = replace(reference, parts=[replace(reference.parts[0], notes=notes, marks=[])])
    result = compare_notation(candidate, reference)
    if case == "complete":
        assert result["pitch"] == result["rhythm"] == result["instruments"] == 1
    else:
        assert result["rhythm"] < 1


def test_measured_tremolo_uses_local_tempo_and_subdivision_for_the_entire_span():
    note = Note(60, Fraction(0), Fraction(2), 46, False, "1", "1", tremolo=1)
    notation = Score([Part("Harp", [note])], [(Fraction(0), 60), (Fraction(1), 120)], "", "")
    valid = performance([60] * 4, [0, .5, 1, 1.25])
    assert compare_playback(notation, valid)["notation_coverage"] == 1
    assert compare_playback(notation, performance([60] * 3, [0, .5, 1]))["notation_coverage"] == 0
