from fractions import Fraction

import pretty_midi
import pytest

from tasks.visual_media.music_transcription.main import TrackInfo
from tasks.visual_media.music_transcription.notation import Mark, Note, Part, Score
from tasks.visual_media.music_transcription.playback import compare_playback


def ordinary_score():
    notes = [Note(pitch, Fraction(index), Fraction(index + 1), 46, False, "1", "1") for index, pitch in enumerate((60, 62, 64, 65))]
    return Score([Part("Harp", notes)], [(Fraction(0), 120.0)], "", "")


def track(pitches, starts, durations=None, program=46):
    durations = durations or [.25] * len(starts)
    notes = [pretty_midi.Note(velocity=90, pitch=pitch, start=start, end=start + duration) for pitch, start, duration in zip(pitches, starts, durations, strict=True)]
    return TrackInfo("Arbitrary name", program, False, notes, [])


def test_native_release_lengths_do_not_change_written_rhythm():
    notation = ordinary_score()
    result = compare_playback(notation, [track((60, 62, 64, 65), (0, .5, 1, 1.5), (.1, .3, .45, .5))])
    assert result["notation_coverage"] == result["midi_precision"] == result["program_accuracy"] == 1


def test_piano_voices_in_unison_can_share_a_midi_attack():
    first = Note(60, Fraction(0), Fraction(1), 0, False, "1", "1")
    second = Note(60, Fraction(0), Fraction(2), 0, False, "2", "2")
    notation = Score([Part("Piano", [first, second])], [(Fraction(0), 120)], "", "")
    for pitches, starts in (([60], [0]), ([60, 60], [0, 0])):
        result = compare_playback(notation, [track(pitches, starts, program=0)])
        assert result["notation_coverage"] == result["midi_precision"] == result["program_accuracy"] == 1
    result = compare_playback(notation, [track([60, 60, 60], [0, 0, 0], program=0)])
    assert result["midi_precision"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("kind", ["empty", "unrelated", "late"])
def test_absent_or_unrelated_performance_does_not_pass(kind):
    notation = ordinary_score()
    if kind == "empty":
        tracks = []
    elif kind == "unrelated":
        tracks = [track((61, 63, 66, 68), (0, .5, 1, 1.5))]
    else:
        tracks = [track((60, 62, 64, 65), (10, 10.5, 11, 11.5))]
    result = compare_playback(notation, tracks)
    assert result["notation_coverage"] == result["midi_precision"] == 0


def test_wrong_program_and_extra_notes_are_reported_separately():
    notation = ordinary_score()
    result = compare_playback(notation, [track((60, 62, 64, 65, 30), (0, .5, 1, 1.5, 3), program=0)])
    assert result["notation_coverage"] == 1
    assert result["midi_precision"] == .8
    assert result["program_accuracy"] == 0


def test_tremolo_requires_real_repeated_playback_without_fixed_sampler_gate_length():
    notation = Score([Part("Harp", [Note(60, Fraction(0), Fraction(2), 46, False, "1", "1", tremolo=3)])], [(Fraction(0), 120.0)], "", "")
    valid = track([60] * 16, [index / 16 for index in range(16)], [.04] * 16)
    assert compare_playback(notation, [valid])["notation_coverage"] == 1
    for candidate in (track([60], [0]), track([60, 60], [0, .95]), track([60] * 8, [index / 16 for index in range(8)])):
        result = compare_playback(notation, [candidate])
        assert result["notation_coverage"] == 0


def test_glissando_interior_notes_do_not_count_as_spurious_extra_notes():
    notes = [Note(60, Fraction(0), Fraction(2), 46, False, "1", "1"), Note(72, Fraction(2), Fraction(4), 46, False, "2", "1")]
    part = Part("Harp", notes, [Mark("glissando", "start", Fraction(0), "1"), Mark("glissando", "stop", Fraction(2), "2")])
    notation = Score([part], [(Fraction(0), 120.0)], "", "")
    candidate = track((60, 62, 64, 65, 67, 69, 71, 72), (0, .125, .25, .375, .5, .625, .75, 1))
    result = compare_playback(notation, [candidate])
    assert result["notation_coverage"] == result["midi_precision"] == 1
    candidate.notes[2].pitch = 80
    assert compare_playback(notation, [candidate])["midi_precision"] < 1
    assert compare_playback(notation, [track((60, 72), (0, 1))])["notation_coverage"] == 0


def test_glissando_endpoints_are_identified_with_other_simultaneous_notes():
    notes = [Note(pitch, Fraction(start), Fraction(start + 2), 46, False, "1", str(voice)) for pitch, start, voice in ((48, 0, 2), (60, 0, 1), (55, 2, 2), (72, 2, 1))]
    marks = [Mark("glissando", "start", Fraction(0), "1", note_pitch=60), Mark("glissando", "stop", Fraction(2), "1", note_pitch=72)]
    notation = Score([Part("Harp", notes, marks)], [(Fraction(0), 120)], "", "")
    candidate = track((48, 60, 62, 64, 67, 69, 71, 72, 55), (0, 0, .15, .3, .45, .6, .75, 1, 1))
    result = compare_playback(notation, [candidate])
    assert result["notation_coverage"] == result["midi_precision"] == 1


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("starts", [(.5, .6, .7), (.3, .4, .5), (.5, .6, 1.1)])
def test_rolled_chord_accepts_native_spread_beyond_ordinary_onset_tolerance(direction, starts):
    notes = [Note(pitch, Fraction(1), Fraction(3), 46, False, "2" if pitch == 60 else "1", "1", arpeggio=("1", direction)) for pitch in (60, 64, 67)]
    notation = Score([Part("Harp", notes)], [(Fraction(0), 120)], "", "")
    pitches = (60, 64, 67) if direction == "up" else (67, 64, 60)
    result = compare_playback(notation, [track(pitches, starts)])
    assert result["notation_coverage"] == result["midi_precision"] == result["program_accuracy"] == 1


@pytest.mark.parametrize("pitches,starts", [
    ((60, 64), (.5, .6)),
    ((60, 64, 67), (.5, .5, .5)),
    ((60, 64, 67), (.7, .8, .9)),
    ((60, 64, 67), (.5, .6, 1.6)),
    ((67, 64, 60), (.5, .6, .7)),
])
def test_missing_flat_late_and_reversed_arpeggios_do_not_pass(pitches, starts):
    notes = [Note(pitch, Fraction(1), Fraction(3), 46, False, "1", "1", arpeggio=("1", "up")) for pitch in (60, 64, 67)]
    notation = Score([Part("Harp", notes)], [(Fraction(0), 120)], "", "")
    result = compare_playback(notation, [track(pitches, starts)])
    assert result["notation_coverage"] == 0


def test_arpeggio_does_not_excuse_extra_attacks_or_wrong_instruments():
    notes = [Note(pitch, Fraction(1), Fraction(3), 46, False, "1", "1", arpeggio=("1", "up")) for pitch in (60, 64, 67)]
    notation = Score([Part("Harp", notes)], [(Fraction(0), 120)], "", "")
    result = compare_playback(notation, [track((60, 64, 67, 64), (.5, .6, .7, .75), program=0)])
    assert result["notation_coverage"] == 1
    assert result["midi_precision"] == .75
    assert result["program_accuracy"] == 0


@pytest.mark.parametrize("starts", [(.35, .425, .5), (.5, .575, .65)])
def test_grace_sequence_can_anticipate_or_delay_the_principal(starts):
    notes = [Note(pitch, Fraction(1), Fraction(1), 46, False, "1", "1", frozenset({"grace"}), grace_order=index) for index, pitch in enumerate((62, 64))]
    notes.append(Note(60, Fraction(1), Fraction(3), 46, False, "1", "1"))
    notation = Score([Part("Harp", notes)], [(Fraction(0), 120)], "", "")
    result = compare_playback(notation, [track((62, 64, 60), starts)])
    assert result["notation_coverage"] == result["midi_precision"] == result["program_accuracy"] == 1


@pytest.mark.parametrize("pitches,starts", [
    ((62, 60), (.4, .5)),
    ((64, 62, 60), (.35, .425, .5)),
    ((62, 64, 60), (.5, .5, .5)),
    ((62, 64, 60), (.7, .775, .85)),
    ((62, 64, 60), (0, .2, .8)),
])
def test_graces_require_complete_ordered_attacks_near_the_written_onset(pitches, starts):
    notes = [Note(pitch, Fraction(1), Fraction(1), 46, False, "1", "1", frozenset({"grace"}), grace_order=index) for index, pitch in enumerate((62, 64))]
    notes.append(Note(60, Fraction(1), Fraction(3), 46, False, "1", "1"))
    notation = Score([Part("Harp", notes)], [(Fraction(0), 120)], "", "")
    assert compare_playback(notation, [track(pitches, starts)])["notation_coverage"] < 1


def test_repeated_grace_pitch_requires_separate_attacks():
    notes = [Note(60, Fraction(1), Fraction(1), 46, False, "1", "1", frozenset({"grace"}), grace_order=index) for index in range(2)]
    notes.append(Note(60, Fraction(1), Fraction(3), 46, False, "1", "1"))
    notation = Score([Part("Harp", notes)], [(Fraction(0), 120)], "", "")
    assert compare_playback(notation, [track((60,), (.5,))])["notation_coverage"] == 0
    assert compare_playback(notation, [track((60, 60, 60), (.35, .425, .5))])["notation_coverage"] == 1
