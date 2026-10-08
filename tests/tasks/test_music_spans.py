from dataclasses import replace
from fractions import Fraction

import pretty_midi

from tasks.visual_media.music_transcription.main import TrackInfo
from tasks.visual_media.music_transcription.notation import Mark, Note, Part, Score
from tasks.visual_media.music_transcription.notation_scoring import compare_notation
from tasks.visual_media.music_transcription.playback import compare_playback


def test_pedal_applies_to_both_piano_staves_and_cannot_be_silently_omitted():
    notes = [Note(60 + index, Fraction(index), Fraction(index + 1), 0, False, str(index % 2 + 1), "1") for index in range(4)]
    marks = [Mark("pedal", "start", Fraction(0), "1"), Mark("pedal", "stop", Fraction(3), "1")]
    reference = Score([Part("Piano", notes, marks)], [(Fraction(0), 120)], "", "")
    candidate = replace(reference, parts=[replace(reference.parts[0], marks=[])])
    assert compare_notation(reference, reference)["rhythm"] == 1
    assert compare_notation(candidate, reference)["rhythm"] == .25
    alternate = replace(reference, parts=[replace(reference.parts[0], marks=[replace(mark, staff="2") for mark in marks])])
    assert compare_notation(alternate, reference)["rhythm"] == 1


def test_native_pedal_release_rounding_uses_the_existing_timing_tolerance():
    notes = [Note(60 + index, Fraction(index), Fraction(index + 1), 0, False, "1", "1") for index in range(4)]
    marks = [Mark("pedal", "start", Fraction(0), "1"), Mark("pedal", "stop", Fraction(3) + Fraction(1, 256), "1")]
    reference = Score([Part("Piano", notes, marks)], [(Fraction(0), 120)], "", "")
    candidate = replace(reference, parts=[replace(reference.parts[0], marks=[marks[0], replace(marks[1], position=Fraction(3))])])
    assert compare_notation(candidate, reference)["rhythm"] == 1
    assert compare_notation(reference, candidate)["rhythm"] == 1
    candidate.parts[0].marks[-1] = replace(marks[-1], position=Fraction(3) + Fraction(1, 4))
    assert compare_notation(candidate, reference)["rhythm"] < 1


def test_pedal_released_too_early_within_a_written_note_loses_credit():
    notes = [Note(60, Fraction(0), Fraction(2), 0, False, "1", "1")]
    marks = [Mark("pedal", "start", Fraction(0), "1"), Mark("pedal", "stop", Fraction(2), "1")]
    reference = Score([Part("Piano", notes, marks)], [(Fraction(0), 120)], "", "")
    candidate = replace(reference, parts=[replace(reference.parts[0], marks=[marks[0], replace(marks[1], position=Fraction(1))])])
    assert compare_notation(candidate, reference)["rhythm"] == 0


def glissando_score():
    notes = [Note(60, Fraction(0), Fraction(2), 46, False, "1", "1"), Note(72, Fraction(2), Fraction(3), 46, False, "1", "1")]
    marks = [Mark("glissando", "start", Fraction(0), "1", note_pitch=60), Mark("glissando", "stop", Fraction(2), "1", note_pitch=72)]
    return Score([Part("Harp", notes, marks)], [(Fraction(0), 120)], "", "")


def test_glissando_mark_cannot_be_deleted_to_avoid_playing_it():
    reference = glissando_score()
    candidate = replace(reference, parts=[replace(reference.parts[0], marks=[])])
    assert compare_notation(reference, reference)["rhythm"] == 1
    assert compare_notation(candidate, reference)["rhythm"] == 0
    alternate = replace(reference, parts=[replace(reference.parts[0], marks=[replace(mark, kind="slide") for mark in reference.parts[0].marks])])
    assert compare_notation(alternate, reference)["rhythm"] == 1


def test_explicit_complete_glissando_retains_notation_credit():
    reference = glissando_score()
    pitches = (60, 62, 64, 65, 67, 69, 71, 72)
    notes = [replace(reference.parts[0].notes[0], pitch=pitch, start=Fraction(index * 2, 7), end=Fraction((index + 1) * 2, 7)) for index, pitch in enumerate(pitches[:-1])]
    notes.append(reference.parts[0].notes[-1])
    candidate = replace(reference, parts=[replace(reference.parts[0], notes=notes, marks=[])])
    result = compare_notation(candidate, reference)
    assert result["pitch"] == result["rhythm"] == 1
    notes[3] = replace(notes[3], pitch=80)
    assert compare_notation(candidate, reference)["pitch"] < 1


def test_pedal_playback_accepts_controller_or_equivalent_sustained_notes():
    notes = [Note(60 + index, Fraction(index), Fraction(index + 1), 0, False, str(index + 1), "1") for index in range(2)]
    marks = [Mark("pedal", "start", Fraction(0), "1"), Mark("pedal", "stop", Fraction(3), "1")]
    notation = Score([Part("Piano", notes, marks)], [(Fraction(0), 120)], "", "")
    performance = [pretty_midi.Note(90, 60 + index, index / 2, index / 2 + .4) for index in range(2)]
    track = TrackInfo("Piano", 0, False, performance, [pretty_midi.ControlChange(64, 127, 0), pretty_midi.ControlChange(64, 0, 1.5)])
    assert compare_playback(notation, [track])["notation_coverage"] == 1
    track.control_changes = []
    assert compare_playback(notation, [track])["notation_coverage"] == 0
    for note in track.notes:
        note.end = 1.5
    assert compare_playback(notation, [track])["notation_coverage"] == 1


def test_a_simultaneous_scale_in_another_instrument_is_not_collapsed():
    reference = glissando_score()
    pitches = (60, 62, 64, 65, 67, 69, 71, 72)
    notes = [Note(pitch, Fraction(index * 2, 7), Fraction((index + 1) * 2, 7), 68, False, "1", "1") for index, pitch in enumerate(pitches)]
    reference = replace(reference, parts=[*reference.parts, Part("Oboe", notes)])
    result = compare_notation(reference, reference)
    assert result["pitch"] == result["rhythm"] == 1
