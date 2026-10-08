from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pretty_midi
import pytest

from tasks.visual_media.music_transcription.main import TrackInfo
from tasks.visual_media.music_transcription.notation import Mark, Note, Part, Score
from tasks.visual_media.music_transcription.playback import compare_playback, reference_tremolo_timing


def passage(paired, rounded):
    beginning = Fraction(274 if paired else 30)
    ending = beginning + (4 if paired else 6)
    first = Note(81 if paired else 62, beginning, beginning + 2 if paired else ending, 40 if paired else 46, False, "1", "1", tremolo=3)
    notes = [first]
    marks = []
    if paired:
        notes.append(replace(first, pitch=86, start=beginning + 2, end=ending))
        marks = [Mark("tremolo", "start", beginning, "1", "1:3"), Mark("tremolo", "stop", beginning + 2, "1", "1:3")]
    notation = Score([Part("Strings" if paired else "Harp", notes, marks)], [(Fraction(0), 60_000_000 / 428571 if rounded else 140)], "", "")
    start = notation.seconds(beginning)
    source_cadence, source_count = (.189, 9) if paired else (.0999999, 25)
    source_attacks = [pretty_midi.Note(80, int(note.pitch), start + (index + phase / len(notes)) * source_cadence, start + (index + phase / len(notes)) * source_cadence + .025) for index in range(source_count) for phase, note in enumerate(notes)]
    candidate_attacks = []
    for index in range(int((ending - beginning) * 8)):
        note = notes[index % len(notes)]
        instant = notation.seconds(beginning + Fraction(index, 8))
        candidate_attacks.append(pretty_midi.Note(80, int(note.pitch), instant, instant + .025))
    source = [TrackInfo(notation.parts[0].name, first.program, False, source_attacks, [])]
    candidate = [TrackInfo(notation.parts[0].name, first.program, False, candidate_attacks, [])]
    return notation, source, candidate


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("rounded", [False, True])
def test_complete_written_grid_with_shorter_reference(paired, rounded):
    notation, source, candidate = passage(paired, rounded)
    timing = reference_tremolo_timing(notation, source)
    result = compare_playback(notation, candidate, tremolo_timing=timing)
    assert result["notation_coverage"] == result["midi_precision"] == result["program_accuracy"] == 1


@pytest.mark.parametrize("mutation", ["first", "middle", "short_both", "wrong_pitch", "held", "too_slow"])
def test_incomplete_or_wrong_written_roll_is_rejected(mutation):
    notation, source, candidate = passage(False, True)
    timing = reference_tremolo_timing(notation, source)
    attacks = candidate[0].notes
    if mutation == "first":
        attacks.pop(0)
    elif mutation == "middle":
        attacks.pop(len(attacks) // 2)
    elif mutation == "short_both":
        del attacks[-3:]
    elif mutation == "wrong_pitch":
        attacks[len(attacks) // 2].pitch += 1
    elif mutation == "held":
        candidate[0].notes = attacks[:1]
        attacks[0].end = notation.seconds(notation.parts[0].notes[0].end)
    else:
        candidate[0].notes = attacks[::2]
    result = compare_playback(notation, candidate, tremolo_timing=timing)
    assert result["notation_coverage"] < 1
    assert result["midi_precision"] < 1


@pytest.mark.parametrize("count", [46, 47])
def test_independently_valid_reference_span_remains_accepted(count):
    notation, source, candidate = passage(False, True)
    timing = reference_tremolo_timing(notation, source)
    candidate[0].notes = candidate[0].notes[:count]
    result = compare_playback(notation, candidate, tremolo_timing=timing)
    assert result["notation_coverage"] == result["midi_precision"] == 1


def test_complete_written_grid_keeps_program_penalty():
    notation, source, candidate = passage(False, True)
    timing = reference_tremolo_timing(notation, source)
    candidate[0].program = 0
    result = compare_playback(notation, candidate, tremolo_timing=timing)
    assert result["notation_coverage"] == result["midi_precision"] == 1
    assert result["program_accuracy"] == 0


def test_written_roll_cannot_consume_next_same_pitch_note():
    notation, source, candidate = passage(False, True)
    first = notation.parts[0].notes[0]
    following = replace(first, start=first.end, end=first.end + 1, tremolo=0)
    notation.parts[0].notes.append(following)
    instant = notation.seconds(following.start)
    for tracks in (source, candidate):
        tracks[0].notes.append(pretty_midi.Note(80, int(first.pitch), instant, instant + .1))
    timing = reference_tremolo_timing(notation, source)
    assert compare_playback(notation, candidate, tremolo_timing=timing)["notation_coverage"] == 1
    missing_following = deepcopy(candidate)
    missing_following[0].notes.pop()
    assert compare_playback(notation, missing_following, tremolo_timing=timing)["notation_coverage"] < 1
