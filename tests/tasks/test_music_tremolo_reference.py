from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pretty_midi
import pytest

from tasks.visual_media.music_transcription.main import TrackInfo
from tasks.visual_media.music_transcription.notation import Mark, Note, Part, Score
from tasks.visual_media.music_transcription.playback import compare_playback, reference_tremolo_timing


def prelude_passage(paired):
    beginning = Fraction(274 if paired else 30)
    ending = beginning + (4 if paired else 6)
    notes = [Note(81 if paired else 62, beginning, beginning + 2 if paired else ending, 40 if paired else 46, False, "1", "1", tremolo=3)]
    marks = []
    if paired:
        notes.append(replace(notes[0], pitch=86, start=beginning + 2, end=ending))
        marks = [Mark("tremolo", "start", beginning, "1", "1:3"), Mark("tremolo", "stop", beginning + 2, "1", "1:3")]
    notation = Score([Part("Strings" if paired else "Harp", notes, marks)], [(Fraction(0), 140)], "", "")
    start = notation.seconds(beginning) * .999999
    source_cadence, native_cadence = (.189, .10625) if paired else (.1, .05142857142857143)
    source_count, native_count = (9, 16) if paired else (25, 48)
    performances = []
    for cadence, count in ((source_cadence, source_count), (native_cadence, native_count)):
        attacks = [pretty_midi.Note(80, int(note.pitch), start + (index + phase / len(notes)) * cadence, start + (index + phase / len(notes)) * cadence + .025) for index in range(count) for phase, note in enumerate(notes)]
        performances.append([TrackInfo(notation.parts[0].name, notes[0].program, False, attacks, [])])
    return notation, *performances


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("realization", ["source", "native"])
def test_source_and_native_tremolos_keep_complete_attack_patterns(paired, realization):
    notation, source, native = prelude_passage(paired)
    timing = reference_tremolo_timing(notation, source)
    result = compare_playback(notation, source if realization == "source" else native, tremolo_timing=timing)
    assert result["notation_coverage"] == result["midi_precision"] == result["program_accuracy"] == 1


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("realization", ["source", "native"])
@pytest.mark.parametrize("mutation", ["delete_first", "delete_middle", "delete_last", "wrong_pitch", "alternate_deleted", "held"])
def test_source_tremolo_allowance_rejects_deleted_or_wrong_attacks(paired, realization, mutation):
    notation, source, native = prelude_passage(paired)
    timing = reference_tremolo_timing(notation, source)
    candidate = deepcopy(source if realization == "source" else native)
    attacks = candidate[0].notes
    if mutation.startswith("delete_"):
        index = {"delete_first": 0, "delete_middle": len(attacks) // 2, "delete_last": -1}[mutation]
        del attacks[index]
    elif mutation == "wrong_pitch":
        attacks[len(attacks) // 2].pitch += 1
    elif mutation == "alternate_deleted":
        candidate[0].notes = attacks[::2]
    else:
        candidate[0].notes = attacks[:1]
        candidate[0].notes[0].end = notation.seconds(notation.parts[0].notes[-1].end)
    result = compare_playback(notation, candidate, tremolo_timing=timing)
    assert result["notation_coverage"] < 1
    assert result["midi_precision"] < 1


@pytest.mark.parametrize("paired", [False, True])
def test_tremolo_reference_cannot_borrow_next_attack_with_rounded_midi_tempo(paired):
    notation, source, native = prelude_passage(paired)
    last = notation.parts[0].notes[-1]
    following = replace(notation.parts[0].notes[0], start=last.end, end=last.end + 1, tremolo=0)
    notation.parts[0].notes.append(following)
    instant = notation.seconds(following.start) * .999999
    for performance in (source, native):
        performance[0].notes.append(pretty_midi.Note(80, int(following.pitch), instant, instant + .1))
    timing = reference_tremolo_timing(notation, source)
    assert compare_playback(notation, native, tremolo_timing=timing)["notation_coverage"] == 1
    native[0].notes.pop()
    assert compare_playback(notation, native, tremolo_timing=timing)["notation_coverage"] < 1


def test_paired_tremolo_requires_alternation_without_reference_performance():
    beginning = Note(60, Fraction(0), Fraction(1), 40, False, "1", "1", tremolo=1)
    ending = replace(beginning, pitch=67, start=Fraction(1), end=Fraction(2))
    marks = [Mark("tremolo", "start", Fraction(0), "1", "1:1"), Mark("tremolo", "stop", Fraction(1), "1", "1:1")]
    notation = Score([Part("Strings", [beginning, ending], marks)], [(Fraction(0), 60)], "", "")
    attacks = [pretty_midi.Note(80, 60 if index % 2 == 0 else 67, index / 2, index / 2 + .1) for index in range(4)]
    tracks = [TrackInfo("Strings", 40, False, attacks, [])]
    assert compare_playback(notation, tracks)["notation_coverage"] == 1
    attacks[1].start = 0
    assert compare_playback(notation, tracks)["notation_coverage"] == 0


def test_reference_tremolo_keeps_program_checks():
    notation, source, native = prelude_passage(False)
    timing = reference_tremolo_timing(notation, source)
    native[0].program = 0
    result = compare_playback(notation, native, tremolo_timing=timing)
    assert result["notation_coverage"] == 1
    assert result["program_accuracy"] == 0


@pytest.mark.parametrize("paired", [False, True])
def test_source_tremolo_survives_native_midi_tempo_rounding(paired):
    notation, source, _ = prelude_passage(paired)
    timing = reference_tremolo_timing(notation, source)
    native_notation = replace(notation, tempos=[(Fraction(0), 60_000_000 / 428571)])
    result = compare_playback(native_notation, source, tremolo_timing=timing)
    assert result["notation_coverage"] == result["midi_precision"] == 1
