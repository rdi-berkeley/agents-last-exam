from dataclasses import replace
from fractions import Fraction

import pretty_midi
import pytest

from tasks.visual_media.music_transcription.main import TrackInfo
from tasks.visual_media.music_transcription.notation import parse_musicxml
from tasks.visual_media.music_transcription.notation_scoring import compare_notation
from tasks.visual_media.music_transcription.playback import compare_playback


def trill_score(step="C", alter=1, fifths=-1, ornament="", preceding="", transpose=""):
    return parse_musicxml(f'''<score-partwise><part-list>
      <score-part id="piccolo"><part-name>Piccolo</part-name>
        <midi-instrument id="instrument"><midi-program>73</midi-program></midi-instrument>
      </score-part></part-list><part id="piccolo"><measure implicit="yes">
        <attributes><divisions>2</divisions><key><fifths>{fifths}</fifths></key>{transpose}</attributes>
        <direction><sound tempo="120"/></direction>{preceding}
        <note><pitch><step>{step}</step><alter>{alter}</alter><octave>7</octave></pitch>
        <duration>3</duration><notations><ornaments>{ornament or '<trill-mark/>'}</ornaments></notations></note>
      </measure></part></score-partwise>'''.encode())


@pytest.mark.parametrize("step,alter,fifths,upper", [("C", 1, -1, 98), ("A", 0, -1, 106), ("E", 0, 1, 102), ("B", 0, 0, 108)])
def test_trill_uses_written_neighbor_and_key_signature(step, alter, fifths, upper):
    assert trill_score(step, alter, fifths).parts[0].notes[0].trill_pitch == upper


def test_trill_accidental_and_transposition():
    score = trill_score(ornament='<trill-mark/><accidental-mark>sharp</accidental-mark>', transpose='<transpose><chromatic>-12</chromatic></transpose>')
    assert (score.parts[0].notes[0].pitch, score.parts[0].notes[0].trill_pitch) == (85, 87)
    assert trill_score(ornament='<trill-mark trill-step="whole"/>').parts[0].notes[0].trill_pitch == 99


def test_trill_respects_previous_accidental_in_the_measure():
    preceding = '<note><pitch><step>D</step><alter>-1</alter><octave>7</octave></pitch><duration>2</duration></note>'
    assert trill_score(preceding=preceding).parts[0].notes[-1].trill_pitch == 97


@pytest.mark.parametrize("first", [97, 98])
def test_trill_playback_accepts_alternating_native_attacks(first):
    score = trill_score()
    pitches = [first if index % 2 == 0 else 195 - first for index in range(12)]
    notes = [pretty_midi.Note(90, pitch, index / 16, (index + 1) / 16) for index, pitch in enumerate(pitches)]
    track = TrackInfo("Renamed", 72, False, notes, [])
    result = compare_playback(score, [track])
    assert result["notation_coverage"] == result["midi_precision"] == result["program_accuracy"] == 1
    track.program = 0
    assert compare_playback(score, [track])["program_accuracy"] == 0


@pytest.mark.parametrize("case", ["held", "repeated", "wrong_neighbor", "too_short", "late", "simultaneous"])
def test_incorrect_trill_performance_is_not_excused(case):
    pitches = [97 if index % 2 == 0 else 98 for index in range(12)]
    starts = [index / 16 for index in range(12)]
    if case == "held":
        pitches, starts = [97], [0]
    elif case == "repeated":
        pitches = [97] * 12
    elif case == "wrong_neighbor":
        pitches = [97 if index % 2 == 0 else 99 for index in range(12)]
    elif case == "too_short":
        pitches, starts = pitches[:4], starts[:4]
    elif case == "late":
        starts = [start + .2 for start in starts]
    elif case == "simultaneous":
        starts = [0] * 12
    notes = [pretty_midi.Note(90, pitch, start, start + .05) for pitch, start in zip(pitches, starts, strict=True)]
    result = compare_playback(trill_score(), [TrackInfo("Renamed", 72, False, notes, [])])
    assert result["notation_coverage"] == 0


def test_deleting_trill_notation_does_not_restore_rhythm_credit():
    reference = trill_score()
    assert compare_notation(reference, reference)["rhythm"] == 1
    part = reference.parts[0]
    candidate = replace(reference, parts=[replace(part, notes=[replace(part.notes[0], trill_pitch=None)])])
    result = compare_notation(candidate, reference)
    assert result["pitch"] == 1
    assert result["rhythm"] == 0


def test_fully_written_out_trill_is_equivalent_to_the_ornament():
    reference = trill_score()
    part = reference.parts[0]
    source = part.notes[0]
    notes = [replace(source, pitch=97 if index % 2 == 0 else 98, start=Fraction(index, 8), end=Fraction(index + 1, 8), trill_pitch=None) for index in range(12)]
    candidate = replace(reference, parts=[replace(part, notes=notes)])
    result = compare_notation(candidate, reference)
    assert result["pitch"] == result["rhythm"] == result["instruments"] == 1
    notes[5] = replace(notes[5], pitch=99)
    result = compare_notation(candidate, reference)
    assert result["pitch"] < 1 and result["rhythm"] < 1
