from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest

from tasks.visual_media.music_transcription.notation import Mark, Note, Part, Score, group_reference_parts
from tasks.visual_media.music_transcription.notation_scoring import compare_notation, expanded_notes


def control_score():
    notes = [Note(pitch, Fraction(index), Fraction(index + 1), 46, False, "1", "1") for index, pitch in enumerate((60, 62, 64, 65))]
    notes += [replace(note, pitch=note.pitch - 12, staff="2", voice="5") for note in notes]
    marks = [Mark("dynamic", "p", Fraction(0), staff) for staff in ("1", "2")]
    marks += [Mark("dynamic", "f", Fraction(2), staff) for staff in ("1", "2")]
    return Score([Part("Harp", notes, marks)], [(Fraction(0), 120.0)], "Study", "Author")


def metrics(result):
    return tuple(result[name] for name in ("pitch", "rhythm", "dynamics", "instruments"))


def test_separate_parts_and_renumbered_staves_preserve_the_same_instrument():
    reference = control_score()
    candidate = deepcopy(reference)
    candidate.parts = []
    for staff in ("2", "1"):
        original = reference.parts[0]
        candidate.parts.append(Part(
            "Arbitrary staff " + staff,
            [replace(note, staff="1", voice="1") for note in original.notes if note.staff == staff],
            [replace(mark, staff="1") for mark in original.marks if mark.staff == staff],
        ))
    assert metrics(compare_notation(candidate, reference)) == pytest.approx((1, 1, 1, 1))


def test_empty_wrong_and_duplicate_notes_cannot_receive_full_credit():
    reference = control_score()
    candidate = deepcopy(reference)
    candidate.parts[0].notes = []
    assert metrics(compare_notation(candidate, reference)) == (0, 0, 0, 0)
    candidate = deepcopy(reference)
    candidate.parts[0].notes *= 2
    result = compare_notation(candidate, reference)
    assert result["pitch"] == pytest.approx(2 / 3)
    assert result["rhythm"] == pytest.approx(2 / 3)
    candidate.parts[0].notes = [replace(note, pitch=note.pitch + .5) for note in candidate.parts[0].notes]
    assert compare_notation(candidate, reference)["pitch"] == 0


def test_flat_missing_and_reversed_dynamics_are_rejected():
    reference = control_score()
    for mode in ("missing", "flat", "reversed"):
        candidate = deepcopy(reference)
        if mode == "missing":
            candidate.parts[0].marks = []
        elif mode == "flat":
            candidate.parts[0].marks = [replace(mark, value="mf") for mark in candidate.parts[0].marks]
        else:
            candidate.parts[0].marks = [replace(mark, value="f" if mark.value == "p" else "p") for mark in candidate.parts[0].marks]
        assert compare_notation(candidate, reference)["dynamics"] == 0


def test_equivalent_relative_dynamic_levels_keep_credit():
    reference = control_score()
    candidate = deepcopy(reference)
    candidate.parts[0].marks = [replace(mark, value="mp" if mark.value == "p" else "ff") for mark in candidate.parts[0].marks]
    assert compare_notation(candidate, reference)["dynamics"] == pytest.approx(1)


def test_instrument_program_mismatch_does_not_change_note_accuracy():
    reference = control_score()
    candidate = deepcopy(reference)
    candidate.parts[0].notes = [replace(note, program=0) for note in candidate.parts[0].notes]
    result = compare_notation(candidate, reference)
    assert result["pitch"] == result["rhythm"] == 1
    assert result["instruments"] == 0


def test_rhythm_can_still_match_when_every_pitch_is_wrong():
    reference = control_score()
    candidate = deepcopy(reference)
    candidate.parts[0].notes = [replace(note, pitch=note.pitch + .5) for note in candidate.parts[0].notes]
    result = compare_notation(candidate, reference)
    assert result["pitch"] == 0
    assert result["rhythm"] == 1


def test_missing_duration_and_force_articulations_lose_credit():
    reference = control_score()
    reference.parts[0].notes = [replace(note, articulations=frozenset({"staccato", "accent"})) for note in reference.parts[0].notes]
    candidate = deepcopy(reference)
    assert metrics(compare_notation(candidate, reference)) == pytest.approx((1, 1, 1, 1))
    candidate.parts[0].notes = [replace(note, articulations=frozenset()) for note in candidate.parts[0].notes]
    result = compare_notation(candidate, reference)
    assert result["pitch"] == 1 and result["rhythm"] == 0
    assert result["dynamics"] == pytest.approx(.8)


def test_chord_notes_in_separate_voices_remain_equivalent():
    reference = control_score()
    reference.parts[0].notes = [replace(note, staff="1", voice="1") for note in reference.parts[0].notes]
    candidate = deepcopy(reference)
    candidate.parts[0].notes = [replace(note, voice="2" if note.pitch < 60 else "1") for note in candidate.parts[0].notes]
    assert metrics(compare_notation(candidate, reference)) == pytest.approx((1, 1, 1, 1))


def test_reference_grouping_preserves_events_and_original_instrument_weight():
    candidate = control_score()
    candidate.parts = [
        Part("Upper", [note for note in candidate.parts[0].notes if note.staff == "1"], candidate.parts[0].marks),
        Part("Lower", [replace(note, staff="1", voice="1") for note in candidate.parts[0].notes if note.staff == "2"], candidate.parts[0].marks),
    ]
    reference = group_reference_parts(candidate, [[0, 1]])
    assert len(reference.parts) == 1 and len(reference.parts[0].notes) == 8
    assert len(candidate.parts) == 2
    assert metrics(compare_notation(candidate, reference)) == pytest.approx((1, 1, 1, 1))
    result = compare_notation(Score(candidate.parts[:1], candidate.tempos, "", ""), reference)
    assert result["pitch"] == pytest.approx(2 / 3)


def test_simultaneous_identical_pitches_can_keep_distinct_dynamic_voices():
    notes = [Note(49, Fraction(index), Fraction(index + 1), None, True, "1", "1") for index in range(4)]
    reference = Score([
        Part("Loud cymbal", notes, [Mark("dynamic", "f", Fraction(0), "1")]),
        Part("Soft cymbal", deepcopy(notes), [Mark("dynamic", "p", Fraction(0), "1")]),
    ], [(Fraction(0), 120)], "", "")
    candidate = deepcopy(reference)
    candidate.parts.reverse()
    grouped = group_reference_parts(reference, [[0, 1]])
    assert metrics(compare_notation(candidate, grouped)) == pytest.approx((1, 1, 1, 1))
    candidate.parts[0].marks = [Mark("dynamic", "f", Fraction(0), "1")]
    assert compare_notation(candidate, grouped)["dynamics"] == 0


def test_simultaneous_unisons_keep_distinct_playing_techniques():
    notes = [Note(60, Fraction(0), Fraction(1), 42, False, "1", "1"), Note(60, Fraction(0), Fraction(1), 42, False, "1", "2", frozenset({"tech:pizzicato"}))]
    reference = Score([Part("Cello", notes)], [(Fraction(0), 120)], "", "")
    candidate = deepcopy(reference)
    candidate.parts[0].notes.reverse()
    assert metrics(compare_notation(candidate, reference)) == (1, 1, 1, 1)
    candidate.parts[0].notes = [replace(note, articulations=frozenset()) for note in candidate.parts[0].notes]
    assert compare_notation(candidate, reference)["instruments"] == .5


@pytest.mark.parametrize("groups", [[[0, 0]], [[0], [0]], [[9]], [[]]])
def test_corrupt_reference_groups_fail_explicitly(groups):
    with pytest.raises(ValueError):
        group_reference_parts(control_score(), groups)


def test_wrong_tempo_and_written_durations_lose_credit():
    reference = control_score()
    candidate = deepcopy(reference)
    candidate.tempos = [(Fraction(0), 90)]
    assert compare_notation(candidate, reference)["rhythm"] < 1
    candidate = deepcopy(reference)
    candidate.parts[0].notes = [replace(note, end=note.end + Fraction(1, 2)) for note in candidate.parts[0].notes]
    result = compare_notation(candidate, reference)
    assert result["pitch"] == 1 and result["rhythm"] == 0


def test_extra_entire_part_loses_credit_without_replacing_a_reference_part():
    reference = control_score()
    candidate = deepcopy(reference)
    candidate.parts.append(Part("Noise", [Note(20, Fraction(20), Fraction(21), 0, False, "1", "1")]))
    result = compare_notation(candidate, reference)
    assert result["unassigned_extra_notes"] == 1
    assert 0 < result["pitch"] < 1


def test_single_tremolo_and_explicit_notes_are_equivalent():
    original = Note(60, Fraction(0), Fraction(1), 48, False, "1", "1", tremolo=3)
    part = Part("Violin", [original], [Mark("dynamic", "f", Fraction(0), "1")])
    reference = Score([part], [(Fraction(0), 120)], "", "")
    candidate = deepcopy(reference)
    candidate.parts[0].notes = [replace(original, start=Fraction(index, 8), end=Fraction(index + 1, 8), tremolo=0) for index in range(8)]
    assert len(expanded_notes(part)) == 8
    assert metrics(compare_notation(candidate, reference)) == pytest.approx((1, 1, 1, 1))


def test_two_note_tremolo_preserves_alternation_and_total_duration():
    first = Note(60, Fraction(0), Fraction(1), 48, False, "1", "1", tremolo=3)
    second = replace(first, pitch=64, start=Fraction(1), end=Fraction(2))
    part = Part("Violin", [first, second], [
        Mark("tremolo", "start", Fraction(0), "1", "1:3", Fraction(1)),
        Mark("tremolo", "stop", Fraction(1), "1", "1:3", Fraction(2)),
    ])
    notes = expanded_notes(part)
    assert len(notes) == 16
    assert [note.pitch for note in notes] == [60, 64] * 8
    assert notes[-1].end == 2
    reference = Score([part], [(Fraction(0), 120)], "", "")
    candidate = deepcopy(reference)
    candidate.parts[0].notes = notes
    candidate.parts[0].marks = []
    assert metrics(compare_notation(candidate, reference)) == pytest.approx((1, 1, 1, 1))


def test_missing_or_reversed_arpeggio_notation_loses_rhythm_credit():
    notes = [Note(pitch, Fraction(0), Fraction(2), 0, False, "1", "1", arpeggio=("1", "up")) for pitch in (60, 64, 67)]
    reference = Score([Part("Piano", notes)], [(Fraction(0), 120)], "", "")
    for arpeggio, expected_rhythm in ((("9", "up"), 1), (("9", "down"), 0), (None, 0)):
        candidate = deepcopy(reference)
        candidate.parts[0].notes = [replace(note, arpeggio=arpeggio) for note in notes]
        result = compare_notation(candidate, reference)
        assert result["pitch"] == result["instruments"] == 1
        assert result["rhythm"] == expected_rhythm
