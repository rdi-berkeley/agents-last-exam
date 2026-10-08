from copy import deepcopy
from fractions import Fraction

import pytest

from tasks.visual_media.music_transcription.notation import Mark, Note, Part, Score
from tasks.visual_media.music_transcription.notation_scoring import compare_notation


def example(marks):
    notes = [Note(pitch, Fraction(index), Fraction(index + 1), 40, False, "1", "1") for index, pitch in enumerate((60, 62, 64, 65))]
    return Score([Part("Violin", notes, marks)], [(Fraction(0), 120)], "", "")


def test_slur_and_legato_text_have_equivalent_phrasing():
    reference = example([Mark("slur", "start", Fraction(0), "1"), Mark("slur", "stop", Fraction(3), "1")])
    candidate = example([Mark("words", "legato", Fraction(0), "1")])
    assert compare_notation(candidate, reference)["rhythm"] == pytest.approx(1)
    candidate.parts[0].marks = []
    result = compare_notation(candidate, reference)
    assert result["pitch"] == 1 and result["rhythm"] == 0


@pytest.mark.parametrize("original,equivalent", [("pizz.", "pizzicato"), ("con sord.", "muted"), ("harm.", "natural harmonic"), ("flautando", "flautato")])
def test_standard_technique_synonyms_receive_the_same_credit(original, equivalent):
    reference = example([Mark("words", original, Fraction(0), "1")])
    candidate = example([Mark("words", equivalent, Fraction(0), "1")])
    assert compare_notation(candidate, reference)["instruments"] == 1
    candidate.parts[0].marks = []
    assert compare_notation(candidate, reference)["instruments"] < 1


def test_cancelled_mute_and_plucking_do_not_affect_later_notes():
    reference = example([Mark("words", "pizzicato", Fraction(0), "1"), Mark("words", "arco", Fraction(2), "1")])
    candidate = deepcopy(reference)
    assert compare_notation(candidate, reference)["instruments"] == 1
    candidate.parts[0].marks.pop()
    assert compare_notation(candidate, reference)["instruments"] == .5


def test_harmonic_symbol_and_text_are_equivalent():
    reference = example([Mark("words", "harm.", Fraction(0), "1")])
    candidate = example([])
    candidate.parts[0].notes[0].articulations = frozenset({"tech:harmonic"})
    assert compare_notation(candidate, reference)["instruments"] == 1


def test_dangling_slur_is_not_silently_accepted():
    reference = example([Mark("slur", "start", Fraction(0), "1")])
    with pytest.raises(ValueError, match="Unterminated slur"):
        compare_notation(reference, reference)


def test_grace_slur_at_one_written_onset_accepts_stop_first_storage_order():
    from tasks.visual_media.music_transcription.expression import expressive_notes

    grace = Note(62, Fraction(0), Fraction(0), 40, False, "1", "1", frozenset({"grace"}))
    principal = Note(60, Fraction(0), Fraction(1), 40, False, "1", "1")
    part = Part("Violin", [grace, principal], [Mark("slur", "stop", Fraction(0), "1"), Mark("slur", "start", Fraction(0), "1")])
    assert all("legato" in note.articulations for note in expressive_notes(part, part.notes))
