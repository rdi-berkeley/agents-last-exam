from copy import deepcopy
from fractions import Fraction

import pretty_midi
import pytest

from tasks.visual_media.music_transcription.main import TrackInfo
from tasks.visual_media.music_transcription.notation import Note, Part, Score
from tasks.visual_media.music_transcription.performance_dynamics import compare_performance_dynamics


def example():
    notation = Score([Part("Piano", [Note(60 + index, Fraction(index), Fraction(index + 1), 0, False, "1", "1") for index in range(6)])], [(Fraction(0), 120)], "", "")
    notes = [pretty_midi.Note(velocity=value, pitch=60 + index, start=index / 2, end=(index + .8) / 2) for index, value in enumerate((40, 50, 70, 80, 65, 35))]
    return notation, [TrackInfo("Piano", 0, False, notes, [])]


def test_same_performance_and_scaled_volume_keep_credit():
    notation, reference = example()
    candidate = deepcopy(reference)
    for event in candidate[0].notes:
        event.velocity += 10
    assert compare_performance_dynamics(notation, reference, candidate, [0])[0]["score"] == pytest.approx(1)


@pytest.mark.parametrize("mode", ["flat", "reversed", "absent"])
def test_missing_or_wrong_performance_expression_loses_credit(mode):
    notation, reference = example()
    candidate = deepcopy(reference)
    if mode == "absent":
        candidate = []
    else:
        for event in candidate[0].notes:
            event.velocity = 64 if mode == "flat" else 127 - event.velocity
    assert compare_performance_dynamics(notation, reference, candidate, [0])[0]["score"] == 0


def test_controller_expression_and_note_velocity_can_be_equivalent():
    notation, reference = example()
    candidate = deepcopy(reference)
    for event in candidate[0].notes:
        candidate[0].control_changes.append(pretty_midi.ControlChange(11, event.velocity, event.start))
        event.velocity = 127
    assert compare_performance_dynamics(notation, reference, candidate, [0])[0]["score"] == pytest.approx(1)


def test_split_piano_tracks_and_track_names_do_not_change_credit():
    notation, reference = example()
    candidate = [TrackInfo("Upper", 0, False, deepcopy(reference[0].notes[:3]), []), TrackInfo("Lower", 0, False, deepcopy(reference[0].notes[3:]), [])]
    assert compare_performance_dynamics(notation, reference, candidate, [0])[0]["score"] == pytest.approx(1)


def test_partial_music_cannot_borrow_full_expression_credit():
    notation, reference = example()
    candidate = deepcopy(reference)
    candidate[0].notes = candidate[0].notes[:3]
    assert compare_performance_dynamics(notation, reference, candidate, [0])[0]["score"] == pytest.approx(2 / 3)


def test_bad_reference_is_an_explicit_error():
    notation, candidate = example()
    with pytest.raises(ValueError, match="Reference MIDI"):
        compare_performance_dynamics(notation, [], candidate, [0])


def test_roll_expression_accepts_different_complete_attack_rates():
    notation = Score([Part("Drum", [Note(38, Fraction(index * 2), Fraction(index * 2 + 2), None, True, "1", "1", tremolo=3) for index in range(3)])], [(Fraction(0), 120)], "", "")
    performances = []
    for strikes in (16, 20):
        notes = [pretty_midi.Note(velocity, 38, phrase + strike / strikes, phrase + (strike + .8) / strikes) for phrase, velocity in enumerate((40, 90, 60)) for strike in range(strikes)]
        performances.append([TrackInfo("Snare", None, True, notes, [])])
    assert compare_performance_dynamics(notation, performances[0], performances[1], [0])[0]["score"] == pytest.approx(1)
    for note in performances[1][0].notes:
        note.velocity = 70
    assert compare_performance_dynamics(notation, performances[0], performances[1], [0])[0]["score"] == 0


def test_grace_expression_accepts_anticipated_and_delayed_playback():
    notes = [Note(62, Fraction(1), Fraction(1), 0, False, "1", "1", frozenset({"grace"}), grace_order=0), Note(64, Fraction(1), Fraction(1), 0, False, "1", "1", frozenset({"grace"}), grace_order=1), Note(60, Fraction(1), Fraction(3), 0, False, "1", "1")]
    notation = Score([Part("Piano", notes)], [(Fraction(0), 120)], "", "")
    performances = []
    for starts in ((.35, .425, .5), (.5, .575, .65)):
        performed = [pretty_midi.Note(velocity, pitch, start, start + .05) for velocity, pitch, start in zip((40, 60, 90), (62, 64, 60), starts, strict=True)]
        performances.append([TrackInfo("Piano", 0, False, performed, [])])
    assert compare_performance_dynamics(notation, performances[0], performances[1], [0])[0]["score"] == pytest.approx(1)


def test_duplicate_voices_may_share_the_louder_simultaneous_attack():
    notes = [Note(60, Fraction(index), Fraction(index + 1), 0, False, "1", str(voice)) for index in range(3) for voice in (1, 2)]
    notation = Score([Part("Piano", notes)], [(Fraction(0), 120)], "", "")
    source_notes = [pretty_midi.Note(velocity, 60, index / 2, (index + .8) / 2) for index, velocities in enumerate(((40, 50), (80, 90), (60, 70))) for velocity in velocities]
    candidate_notes = [pretty_midi.Note(velocity, 60, index / 2, (index + .8) / 2) for index, velocity in enumerate((50, 90, 70))]
    source = [TrackInfo("Piano", 0, False, source_notes, [])]
    candidate = [TrackInfo("Piano", 0, False, candidate_notes, [])]
    assert compare_performance_dynamics(notation, source, candidate, [0])[0]["score"] == pytest.approx(1)
