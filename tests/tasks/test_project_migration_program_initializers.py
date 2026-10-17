import pytest

from tasks.visual_media.project_migration.scripts.project_check import ordered_controls_preserved


def program(value, *, channel=0):
    return {"type": "program_change", "channel": channel, "program": value}


def controller(number, value):
    return {"type": "control_change", "channel": 0, "control": number, "value": value}


NOTE = {"type": "note_on", "channel": 0, "note": 60, "velocity": 90}
ORIGINAL = [(0, program(0)), (10, NOTE), (30, controller(64, 127)), (40, controller(64, 0))]


def test_preserved_original_with_instrument_initializer():
    additions = [(0, controller(0, 0)), (0, controller(32, 0)), (0, program(73))]
    actual = ORIGINAL[:1] + additions + ORIGINAL[1:]
    passed, offset, initializers = ordered_controls_preserved(ORIGINAL, actual)
    assert passed and offset == 0 and initializers == additions


def test_initializer_at_same_tick_before_first_note():
    actual = ORIGINAL[:1] + [(10, program(73))] + ORIGINAL[1:]
    assert ordered_controls_preserved(ORIGINAL, actual)[0]


@pytest.mark.parametrize("actual", [
    [(0, program(73))] + ORIGINAL[1:],
    ORIGINAL[:-1],
    ORIGINAL[:2] + [(10, program(73))] + ORIGINAL[2:],
    ORIGINAL[:2] + [(11, controller(0, 1))] + ORIGINAL[2:],
    ORIGINAL[:1] + [(0, controller(64, 127))] + ORIGINAL[1:],
    ORIGINAL[:1] + [(0, program(73, channel=1))] + ORIGINAL[1:],
    ORIGINAL[:2] + ORIGINAL[3:] + ORIGINAL[2:3],
])
def test_original_controls_and_post_onset_sequence_stay_required(actual):
    assert not ordered_controls_preserved(ORIGINAL, actual)[0]


def test_control_only_material_cannot_gain_program_events():
    original = [(0, program(0))]
    assert not ordered_controls_preserved(original, original + [(0, program(73))])[0]


def test_retained_original_event_time_remains_required():
    actual = [(6, program(0))] + ORIGINAL[1:]
    assert not ordered_controls_preserved(ORIGINAL, actual)[0]


def test_initializer_does_not_consume_identical_later_program_change():
    original = ORIGINAL + [(50, program(73))]
    actual = original[:1] + [(0, program(73))] + original[1:]
    passed, offset, initializers = ordered_controls_preserved(original, actual)
    assert passed and offset == 0 and initializers == [(0, program(73))]
