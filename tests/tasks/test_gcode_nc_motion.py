import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "tasks/engineering/gcode/scripts/nc_motion.py"
SPEC = importlib.util.spec_from_file_location("nc_motion", SCRIPT)
motion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(motion)


def records(*commands):
    for ordinal, (name, arguments) in enumerate(commands, 1):
        yield {"ordinal": ordinal, "source_block": None, "name": name, "arguments": arguments}


def run(*commands, **kwargs):
    kwargs.setdefault("tool_gauge_offsets_mm", {57: (0, 0, 0), 84: (0, 0, 0)})
    return list(
        motion.iter_motions(
            records(("USE_LENGTH_UNITS", "CANON_UNITS_MM"), *commands),
            initial_machine_mm=(0, 0, 0),
            initial_tool=57,
            chord_error_mm=0.01,
            **kwargs,
        )
    )


def test_unit_change_preserves_physical_position_and_feed():
    rows = run(
        ("SET_FEED_RATE", "254"),
        ("STRAIGHT_FEED", "25.4, 0, 0, 0, 0, 0"),
        ("USE_LENGTH_UNITS", "CANON_UNITS_INCHES"),
        ("STRAIGHT_FEED", "2, 0, 0, 0, 0, 0"),
    )
    assert rows[1]["start_mm"] == (25.4, 0, 0)
    assert rows[1]["end_mm"] == (50.8, 0, 0)
    assert rows[1]["feed_mm_min"] == 254


def test_frame_rotation_and_tool_offset_order_and_no_phantom_motion():
    rows = run(
        ("STRAIGHT_TRAVERSE", "2, 3, 4, 0, 0, 0"),
        ("SET_G5X_OFFSET", "2, 10, 20, 30, 0, 0, 0"),
        ("SET_G92_OFFSET", "1, 2, 3, 0, 0, 0"),
        ("SET_XY_ROTATION", "90"),
        ("USE_TOOL_LENGTH_OFFSET", "4 5 6, 0 0 0, 0 0 0"),
        ("STRAIGHT_TRAVERSE", "6, 7, 8, 0, 0, 0"),
    )
    assert len(rows) == 2
    assert rows[1]["start_mm"] == (2, 3, 4)
    assert rows[1]["end_mm"] == pytest.approx((5, 32, 47))
    assert rows[1]["tool_offset_mm"] == (4, 5, 6)


def test_offsets_stay_in_mm_across_units():
    rows = run(
        ("SET_G5X_OFFSET", "1, 25.4, 0, 0, 0, 0, 0"),
        ("USE_TOOL_LENGTH_OFFSET", "0 0 25.4, 0 0 0, 0 0 0"),
        ("USE_LENGTH_UNITS", "CANON_UNITS_INCHES"),
        ("STRAIGHT_TRAVERSE", "1, 0, 1, 0, 0, 0"),
    )
    assert rows[0]["end_mm"] == (50.8, 0, 50.8)


def test_selection_is_not_a_change_and_pocket_is_not_tool_number():
    rows = run(
        ("SELECT_TOOL", "84"),
        ("STRAIGHT_TRAVERSE", "1, 0, 0, 0, 0, 0"),
        ("CHANGE_TOOL", "2"),
        ("STRAIGHT_TRAVERSE", "2, 0, 0, 0, 0, 0"),
        ("CHANGE_TOOL_NUMBER", "1"),
        ("STRAIGHT_TRAVERSE", "3, 0, 0, 0, 0, 0"),
        pocket_tools={1: 57, 2: 84},
    )
    assert [row["tool"] for row in rows] == [57, 84, 57]
    assert rows[-1]["start_mm"] == (2, 0, 0)


@pytest.mark.parametrize(
    "command",
    [
        ("STRAIGHT_PROBE", "1, 0, 0, 0, 0, 0"),
        ("RIGID_TAP", "1, 0, 0"),
        ("NURBS_FEED", "3, ..."),
        ("USER_DEFINED_FUNCTION", "1, 2, 3"),
        ("SET_TOOL_TABLE_ENTRY", "1, 57, 0"),
        ("NOT_A_CANONICAL_EVENT", ""),
        ("STRAIGHT_TRAVERSE", "1, 2, 3, 1, 0, 0"),
        ("USE_TOOL_LENGTH_OFFSET", "0 0 1, 0 0 0, 0 0 1"),
        ("SELECT_PLANE", "CANON_PLANE_UV"),
        ("SET_FEED_MODE", "2, 0"),
        ("SET_SPINDLE_MODE", "0 120"),
        ("STRAIGHT_TRAVERSE", "nan, 0, 0, 0, 0, 0"),
        ("CHANGE_TOOL_NUMBER", "2"),
    ],
)
def test_unsupported_or_invalid_semantics_raise(command):
    with pytest.raises(motion.MotionError, match="Canonical event 2:"):
        run(command)


def test_supported_non_motion_records_are_exposed():
    seen = []
    rows = run(("DWELL", "0.25"), ("FLOOD_ON", ""), ("PROGRAM_STOP", ""), on_event=seen.append)
    assert rows == []
    assert [event["name"] for event in seen] == [
        "USE_LENGTH_UNITS",
        "DWELL",
        "FLOOD_ON",
        "PROGRAM_STOP",
    ]


def test_stream_is_lazy_and_late_failure_is_not_dropped():
    def source():
        yield from records(
            ("USE_LENGTH_UNITS", "CANON_UNITS_MM"),
            ("STRAIGHT_TRAVERSE", "1, 2, 3, 0, 0, 0"),
        )
        raise RuntimeError("input stream failed late")

    rows = motion.iter_motions(
        source(),
        initial_machine_mm=(0, 0, 0),
        initial_tool=57,
        tool_gauge_offsets_mm={57: (0, 0, 0)},
        chord_error_mm=0.01,
    )
    assert next(rows)["end_mm"] == (1, 2, 3)
    with pytest.raises(RuntimeError, match="failed late"):
        next(rows)


def test_feed_per_revolution_is_not_mislabelled_as_feed_per_minute():
    rows = run(
        ("SET_FEED_MODE", "0, 1"),
        ("SET_FEED_RATE", "0.1"),
        ("SET_SPINDLE_SPEED", "0, 1000"),
        ("START_SPINDLE_CLOCKWISE", "0"),
        ("STRAIGHT_FEED", "1, 2, 3, 0, 0, 0"),
    )
    assert rows[0]["feed_mm_min"] is None
    assert rows[0]["feed_mm_rev"] == 0.1
    assert rows[0]["spindle_rpm"] == 1000


def test_post_end_motion_is_rejected():
    with pytest.raises(motion.MotionError, match="after PROGRAM_END"):
        run(("PROGRAM_END", ""), ("STRAIGHT_TRAVERSE", "1, 0, 0, 0, 0, 0"))


def test_zero_turn_arc_is_axial_translation():
    rows = run(
        ("SET_FEED_RATE", "100"),
        ("ARC_FEED", "0, 0, 1, 0, 0, 2, 0, 0, 0"),
    )
    assert len(rows) == 1 and rows[0]["end_mm"] == (0, 0, 2)
    assert rows[0]["chord_error_mm"] == 0


def test_g43_and_uncompensated_programs_follow_the_same_physical_tip_path():
    compensated = run(
        ("USE_TOOL_LENGTH_OFFSET", "0 0 5, 0 0 0, 0 0 0"),
        ("STRAIGHT_TRAVERSE", "1, 2, 3, 0, 0, 0"),
        tool_gauge_offsets_mm={57: (0, 0, 5)},
    )
    uncompensated = run(
        ("STRAIGHT_TRAVERSE", "1, 2, 8, 0, 0, 0"),
        tool_gauge_offsets_mm={57: (0, 0, 5)},
    )
    for field in ("start_mm", "end_mm", "gauge_start_mm", "gauge_end_mm"):
        assert compensated[0][field] == uncompensated[0][field]
    assert compensated[0]["start_mm"] == (0, 0, -5)
    assert compensated[0]["end_mm"] == (1, 2, 3)
    assert compensated[0]["gauge_end_mm"] == (1, 2, 8)


def test_wrong_h_offset_is_not_cancelled_as_if_it_were_the_actual_tool_length():
    rows = run(
        ("USE_TOOL_LENGTH_OFFSET", "0 0 10, 0 0 0, 0 0 0"),
        ("STRAIGHT_TRAVERSE", "1, 2, 3, 0, 0, 0"),
        tool_gauge_offsets_mm={57: (0, 0, 5)},
    )
    assert rows[0]["end_mm"] == (1, 2, 8)


def test_compensation_changes_do_not_teleport_the_physical_tip():
    rows = run(
        ("STRAIGHT_TRAVERSE", "0, 0, 8, 0, 0, 0"),
        ("USE_TOOL_LENGTH_OFFSET", "0 0 5, 0 0 0, 0 0 0"),
        ("STRAIGHT_TRAVERSE", "0, 0, 3, 0, 0, 0"),
        ("USE_TOOL_LENGTH_OFFSET", "0 0 0, 0 0 0, 0 0 0"),
        ("STRAIGHT_TRAVERSE", "0, 0, 8, 0, 0, 0"),
        tool_gauge_offsets_mm={57: (0, 0, 5)},
    )
    assert all(row["end_mm"] == (0, 0, 3) for row in rows)
    assert rows[1]["start_mm"] == rows[2]["start_mm"] == (0, 0, 3)


def test_tool_change_preserves_gauge_and_changes_tip_with_declared_tool_length():
    rows = run(
        ("STRAIGHT_TRAVERSE", "0, 0, 8, 0, 0, 0"),
        ("SELECT_TOOL", "84"),
        ("CHANGE_TOOL", "2"),
        ("STRAIGHT_TRAVERSE", "0, 0, 9, 0, 0, 0"),
        tool_gauge_offsets_mm={57: (0, 0, 5), 84: (0, 0, 10)},
    )
    assert rows[0]["end_mm"] == (0, 0, 3)
    assert rows[1]["start_mm"] == (0, 0, -2)
    assert rows[0]["gauge_end_mm"] == rows[1]["gauge_start_mm"]


def test_missing_calibration_fails_instead_of_guessing_from_compensation():
    with pytest.raises(motion.MotionError, match="no declared gauge calibration"):
        run(("STRAIGHT_TRAVERSE", "0, 0, 1, 0, 0, 0"), tool_gauge_offsets_mm={})
