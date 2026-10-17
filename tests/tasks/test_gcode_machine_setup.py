import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "gcode_machine_setup",
    Path(__file__).parents[2] / "tasks/engineering/gcode/scripts/machine_setup.py",
)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def test_initial_clearance_precedes_first_tool_installation(tmp_path):
    program = tmp_path / "saved.ngc"
    original = b"G21 G90\nT91 M6\nG0 Z5.15\nM2\n"
    program.write_bytes(original)
    output = tmp_path / "replay.ngc"
    receipt = setup.prepare_program(program, output, stock_top_mm=0.15, clearance_mm=5)
    result = output.read_bytes()
    assert result.endswith(original)
    assert result.index(b"G53 G0 Z5.150000000000") < result.index(b"T91 M6")
    assert receipt["initial_tool"] == 0
    assert receipt["initial_machine_mm"] == [0, 0, 0]
    assert receipt["unloaded_reposition_mm"] == [0, 0, 5.15]
    assert program.read_bytes() == original


@pytest.mark.parametrize("clearance", [0, -1, float("nan"), float("inf")])
def test_invalid_clearance_cannot_publish_a_replay(tmp_path, clearance):
    program = tmp_path / "saved.ngc"
    program.write_text("M2\n")
    output = tmp_path / "replay.ngc"
    with pytest.raises(ValueError):
        setup.prepare_program(program, output, stock_top_mm=0.15, clearance_mm=clearance)
    assert not output.exists()


def test_existing_program_and_evidence_are_not_overwritten(tmp_path):
    program = tmp_path / "saved.ngc"
    program.write_text("M2\n")
    with pytest.raises(ValueError, match="overwrite"):
        setup.prepare_program(program, program, stock_top_mm=0.15, clearance_mm=5)
    output = tmp_path / "replay.ngc"
    output.write_text("retained evidence")
    with pytest.raises(FileExistsError):
        setup.prepare_program(program, output, stock_top_mm=0.15, clearance_mm=5)
    assert output.read_text() == "retained evidence"
