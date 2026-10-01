from __future__ import annotations

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[2]
TASK_DATA = (
    ROOT
    / "task-data-hf"
    / "extracted"
    / "engineering"
    / "sumo_urban_am_peak_calibration"
    / "base"
)
EVALUATOR = TASK_DATA / "reference" / "evaluator_only"

pytestmark = pytest.mark.skipif(
    not EVALUATOR.is_dir(), reason="Requires the gated SUMO evaluator fixture"
)


def _load_rebuild_module():
    spec = importlib.util.spec_from_file_location("sumo_rebuild_all", EVALUATOR / "rebuild_all.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rebuild_uses_staged_prompt_when_evaluator_copy_is_absent() -> None:
    module = _load_rebuild_module()
    assert module.task_prompt_path() == TASK_DATA / "input" / "task_prompt.md"
    assert module.starter_project_path() == TASK_DATA / "input" / "starter_project"


def test_agent_package_uses_current_five_hour_prompt() -> None:
    archive = EVALUATOR / "submission" / "agent_starter_project.zip"
    assert archive.exists()
    with ZipFile(archive) as zipped:
        prompt = zipped.read("agent_input/TASK_PROMPT.md").decode()
    assert "5 hours of agent wall-time budget" in prompt
    assert "up to 5 hours" in prompt
    assert "8 hours" not in prompt
    with ZipFile(archive) as zipped:
        members = set(zipped.namelist())
    expected = {
        "agent_input/TASK_PROMPT.md",
        "agent_input/calibration_report.schema.json",
    }
    expected.update(
        f"agent_input/starter_project/{path.relative_to(TASK_DATA / 'input' / 'starter_project').as_posix()}"
        for path in (TASK_DATA / "input" / "starter_project").rglob("*")
        if path.is_file()
    )
    assert members == expected


def test_prompt_discloses_all_twelve_hard_gates() -> None:
    prompt = (TASK_DATA / "input" / "task_prompt.md").read_text()
    for gate in range(1, 13):
        assert f"| {gate} |" in prompt
    assert "RMSE / mean observed flow ≤ 12%" in prompt
    assert "GEH < 5 on at least 85%" in prompt
    assert "every declared and hidden seed completes without an error" in prompt
