from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tasks.computing_math.dit_pipeline_cfg_alignment_fid_256_001.scripts import (
    score_outputs,
)
from tasks.physical_sciences._shared.materials_science._common import (
    command_result_parts,
)


ROOT = Path(__file__).resolve().parents[2]
DIT_REFERENCE = (
    ROOT
    / "task-data-hf"
    / "extracted"
    / "computing_math"
    / "dit_pipeline_cfg_alignment_fid_256_001"
    / "base"
    / "reference"
    / "pipeline_dit.py"
)


@pytest.fixture
def reference_text() -> str:
    if not DIT_REFERENCE.exists():
        pytest.skip("DIT release data is not installed")
    return DIT_REFERENCE.read_text(encoding="utf-8")


def test_dit_scorer_provides_declared_diffusers_type(reference_text: str) -> None:
    candidate_text = "from diffusers.models import DiTTransformer2DModel\n" + reference_text

    result = score_outputs.score_submission_text(candidate_text, reference_text)

    assert result.passed, result.to_dict()
    assert result.score == 1.0


def test_dit_scheduler_supports_config_variance_access(reference_text: str) -> None:
    candidate_text = reference_text.replace(
        'getattr(self.scheduler, "variance_type", None)',
        "self.scheduler.config.variance_type",
    )
    assert candidate_text != reference_text
    result = score_outputs.score_submission_text(candidate_text, reference_text)
    assert result.score == 1.0, result.to_dict()
    assert result.passed


@pytest.mark.parametrize(
    "module_name",
    [
        "tasks.physical_sciences.mose2_bse_absorption_soc.main",
        "tasks.physical_sciences.silicon_bse_absorption.main",
    ],
)
@pytest.mark.asyncio
async def test_bse_setup_uses_guest_timeout_and_current_command_result(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(module_name)

    async def skip_setup(_task_cfg, _session) -> None:
        return None

    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        async def run_command(self, command: str, *, check: bool = True) -> dict:
            self.calls.append((command, check))
            return {"return_code": 0, "stdout": "", "stderr": ""}

    session = Session()
    monkeypatch.setattr(module, "_setup", skip_setup)

    await module.start(
        type("TaskConfig", (), {"metadata": {"software_dir": "/opt/task"}})(),
        session,
    )

    assert session.calls == [
        ("timeout --foreground 600s bash /opt/task/install_software.sh", False)
    ]


def test_command_result_parts_accepts_cua_dict_and_object() -> None:
    assert command_result_parts({"return_code": 0, "stdout": "ok", "stderr": ""}) == (0, "ok", "")

    class Result:
        returncode = 2
        stdout = ""
        stderr = "failed"

    assert command_result_parts(Result()) == (2, "", "failed")
