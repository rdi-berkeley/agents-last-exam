"""AgentHLE task: NHANES confounder sensitivity analysis."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local import fallback only

    class _FallbackTask:
        def __init__(self, description, metadata, computer):
            self.description = description
            self.metadata = metadata
            self.computer = computer

    def _identity_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    cb = SimpleNamespace(
        Task=_FallbackTask,
        DesktopSession=object,
        tasks_config=_identity_decorator,
        setup_task=_identity_decorator,
        evaluate_task=_identity_decorator,
    )

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import ScoreResult, score_output_bundle  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "nhanes_confounder_sensitivity_analysis"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    check: bool = False,
) -> dict[str, Any]:
    try:
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command)


def _log_score(result: ScoreResult) -> None:
    logger.info(
        "score=%.6f passed=%s reason=%s hard_gate=%s",
        result.score,
        result.passed,
        result.reason,
        result.hard_gate,
    )
    logger.info("details=%s", json.dumps(result.to_dict(), ensure_ascii=True))


class NHANESConfounderSensitivityConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    OS_TYPE: str = "linux"

    def __init__(self, variant_name: str = VARIANT_NAME) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=variant_name,
            OS_TYPE="linux",
            REMOTE_ROOT_DIR=os.environ.get("REMOTE_ROOT_DIR", "/media/user/data/agenthle"),
            REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def task_spec_file(self) -> str:
        return f"{self.input_dir}/task_spec.txt"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def demographics_file(self) -> str:
        return f"{self.input_dir}/3.demographics_clean_cycle1.csv"

    @property
    def questionnaire_file(self) -> str:
        return f"{self.input_dir}/7.questionnaire_clean_cycle1.csv"

    @property
    def response_file(self) -> str:
        return f"{self.input_dir}/8.response_clean_cycle1.csv"

    @property
    def medications_file(self) -> str:
        return f"{self.input_dir}/5.medications_clean_cycle1.csv"

    @property
    def runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def runtime_pyproject(self) -> str:
        return f"{self.runtime_env_dir}/pyproject.toml"

    @property
    def runtime_lock(self) -> str:
        return f"{self.runtime_env_dir}/uv.lock"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_with_task_deps.sh"

    @property
    def subset_output_file(self) -> str:
        return f"{self.remote_output_dir}/hpyl_nsaid_subset.csv"

    @property
    def summary_output_file(self) -> str:
        return f"{self.remote_output_dir}/sensitivity_summary.csv"

    @property
    def reference_subset_file(self) -> str:
        return f"{self.reference_dir}/hpyl_nsaid_subset.csv"

    @property
    def reference_summary_file(self) -> str:
        return f"{self.reference_dir}/sensitivity_summary.csv"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux precision-health data-analysis task.

Read the staged benchmark brief and output contract first:
- `{self.task_spec_file}`
- `{self.output_contract_file}`

Visible task inputs:
- `{self.demographics_file}`
- `{self.questionnaire_file}`
- `{self.response_file}`
- `{self.medications_file}`

If you want the pinned Python environment, use:
- `{self.python_wrapper}`

Your job is to rebuild the analytic cohort and export exactly these files under `{self.remote_output_dir}`:
- `{self.subset_output_file}`
- `{self.summary_output_file}`

Requirements:
1. Follow the exact cohort construction, variable derivations, and model definitions in `task_spec.txt`.
2. Use `output_contract.json` for the exact output filenames, column order, formulas, and metadata fields.
3. Do not modify `input/`, `reference/`, `output_test_pos/`, or `output_test_neg/`.
4. Write only the required task outputs under `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": self.VARIANT_NAME,
                "task_spec_file": self.task_spec_file,
                "output_contract_file": self.output_contract_file,
                "demographics_file": self.demographics_file,
                "questionnaire_file": self.questionnaire_file,
                "response_file": self.response_file,
                "medications_file": self.medications_file,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lock": self.runtime_lock,
                "python_wrapper": self.python_wrapper,
                "subset_output_file": self.subset_output_file,
                "summary_output_file": self.summary_output_file,
                "reference_subset_file": self.reference_subset_file,
                "reference_summary_file": self.reference_summary_file,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = NHANESConfounderSensitivityConfig()


@cb.tasks_config(split="train")
def load():
    cfg = NHANESConfounderSensitivityConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    required_paths = [
        meta["subset_output_file"],
        meta["summary_output_file"],
        meta["reference_subset_file"],
        meta["reference_summary_file"],
    ]
    missing = [path for path in required_paths if not await session.exists(path)]
    if missing:
        logger.error("Missing evaluation paths: %s", missing)
        return [0.0]

    try:
        candidate_subset = _as_text(await session.read_file(meta["subset_output_file"]))
        candidate_summary = _as_text(await session.read_file(meta["summary_output_file"]))
        reference_subset = _as_text(await session.read_file(meta["reference_subset_file"]))
        reference_summary = _as_text(await session.read_file(meta["reference_summary_file"]))
    except Exception as exc:
        logger.error("Failed to read task outputs/reference files: %s", exc)
        return [0.0]

    result = score_output_bundle(
        candidate_subset_csv=candidate_subset,
        candidate_summary_csv=candidate_summary,
        reference_subset_csv=reference_subset,
        reference_summary_csv=reference_summary,
    )
    _log_score(result)
    return [float(result.score)]
