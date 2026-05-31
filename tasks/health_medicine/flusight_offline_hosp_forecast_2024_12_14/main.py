"""AgentHLE task: flusight_offline_hosp_forecast_2024_12_14."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.health_medicine.flusight_offline_hosp_forecast_2024_12_14.scripts.score_outputs import (
    score_submission_bundle,
)


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)


async def _missing(session: cb.DesktopSession, path: str, *, label: str) -> bool:
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("Missing %s: %s", label, path)
    return True


@dataclass
class FluSightOfflineHospForecastConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "health_medicine"
    TASK_NAME: str = "flusight_offline_hosp_forecast_2024_12_14"
    VARIANT_NAME: str = "base"
    OUTPUT_SUBDIR: str = os.environ.get("OUTPUT_SUBDIR", "output")

    @property
    def task_prompt_file(self) -> str:
        return f"{self.input_dir}/task_prompt.md"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def task_instructions_file(self) -> str:
        return f"{self.input_dir}/TASK_INSTRUCTIONS.md"

    @property
    def schema_file(self) -> str:
        return f"{self.input_dir}/forecast_output_schema.md"

    @property
    def template_file(self) -> str:
        return f"{self.input_dir}/forecast_output_template.csv"

    @property
    def historical_file(self) -> str:
        return f"{self.input_dir}/historical_weekly_hospital_admissions_asof_2024-12-14.csv"

    @property
    def locations_file(self) -> str:
        return f"{self.input_dir}/locations.csv"

    @property
    def source_provenance_file(self) -> str:
        return f"{self.input_dir}/source_provenance.md"

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
    def bootstrap_wrapper(self) -> str:
        return f"{self.software_dir}/bootstrap_runtime.sh"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_with_task_deps.sh"

    @property
    def output_submission(self) -> str:
        return f"{self.output_dir}/submission.csv"

    @property
    def evaluator_script(self) -> str:
        return f"{self.reference_dir}/evaluator_only/evaluate_submission.py"

    @property
    def truth_file(self) -> str:
        return f"{self.reference_dir}/evaluator_only/ground_truth_future_weeks.csv"

    @property
    def scoring_baseline_file(self) -> str:
        return f"{self.reference_dir}/scoring_baseline.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to produce a FluSight-style offline influenza hospitalization forecast.

Visible task files:
- `{self.task_prompt_file}`
- `{self.output_contract_file}`
- `{self.task_instructions_file}`
- `{self.schema_file}`
- `{self.template_file}`
- `{self.historical_file}`
- `{self.locations_file}`
- `{self.source_provenance_file}`
- `{self.runtime_pyproject}`
- `{self.runtime_lock}`
- `{self.bootstrap_wrapper}`
- `{self.python_wrapper}`

What you must do:
1. Read `{self.task_prompt_file}` and `{self.output_contract_file}` first.
2. Use the archived historical snapshot in `{self.historical_file}` to forecast weekly admissions for the 53 required jurisdictions and four required target weeks.
3. Fill the staged template contract exactly and write one final file under `{self.output_dir}`:
   - `{self.output_submission}`
4. If you want the staged Python environment, materialize it with `{self.bootstrap_wrapper}` and run Python with `{self.python_wrapper}`.

Rules:
- Do not use the internet or external files.
- Do not modify files under `{self.input_dir}`.
- Output must contain exactly 212 rows with the required columns and integer constraints.
- Write only the required `submission.csv` into the writable output directory.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": f"{self.DOMAIN_NAME}/{self.TASK_NAME}",
                "task_prompt_file": self.task_prompt_file,
                "output_contract_file": self.output_contract_file,
                "task_instructions_file": self.task_instructions_file,
                "schema_file": self.schema_file,
                "template_file": self.template_file,
                "historical_file": self.historical_file,
                "locations_file": self.locations_file,
                "source_provenance_file": self.source_provenance_file,
                "runtime_env_dir": self.runtime_env_dir,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lock": self.runtime_lock,
                "bootstrap_wrapper": self.bootstrap_wrapper,
                "python_wrapper": self.python_wrapper,
                "output_submission": self.output_submission,
                "evaluator_script": self.evaluator_script,
                "truth_file": self.truth_file,
                "scoring_baseline_file": self.scoring_baseline_file,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


config = FluSightOfflineHospForecastConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    for key, label in [
        ("output_submission", "submission CSV"),
        ("evaluator_script", "hidden evaluator script"),
        ("truth_file", "hidden truth CSV"),
        ("scoring_baseline_file", "scoring baseline JSON"),
    ]:
        if not (await session.file_exists(meta[key]) or await session.directory_exists(meta[key])):
            logger.error("Missing %s at %s", label, meta[key])
            return [0.0]

    with tempfile.TemporaryDirectory(prefix="flusight_offline_hosp_forecast_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_submission = tmp / "submission.csv"
        local_evaluator = tmp / "evaluate_submission.py"
        local_truth = tmp / "ground_truth_future_weeks.csv"
        local_baseline = tmp / "scoring_baseline.json"

        try:
            local_submission.write_bytes(await session.read_bytes(meta["output_submission"]))
            local_evaluator.write_bytes(await session.read_bytes(meta["evaluator_script"]))
            local_truth.write_bytes(await session.read_bytes(meta["truth_file"]))
            local_baseline.write_bytes(await session.read_bytes(meta["scoring_baseline_file"]))
            result = score_submission_bundle(
                submission_path=local_submission,
                evaluator_script_path=local_evaluator,
                truth_path=local_truth,
                scoring_baseline_path=local_baseline,
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Evaluation failed: %s", exc)
            return [0.0]

    logger.info("evaluation=%s", json.dumps(result, sort_keys=True)[:2000])
    return [float(result.get("score", 0.0))]
