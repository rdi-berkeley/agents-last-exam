"""AgentHLE task: Obermeyer healthcare bias reproduction."""

from __future__ import annotations

import json
import logging
import sys
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

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import ScoreResult, score_output_bundle  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "obermeyer_bias_reproduction"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAMES = ("base", "variant_1")


def _remote_join(*parts: str) -> str:
    return str(PurePosixPath(*parts))


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    check: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, check=check, timeout=timeout)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


class ObermeyerBiasConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    OS_TYPE: str = "linux"

    def __init__(self, variant_name: str = "base") -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=variant_name,
            OS_TYPE="linux",
        )

    @property
    def runtime_env_dir(self) -> str:
        return _remote_join(self.input_dir, "runtime_env")

    @property
    def output_test_pos_dir(self) -> str:
        return _remote_join(self.task_dir, "output_test_pos")

    @property
    def output_test_neg_dir(self) -> str:
        return _remote_join(self.task_dir, "output_test_neg")

    @property
    def output_dir(self) -> str:
        if self.OUTPUT_SUBDIR == "output_test_pos":
            return self.output_test_pos_dir
        if self.OUTPUT_SUBDIR == "output_test_neg":
            return self.output_test_neg_dir
        return _remote_join(self.task_dir, self.OUTPUT_SUBDIR)

    @property
    def analysis_data_file(self) -> str:
        return _remote_join(self.input_dir, "analysis_data.csv")

    @property
    def dummy_test_file(self) -> str:
        return _remote_join(self.input_dir, "dummy_test.csv")

    @property
    def instructions_file(self) -> str:
        return _remote_join(self.input_dir, "task_instructions.md")

    @property
    def runtime_pyproject(self) -> str:
        return _remote_join(self.runtime_env_dir, "pyproject.toml")

    @property
    def runtime_lock(self) -> str:
        return _remote_join(self.runtime_env_dir, "uv.lock")

    @property
    def python_wrapper(self) -> str:
        return _remote_join(self.software_dir, "python_with_task_deps.sh")

    @property
    def predictions_output(self) -> str:
        return _remote_join(self.output_dir, "full_predictions.csv")

    @property
    def baseline_report_output(self) -> str:
        return _remote_join(self.output_dir, "baseline_analysis_report.md")

    @property
    def revised_report_output(self) -> str:
        return _remote_join(self.output_dir, "revised_analysis_report.md")

    @property
    def reference_metrics_file(self) -> str:
        return _remote_join(self.reference_dir, "reference_metrics.json")

    @property
    def reference_predictions_file(self) -> str:
        return _remote_join(self.reference_dir, "full_predictions.csv")

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to reproduce an Obermeyer-style healthcare algorithmic bias audit.

Read the task instructions and cohort data staged under:
- `{self.instructions_file}`
- `{self.analysis_data_file}`
- `{self.dummy_test_file}`

The helper `{self.python_wrapper}` runs Python from the staged UV runtime environment if you want pandas and numpy available. It stores its virtual environment and cache under the writable output directory.

Your job is to:
1. Audit the observed healthcare risk score using `risk_score_t` directly as the baseline ranking.
2. Build a reproducible counterfactual revised ranking based on medical need, not by manually assigning scores by race.
3. Show whether Black patients are sicker at similar baseline risk and whether White patients incur higher cost at similar health burden.
4. Save exactly these files under `{self.output_dir}`:
   - `{self.predictions_output}`
   - `{self.baseline_report_output}`
   - `{self.revised_report_output}`

`full_predictions.csv` must have exactly these columns:
`patient_id,baseline_score,revised_score`

"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": self.VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "runtime_env_dir": self.runtime_env_dir,
                "software_dir": self.software_dir,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "output_dir": self.output_dir,
                "analysis_data_file": self.analysis_data_file,
                "dummy_test_file": self.dummy_test_file,
                "instructions_file": self.instructions_file,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lock": self.runtime_lock,
                "python_wrapper": self.python_wrapper,
                "predictions_output": self.predictions_output,
                "baseline_report_output": self.baseline_report_output,
                "revised_report_output": self.revised_report_output,
                "reference_metrics_file": self.reference_metrics_file,
                "reference_predictions_file": self.reference_predictions_file,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = ObermeyerBiasConfig()


@cb.tasks_config(split="train")
def load():
    tasks = []
    for variant_name in VARIANT_NAMES:
        cfg = ObermeyerBiasConfig(variant_name)
        tasks.append(
            cb.Task(
                description=cfg.task_description,
                metadata=cfg.to_metadata(),
                computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
            )
        )
    return tasks


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _log_score(variant_name: str, result: ScoreResult) -> None:
    logger.info(
        "[%s/%s] score=%.6f passed=%s reason=%s hard_gate=%s",
        TASK_NAME,
        variant_name,
        result.score,
        result.passed,
        result.reason,
        result.hard_gate,
    )
    logger.info(
        "[%s/%s] details=%s",
        TASK_NAME,
        variant_name,
        json.dumps(result.to_dict(), ensure_ascii=False),
    )


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if not (await session.file_exists(meta["predictions_output"]) or await session.directory_exists(meta["predictions_output"])):
        logger.error("[%s] missing output: %s", TASK_NAME, meta["predictions_output"])
        return [0.0]

    for ref_key in ("analysis_data_file", "reference_metrics_file", "reference_predictions_file"):
        if not (await session.file_exists(meta[ref_key]) or await session.directory_exists(meta[ref_key])):
            raise RuntimeError(
                f"[{TASK_NAME}] evaluator-controlled {ref_key} missing: {meta[ref_key]}"
            )

    baseline_report = ""
    revised_report = ""
    if (await session.file_exists(meta["baseline_report_output"]) or await session.directory_exists(meta["baseline_report_output"])):
        baseline_report = _as_text(await session.read_file(meta["baseline_report_output"]))
    if (await session.file_exists(meta["revised_report_output"]) or await session.directory_exists(meta["revised_report_output"])):
        revised_report = _as_text(await session.read_file(meta["revised_report_output"]))

    result = score_output_bundle(
        predictions_csv=_as_text(await session.read_file(meta["predictions_output"])),
        analysis_data_csv=_as_text(await session.read_file(meta["analysis_data_file"])),
        reference_metrics_json=_as_text(await session.read_file(meta["reference_metrics_file"])),
        reference_predictions_csv=_as_text(
            await session.read_file(meta["reference_predictions_file"])
        ),
        baseline_report_md=baseline_report,
        revised_report_md=revised_report,
    )
    _log_score(meta["variant_name"], result)
    return [result.score]
