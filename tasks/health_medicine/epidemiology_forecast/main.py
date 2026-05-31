"""AgentHLE task: epidemiology_forecast."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local fallback only

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

from verify_outputs import ScoreResult, verify_output_bundle  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "epidemiology_forecast"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"


def _remote_join(*parts: str) -> str:
    return str(PurePosixPath(*parts))


def _as_text(payload: Any) -> str:
    return payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)


class EpidemiologyForecastConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    def __init__(self) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
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
    def forecasts_file(self) -> str:
        return _remote_join(self.input_dir, "forecasts_2021-22.parquet")

    @property
    def truth_file(self) -> str:
        return _remote_join(self.input_dir, "truth_2021-22.csv")

    @property
    def runtime_pyproject(self) -> str:
        return _remote_join(self.runtime_env_dir, "pyproject.toml")

    @property
    def runtime_lock(self) -> str:
        return _remote_join(self.runtime_env_dir, "uv.lock")

    @property
    def submission_output(self) -> str:
        return _remote_join(self.output_dir, "submission.csv")

    @property
    def per_cell_output(self) -> str:
        return _remote_join(self.output_dir, "per_cell_scores.csv")

    @property
    def reference_submission(self) -> str:
        return _remote_join(self.reference_dir, "expected_table1_2021-22.csv")

    @property
    def reference_per_cell(self) -> str:
        return _remote_join(self.reference_dir, "expected_per_cell_2021-22.csv")

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to produce the 2021-22 CDC FluSight influenza-hospitalization forecast skill scorecard.

Visible task files:
- `{self.forecasts_file}`
- `{self.truth_file}`
- `{self.runtime_pyproject}`
- `{self.runtime_lock}`

Write exactly these outputs under `{self.output_dir}`:
- `{self.submission_output}`
- `{self.per_cell_output}`

Implementation rules:
- Use IEEE 754 double precision throughout.
- Keep only `forecast_date` in the closed interval `[2022-02-21, 2022-06-20]`.
- Keep only targets `1 wk ahead inc flu hosp` through `4 wk ahead inc flu hosp`.
- Keep only `type == "quantile"`.
- Drop every row with `location == "US"` before aggregation.
- Join truth on `(location, target_end_date)` using `{self.truth_file}` exactly as staged.
- Quantile levels are the 23 values:
  `0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990`.
- Round quantile levels to 3 decimals before matching.
- Apply the round-outward rule to forecast values before scoring:
  `q < 0.5 -> floor(value)`, `q > 0.5 -> ceil(value)`, `q == 0.5 -> round half to even`.
- A model is eligible iff its non-US quantile-row count is at least 75% of `Flusight-baseline`'s count after the window/type filters.
- For each eligible `(model, location, target_end_date, target)` cell, compute:
  `wis`, `ae_median`, `cov50_hit`, `cov95_hit`.
- Use the proper-score WIS normalization `1 / (K + 1/2)` with `K = 11` and alphas
  `0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90`.
- If any required quantile is missing for a cell, treat that cell's `wis` and `ae_median` as NaN, then aggregate with `fillna(0).mean()`.
- Aggregate per model into `wis`, `mae`, `cov50`, and `cov95`.
- Compute `rel_wis` from pairwise overlapping-cell WIS ratios using the geometric mean over all eligible models, including the self-ratio of 1, then normalize by `Flusight-baseline`.

Output requirements:
- `per_cell_scores.csv` columns:
  `model, location, target_end_date, target, wis, ae_median, cov50_hit, cov95_hit`
- `submission.csv` columns:
  `model, wis, rel_wis, mae, cov50, cov95`
- `submission.csv` must contain exactly one row per eligible model.
- `per_cell_scores.csv` must contain exactly one row per eligible scored cell.

Runtime guidance:
- The staged `input/runtime_env/` folder is the agent-facing dependency manifest.
- If you need `numpy`, `pandas`, or `pyarrow`, install them from the staged UV project rather than relying on global packages.
- You may materialize the task-local runtime by running the canonical entry point `{self.software_dir}/bootstrap_runtime.sh`, which invokes `uv sync --project input/runtime_env` and writes the venv under `input/runtime_env/.venv`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "runtime_env_dir": self.runtime_env_dir,
                "software_dir": self.software_dir,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "output_dir": self.output_dir,
                "forecasts_file": self.forecasts_file,
                "truth_file": self.truth_file,
                "runtime_pyproject": self.runtime_pyproject,
                "runtime_lock": self.runtime_lock,
                "submission_output": self.submission_output,
                "per_cell_output": self.per_cell_output,
                "reference_submission": self.reference_submission,
                "reference_per_cell": self.reference_per_cell,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = EpidemiologyForecastConfig()


@cb.tasks_config(split="train")
def load():
    cfg = EpidemiologyForecastConfig()
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


def _log_score(result: ScoreResult) -> None:
    logger.info(
        "[%s] score=%.3f passed=%s reason=%s hard_gate=%s",
        TASK_NAME,
        result.score,
        result.passed,
        result.reason,
        result.hard_gate,
    )
    logger.info("[%s] details=%s", TASK_NAME, json.dumps(result.to_dict(), ensure_ascii=False))


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    required_outputs = [meta["submission_output"], meta["per_cell_output"]]
    missing_outputs = [path for path in required_outputs if not (await session.file_exists(path) or await session.directory_exists(path))]
    if missing_outputs:
        logger.error("[%s] missing output files: %s", TASK_NAME, "; ".join(missing_outputs))
        return [0.0]

    try:
        result = verify_output_bundle(
            submission_csv=_as_text(await session.read_file(meta["submission_output"])),
            per_cell_csv=_as_text(await session.read_file(meta["per_cell_output"])),
            reference_submission_csv=_as_text(
                await session.read_file(meta["reference_submission"])
            ),
            reference_per_cell_csv=_as_text(await session.read_file(meta["reference_per_cell"])),
        )
    except Exception as exc:
        logger.error("[%s] evaluation failure: %s", TASK_NAME, exc)
        return [0.0]

    _log_score(result)
    return [result.score]
