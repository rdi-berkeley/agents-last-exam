"""AgentHLE task: digital_marketing_ab_test_analysis_1."""

from __future__ import annotations

import json
import logging
import os
import shlex
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

logger = logging.getLogger(__name__)

DOMAIN_NAME = "business_finance"
TASK_NAME = "digital_marketing_ab_test_analysis_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"


def _remote_join(*parts: str) -> str:
    return str(PurePosixPath(*parts))


async def _run_command(
    session: cb.DesktopSession, command: str, *, check: bool = False
) -> dict[str, Any]:
    return await session.run_command(command, check=check)


def _parse_json_stdout(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("verifier returned empty stdout")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"unable to parse verifier JSON: {text[:1000]}")


class DigitalMarketingABTestConfig(LinuxTaskConfig):
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
            REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def output_test_pos_dir(self) -> str:
        return _remote_join(self.task_dir, "output_test_pos")

    @property
    def output_test_neg_dir(self) -> str:
        return _remote_join(self.task_dir, "output_test_neg")

    @property
    def runtime_env_dir(self) -> str:
        return _remote_join(self.input_dir, "runtime_env")

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM to complete a digital marketing A/B test analysis.

Visible task files:
- `{self.input_dir}/experiment_brief.md`
- `{self.input_dir}/historical_metrics.csv`
- `{self.input_dir}/eligible_population.csv`
- `{self.input_dir}/exclusion_rules.yaml`
- `{self.input_dir}/active_experiment_customers.csv`
- `{self.input_dir}/experiment_results_raw.csv`
- `{self.input_dir}/runtime_env/pyproject.toml`
- `{self.input_dir}/runtime_env/uv.lock`

Write exactly these outputs under `{self.remote_output_dir}`:

1. `randomization_assignment.csv` — CSV with columns `metric,value`.
   Required rows (one per metric): `n_control`, `n_treatment`, `ratio`
   (treatment/control ratio), `srm_chi2`, `srm_pvalue`, `srm_pass`.
   Compute the sample-ratio mismatch (SRM) chi-squared test over the
   **full** `experiment_results_raw.csv` population (do NOT filter by
   exclusion rules — use every row in the raw results file).

2. `experiment_results.tsv` — tab-separated file with exactly 4 rows
   (one per metric: `opened_rate`, `clicked_rate`, `converted_rate`,
   `unsubscribed_rate`). Columns: `metric`, `is_primary` (True/False),
   `control_rate`, `treatment_rate`, `absolute_lift`,
   `relative_lift_pct`, `ci_lower_95`, `ci_upper_95`, `z_statistic`,
   `p_value_raw`, `significant_at_05` (True/False), `bh_rank`,
   `bh_threshold`, `bh_significant` (True/False). The last three
   Benjamini-Hochberg columns apply only to secondary metrics; leave
   them blank for the primary metric. Rates are computed over delivered
   emails from the **full** `experiment_results_raw.csv` (do NOT apply
   exclusion rules).

3. `experiment_report.md` — Markdown report that includes:
   the required per-arm sample size (from a power analysis on
   `historical_metrics.csv`), a clear **Recommendation** section
   containing the word "ship" or "hold", and the observed absolute
   lift on the primary metric.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "variant_name": VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "reference_dir": self.reference_dir,
                "software_dir": self.software_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "remote_output_dir": self.remote_output_dir,
                "runtime_env_dir": self.runtime_env_dir,
                "canonical_gcs_root": "gs://ale-data-all/business_finance/digital_marketing_ab_test_analysis_1/base/",
            }
        )
        return metadata


config = DigitalMarketingABTestConfig()


@cb.tasks_config(split="train")
def load():
    cfg = DigitalMarketingABTestConfig()
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


def _read_script(name: str) -> str:
    return (Path(__file__).resolve().parent / "scripts" / name).read_text(encoding="utf-8")


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    await session.interface.create_dir(EVAL_TMP_DIR)
    verify_script_path = f"{EVAL_TMP_DIR}/verify_ab_test_outputs.py"
    await session.write_file(verify_script_path, _read_script("verify_ab_test_outputs.py"))
    cmd = " ".join(
        [
            "python3",
            shlex.quote(verify_script_path),
            "--input-dir",
            shlex.quote(meta["input_dir"]),
            "--reference-dir",
            shlex.quote(meta["reference_dir"]),
            "--output-dir",
            shlex.quote(meta["remote_output_dir"]),
        ]
    )
    result = await _run_command(session, cmd, check=False)
    try:
        payload = _parse_json_stdout(
            (result.get("stdout") or "") + "\n" + (result.get("stderr") or "")
        )
    except ValueError:
        logger.error("verifier stdout: %s", result.get("stdout", "")[:1000])
        logger.error("verifier stderr: %s", result.get("stderr", "")[:1000])
        return [0.0]
    if result.get("return_code", 0) != 0:
        logger.error("verifier command failed: %s", payload)
        return [0.0]
    logger.info(
        "digital marketing verifier payload: %s", json.dumps(payload, ensure_ascii=False)[:2000]
    )
    return [float(payload.get("score", 0.0))]
