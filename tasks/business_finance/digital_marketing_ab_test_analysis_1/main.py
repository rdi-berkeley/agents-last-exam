"""AgentHLE task: digital_marketing_ab_test_analysis_1."""

from __future__ import annotations

import json
import logging
import math
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
from tasks.business_finance.digital_marketing_ab_test_analysis_1.report_judge import (
    MAX_REPORT_CHARS,
    judge_report,
)
from tasks.linux_runtime import LinuxTaskConfig
from tasks.utils.evaluation import JudgeInfrastructureError

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
   Test against equal 1:1 allocation using Pearson chi-square with
   1 degree of freedom (no continuity correction); `srm_pass` is True
   exactly when `srm_pvalue > 0.01`.

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
   exclusion rules). If both arms have the same boundary rate (both 0%
   or both 100%), `z_statistic` and `p_value_raw` may be left blank as
   undefined, or reported as the conventional `0.0` and `1.0`. Use
   `p_value_raw=1.0` when ranking that metric for BH correction.
   Use the pooled, two-sided two-proportion z test without continuity correction
   and unpooled Wald 95% confidence intervals. `absolute_lift` is the treatment
   rate minus the control rate; `relative_lift_pct` is the relative increase
   over control expressed as a percentage. `significant_at_05` means p < 0.05.
   Apply Benjamini-Hochberg step-up correction to the three secondary metrics
   at FDR 0.05. `bh_rank` is the 1-based ascending raw-p-value rank,
   `bh_threshold` is its BH critical value, and `bh_significant` is the
   resulting step-up rejection decision.

3. `experiment_report.md` — Markdown report that includes:
   the required per-arm sample size (from a power analysis on
   `historical_metrics.csv`), a clear **Recommendation** section
   containing the word "ship" or "hold", and the observed absolute
   lift on the primary metric. Recommend **ship** only when the primary
   metric is significant at 0.05 and the treatment unsubscribe lift is
   at most 0.5 percentage points; otherwise recommend **hold**.
   For power, use equal 1:1 allocation, two-sided alpha=0.05, 80% power,
   and the arithmetic mean of historical `open_rate` as the baseline rate.
   The minimum detectable effect is an absolute increase of 0.03.
   Report the normal-approximation two-proportion per-arm sample size,
   rounded up to an integer. Cohen's h normal-power analysis with the same
   inputs is also supported.
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
            "--external-report",
            "--report-byte-limit",
            str(MAX_REPORT_CHARS * 4),
        ]
    )
    result = await _run_command(session, cmd, check=False)
    if result.get("return_code", 0) != 0:
        raise RuntimeError(
            "digital marketing verifier failed: "
            f"{(result.get('stderr') or result.get('stdout') or '')[:1000]}"
        )
    try:
        payload = _parse_json_stdout(result.get("stdout") or "")
        score = float(payload["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score is not finite or outside [0, 1]")
        report_score = float(payload["component_scores"]["report"])
        if report_score not in {0.0, 0.2}:
            raise ValueError("invalid report component")
        missing_files = payload["missing_files"]
        if not isinstance(missing_files, list):
            raise ValueError("invalid missing file list")
        sample_bounds = tuple(payload["sample_bounds"])
        if len(sample_bounds) != 2 or not all(type(value) is int for value in sample_bounds):
            raise ValueError("invalid sample-size bounds")
        expected_lift = float(payload["primary_absolute_lift"])
        expected_recommendation = payload["expected_recommendation"]
        if not math.isfinite(expected_lift) or expected_recommendation not in {"ship", "hold"}:
            raise ValueError("invalid report reference facts")
        report_size = payload["report_size"]
        if type(report_size) is not int or report_size < 0:
            raise ValueError("invalid report size")
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeError("invalid digital marketing verifier output") from exc
    logger.info(
        "digital marketing verifier payload: %s", json.dumps(payload, ensure_ascii=False)[:2000]
    )
    if "report" in missing_files or report_size == 0:
        return [round(score - report_score, 10)]
    if report_size > MAX_REPORT_CHARS * 4:
        raise JudgeInfrastructureError("Report exceeds judge input capacity")
    report_path = _remote_join(meta["remote_output_dir"], "experiment_report.md")
    report = await session.read_file(report_path)
    if not isinstance(report, str):
        raise JudgeInfrastructureError("Report could not be read as text")
    receipt = await judge_report(
        report,
        sample_bounds=sample_bounds,
        expected_lift=expected_lift,
        expected_recommendation=expected_recommendation,
    )
    logger.info("digital marketing report judge: %s", json.dumps(receipt, ensure_ascii=False))
    return [round(score - report_score + (0.2 if receipt["passed"] else 0.0), 10)]
