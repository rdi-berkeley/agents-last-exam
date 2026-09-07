"""ALE task: business_finance/ashare_pit_factor_ic_01.

Point-in-time multi-factor construction and rank-IC evaluation on a simulated China A-share
market. Input data (daily bars, cumulative quarterly statements with announcement dates and
restatements, security master, trading calendar, rebalance dates) is staged by the framework under
input/. The hidden reference (factor panel and IC report produced by the normative pipeline in
input/factor_spec.md) is staged under reference/ only at evaluation time.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_factor_outputs import score_submission

logger = logging.getLogger(__name__)

DOMAIN_NAME = "business_finance"
TASK_NAME = "ashare_pit_factor_ic_01"

VARIANTS = [
    ("base", "300 securities, 2022-07 to 2025-12, 30 monthly rebalance dates, seed 20260907."),
    ("variant_2", "320 securities, same period and rules, seed 20260908."),
    ("variant_3", "280 securities, same period and rules, seed 20260909."),
]


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = "base"

    @property
    def data_dir(self) -> str:
        return f"{self.input_dir}/data"

    @property
    def task_brief_file(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def factor_spec_file(self) -> str:
        return f"{self.input_dir}/factor_spec.md"

    @property
    def output_contract_file(self) -> str:
        return f"{self.input_dir}/output_contract.json"

    @property
    def requirements_file(self) -> str:
        return f"{self.input_dir}/runtime_env/requirements.txt"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python.sh"

    @property
    def output_files(self) -> dict[str, str]:
        return {
            "factor_panel.csv": f"{self.remote_output_dir}/factor_panel.csv",
            "ic_report.json": f"{self.remote_output_dir}/ic_report.json",
        }

    @property
    def reference_files(self) -> dict[str, str]:
        return {
            "factor_panel.csv": f"{self.reference_dir}/factor_panel.csv",
            "ic_report.json": f"{self.reference_dir}/ic_report.json",
        }

    @property
    def task_description(self) -> str:
        return f"""\
You are the quant researcher on a China A-share equity team. Build the team's point-in-time factor pipeline from the staged vendor data and evaluate the factors with rank IC.

Task directory: `{self.task_dir}`

Read these first, in this order:
- Task brief: `{self.task_brief_file}`
- Normative factor specification (the grader implements exactly this): `{self.factor_spec_file}`
- Output schema: `{self.output_contract_file}`

Input data under `{self.data_dir}`:
- `daily_bars.csv` (unadjusted OHLC, volume, amount, adj_factor, share counts, trade_status)
- `financials.csv` (quarterly statements, cumulative year-to-date, with announce_date; restatements appear as a second row for the same report_period)
- `securities.csv` (industry, list_date, delist_date)
- `trading_calendar.csv`, `rebalance_dates.csv`

Deliverables, written under `{self.remote_output_dir}`:
- `factor_panel.csv`: one row per (rebalance date, eligible ticker) with the six processed factors `mom_6_1, rev_1m, vol_20, turnover_20, ep_ttm, size` and `fwd_ret_20`
- `ic_report.json`: per-factor rank IC statistics (`mean_ic`, `ic_std`, `icir`, `n_periods`, `by_date`)

Requirements:
- Every statement is usable only from its `announce_date`; restatements replace the original only after their own announcement. Trailing-twelve-month net profit must be assembled from the cumulative year-to-date rows as the specification prescribes.
- Eligibility, winsorisation, industry neutralisation, standardisation, forward-return and IC rules follow `{self.factor_spec_file}` exactly. No look-ahead, no survivorship filtering.
- Python with pandas and numpy is available via `{self.python_wrapper}` (first call creates a virtual environment from `{self.requirements_file}`). Any other tooling is acceptable if the output files match the contract.
- Do not modify files under `{self.input_dir}`.
- The grader recomputes the rank IC from your own panel and rejects a report whose `mean_ic` disagrees with that recomputation.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "data_dir": self.data_dir,
                "task_brief_file": self.task_brief_file,
                "factor_spec_file": self.factor_spec_file,
                "output_contract_file": self.output_contract_file,
                "requirements_file": self.requirements_file,
                "python_wrapper": self.python_wrapper,
                "output_files": self.output_files,
                "reference_files": self.reference_files,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    tasks = []
    for variant_name, _label in VARIANTS:
        cfg = TaskConfig(VARIANT_NAME=variant_name)
        tasks.append(
            cb.Task(
                description=cfg.task_description,
                metadata=cfg.to_metadata(),
                computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
            )
        )
    return tasks


async def _exists(session: cb.DesktopSession, path: str) -> bool:
    return bool(await session.file_exists(path) or await session.directory_exists(path))


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)
    meta = task_cfg.metadata
    out_dir = meta["remote_output_dir"]
    await session.run_command(f"rm -rf {out_dir!r} && mkdir -p {out_dir!r}", check=False)
    await session.run_command(f"chmod +x {meta['python_wrapper']!r}", check=False)
    for path in (
        meta["factor_spec_file"],
        meta["output_contract_file"],
        f"{meta['data_dir']}/daily_bars.csv",
    ):
        if not await _exists(session, path):
            raise RuntimeError(f"staged input missing: {path}")
    if await _exists(session, meta["reference_dir"]):
        raise RuntimeError(
            f"reference directory must not be visible during setup: {meta['reference_dir']}"
        )
    logger.info("[%s] input staged, output dir ready at %s", meta["variant_name"], out_dir)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]

    reference: dict[str, bytes] = {}
    for name, path in meta["reference_files"].items():
        if not await _exists(session, path):
            raise RuntimeError(f"[{tag}] hidden reference missing: {path}")
        reference[name] = await session.read_bytes(path)

    outputs: dict[str, bytes | None] = {}
    for name, path in meta["output_files"].items():
        outputs[name] = await session.read_bytes(path) if await _exists(session, path) else None

    try:
        result = score_submission(
            outputs["factor_panel.csv"],
            outputs["ic_report.json"],
            reference["factor_panel.csv"],
            reference["ic_report.json"],
        )
    except RuntimeError:
        raise
    except Exception:
        logger.exception("[%s] scoring failed on submitted artifacts", tag)
        return [0.0]

    logger.info("[%s] evaluation=%s", tag, json.dumps(result.to_dict(), sort_keys=True))
    return [float(result.score)]


if __name__ == "__main__":
    for task in load():
        print(task.description)
