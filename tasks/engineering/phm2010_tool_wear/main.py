"""engineering/phm2010_tool_wear (Linux).

Predict per-cut flank wear of milling cutter c4 from its force / vibration / acoustic-emission signals,
given two fully labelled cutters (c1, c6) for training. Deliverable: output/c4_wear_pred.csv.
The hidden reference is the measured c4 wear; evaluate() maps MAE to a 0-1 score.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.linux_runtime import LinuxTaskConfig
from tasks.engineering.phm2010_tool_wear.scripts.grade import parse_wear_csv, score_wear

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "phm2010_tool_wear"
VARIANT_NAME = "base"


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def train_dir(self) -> str:
        return f"{self.input_dir}/train"

    @property
    def test_cuts_dir(self) -> str:
        return f"{self.input_dir}/test/c4/cuts"

    @property
    def readme_path(self) -> str:
        return f"{self.input_dir}/README.md"

    @property
    def output_csv(self) -> str:
        return f"{self.remote_output_dir}/c4_wear_pred.csv"

    @property
    def reference_csv(self) -> str:
        return f"{self.reference_dir}/c4_wear.csv"

    @property
    def task_description(self) -> str:
        return (
            "You are a machining process engineer building a tool-condition-monitoring model for a high-speed CNC "
            "milling operation.\n\n"
            "## Input\n"
            f"- Training cutters c1 and c6 in `{self.train_dir}/<cutter>/`: `cuts/cut_001.csv` ... `cut_315.csv` hold the "
            "raw signals of one cut each (about 127,000 rows sampled at 50 kHz; columns Fx_N, Fy_N, Fz_N, Vx_g, Vy_g, Vz_g, "
            "AE_RMS_V), and `wear.csv` gives the flank wear measured after each cut on the three flutes (cut,flute_1,"
            "flute_2,flute_3 in micrometres).\n"
            f"- Test cutter c4 in `{self.test_cuts_dir}/`: the same 315 cut files, but no wear file.\n"
            f"- `{self.readme_path}` repeats these instructions.\n\n"
            "## What you must do\n"
            "1. Engineer features from the signals (statistics, spectral content, cumulative trends across cuts, ...) and "
            "fit a model on c1 and c6.\n"
            "2. Predict the flank wear of c4 after every cut for all three flutes.\n"
            f"3. Write `{self.output_csv}` with header `cut,flute_1,flute_2,flute_3` and exactly 315 rows for cuts 1..315, "
            "values in micrometres.\n\n"
            "## Scoring\n"
            "Mean absolute error over the 945 predicted values against the measured wear: MAE <= 6 um scores 1.0, "
            "MAE >= 24 um scores 0.0, linear in between. A missing, malformed or incomplete file scores 0."
        )

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({
            "train_dir": self.train_dir,
            "test_cuts_dir": self.test_cuts_dir,
            "readme_path": self.readme_path,
            "output_csv": self.output_csv,
            "reference_csv": self.reference_csv,
        })
        return m


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [cb.Task(
        description=cfg.task_description,
        metadata=cfg.to_metadata(),
        computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
    )]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    await session.run_command(f"mkdir -p {meta['remote_output_dir']!r} && rm -f {meta['output_csv']!r}", check=False)
    for path in (f"{meta['train_dir']}/c1/wear.csv", f"{meta['train_dir']}/c6/wear.csv", meta["readme_path"]):
        try:
            await session.read_file(path)
        except Exception as exc:
            raise RuntimeError(f"staged input missing: {path} ({exc})")
    listing = await session.run_command(f"ls {meta['test_cuts_dir']!r} | wc -l", check=False)
    n = int((listing.get("stdout") if isinstance(listing, dict) else getattr(listing, "stdout", "0")).strip() or 0)
    if n < 315:
        raise RuntimeError(f"expected 315 c4 cut files, found {n}")
    if await session.file_exists(meta["reference_csv"]):
        raise RuntimeError("reference wear file is visible to the agent; staging order is wrong")
    logger.info("[%s] input staged; reference hidden", TASK_NAME)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        ref = parse_wear_csv(await session.read_file(meta["reference_csv"]))
    except Exception as exc:
        raise RuntimeError(f"reference unreadable at {meta['reference_csv']}: {exc}")
    if not ref:
        raise RuntimeError("reference wear file is malformed")
    try:
        text = await session.read_file(meta["output_csv"])
    except Exception as exc:
        logger.info("[%s] no output at %s: %s", TASK_NAME, meta["output_csv"], exc)
        return [0.0]
    result = score_wear(parse_wear_csv(text, expect_cuts=len(ref)), ref)
    logger.info("[%s] %s MAE=%s um RMSE=%s um score=%.4f", TASK_NAME, result["gate"], result["mae_um"], result["rmse_um"], result["score"])
    return [result["score"]]
