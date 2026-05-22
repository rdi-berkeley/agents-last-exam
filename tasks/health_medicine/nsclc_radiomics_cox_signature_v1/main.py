"""Task definition for health_medicine/nsclc_radiomics_cox_signature_v1."""

import json
import logging
import os
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "nsclc_radiomics_cox_signature_v1"
VARIANT_NAME = "base"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


class NSCLCRadiomicsConfig(LinuxTaskConfig):
    def __init__(self, *, REMOTE_OUTPUT_DIR: str | None = None) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=REMOTE_OUTPUT_DIR or os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def risk_scores_file(self) -> str:
        return f"{self.remote_output_dir}/risk_scores.csv"

    @property
    def ground_truth_file(self) -> str:
        return f"{self.reference_dir}/ground_truth.csv"

    @property
    def task_description(self) -> str:
        return f"""\
You are building an Aerts-2014-style radiomic prognostic risk model for \
non-small-cell lung cancer.

## Input

Use the data under `{self.input_dir}`:

- `images/` contains 422 lung CT NIfTI volumes (`.nii.gz`).
- `masks/` contains matched GTV-1 tumor segmentation masks for the usable cohort.
- `clinical/train_clinical.csv` contains survival labels for the training patients only.
- `prediction_ids.csv` lists held-out patients that need risk scores.
- `split.json` documents the fixed train/test split and the one excluded missing-mask case.
- `runtime_env/pyproject.toml` pins a Python 3.10-compatible scientific stack for this task.

## Task

Build any CPU-only radiomics survival pipeline you choose. You may use \
PyRadiomics, SimpleITK, lifelines, scikit-learn, pandas, and NumPy. Extract \
features from the CT/mask pairs, fit a Cox-style or other survival-risk model \
using the labeled training patients, and generate risk scores for the held-out \
patients in `prediction_ids.csv`.

## Required Output

Write exactly these files under `{self.remote_output_dir}`:

1. `risk_scores.csv` with one row for each held-out patient and columns:
   - `PatientID`
   - `risk_score`

Higher `risk_score` must mean higher predicted hazard / shorter survival. \
Do not use hidden test labels or modify input files. CPU time budget is 2 hours.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "risk_scores_file": self.risk_scores_file,
                "ground_truth_file": self.ground_truth_file,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    cfg = NSCLCRadiomicsConfig()
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
    for path in (meta["risk_scores_file"],):
        result = await session.run_command(f'test -f "{path}" && echo OK || echo MISSING')
        stdout = result.get("output", result.get("stdout", ""))
        if "MISSING" in stdout:
            logger.info("missing output file: %s", path)
            return [0.0]

    await session.run_command(f'mkdir -p "{EVAL_TMP_DIR}"')
    await session.write_file(f"{EVAL_TMP_DIR}/verify_outputs.py", _read_script("verify_outputs.py"))
    result = await session.run_command(
        f'python "{EVAL_TMP_DIR}/verify_outputs.py" '
        f'--output-dir "{meta["remote_output_dir"]}" '
        f'--reference-dir "{meta["reference_dir"]}"'
    )
    stdout = result.get("output", result.get("stdout", ""))
    try:
        data = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        logger.error("verify_outputs.py did not produce JSON: %s", stdout)
        return [0.0]
    logger.info("NSCLC radiomics eval result: %s", data)
    return [float(data.get("score", 0.0))]
