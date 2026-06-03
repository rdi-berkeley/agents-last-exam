"""AgentHLE task: yeast_colony_detection."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_colonies import ColonyScoreResult, score_colony_outputs  # noqa: E402

logger = logging.getLogger(__name__)

DOMAIN_NAME = "life_sciences"
TASK_NAME = "yeast_colony_detection"
VARIANT_NAME = "base"
REFERENCE_CSV_NAME = "RedColonies.csv"


def _as_text(payload: Any) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8")
    return str(payload)


class YeastColonyDetectionConfig(LinuxTaskConfig):

    @property
    def plate_image(self) -> str:
        return f"{self.input_dir}/6-1.jpg"

    @property
    def plate_mask(self) -> str:
        return f"{self.input_dir}/PlateTemplate.png"

    @property
    def cellprofiler_entry(self) -> str:
        return f"{self.software_dir}/cellprofiler"

    @property
    def answer_file(self) -> str:
        return f"{self.remote_output_dir}/answer.json"

    @property
    def measurements_dir(self) -> str:
        return f"{self.remote_output_dir}/measurements"

    @property
    def measurements_file(self) -> str:
        return f"{self.measurements_dir}/{REFERENCE_CSV_NAME}"

    @property
    def reference_measurements_file(self) -> str:
        return f"{self.reference_dir}/ground_truth/{REFERENCE_CSV_NAME}"

    @property
    def task_description(self) -> str:
        return f"""You are analyzing a yeast colony plate image on Linux.

## Your Task
Detect the red yeast colonies growing on the agar plate while excluding white dots and other visual noise.

## Visible Inputs
- Plate image: `{self.plate_image}`
- Plate-region mask: `{self.plate_mask}`
- CellProfiler entry point, if provisioned: `{self.cellprofiler_entry}`

## What You Must Do
1. Use the plate image and mask to identify red yeast colonies inside the plate region.
2. Create a reproducible detection workflow in CellProfiler 4.2.8 or another image-analysis workflow available on the VM.
3. Count the detected red colonies.
4. Save all required outputs under `{self.remote_output_dir}`.

## Required Outputs
- JSON count file: `{self.answer_file}`
  - exact schema: `{{"colony_count": <integer>}}`
- Measurement CSV: `{self.measurements_file}`
  - include one row per detected red colony
  - include numeric centroid columns named `Location_Center_X` and `Location_Center_Y`
  - include `ObjectNumber` if your workflow can provide it

## Constraints
- Do not modify files under `{self.input_dir}`.
- Only read the visible files under `{self.input_dir}`.
- Keep all task-produced files inside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "data_task_dir": self.data_task_dir,
                "input_dir": self.input_dir,
                "software_dir": self.software_dir,
                "plate_image": self.plate_image,
                "plate_mask": self.plate_mask,
                "cellprofiler_entry": self.cellprofiler_entry,
                "answer_file": self.answer_file,
                "measurements_dir": self.measurements_dir,
                "measurements_file": self.measurements_file,
                "reference_measurements_file": self.reference_measurements_file,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return metadata


config = YeastColonyDetectionConfig(
    DOMAIN_NAME=DOMAIN_NAME,
    TASK_NAME=TASK_NAME,
    VARIANT_NAME=VARIANT_NAME,
)


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _log_score(result: ColonyScoreResult) -> None:
    logger.info("Colony evaluation: %s", json.dumps(result.to_dict(), sort_keys=True))


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    for path in (meta["answer_file"], meta["measurements_file"]):
        if not (await session.file_exists(path) or await session.directory_exists(path)):
            logger.error("agent missing output: %s", path)
            return [0.0]

    if not (await session.file_exists(meta["reference_measurements_file"]) or await session.directory_exists(meta["reference_measurements_file"])):
        raise RuntimeError(
            f"evaluator-controlled reference missing: {meta['reference_measurements_file']}"
        )

    answer_text = _as_text(await session.read_file(meta["answer_file"]))
    prediction_csv = _as_text(await session.read_file(meta["measurements_file"]))
    reference_csv = _as_text(await session.read_file(meta["reference_measurements_file"]))
    result = score_colony_outputs(answer_text, prediction_csv, reference_csv)
    _log_score(result)
    return [float(result.score)]
