"""Windows manuscript-translation task for a classical Yi inscription."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

# cua_bench loads task modules via exec_module without always pre-registering
# them in sys.modules; dataclass needs this for string annotation handling.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.education_info.yi_manuscript_translation_1.scripts.score_outputs import (
    evaluate_submission,
)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "education_info"
TASK_NAME = "yi_manuscript_translation_1"
VARIANT_NAME = "base"
ALLOWED_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}


def _output_dir_name(remote_output_dir: str) -> str:
    return remote_output_dir.strip().strip("\\/") or "output"


@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "windows"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def output_dir_name(self) -> str:
        return _output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        return rf"{self.task_dir}\{self.output_dir_name}"

    @property
    def question_file(self) -> str:
        return rf"{self.input_dir}\question.txt"

    @property
    def image_file(self) -> str:
        return rf"{self.input_dir}\inscription.png"

    @property
    def reference_materials_dir(self) -> str:
        return rf"{self.input_dir}\reference_materials"

    @property
    def viewer_launcher(self) -> str:
        return rf"{self.software_dir}\launch_viewer.cmd"

    @property
    def output_files(self) -> dict[str, str]:
        return {
            "bounding_box": rf"{self.remote_output_dir}\bounding_box.json",
            "report": rf"{self.remote_output_dir}\report.txt",
        }

    @property
    def hidden_reference_bbox(self) -> str:
        return rf"{self.reference_dir}\reference_bbox.json"

    @property
    def hidden_ground_truth(self) -> str:
        return rf"{self.reference_dir}\ground_truth.json"

    @property
    def hidden_reference_materials_snapshot(self) -> str:
        return rf"{self.reference_dir}\reference_materials_snapshot.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are analyzing a classical Yi manuscript image on a Windows VM.

## Task Folder
`{self.task_dir}`

## Visible Input Files
- Task brief: `{self.question_file}`
- Manuscript image: `{self.image_file}`
- Bundled source texts: `{self.reference_materials_dir}`

## Software
- Image launcher: `{self.viewer_launcher}`
- Use any plain-text editor already present on the VM for the final report and JSON output

## What You Must Do
1. Read `{self.question_file}`.
2. Open the manuscript image with `{self.viewer_launcher}` and inspect the phonetic gloss in the upper-left margin.
3. Determine which single glyph the gloss refers to by combining the image inspection with the bundled files in `{self.reference_materials_dir}`.
4. Save the target glyph bounding box exactly to:
   `{self.output_files["bounding_box"]}`
5. Save the final structured report exactly to:
   `{self.output_files["report"]}`

## Output Requirements
- `bounding_box.json` must be UTF-8 JSON with integer keys `x1`, `y1`, `x2`, `y2`
- coordinates must use the original image pixels, not screenshot coordinates
- `report.txt` must contain these seven section headers exactly:
  `EXAMINATION`
  `SOURCE IMAGE`
  `TECHNIQUE`
  `IDENTIFICATION OF TARGET CHARACTER POSITION`
  `REFERENCE MATERIALS CONSULTED`
  `TRANSLATION`
  `IMPRESSION`
- Do not use the internet or external sources; rely only on the bundled files
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "question_file": self.question_file,
                "image_file": self.image_file,
                "reference_materials_dir": self.reference_materials_dir,
                "viewer_launcher": self.viewer_launcher,
                "output_dir_name": self.output_dir_name,
                "output_files": self.output_files,
                "hidden_reference_bbox": self.hidden_reference_bbox,
                "hidden_ground_truth": self.hidden_ground_truth,
                "hidden_reference_materials_snapshot": self.hidden_reference_materials_snapshot,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    bbox_path = meta["output_files"]["bounding_box"]
    report_path = meta["output_files"]["report"]
    reference_bbox_path = meta["hidden_reference_bbox"]
    ground_truth_path = meta["hidden_ground_truth"]
    reference_snapshot_path = meta["hidden_reference_materials_snapshot"]

    try:
        bbox_bytes = await session.read_bytes(bbox_path)
        report_bytes = await session.read_bytes(report_path)
        reference_bbox_bytes = await session.read_bytes(reference_bbox_path)
        ground_truth_bytes = await session.read_bytes(ground_truth_path)
        reference_snapshot_bytes = await session.read_bytes(reference_snapshot_path)
    except Exception as exc:  # pragma: no cover - remote I/O failure path
        logger.info("required output/reference file missing: %s", exc)
        return [0.0]

    result = evaluate_submission(
        bbox_bytes=bbox_bytes,
        report_bytes=report_bytes,
        reference_bbox_bytes=reference_bbox_bytes,
        ground_truth_bytes=ground_truth_bytes,
        reference_materials_snapshot_bytes=reference_snapshot_bytes,
    )
    logger.info("evaluation result: %s", json.dumps(result, sort_keys=True, ensure_ascii=False))
    return [float(result.get("score", 0.0))]


if __name__ == "__main__":
    for task in load():
        print(task.description)
