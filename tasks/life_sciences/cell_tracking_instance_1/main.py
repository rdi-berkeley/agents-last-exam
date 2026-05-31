"""Cell tracking benchmark over a fluorescence microscopy time-lapse sequence."""

import logging
import posixpath
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_setup import BaseTaskSetup
from tasks.life_sciences.cell_tracking_instance_1.eval import (
    EvaluationError,
    evaluate_tracking_submission,
    load_tiff_array,
)
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "life_sciences"
TASK_NAME = "cell_tracking_instance_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
FRAME_COUNT = 30
ALLOWED_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(path: str) -> str:
    normalized = posixpath.normpath(path.replace("\\", "/"))
    if normalized not in ALLOWED_OUTPUT_DIRS:
        raise ValueError(
            "OUTPUT_SUBDIR must normalize to one of: " + ", ".join(sorted(ALLOWED_OUTPUT_DIRS))
        )
    return normalized


@dataclass
class CellTrackingConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = "base"

    @property
    def sequence_dir(self) -> str:
        return f"{self.input_dir}/01"

    @property
    def runtime_manifest(self) -> str:
        return f"{self.input_dir}/runtime_env/pyproject.toml"

    @property
    def runtime_lockfile(self) -> str:
        return f"{self.input_dir}/runtime_env/uv.lock"

    @property
    def python_wrapper(self) -> str:
        return f"{self.software_dir}/python_cell_tracking.sh"

    @property
    def output_dir(self) -> str:
        output_dir_name = _canonical_output_dir_name(self.OUTPUT_SUBDIR)
        return f"{self.task_dir}/{output_dir_name}"

    @property
    def track_output_file(self) -> str:
        return f"{self.output_dir}/res_track.txt"

    @property
    def reference_track_file(self) -> str:
        return f"{self.reference_dir}/01_GT/TRA/man_track.txt"

    @property
    def task_description(self) -> str:
        return f"""You are performing cell tracking on a fluorescence microscopy time-lapse sequence.

Task directory:
- `{self.task_dir}`

Visible inputs:
- 30 grayscale TIFF frames: `{self.sequence_dir}/t000.tif` through `{self.sequence_dir}/t029.tif`
- Optional Python dependency manifest: `{self.runtime_manifest}`
- Optional Python dependency lockfile: `{self.runtime_lockfile}`
- Optional Python wrapper: `{self.python_wrapper}`

Recommended setup:
- Work on Linux in `{self.task_dir}`.
- If you need common image-processing packages, run:
  `UV_PROJECT_ENVIRONMENT=/tmp/agenthle_cell_tracking_runtime_env uv sync --project "{self.input_dir}/runtime_env"`
- You may run scripts through `{self.python_wrapper}`.

Your task:
1. Segment individual cells in every frame.
2. Track cells across all 30 frames so the same cell keeps a consistent positive integer label.
3. Save labeled tracking masks as `{self.output_dir}/mask000.tif` through `{self.output_dir}/mask029.tif`.
4. Save the lineage table as `{self.track_output_file}`.

Output format:
- Masks must be 2D labeled TIFF images with the same dimensions as the input frames.
- Pixel value `0` is background; positive integer values are cell IDs.
- `res_track.txt` must use Cell Tracking Challenge format with four whitespace-separated integer columns per row: `L B E P`.
- `L` is the cell label, `B` is the first frame index, `E` is the last frame index, and `P` is the parent label or `0`.

A successful result should accurately segment the cells and keep their identities consistent over time. Do not modify files under `input/`; write final result files only under `{self.output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "sequence_dir": self.sequence_dir,
                "runtime_manifest": self.runtime_manifest,
                "runtime_lockfile": self.runtime_lockfile,
                "python_wrapper": self.python_wrapper,
                "track_output_file": self.track_output_file,
                "reference_track_file": self.reference_track_file,
                "output_dir_name": _canonical_output_dir_name(self.OUTPUT_SUBDIR),
            }
        )
        return metadata


config = CellTrackingConfig()


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


async def _read_mask(session: cb.DesktopSession, path: str, expected_shape=None):
    return load_tiff_array(await session.read_bytes(path), expected_shape=expected_shape)


def _frame_from_name(path: str, prefix: str) -> int:
    name = posixpath.basename(path)
    match = re.fullmatch(rf"{re.escape(prefix)}(\d{{3}})\.tif", name)
    if not match:
        raise EvaluationError(f"unexpected mask filename: {name}")
    return int(match.group(1))


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        input0 = await _read_mask(session, f'{meta["sequence_dir"]}/t000.tif')
        expected_shape = tuple(input0.shape)

        pred_masks = {}
        for frame in range(FRAME_COUNT):
            mask_path = f'{meta["output_dir"]}/mask{frame:03d}.tif'
            if not (await session.file_exists(mask_path) or await session.directory_exists(mask_path)):
                logger.info("Missing output mask: %s", mask_path)
                return [0.0]
            pred_masks[frame] = await _read_mask(session, mask_path, expected_shape=expected_shape)

        if not (await session.file_exists(meta["track_output_file"]) or await session.directory_exists(meta["track_output_file"])):
            logger.info("Missing res_track.txt: %s", meta["track_output_file"])
            return [0.0]
        pred_track_text = await session.read_file(meta["track_output_file"])

        ref_track_masks = {}
        for frame in range(FRAME_COUNT):
            ref_path = f'{meta["reference_dir"]}/01_GT/TRA/man_track{frame:03d}.tif'
            ref_track_masks[frame] = await _read_mask(
                session, ref_path, expected_shape=expected_shape
            )

        seg_listing = await session.run_command(
            f'find "{meta["reference_dir"]}/01_GT/SEG" -maxdepth 1 -name "man_seg*.tif" | sort'
        )
        seg_paths = [
            line.strip() for line in seg_listing.get("stdout", "").splitlines() if line.strip()
        ]
        ref_seg_masks = {
            _frame_from_name(path, "man_seg"): await _read_mask(
                session, path, expected_shape=expected_shape
            )
            for path in seg_paths
        }
        ref_track_text = await session.read_file(meta["reference_track_file"])

        result = evaluate_tracking_submission(
            pred_masks,
            pred_track_text,
            ref_track_masks,
            ref_track_text,
            ref_seg_masks,
        )
        logger.info("Evaluation result: %s", result)
        return [float(result["score"])]
    except Exception as exc:
        logger.info("Evaluation failed: %s", exc)
        return [0.0]


if __name__ == "__main__":
    for task in load():
        print(task.description)
