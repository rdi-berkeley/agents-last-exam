"""AgentHLE task: visual_media/butterfly_flap_animation."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
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

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_outputs import (PASS_THRESHOLD, ScoreResult,  # noqa: E402
                           score_video)

logger = logging.getLogger(__name__)

DOMAIN_NAME = "visual_media"
TASK_NAME = "butterfly_flap_animation"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
OUTPUT_VIDEO_NAME = "output.mp4"
ALLOWED_OUTPUT_DIRS = {"output", "output_test_pos", "output_test_neg"}


def _canonical_output_dir_name(raw: str) -> str:
    normalized = raw.replace("\\", "/").strip("/")
    if normalized not in ALLOWED_OUTPUT_DIRS:
        raise ValueError(
            "REMOTE_OUTPUT_DIR must be one of: " + ", ".join(sorted(ALLOWED_OUTPUT_DIRS))
        )
    return normalized


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    check: bool = False,
) -> dict[str, Any]:
    try:
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command)


def _log_score(result: ScoreResult) -> None:
    logger.info(
        "[%s] score=%.6f passed=%s reason=%s hard_gate=%s",
        TASK_NAME,
        result.score,
        result.passed,
        result.reason,
        result.hard_gate,
    )
    logger.info("[%s] details=%s", TASK_NAME, json.dumps(result.to_dict(), ensure_ascii=True))


@dataclass
class ButterflyFlapConfig(GeneralTaskConfig):
    REMOTE_ROOT_DIR: str = os.environ.get("REMOTE_ROOT_DIR", r"E:\agenthle")
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OUTPUT_VIDEO_NAME: str = OUTPUT_VIDEO_NAME
    PASS_THRESHOLD: float = PASS_THRESHOLD

    @property
    def task_dir(self) -> str:
        return rf"{self.REMOTE_ROOT_DIR}\{self.DOMAIN_NAME}\{self.TASK_NAME}\{self.VARIANT_NAME}"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def source_image(self) -> str:
        return rf"{self.input_dir}\butterfly.jpeg"

    @property
    def task_spec(self) -> str:
        return rf"{self.input_dir}\task_spec.md"

    @property
    def ae_launcher(self) -> str:
        return rf"{self.software_dir}\launch_after_effects.bat"

    @property
    def remote_output_dir(self) -> str:
        return rf"{self.task_dir}\{_canonical_output_dir_name(self.REMOTE_OUTPUT_DIR)}"

    @property
    def remote_output_video(self) -> str:
        return rf"{self.remote_output_dir}\{self.OUTPUT_VIDEO_NAME}"

    @property
    def task_description(self) -> str:
        return rf"""Create a butterfly wing-flap animation in Adobe After Effects.

Remote workspace:
- Task folder: {self.task_dir}

Software:
- Launch Adobe After Effects using: {self.ae_launcher}
- The installed After Effects version may differ from the original submission; use the available AE version opened by this launcher.

Visible inputs:
- Source butterfly artwork: {self.source_image}
- Detailed task spec: {self.task_spec}

What to do:
1. Import the butterfly image into After Effects.
2. Split or mask the butterfly into a central body plus left and right wings.
3. Set wing anchor points near the wing-body joints.
4. Animate the wings with repeated Y-rotation or equivalent perspective deformation.
5. Produce exactly 4 complete wing flap cycles: open -> closed -> open.
6. Move the butterfly along a smooth curved 2.5D-style path through the frame with at least 2 Y-direction turning points.
7. Keep the butterfly texture recognizably tied to the source image at fully open moments.
8. Use a plain dark or black background so the butterfly motion remains clear.

Export requirements:
- Export one H.264 MP4.
- Frame rate: 30 fps.
- Duration: 4.0 to 5.5 seconds.
- Save the final render exactly here: {self.remote_output_video}

Do not modify files under input/. Write only the final MP4 under output/.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "source_image": self.source_image,
                "task_spec": self.task_spec,
                "ae_launcher": self.ae_launcher,
                "output_dir_name": _canonical_output_dir_name(self.REMOTE_OUTPUT_DIR),
                "output_video": self.remote_output_video,
                "output_video_name": self.OUTPUT_VIDEO_NAME,
                "pass_threshold": self.PASS_THRESHOLD,
                "canonical_gcs_root": f"gs://ale-data-all/{TASK_ID}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = ButterflyFlapConfig(REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"))


@cb.tasks_config(split="train")
def load():
    cfg = ButterflyFlapConfig(REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"))
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    required_paths = [meta["source_image"], meta["output_video"]]
    missing = [path for path in required_paths if not await session.exists(path)]
    if missing:
        logger.error("[%s] missing evaluation paths: %s", TASK_NAME, "; ".join(missing))
        return [0.0]

    try:
        with tempfile.TemporaryDirectory(prefix="butterfly_flap_eval_") as tmp_name:
            tmp_dir = Path(tmp_name)
            local_input = tmp_dir / "butterfly.jpeg"
            local_output = tmp_dir / "output.mp4"
            local_input.write_bytes(await session.read_bytes(meta["source_image"]))
            local_output.write_bytes(await session.read_bytes(meta["output_video"]))
            result = score_video(
                local_output, local_input, pass_threshold=float(meta["pass_threshold"])
            )
    except Exception as exc:
        logger.error("[%s] evaluation failure: %s", TASK_NAME, exc)
        return [0.0]

    _log_score(result)
    return [result.score]


if __name__ == "__main__":
    for task in load():
        print(task.description)
