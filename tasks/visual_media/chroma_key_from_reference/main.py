"""Chroma-key family task under the canonical media task layout."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.visual_media.chroma_key_from_reference._shared_eval import (
    ROI_INPUT_GATE_THRESHOLD, SOFT_GATE_PASS_THRESHOLD,
    compute_final_frame_result, ps_literal, remote_child)
from tasks.visual_media.chroma_key_from_reference.scripts.local_soft_eval import \
    run_soft_eval_local

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
REMOTE_HARD_EVAL_SCRIPT = THIS_DIR / "scripts" / "remote_hard_eval.py"

REMOTE_TEMP_ROOT = os.environ.get(
    "CHROMA_REMOTE_EVAL_ROOT",
    r"C:\Users\User\AppData\Local\Temp\agenthle_eval\chroma",
)
DEFAULT_OUTPUT_VIDEO_NAME = os.environ.get("TARGET_VIDEO_NAME", "output.mp4")
SAMPLE_COUNT = 5
SOFT_EVAL_MODEL = os.environ.get("CHROMA_SOFT_EVAL_MODEL", "gpt-5.2")


@dataclass(frozen=True)
class ChromaVariantSpec:
    dir_name: str
    task_tag: str
    remote_task_dir_name: str
    input_video_name: str
    input_hint_image_name: str
    reference_video_name: str
    software_backend: str = "DaVinci Resolve"
    software_launcher_name: str = "DaVinci Resolve.lnk"


VARIANTS = [
    ChromaVariantSpec(
        dir_name="crane_flyingsky",
        task_tag="crane_flyingsky",
        remote_task_dir_name="crane_flyingsky",
        input_video_name="input.mp4",
        input_hint_image_name="input.png",
        reference_video_name="reference.mp4",
    ),
    ChromaVariantSpec(
        dir_name="dolphin_underwater",
        task_tag="dolphin_underwater",
        remote_task_dir_name="dolphin_underwater",
        input_video_name="input.mp4",
        input_hint_image_name="input.png",
        reference_video_name="reference.mp4",
    ),
    ChromaVariantSpec(
        dir_name="explosion_firesmoke",
        task_tag="explosion_firesmoke",
        remote_task_dir_name="explosion_firesmoke",
        input_video_name="input.mp4",
        input_hint_image_name="input.png",
        reference_video_name="reference.mp4",
    ),
    ChromaVariantSpec(
        dir_name="jerry_stage",
        task_tag="jerry_stage",
        remote_task_dir_name="jerry_stage",
        input_video_name="input.mp4",
        input_hint_image_name="input.png",
        reference_video_name="reference.mp4",
    ),
    ChromaVariantSpec(
        dir_name="nezha_courtyard",
        task_tag="nezha_courtyard",
        remote_task_dir_name="nezha_courtyard",
        input_video_name="input.mp4",
        input_hint_image_name="input.png",
        reference_video_name="reference.mp4",
    ),
    ChromaVariantSpec(
        dir_name="polarbear_snowmountain",
        task_tag="polarbear_snowmountain",
        remote_task_dir_name="polarbear_snowmountain",
        input_video_name="input.mp4",
        input_hint_image_name="input.png",
        reference_video_name="reference.mp4",
    ),
]

VARIANT_MAP = {spec.task_tag: spec for spec in VARIANTS}


@dataclass
class ChromaTaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "chroma_key_from_reference"
    VARIANT_NAME: str = ""
    REMOTE_TASK_DIR_NAME: str = ""
    INPUT_VIDEO_NAME: str = ""
    INPUT_HINT_IMAGE_NAME: str = ""
    REFERENCE_VIDEO_NAME: str = ""
    SOFTWARE_BACKEND: str = "DaVinci Resolve"
    SOFTWARE_LAUNCHER_NAME: str = "DaVinci Resolve.lnk"

    OUTPUT_VIDEO_NAME: str = DEFAULT_OUTPUT_VIDEO_NAME
    EVAL_MODEL: str = SOFT_EVAL_MODEL
    DEFAULT_SAMPLE_COUNT: int = SAMPLE_COUNT

    @property
    def software_launcher_path(self) -> str:
        return rf"{self.software_dir}\{self.SOFTWARE_LAUNCHER_NAME}"

    @property
    def remote_input_video(self) -> str:
        return rf"{self.task_dir}\input\{self.INPUT_VIDEO_NAME}"

    @property
    def remote_input_hint_image(self) -> str:
        return rf"{self.task_dir}\input\{self.INPUT_HINT_IMAGE_NAME}"

    @property
    def remote_reference_video(self) -> str:
        return rf"{self.reference_dir}\{self.REFERENCE_VIDEO_NAME}"

    @property
    def remote_breakpoint_json(self) -> str:
        return rf"{self.reference_dir}\breakpoint.json"

    @property
    def remote_output_video(self) -> str:
        return rf"{self.output_dir}\{self.OUTPUT_VIDEO_NAME}"

    @property
    def task_description(self) -> str:
        return rf"""
Goal: Remove the green-screen background from the source clip, create a new Resolve project yourself, composite the correct foreground subject into the intended target scene indicated by the first-frame target screenshot, and export the finished composite video.

Remote workspace:
- Task folder: {self.task_dir}

Software:
- Launch {self.SOFTWARE_BACKEND} using: {self.software_launcher_path}
- No starter `.drp` project is packaged with this task. Create a new Resolve project yourself for the task run.

Inputs (already on remote desktop):
- Input assets live under: {self.task_dir}\input
- Foreground source clip with green background: {self.remote_input_video}
- First-frame target screenshot / disambiguation hint: {self.remote_input_hint_image}

What to do:
- Create a new Resolve project and import the task assets from the task folder.
- Key out the green background from the foreground source clip.
- Build the composite yourself in Resolve so the result matches the intended target scene shown by the first-frame screenshot.
- Use the screenshot to determine which foreground subject to preserve and how the finished composite should look in placement, scale, and overall composition.
- Treat the screenshot as a visual specification for the background scene, framing, and composition that your final render should match.
- Do not export the untouched source clip and do not leave visible green-screen background in the final render.

Output:
- Export/render to: {self.remote_output_video}
"""

    def to_metadata(self) -> dict:
        md = super().to_metadata()
        md.update(
            {
                "variant_name": self.VARIANT_NAME,
                "software_backend": self.SOFTWARE_BACKEND,
                "software_launcher_name": self.SOFTWARE_LAUNCHER_NAME,
                "software_launcher_path": self.software_launcher_path,
                "input_video": self.remote_input_video,
                "input_hint_image": self.remote_input_hint_image,
                "output_video": self.remote_output_video,
                "sample_count": self.DEFAULT_SAMPLE_COUNT,
                "soft_eval_model": self.EVAL_MODEL,
                "remote_task_dir_name": self.REMOTE_TASK_DIR_NAME,
                "remote_task_dir": self.task_dir,
            }
        )
        return md


def _build_config(spec: ChromaVariantSpec) -> ChromaTaskConfig:
    return ChromaTaskConfig(
        VARIANT_NAME=spec.task_tag,
        REMOTE_TASK_DIR_NAME=spec.remote_task_dir_name,
        INPUT_VIDEO_NAME=spec.input_video_name,
        INPUT_HINT_IMAGE_NAME=spec.input_hint_image_name,
        REFERENCE_VIDEO_NAME=spec.reference_video_name,
        SOFTWARE_BACKEND=spec.software_backend,
        SOFTWARE_LAUNCHER_NAME=spec.software_launcher_name,
    )


def _parse_json_from_stdout(stdout: str) -> dict | None:
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            continue
    return None


async def _run_remote_hard_eval(
    task_cfg, session: cb.DesktopSession
) -> tuple[float, list[dict], list[dict], dict]:
    task_tag = task_cfg.metadata["variant_name"]
    spec = VARIANT_MAP[task_tag]
    eval_cfg = _build_config(spec)
    output_video = task_cfg.metadata["output_video"]
    reference_video = eval_cfg.remote_reference_video
    input_video = task_cfg.metadata["input_video"]
    if not (await session.file_exists(output_video) or await session.directory_exists(output_video)):
        return 0.0, [], [], {"reason": "missing_output_video"}
    if not (await session.file_exists(reference_video) or await session.directory_exists(reference_video)):
        return 0.0, [], [], {"reason": "missing_reference_video"}
    if not (await session.file_exists(input_video) or await session.directory_exists(input_video)):
        return 0.0, [], [], {"reason": "missing_input_video"}

    remote_breakpoint_path = eval_cfg.remote_breakpoint_json
    sample_count = int(task_cfg.metadata.get("sample_count", SAMPLE_COUNT))

    remote_eval_dir = remote_child(REMOTE_TEMP_ROOT, task_tag)
    remote_script_path = remote_child(remote_eval_dir, "remote_hard_eval.py")
    remote_report_path = remote_child(remote_eval_dir, "hard_eval_report.json")
    remote_frames_dir = remote_child(remote_eval_dir, "frames")

    await session.interface.create_dir(remote_eval_dir)
    await session.interface.create_dir(remote_frames_dir)
    await session.write_file(
        remote_script_path, REMOTE_HARD_EVAL_SCRIPT.read_text(encoding="utf-8")
    )
    if not (await session.file_exists(remote_breakpoint_path) or await session.directory_exists(remote_breakpoint_path)):
        return 0.0, [], [], {"reason": "missing_breakpoint_json", "path": remote_breakpoint_path}

    argv = " ".join(
        ps_literal(value)
        for value in [
            remote_script_path,
            "--input-video",
            input_video,
            "--reference-video",
            reference_video,
            "--output-video",
            output_video,
            "--breakpoint-json",
            remote_breakpoint_path,
            "--report-json",
            remote_report_path,
            "--frames-dir",
            remote_frames_dir,
            "--sample-count",
            str(sample_count),
        ]
    )
    ps = (
        "$ErrorActionPreference='Stop'; "
        f"Set-Location -LiteralPath {ps_literal(remote_eval_dir)}; "
        f"& python {argv}"
    )
    result = await session.run_command(f'powershell -NoProfile -Command "{ps}"', check=False)
    stdout = result.get("stdout", "") if isinstance(result, dict) else ""
    stderr = result.get("stderr", "") if isinstance(result, dict) else ""
    return_code = int(result.get("return_code", 1)) if isinstance(result, dict) else 1

    payload = _parse_json_from_stdout(stdout)
    if return_code != 0:
        logger.error("[chroma hard eval] remote script failed for %s: %s", task_tag, stderr[:1000])
        return 0.0, [], [], {"reason": "remote_script_failed", "stderr": stderr, "stdout": stdout}
    if payload is None:
        try:
            payload = json.loads((await session.read_bytes(remote_report_path)).decode("utf-8-sig"))
            hard_score = float(payload.get("summary", {}).get("hard_score", 0.0))
            return (
                hard_score,
                payload.get("details", []),
                payload.get("frame_pairs", []),
                {"report_path": remote_report_path},
            )
        except Exception as exc:
            logger.error("[chroma hard eval] could not parse payload for %s: %s", task_tag, exc)
            return (
                0.0,
                [],
                [],
                {"reason": "invalid_remote_payload", "stdout": stdout, "stderr": stderr},
            )

    return (
        float(payload.get("score", 0.0)),
        payload.get("details", []),
        payload.get("frame_pairs", []),
        payload,
    )


async def _run_local_soft_eval(
    task_cfg, session: cb.DesktopSession, frame_pairs: list[dict]
) -> tuple[float, dict]:
    task_tag = task_cfg.metadata["variant_name"]
    model = task_cfg.metadata.get("soft_eval_model", SOFT_EVAL_MODEL)
    frame_items = []
    for pair in frame_pairs:
        try:
            input_bytes = await session.read_bytes(pair["input_frame_path"])
            output_bytes = await session.read_bytes(pair["output_frame_path"])
        except Exception as exc:
            logger.warning(
                "[chroma soft eval] could not download frame pair for %s: %s", task_tag, exc
            )
            continue
        frame_items.append(
            {
                "identifier": pair.get("identifier")
                or f"{task_tag}_{int(pair.get('index', 0)):03d}",
                "index": int(pair.get("index", 0)),
                "time_sec": float(pair.get("time_sec", 0.0)),
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
            }
        )
    if not frame_items:
        return 0.0, {"summary": {"final_score": 0.0, "reason": "no_frame_pairs"}, "evaluations": []}
    return await run_soft_eval_local(task_tag=task_tag, frame_items=frame_items, model=model)


@cb.tasks_config(split="train")
def load():
    tasks = []
    for spec in VARIANTS:
        config = _build_config(spec)
        tasks.append(
            cb.Task(
                description=config.task_description,
                metadata=config.to_metadata(),
                computer={
                    "provider": "computer",
                    "setup_config": {"os_type": config.OS_TYPE},
                },
            )
        )
    return tasks


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    hard_score, hard_details, frame_pairs, hard_payload = await _run_remote_hard_eval(
        task_cfg, session
    )
    soft_score = 0.0
    soft_details: dict = {"evaluations": []}
    if frame_pairs:
        try:
            soft_score, soft_details = await _run_local_soft_eval(task_cfg, session, frame_pairs)
        except Exception as exc:
            logger.warning(
                "[chroma soft eval] local LLM evaluation failed for %s: %s",
                task_cfg.metadata["variant_name"],
                exc,
            )
            soft_score = 0.0

    soft_by_identifier = {
        str(item.get("identifier")): float(item.get("score", 0.0) or 0.0)
        for item in soft_details.get("evaluations", [])
    }
    soft_eval_available = bool(soft_details.get("evaluations"))
    combined_scores = []
    for detail in hard_details:
        identifier = str(detail.get("identifier", ""))
        frame_result = compute_final_frame_result(
            roi_input_cv=float(detail.get("roi_input_cv", 0.0) or 0.0),
            soft_frame_score=soft_by_identifier.get(identifier, 0.0),
            full_frame_edge_iou=float(detail.get("full_frame_edge_iou", 0.0) or 0.0),
            roi_edge_iou=float(detail.get("roi_edge_iou", 0.0) or 0.0),
        )
        detail.update(frame_result)
        combined_scores.append(float(frame_result["final_frame_score"]))

    if not soft_eval_available:
        final_score = 0.0
    elif combined_scores:
        final_score = float(sum(combined_scores) / len(combined_scores))
    else:
        final_score = 0.0

    logger.info(
        "[chroma eval] task=%s hard=%.4f soft=%.4f final=%.4f",
        task_cfg.metadata["variant_name"],
        hard_score,
        soft_score,
        final_score,
    )
    if hard_payload.get("reason"):
        logger.info(
            "[chroma eval] task=%s hard_eval_reason=%s",
            task_cfg.metadata["variant_name"],
            hard_payload.get("reason"),
        )
    logger.info(
        "[chroma eval] task=%s thresholds roi_input_cv>=%.2f soft>=%.2f final_score=hard_quality_when_gates_pass",
        task_cfg.metadata["variant_name"],
        ROI_INPUT_GATE_THRESHOLD,
        SOFT_GATE_PASS_THRESHOLD,
    )
    return [final_score]
