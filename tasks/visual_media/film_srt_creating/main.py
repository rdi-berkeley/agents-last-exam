"""Jianying Task - Add subtitles based on reference style/position.

Single variant: ``train``.
Software: Jianying (剪映专业版, free).
VM: agenthle-dev-gpu-free (E:\\agenthle\\visual_media\\film_srt_creating\\train\\).
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.utils.evaluation import llm_vision_judge

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

#################################################################
############################# Setup #############################
#################################################################

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "gpt-5.2"

# Time samples in `hh:mm:ss:ff` (frame index at the video FPS).
# These are internal to the evaluator and NOT exposed to the agent.
SAMPLE_TIMECODES = [
    "00:00:20:23",
    "00:01:39:06",
    "00:02:08:13",
    "00:02:25:12",
]

VARIANTS = ["train"]


@dataclass
class TaskConfig(GeneralTaskConfig):
    REMOTE_ROOT_DIR: str = os.environ.get("REMOTE_ROOT_DIR", r"E:\agenthle")
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "film_srt_creating"
    VARIANT_NAME: str = ""

    OUTPUT_VIDEO_NAME: str = os.environ.get("TARGET_VIDEO_NAME", "agent_output.mp4")

    @property
    def remote_input_video(self) -> str:
        return rf"{self.task_dir}\input\input.mp4"

    @property
    def remote_reference_video(self) -> str:
        return rf"{self.task_dir}\reference\reference.mp4"

    @property
    def target_video_path(self) -> str:
        # Exported video lives under the task output directory.
        # Default is `agent_output.mp4`. For eval testing you can set:
        # `TARGET_VIDEO_NAME=output-test.mp4`.
        return rf"{self.remote_output_dir}\{self.OUTPUT_VIDEO_NAME}"

    @property
    def task_description(self) -> str:
        return f"""
Goal: Use Jianying (CapCut) to add subtitles to the input video.

Requirements:
- Subtitle layout: vertical (竖排), positioned at the right side of the frame
- Font style: artistic/calligraphic Chinese style matching the reference video
- Text content: use the lyrics/narration from the input video (not the reference text)

Inputs:
- Input video: {self.remote_input_video}
- Reference video (different film) showing the desired subtitle style, layout, and position: {self.remote_reference_video}

Output:
- Export the edited video to: {self.target_video_path}

Instructions:
1. Watch the reference video to understand the desired vertical calligraphic subtitle style and positioning.
2. Open the input video in Jianying.
3. Add subtitles matching the input video's audio/narration content.
4. Style the subtitles to match the reference: vertical layout, calligraphic font, positioned on the right side of the frame.
5. Export the final video.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                # Remote resources
                "remote_task_dir": self.task_dir,
                "remote_input_video": self.remote_input_video,
                "remote_reference_video": self.remote_reference_video,
                "target_video": self.target_video_path,
                "reference_video": self.remote_reference_video,
                # Evaluation config
                "sample_timecodes": SAMPLE_TIMECODES,
                "model": DEFAULT_MODEL,
            }
        )
        return metadata


def _make_config(variant: str) -> TaskConfig:
    return TaskConfig(VARIANT_NAME=variant)


@cb.tasks_config(split="train")
def load():
    """Define the Jianying subtitles task (one task per variant)."""
    tasks = []
    for variant in VARIANTS:
        cfg = _make_config(variant)
        tasks.append(
            cb.Task(
                description=cfg.task_description,
                metadata=cfg.to_metadata(),
                computer={
                    "provider": "computer",
                    "setup_config": {"os_type": cfg.OS_TYPE},
                },
            )
        )
    return tasks


#################################################################
######################### Initialization ########################
#################################################################


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _timecode_to_ms(tc: str, fps: float) -> int:
    """Convert timecode to milliseconds.

    Expected format: `hh:mm:ss:ff` where `ff` is a frame index at the given FPS.
    Fallback: supports `hh:mm:ss.mmm` (milliseconds) if encountered.
    """
    tc = tc.strip()

    # Support common millisecond format too (not used by this task, but helps robustness).
    if "." in tc:
        try:
            base, ms_s = tc.rsplit(".", 1)
            hh, mm, ss = base.split(":")
            h = int(hh)
            m = int(mm)
            s = int(ss)
            ms_i = int(ms_s)
            ms_i = max(0, min(999, ms_i))
            return ((h * 3600 + m * 60 + s) * 1000) + ms_i
        except Exception:
            return 0

    try:
        hh, mm, ss, ff = tc.split(":")
        h = int(hh)
        m = int(mm)
        s = int(ss)
        f = int(ff)
    except Exception:
        return 0

    fps = float(fps) if fps else 30.0
    if fps <= 0:
        fps = 30.0

    # Convert to frame number then to ms; clamp negative frames to 0.
    base_frames = int(round((h * 3600 + m * 60 + s) * fps))
    total_frames = max(0, base_frames + f)
    return int(round((total_frames / fps) * 1000.0))


async def _extract_frame_remote(
    session: cb.DesktopSession,
    video_path: str,
    out_png_path: str,
    *,
    ss_ms: int,
) -> bool:
    """Extract a single frame on the remote machine using ffmpeg.

    This avoids downloading large mp4s to local disk (which can be truncated).
    """
    # Use PowerShell on Windows; runner uses windows environments.
    # Quote paths defensively.
    ss_s = f"{ss_ms/1000.0:.3f}"
    cmd = (
        "powershell -NoProfile -Command "
        '"'
        "$ErrorActionPreference='Stop'; "
        f"ffmpeg -hide_banner -loglevel error -y -ss {ss_s} -i '{video_path}' -frames:v 1 '{out_png_path}'"
        '"'
    )
    try:
        await session.run_command(cmd, check=True)
        return True
    except Exception as e:
        print(f"[eval] remote ffmpeg extract failed: {e}")
        return False


#################################################################
########################### Evaluation ##########################
#################################################################


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Compare subtitle style/position with reference using full-frame VLM judgments.

    Prefer remote frame extraction via ffmpeg to avoid downloading large mp4s.
    """
    target_video = task_cfg.metadata["target_video"]
    reference_video = task_cfg.metadata["reference_video"]
    model = task_cfg.metadata.get("model", DEFAULT_MODEL)
    sample_timecodes = task_cfg.metadata.get("sample_timecodes", SAMPLE_TIMECODES)

    output_dir = os.environ.get("EVALUATION_OUTPUT_DIR", "./trycua/cua-bench/film_srt_creating")
    eval_dir = os.path.join(output_dir, "eval")

    # Use prints because logging may not be configured in the batch runner.
    print(f"[eval] target_video={target_video}")
    print(f"[eval] reference_video={reference_video}")
    print(f"[eval] model={model}")
    print(f"[eval] sample_timecodes={sample_timecodes}")

    # Compute sample times in ms using a default FPS; the ffmpeg extraction uses ms.
    # If the project FPS differs, tune SAMPLE_TIMECODES or set an env var later.
    fps = float(os.environ.get("VIDEO_FPS", "30") or 30.0)
    times_ms = [_timecode_to_ms(tc, fps) for tc in sample_timecodes]
    print(f"[eval] fps={fps} times_ms={times_ms}")

    scores = []
    details = []

    for idx, time_ms in enumerate(times_ms):
        tc = sample_timecodes[idx] if idx < len(sample_timecodes) else None
        print(f"[eval] sampling idx={idx} timecode={tc} time_ms={time_ms}")
        # Extract frames on remote, then read the small PNGs back.
        remote_tmp_dir = rf"{task_cfg.metadata['remote_output_dir']}\eval_frames"
        try:
            await session.makedirs(remote_tmp_dir)
        except Exception:
            pass

        remote_target_png = rf"{remote_tmp_dir}\target_{idx:03d}.png"
        remote_ref_png = rf"{remote_tmp_dir}\ref_{idx:03d}.png"

        ok_t = await _extract_frame_remote(session, target_video, remote_target_png, ss_ms=time_ms)
        ok_r = await _extract_frame_remote(session, reference_video, remote_ref_png, ss_ms=time_ms)
        if not ok_t or not ok_r:
            scores.append(0.0)
            details.append({"time_ms": time_ms, "error": "remote_frame_extract_failed"})
            continue

        try:
            target_png_bytes = await session.read_bytes(remote_target_png)
            ref_png_bytes = await session.read_bytes(remote_ref_png)
        except Exception as e:
            scores.append(0.0)
            details.append({"time_ms": time_ms, "error": f"remote_png_read_failed: {e}"})
            continue

        # Save local copies for debugging.
        target_frame_path = os.path.join(
            eval_dir, "frames_target", f"frame_{idx:03d}_{time_ms:06d}ms.png"
        )
        ref_frame_path = os.path.join(
            eval_dir, "frames_ref", f"frame_{idx:03d}_{time_ms:06d}ms.png"
        )
        os.makedirs(os.path.dirname(target_frame_path), exist_ok=True)
        os.makedirs(os.path.dirname(ref_frame_path), exist_ok=True)
        with open(target_frame_path, "wb") as f:
            f.write(target_png_bytes)
        with open(ref_frame_path, "wb") as f:
            f.write(ref_png_bytes)

        prompt = (
            "You are comparing subtitle style and position.\n\n"
            "The reference image shows the desired subtitle style and placement.\n"
            "The target image is from the edited video.\n\n"
            "Ignore background differences and the exact subtitle text content.\n"
            "Focus on: subtitle orientation (vertical/竖排 vs horizontal), "
            "font style (artistic/calligraphic Chinese characters), "
            "and position (right side of frame vs bottom center).\n\n"
            "If no subtitles are visible in BOTH images, answer YES.\n"
            "Question: Does the target subtitle style and layout match the reference?\n"
            'Answer with ONLY "YES" or "NO".'
        )

        eval_result = await llm_vision_judge(
            prompt=prompt,
            image_bytes=target_png_bytes,
            reference_image_bytes=ref_png_bytes,
            model=model,
            return_details=True,
            max_tokens=10,
        )
        print(
            f"[eval] vlm_response={eval_result.get('vlm_response')} score={eval_result.get('score')}"
        )

        score = float(eval_result.get("score", 0.0))
        scores.append(score)
        details.append(
            {
                "index": idx,
                "time_ms": time_ms,
                "timecode": sample_timecodes[idx] if idx < len(sample_timecodes) else None,
                "target_frame_path": target_frame_path,
                "ref_frame_path": ref_frame_path,
                "vlm": eval_result,
            }
        )

    avg_score = sum(scores) / len(scores) if scores else 0.0
    result_path = os.path.join(eval_dir, "evaluation.json")
    os.makedirs(eval_dir, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_video": target_video,
                "reference_video": reference_video,
                "score": avg_score,
                "details": details,
            },
            f,
            indent=2,
        )

    return [avg_score]
