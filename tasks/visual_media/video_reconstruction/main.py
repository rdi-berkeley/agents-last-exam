"""Video reconstruction family task: recut from raw clips using time-cut prompt."""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup
from tasks.utils.evaluation import gemini_video_json_judge

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

TASK_DIR = Path(__file__).resolve().parent
VARIANT_NAME_CONST = "international_festival"
VARIANT_NAME = VARIANT_NAME_CONST
DOMAIN_NAME = "visual_media"
TASK_NAME_CONST = "video_reconstruction"
REMOTE_ROOT_DIR = os.environ.get("IF_REMOTE_ROOT_DIR", r"E:\agenthle")
PROJECT_FILE_NAME = "blank.drp"
OUTPUT_VIDEO_NAME = os.environ.get("TARGET_VIDEO_NAME", "output-test.mp4")
# Do not pin a preview build that Google will retire: `gemini-3-pro-preview`
# started returning 404 NOT_FOUND and silently zeroed every run. Override via env.
EVAL_MODEL = os.environ.get("GEMINI_EVAL_MODEL", "gemini-3.1-pro-preview")
EVAL_THRESHOLD = 0.75
EVAL_AUTH_MODE = os.environ.get("GEMINI_AUTH_MODE", "auto")
EVAL_REPEATS = int(os.environ.get("GEMINI_EVAL_REPEATS", "3"))
TARGET_DURATION_SEC = 138.0
DURATION_TOLERANCE_SEC = 10.0
SETUP_SLEEP_SEC = 18.0

FALLBACK_SEGMENT_PROMPT = """
Video reconstruction task for Syracuse University International Festival 2025.

Rebuild the shot sequence and pacing using these 4 sections:
1) Behind-the-scenes preparation and event warm-up (00:00-00:40)
2) Food and cultural experience (00:41-01:06)
3) Multicultural stage performances (01:07-01:19)
4) Friendship interactions and multilingual greetings ending (01:20-02:18)

Key requirements:
- Retrieve source material from raw_clips and assemble in time-cut order.
- Keep each segment theme aligned with the reference prompt
  (scene, subject actions, and event content).
- Match temporal order as closely as possible; avoid wrong, missing, or swapped segments.
- Keep subtitle/title structure consistent with the prompt
  (such as opening title and ending credits).
"""

REQUIRED_SCORE_KEYS = [
    "shot_selection",
    "temporal_order",
    "timing_alignment",
    "content_fidelity",
    "overall_coherence",
]


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in REQUIRED_SCORE_KEYS:
        normalized[key] = _clamp_score(payload.get(key, 0.0))

    summary = payload.get("summary", "")
    normalized["summary"] = str(summary) if summary is not None else ""

    major_errors = payload.get("major_errors", [])
    if isinstance(major_errors, list):
        normalized["major_errors"] = [str(item) for item in major_errors]
    elif major_errors is None:
        normalized["major_errors"] = []
    else:
        normalized["major_errors"] = [str(major_errors)]

    final_score = sum(normalized[key] for key in REQUIRED_SCORE_KEYS) / len(REQUIRED_SCORE_KEYS)
    normalized["final_score"] = _clamp_score(final_score)
    return normalized


def _probe_video_duration_seconds(video_bytes: bytes) -> float | None:
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as fp:
            fp.write(video_bytes)
            temp_path = fp.name
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                temp_path,
            ],
            text=True,
        ).strip()
        return float(output)
    except Exception:
        return None
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


async def evaluate_video_against_prompt(
    *,
    video_bytes: bytes,
    segment_prompt: str,
    model: str = "gemini-3.0-pro",
    api_key: str | None = None,
    auth_mode: str | None = None,
) -> dict[str, Any]:
    prompt = (
        "You are a strict video evaluator.\n"
        "Given one output video and the time-cut editing reference prompt below, "
        "judge whether the output follows the required shot selection, temporal order, "
        "timing alignment, content fidelity, and overall coherence.\n\n"
        "Return ONLY valid JSON with these exact keys:\n"
        "{\n"
        '  "shot_selection": number,\n'
        '  "temporal_order": number,\n'
        '  "timing_alignment": number,\n'
        '  "content_fidelity": number,\n'
        '  "overall_coherence": number,\n'
        '  "summary": string,\n'
        '  "major_errors": string[]\n'
        "}\n"
        "Scoring rules:\n"
        "- each score must be in [0,1]\n"
        "- be strict on mismatched segments and wrong sequence\n"
        "- if the video clearly fails, return low scores\n\n"
        f"Reference prompt:\n{segment_prompt}"
    )
    result = await gemini_video_json_judge(
        prompt=prompt,
        video_bytes=video_bytes,
        model=model,
        api_key=api_key,
        auth_mode=auth_mode,
        temperature=0.0,
    )
    parsed_raw = result.get("parsed") if isinstance(result, dict) else None
    if not isinstance(parsed_raw, dict):
        payload = dict(result) if isinstance(result, dict) else {}
        payload.setdefault("score", 0.0)
        return payload

    parsed = _normalize_payload(parsed_raw)
    payload = dict(result)
    payload["parsed"] = parsed
    payload["score"] = float(parsed["final_score"])
    return payload


def _ps_quote(text: str) -> str:
    return text.replace("'", "''")


async def _run_command(session: cb.DesktopSession, command: str, check: bool = True):
    return await session.run_command(command, check=check)


async def _prepare_eval_video(
    session: cb.DesktopSession,
    *,
    output_video: str,
    remote_tmp_video: str,
) -> str:
    """
    Try compressing output video before sending to Gemini.
    Falls back to original output if compression fails.
    """
    cmd = (
        "powershell -NoProfile -Command "
        '"'
        "$ErrorActionPreference='Continue'; "
        f"$in='{_ps_quote(output_video)}'; "
        f"$out='{_ps_quote(remote_tmp_video)}'; "
        "ffmpeg -hide_banner -loglevel error -y "
        "-i $in -vf 'scale=960:-2,fps=12' -an -c:v libx264 -preset veryfast -crf 34 $out"
        '"'
    )
    try:
        await _run_command(session, cmd, check=True)
        if await session.exists(remote_tmp_video):
            return remote_tmp_video
    except Exception as exc:
        logger.warning("[eval] failed to compress output for Gemini: %s", exc)
    return output_video


@dataclass
class TaskConfig(GeneralTaskConfig):
    REMOTE_ROOT_DIR: str = REMOTE_ROOT_DIR
    DOMAIN_NAME: str = "visual_media"

    TASK_NAME: str = "video_reconstruction"
    VARIANT_NAME: str = VARIANT_NAME_CONST

    @property
    def raw_clips_dir(self) -> str:
        return rf"{self.task_dir}\input\raw_clips"

    @property
    def segment_prompt_path(self) -> str:
        return rf"{self.task_dir}\input\segment_prompt.md"

    @property
    def project_path(self) -> str:
        return rf"{self.software_dir}\{PROJECT_FILE_NAME}"

    @property
    def output_video(self) -> str:
        return rf"{self.remote_output_dir}\{OUTPUT_VIDEO_NAME}"

    @property
    def task_description(self) -> str:
        return f"""
Goal: Use DaVinci Resolve to rebuild the International Festival video from raw clips.

Workflow:
1. Open DaVinci Resolve and load the project file: {self.project_path}
2. Review the source clips in: {self.raw_clips_dir}
3. Follow the editing brief in: {self.segment_prompt_path}
4. Rebuild the recap sequence and pacing so the sections appear in the requested order.
5. Export the final video to: {self.output_video}

Editing brief summary:
- 00:00-00:40: behind-the-scenes preparation and event warm-up
- 00:41-01:06: food and cultural experience
- 01:07-01:19: multicultural stage performances
- 01:20-02:18: friendship interactions, multilingual greetings, and ending credits
- `segment_prompt.md` contains ordered timestamped natural-language cut instructions describing the expected scenes, actions, greetings, titles, and ending credits within those ranges.

Available input:
- Raw source clips: {self.raw_clips_dir}
- Editing brief: {self.segment_prompt_path}

Output requirements:
- Produce one coherent recap video that follows the requested sections and keeps the overall runtime close to 2 minutes 18 seconds.
- Save the exported file exactly to: {self.output_video}
"""

    def to_metadata(self) -> dict:
        md = super().to_metadata()
        md.update(
            {
                "raw_clips_dir": self.raw_clips_dir,
                "segment_prompt_path": self.segment_prompt_path,
                "project_path": self.project_path,
                "output_video": self.output_video,
                "eval_provider": "gemini",
                "eval_model": EVAL_MODEL,
                "eval_threshold": EVAL_THRESHOLD,
                "eval_auth_mode": EVAL_AUTH_MODE,
                "eval_repeats": EVAL_REPEATS,
                "target_duration_sec": TARGET_DURATION_SEC,
                "duration_tolerance_sec": DURATION_TOLERANCE_SEC,
                "setup_sleep_sec": SETUP_SLEEP_SEC,
            }
        )
        return md


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    output_video = task_cfg.metadata["output_video"]
    segment_prompt_path = task_cfg.metadata["segment_prompt_path"]
    eval_model = task_cfg.metadata.get("eval_model", EVAL_MODEL)
    eval_threshold = float(task_cfg.metadata.get("eval_threshold", EVAL_THRESHOLD))
    eval_auth_mode = task_cfg.metadata.get("eval_auth_mode", EVAL_AUTH_MODE)
    eval_repeats = max(1, int(task_cfg.metadata.get("eval_repeats", EVAL_REPEATS)))
    target_duration_sec = float(task_cfg.metadata.get("target_duration_sec", TARGET_DURATION_SEC))
    duration_tolerance_sec = float(
        task_cfg.metadata.get("duration_tolerance_sec", DURATION_TOLERANCE_SEC)
    )
    task_tag = task_cfg.metadata.get("variant_name", VARIANT_NAME)

    eval_root = Path(os.environ.get("EVALUATION_OUTPUT_DIR", f"./trycua/cua-bench/{task_tag}"))
    eval_dir = eval_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / "evaluation_gemini.json"

    def _save_report(payload: dict):
        report_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if not await session.exists(output_video):
        _save_report(
            {
                "variant_name": task_tag,
                "error": "missing_output_video",
                "output_video": output_video,
                "final_score": 0.0,
                "pass": False,
            }
        )
        return [0.0]

    prompt_source = "embedded_fallback"
    segment_prompt = FALLBACK_SEGMENT_PROMPT
    try:
        if await session.exists(segment_prompt_path):
            prompt_text = await session.read_file(segment_prompt_path)
            if isinstance(prompt_text, str) and prompt_text.strip():
                segment_prompt = prompt_text
                prompt_source = "remote_input_file"
    except Exception as exc:
        logger.warning("[eval] failed reading segment prompt file: %s", exc)

    remote_eval_dir = rf"{task_cfg.metadata['remote_output_dir']}\eval_tmp"
    try:
        await session.makedirs(remote_eval_dir)
    except Exception:
        pass
    remote_compact_video = rf"{remote_eval_dir}\gemini_eval_compact.mp4"
    eval_video_path = await _prepare_eval_video(
        session,
        output_video=output_video,
        remote_tmp_video=remote_compact_video,
    )

    try:
        video_bytes = await session.read_bytes(eval_video_path)
    except Exception as exc:
        _save_report(
            {
                "variant_name": task_tag,
                "error": f"read_video_failed:{exc}",
                "output_video": output_video,
                "eval_video_path": eval_video_path,
                "final_score": 0.0,
                "pass": False,
            }
        )
        return [0.0]

    duration_sec = await asyncio.to_thread(_probe_video_duration_seconds, video_bytes)
    min_duration_sec = max(0.0, target_duration_sec - duration_tolerance_sec)
    max_duration_sec = target_duration_sec + duration_tolerance_sec
    if duration_sec is None:
        _save_report(
            {
                "variant_name": task_tag,
                "error": "duration_probe_failed",
                "output_video": output_video,
                "eval_video_path": eval_video_path,
                "final_score": 0.0,
                "pass": False,
            }
        )
        return [0.0]
    if not (min_duration_sec <= duration_sec <= max_duration_sec):
        _save_report(
            {
                "variant_name": task_tag,
                "error": "duration_out_of_range",
                "output_video": output_video,
                "eval_video_path": eval_video_path,
                "duration_sec": duration_sec,
                "target_duration_sec": target_duration_sec,
                "duration_tolerance_sec": duration_tolerance_sec,
                "final_score": 0.0,
                "pass": False,
            }
        )
        return [0.0]

    attempts: list[dict[str, Any]] = []
    scores: list[float] = []
    for idx in range(eval_repeats):
        result = await evaluate_video_against_prompt(
            video_bytes=video_bytes,
            segment_prompt=segment_prompt,
            model=eval_model,
            auth_mode=eval_auth_mode,
        )
        parsed = result.get("parsed") if isinstance(result, dict) else None
        gemini_ok = bool(result.get("ok")) if isinstance(result, dict) else False
        score = float(parsed.get("final_score", 0.0) or 0.0) if isinstance(parsed, dict) else 0.0
        score = max(0.0, min(1.0, score))
        # Only count attempts where the judge actually returned a parsed verdict.
        # A failed judge call (retired model / auth / truncated JSON) must NOT be
        # averaged in as a 0 — that silently turns a judge outage into a fake low score.
        if gemini_ok and isinstance(parsed, dict):
            scores.append(score)
        attempts.append(
            {
                "attempt_index": idx,
                "gemini_ok": gemini_ok,
                "gemini_error": (
                    result.get("error") if isinstance(result, dict) else "unknown_error"
                ),
                "gemini_auth_mode": result.get("auth_mode") if isinstance(result, dict) else None,
                "gemini_model_used": result.get("model_used") if isinstance(result, dict) else None,
                "gemini_project": result.get("project") if isinstance(result, dict) else None,
                "gemini_location": result.get("location") if isinstance(result, dict) else None,
                "gemini_raw_text": result.get("raw_text") if isinstance(result, dict) else "",
                "scores": parsed or {},
                "final_score": score,
            }
        )

    # Hard-fail (do NOT record 0.0) when the judge itself never succeeded. A
    # retired model, missing credential, or truncated/invalid JSON must surface
    # as an evaluation error so the run is retried/flagged — not silently scored
    # 0 for every submission (which masked both a broken judge and an answer leak).
    if not scores:
        gemini_errors = [a.get("gemini_error") for a in attempts]
        _save_report(
            {
                "variant_name": task_tag,
                "error": "gemini_judge_failed_all_attempts",
                "gemini_errors": gemini_errors,
                "output_video": output_video,
                "eval_video_path": eval_video_path,
                "duration_sec": duration_sec,
                "final_score": None,
                "pass": False,
            }
        )
        raise RuntimeError(
            f"Gemini judge failed on all {len(attempts)} attempt(s); refusing to "
            f"score as 0.0. Errors: {gemini_errors}"
        )

    final_score = sum(scores) / len(scores)
    final_score = max(0.0, min(1.0, final_score))

    passed = bool(final_score >= eval_threshold)
    latest_attempt = attempts[-1] if attempts else {}

    report = {
        "variant_name": task_tag,
        "eval_provider": task_cfg.metadata.get("eval_provider", "gemini"),
        "eval_model": eval_model,
        "eval_auth_mode": eval_auth_mode,
        "eval_repeats": eval_repeats,
        "eval_threshold": eval_threshold,
        "target_duration_sec": target_duration_sec,
        "duration_tolerance_sec": duration_tolerance_sec,
        "output_video": output_video,
        "eval_video_path": eval_video_path,
        "duration_sec": duration_sec,
        "segment_prompt_path": segment_prompt_path,
        "segment_prompt_source": prompt_source,
        "gemini_ok": all(bool(attempt.get("gemini_ok")) for attempt in attempts),
        "gemini_error": (
            None
            if all(bool(attempt.get("gemini_ok")) for attempt in attempts)
            else [attempt.get("gemini_error") for attempt in attempts]
        ),
        "gemini_auth_mode": latest_attempt.get("gemini_auth_mode"),
        "gemini_model_used": latest_attempt.get("gemini_model_used"),
        "gemini_project": latest_attempt.get("gemini_project"),
        "gemini_location": latest_attempt.get("gemini_location"),
        "gemini_raw_text": latest_attempt.get("gemini_raw_text", ""),
        "scores": latest_attempt.get("scores", {}),
        "score_runs": scores,
        "gemini_attempts": attempts,
        "final_score": final_score,
        "pass": passed,
    }
    _save_report(report)
    return [final_score]
