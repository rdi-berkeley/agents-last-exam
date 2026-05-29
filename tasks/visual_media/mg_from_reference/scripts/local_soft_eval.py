"""Shared helpers for MG family evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path, PureWindowsPath
from typing import Any

from tasks.utils.evaluation import EvaluationContext, llm_vision_binary_checklist_judge

logger = logging.getLogger(__name__)


def remote_child(base: str, *parts: str) -> str:
    path = PureWindowsPath(base)
    for part in parts:
        path = path / part
    return str(path)


def ps_quote(text: str) -> str:
    return text.replace("'", "''")


def ps_literal(text: str) -> str:
    return f"'{ps_quote(text)}'"


def read_breakpoint_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_sample_times(point_path: Path, default_count: int) -> list[float]:
    payload = read_breakpoint_payload(point_path)
    sample_count = max(1, int(payload.get("sample_count", default_count)))
    raw = payload.get("breakpoints", payload.get("points", []))
    points: list[float] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            time_sec = float(item.get("time_sec", 0.0))
        except Exception:
            continue
        if time_sec >= 0:
            points.append(round(time_sec, 6))
    return sorted(set(points))[:sample_count]


def build_soft_eval_prompt(task_tag: str, index: int, time_sec: float) -> str:
    return f"""You are evaluating a motion-graphics recreation task.

Compare these two images:
1. First image: the agent's rendered output frame
2. Second image: the reference frame

Task: {task_tag}
Frame index: {index}
Timestamp: {time_sec:.3f} seconds

Judge each checklist item independently.
Use a strict binary YES/NO decision for each item.
Do not invent fractional scores. A frame-level soft score will be computed by averaging the binary checklist answers."""


def build_soft_eval_checklist() -> list[tuple[str, str]]:
    return [
        (
            "composition_layout_match",
            "Does the output frame match the reference frame in overall composition and layout?",
        ),
        (
            "element_geometry_match",
            "Do the visible motion-graphics elements match in geometry, placement, and ordering?",
        ),
        (
            "timing_state_match",
            "Does the output appear to be at the same animation or timing state as the reference frame?",
        ),
        (
            "style_color_match",
            "Do the styling cues such as colors, typography, and visual treatment match closely enough?",
        ),
    ]


async def run_soft_eval_local(
    *,
    task_tag: str,
    frame_items: list[dict[str, Any]],
    model: str,
) -> tuple[float, dict[str, Any]]:
    async with EvaluationContext(
        task_tag=task_tag,
        mode="mg_soft_eval",
        auto_save=False,
        model=model,
        frame_count=len(frame_items),
    ) as ctx:
        total_score = 0.0
        evaluated = 0
        evaluations: list[dict[str, Any]] = []
        for item in frame_items:
            identifier = str(item["identifier"])
            prompt = build_soft_eval_prompt(
                task_tag=task_tag,
                index=int(item["index"]),
                time_sec=float(item["time_sec"]),
            )
            try:
                result = await llm_vision_binary_checklist_judge(
                    prompt_intro=prompt,
                    checklist_items=build_soft_eval_checklist(),
                    image_bytes=item["output_bytes"],
                    reference_image_bytes=item["reference_bytes"],
                    model=model,
                    max_tokens=256,
                    return_details=True,
                    eval_context=ctx,
                    identifier=identifier,
                )
                score = float(result.get("score") or 0.0)
                total_score += score
                evaluated += 1
                evaluations.append(
                    {
                        "identifier": identifier,
                        "index": int(item["index"]),
                        "time_sec": float(item["time_sec"]),
                        "score": score,
                        "vlm_response": result.get("vlm_response"),
                        "checklist_answers": result.get("checklist_answers"),
                        "checklist_scores": result.get("checklist_scores"),
                        "summary": result.get("summary"),
                    }
                )
            except Exception as exc:
                logger.warning("[mg soft eval] frame %s failed: %s", identifier, exc)
                ctx.log_error(identifier, exc, score=0.0)
                evaluations.append(
                    {
                        "identifier": identifier,
                        "index": int(item["index"]),
                        "time_sec": float(item["time_sec"]),
                        "score": 0.0,
                        "error": str(exc),
                    }
                )

        final_score = total_score / max(1, evaluated)
        _, details = ctx.finalize(
            final_score=final_score,
            evaluated_frames=evaluated,
            requested_frames=len(frame_items),
        )
        details["evaluations"] = evaluations
        return final_score, details
