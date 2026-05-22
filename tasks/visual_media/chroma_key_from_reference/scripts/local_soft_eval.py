"""Local soft-eval runner for chroma-key tasks."""

from __future__ import annotations

import logging
from typing import Any

from utils.evaluation import EvaluationContext, llm_vision_binary_checklist_judge

logger = logging.getLogger(__name__)


def build_soft_eval_prompt(task_tag: str, index: int, time_sec: float) -> str:
    return f"""You are evaluating task completion for a keying-related editing task.

Compare these two images:
1. First image: the original input frame before editing
2. Second image: the agent's rendered output frame after editing

Task: {task_tag}
Frame index: {index}
Timestamp: {time_sec:.3f} seconds

Decide only whether the second image shows that the agent actually performed a real keying-related edit on the first image.

Do NOT judge whether the result is high quality.
Do NOT judge whether it matches any hidden reference.
Do NOT require a final composited shot.
Do NOT require a perfect background replacement.

Judge each checklist item independently.
Use a strict binary YES/NO decision for each item.
Do not invent fractional scores. A frame-level soft score will be computed by averaging the binary checklist answers."""


def build_soft_eval_checklist() -> list[tuple[str, str]]:
    return [
        (
            "real_keying_edit_done",
            "Does the output frame show that a real keying-related edit was performed on the input frame?",
        ),
        (
            "not_raw_input_copy",
            "Is the output clearly not just the raw input left essentially unchanged?",
        ),
        (
            "foreground_subject_preserved",
            "Is the foreground subject still present and reasonably preserved after the edit?",
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
        mode="chroma_soft_eval",
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
                    image_bytes=item["input_bytes"],
                    reference_image_bytes=item["output_bytes"],
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
                logger.warning("[chroma soft eval] frame %s failed: %s", identifier, exc)
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
