from __future__ import annotations

import json
import os
from pathlib import Path

from utils.evaluation import llm_vision_binary_questions_sync, resolve_llm_judge_model

MODEL = resolve_llm_judge_model(
    env_var="BLENDER_TASK_SOFT_EVAL_MODEL",
    default="gpt-5.2",
)


def run_local_soft_eval(
    *,
    reference_sheet: Path,
    preview_sheet: Path,
    replay_sheet: Path,
    pose_state_sheet: Path,
) -> float:
    if not os.environ.get("OPENAI_API_KEY"):
        return 0.0

    prompt_context = (
        "You are an expert animation and rigging judge. "
        "This is a body-motion reproduction task. "
        "Image 1 is the hidden clean reference video sampled into a contact sheet. "
        "Image 2 is the agent-submitted preview.mp4 sampled at matching normalized times. "
        "Image 3 is the evaluator replay rendered from final.blend. "
        "Image 4 shows pose-state triads with reference video pose, hidden GT pose, and candidate pose. "
        "Ignore material and lighting differences. Do not judge lip sync or facial expression. "
        "Focus only on: whether the preview video matches the reference body motion, whether the replay from the Blender file agrees with the submitted preview, and whether the skeleton states and poses look natural rather than broken or collapsed. "
        "Judge each question independently using only YES or NO."
    )
    questions = [
        "Does the submitted preview match the reference body motion well enough to pass?",
        "Does the replay rendered from final.blend agree with the submitted preview well enough to pass?",
        "Do the visible skeleton states and poses look natural and non-broken enough to pass?",
    ]

    data = llm_vision_binary_questions_sync(
        prompt_context=prompt_context,
        questions=questions,
        image_bytes_list=[
            reference_sheet.read_bytes(),
            preview_sheet.read_bytes(),
            replay_sheet.read_bytes(),
            pose_state_sheet.read_bytes(),
        ],
        model=MODEL,
        temperature=0,
    )
    report_path = reference_sheet.parent / "soft_eval_report.json"
    report_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return max(0.0, min(1.0, float(data.get("final_score", 0.0))))
