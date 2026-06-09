"""Image-only VLM judge for the drawings_to_3d_building task family.

This scorer NEVER reads the agent's OBJ or 3DM geometry. It compares the
rendered 14 canonical views (4 plans + 4 elevations + 2 sections + 4 axons)
of the candidate against the frozen reference renders using a multimodal LLM.

The judge asks N binary YES/NO questions defined in a variant-specific
`eval_config.json`. Final score = yes_count / question_count.

Usage:
    python score_outputs.py \\
        --reference-render-dir <path to reference_renders_v1/> \\
        --candidate-render-dir <path to agent's rendered views/> \\
        --config <path to variant's eval_config.json> \\
        --output-json <path to write report>

Config schema (per-variant, e.g. tmp/betonwerk/eval_config.json):
    {
      "task_description": "human-readable variant description shown in the prompt",
      "judge_questions": ["yes/no question 1", "yes/no question 2", ...],
      "pass_threshold": 0.5,
      "view_names": ["plan_hall_ground", ..., "axon_SE"]   # optional, defaults to standard 14
    }
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
# tasks/engineering/<task>/scripts/ → repo root is 4 levels up
REPO_ROOT = THIS_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.utils.evaluation import llm_multimodal_binary_questions_sync  # noqa: E402


DEFAULT_VIEW_NAMES: tuple[str, ...] = (
    "plan_hall_ground",
    "plan_hall_first",
    "plan_hall_second",
    "plan_tower_typical",
    "elevation_north",
    "elevation_south",
    "elevation_east",
    "elevation_west",
    "section_NS",
    "section_EW",
    "axon_NE",
    "axon_NW",
    "axon_SE",
    "axon_SW",
)

DEFAULT_PASS_THRESHOLD = 0.5


def _build_prompt_context(task_description: str) -> str:
    return (
        "You are evaluating a 3D architectural reconstruction benchmark.\n"
        f"{task_description.strip()}\n\n"
        "You will see 14 paired renders of the building. For each view, the Reference image "
        "(ground truth) is shown first, then the Candidate image (agent submission).\n"
        "The 14 views are: 4 floor plans, 4 cardinal elevations, 2 vertical sections, and 4 corner axonometrics.\n"
        "Judge each question independently using ONLY the visual evidence in the renders; do not "
        "infer hidden geometry from outside knowledge.\n"
        "Respond with ONLY YES or NO for each question."
    )


def _load_config(config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text())
    if not raw.get("judge_questions"):
        raise ValueError(f"{config_path}: missing or empty 'judge_questions'")
    if not raw.get("task_description"):
        raise ValueError(f"{config_path}: missing 'task_description'")
    raw.setdefault("view_names", list(DEFAULT_VIEW_NAMES))
    raw.setdefault("pass_threshold", DEFAULT_PASS_THRESHOLD)
    return raw


_JUDGE_MAX_EDGE = 1024


def _image_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    # Downscale before base64 so the per-question image payload stays under the 50 MB OpenAI cap.
    # Render fidelity stays on disk; only the judge sees the downscaled copy.
    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
            if max(w, h) > _JUDGE_MAX_EDGE:
                scale = _JUDGE_MAX_EDGE / float(max(w, h))
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="PNG", optimize=True)
                raw = buf.getvalue()
    except Exception:
        pass
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_content(
    ref_dir: Path, cand_dir: Path, view_names: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    content: list[dict[str, Any]] = []
    missing: list[str] = []
    for view in view_names:
        ref = ref_dir / f"{view}.png"
        cand = cand_dir / f"{view}.png"
        if not ref.exists():
            missing.append(f"reference/{view}.png")
            continue
        if not cand.exists():
            missing.append(f"candidate/{view}.png")
            continue
        content.extend(
            [
                {"type": "text", "text": f"Reference render — {view}:"},
                {"type": "image_url", "image_url": {"url": _image_to_data_url(ref)}},
                {"type": "text", "text": f"Candidate render — {view}:"},
                {"type": "image_url", "image_url": {"url": _image_to_data_url(cand)}},
            ]
        )
    return content, missing


def evaluate_renders(
    reference_render_dir: Path,
    candidate_render_dir: Path,
    config_path: Path,
    model: str | None = None,
) -> dict[str, Any]:
    """Run the image-only judge over the per-variant view pairs.

    Returns a dict with:
      score              : final_score in [0, 1] (yes_count / question_count)
      passed             : bool (score >= pass_threshold from config)
      yes_count, no_count, question_count
      per_question       : list of {question, result, score, raw_response}
      missing_views      : views without both ref + cand pngs
      pass_threshold     : float (from config)
      variant_task_description : echo of the variant config
    """
    config = _load_config(config_path)
    questions = config["judge_questions"]
    view_names = config["view_names"]
    pass_threshold = config["pass_threshold"]

    content, missing = _build_content(reference_render_dir, candidate_render_dir, view_names)
    if not content:
        return {
            "score": 0.0,
            "passed": False,
            "yes_count": 0,
            "no_count": len(questions),
            "question_count": len(questions),
            "per_question": [],
            "missing_views": missing,
            "pass_threshold": pass_threshold,
            "variant_task_description": config["task_description"],
            "error": "no_renderable_view_pairs",
        }

    judge_result = llm_multimodal_binary_questions_sync(
        prompt_context=_build_prompt_context(config["task_description"]),
        questions=questions,
        content=content,
        model=model,
        max_tokens=32,
        temperature=0,
    )

    return {
        "score": float(judge_result["final_score"]),
        "passed": judge_result["final_score"] >= pass_threshold,
        "yes_count": judge_result["yes_count"],
        "no_count": judge_result["no_count"],
        "question_count": judge_result["question_count"],
        "per_question": judge_result["results"],
        "missing_views": missing,
        "pass_threshold": pass_threshold,
        "variant_task_description": config["task_description"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-only VLM judge for drawings_to_3d_building variants")
    parser.add_argument("--reference-render-dir", required=True)
    parser.add_argument("--candidate-render-dir", required=True)
    parser.add_argument("--config", required=True, help="Per-variant eval_config.json")
    parser.add_argument("--output-json", default=None, help="Optional path to write the JSON report")
    parser.add_argument("--model", default=None, help="Override the judge model (otherwise resolved from env)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_renders(
        Path(args.reference_render_dir).resolve(),
        Path(args.candidate_render_dir).resolve(),
        Path(args.config).resolve(),
        model=args.model,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
