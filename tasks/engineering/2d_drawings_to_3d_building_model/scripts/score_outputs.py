"""Image-only VLM judge for the drawings_to_3d_building task family.

This scorer NEVER reads the agent's OBJ or 3DM geometry. It compares the
rendered 14 canonical views (4 plans + 4 elevations + 2 sections + 4 axons)
of the candidate against the frozen reference renders using a multimodal LLM.

The judge asks the eight original binary YES/NO questions using only relevant
view pairs for each criterion. Final score = yes_count / question_count.

Usage:
    python score_outputs.py \\
        --reference-render-dir <path to reference_renders_v1/> \\
        --candidate-render-dir <path to agent's rendered views/> \\
        --config <path to variant's eval_config.json> \\
        --output-json <path to write report>

The config retains the original task_description and eight judge_questions.
view_names defaults to the standard 14; pass_threshold defaults to 0.5.
Questions map to views by their full text, independently of question order.
See ../README.md for judge configuration and evidence persistence.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import OpenAI


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
DEFAULT_JUDGE_MODEL = "gpt-6-astra"
CRITERION_VIEWS: dict[str, tuple[str, ...]] = {
    "Does the candidate's overall building massing — relative proportions of the workshop Hall and the residential Tower — match the reference?": (
        "axon_NE",
        "axon_SW",
    ),
    "Does the candidate have approximately the same number of distinct horizontal floor levels (slabs) as the reference, especially within the Tower zone?": (
        "section_NS",
        "section_EW",
        "elevation_east",
        "elevation_west",
    ),
    "Is the Tower in the candidate located at the correct position (one end of the building) with a similar narrow footprint and tall extrusion?": (
        "plan_hall_ground",
        "plan_tower_typical",
        "axon_NE",
    ),
    "Does the candidate's north Hall facade show a continuous solid wall (with vertical metal-panel cladding), rather than an open structural framework?": (
        "elevation_north",
        "axon_NW",
    ),
    "Does the candidate's south Hall facade show an open structural framework (steel members with the interior visible through), rather than a solid wall?": (
        "elevation_south",
        "axon_SE",
    ),
    "Does the Tower facade in the candidate show a glass curtain wall with a visible mullion grid (vertical posts plus per-floor horizontal divisions), rather than a featureless solid block?": (
        "elevation_east",
        "elevation_west",
        "axon_NE",
    ),
    "Are the workshop module structures inside the Hall ground floor at approximately the same positions and orientations as in the reference?": (
        "plan_hall_ground",
    ),
    "Do the overall elevation silhouettes of the candidate building match the reference reasonably well across the four cardinal directions?": (
        "elevation_north",
        "elevation_south",
        "elevation_east",
        "elevation_west",
    ),
}


def _build_prompt_context(task_description: str, view_names: list[str]) -> str:
    return (
        "You are evaluating a 3D architectural reconstruction benchmark.\n"
        f"{task_description.strip()}\n\n"
        f"You will see {len(view_names)} paired renders relevant to this criterion: "
        f"{', '.join(view_names)}. For each view, the Reference image "
        "(ground truth) is shown first, then the Candidate image (agent submission).\n"
        "Judge each question independently using ONLY the visual evidence in the renders; do not "
        "infer hidden geometry from outside knowledge.\n"
        "Materials, textures, mesh organization and a common horizontal translation are not scored.\n"
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
    questions = raw["judge_questions"]
    if (
        not isinstance(questions, list)
        or len(questions) != len(CRITERION_VIEWS)
        or any(not isinstance(question, str) for question in questions)
        or set(questions) != set(CRITERION_VIEWS)
    ):
        raise ValueError(f"{config_path}: expected the eight original architectural criteria")
    views = raw["view_names"]
    if (
        not isinstance(views, list)
        or any(view not in DEFAULT_VIEW_NAMES for view in views)
        or len(set(views)) != len(views)
        or any(view not in views for selected in CRITERION_VIEWS.values() for view in selected)
    ):
        raise ValueError(f"{config_path}: view_names must include every criterion's required views")
    if not isinstance(raw["pass_threshold"], (int, float)) or not 0 <= raw["pass_threshold"] <= 1:
        raise ValueError(f"{config_path}: pass_threshold must be between zero and one")
    return raw


def _judge_settings(model: str | None = None) -> dict[str, str]:
    secret_path = Path(__file__).resolve().parents[4] / "secret" / ".env"
    values = {**dotenv_values(secret_path), **os.environ}
    api_key = values.get("D2T3B_JUDGE_API_KEY") or values.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Building judge requires D2T3B_JUDGE_API_KEY or OPENAI_API_KEY")
    return {
        "model": model
        or values.get("D2T3B_JUDGE_MODEL")
        or values.get("LLM_JUDGE_MODEL")
        or DEFAULT_JUDGE_MODEL,
        "base_url": values.get("D2T3B_JUDGE_BASE_URL")
        or values.get("OPENAI_BASE_URL")
        or values.get("OPENAI_API_BASE")
        or "https://api.openai.com/v1",
        "api_key": api_key,
    }


_JUDGE_MAX_EDGE = 1024


def _image_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
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
    report_path: Path | None = None,
    on_report: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Judge eight criteria, checkpointing full replies before validating each answer."""
    config = _load_config(config_path)
    questions = config["judge_questions"]
    pass_threshold = config["pass_threshold"]
    settings = _judge_settings(model)
    report: dict[str, Any] = {
        "status": "in_progress",
        "score": None,
        "passed": None,
        "yes_count": 0,
        "no_count": 0,
        "question_count": len(questions),
        "per_question": [],
        "missing_views": [],
        "pass_threshold": pass_threshold,
        "variant_task_description": config["task_description"],
        "model": settings["model"],
        "base_url": settings["base_url"],
    }

    def persist() -> None:
        if report_path is not None:
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if on_report is not None:
            on_report(report)

    persist()
    try:
        for view in config["view_names"]:
            for label, directory in (
                ("reference", reference_render_dir),
                ("candidate", candidate_render_dir),
            ):
                if not (directory / f"{view}.png").is_file():
                    report["missing_views"].append(f"{label}/{view}.png")
        if report["missing_views"]:
            raise RuntimeError("Building judge is missing required render evidence")

        with OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            timeout=120,
            max_retries=2,
        ) as client:
            for index, question in enumerate(questions, start=1):
                view_names = list(CRITERION_VIEWS[question])
                content, missing = _build_content(
                    reference_render_dir,
                    candidate_render_dir,
                    view_names,
                )
                if missing:
                    report["missing_views"] = missing
                    raise RuntimeError("Building judge lost required render evidence")
                prompt = (
                    _build_prompt_context(config["task_description"], view_names)
                    + f"\n\nQuestion {index}/{len(questions)}: {question}"
                )
                response = client.chat.completions.create(
                    model=settings["model"],
                    messages=[
                        {"role": "user", "content": [{"type": "text", "text": prompt}, *content]}
                    ],
                    max_completion_tokens=2048,
                )
                choice = response.choices[0] if response.choices else None
                raw_response = choice.message.content if choice else None
                result = {
                    "question": question,
                    "question_index": index,
                    "view_names": view_names,
                    "result": None,
                    "score": None,
                    "raw_response": raw_response,
                    "response": response.model_dump(mode="json"),
                }
                report["per_question"].append(result)
                persist()
                answer = (raw_response or "").strip().upper()
                if choice is None or choice.finish_reason != "stop" or answer not in {"YES", "NO"}:
                    raise RuntimeError(
                        f"Building judge returned an incomplete or non-binary reply for Q{index}"
                    )
                result.update(result=answer, score=float(answer == "YES"))
                report["yes_count"] += int(answer == "YES")
                report["no_count"] += int(answer == "NO")
                persist()
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__)
        persist()
        raise

    score = report["yes_count"] / len(questions)
    report.update(status="completed", score=score, passed=score >= pass_threshold)
    persist()
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image-only VLM judge for drawings_to_3d_building variants"
    )
    parser.add_argument("--reference-render-dir", required=True)
    parser.add_argument("--candidate-render-dir", required=True)
    parser.add_argument("--config", required=True, help="Per-variant eval_config.json")
    parser.add_argument(
        "--output-json", default=None, help="Optional path to write the JSON report"
    )
    parser.add_argument(
        "--model", default=None, help="Override the judge model (default: gpt-6-astra)"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_renders(
        Path(args.reference_render_dir).resolve(),
        Path(args.candidate_render_dir).resolve(),
        Path(args.config).resolve(),
        model=args.model,
        report_path=Path(args.output_json).resolve() if args.output_json else None,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
