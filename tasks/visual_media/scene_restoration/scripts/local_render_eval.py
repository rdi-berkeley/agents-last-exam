from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from tasks.utils.evaluation import EvaluationContext, llm_vision_binary_questions_sync, resolve_llm_judge_model

logger = logging.getLogger(__name__)

MODEL = resolve_llm_judge_model(
    env_var="UNREAL_TASK_SOFT_EVAL_MODEL",
    default="gpt-4.1-mini",
)


def _load_api_key() -> str | None:
    direct = os.environ.get("OPENAI_API_KEY")
    if direct:
        return direct
    for parent in Path(__file__).resolve().parents:
        repo_env = parent / ".env"
        if not repo_env.exists():
            continue
        for line in repo_env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or not line.startswith("OPENAI_API_KEY="):
                continue
            value = line.split("=", 1)[1].strip()
            if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
                value = value[1:-1]
            if value:
                return value
    return None


def _load_panel(path: Path, *, fallback_size: tuple[int, int] | None = None) -> Image.Image:
    if path.exists():
        return Image.open(path).convert("RGBA")
    width, height = fallback_size or (1600, 900)
    canvas = Image.new("RGBA", (width, height), (245, 245, 245, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 24), "candidate render missing", fill=(30, 30, 30, 255))
    return canvas


def _pair_sheet(reference_path: Path, candidate_path: Path, label: str, out_path: Path) -> Path:
    ref = _load_panel(reference_path)
    cand = _load_panel(candidate_path, fallback_size=ref.size)
    ref = ImageOps.contain(ref, (960, 540))
    cand = ImageOps.contain(cand, (960, 540))
    panel_height = max(ref.height, cand.height)
    canvas = Image.new("RGBA", (ref.width + cand.width, panel_height + 64), (255, 255, 255, 255))
    canvas.paste(ref, (0, 40))
    canvas.paste(cand, (ref.width, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f"{label} | left=reference | right=candidate", fill=(20, 20, 20, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _stack(paths: list[Path], out_path: Path) -> Path:
    images = [Image.open(p).convert("RGBA") for p in paths]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    y = 0
    for image in images:
        if image.width != width:
            image = ImageOps.pad(image, (width, image.height), color=(255, 255, 255, 255))
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(out_path)
    return out_path


def _build_bundle_images(pair_sheets: list[Path], out_dir: Path, *, rows_per_bundle: int = 5) -> list[Path]:
    bundles: list[Path] = []
    for idx in range(0, len(pair_sheets), rows_per_bundle):
        chunk = pair_sheets[idx : idx + rows_per_bundle]
        bundles.append(_stack(chunk, out_dir / f"bundle_{len(bundles):02d}.png"))
    return bundles


def _build_soft_eval_prompt(task_summary: str) -> str:
    summary = task_summary.strip() or "Restore the removed scene region using the visible Unreal project assets."
    return (
        "You are evaluating an Unreal scene-restoration task. "
        "Each image contains multiple rows, and each row shows left=reference and right=candidate from the same fixed camera. "
        f"Task summary: {summary} "
        "Judge whether the removed region is restored correctly, object placement and scale are plausible, preserved context remains intact, and the overall scene appearance matches the reference views. "
        "Ignore tiny lighting/exposure differences and minor anti-aliasing noise. "
        "Large missing structures, obviously missing props, badly misplaced objects, or visible damage to scene context outside the removed region are not acceptable. "
        "Judge each question independently using only YES or NO."
    )


def run_local_soft_eval(
    *,
    task_tag: str,
    task_summary: str,
    frame_pairs: list[dict[str, str]],
    local_tmp_dir: Path,
) -> float | None:
    api_key = _load_api_key()
    if not api_key or not frame_pairs:
        return None

    local_tmp_dir.mkdir(parents=True, exist_ok=True)
    prompt = _build_soft_eval_prompt(task_summary)
    with EvaluationContext(
        task_tag=task_tag,
        mode="unreal_scene_restoration_soft_eval",
        output_dir=str(local_tmp_dir),
        auto_save=False,
        model=MODEL,
        frame_count=len(frame_pairs),
    ) as ctx:
        pair_sheets: list[Path] = []
        for idx, pair in enumerate(frame_pairs):
            pair_sheets.append(
                _pair_sheet(
                    Path(pair["reference_image"]),
                    Path(pair["candidate_image"]),
                    pair.get("view", f"view_{idx:02d}"),
                    local_tmp_dir / f"pair_{idx:02d}.png",
                )
            )
        bundle_images = _build_bundle_images(pair_sheets, local_tmp_dir)
        questions = [
            "Is the removed scene region restored well enough to pass?",
            "Are object placement and scale plausible enough to pass?",
            "Is the preserved scene context left intact enough to pass?",
            "Does the overall candidate scene match the reference views well enough to pass?",
        ]
        data = llm_vision_binary_questions_sync(
            prompt_context=prompt,
            questions=questions,
            image_bytes_list=[path.read_bytes() for path in bundle_images],
            model=MODEL,
            temperature=0,
            api_key=api_key,
        )
        question_scores = [
            float(item.get("score", 0.0))
            for item in data.get("results", [])
            if isinstance(item, dict)
        ]
        if question_scores:
            final_score = 1.0 if min(question_scores) >= 1.0 else 0.0
        else:
            final_score = max(0.0, min(1.0, float(data.get("final_score", 0.0))))
        data["final_score"] = final_score
        ctx.log_evaluation(
            identifier="scene_bundle",
            score=final_score,
            prompt=prompt,
            model=MODEL,
            vlm_response=json.dumps(data, ensure_ascii=False),
            frame_pair_count=len(frame_pairs),
            bundle_count=len(bundle_images),
            yes_count=data.get("yes_count"),
            question_count=data.get("question_count"),
        )
        ctx.add_score(final_score)
        _, details = ctx.finalize(
            final_score=final_score,
            frame_pair_count=len(frame_pairs),
            bundle_count=len(bundle_images),
        )
        details["judge_report"] = data
        (local_tmp_dir / "judge_report.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        logger.info("[unreal soft eval] task=%s final=%.4f bundles=%d", task_tag, final_score, len(bundle_images))
        return final_score
