from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import dotenv_values
from PIL import Image, ImageOps, ImageDraw
from tasks.utils.evaluation import (
    eval_credentials_dir,
    llm_vision_binary_questions_sync,
    resolve_llm_judge_model,
)

MODEL = resolve_llm_judge_model(
    env_var='BLENDER_TASK_SOFT_EVAL_MODEL',
    default='gpt-4.1-mini',
)


def _load_api_key() -> str | None:
    evaluator_env = eval_credentials_dir() / "openai.env"
    if evaluator_env.is_file():
        value = dotenv_values(evaluator_env).get("OPENAI_API_KEY")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if os.environ.get("AGENTHLE_EVAL_CREDENTIALS_DIR"):
        return None
    repo_env = Path(__file__).resolve().parents[4] / "secret" / ".env"
    value = dotenv_values(repo_env).get("OPENAI_API_KEY")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _pair_sheet(reference_path: Path, candidate_path: Path, label: str, out_path: Path) -> Path:
    ref = Image.open(reference_path).convert('RGBA')
    cand = Image.open(candidate_path).convert('RGBA')
    height = max(ref.height, cand.height) + 64
    width = ref.width + cand.width
    canvas = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    canvas.paste(ref, (0, 40))
    canvas.paste(cand, (ref.width, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), f'{label} | left=reference | right=candidate', fill=(20, 20, 20, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _stack(paths: list[Path], out_path: Path) -> Path:
    images = [Image.open(p).convert('RGBA') for p in paths]
    width = max(i.width for i in images)
    height = sum(i.height for i in images)
    canvas = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    y = 0
    for img in images:
        if img.width != width:
            img = ImageOps.pad(img, (width, img.height), color=(255, 255, 255, 255))
        canvas.paste(img, (0, y))
        y += img.height
    canvas.save(out_path)
    return out_path


def run_local_soft_eval(frame_pairs: list[dict], local_tmp_dir: Path) -> float:
    if not frame_pairs:
        return 0.0
    if not _load_api_key():
        raise RuntimeError("OPENAI_API_KEY is not set for UV soft evaluation")
    local_tmp_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    for pair in frame_pairs:
        sheets.append(_pair_sheet(Path(pair['reference_image']), Path(pair['candidate_image']), pair['view'], local_tmp_dir / f"pair_{pair['view']}.png"))
    bundle = _stack(sheets, local_tmp_dir / 'soft_eval_bundle.png')
    prompt_context = (
        'You are evaluating a UV and material reproduction task. '
        'Each row shows left=reference and right=candidate. '
        'Judge whether the candidate has the correct texture placement, material appearance, and overall look. '
        'Minor lighting or tiny seam differences are acceptable. Large texture misalignment, wrong orientation, loss of the original color palette, grayscale/desaturated appearance when the reference is colorful, or clearly wrong material look are not acceptable. '
        'If the candidate loses important color information from the reference, it should fail the material-appearance question.'
    )
    questions = [
        'Is the texture placement and orientation correct enough to pass?',
        'Does the candidate preserve the reference material appearance and color palette well enough to pass?',
        'Are obvious UV seams, stretching, or texture artifacts absent enough for the result to pass?',
    ]
    data = llm_vision_binary_questions_sync(
        prompt_context=prompt_context,
        questions=questions,
        image_bytes_list=[bundle.read_bytes()],
        model=MODEL,
        temperature=0,
        api_key=_load_api_key(),
    )
    (local_tmp_dir / 'soft_eval_report.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return max(0.0, min(1.0, float(data.get('final_score', 0.0))))
