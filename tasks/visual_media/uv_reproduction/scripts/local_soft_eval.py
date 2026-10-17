from __future__ import annotations

import json
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw
from tasks.utils.evaluation import (
    judge_config,
    llm_vision_binary_questions_sync,
    resolve_llm_judge_model,
)


def _pair_sheet(reference: str | Path | bytes, candidate: str | Path | bytes, label: str) -> Image.Image:
    with ExitStack() as resources:
        with Image.open(BytesIO(reference) if isinstance(reference, bytes) else reference) as source:
            ref = source.convert('RGBA')
            resources.callback(ref.close)
        with Image.open(BytesIO(candidate) if isinstance(candidate, bytes) else candidate) as source:
            cand = source.convert('RGBA')
            resources.callback(cand.close)
        height = max(ref.height, cand.height) + 64
        width = ref.width + cand.width
        canvas = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        canvas.paste(ref, (0, 40))
        canvas.paste(cand, (ref.width, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), f'{label} | left=reference | right=candidate', fill=(20, 20, 20, 255))
        return canvas


def _stack(images: list[Image.Image]) -> Image.Image:
    width = max(i.width for i in images)
    height = sum(i.height for i in images)
    canvas = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    y = 0
    for img in images:
        if img.width != width:
            padded = ImageOps.pad(img, (width, img.height), color=(255, 255, 255, 255))
            try:
                canvas.paste(padded, (0, y))
            finally:
                padded.close()
        else:
            canvas.paste(img, (0, y))
        y += img.height
    return canvas


def run_local_soft_eval(
    frame_pairs: list[dict],
    local_tmp_dir: Path | None = None,
    *,
    report_out: dict | None = None,
) -> float:
    """Judge image bytes or paths; optionally collect the full report or write diagnostics."""
    if not frame_pairs:
        return 0.0
    if not judge_config()["api_key"]:
        raise RuntimeError("JUDGE_API_KEY is not set for UV soft evaluation")
    if local_tmp_dir is not None:
        local_tmp_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    with BytesIO() as encoded_bundle:
        try:
            for pair in frame_pairs:
                sheet = _pair_sheet(pair['reference_image'], pair['candidate_image'], pair['view'])
                sheets.append(sheet)
                if local_tmp_dir is not None:
                    sheet.save(local_tmp_dir / f"pair_{pair['view']}.png")
            bundle = _stack(sheets)
            try:
                bundle.save(encoded_bundle, format='PNG')
            finally:
                bundle.close()
            bundle_bytes = encoded_bundle.getvalue()
        finally:
            for sheet in sheets:
                sheet.close()
    if local_tmp_dir is not None:
        (local_tmp_dir / 'soft_eval_bundle.png').write_bytes(bundle_bytes)
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
        image_bytes_list=[bundle_bytes],
        model=resolve_llm_judge_model(),
        temperature=0,
        api_key=judge_config()["api_key"],
    )
    if report_out is not None:
        report_out.update(data)
    if local_tmp_dir is not None:
        (local_tmp_dir / 'soft_eval_report.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return max(0.0, min(1.0, float(data.get('final_score', 0.0))))
