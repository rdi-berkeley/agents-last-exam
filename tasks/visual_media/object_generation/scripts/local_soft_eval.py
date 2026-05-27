from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw
from tasks.utils.evaluation import llm_vision_binary_questions_sync, resolve_llm_judge_model

MODEL = resolve_llm_judge_model(
    env_var='BLENDER_TASK_SOFT_EVAL_MODEL',
    default='gpt-4.1-mini',
)


def _load_api_key() -> str | None:
    direct = os.environ.get('OPENAI_API_KEY')
    if direct:
        return direct
    repo_env = Path(__file__).resolve().parents[4] / '.env'
    if not repo_env.exists():
        return None
    for line in repo_env.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or not line.startswith('OPENAI_API_KEY='):
            continue
        value = line.split('=', 1)[1].strip()
        if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
            value = value[1:-1]
        if value:
            return value
    return None


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
    api_key = _load_api_key()
    if not api_key or not frame_pairs:
        return 0.0
    local_tmp_dir.mkdir(parents=True, exist_ok=True)
    sheets = []
    for idx, pair in enumerate(frame_pairs):
        label = pair.get('view', f'pair_{idx}')
        sheets.append(
            _pair_sheet(
                Path(pair['reference_image']),
                Path(pair['candidate_image']),
                label,
                local_tmp_dir / f'pair_{idx:02d}.png',
            )
        )
    bundle = _stack(sheets, local_tmp_dir / 'soft_eval_bundle.png')
    prompt_context = (
        'You are evaluating a Blender static-object completion task. '
        'Each row shows left=reference and right=candidate. '
        'Judge whether the candidate correctly restores the missing geometry, keeps the object coherent as a whole, and preserves plausible material/UV appearance. '
        'Large part omissions, wrong placement, obviously broken proportions, or severe material mismatch are not acceptable. '
        'If a clearly visible part exists in the reference but is absent in the candidate, the restoration and coherence questions should fail; do not treat the result as acceptable just because the rest of the object looks correct. '
        'Minor lighting, tiny seam differences, or small texture variation are acceptable. '
        'Judge each question independently using only YES or NO.'
    )
    questions = [
        'Is the missing geometry restored well enough to pass?',
        'Is the part placement and alignment correct enough to pass?',
        'Is the whole object coherent and complete enough to pass?',
        'Is the final material appearance acceptable enough to pass?',
    ]
    data = llm_vision_binary_questions_sync(
        prompt_context=prompt_context,
        questions=questions,
        image_bytes_list=[bundle.read_bytes()],
        model=MODEL,
        temperature=0,
        api_key=api_key,
    )
    (local_tmp_dir / 'soft_eval_report.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return max(0.0, min(1.0, float(data.get('final_score', 0.0))))
