from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

VIEW_NAMES = ['front', 'back', 'left', 'right', 'top_front', 'bottom_front']
SCENE_VIEW_NAMES = ['front', 'back', 'left', 'right']
DEFAULT_BLENDER = os.environ.get('BLENDER_TASK_REMOTE_BLENDER', r'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe')
DEFAULT_MULTIPART_SAMPLE_COUNT = 5
DEFAULT_MULTIPART_SAMPLE_SEED = 'uv_reproduction_multipart_eval_v1'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Remote hard eval for UV+material reproduction')
    p.add_argument('--task-shape', choices=['single_part', 'multi_part'], default='single_part')
    p.add_argument('--reference-obj')
    p.add_argument('--reference-mtl')
    p.add_argument('--reference-texture-dir')
    p.add_argument('--candidate-obj')
    p.add_argument('--candidate-mtl')
    p.add_argument('--candidate-texture-dir')
    p.add_argument('--reference-manifest')
    p.add_argument('--input-manifest')
    p.add_argument('--reference-dir')
    p.add_argument('--reference-images-dir')
    p.add_argument('--candidate-scene')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--blender-binary', default=DEFAULT_BLENDER)
    p.add_argument('--requires-uv', default='1')
    p.add_argument('--requires-mtl', default='0')
    p.add_argument('--requires-basecolor-texture', default='0')
    p.add_argument('--requires-color-match-gate', default='0')
    p.add_argument('--color-match-gate-threshold', type=float, default=0.0)
    p.add_argument('--multipart-sample-ratio', type=float, default=1.0)
    p.add_argument('--multipart-sample-count', type=int, default=DEFAULT_MULTIPART_SAMPLE_COUNT)
    p.add_argument('--multipart-sample-seed', default=DEFAULT_MULTIPART_SAMPLE_SEED)
    p.add_argument('--multipart-required-parts-json', default='')
    p.add_argument('--renderer-script', required=True)
    return p.parse_args()


def _flag(value: str | int | bool) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def _obj_has_uv(path: Path) -> bool:
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        return any(line.startswith('vt ') for line in f)


def _find_basecolor(mtl_path: Path) -> Path | None:
    if not mtl_path.exists():
        return None
    with mtl_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('map_Kd '):
                rel = line.strip().split(None, 1)[1]
                candidate = (mtl_path.parent / rel).resolve()
                if candidate.exists():
                    return candidate
    return None


def _render_single(obj_path: Path, out_dir: Path, blender_binary: str, renderer_script: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [blender_binary, '-b', '--factory-startup', '--python', str(renderer_script), '--', '--obj', str(obj_path), '--output-dir', str(out_dir)]
    subprocess.run(cmd, check=True)


def _render_multipart(scene_path: Path, parts: list[str], out_dir: Path, blender_binary: str, renderer_script: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender_binary,
        '-b',
        str(scene_path),
        '--python',
        str(renderer_script),
        '--',
        '--output-dir',
        str(out_dir),
        '--parts-json',
        json.dumps(parts),
    ]
    subprocess.run(cmd, check=True)
    manifest_path = out_dir / 'candidate_manifest.json'
    return json.loads(manifest_path.read_text(encoding='utf-8'))


def _render_multipart_with_required_parts(
    scene_path: Path,
    *,
    render_parts: list[str],
    required_parts: list[str],
    out_dir: Path,
    blender_binary: str,
    renderer_script: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        blender_binary,
        '-b',
        str(scene_path),
        '--python',
        str(renderer_script),
        '--',
        '--output-dir',
        str(out_dir),
        '--parts-json',
        json.dumps(render_parts),
        '--required-parts-json',
        json.dumps(required_parts),
    ]
    subprocess.run(cmd, check=True)
    manifest_path = out_dir / 'candidate_manifest.json'
    return json.loads(manifest_path.read_text(encoding='utf-8'))


def _load_rgb(path: Path) -> np.ndarray:
    if Image is not None:
        return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f'Failed to load image: {path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0
    raise RuntimeError('Neither Pillow nor OpenCV is available for image loading')


def _foreground_mask(img: np.ndarray) -> np.ndarray:
    bg = np.array([0.18, 0.19, 0.21], dtype=np.float32)
    diff = np.max(np.abs(img - bg[None, None, :]), axis=2)
    return diff > 0.04


def _silhouette_score(ref: np.ndarray, cand: np.ndarray) -> float:
    rm = _foreground_mask(ref)
    cm = _foreground_mask(cand)
    union = np.logical_or(rm, cm).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(rm, cm).sum()
    return float(inter / union)


def _color_score(ref: np.ndarray, cand: np.ndarray) -> float:
    mask = np.logical_or(_foreground_mask(ref), _foreground_mask(cand))
    if not mask.any():
        return 0.0
    diff = np.abs(ref - cand).mean(axis=2)
    mae = float(diff[mask].mean())
    return max(0.0, 1.0 - mae / 0.18)


def _saturation(img: np.ndarray) -> np.ndarray:
    maxc = img.max(axis=2)
    minc = img.min(axis=2)
    denom = np.maximum(maxc, 1e-6)
    return (maxc - minc) / denom


def _chroma_score(ref: np.ndarray, cand: np.ndarray) -> float:
    mask = np.logical_or(_foreground_mask(ref), _foreground_mask(cand))
    if not mask.any():
        return 0.0
    ref_sat = _saturation(ref)
    cand_sat = _saturation(cand)
    sat_diff = float(np.abs(ref_sat[mask] - cand_sat[mask]).mean())
    return max(0.0, 1.0 - sat_diff / 0.10)


def _rgb_chroma_magnitude(img: np.ndarray, mask: np.ndarray) -> float:
    pixels = img[mask]
    if pixels.size == 0:
        return 0.0
    centered = pixels - pixels.mean(axis=1, keepdims=True)
    mag = np.linalg.norm(centered, axis=1)
    return float(mag.mean())


def _structural_score(ref: np.ndarray, cand: np.ndarray) -> float:
    ref_g = ref.mean(axis=2)
    cand_g = cand.mean(axis=2)
    mu_x = float(ref_g.mean())
    mu_y = float(cand_g.mean())
    sigma_x = float(ref_g.var())
    sigma_y = float(cand_g.var())
    sigma_xy = float(((ref_g - mu_x) * (cand_g - mu_y)).mean())
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    if den <= 0:
        return 0.0
    return max(0.0, min(1.0, num / den))


def _image_metrics(ref_img: Path, cand_img: Path) -> dict[str, float]:
    ref = _load_rgb(ref_img)
    cand = _load_rgb(cand_img)
    silhouette = _silhouette_score(ref, cand)
    color = _color_score(ref, cand)
    chroma = _chroma_score(ref, cand)
    structural = _structural_score(ref, cand)
    mask = np.logical_or(_foreground_mask(ref), _foreground_mask(cand))
    ref_mag = _rgb_chroma_magnitude(ref, mask)
    cand_mag = _rgb_chroma_magnitude(cand, mask)
    ratio = 1.0 if ref_mag <= 1e-6 else float(cand_mag / ref_mag)
    score = 0.20 * silhouette + 0.25 * color + 0.40 * chroma + 0.15 * structural
    return {
        'silhouette_score': float(silhouette),
        'color_score': float(color),
        'chroma_score': float(chroma),
        'structural_score': float(structural),
        'colorfulness_ratio': max(0.0, min(1.0, ratio)),
        'hard_score': float(score),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _load_reference_parts(reference_manifest_path: Path) -> list[dict[str, Any]]:
    data = _load_json(reference_manifest_path)
    return data['parts'] if isinstance(data, dict) and 'parts' in data else data


def _resolve(path_value: str, package_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (package_root / path).resolve()


def _sample_names(names: list[str], limit: int) -> list[str]:
    if len(names) <= limit:
        return names
    step = max(1, len(names) // limit)
    return names[::step][:limit]


def _project_name_from_manifest(reference_manifest_path: Path) -> str:
    data = _load_json(reference_manifest_path)
    if isinstance(data, dict):
        value = data.get('project_name')
        if value:
            return str(value)
    return reference_manifest_path.parent.parent.name


def _stable_sample_parts(parts: list[dict[str, Any]], sample_count: int, *, seed: str, project_name: str) -> list[dict[str, Any]]:
    if not parts:
        return []
    sample_count = max(1, int(sample_count))
    sample_count = min(sample_count, len(parts))
    pool = sorted(parts, key=lambda item: str(item.get('name', '')))
    if sample_count >= len(pool):
        return list(pool)
    rng = random.Random(f'{seed}:{project_name}')
    sampled = rng.sample(pool, sample_count)
    return sorted(sampled, key=lambda item: str(item.get('name', '')))


def _eval_single(args: argparse.Namespace, report_path: Path) -> dict[str, Any]:
    ref_obj = Path(args.reference_obj).resolve()
    ref_mtl = Path(args.reference_mtl).resolve()
    cand_obj = Path(args.candidate_obj).resolve()
    cand_mtl = Path(args.candidate_mtl).resolve()
    cand_tex = Path(args.candidate_texture_dir).resolve()
    renderer_script = Path(args.renderer_script).resolve()
    out_dir = report_path.parent
    requires_uv = _flag(args.requires_uv)
    requires_mtl = _flag(args.requires_mtl)
    requires_basecolor_texture = _flag(args.requires_basecolor_texture)
    requires_color_match_gate = _flag(args.requires_color_match_gate)

    gate = {
        'candidate_obj_exists': cand_obj.exists(),
        'candidate_obj_has_uv': (cand_obj.exists() and _obj_has_uv(cand_obj)) if requires_uv else True,
        'candidate_mtl_exists': cand_mtl.exists() if requires_mtl else True,
        'candidate_texture_dir_exists': cand_tex.exists(),
        'basecolor_exists': bool(_find_basecolor(cand_mtl)) if (requires_basecolor_texture and cand_mtl.exists()) else True,
    }
    if not all(gate.values()):
        return {'score': 0.0, 'report_path': str(report_path), 'metrics': {'gate': gate}, 'frame_pairs': []}

    cand_render_dir = out_dir / 'candidate_renders'
    _render_single(cand_obj, cand_render_dir, args.blender_binary, renderer_script)

    # Prefer pre-rendered reference views when a complete set is staged. These MUST be
    # produced by this same renderer script (identical camera rig, lighting, resolution)
    # so the per-view metrics stay comparable. If the set is missing or incomplete, fall
    # back to rendering the reference OBJ live, which is always parity-safe.
    prerendered_dir = Path(args.reference_images_dir).resolve() if args.reference_images_dir else None
    if prerendered_dir is not None and all((prerendered_dir / f'{name}.png').exists() for name in VIEW_NAMES):
        ref_render_dir = prerendered_dir
    else:
        ref_render_dir = out_dir / 'reference_renders'
        _render_single(ref_obj, ref_render_dir, args.blender_binary, renderer_script)

    per_view = []
    pairs = []
    metrics_rows = []
    for name in VIEW_NAMES:
        ref_img = ref_render_dir / f'{name}.png'
        cand_img = cand_render_dir / f'{name}.png'
        row = _image_metrics(ref_img, cand_img)
        row['view'] = name
        per_view.append(row)
        metrics_rows.append(row)
        pairs.append({'view': name, 'reference_image': str(ref_img), 'candidate_image': str(cand_img)})

    silhouette_score = float(np.mean([row['silhouette_score'] for row in metrics_rows])) if metrics_rows else 0.0
    color_score = float(np.mean([row['color_score'] for row in metrics_rows])) if metrics_rows else 0.0
    chroma_score = float(np.mean([row['chroma_score'] for row in metrics_rows])) if metrics_rows else 0.0
    structural_score = float(np.mean([row['structural_score'] for row in metrics_rows])) if metrics_rows else 0.0
    colorfulness_ratio = float(np.mean([row['colorfulness_ratio'] for row in metrics_rows])) if metrics_rows else 0.0
    color_match_ok = (
        min(color_score, chroma_score) >= float(args.color_match_gate_threshold)
        and colorfulness_ratio >= 0.85
    ) if requires_color_match_gate else True
    hard_score = 0.0 if not color_match_ok else float(np.mean([row['hard_score'] for row in metrics_rows]))
    return {
        'task_shape': 'single_part',
        'score': hard_score,
        'report_path': str(report_path),
        'expected_part_count': 1,
        'evaluated_part_count': 1 if hard_score > 0.0 else 0,
        'missing_parts': [],
        'metrics': {
            'gate': {**gate, 'color_match_ok': color_match_ok},
            'per_view': per_view,
            'silhouette_score': silhouette_score,
            'color_score': color_score,
            'chroma_score': chroma_score,
            'colorfulness_ratio': colorfulness_ratio,
            'structural_score': structural_score,
            'hard_score': hard_score,
        },
        'frame_pairs': [
            {**pair, 'pair_type': 'part'}
            for pair in pairs
        ],
    }


def _eval_multi(args: argparse.Namespace, report_path: Path) -> dict[str, Any]:
    candidate_scene = Path(args.candidate_scene).resolve()
    reference_manifest = Path(args.reference_manifest).resolve()
    input_manifest = Path(args.input_manifest).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    renderer_script = Path(args.renderer_script).resolve()
    out_dir = report_path.parent
    requires_uv = _flag(args.requires_uv)
    requires_mtl = _flag(args.requires_mtl)
    requires_basecolor_texture = _flag(args.requires_basecolor_texture)
    requires_color_match_gate = _flag(args.requires_color_match_gate)

    if not candidate_scene.exists():
        return {'score': 0.0, 'report_path': str(report_path), 'metrics': {'gate': {'candidate_scene_exists': False}}, 'frame_pairs': []}

    ref_parts = _load_reference_parts(reference_manifest)
    project_name = _project_name_from_manifest(reference_manifest)
    all_expected_names = [entry['name'] for entry in ref_parts]
    sampled_ref_parts = _stable_sample_parts(
        ref_parts,
        args.multipart_sample_count,
        seed=str(args.multipart_sample_seed),
        project_name=project_name,
    )
    sampled_names = [entry['name'] for entry in sampled_ref_parts]
    candidate_render_dir = out_dir / 'candidate_scene_renders'
    candidate_manifest = _render_multipart_with_required_parts(
        candidate_scene,
        render_parts=sampled_names,
        required_parts=all_expected_names,
        out_dir=candidate_render_dir,
        blender_binary=args.blender_binary,
        renderer_script=renderer_script,
    )
    cand_parts = {part['name']: part for part in candidate_manifest.get('parts', [])}
    missing_parts = list(candidate_manifest.get('missing_parts', []))

    reference_package_root = reference_manifest.parent.parent
    input_root = input_manifest.parent
    scene_reference_dir = input_root / 'scene_reference_images'

    per_part = []
    part_scores = []
    scene_pairs = []
    frame_pairs = []
    part_color_scores = []
    part_chroma_scores = []
    part_colorfulness = []
    for entry in sampled_ref_parts:
        name = entry['name']
        cand_entry = cand_parts.get(name)
        ref_meta = _load_json(_resolve(entry['meta'], reference_package_root))
        ref_view = (ref_meta.get('views') or ref_meta.get('angles') or [])[0]
        ref_detail = _resolve(ref_view['detail_image'], reference_package_root)
        if not cand_entry:
            per_part.append({'name': name, 'status': 'missing', 'hard_score': 0.0})
            part_scores.append(0.0)
            continue
        gate = {
            'candidate_has_uv': cand_entry.get('has_uv', False) if requires_uv else True,
            'candidate_has_mtl': cand_entry.get('has_mtl', False) if requires_mtl else True,
            'candidate_has_basecolor_texture': cand_entry.get('has_basecolor_texture', False) if requires_basecolor_texture else True,
        }
        if not all(gate.values()):
            per_part.append({'name': name, 'status': 'gate_failed', 'gate': gate, 'hard_score': 0.0})
            part_scores.append(0.0)
            continue
        cand_view = (cand_entry.get('views') or [])[0]
        cand_detail = Path(cand_view['detail_image']).resolve()
        row = _image_metrics(ref_detail, cand_detail)
        row.update({'name': name, 'status': 'ok', 'gate': gate})
        per_part.append(row)
        part_scores.append(row['hard_score'])
        part_color_scores.append(row['color_score'])
        part_chroma_scores.append(row['chroma_score'])
        part_colorfulness.append(row['colorfulness_ratio'])

    scene_metrics = []
    candidate_scene_views = {item['view']: Path(item['image']).resolve() for item in candidate_manifest.get('scene_views', [])}
    for view_name in SCENE_VIEW_NAMES:
        ref_img = scene_reference_dir / f'{view_name}.png'
        cand_img = candidate_scene_views.get(view_name)
        if not ref_img.exists() or cand_img is None or not cand_img.exists():
            continue
        row = _image_metrics(ref_img, cand_img)
        row['view'] = view_name
        scene_metrics.append(row)
        scene_pairs.append({'view': f'scene:{view_name}', 'reference_image': str(ref_img), 'candidate_image': str(cand_img)})

    mean_part_score = float(np.mean(part_scores)) if part_scores else 0.0
    mean_scene_score = float(np.mean([row['hard_score'] for row in scene_metrics])) if scene_metrics else 0.0
    mean_color = float(np.mean(part_color_scores)) if part_color_scores else 0.0
    mean_chroma = float(np.mean(part_chroma_scores)) if part_chroma_scores else 0.0
    mean_colorfulness = float(np.mean(part_colorfulness)) if part_colorfulness else 0.0
    missing_penalty = 0.0 if missing_parts else 1.0
    badly_scored_parts = sum(
        1
        for row in per_part
        if row.get('status') == 'ok' and float(row.get('hard_score', 0.0)) < 0.5
    )
    quality_penalty = (0.75 ** badly_scored_parts) if badly_scored_parts else 1.0
    color_match_ok = (
        min(mean_color, mean_chroma) >= float(args.color_match_gate_threshold)
        and mean_colorfulness >= 0.85
    ) if requires_color_match_gate else True
    hard_score = 0.0 if not color_match_ok else (
        (0.85 * mean_part_score + 0.15 * mean_scene_score)
        * missing_penalty
        * quality_penalty
    )

    sampled_part_names = _sample_names([row['name'] for row in per_part if row.get('status') == 'ok'], 5)
    for row in per_part:
        if row.get('status') != 'ok' or row['name'] not in sampled_part_names:
            continue
        ref_entry = next(item for item in sampled_ref_parts if item['name'] == row['name'])
        ref_meta = _load_json(_resolve(ref_entry['meta'], reference_package_root))
        ref_view = (ref_meta.get('views') or ref_meta.get('angles') or [])[0]
        ref_detail = _resolve(ref_view['detail_image'], reference_package_root)
        cand_entry = cand_parts[row['name']]
        cand_view = (cand_entry.get('views') or [])[0]
        frame_pairs.append({
            'view': f"part:{row['name']}:front",
            'pair_type': 'part',
            'reference_image': str(ref_detail),
            'candidate_image': str(Path(cand_view['detail_image']).resolve()),
        })
    frame_pairs.extend(
        {**pair, 'pair_type': 'scene'}
        for pair in scene_pairs[:2]
    )

    return {
        'task_shape': 'multi_part',
        'score': float(hard_score),
        'report_path': str(report_path),
        'expected_part_count': len(all_expected_names),
        'sampled_part_count': len(sampled_names),
        'sampled_parts': sampled_names,
        'multipart_sample_seed': str(args.multipart_sample_seed),
        'evaluated_part_count': sum(1 for row in per_part if row.get('status') == 'ok'),
        'missing_parts': missing_parts,
        'metrics': {
            'gate': {
                'candidate_scene_exists': True,
                'color_match_ok': color_match_ok,
            },
            'part_count': len(all_expected_names),
            'sampled_part_count': len(sampled_names),
            'sampled_parts': sampled_names,
            'multipart_sample_seed': str(args.multipart_sample_seed),
            'evaluated_part_count': sum(1 for row in per_part if row.get('status') == 'ok'),
            'missing_parts': missing_parts,
            'mean_part_score': mean_part_score,
            'mean_scene_score': mean_scene_score,
            'mean_color_score': mean_color,
            'mean_chroma_score': mean_chroma,
            'mean_colorfulness_ratio': mean_colorfulness,
            'missing_penalty': missing_penalty,
            'badly_scored_part_count': badly_scored_parts,
            'quality_penalty': quality_penalty,
            'per_part': per_part,
            'scene_views': scene_metrics,
            'hard_score': float(hard_score),
        },
        'frame_pairs': frame_pairs,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'hard_eval_report.json'
    if args.task_shape == 'multi_part':
        payload = _eval_multi(args, report_path)
    else:
        payload = _eval_single(args, report_path)
    report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
