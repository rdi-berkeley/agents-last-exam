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

SCENE_VIEW_NAMES = ['front', 'back', 'left', 'right']
DEFAULT_BLENDER = os.environ.get('BLENDER_TASK_REMOTE_BLENDER', r'C:\Program Files\Blender Foundation\Blender 5.0\blender.exe')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Remote hard eval for Blender static object generation')
    p.add_argument('--reference-manifest', required=True)
    p.add_argument('--evaluation-config', required=True)
    p.add_argument('--reference-dir', required=True)
    p.add_argument('--candidate-scene', required=True)
    p.add_argument('--candidate-objects-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--blender-binary', default=DEFAULT_BLENDER)
    p.add_argument('--sample-ratio', type=float, default=0.20)
    p.add_argument('--min-samples', type=int, default=3)
    p.add_argument('--max-samples', type=int, default=8)
    p.add_argument('--renderer-script', required=True)
    return p.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _resolve(path_value: str, package_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (package_root / path).resolve()


def _safe_part_filename(entry: dict[str, Any]) -> str:
    return f"{entry.get('path_name') or entry['name']}.obj"


def _obj_bbox(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    mins = None
    maxs = None
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            if not line.startswith('v '):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
            mins = xyz.copy() if mins is None else np.minimum(mins, xyz)
            maxs = xyz.copy() if maxs is None else np.maximum(maxs, xyz)
    if mins is None or maxs is None:
        return None
    return mins, maxs


def _placement_score(reference_obj: Path, candidate_obj: Path) -> float:
    ref_bbox = _obj_bbox(reference_obj)
    cand_bbox = _obj_bbox(candidate_obj)
    if ref_bbox is None or cand_bbox is None:
        return 0.0
    ref_min, ref_max = ref_bbox
    cand_min, cand_max = cand_bbox
    ref_center = (ref_min + ref_max) * 0.5
    cand_center = (cand_min + cand_max) * 0.5
    ref_size = np.maximum(ref_max - ref_min, 1e-6)
    cand_size = np.maximum(cand_max - cand_min, 1e-6)
    center_delta = float(np.linalg.norm(ref_center - cand_center))
    size_delta = float(np.linalg.norm(ref_size - cand_size) / np.linalg.norm(ref_size))
    center_score = max(0.0, 1.0 - center_delta / (max(float(ref_size.max()), 1e-6) * 1.5))
    size_score = max(0.0, 1.0 - size_delta)
    return float(0.8 * center_score + 0.2 * size_score)


def _render_scene(scene_path: Path, parts: list[str], out_dir: Path, blender_binary: str, renderer_script: Path) -> dict[str, Any]:
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
    manifest_path = out_dir / 'candidate_manifest.json'
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        return {
            'parts': [],
            'scene_views': [],
            'missing_parts': list(parts),
            'render_error': 'renderer_failed',
        }
    if not manifest_path.exists():
        return {
            'parts': [],
            'scene_views': [],
            'missing_parts': list(parts),
            'render_error': 'candidate_manifest_missing',
        }
    return _load_json(manifest_path)


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
    structural = _structural_score(ref, cand)
    score = 0.45 * silhouette + 0.30 * color + 0.25 * structural
    return {
        'silhouette_score': float(silhouette),
        'color_score': float(color),
        'structural_score': float(structural),
        'hard_score': float(score),
    }


def _write_missing_placeholder(reference_image: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if Image is not None:
        with Image.open(reference_image).convert('RGBA') as ref_img:
            placeholder = Image.new('RGBA', ref_img.size, (46, 48, 54, 255))
            placeholder.save(out_path)
        return out_path
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    if cv2 is None:
        raise RuntimeError('Neither Pillow nor OpenCV is available for placeholder generation')
    ref = cv2.imread(str(reference_image), cv2.IMREAD_UNCHANGED)
    if ref is None:
        raise RuntimeError(f'Failed to load reference image for placeholder: {reference_image}')
    placeholder = np.zeros_like(ref)
    if placeholder.ndim == 3 and placeholder.shape[2] >= 3:
        placeholder[..., 0] = 54
        placeholder[..., 1] = 48
        placeholder[..., 2] = 46
        if placeholder.shape[2] == 4:
            placeholder[..., 3] = 255
    cv2.imwrite(str(out_path), placeholder)
    return out_path


def _sample_editable_parts(parts: list[dict[str, Any]], ratio: float, minimum: int, maximum: int) -> list[dict[str, Any]]:
    editable = [part for part in parts if part.get('part_role', 'editable') == 'editable']
    if not editable:
        return []
    sample_count = max(minimum, int(round(len(editable) * ratio)))
    sample_count = min(maximum, sample_count, len(editable))
    if sample_count >= len(editable):
        return list(editable)
    indexes = sorted(random.sample(range(len(editable)), sample_count))
    return [editable[idx] for idx in indexes]


def _load_reference_parts(reference_manifest_path: Path) -> list[dict[str, Any]]:
    data = _load_json(reference_manifest_path)
    return data['parts'] if isinstance(data, dict) and 'parts' in data else data


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'hard_eval_report.json'

    candidate_scene = Path(args.candidate_scene).resolve()
    candidate_objects_dir = Path(args.candidate_objects_dir).resolve()
    reference_manifest = Path(args.reference_manifest).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    renderer_script = Path(args.renderer_script).resolve()
    evaluation_config = _load_json(Path(args.evaluation_config).resolve())

    if not candidate_scene.exists():
        payload = {'score': 0.0, 'report_path': str(report_path), 'metrics': {'gate': {'candidate_scene_exists': False}}, 'frame_pairs': []}
        report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(json.dumps(payload))
        return

    ref_parts = _load_reference_parts(reference_manifest)
    all_expected_names = [entry['name'] for entry in ref_parts]
    sampled_ref_parts = _sample_editable_parts(ref_parts, args.sample_ratio, args.min_samples, args.max_samples)
    sampled_names = [entry['name'] for entry in sampled_ref_parts]

    candidate_render_dir = out_dir / 'candidate_scene_renders'
    candidate_manifest = _render_scene(candidate_scene, sampled_names, candidate_render_dir, args.blender_binary, renderer_script)
    cand_parts = {part['name']: part for part in candidate_manifest.get('parts', [])}
    missing_in_scene = list(candidate_manifest.get('missing_parts', []))

    reference_package_root = reference_manifest.parent.parent
    per_part = []
    sampled_render_scores = []
    bundle_rows = []
    frame_pairs = []
    placement_scores = []

    expected_object_names = []
    missing_bundle_parts = []
    for entry in ref_parts:
        obj_name = _safe_part_filename(entry)
        expected_object_names.append(obj_name)
        obj_path = candidate_objects_dir / obj_name
        if not obj_path.exists():
            missing_bundle_parts.append(entry['name'])

    for entry in sampled_ref_parts:
        name = entry['name']
        obj_filename = _safe_part_filename(entry)
        obj_path = candidate_objects_dir / obj_filename
        cand_entry = cand_parts.get(name)
        ref_meta = _load_json(_resolve(entry['meta'], reference_package_root))
        ref_view = (ref_meta.get('views') or [])[0]
        ref_detail = _resolve(ref_view['detail_image'], reference_package_root)
        ref_object = _resolve(entry['object_path'], reference_package_root)
        bundle_gate = {
            'obj_exists': obj_path.exists(),
        }
        bundle_score = float(bundle_gate['obj_exists'])
        bundle_rows.append({'name': name, 'bundle_gate': bundle_gate, 'bundle_score': bundle_score})
        if not cand_entry:
            per_part.append({'name': name, 'status': 'missing_in_scene', 'bundle_gate': bundle_gate, 'bundle_score': bundle_score, 'hard_score': 0.0})
            sampled_render_scores.append(0.0)
            placeholder = _write_missing_placeholder(ref_detail, out_dir / 'missing_placeholders' / f'{name}.png')
            frame_pairs.append({'view': f'part:{name}:front', 'pair_type': 'part', 'reference_image': str(ref_detail), 'candidate_image': str(placeholder)})
            continue
        cand_view = (cand_entry.get('views') or [])[0]
        cand_detail = Path(cand_view['detail_image']).resolve()
        row = _image_metrics(ref_detail, cand_detail)
        placement_score = _placement_score(ref_object, obj_path)
        placement_scores.append(placement_score)
        row.update({'name': name, 'status': 'ok', 'bundle_gate': bundle_gate, 'bundle_score': bundle_score, 'placement_score': placement_score})
        render_score = row['hard_score']
        part_score = 0.70 * render_score + 0.30 * placement_score
        row['part_score'] = part_score
        per_part.append(row)
        sampled_render_scores.append(part_score)
        frame_pairs.append({'view': f'part:{name}:front', 'pair_type': 'part', 'reference_image': str(ref_detail), 'candidate_image': str(cand_detail)})

    scene_metrics = []
    scene_pairs = []
    candidate_scene_views = {item['view']: Path(item['image']).resolve() for item in candidate_manifest.get('scene_views', [])}
    for scene_cfg in evaluation_config.get('scene_views', []):
        view_name = scene_cfg['view']
        ref_img = _resolve(scene_cfg['image'], reference_package_root)
        cand_img = candidate_scene_views.get(view_name)
        if not ref_img.exists() or cand_img is None or not cand_img.exists():
            continue
        row = _image_metrics(ref_img, cand_img)
        row['view'] = view_name
        scene_metrics.append(row)
        scene_pairs.append({'view': f'scene:{view_name}', 'pair_type': 'scene', 'reference_image': str(ref_img), 'candidate_image': str(cand_img)})

    mean_part_score = float(np.mean(sampled_render_scores)) if sampled_render_scores else 0.0
    mean_scene_score = float(np.mean([row['hard_score'] for row in scene_metrics])) if scene_metrics else 0.0
    mean_bundle_score = float(np.mean([row['bundle_score'] for row in bundle_rows])) if bundle_rows else 0.0

    missing_scene_penalty = (0.50 ** len(missing_in_scene)) if missing_in_scene else 1.0
    missing_bundle_penalty = (0.75 ** len(missing_bundle_parts)) if missing_bundle_parts else 1.0
    placement_penalty = (0.50 + 0.50 * min(placement_scores)) if placement_scores else 1.0

    hard_score = (
        (0.60 * mean_part_score + 0.35 * mean_scene_score + 0.05 * mean_bundle_score)
        * missing_scene_penalty
        * missing_bundle_penalty
        * placement_penalty
    )
    hard_score = float(max(0.0, min(1.0, hard_score)))

    frame_pairs = frame_pairs[:5] + scene_pairs[:2]
    payload = {
        'task_shape': 'multi_part',
        'score': hard_score,
        'report_path': str(report_path),
        'expected_part_count': len(all_expected_names),
        'sampled_part_count': len(sampled_names),
        'sampled_parts': sampled_names,
        'evaluated_part_count': sum(1 for row in per_part if row.get('status') == 'ok'),
        'missing_parts': missing_in_scene,
        'missing_bundle_parts': missing_bundle_parts,
        'metrics': {
            'gate': {
                'candidate_scene_exists': True,
                'candidate_objects_dir_exists': candidate_objects_dir.exists(),
            },
            'mean_part_score': mean_part_score,
            'mean_scene_score': mean_scene_score,
            'mean_bundle_score': mean_bundle_score,
            'missing_scene_penalty': missing_scene_penalty,
            'missing_bundle_penalty': missing_bundle_penalty,
            'placement_penalty': placement_penalty,
            'per_part': per_part,
            'scene_views': scene_metrics,
            'bundle_rows': bundle_rows,
            'hard_score': hard_score,
        },
        'frame_pairs': frame_pairs,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
