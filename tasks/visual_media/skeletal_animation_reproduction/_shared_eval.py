from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

VALIDITY_MIN_MOTION_EXTENT = 0.01
PREVIEW_MIN_FOREGROUND_COVERAGE = 0.005


def evenly_spaced_positions(sample_count: int) -> list[float]:
    if sample_count <= 1:
        return [0.0]
    return [idx / (sample_count - 1) for idx in range(sample_count)]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def extract_reference_frames(
    video_path: Path,
    sample_positions: list[float],
    output_dir: Path,
    prefix: str = "reference",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,duration,r_frame_rate",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    stream = meta["streams"][0]
    total_frames = int(stream.get("nb_frames") or 0)
    if total_frames <= 0:
        fps_num, fps_den = [int(x) for x in stream["r_frame_rate"].split("/")]
        fps = fps_num / max(fps_den, 1)
        duration = float(stream.get("duration") or 0.0)
        total_frames = max(1, int(round(duration * fps)))
    frame_indices = []
    for pos in sample_positions:
        raw = int(round(pos * max(total_frames - 1, 0)))
        frame_indices.append(min(max(raw, 0), max(total_frames - 1, 0)))

    outputs: list[Path] = []
    for idx, frame_index in enumerate(frame_indices):
        out_path = output_dir / f"{prefix}_{idx:02d}.png"
        _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"select='eq(n\\,{frame_index})'",
                "-vsync",
                "0",
                "-frames:v",
                "1",
                str(out_path),
            ]
        )
        outputs.append(out_path)
    return outputs


def _load_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)


def _resize_like(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image, mode="RGBA")
    resized = pil.resize((target_shape[1], target_shape[0]), Image.Resampling.BILINEAR)
    return np.array(resized, dtype=np.uint8)


def _estimate_background(image: np.ndarray) -> np.ndarray:
    patch = np.concatenate(
        [
            image[:12, :, :3].reshape(-1, 3),
            image[-12:, :, :3].reshape(-1, 3),
            image[:, :12, :3].reshape(-1, 3),
            image[:, -12:, :3].reshape(-1, 3),
        ],
        axis=0,
    )
    luminance = patch.astype(np.float32).mean(axis=1)
    keep = patch[luminance >= np.quantile(luminance, 0.5)]
    if keep.size == 0:
        keep = patch
    return np.median(keep, axis=0)


def extract_foreground_mask(image: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    alpha = image[:, :, 3]
    alpha_coverage = float(np.mean(alpha > 8))
    if float(alpha.max()) > 8.0 and 0.01 < alpha_coverage < 0.99:
        return _largest_component(alpha > 8)
    background = _estimate_background(image)
    diff = np.linalg.norm(image[:, :, :3].astype(np.float32) - background.astype(np.float32), axis=2)
    similar = diff <= threshold
    background_mask = _border_connected(similar)
    return _largest_component(~background_mask)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_coords: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(coords) > len(best_coords):
                best_coords = coords
    out = np.zeros_like(mask, dtype=bool)
    for y, x in best_coords:
        out[y, x] = True
    return out


def _border_connected(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(w):
        if mask[0, x]:
            stack.append((0, x))
        if mask[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if mask[y, 0]:
            stack.append((y, 0))
        if mask[y, w - 1]:
            stack.append((y, w - 1))
    while stack:
        y, x = stack.pop()
        if visited[y, x] or not mask[y, x]:
            continue
        visited[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx]:
                stack.append((ny, nx))
    return visited


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return (0.5, 0.5, 0.0, 0.0)
    y0 = ys.min()
    y1 = ys.max()
    x0 = xs.min()
    x1 = xs.max()
    h, w = mask.shape
    cx = ((x0 + x1) / 2.0) / max(w, 1)
    cy = ((y0 + y1) / 2.0) / max(h, 1)
    bw = (x1 - x0 + 1) / max(w, 1)
    bh = (y1 - y0 + 1) / max(h, 1)
    return (cx, cy, bw, bh)


def silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union <= 0:
        return 0.0
    return float(inter / union)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0 = a[0] - a[2] / 2.0
    ay0 = a[1] - a[3] / 2.0
    ax1 = a[0] + a[2] / 2.0
    ay1 = a[1] + a[3] / 2.0
    bx0 = b[0] - b[2] / 2.0
    by0 = b[1] - b[3] / 2.0
    bx1 = b[0] + b[2] / 2.0
    by1 = b[1] + b[3] / 2.0
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _bbox_alignment_score(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    center_error = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    size_error = abs(a[2] - b[2]) + abs(a[3] - b[3])
    center_score = _score_from_error(center_error, 0.20)
    size_score = _score_from_error(size_error, 0.30)
    bbox_iou = _bbox_iou(a, b)
    return 0.5 * bbox_iou + 0.25 * center_score + 0.25 * size_score


def reference_framing_targets(frame_paths: list[Path]) -> list[dict[str, float]]:
    targets: list[dict[str, float]] = []
    for path in frame_paths:
        image = _load_rgba(path)
        bbox = mask_bbox(extract_foreground_mask(image))
        targets.append(
            {
                "center_x": float(bbox[0]),
                "center_y": float(bbox[1]),
                "width": float(max(bbox[2], 1e-3)),
                "height": float(max(bbox[3], 1e-3)),
            }
        )
    return targets


def masked_character_similarity(reference_image: np.ndarray, candidate_image: np.ndarray) -> float:
    target_shape = reference_image.shape[:2]
    candidate_image = _resize_like(candidate_image, target_shape)
    ref_mask = extract_foreground_mask(reference_image)
    cand_mask = extract_foreground_mask(candidate_image)
    union = np.logical_or(ref_mask, cand_mask)
    if union.sum() <= 0:
        return 0.0

    ref_gray = np.array(Image.fromarray(reference_image[:, :, :3]).convert("L"), dtype=np.float32)
    cand_gray = np.array(Image.fromarray(candidate_image[:, :, :3]).convert("L"), dtype=np.float32)
    ref_edges = np.array(Image.fromarray(ref_gray.astype(np.uint8)).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    cand_edges = np.array(Image.fromarray(cand_gray.astype(np.uint8)).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    diff = np.abs(ref_edges - cand_edges) / 255.0
    score = 1.0 - float(diff[union].mean())
    return max(0.0, min(1.0, score))


def _preview_health_metrics(frame_paths: list[Path]) -> dict[str, float]:
    if not frame_paths:
        return {"foreground_coverage_mean": 0.0}
    coverages = []
    for path in frame_paths:
        mask = extract_foreground_mask(_load_rgba(path))
        coverages.append(float(mask.mean()))
    return {"foreground_coverage_mean": float(np.mean(coverages)) if coverages else 0.0}


def _stack_horizontal(paths: list[Path], out_path: Path, title: str | None = None) -> Path:
    images = [_composite_on_white(path) for path in paths]
    if not images:
        blank = Image.new("RGBA", (32, 32), (255, 255, 255, 255))
        blank.save(out_path)
        return out_path
    header = 36 if title else 0
    width = sum(image.width for image in images)
    height = max(image.height for image in images) + header
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((12, 10), title, fill=(20, 20, 20, 255))
    x = 0
    for image in images:
        canvas.paste(image, (x, header))
        x += image.width
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _stack_vertical(paths: list[Path], out_path: Path) -> Path:
    images = [_composite_on_white(path) for path in paths]
    if not images:
        blank = Image.new("RGBA", (32, 32), (255, 255, 255, 255))
        blank.save(out_path)
        return out_path
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    y = 0
    for image in images:
        if image.width != width:
            image = ImageOps.pad(image, (width, image.height), color=(255, 255, 255, 255))
        canvas.paste(image, (0, y))
        y += image.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _composite_on_white(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
    canvas.alpha_composite(image)
    return canvas


def _score_from_error(error: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - error / tolerance))


def _compute_video_comparison(reference_frame_paths: list[Path], candidate_frame_paths: list[Path]) -> dict[str, float]:
    reference_images = [_load_rgba(path) for path in reference_frame_paths]
    candidate_images = [_load_rgba(path) for path in candidate_frame_paths]
    reference_masks = [extract_foreground_mask(image) for image in reference_images]
    candidate_masks = [extract_foreground_mask(image) for image in candidate_images]

    ious = []
    masked_scores = []
    temporal_delta_scores = []
    for ref_image, ref_mask, cand_image, cand_mask in zip(reference_images, reference_masks, candidate_images, candidate_masks):
        cand_image_resized = _resize_like(cand_image, ref_image.shape[:2])
        cand_mask_resized = extract_foreground_mask(cand_image_resized)
        ious.append(silhouette_iou(ref_mask, cand_mask_resized))
        masked_scores.append(masked_character_similarity(ref_image, cand_image_resized))
    for idx in range(1, min(len(reference_masks), len(candidate_masks))):
        ref_delta = np.logical_xor(reference_masks[idx - 1], reference_masks[idx]).mean()
        cand_delta = np.logical_xor(candidate_masks[idx - 1], candidate_masks[idx]).mean()
        temporal_delta_scores.append(1.0 - min(1.0, abs(ref_delta - cand_delta) / 0.25))

    silhouette_iou_mean = float(np.mean(ious)) if ious else 0.0
    masked_similarity = float(np.mean(masked_scores)) if masked_scores else 0.0
    temporal_delta_score = float(np.mean(temporal_delta_scores)) if temporal_delta_scores else 0.0
    score = 0.5 * silhouette_iou_mean + 0.35 * masked_similarity + 0.15 * temporal_delta_score
    return {
        "silhouette_iou_mean": silhouette_iou_mean,
        "masked_character_similarity": masked_similarity,
        "temporal_delta_score": temporal_delta_score,
        "score": score,
    }


def _compute_replay_comparison(preview_frame_paths: list[Path], replay_frame_paths: list[Path]) -> dict[str, float]:
    preview_images = [_load_rgba(path) for path in preview_frame_paths]
    replay_images = [_load_rgba(path) for path in replay_frame_paths]
    preview_masks = [extract_foreground_mask(image) for image in preview_images]
    replay_masks = [extract_foreground_mask(image) for image in replay_images]

    silhouette_scores = []
    masked_scores = []
    temporal_delta_scores = []
    bbox_alignment_scores = []
    for prev_image, prev_mask, replay_image in zip(preview_images, preview_masks, replay_images):
        replay_resized = _resize_like(replay_image, prev_image.shape[:2])
        replay_mask = extract_foreground_mask(replay_resized)
        silhouette_scores.append(silhouette_iou(prev_mask, replay_mask))
        masked_scores.append(masked_character_similarity(prev_image, replay_resized))
        bbox_alignment_scores.append(_bbox_alignment_score(mask_bbox(prev_mask), mask_bbox(replay_mask)))
    for idx in range(1, min(len(preview_masks), len(replay_masks))):
        prev_delta = np.logical_xor(preview_masks[idx - 1], preview_masks[idx]).mean()
        replay_delta = np.logical_xor(replay_masks[idx - 1], replay_masks[idx]).mean()
        temporal_delta_scores.append(1.0 - min(1.0, abs(prev_delta - replay_delta) / 0.25))

    silhouette_iou_mean = float(np.mean(silhouette_scores)) if silhouette_scores else 0.0
    masked_similarity = float(np.mean(masked_scores)) if masked_scores else 0.0
    temporal_delta_score = float(np.mean(temporal_delta_scores)) if temporal_delta_scores else 0.0
    bbox_alignment_mean = float(np.mean(bbox_alignment_scores)) if bbox_alignment_scores else 0.0
    score = (
        0.20 * silhouette_iou_mean
        + 0.35 * masked_similarity
        + 0.20 * temporal_delta_score
        + 0.25 * bbox_alignment_mean
    )
    return {
        "silhouette_iou_mean": silhouette_iou_mean,
        "masked_character_similarity": masked_similarity,
        "temporal_delta_score": temporal_delta_score,
        "bbox_alignment_score": bbox_alignment_mean,
        "score": score,
    }


def _resolve_package_bone_name(package: dict[str, Any], semantic: str) -> str | None:
    meta = package.get("bone_semantics", {}).get(semantic, {})
    primary = meta.get("primary")
    if primary:
        return str(primary)
    aliases = meta.get("aliases") or []
    return str(aliases[0]) if aliases else None


def _score_joint_ranges(skeleton_package: dict[str, Any], candidate_joint_ranges: dict[str, Any]) -> dict[str, float]:
    per_bone_scores = []
    for semantic in skeleton_package.get("required_bones", []):
        target = skeleton_package.get("joint_range_targets", {}).get(semantic, {})
        observed = candidate_joint_ranges.get(semantic, {})
        if not target or not observed:
            per_bone_scores.append(0.0)
            continue
        target_mag = max(float(target.get("magnitude", 0.0)), 1e-6)
        observed_mag = float(observed.get("magnitude", 0.0))
        magnitude_score = max(0.0, min(1.0, observed_mag / target_mag))
        axis_scores = []
        for axis in ("x", "y", "z"):
            axis_target = max(float(target.get(axis, 0.0)), 1e-6)
            axis_value = float(observed.get(axis, 0.0))
            axis_scores.append(max(0.0, min(1.0, axis_value / axis_target)))
        per_bone_scores.append(0.55 * magnitude_score + 0.45 * float(np.mean(axis_scores)))
    joint_range_score = float(np.mean(per_bone_scores)) if per_bone_scores else 0.0
    return {"joint_range_score": joint_range_score}


def _score_pose_states(
    *,
    skeleton_package: dict[str, Any],
    candidate_pose_states: list[dict[str, Any]],
    package_root: Path,
) -> dict[str, float]:
    candidate_by_name = {str(item.get("name")): item for item in candidate_pose_states}
    silhouette_scores = []
    endpoint_scores = []
    angle_scores = []
    pose_scores = []

    for pose_state in skeleton_package.get("pose_states", []):
        name = str(pose_state.get("name"))
        candidate = candidate_by_name.get(name)
        if not candidate:
            silhouette_scores.append(0.0)
            endpoint_scores.append(0.0)
            angle_scores.append(0.0)
            pose_scores.append(0.0)
            continue

        gt_image_path = package_root / str(pose_state.get("gt_pose_image", ""))
        candidate_image_path = Path(str(candidate.get("image_path", "")))
        if not gt_image_path.exists() or not candidate_image_path.exists():
            silhouette_scores.append(0.0)
            endpoint_scores.append(0.0)
            angle_scores.append(0.0)
            pose_scores.append(0.0)
            continue

        gt_mask = extract_foreground_mask(_load_rgba(gt_image_path))
        cand_mask = extract_foreground_mask(_load_rgba(candidate_image_path))
        silhouette_score = silhouette_iou(gt_mask, cand_mask)

        endpoint_errors = []
        angle_terms = []
        gt_bones = pose_state.get("bone_states", {})
        cand_bones = candidate.get("bone_states", {})
        for semantic in skeleton_package.get("required_bones", []):
            gt_bone = gt_bones.get(semantic)
            cand_bone = cand_bones.get(semantic)
            if not gt_bone or not cand_bone:
                endpoint_errors.append(1.0)
                angle_terms.append(0.0)
                continue
            gt_head = np.array(gt_bone.get("head", [0.0, 0.0, 0.0]), dtype=np.float32)
            gt_tail = np.array(gt_bone.get("tail", [0.0, 0.0, 0.0]), dtype=np.float32)
            cand_head = np.array(cand_bone.get("head", [0.0, 0.0, 0.0]), dtype=np.float32)
            cand_tail = np.array(cand_bone.get("tail", [0.0, 0.0, 0.0]), dtype=np.float32)
            endpoint_errors.append(float((np.linalg.norm(gt_head - cand_head) + np.linalg.norm(gt_tail - cand_tail)) / 2.0))

            gt_dir = np.array(gt_bone.get("direction", [0.0, 0.0, 0.0]), dtype=np.float32)
            cand_dir = np.array(cand_bone.get("direction", [0.0, 0.0, 0.0]), dtype=np.float32)
            gt_norm = float(np.linalg.norm(gt_dir))
            cand_norm = float(np.linalg.norm(cand_dir))
            if gt_norm <= 1e-6 or cand_norm <= 1e-6:
                angle_terms.append(0.0)
            else:
                dot = float(np.clip(np.dot(gt_dir / gt_norm, cand_dir / cand_norm), -1.0, 1.0))
                angle_terms.append((dot + 1.0) / 2.0)

        endpoint_score = _score_from_error(float(np.mean(endpoint_errors)) if endpoint_errors else 1.0, 0.25)
        angle_score = float(np.mean(angle_terms)) if angle_terms else 0.0
        pose_score = 0.4 * silhouette_score + 0.3 * endpoint_score + 0.3 * angle_score

        silhouette_scores.append(silhouette_score)
        endpoint_scores.append(endpoint_score)
        angle_scores.append(angle_score)
        pose_scores.append(pose_score)

    return {
        "pose_state_silhouette_score": float(np.mean(silhouette_scores)) if silhouette_scores else 0.0,
        "pose_state_endpoint_score": float(np.mean(endpoint_scores)) if endpoint_scores else 0.0,
        "pose_state_limb_angle_score": float(np.mean(angle_scores)) if angle_scores else 0.0,
        "pose_state_score": float(np.mean(pose_scores)) if pose_scores else 0.0,
    }


def write_eval_bundle(
    *,
    output_dir: Path,
    reference_frame_paths: list[Path],
    preview_frame_paths: list[Path],
    replay_frame_paths: list[Path],
    pose_state_rows: list[dict[str, Path | str]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_sheet = _stack_horizontal(
        reference_frame_paths,
        output_dir / "reference_contact_sheet.png",
        "Hidden clean reference video sampled frames",
    )
    preview_sheet = _stack_horizontal(
        preview_frame_paths,
        output_dir / "preview_contact_sheet.png",
        "Agent submitted preview.mp4 sampled frames",
    )
    replay_sheet = _stack_horizontal(
        replay_frame_paths,
        output_dir / "replay_contact_sheet.png",
        "Evaluator replay renders from final.blend",
    )

    pose_rows = []
    for row in pose_state_rows:
        ref_path = Path(str(row["reference_pose"]))
        gt_path = Path(str(row["gt_pose"]))
        cand_path = Path(str(row["candidate_pose"]))
        name = str(row["name"])
        ref_image = _composite_on_white(ref_path)
        gt_image = _composite_on_white(gt_path)
        cand_image = _composite_on_white(cand_path)
        panel = Image.new(
            "RGBA",
            (ref_image.width * 3, ref_image.height + 36),
            (255, 255, 255, 255),
        )
        draw = ImageDraw.Draw(panel)
        draw.text(
            (12, 10),
            f"{name} | left=reference video | center=hidden GT pose | right=candidate pose",
            fill=(20, 20, 20, 255),
        )
        panel.paste(ref_image, (0, 36))
        panel.paste(gt_image, (ref_image.width, 36))
        panel.paste(cand_image, (ref_image.width * 2, 36))
        out_path = output_dir / f"{name}_pose_row.png"
        panel.save(out_path)
        pose_rows.append(out_path)
    pose_state_sheet = _stack_vertical(pose_rows, output_dir / "pose_state_sheet.png")

    return {
        "reference_sheet": reference_sheet,
        "preview_sheet": preview_sheet,
        "replay_sheet": replay_sheet,
        "pose_state_sheet": pose_state_sheet,
    }


def compute_local_metrics(
    *,
    reference_frame_paths: list[Path],
    preview_frame_paths: list[Path],
    replay_frame_paths: list[Path],
    skeleton_package: dict[str, Any],
    package_root: Path,
    validity_payload: dict[str, Any],
) -> dict[str, Any]:
    validity_gate_passed = bool(validity_payload.get("validity_gate_passed", False))
    gate_fail_reasons = list(validity_payload.get("gate_fail_reasons", []))
    if not validity_gate_passed:
        return {
            "validity_gate_passed": False,
            "gate_fail_reasons": gate_fail_reasons,
            "video_match_score": 0.0,
            "replay_consistency_score": 0.0,
            "joint_range_score": 0.0,
            "pose_state_score": 0.0,
            "minimal_skeleton_score": 0.0,
        }

    if not preview_frame_paths or len(preview_frame_paths) != len(reference_frame_paths):
        gate_fail_reasons.append("preview_frames_missing_or_misaligned")
        validity_gate_passed = False
    if not replay_frame_paths or len(replay_frame_paths) != len(preview_frame_paths):
        gate_fail_reasons.append("replay_frames_missing_or_misaligned")
        validity_gate_passed = False
    if not validity_gate_passed:
        return {
            "validity_gate_passed": False,
            "gate_fail_reasons": gate_fail_reasons,
            "video_match_score": 0.0,
            "replay_consistency_score": 0.0,
            "joint_range_score": 0.0,
            "pose_state_score": 0.0,
            "minimal_skeleton_score": 0.0,
        }

    preview_health = _preview_health_metrics(preview_frame_paths)
    if float(preview_health["foreground_coverage_mean"]) < PREVIEW_MIN_FOREGROUND_COVERAGE:
        gate_fail_reasons.append("preview_missing_character")
        return {
            "validity_gate_passed": False,
            "gate_fail_reasons": gate_fail_reasons,
            "preview_foreground_coverage_mean": float(preview_health["foreground_coverage_mean"]),
            "video_match_score": 0.0,
            "replay_consistency_score": 0.0,
            "joint_range_score": 0.0,
            "pose_state_score": 0.0,
            "minimal_skeleton_score": 0.0,
        }

    video_metrics = _compute_video_comparison(reference_frame_paths, preview_frame_paths)
    replay_metrics = _compute_replay_comparison(preview_frame_paths, replay_frame_paths)
    joint_metrics = _score_joint_ranges(skeleton_package, validity_payload.get("joint_ranges", {}))
    pose_metrics = _score_pose_states(
        skeleton_package=skeleton_package,
        candidate_pose_states=list(validity_payload.get("pose_states", [])),
        package_root=package_root,
    )
    minimal_skeleton_score = 0.5 * joint_metrics["joint_range_score"] + 0.5 * pose_metrics["pose_state_score"]

    return {
        "validity_gate_passed": True,
        "gate_fail_reasons": gate_fail_reasons,
        "preview_foreground_coverage_mean": float(preview_health["foreground_coverage_mean"]),
        "video_match_score": float(video_metrics["score"]),
        "video_silhouette_iou_mean": float(video_metrics["silhouette_iou_mean"]),
        "video_masked_character_similarity": float(video_metrics["masked_character_similarity"]),
        "video_temporal_delta_score": float(video_metrics["temporal_delta_score"]),
        "replay_consistency_score": float(replay_metrics["score"]),
        "replay_silhouette_iou_mean": float(replay_metrics["silhouette_iou_mean"]),
        "replay_masked_character_similarity": float(replay_metrics["masked_character_similarity"]),
        "replay_temporal_delta_score": float(replay_metrics["temporal_delta_score"]),
        "replay_bbox_alignment_score": float(replay_metrics["bbox_alignment_score"]),
        "joint_range_score": float(joint_metrics["joint_range_score"]),
        "pose_state_score": float(pose_metrics["pose_state_score"]),
        "pose_state_silhouette_score": float(pose_metrics["pose_state_silhouette_score"]),
        "pose_state_endpoint_score": float(pose_metrics["pose_state_endpoint_score"]),
        "pose_state_limb_angle_score": float(pose_metrics["pose_state_limb_angle_score"]),
        "minimal_skeleton_score": float(minimal_skeleton_score),
    }


def compute_final_score(
    *,
    validity_gate: bool,
    video_match_score: float,
    replay_consistency_score: float,
    minimal_skeleton_score: float,
    vlm_score: float,
) -> float:
    if not validity_gate:
        return 0.0
    final_score = (
        0.35 * video_match_score
        + 0.20 * replay_consistency_score
        + 0.30 * minimal_skeleton_score
        + 0.15 * vlm_score
    )
    return max(0.0, min(1.0, float(final_score)))
