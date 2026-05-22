"""Evaluator helpers for life_sciences/cell_tracking_instance_1."""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

FRAME_COUNT = 30
SEG_PASS_THRESHOLD = 0.5
TRA_PASS_THRESHOLD = 0.9


@dataclass(frozen=True)
class TrackRow:
    label: int
    begin: int
    end: int
    parent: int


class EvaluationError(ValueError):
    """Raised when the submitted output is invalid."""


def load_tiff_array(data: bytes, *, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    try:
        with Image.open(BytesIO(data)) as image:
            arr = np.array(image)
    except Exception as exc:
        raise EvaluationError(f"cannot decode TIFF: {exc}") from exc

    if arr.ndim != 2:
        raise EvaluationError(f"mask must be a 2D labeled image, got shape {arr.shape}")
    if expected_shape is not None and tuple(arr.shape) != expected_shape:
        raise EvaluationError(f"mask shape {arr.shape} does not match expected {expected_shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.isfinite(arr)) or not np.all(arr == np.floor(arr)):
            raise EvaluationError("mask contains non-integer labels")
    arr = arr.astype(np.int64, copy=False)
    if np.any(arr < 0):
        raise EvaluationError("mask contains negative labels")
    return arr


def parse_track_table(text: str) -> dict[int, TrackRow]:
    rows: dict[int, TrackRow] = {}
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        if len(parts) != 4:
            raise EvaluationError(f"res_track.txt line {lineno} must have 4 columns")
        try:
            label, begin, end, parent = (int(part) for part in parts)
        except ValueError as exc:
            raise EvaluationError(f"res_track.txt line {lineno} contains non-integers") from exc
        if label <= 0:
            raise EvaluationError(f"res_track.txt line {lineno} has nonpositive label")
        if label in rows:
            raise EvaluationError(f"duplicate track label {label}")
        if begin < 0 or end < begin or end >= FRAME_COUNT:
            raise EvaluationError(f"invalid begin/end for label {label}")
        if parent < 0:
            raise EvaluationError(f"invalid parent for label {label}")
        rows[label] = TrackRow(label=label, begin=begin, end=end, parent=parent)
    if not rows:
        raise EvaluationError("res_track.txt contains no tracks")
    missing_parent = sorted(
        row.parent for row in rows.values() if row.parent != 0 and row.parent not in rows
    )
    if missing_parent:
        raise EvaluationError(f"parent labels missing from res_track.txt: {missing_parent[:5]}")
    return rows


def _label_areas(mask: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(mask, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts) if int(label) > 0}


def _best_object_iou(ref_object: np.ndarray, pred_mask: np.ndarray, pred_areas: dict[int, int]) -> float:
    ref_area = int(np.count_nonzero(ref_object))
    if ref_area == 0:
        return 0.0
    labels, counts = np.unique(pred_mask[ref_object], return_counts=True)
    best = 0.0
    for label, intersection in zip(labels, counts):
        label_int = int(label)
        if label_int <= 0:
            continue
        union = ref_area + pred_areas.get(label_int, 0) - int(intersection)
        if union > 0:
            best = max(best, int(intersection) / union)
    return best


def compute_seg_score(pred_masks: dict[int, np.ndarray], ref_seg_masks: dict[int, np.ndarray]) -> float:
    scores: list[float] = []
    for frame, ref_mask in sorted(ref_seg_masks.items()):
        pred_mask = pred_masks[frame]
        pred_areas = _label_areas(pred_mask)
        for ref_label in np.unique(ref_mask):
            ref_label = int(ref_label)
            if ref_label <= 0:
                continue
            scores.append(_best_object_iou(ref_mask == ref_label, pred_mask, pred_areas))
    return float(np.mean(scores)) if scores else 0.0


def _build_ref_to_pred_mapping(
    pred_masks: dict[int, np.ndarray],
    ref_track_masks: dict[int, np.ndarray],
) -> dict[int, int]:
    intersections: dict[tuple[int, int], int] = {}
    for frame, ref_mask in sorted(ref_track_masks.items()):
        pred_mask = pred_masks[frame]
        for ref_label in np.unique(ref_mask):
            ref_label = int(ref_label)
            if ref_label <= 0:
                continue
            labels, counts = np.unique(pred_mask[ref_mask == ref_label], return_counts=True)
            for pred_label, count in zip(labels, counts):
                pred_label = int(pred_label)
                if pred_label <= 0:
                    continue
                key = (ref_label, pred_label)
                intersections[key] = intersections.get(key, 0) + int(count)

    mapping: dict[int, int] = {}
    by_ref: dict[int, list[tuple[int, int]]] = {}
    for (ref_label, pred_label), count in intersections.items():
        by_ref.setdefault(ref_label, []).append((count, pred_label))
    for ref_label, candidates in by_ref.items():
        count, pred_label = max(candidates)
        if count > 0:
            mapping[ref_label] = pred_label
    return mapping


def compute_tra_score(
    pred_masks: dict[int, np.ndarray],
    ref_track_masks: dict[int, np.ndarray],
    pred_tracks: dict[int, TrackRow],
    ref_tracks: dict[int, TrackRow],
) -> float:
    mapping = _build_ref_to_pred_mapping(pred_masks, ref_track_masks)
    if not mapping:
        return 0.0

    frame_scores: list[float] = []
    for frame, ref_mask in sorted(ref_track_masks.items()):
        pred_mask = pred_masks[frame]
        pred_areas = _label_areas(pred_mask)
        for ref_label in np.unique(ref_mask):
            ref_label = int(ref_label)
            if ref_label <= 0:
                continue
            pred_label = mapping.get(ref_label)
            if pred_label is None:
                frame_scores.append(0.0)
                continue
            ref_object = ref_mask == ref_label
            intersection = int(np.count_nonzero(ref_object & (pred_mask == pred_label)))
            union = int(np.count_nonzero(ref_object)) + pred_areas.get(pred_label, 0) - intersection
            # TRA is about temporal graph consistency; SEG handles exact outlines.
            # Count an object as tracked for a frame when the mapped predicted
            # label overlaps it at all, then let the lineage table check spans.
            frame_scores.append(float(intersection > 0))
    mask_consistency = float(np.mean(frame_scores)) if frame_scores else 0.0

    track_scores: list[float] = []
    for ref_label, ref_row in ref_tracks.items():
        pred_label = mapping.get(ref_label)
        if pred_label is None or pred_label not in pred_tracks:
            track_scores.append(0.0)
            continue
        pred_row = pred_tracks[pred_label]
        parent_expected = 0 if ref_row.parent == 0 else mapping.get(ref_row.parent, -1)
        track_scores.append(
            0.4 * float(pred_row.begin == ref_row.begin)
            + 0.4 * float(pred_row.end == ref_row.end)
            + 0.2 * float(pred_row.parent == parent_expected)
        )
    lineage = float(np.mean(track_scores)) if track_scores else 0.0
    return 0.85 * mask_consistency + 0.15 * lineage


def evaluate_tracking_submission(
    pred_masks: dict[int, np.ndarray],
    pred_track_text: str,
    ref_track_masks: dict[int, np.ndarray],
    ref_track_text: str,
    ref_seg_masks: dict[int, np.ndarray],
) -> dict[str, Any]:
    if sorted(pred_masks) != list(range(FRAME_COUNT)):
        raise EvaluationError("prediction must include mask000.tif through mask029.tif")
    if all(not np.any(mask > 0) for mask in pred_masks.values()):
        raise EvaluationError("all predicted masks are empty")

    pred_tracks = parse_track_table(pred_track_text)
    ref_tracks = parse_track_table(ref_track_text)
    mask_labels = {int(label) for mask in pred_masks.values() for label in np.unique(mask) if label > 0}
    missing_from_tracks = sorted(mask_labels - set(pred_tracks))
    if missing_from_tracks:
        raise EvaluationError(
            f"predicted mask labels missing from res_track.txt: {missing_from_tracks[:10]}"
        )

    seg = compute_seg_score(pred_masks, ref_seg_masks)
    tra = compute_tra_score(pred_masks, ref_track_masks, pred_tracks, ref_tracks)
    passes = seg >= SEG_PASS_THRESHOLD and tra >= TRA_PASS_THRESHOLD
    if passes:
        score = 1.0
    else:
        partial = 0.5 * min(seg / SEG_PASS_THRESHOLD, 1.0) + 0.5 * min(
            tra / TRA_PASS_THRESHOLD, 1.0
        )
        score = min(0.99, partial)
    return {
        "score": round(float(score), 6),
        "seg": round(float(seg), 6),
        "tra": round(float(tra), 6),
        "passes": bool(passes),
    }
