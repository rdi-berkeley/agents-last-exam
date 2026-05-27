"""Structured scorer for geospatial_cv_vehicle_detection_1 outputs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

MATCH_TOLERANCE_PX = 12.0


@dataclass
class ScoreResult:
    score: float
    raw_f1: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_predictions: int
    total_references: int
    summary_matches_files: bool
    format_valid: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "raw_f1": self.raw_f1,
            "precision": self.precision,
            "recall": self.recall,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_predictions": self.total_predictions,
            "total_references": self.total_references,
            "summary_matches_files": self.summary_matches_files,
            "format_valid": self.format_valid,
            "error": self.error,
        }


def _error(message: str) -> ScoreResult:
    return ScoreResult(
        score=0.0,
        raw_f1=0.0,
        precision=0.0,
        recall=0.0,
        true_positives=0,
        false_positives=0,
        false_negatives=0,
        total_predictions=0,
        total_references=0,
        summary_matches_files=False,
        format_valid=False,
        error=message,
    )


def _parse_detection_text(
    text: str,
    *,
    sequence_id: str,
    frame_index_range: tuple[int, int],
    valid_x_range: tuple[int, int],
    valid_y_range: tuple[int, int],
) -> dict[int, list[tuple[int, int]]]:
    by_frame: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"{sequence_id}:{line_no} must have exactly 3 comma-separated columns")
        try:
            frame_idx, x_pixel, y_pixel = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"{sequence_id}:{line_no} contains a non-integer value") from exc
        if not frame_index_range[0] <= frame_idx <= frame_index_range[1]:
            raise ValueError(
                f"{sequence_id}:{line_no} frame_index {frame_idx} outside {frame_index_range}"
            )
        if not valid_x_range[0] <= x_pixel <= valid_x_range[1]:
            raise ValueError(f"{sequence_id}:{line_no} x_pixel {x_pixel} outside {valid_x_range}")
        if not valid_y_range[0] <= y_pixel <= valid_y_range[1]:
            raise ValueError(f"{sequence_id}:{line_no} y_pixel {y_pixel} outside {valid_y_range}")
        by_frame[frame_idx].append((x_pixel, y_pixel))
    return {frame_idx: by_frame.get(frame_idx, []) for frame_idx in range(frame_index_range[0], frame_index_range[1] + 1)}


def _build_summary(
    *,
    task_id: str,
    variant: str,
    sequence_ids: list[str],
    frame_count_per_sequence: int,
    parsed_outputs: dict[str, dict[int, list[tuple[int, int]]]],
) -> dict[str, Any]:
    per_sequence: dict[str, dict[str, Any]] = {}
    total_points = 0
    nonempty_sequences = 0
    for sequence_id in sequence_ids:
        frame_points = parsed_outputs[sequence_id]
        frame_counts = {
            str(frame_idx): len(frame_points[frame_idx])
            for frame_idx in range(1, frame_count_per_sequence + 1)
            if frame_points[frame_idx]
        }
        sequence_total = sum(frame_counts.values())
        per_sequence[sequence_id] = {
            "total_points": sequence_total,
            "frames_with_points": len(frame_counts),
            "frame_counts": frame_counts,
        }
        total_points += sequence_total
        if sequence_total > 0:
            nonempty_sequences += 1

    return {
        "task_id": task_id,
        "variant": variant,
        "sequence_count": len(sequence_ids),
        "frame_count_per_sequence": frame_count_per_sequence,
        "total_points": total_points,
        "nonempty_sequence_count": nonempty_sequences,
        "empty_sequence_count": len(sequence_ids) - nonempty_sequences,
        "per_sequence": per_sequence,
    }


def _normalize_frame_counts(raw_frame_counts: Any) -> dict[str, int]:
    if not isinstance(raw_frame_counts, dict):
        raise ValueError("summary.json frame_counts entries must be JSON objects")
    normalized: dict[str, int] = {}
    for raw_key, raw_value in raw_frame_counts.items():
        try:
            frame_key = str(int(raw_key))
            count_value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("summary.json frame_counts must map integer-like keys to integers") from exc
        if count_value < 0:
            raise ValueError("summary.json frame_counts cannot contain negative counts")
        if count_value > 0:
            normalized[frame_key] = count_value
    return normalized


def _validate_summary(summary_text: str, computed_summary: dict[str, Any], sequence_ids: list[str]) -> bool:
    try:
        summary = json.loads(summary_text)
    except json.JSONDecodeError as exc:
        raise ValueError("summary.json is not valid JSON") from exc

    required_keys = {
        "task_id",
        "variant",
        "sequence_count",
        "frame_count_per_sequence",
        "total_points",
        "nonempty_sequence_count",
        "empty_sequence_count",
        "per_sequence",
    }
    missing_keys = sorted(required_keys - set(summary))
    if missing_keys:
        raise ValueError("summary.json missing required keys: " + ", ".join(missing_keys))
    if sorted(summary["per_sequence"]) != sorted(sequence_ids):
        raise ValueError("summary.json per_sequence keys do not match the required sequence ids")

    top_level_keys = [
        "task_id",
        "variant",
        "sequence_count",
        "frame_count_per_sequence",
        "total_points",
        "nonempty_sequence_count",
        "empty_sequence_count",
    ]
    for key in top_level_keys:
        if summary.get(key) != computed_summary[key]:
            raise ValueError(f"summary.json field {key!r} does not match the detection files")

    for sequence_id in sequence_ids:
        sequence_summary = summary["per_sequence"].get(sequence_id)
        if not isinstance(sequence_summary, dict):
            raise ValueError(f"summary.json per_sequence[{sequence_id!r}] must be an object")
        for key in ("total_points", "frames_with_points", "frame_counts"):
            if key not in sequence_summary:
                raise ValueError(f"summary.json per_sequence[{sequence_id!r}] missing key {key!r}")
        expected = computed_summary["per_sequence"][sequence_id]
        if sequence_summary["total_points"] != expected["total_points"]:
            raise ValueError(
                f"summary.json per_sequence[{sequence_id!r}].total_points does not match detections"
            )
        if sequence_summary["frames_with_points"] != expected["frames_with_points"]:
            raise ValueError(
                f"summary.json per_sequence[{sequence_id!r}].frames_with_points does not match detections"
            )
        normalized_counts = _normalize_frame_counts(sequence_summary["frame_counts"])
        if normalized_counts != expected["frame_counts"]:
            raise ValueError(
                f"summary.json per_sequence[{sequence_id!r}].frame_counts does not match detections"
            )
    return True


def _match_frame_points(
    predicted: list[tuple[int, int]],
    reference: list[tuple[int, int]],
    tolerance_px: float,
) -> tuple[int, int, int]:
    if not predicted and not reference:
        return 0, 0, 0

    cell_size = max(1.0, tolerance_px)
    tolerance_sq = tolerance_px * tolerance_px
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (x_pixel, y_pixel) in enumerate(reference):
        grid[(int(x_pixel // cell_size), int(y_pixel // cell_size))].append(index)

    adjacency: list[list[int]] = []
    for pred_x, pred_y in predicted:
        grid_x = int(pred_x // cell_size)
        grid_y = int(pred_y // cell_size)
        candidates: list[tuple[float, int]] = []
        for candidate_x in range(grid_x - 1, grid_x + 2):
            for candidate_y in range(grid_y - 1, grid_y + 2):
                for ref_index in grid.get((candidate_x, candidate_y), []):
                    ref_x, ref_y = reference[ref_index]
                    distance_sq = float((pred_x - ref_x) ** 2 + (pred_y - ref_y) ** 2)
                    if distance_sq <= tolerance_sq:
                        candidates.append((distance_sq, ref_index))
        candidates.sort()
        adjacency.append([ref_index for _, ref_index in candidates])

    matched_reference: list[int | None] = [None] * len(reference)

    def _augment(pred_index: int, seen: set[int]) -> bool:
        for ref_index in adjacency[pred_index]:
            if ref_index in seen:
                continue
            seen.add(ref_index)
            if matched_reference[ref_index] is None or _augment(matched_reference[ref_index], seen):
                matched_reference[ref_index] = pred_index
                return True
        return False

    true_positives = 0
    for pred_index in range(len(predicted)):
        if _augment(pred_index, set()):
            true_positives += 1

    false_positives = len(predicted) - true_positives
    false_negatives = len(reference) - true_positives
    return true_positives, false_positives, false_negatives


def score_output_bundle(
    *,
    manifest: dict[str, Any],
    output_texts: dict[str, str],
    summary_text: str,
    reference_texts: dict[str, str],
    tolerance_px: float = MATCH_TOLERANCE_PX,
) -> ScoreResult:
    try:
        sequence_ids = list(manifest["sequence_ids"])
        frame_count_per_sequence = int(manifest["frame_count_per_sequence"])
        frame_index_range = tuple(int(value) for value in manifest["frame_index_range"])
        coordinate_convention = manifest["coordinate_convention"]
        valid_x_range = tuple(int(value) for value in coordinate_convention["valid_x_range"])
        valid_y_range = tuple(int(value) for value in coordinate_convention["valid_y_range"])

        parsed_outputs = {
            sequence_id: _parse_detection_text(
                output_texts[sequence_id],
                sequence_id=sequence_id,
                frame_index_range=frame_index_range,
                valid_x_range=valid_x_range,
                valid_y_range=valid_y_range,
            )
            for sequence_id in sequence_ids
        }
        parsed_references = {
            sequence_id: _parse_detection_text(
                reference_texts[sequence_id],
                sequence_id=sequence_id,
                frame_index_range=frame_index_range,
                valid_x_range=valid_x_range,
                valid_y_range=valid_y_range,
            )
            for sequence_id in sequence_ids
        }

        computed_summary = _build_summary(
            task_id=str(manifest["task_id"]),
            variant=str(manifest["variant"]),
            sequence_ids=sequence_ids,
            frame_count_per_sequence=frame_count_per_sequence,
            parsed_outputs=parsed_outputs,
        )
        _validate_summary(summary_text, computed_summary, sequence_ids)

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for sequence_id in sequence_ids:
            for frame_idx in range(1, frame_count_per_sequence + 1):
                tp, fp, fn = _match_frame_points(
                    parsed_outputs[sequence_id][frame_idx],
                    parsed_references[sequence_id][frame_idx],
                    tolerance_px=tolerance_px,
                )
                true_positives += tp
                false_positives += fp
                false_negatives += fn

        total_predictions = true_positives + false_positives
        total_references = true_positives + false_negatives
        precision = true_positives / total_predictions if total_predictions else 0.0
        recall = true_positives / total_references if total_references else 0.0
        raw_f1 = (
            2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        )
        return ScoreResult(
            score=raw_f1,
            raw_f1=raw_f1,
            precision=precision,
            recall=recall,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            total_predictions=total_predictions,
            total_references=total_references,
            summary_matches_files=True,
            format_valid=True,
        )
    except Exception as exc:
        return _error(str(exc))
