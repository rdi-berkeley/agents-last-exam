"""Scoring helpers for echo_ef_segmentation_instance_1."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from typing import Any

import numpy as np
from skimage.draw import polygon

MEASUREMENTS_HEADERS = ["EF", "ESV_px", "EDV_px"]
SEGMENTATION_HEADERS = ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
INTEGER_LITERAL_RE = re.compile(r"^-?\d+$")


class SubmissionValidationError(ValueError):
    """Raised when the agent submission violates the public output contract."""


def _parse_csv_rows(
    text: str,
    *,
    csv_name: str,
    expected_headers: list[str],
) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    headers = reader.fieldnames or []
    if headers != expected_headers:
        raise SubmissionValidationError(
            f"{csv_name} headers must be {expected_headers}, got {headers}"
        )

    rows: list[dict[str, str]] = []
    for row_index, row in enumerate(reader, start=2):
        if None in row:
            raise SubmissionValidationError(
                f"{csv_name} row {row_index} has extra columns beyond {expected_headers}"
            )
        missing_fields = [header for header in expected_headers if row.get(header) is None]
        if missing_fields:
            raise SubmissionValidationError(
                f"{csv_name} row {row_index} is missing fields: {missing_fields}"
            )
        rows.append(row)
    return rows


def _parse_measurements(text: str) -> dict[str, float]:
    rows = _parse_csv_rows(
        text,
        csv_name="measurements.csv",
        expected_headers=MEASUREMENTS_HEADERS,
    )
    if len(rows) != 1:
        raise SubmissionValidationError(
            f"measurements.csv must contain exactly one row, got {len(rows)}"
        )
    row = rows[0]
    parsed: dict[str, float] = {}
    for key in MEASUREMENTS_HEADERS:
        try:
            parsed[key] = float(row[key])
        except (TypeError, ValueError) as exc:
            raise SubmissionValidationError(
                f"measurements.csv field {key} must be numeric, got {row[key]!r}"
            ) from exc
        if not math.isfinite(parsed[key]):
            raise SubmissionValidationError(
                f"measurements.csv field {key} must be finite, got {row[key]!r}"
            )
    return parsed


def _parse_exact_int(raw: str, *, field_name: str) -> int:
    stripped = raw.strip()
    if not INTEGER_LITERAL_RE.fullmatch(stripped):
        raise SubmissionValidationError(f"{field_name} must be an integer literal, got {raw!r}")
    return int(stripped)


def _parse_segmentation_rows(text: str) -> list[dict[str, Any]]:
    rows = _parse_csv_rows(
        text,
        csv_name="segmentation.csv",
        expected_headers=SEGMENTATION_HEADERS,
    )
    parsed: list[dict[str, Any]] = []
    for row in rows:
        try:
            x1 = float(row["X1"])
            y1 = float(row["Y1"])
            x2 = float(row["X2"])
            y2 = float(row["Y2"])
        except (TypeError, ValueError) as exc:
            raise SubmissionValidationError(
                f"segmentation.csv coordinate fields must be numeric, got row {row}"
            ) from exc
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise SubmissionValidationError(
                f"segmentation.csv coordinate fields must be finite, got row {row}"
            )
        parsed.append(
            {
                "FileName": row["FileName"],
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2,
                "Frame": _parse_exact_int(row["Frame"], field_name="segmentation.csv Frame"),
            }
        )
    return parsed


def _frame_rows(rows: list[dict[str, Any]], frame_id: int) -> list[dict[str, Any]]:
    return [row for row in rows if row["Frame"] == frame_id]


def _validate_predicted_frames(rows: list[dict[str, Any]], expected_frames: set[int]) -> None:
    observed_frames = {int(row["Frame"]) for row in rows}
    extra_frames = sorted(observed_frames - expected_frames)
    missing_frames = sorted(expected_frames - observed_frames)
    if extra_frames:
        raise SubmissionValidationError(
            f"segmentation.csv contains unexpected frame ids: {extra_frames}"
        )
    if missing_frames:
        raise SubmissionValidationError(
            f"segmentation.csv is missing required frame ids: {missing_frames}"
        )


def _validate_predicted_filenames(rows: list[dict[str, Any]], expected_filename: str) -> None:
    invalid = sorted({str(row["FileName"]) for row in rows if str(row["FileName"]) != expected_filename})
    if invalid:
        raise SubmissionValidationError(
            "segmentation.csv FileName values must all match "
            f"{expected_filename!r}, got {invalid}"
        )


def _contour_to_mask(rows: list[dict[str, Any]], frame_id: int, height: int, width: int) -> np.ndarray:
    frame_rows = _frame_rows(rows, frame_id)
    if len(frame_rows) < 5:
        raise SubmissionValidationError(
            f"frame {frame_id} has too few contour rows: {len(frame_rows)}"
        )
    left = np.array([[row["X1"], row["Y1"]] for row in frame_rows], dtype=float)
    right = np.array([[row["X2"], row["Y2"]] for row in frame_rows], dtype=float)[::-1]
    boundary = np.concatenate([left, right], axis=0)
    mask = np.zeros((height, width), dtype=np.uint8)
    rr, cc = polygon(boundary[:, 1], boundary[:, 0], shape=(height, width))
    mask[rr, cc] = 1
    if int(mask.sum()) == 0:
        raise SubmissionValidationError(f"frame {frame_id} produced an empty mask")
    return mask


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    denom = float(mask_a.sum() + mask_b.sum())
    if denom == 0.0:
        return 0.0
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    return float(2.0 * intersection / denom)


def score_submission(
    *,
    measurements_csv: str,
    segmentation_csv: str,
    reference_measurements_json: str,
    reference_segmentation_csv: str,
    variant_metadata_json: str,
) -> dict[str, Any]:
    measurements_truth = json.loads(reference_measurements_json)
    variant_metadata = json.loads(variant_metadata_json)
    try:
        reference_rows = _parse_segmentation_rows(reference_segmentation_csv)
    except SubmissionValidationError as exc:
        raise RuntimeError(f"invalid evaluator reference segmentation.csv: {exc}") from exc

    height = int(measurements_truth["frame_height"])
    width = int(measurements_truth["frame_width"])
    ed_frame = int(variant_metadata["ed_frame"])
    es_frame = int(variant_metadata["es_frame"])
    video_filename = str(variant_metadata["video_filename"])
    gt_ef = float(measurements_truth["EF"])
    try:
        measurements = _parse_measurements(measurements_csv)
        predicted_rows = _parse_segmentation_rows(segmentation_csv)
        _validate_predicted_filenames(predicted_rows, video_filename)
        _validate_predicted_frames(predicted_rows, {ed_frame, es_frame})
        pred_ed = _contour_to_mask(predicted_rows, ed_frame, height, width)
        pred_es = _contour_to_mask(predicted_rows, es_frame, height, width)
    except SubmissionValidationError as exc:
        return {
            "score": 0.0,
            "error": str(exc),
        }

    ref_ed = _contour_to_mask(reference_rows, ed_frame, height, width)
    ref_es = _contour_to_mask(reference_rows, es_frame, height, width)

    mean_dice = float((_dice(pred_ed, ref_ed) + _dice(pred_es, ref_es)) / 2.0)
    ef_error = abs(float(measurements["EF"]) - gt_ef)
    score = 1.0 if ef_error <= 8.0 and mean_dice >= 0.80 else 0.0
    return {
        "score": score,
        "ef_error": ef_error,
        "mean_dice": mean_dice,
        "ed_frame": ed_frame,
        "es_frame": es_frame,
    }
