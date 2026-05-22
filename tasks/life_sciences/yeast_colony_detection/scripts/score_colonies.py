"""Scoring helpers for yeast_colony_detection."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

COUNT_MIN = 68
COUNT_MAX = 78
MMD_GAMMA = 1.0
MMD_ACCURACY_THRESHOLD = 0.9
EXCLUDED_NUMERIC_COLUMNS = {"ImageNumber", "ObjectNumber"}
REQUIRED_CENTROID_COLUMNS = ["Location_Center_X", "Location_Center_Y"]


@dataclass(frozen=True)
class ColonyScoreResult:
    score: float
    count_pass: bool
    mmd_pass: bool
    reported_count: int | None
    predicted_rows: int
    reference_rows: int
    mmd_distance: float | None
    mmd_accuracy: float | None
    columns_used: list[str]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_answer_json(text: str) -> tuple[int | None, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid answer.json: {exc}"
    if not isinstance(payload, dict):
        return None, "answer.json must be a JSON object"
    value = payload.get("colony_count")
    if isinstance(value, bool) or not isinstance(value, int):
        return None, "answer.json must contain integer colony_count"
    return value, None


def _read_csv_rows(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV has no header")
    return list(reader)


def _numeric_columns(rows: list[dict[str, str]]) -> set[str]:
    if not rows:
        return set()
    result: set[str] = set()
    for column in rows[0].keys():
        if column in EXCLUDED_NUMERIC_COLUMNS:
            continue
        ok = True
        saw_value = False
        for row in rows:
            raw = row.get(column, "")
            if raw == "":
                ok = False
                break
            try:
                value = float(raw)
            except ValueError:
                ok = False
                break
            if math.isfinite(value):
                saw_value = True
        if ok and saw_value:
            result.add(column)
    return result


def _matrix(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    data: list[list[float]] = []
    for row in rows:
        values: list[float] = []
        keep = True
        for column in columns:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                keep = False
                break
            if not math.isfinite(value):
                keep = False
                break
            values.append(value)
        if keep:
            data.append(values)
    if not data:
        return np.empty((0, len(columns)), dtype=float)
    return np.asarray(data, dtype=float)


def _normalize_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_data = np.vstack([x, y])
    mins = np.min(all_data, axis=0)
    maxs = np.max(all_data, axis=0)
    ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    x_norm = np.nan_to_num((x - mins) / ranges)
    y_norm = np.nan_to_num((y - mins) / ranges)
    return x_norm, y_norm


def _rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    squared = np.sum(diff * diff, axis=2)
    return np.exp(-gamma * squared)


def mmd_distance(x: np.ndarray, y: np.ndarray, gamma: float = MMD_GAMMA) -> float:
    m = x.shape[0]
    n = y.shape[0]
    if m == 0 or n == 0:
        raise ValueError("MMD requires non-empty matrices")
    xx = _rbf_kernel(x, x, gamma)
    yy = _rbf_kernel(y, y, gamma)
    xy = _rbf_kernel(x, y, gamma)
    mmd_squared = (np.sum(xx) / (m * m)) + (np.sum(yy) / (n * n)) - (2 * np.sum(xy) / (m * n))
    return float(np.sqrt(max(0.0, mmd_squared)))


def score_colony_outputs(answer_json: str, prediction_csv: str, reference_csv: str) -> ColonyScoreResult:
    reported_count, answer_error = parse_answer_json(answer_json)
    if answer_error:
        return ColonyScoreResult(0.0, False, False, None, 0, 0, None, None, [], answer_error)

    assert reported_count is not None
    count_pass = COUNT_MIN <= reported_count <= COUNT_MAX

    try:
        pred_rows = _read_csv_rows(prediction_csv)
        ref_rows = _read_csv_rows(reference_csv)
    except Exception as exc:
        return ColonyScoreResult(0.0, count_pass, False, reported_count, 0, 0, None, None, [], f"invalid CSV: {exc}")

    if len(pred_rows) != reported_count:
        return ColonyScoreResult(
            0.0,
            count_pass,
            False,
            reported_count,
            len(pred_rows),
            len(ref_rows),
            None,
            None,
            [],
            "measurement row count does not match reported colony_count",
        )

    pred_numeric = _numeric_columns(pred_rows)
    ref_numeric = _numeric_columns(ref_rows)
    missing_centroids = [
        column
        for column in REQUIRED_CENTROID_COLUMNS
        if column not in pred_numeric or column not in ref_numeric
    ]
    if missing_centroids:
        return ColonyScoreResult(
            0.0,
            count_pass,
            False,
            reported_count,
            len(pred_rows),
            len(ref_rows),
            None,
            None,
            [],
            "missing required numeric centroid columns: " + ", ".join(missing_centroids),
        )

    common_columns = sorted((pred_numeric & ref_numeric) - EXCLUDED_NUMERIC_COLUMNS)
    if not common_columns:
        return ColonyScoreResult(
            0.0,
            count_pass,
            False,
            reported_count,
            len(pred_rows),
            len(ref_rows),
            None,
            None,
            [],
            "no shared numeric measurement columns",
        )

    x = _matrix(pred_rows, common_columns)
    y = _matrix(ref_rows, common_columns)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return ColonyScoreResult(
            0.0,
            count_pass,
            False,
            reported_count,
            x.shape[0],
            y.shape[0],
            None,
            None,
            common_columns,
            "no valid numeric rows after dropping invalid values",
        )

    x_norm, y_norm = _normalize_pair(x, y)
    distance = mmd_distance(x_norm, y_norm, gamma=MMD_GAMMA)
    accuracy = float(np.exp(-distance))
    mmd_pass = accuracy >= MMD_ACCURACY_THRESHOLD
    score = 1.0 if count_pass and mmd_pass else 0.0
    reason = "pass" if score == 1.0 else "count and/or MMD check failed"
    return ColonyScoreResult(
        score,
        count_pass,
        mmd_pass,
        reported_count,
        len(pred_rows),
        len(ref_rows),
        distance,
        accuracy,
        common_columns,
        reason,
    )
