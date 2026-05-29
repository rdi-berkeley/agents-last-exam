"""Scoring helpers for ham10000_skin_lesion_classification."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


CLASS_ORDER = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
EXPECTED_COLUMNS = ["image_id", "predicted_class"]


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hard_fail(reason: str, details: dict[str, Any] | None = None) -> ScoreResult:
    return ScoreResult(score=0.0, passed=False, reason=reason, hard_gate=reason, details=details or {})


def _read_csv(text: str, label: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")))
    try:
        raw_fieldnames = next(reader)
    except StopIteration as exc:
        raise ValueError(f"{label} has no header row") from exc
    fieldnames = list(raw_fieldnames)
    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
        if len(row) != len(fieldnames):
            raise ValueError(f"{label} row {row_number} has {len(row)} fields; expected {len(fieldnames)}")
        rows.append({key: value for key, value in zip(fieldnames, row)})
    return fieldnames, rows


def _parse_metric_json(text: str) -> float:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("balanced_accuracy.json is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("balanced_accuracy.json must be a JSON object")
    if set(payload.keys()) != {"balanced_accuracy"}:
        raise ValueError("balanced_accuracy.json must contain exactly one key: balanced_accuracy")
    value = payload["balanced_accuracy"]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("balanced_accuracy must be a finite number")
    return float(value)


def score_submission(
    *,
    predictions_csv: str,
    balanced_accuracy_json: str,
    reference_csv: str,
    pass_threshold: float,
) -> ScoreResult:
    try:
        reported_metric = _parse_metric_json(balanced_accuracy_json)
        observed_columns, observed_rows = _read_csv(predictions_csv, "predictions.csv")
        reference_columns, reference_rows = _read_csv(reference_csv, "test_gold.csv")
    except ValueError as exc:
        return _hard_fail(str(exc))

    if observed_columns != EXPECTED_COLUMNS:
        return _hard_fail("wrong_columns", {"observed": observed_columns, "expected": EXPECTED_COLUMNS})
    if reference_columns != EXPECTED_COLUMNS:
        return _hard_fail("reference_wrong_columns", {"reference": reference_columns, "expected": EXPECTED_COLUMNS})

    reference_map: dict[str, str] = {}
    reference_counts = {label: 0 for label in CLASS_ORDER}
    for row in reference_rows:
        image_id = row["image_id"]
        label = row["predicted_class"]
        if not image_id:
            return _hard_fail("reference_empty_image_id")
        if image_id in reference_map:
            return _hard_fail("reference_duplicate_image_id", {"image_id": image_id})
        if label not in CLASS_ORDER:
            return _hard_fail("reference_invalid_class", {"image_id": image_id, "predicted_class": label})
        reference_map[image_id] = label
        reference_counts[label] += 1

    if len(observed_rows) != len(reference_rows):
        return _hard_fail("wrong_row_count", {"observed": len(observed_rows), "expected": len(reference_rows)})

    observed_map: dict[str, str] = {}
    duplicates: list[str] = []
    for row in observed_rows:
        image_id = row.get("image_id", "")
        label = row.get("predicted_class", "")
        if not image_id:
            return _hard_fail("empty_image_id")
        if not label:
            return _hard_fail("empty_predicted_class", {"image_id": image_id})
        if label not in CLASS_ORDER:
            return _hard_fail("invalid_predicted_class", {"image_id": image_id, "predicted_class": label})
        if image_id in observed_map:
            duplicates.append(image_id)
            continue
        observed_map[image_id] = label

    if duplicates:
        return _hard_fail("duplicate_image_id", {"duplicates": sorted(set(duplicates))})

    unknown_ids = sorted(set(observed_map) - set(reference_map))
    missing_ids = sorted(set(reference_map) - set(observed_map))
    if unknown_ids:
        return _hard_fail("unknown_image_id", {"unknown_ids": unknown_ids[:10]})
    if missing_ids:
        return _hard_fail("missing_image_id", {"missing_ids": missing_ids[:10]})

    correct_counts = {label: 0 for label in CLASS_ORDER}
    for image_id, gold_label in reference_map.items():
        if observed_map[image_id] == gold_label:
            correct_counts[gold_label] += 1

    try:
        per_class_recall = {
            label: correct_counts[label] / reference_counts[label]
            for label in CLASS_ORDER
        }
    except ZeroDivisionError as exc:
        raise RuntimeError("reference data unexpectedly omitted one or more classes") from exc

    balanced_accuracy = sum(per_class_recall[label] for label in CLASS_ORDER) / len(CLASS_ORDER)
    if balanced_accuracy >= pass_threshold:
        score = 1.0
        passed = True
        reason = "passed"
    else:
        score = max(0.0, balanced_accuracy / pass_threshold)
        passed = False
        reason = "below_threshold"

    return ScoreResult(
        score=float(score),
        passed=passed,
        reason=reason,
        hard_gate=None,
        details={
            "balanced_accuracy": float(balanced_accuracy),
            "reported_balanced_accuracy": float(reported_metric),
            "reported_absolute_error": abs(float(reported_metric) - float(balanced_accuracy)),
            "pass_threshold": float(pass_threshold),
            "per_class_recall": per_class_recall,
            "row_count": len(reference_rows),
        },
    )
