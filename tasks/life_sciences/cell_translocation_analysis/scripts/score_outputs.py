"""Scorer for the cell translocation analysis task."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


ANSWER_DOSE_TOLERANCE = 5.0
ANSWER_PERCENTAGE_TOLERANCE = 0.05
REFERENCE_ROW_TOLERANCE = 0.25

CSV_NAMES = ("Cells.csv", "Cytoplasm.csv", "Nuclei.csv")


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
    return ScoreResult(0.0, False, reason, reason, details or {})


def _parse_json(text: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _finite_number(payload: dict[str, Any], key: str, label: str) -> float:
    if key not in payload:
        raise ValueError(f"{label} is missing key {key!r}")
    try:
        value = float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{key} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}.{key} must be finite")
    return value


def _read_csv(text: str, label: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if reader.fieldnames is None:
        raise ValueError(f"{label} has no header row")
    fieldnames = [name.strip() for name in reader.fieldnames]
    rows = list(reader)
    return fieldnames, rows


def _column_family_score(columns: list[str], required_prefixes: tuple[str, ...]) -> float:
    if not required_prefixes:
        return 1.0
    hits = 0
    for prefix in required_prefixes:
        if any(column.startswith(prefix) for column in columns):
            hits += 1
    return hits / len(required_prefixes)


def _score_measurement_csv(
    *,
    label: str,
    observed_text: str,
    reference_text: str,
    required_prefixes: tuple[str, ...],
) -> tuple[float, dict[str, Any]]:
    observed_columns, observed_rows = _read_csv(observed_text, label)
    reference_columns, reference_rows = _read_csv(reference_text, f"reference {label}")

    required_columns = {"ImageNumber", "ObjectNumber"}
    missing_required = sorted(required_columns - set(observed_columns))
    if missing_required:
        return 0.0, {"missing_required_columns": missing_required}

    if not observed_rows:
        return 0.0, {"row_count": 0}

    reference_count = len(reference_rows)
    observed_count = len(observed_rows)
    row_low = max(1, int(reference_count * (1.0 - REFERENCE_ROW_TOLERANCE)))
    row_high = int(reference_count * (1.0 + REFERENCE_ROW_TOLERANCE))
    row_score = 1.0 if row_low <= observed_count <= row_high else 0.0

    feature_score = _column_family_score(observed_columns, required_prefixes)

    try:
        image_numbers = {
            int(float(row["ImageNumber"]))
            for row in observed_rows
            if row.get("ImageNumber") not in (None, "")
        }
        object_numbers = [
            int(float(row["ObjectNumber"]))
            for row in observed_rows[: min(1000, len(observed_rows))]
            if row.get("ObjectNumber") not in (None, "")
        ]
    except ValueError:
        return 0.0, {"numeric_id_error": True}

    image_score = 1.0 if len(image_numbers) >= 20 and min(image_numbers) >= 1 else 0.0
    object_score = 1.0 if object_numbers and max(object_numbers) >= 10 else 0.0

    score = 0.35 * row_score + 0.35 * feature_score + 0.15 * image_score + 0.15 * object_score
    details = {
        "row_count": observed_count,
        "reference_row_count": reference_count,
        "column_count": len(observed_columns),
        "reference_column_count": len(reference_columns),
        "row_score": row_score,
        "feature_score": feature_score,
        "image_score": image_score,
        "object_score": object_score,
    }
    return round(score, 6), details


def score_output_bundle(
    *,
    answer_json: str,
    reference_answer_json: str,
    output_csvs: dict[str, str],
    reference_csvs: dict[str, str],
) -> ScoreResult:
    """Return a scalar score for the answer and required measurement tables."""

    try:
        answer = _parse_json(answer_json, "answer.json")
        reference_answer = _parse_json(reference_answer_json, "reference answer.json")
        dose = _finite_number(answer, "minimum_dose", "answer.json")
        percentage = _finite_number(answer, "positive_percentage", "answer.json")
        reference_dose = _finite_number(reference_answer, "minimum_dose", "reference answer.json")
        reference_percentage = _finite_number(
            reference_answer, "positive_percentage", "reference answer.json"
        )
    except ValueError as exc:
        return _hard_fail(str(exc))

    dose_abs_error = abs(dose - reference_dose)
    percentage_abs_error = abs(percentage - reference_percentage)
    if dose_abs_error > ANSWER_DOSE_TOLERANCE:
        return _hard_fail(
            "minimum_dose_outside_tolerance",
            {"observed": dose, "reference": reference_dose, "abs_error": dose_abs_error},
        )
    if percentage_abs_error > ANSWER_PERCENTAGE_TOLERANCE:
        return _hard_fail(
            "positive_percentage_outside_tolerance",
            {
                "observed": percentage,
                "reference": reference_percentage,
                "abs_error": percentage_abs_error,
            },
        )

    missing = [name for name in CSV_NAMES if name not in output_csvs]
    if missing:
        return _hard_fail("missing_measurement_csvs", {"missing": missing})

    family_requirements = {
        "Cells.csv": ("Location_",),
        "Cytoplasm.csv": ("Correlation_", "Intensity_", "Location_", "Math_"),
        "Nuclei.csv": ("Correlation_", "Intensity_", "Location_", "Math_"),
    }

    table_scores: dict[str, float] = {}
    table_details: dict[str, Any] = {}
    for name in CSV_NAMES:
        try:
            table_score, details = _score_measurement_csv(
                label=name,
                observed_text=output_csvs[name],
                reference_text=reference_csvs[name],
                required_prefixes=family_requirements[name],
            )
        except ValueError as exc:
            return _hard_fail(f"{name}_parse_error: {exc}")
        table_scores[name] = table_score
        table_details[name] = details

    answer_score = 1.0
    measurement_score = sum(table_scores.values()) / len(table_scores)
    score = round(0.60 * answer_score + 0.40 * measurement_score, 6)
    passed = score >= 0.85
    reason = "passed" if passed else "score_below_threshold"
    return ScoreResult(
        score,
        passed,
        reason,
        None,
        {
            "dose_abs_error": dose_abs_error,
            "percentage_abs_error": percentage_abs_error,
            "table_scores": table_scores,
            "table_details": table_details,
        },
    )
