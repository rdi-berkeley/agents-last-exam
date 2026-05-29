"""Scoring helpers for fundus_amd_staging."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import asdict, dataclass
from typing import Any


EXPECTED_COLUMNS = ["image_id", "amd_stage"]
ASCII_INTEGER_LITERAL_RE = re.compile(r"^[0-9]+$")


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


def _read_csv(text: str, label: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")))
    try:
        raw_fieldnames = next(reader)
    except StopIteration:
        raise ValueError(f"{label} has no header row")
    fieldnames = list(raw_fieldnames)
    rows = []
    for row_number, row in enumerate(reader, start=2):
        if len(row) != len(fieldnames):
            raise ValueError(f"{label} row {row_number} has {len(row)} fields; expected {len(fieldnames)}")
        rows.append({key: value for key, value in zip(fieldnames, row)})
    return fieldnames, rows


def _parse_stage_label(raw: str, *, field_name: str) -> int:
    if raw == "":
        raise ValueError(f"{field_name} must not be empty")
    if not ASCII_INTEGER_LITERAL_RE.fullmatch(raw):
        raise ValueError(f"{field_name} must be an ASCII integer literal, got {raw!r}")
    if raw not in {"0", "1", "2"}:
        raise ValueError(f"{field_name} must be exactly '0', '1', or '2', got {raw!r}")
    return int(raw)


def score_output_csv(*, output_csv: str, reference_csv: str) -> ScoreResult:
    try:
        observed_columns, observed_rows = _read_csv(output_csv, "amd_staging_predictions.csv")
        reference_columns, reference_rows = _read_csv(reference_csv, "amd_staging_gold_standard.csv")
    except ValueError as exc:
        return _hard_fail(str(exc))

    if observed_columns != EXPECTED_COLUMNS:
        return _hard_fail("wrong_columns", {"observed": observed_columns, "expected": EXPECTED_COLUMNS})
    if reference_columns != EXPECTED_COLUMNS:
        return _hard_fail("reference_wrong_columns", {"reference": reference_columns, "expected": EXPECTED_COLUMNS})

    try:
        reference_map: dict[str, int] = {}
        for row in reference_rows:
            image_id = row["image_id"]
            if image_id in reference_map:
                return _hard_fail("reference_duplicate_image_id", {"image_id": image_id})
            reference_map[image_id] = _parse_stage_label(
                row["amd_stage"], field_name=f"reference amd_stage for {image_id}"
            )
    except ValueError as exc:
        return _hard_fail("reference_invalid_label", {"error": str(exc)})

    if len(observed_rows) != len(reference_rows):
        return _hard_fail("wrong_row_count", {"observed": len(observed_rows), "expected": len(reference_rows)})

    observed_map: dict[str, int] = {}
    duplicates: list[str] = []
    try:
        for row in observed_rows:
            image_id = row.get("image_id", "")
            if image_id == "":
                return _hard_fail("empty_image_id")
            if image_id in observed_map:
                duplicates.append(image_id)
                continue
            observed_map[image_id] = _parse_stage_label(
                row["amd_stage"], field_name=f"amd_stage for {image_id}"
            )
    except ValueError as exc:
        return _hard_fail("invalid_label", {"error": str(exc)})

    if duplicates:
        return _hard_fail("duplicate_image_id", {"duplicates": sorted(set(duplicates))})

    unknown_ids = sorted(set(observed_map) - set(reference_map))
    missing_ids = sorted(set(reference_map) - set(observed_map))
    if unknown_ids:
        return _hard_fail("unknown_image_id", {"unknown_ids": unknown_ids[:10]})
    if missing_ids:
        return _hard_fail("missing_image_id", {"missing_ids": missing_ids[:10]})

    correct = sum(int(observed_map[image_id] == reference_map[image_id]) for image_id in reference_map)
    score = correct / len(reference_map)
    return ScoreResult(
        score=float(score),
        passed=correct == len(reference_map),
        reason="passed" if correct == len(reference_map) else "accuracy_below_perfect",
        hard_gate=None,
        details={"row_count": len(reference_map), "correct": correct, "accuracy": float(score)},
    )
