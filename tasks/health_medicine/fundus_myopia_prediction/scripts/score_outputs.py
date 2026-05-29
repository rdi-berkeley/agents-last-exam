"""Scoring helpers for fundus_myopia_prediction."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPECTED_COLUMNS = ["image_id", "meta_pm_grade"]
VALID_GRADES = {0, 1, 2, 3, 4}
INTEGER_LITERAL_RE = re.compile(r"^-?\d+$")


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
    fieldnames = [name.strip() for name in raw_fieldnames]
    rows = []
    for row_number, row in enumerate(reader, start=2):
        if len(row) != len(fieldnames):
            raise ValueError(
                f"{label} row {row_number} has {len(row)} fields; expected {len(fieldnames)}"
            )
        rows.append({key: value.strip() for key, value in zip(fieldnames, row)})
    return fieldnames, rows


def _parse_grade(raw: str, *, field_name: str) -> int:
    if raw == "":
        raise ValueError(f"{field_name} must not be empty")
    if not INTEGER_LITERAL_RE.fullmatch(raw):
        raise ValueError(f"{field_name} must be an integer literal, got {raw!r}")
    value = int(raw)
    if value not in VALID_GRADES:
        raise ValueError(f"{field_name} must be one of {sorted(VALID_GRADES)}, got {raw!r}")
    return value


def score_output_csv(*, output_csv: str, reference_csv: str) -> ScoreResult:
    try:
        observed_columns, observed_rows = _read_csv(output_csv, "myopia_predictions.csv")
        reference_columns, reference_rows = _read_csv(
            reference_csv, "myopia_gold_standard.csv"
        )
    except ValueError as exc:
        return _hard_fail(str(exc))

    if observed_columns != EXPECTED_COLUMNS:
        return _hard_fail(
            "wrong_columns",
            {"observed": observed_columns, "expected": EXPECTED_COLUMNS},
        )
    if reference_columns != EXPECTED_COLUMNS:
        return _hard_fail(
            "reference_wrong_columns",
            {"reference": reference_columns, "expected": EXPECTED_COLUMNS},
        )

    try:
        reference_map: dict[str, int] = {}
        for row in reference_rows:
            image_id = row["image_id"]
            if image_id in reference_map:
                return _hard_fail("reference_duplicate_image_id", {"image_id": image_id})
            reference_map[image_id] = _parse_grade(
                row["meta_pm_grade"],
                field_name=f"reference meta_pm_grade for {image_id}",
            )
    except ValueError as exc:
        return _hard_fail("reference_invalid_label", {"error": str(exc)})

    if len(observed_rows) != len(reference_rows):
        return _hard_fail(
            "wrong_row_count",
            {"observed": len(observed_rows), "expected": len(reference_rows)},
        )

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
            observed_map[image_id] = _parse_grade(
                row["meta_pm_grade"],
                field_name=f"meta_pm_grade for {image_id}",
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
    num_classes = len(VALID_GRADES)
    chance = 1.0 / num_classes
    raw_accuracy = correct / len(reference_map)
    score = max(0.0, (raw_accuracy - chance) / (1.0 - chance))
    return ScoreResult(
        score=float(score),
        passed=correct == len(reference_map),
        reason="passed" if correct == len(reference_map) else "accuracy_below_perfect",
        hard_gate=None,
        details={
            "row_count": len(reference_map),
            "correct": correct,
            "raw_accuracy": float(raw_accuracy),
            "chance_baseline": float(chance),
            "score": float(score),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a myopia_predictions.csv file.")
    parser.add_argument("--output", required=True, help="Path to the candidate output CSV")
    parser.add_argument("--reference", required=True, help="Path to the hidden gold CSV")
    return parser.parse_args()


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


if __name__ == "__main__":
    args = _parse_args()
    result = score_output_csv(
        output_csv=_read_text(args.output),
        reference_csv=_read_text(args.reference),
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
