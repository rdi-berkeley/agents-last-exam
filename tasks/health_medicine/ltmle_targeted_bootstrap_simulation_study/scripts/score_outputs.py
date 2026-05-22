"""Local structured scoring helpers for ltmle_targeted_bootstrap_simulation_study."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


def _parse_csv_text(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    return list(reader.fieldnames or []), rows


def _build_row_map(
    rows: list[dict[str, str]],
    row_keys: list[str],
) -> tuple[dict[tuple[str, ...], dict[str, str]], list[tuple[str, ...]]]:
    row_map: dict[tuple[str, ...], dict[str, str]] = {}
    duplicates: list[tuple[str, ...]] = []
    for row in rows:
        key = tuple(row[key_name] for key_name in row_keys)
        if key in row_map:
            duplicates.append(key)
        row_map[key] = row
    return row_map, duplicates


def compare_summary_csv(
    candidate_summary_csv: str,
    expected_summary_csv: str,
    contract_section: dict[str, Any],
    *,
    label: str,
) -> ScoreResult:
    expected_columns = list(contract_section["summary_columns"])
    row_keys = list(contract_section["row_match_keys"])
    tolerances = dict(contract_section["metric_tolerances"])

    candidate_fieldnames, candidate_rows = _parse_csv_text(candidate_summary_csv)
    expected_fieldnames, expected_rows = _parse_csv_text(expected_summary_csv)

    if candidate_fieldnames != expected_columns:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"{label}_schema_mismatch",
            details={
                "candidate_fieldnames": candidate_fieldnames,
                "expected_fieldnames": expected_columns,
            },
        )
    if expected_fieldnames != expected_columns:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"{label}_reference_schema_mismatch",
            details={
                "reference_fieldnames": expected_fieldnames,
                "expected_fieldnames": expected_columns,
            },
        )
    if not candidate_rows:
        return ScoreResult(score=0.0, passed=False, reason=f"{label}_candidate_empty")
    if not expected_rows:
        return ScoreResult(score=0.0, passed=False, reason=f"{label}_reference_empty")

    candidate_map, candidate_dupes = _build_row_map(candidate_rows, row_keys)
    expected_map, expected_dupes = _build_row_map(expected_rows, row_keys)
    if candidate_dupes:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"{label}_candidate_duplicate_keys",
            details={"duplicate_keys": [list(key) for key in candidate_dupes]},
        )
    if expected_dupes:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"{label}_reference_duplicate_keys",
            details={"duplicate_keys": [list(key) for key in expected_dupes]},
        )

    candidate_keys = set(candidate_map)
    expected_keys = set(expected_map)
    if candidate_keys != expected_keys:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"{label}_row_key_mismatch",
            details={
                "missing_keys": [list(key) for key in sorted(expected_keys - candidate_keys)],
                "unexpected_keys": [list(key) for key in sorted(candidate_keys - expected_keys)],
            },
        )

    mismatches: list[dict[str, Any]] = []
    for row_key in sorted(expected_keys):
        candidate_row = candidate_map[row_key]
        expected_row = expected_map[row_key]
        for metric_name, tolerance in tolerances.items():
            try:
                candidate_value = float(candidate_row[metric_name])
                expected_value = float(expected_row[metric_name])
            except ValueError:
                mismatches.append(
                    {
                        "row_key": list(row_key),
                        "metric": metric_name,
                        "candidate_value": candidate_row[metric_name],
                        "expected_value": expected_row[metric_name],
                        "type": "invalid_numeric",
                    }
                )
                continue
            if not math.isfinite(candidate_value) or not math.isfinite(expected_value):
                mismatches.append(
                    {
                        "row_key": list(row_key),
                        "metric": metric_name,
                        "candidate_value": candidate_value,
                        "expected_value": expected_value,
                        "type": "non_finite_numeric",
                    }
                )
                continue
            delta = abs(candidate_value - expected_value)
            if delta > float(tolerance):
                mismatches.append(
                    {
                        "row_key": list(row_key),
                        "metric": metric_name,
                        "candidate_value": candidate_value,
                        "expected_value": expected_value,
                        "delta": delta,
                        "tolerance": float(tolerance),
                    }
                )

    if mismatches:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason=f"{label}_metric_mismatch",
            details={"mismatches": mismatches[:20], "mismatch_count": len(mismatches)},
        )

    return ScoreResult(
        score=1.0,
        passed=True,
        reason=f"{label}_passed",
        details={"row_count": len(candidate_rows)},
    )
