"""Scorer for the Northwind high-value customers task."""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass
from typing import Any


EXPECTED_COLUMNS = [
    "employee_full_name",
    "customer_id",
    "company_name",
    "total_adjusted_revenue",
    "avg_shipping_delay_days",
    "first_order_date_2022",
]


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
        rows.append(
            {
                key: value.strip()
                for key, value in zip(fieldnames, row, strict=True)
            }
        )
    return fieldnames, rows


def score_output_csv(*, output_csv: str, reference_csv: str) -> ScoreResult:
    """Return 1.0 only for the exact expected final CSV."""

    try:
        observed_columns, observed_rows = _read_csv(output_csv, "final_result.csv")
        reference_columns, reference_rows = _read_csv(reference_csv, "reference_result.csv")
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

    observed_keys = [row["employee_full_name"] for row in observed_rows]
    if len(observed_keys) != len(set(observed_keys)):
        return _hard_fail("duplicate_employee_rows")

    if len(observed_rows) != len(reference_rows):
        return _hard_fail(
            "wrong_row_count",
            {"observed": len(observed_rows), "expected": len(reference_rows)},
        )

    mismatches = []
    for index, (observed, reference) in enumerate(zip(observed_rows, reference_rows), start=1):
        if observed != reference:
            mismatches.append(
                {
                    "row_number": index,
                    "observed": observed,
                    "expected": reference,
                }
            )
            break

    if mismatches:
        return _hard_fail("row_mismatch", {"first_mismatch": mismatches[0]})

    return ScoreResult(
        1.0,
        True,
        "passed",
        None,
        {"row_count": len(observed_rows), "columns": observed_columns},
    )
