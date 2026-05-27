"""Exact CSV scorer for the Northwind high-value customer task."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED_HEADER = [
    "employee_full_name",
    "customer_id",
    "company_name",
    "total_adjusted_revenue",
    "distinct_categories_purchased",
    "first_order_date_2022",
]

TWO_DECIMAL_RE = re.compile(r"^-?\d+\.\d{2}$")


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _parse_csv(text: str, *, label: str) -> tuple[list[str], list[list[str]]]:
    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"{label} is not valid CSV: {exc}") from exc

    if not rows:
        raise ValueError(f"{label} is empty")

    header = rows[0]
    data_rows = rows[1:]
    return header, data_rows


def score_output_csv(output_csv: str, reference_csv: str) -> ScoreResult:
    try:
        output_header, output_rows = _parse_csv(output_csv, label="output")
        reference_header, reference_rows = _parse_csv(reference_csv, label="reference")
    except ValueError as exc:
        return ScoreResult(0.0, False, "invalid_csv", {"error": str(exc)})

    if output_header != REQUIRED_HEADER:
        return ScoreResult(
            0.0,
            False,
            "wrong_header",
            {"observed_header": output_header, "required_header": REQUIRED_HEADER},
        )

    if reference_header != REQUIRED_HEADER:
        return ScoreResult(
            0.0,
            False,
            "invalid_reference_header",
            {"reference_header": reference_header, "required_header": REQUIRED_HEADER},
        )

    for row_index, row in enumerate(output_rows, start=2):
        if len(row) != len(REQUIRED_HEADER):
            return ScoreResult(
                0.0,
                False,
                "wrong_column_count",
                {"row_number": row_index, "row": row, "expected_columns": len(REQUIRED_HEADER)},
            )
        revenue = row[3]
        if not TWO_DECIMAL_RE.fullmatch(revenue):
            return ScoreResult(
                0.0,
                False,
                "bad_revenue_format",
                {"row_number": row_index, "observed": revenue, "expected": "two decimal places"},
            )

    if output_rows != reference_rows:
        return ScoreResult(
            0.0,
            False,
            "rows_mismatch",
            {
                "output_row_count": len(output_rows),
                "reference_row_count": len(reference_rows),
            },
        )

    return ScoreResult(
        1.0,
        True,
        "exact_match",
        {"row_count": len(output_rows), "header": REQUIRED_HEADER},
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Score final_result.csv against reference_result.csv.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    args = parser.parse_args()

    result = score_output_csv(
        output_csv=args.output.read_text(encoding="utf-8"),
        reference_csv=args.reference.read_text(encoding="utf-8"),
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())
