#!/usr/bin/env python
"""Score Apple FY2024 balance-sheet reconstruction outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


REQUIRED_OUTPUT = "balance_sheet.json"
REFERENCE_FILE = "aapl_fy2024_balance_sheet_reference.json"
TOP_LEVEL_TOTAL_SUFFIXES = {
    "total_assets",
    "total_liabilities",
    "total_shareholders_equity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Directory containing submitted output.")
    parser.add_argument("--reference", required=True, help="Directory containing hidden reference JSON.")
    return parser.parse_args()


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON in {path}: {exc}")
        return None


def normalize_submission(payload: Any) -> Any:
    """Accept either the full reference shape or the submitter's simplified shape."""
    if not isinstance(payload, dict):
        return payload
    if "consolidated_balance_sheet" in payload:
        return payload
    keys = {"assets", "liabilities", "shareholders_equity"}
    if keys.issubset(payload):
        return {
            "metadata": {
                "company": payload.get("company", "Apple Inc."),
                "ticker": payload.get("ticker", "AAPL"),
                "period_of_report": payload.get("period", payload.get("period_of_report")),
                "currency": payload.get("currency"),
                "unit": payload.get("unit", "millions"),
            },
            "consolidated_balance_sheet": {
                "assets": payload.get("assets"),
                "liabilities": payload.get("liabilities"),
                "shareholders_equity": payload.get("shareholders_equity"),
                "total_liabilities_and_shareholders_equity": payload.get(
                    "total_liabilities_and_shareholders_equity"
                ),
            },
        }
    return payload


def numeric_leaves(payload: Any, prefix: str = "") -> dict[str, Decimal]:
    leaves: dict[str, Decimal] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else key
            leaves.update(numeric_leaves(value, child))
    elif isinstance(payload, (int, float, str)) and not isinstance(payload, bool):
        try:
            leaves[prefix] = Decimal(str(payload).replace(",", "").strip())
        except (InvalidOperation, AttributeError):
            pass
    return leaves


def metadata_checks(submission: Any, reference: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(submission, dict):
        return ["submitted JSON must be an object"]
    sub_meta = submission.get("metadata")
    ref_meta = reference.get("metadata", {}) if isinstance(reference, dict) else {}
    if not isinstance(sub_meta, dict):
        errors.append("metadata must be present and must be an object")
        return errors
    for key in ("company", "ticker", "period_of_report"):
        if sub_meta.get(key) != ref_meta.get(key):
            errors.append(f"metadata.{key} mismatch")
    currency = sub_meta.get("currency")
    unit = sub_meta.get("unit")
    if currency not in {"USD", "USD_millions"}:
        errors.append("metadata.currency must be USD or USD_millions")
    if unit not in {"millions", "USD_millions"}:
        errors.append("metadata.unit must be millions or USD_millions")
    return errors


def score(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    output_file = output_dir / REQUIRED_OUTPUT
    if not output_file.is_file():
        return {
            "score": 0.0,
            "pass_fail": False,
            "errors": [f"missing required output: {REQUIRED_OUTPUT}"],
            "correct_fields": 0,
            "total_fields": 0,
        }

    reference = load_json(reference_dir / REFERENCE_FILE, errors)
    submission = normalize_submission(load_json(output_file, errors))
    if errors:
        return {"score": 0.0, "pass_fail": False, "errors": errors, "correct_fields": 0, "total_fields": 0}

    errors.extend(metadata_checks(submission, reference))
    reference_bs = reference.get("consolidated_balance_sheet") if isinstance(reference, dict) else None
    submission_bs = submission.get("consolidated_balance_sheet") if isinstance(submission, dict) else None
    if not isinstance(reference_bs, dict) or not isinstance(submission_bs, dict):
        errors.append("consolidated_balance_sheet must be present and must be an object")
        return {"score": 0.0, "pass_fail": False, "errors": errors, "correct_fields": 0, "total_fields": 0}

    expected = numeric_leaves(reference_bs)
    observed = numeric_leaves(submission_bs)
    details = []
    correct = 0
    totals_correct = True

    for path, expected_value in expected.items():
        observed_value = observed.get(path)
        is_correct = observed_value == expected_value
        correct += int(is_correct)
        suffix = path.split(".")[-1]
        if suffix in TOP_LEVEL_TOTAL_SUFFIXES:
            totals_correct = totals_correct and is_correct
        details.append(
            {
                "path": path,
                "expected": str(expected_value),
                "observed": None if observed_value is None else str(observed_value),
                "correct": is_correct,
            }
        )

    missing_paths = sorted(set(expected) - set(observed))
    extra_paths = sorted(set(observed) - set(expected))
    total = len(expected)
    accuracy = correct / total if total else 0.0
    pass_fail = not errors and totals_correct and accuracy >= 0.95
    return {
        "score": float(accuracy) if not errors else 0.0,
        "pass_fail": pass_fail,
        "errors": errors,
        "correct_fields": correct,
        "total_fields": total,
        "field_accuracy": accuracy,
        "top_level_totals_correct": totals_correct,
        "missing_paths": missing_paths,
        "extra_paths": extra_paths,
        "details": details,
    }


def main() -> int:
    args = parse_args()
    report = score(Path(args.output), Path(args.reference))
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
