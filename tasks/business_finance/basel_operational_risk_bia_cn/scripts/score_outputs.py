#!/usr/bin/env python
"""Score Basel operational risk classification and BIA capital outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


REQUIRED_OUTPUT_FILES = ("classified_events.csv", "capital_calculation.json", "execution_log.txt")
CLASSIFIED_COLUMNS = ("Event_ID", "Loss_Event_Category", "Business_Line", "Loss_Amount_CNY")
CAPITAL_FIELDS = (
    "alpha_coefficient",
    "gross_income_years",
    "positive_gi_sum_cny",
    "number_of_years",
    "average_annual_gi_cny",
    "or_regulatory_capital_cny",
)
VALID_EL_CODES = {f"EL{i}" for i in range(1, 8)}
VALID_BL_CODES = {f"BL{i}" for i in range(1, 9)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Directory containing submitted outputs.")
    parser.add_argument("--reference", required=True, help="Directory containing gold reference files.")
    return parser.parse_args()


def read_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON in {path.name}: {exc}")
        return None


def read_csv(path: Path, errors: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != list(CLASSIFIED_COLUMNS):
                errors.append(
                    f"{path.name} must have header {','.join(CLASSIFIED_COLUMNS)}"
                )
                return []
            return list(reader)
    except Exception as exc:
        errors.append(f"invalid CSV in {path.name}: {exc}")
        return []


def normalize_code(raw: Any) -> str:
    return str(raw or "").strip().upper()


def to_float(raw: Any) -> float:
    if isinstance(raw, str):
        raw = raw.replace(",", "").strip()
    return float(raw)


def build_event_map(rows: list[dict[str, Any]], source: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows, start=2):
        event_id = str(row.get("Event_ID", "")).strip()
        if not event_id:
            errors.append(f"{source} row {idx} missing Event_ID")
            continue
        if event_id in mapped:
            errors.append(f"{source} has duplicate Event_ID {event_id}")
        category = normalize_code(row.get("Loss_Event_Category"))
        business_line = normalize_code(row.get("Business_Line"))
        if category and category not in VALID_EL_CODES:
            errors.append(f"{source} row {idx} invalid Loss_Event_Category {category!r}")
        if business_line and business_line not in VALID_BL_CODES:
            errors.append(f"{source} row {idx} invalid Business_Line {business_line!r}")
        try:
            loss_amount = int(round(to_float(row.get("Loss_Amount_CNY"))))
        except Exception:
            errors.append(f"{source} row {idx} invalid Loss_Amount_CNY")
            loss_amount = None
        mapped[event_id] = {
            "Loss_Event_Category": category,
            "Business_Line": business_line,
            "Loss_Amount_CNY": loss_amount,
        }
    return mapped


def compare_capital(submitted: Any, gold: Any, errors: list[str]) -> dict[str, Any]:
    if not isinstance(submitted, dict):
        errors.append("capital_calculation.json must be a JSON object")
        return {"capital_correct": False, "capital_error": None, "relative_error": None}
    if not isinstance(gold, dict):
        errors.append("gold_capital_calculation.json must be a JSON object")
        return {"capital_correct": False, "capital_error": None, "relative_error": None}

    for field in CAPITAL_FIELDS:
        if field not in submitted:
            errors.append(f"capital_calculation.json missing {field}")

    submitted_years = submitted.get("gross_income_years")
    if not isinstance(submitted_years, dict):
        errors.append("gross_income_years must be an object")
    else:
        missing_years = sorted(set(gold.get("gross_income_years", {})) - set(submitted_years))
        if missing_years:
            errors.append("gross_income_years missing years: " + ", ".join(missing_years))

    try:
        submitted_capital = to_float(submitted["or_regulatory_capital_cny"])
        gold_capital = to_float(gold["or_regulatory_capital_cny"])
    except Exception:
        errors.append("invalid or_regulatory_capital_cny")
        return {"capital_correct": False, "capital_error": None, "relative_error": None}

    abs_error = abs(submitted_capital - gold_capital)
    rel_error = abs_error / abs(gold_capital) if gold_capital else math.inf
    capital_correct = abs_error <= 100.0 or rel_error <= 0.001
    return {
        "capital_correct": capital_correct,
        "capital_error": abs_error,
        "relative_error": rel_error,
        "submitted_capital": submitted_capital,
        "gold_capital": gold_capital,
    }


def score(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (output_dir / name).is_file()]
    if missing:
        errors.append("missing required files: " + ", ".join(missing))
        return {
            "score": 0.0,
            "pass_fail": False,
            "errors": errors,
            "category_accuracy": 0.0,
            "business_line_accuracy": 0.0,
            "capital_correct": False,
        }

    if not (output_dir / "execution_log.txt").read_text(encoding="utf-8", errors="replace").strip():
        errors.append("execution_log.txt is empty")

    submitted_rows = read_csv(output_dir / "classified_events.csv", errors)
    gold_rows = read_csv(reference_dir / "gold_classified_events.csv", errors)
    submitted_capital = read_json(output_dir / "capital_calculation.json", errors)
    gold_capital = read_json(reference_dir / "gold_capital_calculation.json", errors)

    submitted_by_id = build_event_map(submitted_rows, "classified_events.csv", errors)
    gold_by_id = build_event_map(gold_rows, "gold_classified_events.csv", errors)
    expected_ids = set(gold_by_id)
    submitted_ids = set(submitted_by_id)
    if submitted_ids != expected_ids:
        missing_ids = sorted(expected_ids - submitted_ids)
        extra_ids = sorted(submitted_ids - expected_ids)
        if missing_ids:
            errors.append("classified_events.csv missing Event_IDs: " + ", ".join(missing_ids[:10]))
        if extra_ids:
            errors.append("classified_events.csv has unexpected Event_IDs: " + ", ".join(extra_ids[:10]))

    total = len(gold_by_id)
    category_correct = 0
    business_line_correct = 0
    loss_amount_correct = 0
    details = []
    for event_id in sorted(gold_by_id):
        submitted = submitted_by_id.get(event_id)
        gold = gold_by_id[event_id]
        if submitted is None:
            details.append({"Event_ID": event_id, "category_correct": False, "business_line_correct": False})
            continue
        cat_ok = submitted["Loss_Event_Category"] == gold["Loss_Event_Category"]
        bl_ok = submitted["Business_Line"] == gold["Business_Line"]
        loss_ok = submitted["Loss_Amount_CNY"] == gold["Loss_Amount_CNY"]
        category_correct += int(cat_ok)
        business_line_correct += int(bl_ok)
        loss_amount_correct += int(loss_ok)
        details.append(
            {
                "Event_ID": event_id,
                "category_correct": cat_ok,
                "business_line_correct": bl_ok,
                "loss_amount_correct": loss_ok,
            }
        )

    category_accuracy = category_correct / total if total else 0.0
    business_line_accuracy = business_line_correct / total if total else 0.0
    loss_amount_accuracy = loss_amount_correct / total if total else 0.0
    capital_report = compare_capital(submitted_capital, gold_capital, errors)
    passed = (
        not errors
        and total == 60
        and category_accuracy >= 0.9
        and business_line_accuracy >= 0.9
        and loss_amount_accuracy == 1.0
        and bool(capital_report["capital_correct"])
    )
    return {
        "score": 1.0 if passed else 0.0,
        "pass_fail": passed,
        "errors": errors,
        "total_events": total,
        "category_correct": category_correct,
        "business_line_correct": business_line_correct,
        "loss_amount_correct": loss_amount_correct,
        "category_accuracy": category_accuracy,
        "business_line_accuracy": business_line_accuracy,
        "loss_amount_accuracy": loss_amount_accuracy,
        **capital_report,
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
