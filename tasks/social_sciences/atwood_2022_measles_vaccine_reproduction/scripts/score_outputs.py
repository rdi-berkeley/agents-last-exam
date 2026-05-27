#!/usr/bin/env python
"""Score Atwood 2022 Table 2 reproduction outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from pathlib import Path
from typing import Any


TASK_ID = "atwood_2022_table2_vaccination_effect"
REQUIRED_FILES = (
    "execution_log.txt",
    "repair_log.json",
    "paper_coefficients.json",
    "code_coefficients.json",
    "coefficient_matches.csv",
    "verdict.json",
)
MATCH_COLUMNS = (
    "outcome",
    "paper_value",
    "code_value",
    "decimals",
    "rounded_code_value",
    "is_match",
    "source_file",
    "source_line",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Directory containing submitted outputs.")
    parser.add_argument(
        "--reference",
        required=True,
        help="Directory containing hidden gold files, usually reference/hidden_gold.",
    )
    return parser.parse_args()


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON in {path.name}: {exc}")
        return None


def to_decimal(value: Any) -> Decimal:
    if isinstance(value, str):
        normalized = value.replace(",", "").strip()
    else:
        normalized = str(value)
    return Decimal(normalized)


def round_half_up(value: Any, decimals: int) -> Decimal:
    quant = Decimal("1").scaleb(-decimals)
    return to_decimal(value).quantize(quant, rounding=ROUND_HALF_UP)


def verdict_from_rate(rate: float) -> str:
    if rate == 1.0:
        return "fully_reproducible"
    if rate > 0.8:
        return "largely_reproducible"
    if rate >= 0.5:
        return "partially_reproducible"
    return "not_reproducible"


def coefficient_map(payload: Any, name: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        errors.append(f"{name} must be a JSON object")
        return {}
    if payload.get("task_id") != TASK_ID:
        errors.append(f"{name} has wrong task_id")
    coefficients = payload.get("coefficients")
    if not isinstance(coefficients, list):
        errors.append(f"{name} must contain a coefficients list")
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(coefficients):
        if not isinstance(item, dict):
            errors.append(f"{name} coefficient {idx} is not an object")
            continue
        outcome = item.get("outcome")
        if not isinstance(outcome, str) or not outcome:
            errors.append(f"{name} coefficient {idx} missing outcome")
            continue
        if outcome in mapped:
            errors.append(f"{name} has duplicate outcome {outcome!r}")
        mapped[outcome] = item
    return mapped


def validate_support_files(output_dir: Path, expected_outcomes: list[str], errors: list[str]) -> None:
    execution_log = output_dir / "execution_log.txt"
    if not execution_log.read_text(encoding="utf-8", errors="replace").strip():
        errors.append("execution_log.txt is empty")

    repair_log = load_json(output_dir / "repair_log.json", errors)
    if isinstance(repair_log, dict):
        if repair_log.get("task_id") != TASK_ID:
            errors.append("repair_log.json has wrong task_id")
        if not isinstance(repair_log.get("repairs"), list):
            errors.append("repair_log.json must contain a repairs list")

    match_csv = output_dir / "coefficient_matches.csv"
    try:
        with match_csv.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != list(MATCH_COLUMNS):
                errors.append("coefficient_matches.csv has wrong header")
                return
            rows = list(reader)
    except Exception as exc:
        errors.append(f"invalid coefficient_matches.csv: {exc}")
        return

    if len(rows) != len(expected_outcomes):
        errors.append("coefficient_matches.csv must contain one row per scored outcome")
    row_outcomes = [row.get("outcome", "") for row in rows]
    if set(row_outcomes) != set(expected_outcomes):
        errors.append("coefficient_matches.csv outcome set does not match scored outcomes")
    for row in rows:
        if row.get("is_match", "").strip().lower() not in {"true", "false"}:
            errors.append(f"coefficient_matches.csv invalid is_match for {row.get('outcome')!r}")


def score(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (output_dir / name).is_file()]
    empty = [
        name
        for name in REQUIRED_FILES
        if (output_dir / name).is_file() and (output_dir / name).stat().st_size == 0
    ]
    if missing:
        errors.append("missing required files: " + ", ".join(missing))
    if empty:
        errors.append("empty required files: " + ", ".join(empty))

    gold = load_json(reference_dir / "gold_paper_coefficients.json", errors)
    gold_verdict = load_json(reference_dir / "gold_verdict.json", errors)
    sub_paper = load_json(output_dir / "paper_coefficients.json", errors) if not missing else None
    sub_code = load_json(output_dir / "code_coefficients.json", errors) if not missing else None
    sub_verdict = load_json(output_dir / "verdict.json", errors) if not missing else None

    gold_coeffs = coefficient_map(gold, "gold_paper_coefficients.json", errors)
    expected_outcomes = list(gold_coeffs)
    if not missing and expected_outcomes:
        validate_support_files(output_dir, expected_outcomes, errors)

    paper_coeffs = coefficient_map(sub_paper, "paper_coefficients.json", errors)
    code_coeffs = coefficient_map(sub_code, "code_coefficients.json", errors)

    details = []
    paper_correct = 0
    code_matches = 0

    for outcome, gold_item in gold_coeffs.items():
        paper_ok = False
        code_ok = False
        rounded_code: float | None = None

        paper_item = paper_coeffs.get(outcome)
        if paper_item is None:
            errors.append(f"paper_coefficients.json missing outcome {outcome!r}")
        else:
            try:
                paper_ok = to_decimal(paper_item["paper_estimate"]) == to_decimal(
                    gold_item["paper_estimate"]
                )
            except (KeyError, InvalidOperation):
                errors.append(f"paper_coefficients.json invalid estimate for {outcome!r}")

        code_item = code_coeffs.get(outcome)
        if code_item is None:
            errors.append(f"code_coefficients.json missing outcome {outcome!r}")
        else:
            try:
                rounded = round_half_up(
                    code_item["code_estimate"],
                    int(gold_item["paper_estimate_decimals"]),
                )
                rounded_code = float(rounded)
                code_ok = rounded == to_decimal(gold_item["paper_estimate"])
            except (KeyError, InvalidOperation, ValueError):
                errors.append(f"code_coefficients.json invalid estimate for {outcome!r}")

        paper_correct += int(paper_ok)
        code_matches += int(code_ok)
        details.append(
            {
                "outcome": outcome,
                "paper_correct": paper_ok,
                "code_match": code_ok,
                "rounded_code": rounded_code,
                "gold_paper_estimate": gold_item.get("paper_estimate"),
            }
        )

    total = len(gold_coeffs)
    match_rate = code_matches / total if total else 0.0
    expected_verdict = verdict_from_rate(match_rate)
    verdict_correct = False

    if not isinstance(sub_verdict, dict):
        errors.append("verdict.json must be a JSON object")
    else:
        if sub_verdict.get("task_id") != TASK_ID:
            errors.append("verdict.json has wrong task_id")
        try:
            verdict_correct = (
                int(sub_verdict.get("matched")) == code_matches
                and int(sub_verdict.get("total")) == total
                and math.isclose(float(sub_verdict.get("match_rate")), match_rate, abs_tol=1e-12)
                and sub_verdict.get("verdict") == expected_verdict
                and isinstance(gold_verdict, dict)
                and (
                    expected_verdict != "fully_reproducible"
                    or gold_verdict.get("verdict") == expected_verdict
                )
            )
        except (TypeError, ValueError):
            verdict_correct = False
        if not verdict_correct:
            errors.append("verdict.json does not match recomputed verdict")

    passed = paper_correct == total and code_matches == total and verdict_correct and not errors
    return {
        "score": 1.0 if passed else 0.0,
        "pass_fail": passed,
        "errors": errors,
        "paper_extraction_accuracy": paper_correct / total if total else 0.0,
        "code_match_accuracy": code_matches / total if total else 0.0,
        "verdict_correct": verdict_correct,
        "expected_verdict": expected_verdict,
        "expected_matched": code_matches,
        "expected_total": total,
        "expected_match_rate": match_rate,
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
