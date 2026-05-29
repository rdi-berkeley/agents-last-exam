#!/usr/bin/env python
"""Score a submitted Workday HCM reconstruction pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REQUIRED = {
    "workforce_snapshot.csv": [
        "as_of_date", "worker_id", "position_id", "company", "supervisory_org",
        "manager_id", "job_family", "worker_status", "leave_status", "fte",
        "annualized_base_pay", "costing_allocation_hash",
    ],
    "org_summary.csv": [
        "as_of_date", "company", "supervisory_org", "job_family", "headcount",
        "fte", "annualized_base_pay_total", "manager_count", "vacant_positions",
    ],
    "compensation_rollforward.csv": [
        "worker_id", "effective_date", "event_sequence", "reason", "row_status",
        "annualized_base_pay", "prior_annualized_base_pay", "delta_amount",
        "is_retroactive", "is_future_dated",
    ],
    "payroll_reconciliation.csv": [
        "worker_id", "pay_period_end", "expected_gross_pay", "actual_gross_pay",
        "variance_amount", "expected_fte", "actual_fte", "reason_code",
    ],
    "exception_workers.csv": ["worker_id", "as_of_date", "reason_code"],
}

KEYS = {
    "workforce_snapshot.csv": ["as_of_date", "worker_id"],
    "org_summary.csv": ["as_of_date", "company", "supervisory_org", "job_family"],
    "compensation_rollforward.csv": ["worker_id", "effective_date", "event_sequence"],
    "payroll_reconciliation.csv": ["worker_id", "pay_period_end"],
    "exception_workers.csv": ["worker_id", "as_of_date", "reason_code"],
}

TOLERANCES = {
    "fte": 0.001,
    "expected_fte": 0.001,
    "actual_fte": 0.001,
    "annualized_base_pay": 0.01,
    "annualized_base_pay_total": 0.05,
    "prior_annualized_base_pay": 0.01,
    "delta_amount": 0.01,
    "expected_gross_pay": 0.01,
    "actual_gross_pay": 0.01,
    "variance_amount": 0.01,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str], cols: list[str]) -> tuple[str, ...]:
    return tuple(row[col] for col in cols)


def compare_file(actual_path: Path, expected_path: Path, filename: str) -> dict[str, Any]:
    result: dict[str, Any] = {"points": 0.0, "issues": []}
    if not actual_path.exists():
        result["issues"].append(f"missing {filename}")
        return result
    actual = read_csv(actual_path)
    expected = read_csv(expected_path)
    actual_fields = actual[0].keys() if actual else REQUIRED[filename]
    if list(actual_fields) != REQUIRED[filename]:
        result["issues"].append(f"{filename} has wrong columns")
        return result
    keys = KEYS[filename]
    actual_map = {row_key(row, keys): row for row in actual}
    expected_map = {row_key(row, keys): row for row in expected}
    if len(actual_map) != len(actual):
        result["issues"].append(f"{filename} contains duplicate keys")
        return result
    key_credit = len(set(actual_map) & set(expected_map)) / max(1, len(expected_map))
    field_checks = 0
    field_ok = 0
    for key, expected_row in expected_map.items():
        actual_row = actual_map.get(key)
        if not actual_row:
            continue
        for col in REQUIRED[filename]:
            if col in keys:
                continue
            field_checks += 1
            if col in TOLERANCES:
                exp_raw = expected_row[col]
                act_raw = actual_row[col]
                if exp_raw == "" and act_raw == "":
                    field_ok += 1
                elif exp_raw and act_raw and abs(float(exp_raw) - float(act_raw)) <= TOLERANCES[col]:
                    field_ok += 1
            elif actual_row[col] == expected_row[col]:
                field_ok += 1
    field_credit = 1.0 if field_checks == 0 else field_ok / field_checks
    result["points"] = 0.45 * key_credit + 0.55 * field_credit
    if result["points"] < 1.0:
        result["issues"].append(f"{filename} key/field match {result['points']:.3f}")
    return result


def score_scenario(outputs: Path, reference: Path) -> tuple[float, list[str]]:
    weights = {
        "workforce_snapshot.csv": 25,
        "org_summary.csv": 15,
        "compensation_rollforward.csv": 15,
        "payroll_reconciliation.csv": 12,
        "exception_workers.csv": 8,
    }
    score = 0.0
    issues: list[str] = []
    schema_score = 0.0
    for filename, cols in REQUIRED.items():
        path = outputs / filename
        if path.exists():
            rows = read_csv(path)
            if rows and list(rows[0].keys()) == cols:
                schema_score += 15.0 / len(REQUIRED)
            else:
                issues.append(f"{filename} schema mismatch or empty")
        else:
            issues.append(f"{filename} missing")
    score += schema_score
    for filename, weight in weights.items():
        result = compare_file(outputs / filename, reference / filename, filename)
        score += weight * float(result["points"])
        issues.extend(result["issues"])
    note = outputs / "audit_note.md"
    if note.exists():
        text = note.read_text(encoding="utf-8").lower()
        required_terms = [
            "## event precedence",
            "## same-day tie-breaker",
            "## retroactive corrections",
            "## hidden schema compatibility",
            "## no manual edits",
        ]
        if len(text) >= 450:
            score += 10.0 * sum(1 for term in required_terms if term in text) / len(required_terms)
        else:
            issues.append("audit_note.md is too short for process validation")
    else:
        issues.append("audit_note.md missing")
    return score, issues


def run_submission(submission_dir: Path, input_dir: Path, output_dir: Path) -> None:
    script = submission_dir / "run_pipeline.py"
    if not script.exists():
        nested = submission_dir / "starter_project" / "run_pipeline.py"
        script = nested if nested.exists() else script
    if not script.exists():
        raise FileNotFoundError("run_pipeline.py not found in submission directory")
    subprocess.run(
        [sys.executable, str(script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
        cwd=str(script.parent),
        check=True,
        timeout=90,
    )


def score_submission(submission_dir: Path, reference_dir: Path) -> dict[str, Any]:
    scenarios = [
        ("visible", reference_dir / "evaluator_only" / "visible_input", reference_dir / "reference_outputs" / "visible", 70),
        ("hidden", reference_dir / "evaluator_only" / "hidden_input", reference_dir / "reference_outputs" / "hidden", 30),
    ]
    total = 0.0
    report: dict[str, Any] = {"scenario_scores": {}, "issues": []}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, input_dir, expected_dir, weight in scenarios:
            output_dir = tmp_path / f"{name}_outputs"
            try:
                run_submission(submission_dir, input_dir, output_dir)
                raw_score, issues = score_scenario(output_dir, expected_dir)
            except Exception as exc:
                raw_score = 0.0
                issues = [f"{name} scenario failed: {exc}"]
            total += raw_score * weight / 100.0
            report["scenario_scores"][name] = round(raw_score, 2)
            report["issues"].extend(issues[:8])
    report["points"] = round(total, 2)
    report["score"] = round(total / 100.0, 4)
    report["passed"] = total >= 80.0
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(score_submission(args.submission_dir.resolve(), args.reference_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
