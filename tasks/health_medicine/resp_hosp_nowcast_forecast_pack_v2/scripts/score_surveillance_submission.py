from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_FILES = [
    "nowcast_quantiles.csv",
    "forecast_quantiles.csv",
    "trend_classification.csv",
    "delay_diagnostics.json",
    "run_manifest.json",
    "surveillance_brief.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_fieldnames(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return csv.DictReader(handle).fieldnames or []


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_cli(submission_dir: Path, input_dir: Path):
    out = Path(tempfile.mkdtemp(prefix="agenthle-resp-nowcast-"))
    result = subprocess.run(
        [
            sys.executable,
            str(submission_dir / "run_surveillance_nowcast.py"),
            "--input",
            str(input_dir),
            "--output",
            str(out),
        ],
        cwd=submission_dir,
        text=True,
        capture_output=True,
        timeout=60,
    )
    return out, result


def input_schema_violations(visible_input: Path, hidden_input: Path) -> list[str]:
    csv_names = [
        "facility_calendar.csv",
        "population.csv",
        "reporting_snapshot.csv",
        "schema_crosswalk.csv",
    ]
    required_names = csv_names + ["holiday_outage_notes.md", "modeling_contract.md"]
    violations = []
    for name in required_names:
        if not (visible_input / name).exists():
            violations.append(f"visible input missing {name}")
        if not (hidden_input / name).exists():
            violations.append(f"hidden input missing {name}")
    for name in csv_names:
        if (visible_input / name).exists() and (hidden_input / name).exists():
            visible_fields = csv_fieldnames(visible_input / name)
            hidden_fields = csv_fieldnames(hidden_input / name)
            if visible_fields != hidden_fields:
                violations.append(
                    f"{name} hidden columns {hidden_fields} differ from visible columns {visible_fields}"
                )
    return violations


def row_key(row: dict[str, str], columns: list[str]) -> tuple[str, ...]:
    return tuple(row[column] for column in columns)


def compare_numeric_csv(
    actual_path: Path,
    expected_path: Path,
    key_cols: list[str],
    value_tolerances: dict[str, object],
) -> tuple[float, list[dict[str, object]]]:
    if not actual_path.exists():
        return 0.0, [{"check": actual_path.name, "passed": False, "detail": "missing"}]
    if csv_fieldnames(actual_path) != csv_fieldnames(expected_path):
        return 0.0, [
            {"check": actual_path.name + "_schema", "passed": False, "detail": "schema mismatch"}
        ]
    actual = {row_key(row, key_cols): row for row in read_csv(actual_path)}
    expected = {row_key(row, key_cols): row for row in read_csv(expected_path)}
    if set(actual) != set(expected):
        return 0.0, [
            {"check": actual_path.name + "_keys", "passed": False, "detail": "key mismatch"}
        ]
    total = 0.0
    possible = 0.0
    for key, exp in expected.items():
        act = actual[key]
        for column, tol in value_tolerances.items():
            possible += 1.0
            allowed = tol(float(exp[column])) if callable(tol) else tol
            if abs(float(act[column]) - float(exp[column])) <= allowed:
                total += 1.0
    return total / possible if possible else 0.0, []


def compare_trends(actual_path: Path, expected_path: Path) -> tuple[float, list[dict[str, object]]]:
    if not actual_path.exists():
        return 0.0, [{"check": "trend_classification.csv", "passed": False, "detail": "missing"}]
    if csv_fieldnames(actual_path) != csv_fieldnames(expected_path):
        return 0.0, [{"check": "trend_schema", "passed": False, "detail": "schema mismatch"}]
    key_cols = ["region", "pathogen", "age_group", "analysis_week"]
    actual = {row_key(row, key_cols): row for row in read_csv(actual_path)}
    expected = {row_key(row, key_cols): row for row in read_csv(expected_path)}
    if set(actual) != set(expected):
        return 0.0, [{"check": "trend_keys", "passed": False, "detail": "key mismatch"}]
    total = 0.0
    possible = 0.0
    for key, exp in expected.items():
        act = actual[key]
        for column, tol in {
            "nowcast_rate_per_100k": 0.35,
            "forecast_rate_per_100k_h4": 0.45,
            "p_growth": 0.08,
        }.items():
            possible += 1.0
            total += 1.0 if abs(float(act[column]) - float(exp[column])) <= tol else 0.0
        for column in ["trend_category", "action_flag"]:
            possible += 1.0
            total += 1.0 if act[column] == exp[column] else 0.0
    return total / possible if possible else 0.0, []


def compare_diagnostics(actual_path: Path, expected_path: Path) -> tuple[float, list[dict[str, object]]]:
    if not actual_path.exists():
        return 0.0, [{"check": "delay_diagnostics.json", "passed": False, "detail": "missing"}]
    actual = load_json(actual_path)
    expected = load_json(expected_path)
    total = 0.0
    possible = 0.0
    for key, bins in expected["delay_pmfs"].items():
        for delay, exp in bins.items():
            possible += 1.0
            val = float(actual.get("delay_pmfs", {}).get(key, {}).get(delay, -99))
            total += 1.0 if abs(val - float(exp)) <= 0.08 else 0.0
    for key, bins in expected["backfill_factors"].items():
        for age, exp in bins.items():
            possible += 1.0
            val = float(actual.get("backfill_factors", {}).get(key, {}).get(age, -99))
            total += 1.0 if abs(val - float(exp)) <= 0.12 else 0.0
    possible += 1.0
    total += 1.0 if actual.get("excluded_outage_weeks") == expected.get("excluded_outage_weeks") else 0.0
    return total / possible if possible else 0.0, []


def compare_brief(actual_path: Path, expected_trends: Path) -> tuple[float, list[dict[str, object]]]:
    if not actual_path.exists():
        return 0.0, [{"check": "surveillance_brief.md", "passed": False, "detail": "missing"}]
    text = actual_path.read_text(encoding="utf-8").lower()
    total = 0.0
    possible = 0.0
    for row in read_csv(expected_trends):
        for token in [row["region"], row["pathogen"], row["trend_category"], row["action_flag"]]:
            possible += 1.0
            total += 1.0 if token.lower() in text else 0.0
    for phrase in ["reporting delay", "holiday outage", "facility completeness"]:
        possible += 1.0
        total += 1.0 if phrase in text else 0.0
    return total / possible if possible else 0.0, []


def score_output(actual_dir: Path, expected_dir: Path) -> tuple[float, list[dict[str, object]]]:
    missing = [name for name in REQUIRED_FILES if not (actual_dir / name).exists()]
    if missing:
        return 0.0, [{"check": "required_files", "passed": False, "detail": ", ".join(missing)}]

    score = 0.0
    report = []
    frac, checks = compare_numeric_csv(
        actual_dir / "nowcast_quantiles.csv",
        expected_dir / "nowcast_quantiles.csv",
        ["region", "pathogen", "age_group", "admission_week", "quantile"],
        {
            "hospitalizations": lambda exp: max(4.0, abs(exp) * 0.08 if exp else 4.0),
            "rate_per_100k": 0.35,
            "completeness_used": 0.08,
        },
    )
    score += frac * 28.0
    report.extend(checks)
    frac, checks = compare_numeric_csv(
        actual_dir / "forecast_quantiles.csv",
        expected_dir / "forecast_quantiles.csv",
        ["region", "pathogen", "age_group", "forecast_week", "horizon", "quantile"],
        {
            "hospitalizations": lambda exp: max(6.0, abs(exp) * 0.10 if exp else 6.0),
            "rate_per_100k": 0.45,
        },
    )
    score += frac * 24.0
    report.extend(checks)
    frac, checks = compare_trends(
        actual_dir / "trend_classification.csv", expected_dir / "trend_classification.csv"
    )
    score += frac * 20.0
    report.extend(checks)
    frac, checks = compare_diagnostics(
        actual_dir / "delay_diagnostics.json", expected_dir / "delay_diagnostics.json"
    )
    score += frac * 18.0
    report.extend(checks)
    frac, checks = compare_brief(actual_dir / "surveillance_brief.md", expected_dir / "trend_classification.csv")
    score += frac * 8.0
    report.extend(checks)
    try:
        manifest = load_json(actual_dir / "run_manifest.json")
        manifest_ok = (
            manifest.get("proposal_id") == "epidemiology-public-health-wf1-20260421"
            and "EpiNow2 1.8.0.9000" in manifest.get("professional_references", [])
        )
    except Exception:
        manifest_ok = False
    score += 2.0 if manifest_ok else 0.0
    report.append({"check": "run_manifest_contract", "passed": manifest_ok})
    return round(score, 3), report


def candidate_dir(output_dir: Path) -> Path:
    nested = output_dir / "submission"
    if (nested / "run_surveillance_nowcast.py").exists():
        return nested
    return output_dir


def main() -> int:
    args = parse_args()
    reference_dir = Path(args.reference_dir).resolve()
    submission_dir = candidate_dir(Path(args.submission_dir).resolve())
    visible_input = submission_dir / "input"
    hidden_input = reference_dir / "evaluator_only" / "hidden_input"
    report = {
        "score": 0.0,
        "raw_score": 0.0,
        "passed": False,
        "visible_score": 0.0,
        "hidden_score": 0.0,
        "notes": [],
    }

    if not (submission_dir / "run_surveillance_nowcast.py").exists():
        report["notes"].append("missing run_surveillance_nowcast.py")
        print(json.dumps(report, indent=2))
        return 0
    schema_errors = input_schema_violations(visible_input, hidden_input)
    if schema_errors:
        report["notes"].extend(schema_errors)
        print(json.dumps(report, indent=2))
        return 0

    try:
        visible_out, visible_result = run_cli(submission_dir, visible_input)
        if visible_result.returncode != 0:
            report["notes"].append("visible CLI failed: " + visible_result.stderr[-1000:])
            print(json.dumps(report, indent=2))
            return 0
        hidden_out, hidden_result = run_cli(submission_dir, hidden_input)
        if hidden_result.returncode != 0:
            report["notes"].append("hidden CLI failed: " + hidden_result.stderr[-1000:])
            print(json.dumps(report, indent=2))
            return 0
        visible_score, visible_checks = score_output(visible_out, reference_dir / "reference_outputs" / "visible")
        hidden_score, hidden_checks = score_output(hidden_out, reference_dir / "reference_outputs" / "hidden")
    except subprocess.TimeoutExpired:
        report["notes"].append("CLI timeout")
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        report["notes"].append(str(exc))
        print(json.dumps(report, indent=2))
        return 0

    raw_score = round(visible_score * 0.55 + hidden_score * 0.45, 3)
    report.update(
        {
            "score": round(raw_score / 100.0, 4),
            "raw_score": raw_score,
            "passed": raw_score >= 80.0,
            "visible_score": visible_score,
            "hidden_score": hidden_score,
            "check_summary": {
                "visible_failures": sum(1 for item in visible_checks if not item.get("passed", True)),
                "hidden_failures": sum(1 for item in hidden_checks if not item.get("passed", True)),
            },
        }
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
