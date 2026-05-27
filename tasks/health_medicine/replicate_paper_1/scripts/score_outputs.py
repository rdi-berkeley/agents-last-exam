#!/usr/bin/env python
"""Semantic verifier for replicate_paper_1 output bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


METHOD_ALIASES = {
    "iptw": "IPTW",
    "ipw": "IPTW",
    "inverse probability of treatment weighting": "IPTW",
    "a-iptw": "A-IPTW",
    "aiptw": "A-IPTW",
    "a iptw": "A-IPTW",
    "augmented iptw": "A-IPTW",
    "augmented inverse probability of treatment weighting": "A-IPTW",
    "tmle": "TMLE",
    "targeted maximum likelihood estimation": "TMLE",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: object) -> float:
    result = float(str(value).strip())
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {value!r}")
    return result


def normalize_method(value: str) -> str:
    cleaned = " ".join(value.strip().lower().replace("_", " ").split())
    cleaned = cleaned.replace("a iptw", "a-iptw")
    return METHOD_ALIASES.get(cleaned, value.strip())


def parse_sample_size(value: object) -> int:
    return int(float(str(value).strip()))


def check_file(path: Path, label: str, reasons: list[str]) -> bool:
    if not path.exists():
        reasons.append(f"missing {label}: {path}")
        return False
    if path.is_file() and path.stat().st_size <= 0:
        reasons.append(f"empty {label}: {path}")
        return False
    return True


def require_columns(actual: list[str], expected: list[str], label: str, reasons: list[str]) -> None:
    missing = [column for column in expected if column not in actual]
    if missing:
        reasons.append(f"{label} missing columns: {missing}")


def validate_summary(
    output_dir: Path,
    contract: dict[str, Any],
    reasons: list[str],
    details: dict[str, Any],
    *,
    allow_fixture_copy: bool,
) -> dict[tuple[int, str], dict[str, float]]:
    summary_path = output_dir / "summary_results.csv"
    if not check_file(summary_path, "summary_results.csv", reasons):
        return {}

    fixture_hash = contract.get("fixture_copy_guard", {}).get("positive_summary_sha256")
    if fixture_hash and sha256_file(summary_path) == fixture_hash and not allow_fixture_copy:
        reasons.append("summary_results.csv exactly matches the positive fixture")

    fields, rows = load_csv(summary_path)
    require_columns(fields, list(contract["summary_results_columns"]), "summary_results.csv", reasons)
    if not rows:
        reasons.append("summary_results.csv has no data rows")
        return {}

    required_keys = {
        (int(sample_size), method)
        for sample_size in contract["required_sample_sizes"]
        for method in contract["required_methods"]
    }
    seen: dict[tuple[int, str], dict[str, float]] = {}
    truth_low, truth_high = [float(v) for v in contract["truth_estimate_range"]]
    min_reps = int(contract["minimum_replications"])
    min_truth_n = int(contract["minimum_truth_n"])

    for idx, row in enumerate(rows, start=2):
        try:
            key = (parse_sample_size(row["sample_size"]), normalize_method(row["estimator"]))
        except Exception as exc:
            reasons.append(f"summary_results.csv row {idx} has invalid key: {exc}")
            continue
        if key in seen:
            reasons.append(f"summary_results.csv duplicate row for {key}")
            continue

        parsed: dict[str, float] = {}
        for column in [
            "mean_estimate",
            "bias",
            "variance",
            "mcse",
            "ci_lower",
            "ci_upper",
            "coverage",
            "replications",
            "truth_estimate",
            "truth_n",
        ]:
            try:
                parsed[column] = finite_float(row[column])
            except Exception:
                reasons.append(f"summary_results.csv row {idx} has non-finite {column}")
        if not parsed:
            continue

        if not (truth_low <= parsed["truth_estimate"] <= truth_high):
            reasons.append(f"truth_estimate out of range for {key}: {parsed['truth_estimate']}")
        if parsed["truth_n"] < min_truth_n:
            reasons.append(f"truth_n below minimum for {key}: {parsed['truth_n']}")
        if parsed["replications"] < min_reps:
            reasons.append(f"replications below minimum for {key}: {parsed['replications']}")
        if parsed["variance"] < 0:
            reasons.append(f"variance is negative for {key}")
        if not (0 <= parsed["coverage"] <= 1):
            reasons.append(f"coverage out of [0,1] for {key}")
        if parsed["ci_lower"] > parsed["ci_upper"]:
            reasons.append(f"ci_lower > ci_upper for {key}")

        seen[key] = parsed

    if set(seen) != required_keys:
        missing = sorted(required_keys - set(seen))
        extra = sorted(set(seen) - required_keys)
        if missing:
            reasons.append(f"summary_results.csv missing method/sample rows: {missing}")
        if extra:
            reasons.append(f"summary_results.csv has unexpected method/sample rows: {extra}")

    details["summary_rows"] = len(rows)
    return seen


def validate_coverage(
    output_dir: Path,
    contract: dict[str, Any],
    summary: dict[tuple[int, str], dict[str, float]],
    reasons: list[str],
    details: dict[str, Any],
) -> None:
    coverage_path = output_dir / "coverage_results.csv"
    if not check_file(coverage_path, "coverage_results.csv", reasons):
        return

    fields, rows = load_csv(coverage_path)
    require_columns(fields, list(contract["coverage_results_columns"]), "coverage_results.csv", reasons)
    required_keys = {
        (int(sample_size), method)
        for sample_size in contract["required_sample_sizes"]
        for method in contract["required_methods"]
    }
    seen: set[tuple[int, str]] = set()
    for idx, row in enumerate(rows, start=2):
        try:
            key = (parse_sample_size(row["sample_size"]), normalize_method(row["estimator"]))
            coverage = finite_float(row["coverage"])
            mean_ci_width = finite_float(row["mean_ci_width"])
            ci_lower = finite_float(row["ci_lower_mean"])
            ci_upper = finite_float(row["ci_upper_mean"])
        except Exception as exc:
            reasons.append(f"coverage_results.csv row {idx} invalid: {exc}")
            continue
        if key in seen:
            reasons.append(f"coverage_results.csv duplicate row for {key}")
        seen.add(key)
        if not (0 <= coverage <= 1):
            reasons.append(f"coverage_results.csv coverage out of [0,1] for {key}")
        if mean_ci_width < 0:
            reasons.append(f"coverage_results.csv negative mean_ci_width for {key}")
        if ci_lower > ci_upper:
            reasons.append(f"coverage_results.csv ci_lower_mean > ci_upper_mean for {key}")
        if key in summary and abs(coverage - summary[key]["coverage"]) > 0.10:
            reasons.append(f"coverage mismatch between summary and coverage files for {key}")

    if seen != required_keys:
        missing = sorted(required_keys - seen)
        extra = sorted(seen - required_keys)
        if missing:
            reasons.append(f"coverage_results.csv missing method/sample rows: {missing}")
        if extra:
            reasons.append(f"coverage_results.csv has unexpected method/sample rows: {extra}")
    details["coverage_rows"] = len(rows)


def validate_n500_improvements(
    summary: dict[tuple[int, str], dict[str, float]],
    contract: dict[str, Any],
    reasons: list[str],
) -> None:
    rule = contract["n500_required_improvements"]
    baseline = rule["baseline"]
    baseline_row = summary.get((500, baseline))
    if not baseline_row:
        return
    iptw_abs_bias = abs(baseline_row["bias"])
    iptw_variance = baseline_row["variance"]
    iptw_coverage = baseline_row["coverage"]
    min_relative = float(contract["coverage_n500_min_relative_to_iptw"])
    for method in rule["improved_methods"]:
        row = summary.get((500, method))
        if not row:
            continue
        if abs(row["bias"]) >= iptw_abs_bias:
            reasons.append(f"n=500 absolute bias for {method} is not lower than IPTW")
        if row["variance"] >= iptw_variance:
            reasons.append(f"n=500 variance for {method} is not lower than IPTW")
        if row["coverage"] < iptw_coverage + min_relative:
            reasons.append(f"n=500 coverage for {method} is catastrophically below IPTW")


def validate_misc_outputs(output_dir: Path, reasons: list[str], details: dict[str, Any]) -> None:
    check_file(output_dir / "summary_findings.txt", "summary_findings.txt", reasons)
    metadata_path = output_dir / "run_metadata.json"
    if check_file(metadata_path, "run_metadata.json", reasons):
        try:
            details["run_metadata_keys"] = sorted(load_json(metadata_path))
        except Exception as exc:
            reasons.append(f"run_metadata.json is not valid JSON: {exc}")

    plot_dir = output_dir / "comparison_plots"
    if not plot_dir.exists() or not plot_dir.is_dir():
        reasons.append("missing comparison_plots directory")
        return
    plot_files = [
        path
        for path in plot_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"} and path.stat().st_size > 0
    ]
    if not plot_files:
        reasons.append("comparison_plots contains no non-empty .png or .pdf plot")
    details["plot_files"] = sorted(path.name for path in plot_files)


def evaluate(output_dir: Path, reference_dir: Path, *, allow_fixture_copy: bool = False) -> dict[str, Any]:
    reasons: list[str] = []
    details: dict[str, Any] = {}
    contract_path = reference_dir / "evaluation_contract.json"
    if not check_file(contract_path, "evaluation_contract.json", reasons):
        return {"score": 0.0, "passed": False, "reasons": reasons, "details": details}
    contract = load_json(contract_path)

    summary = validate_summary(output_dir, contract, reasons, details, allow_fixture_copy=allow_fixture_copy)
    validate_coverage(output_dir, contract, summary, reasons, details)
    validate_n500_improvements(summary, contract, reasons)
    validate_misc_outputs(output_dir, reasons, details)

    passed = not reasons
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "reasons": reasons,
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score replicate_paper_1 outputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--allow-fixture-copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate(
        Path(args.output_dir),
        Path(args.reference_dir),
        allow_fixture_copy=args.allow_fixture_copy,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

