#!/usr/bin/env python3
"""Verify IDP ensemble scoring output against reference.

Reads the agent's output CSV and the reference CSV, compares structure
and numeric values, and prints a JSON result to stdout.

Usage:
    python verify_output.py --output-file <path> --reference-file <path>

Prints JSON to stdout:
    {"score": float, "passed": bool, "reasons": [...]}
Debug/error info goes to stderr.
"""

import argparse
import csv
import json
import sys

REQUIRED_COLUMNS = ["Method", "Total", "CS", "JC", "NOE/PRE"]
NUMERIC_COLUMNS = ["Total", "CS", "JC", "NOE/PRE"]
EXPECTED_MODELS = {"Model1", "Model2", "Model3", "Model4", "Model5"}
# Observables that should NOT appear (hallucination check)
FORBIDDEN_COLUMNS = {"Rg", "FRET", "SAXS", "RDC", "SANS"}


def load_csv(path):
    """Load CSV file into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def round2(val):
    """Round a float to 2 decimal places."""
    return round(float(val), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--reference-file", required=True)
    args = parser.parse_args()

    reasons = []

    # --- Load agent output ---
    try:
        agent_rows = load_csv(args.output_file)
    except FileNotFoundError:
        result = {"score": 0.0, "passed": False, "reasons": ["Output file not found"]}
        print(json.dumps(result))
        return
    except Exception as e:
        result = {"score": 0.0, "passed": False, "reasons": [f"Cannot parse output: {e}"]}
        print(json.dumps(result))
        return

    # --- Load reference ---
    try:
        ref_rows = load_csv(args.reference_file)
    except Exception as e:
        print(json.dumps({"score": 0.0, "passed": False, "reasons": [f"Cannot load reference: {e}"]}),
              file=sys.stdout)
        return

    # --- Structure checks ---
    if not agent_rows:
        print(json.dumps({"score": 0.0, "passed": False, "reasons": ["Output CSV is empty"]}))
        return

    agent_columns = list(agent_rows[0].keys())

    # Check for forbidden columns (hallucination)
    found_forbidden = FORBIDDEN_COLUMNS & set(agent_columns)
    if found_forbidden:
        reasons.append(f"Forbidden observable columns found: {sorted(found_forbidden)}")
        print(json.dumps({"score": 0.0, "passed": False, "reasons": reasons}))
        return

    # Check required columns exist
    missing_cols = set(REQUIRED_COLUMNS) - set(agent_columns)
    if missing_cols:
        reasons.append(f"Missing required columns: {sorted(missing_cols)}")
        print(json.dumps({"score": 0.0, "passed": False, "reasons": reasons}))
        return

    # Check model rows
    agent_models = {row["Method"].strip() for row in agent_rows}
    missing_models = EXPECTED_MODELS - agent_models
    if missing_models:
        reasons.append(f"Missing model rows: {sorted(missing_models)}")
        print(json.dumps({"score": 0.0, "passed": False, "reasons": reasons}))
        return

    # Build lookup by method
    agent_by_method = {}
    for row in agent_rows:
        method = row["Method"].strip()
        if method in EXPECTED_MODELS:
            agent_by_method[method] = row

    ref_by_method = {}
    for row in ref_rows:
        method = row["Method"].strip()
        ref_by_method[method] = row

    # --- Value range check ---
    for method in sorted(EXPECTED_MODELS):
        for col in NUMERIC_COLUMNS:
            try:
                val = float(agent_by_method[method][col])
            except (ValueError, KeyError):
                reasons.append(f"{method}/{col}: not a valid number")
                print(json.dumps({"score": 0.0, "passed": False, "reasons": reasons}))
                return
            if val < 0.0 or val > 1.0:
                reasons.append(f"{method}/{col}={val} is outside [0,1]")
                print(json.dumps({"score": 0.0, "passed": False, "reasons": reasons}))
                return

    # --- Value accuracy (2 decimal places) ---
    total_cells = 0
    matching_cells = 0
    mismatches = []

    for method in sorted(EXPECTED_MODELS):
        for col in NUMERIC_COLUMNS:
            total_cells += 1
            agent_val = round2(agent_by_method[method][col])
            ref_val = round2(ref_by_method[method][col])
            if agent_val == ref_val:
                matching_cells += 1
            else:
                mismatches.append(f"{method}/{col}: agent={agent_val} ref={ref_val}")

    accuracy = matching_cells / total_cells if total_cells > 0 else 0.0

    if mismatches:
        reasons.append(f"{len(mismatches)}/{total_cells} cells mismatched: {mismatches[:5]}")

    # --- Ranking check ---
    try:
        agent_ranking = sorted(
            EXPECTED_MODELS,
            key=lambda m: float(agent_by_method[m]["Total"]),
            reverse=True,
        )
        ref_ranking = sorted(
            EXPECTED_MODELS,
            key=lambda m: float(ref_by_method[m]["Total"]),
            reverse=True,
        )
        if agent_ranking != ref_ranking:
            reasons.append(
                f"Ranking mismatch: agent={agent_ranking} ref={ref_ranking}"
            )
    except (ValueError, KeyError) as e:
        reasons.append(f"Could not compute ranking: {e}")

    passed = accuracy == 1.0 and not reasons
    score = accuracy

    print(json.dumps({"score": score, "passed": passed, "reasons": reasons}))


if __name__ == "__main__":
    main()
