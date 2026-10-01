#!/usr/bin/env python3
"""Validate both CSV contracts, then award unchanged two-decimal cell credit."""

import argparse
import csv
import json
import math
import sys

REQUIRED_COLUMNS = ["Method", "Total", "CS", "JC", "NOE/PRE"]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]
EXPECTED_MODELS = {"Model1", "Model2", "Model3", "Model4", "Model5"}


def load_csv(path):
    """Return a validated method lookup or raise for an invalid CSV."""
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)):
            raise ValueError("CSV must have nonduplicate column names")
        if set(headers) != set(REQUIRED_COLUMNS):
            raise ValueError("Expected exactly these columns: " + ", ".join(REQUIRED_COLUMNS))
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV row width does not match header")
    if len(rows) != 5 or {row["Method"].strip() for row in rows} != EXPECTED_MODELS:
        raise ValueError("Expected exactly one row for each of Model1 through Model5")
    result = {}
    for row in rows:
        method = row["Method"].strip()
        result[method] = {}
        for column in NUMERIC_COLUMNS:
            value = float(row[column])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{method}/{column}: expected a finite number in [0,1]")
            result[method][column] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--reference-file", required=True)
    args = parser.parse_args()
    try:
        reference = load_csv(args.reference_file)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(json.dumps({"error": "invalid_reference", "message": str(error)}))
        return 2
    try:
        candidate = load_csv(args.output_file)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(json.dumps({"score": 0.0, "passed": False, "reasons": [f"Invalid output: {error}"]}))
        return 0

    mismatches = []
    for method in sorted(EXPECTED_MODELS):
        for column in NUMERIC_COLUMNS:
            actual = round(candidate[method][column], 2)
            expected = round(reference[method][column], 2)
            if actual != expected:
                mismatches.append(f"{method}/{column}: agent={actual} ref={expected}")
    score = (20 - len(mismatches)) / 20
    reasons = []
    if mismatches:
        reasons.append(f"{len(mismatches)}/20 cells mismatched: {mismatches[:5]}")
    candidate_ranking = sorted(
        EXPECTED_MODELS, key=lambda method: (-candidate[method]["Total"], method)
    )
    reference_ranking = sorted(
        EXPECTED_MODELS, key=lambda method: (-reference[method]["Total"], method)
    )
    if candidate_ranking != reference_ranking:
        reasons.append(f"Ranking mismatch: agent={candidate_ranking} ref={reference_ranking}")
    print(json.dumps({"score": score, "passed": score == 1.0, "reasons": reasons}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
