"""Score QSAR prediction CSVs for qsar_model_prediction_instance_1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REQUIRED_COLUMNS = ("Compound_ID", "IC50_pred", "Domain")
DOMAIN_VALUES = {"In", "Out"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")
        return [dict(row) for row in reader]


def _unique_by_compound(rows: list[dict[str, str]], *, label: str) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        compound_id = (row.get("Compound_ID") or "").strip()
        if not compound_id:
            raise ValueError(f"{label} contains a blank Compound_ID")
        if compound_id in by_id:
            duplicates.append(compound_id)
        by_id[compound_id] = row
    if duplicates:
        raise ValueError(f"{label} contains duplicate Compound_ID values: {sorted(set(duplicates))[:5]}")
    return by_id


def _parse_float(value: str, *, compound_id: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"IC50_pred is not numeric for {compound_id}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"IC50_pred is not finite for {compound_id}")
    return parsed


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_ss = sum((x - x_mean) ** 2 for x in xs)
    y_ss = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_ss * y_ss)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _rank_overlap(
    values_a: dict[str, float],
    values_b: dict[str, float],
    *,
    fraction: float,
    highest: bool,
) -> float:
    if not values_a:
        return 0.0
    count = max(1, math.ceil(len(values_a) * fraction))
    ranked_a = sorted(values_a, key=values_a.get, reverse=highest)[:count]
    ranked_b = sorted(values_b, key=values_b.get, reverse=highest)[:count]
    return len(set(ranked_a) & set(ranked_b)) / count


def evaluate_files(
    *,
    output_file: Path,
    reference_file: Path,
    test_file: Path,
) -> dict[str, Any]:
    """Return a score payload for an agent prediction file."""

    if not output_file.exists():
        return {"score": 0.0, "passed": False, "reason": "missing output file"}
    if not reference_file.exists():
        return {"score": 0.0, "passed": False, "reason": "missing reference file"}
    if not test_file.exists():
        return {"score": 0.0, "passed": False, "reason": "missing test input file"}

    try:
        output_rows = _read_csv(output_file)
        reference_rows = _read_csv(reference_file)
        with test_file.open("r", encoding="utf-8-sig", newline="") as handle:
            test_reader = csv.DictReader(handle)
            if test_reader.fieldnames is None or "Compound_ID" not in test_reader.fieldnames:
                raise ValueError("test input is missing Compound_ID")
            test_ids = [(row.get("Compound_ID") or "").strip() for row in test_reader]

        output_by_id = _unique_by_compound(output_rows, label="output")
        reference_by_id = _unique_by_compound(reference_rows, label="reference")
        expected_ids = set(test_ids)
        if "" in expected_ids:
            raise ValueError("test input contains a blank Compound_ID")
        if set(reference_by_id) != expected_ids:
            raise ValueError("reference row set does not match test input row set")
        if set(output_by_id) != expected_ids:
            missing = sorted(expected_ids - set(output_by_id))
            extra = sorted(set(output_by_id) - expected_ids)
            return {
                "score": 0.0,
                "passed": False,
                "reason": "output Compound_ID row set mismatch",
                "missing_count": len(missing),
                "extra_count": len(extra),
                "missing_sample": missing[:5],
                "extra_sample": extra[:5],
            }

        ordered_ids = [compound_id for compound_id in test_ids if compound_id]
        output_values: dict[str, float] = {}
        reference_values: dict[str, float] = {}
        domain_matches = 0

        for compound_id in ordered_ids:
            output_domain = (output_by_id[compound_id].get("Domain") or "").strip()
            reference_domain = (reference_by_id[compound_id].get("Domain") or "").strip()
            if output_domain not in DOMAIN_VALUES:
                raise ValueError(f"Domain must be In or Out for {compound_id}")
            if reference_domain not in DOMAIN_VALUES:
                raise ValueError(f"reference Domain must be In or Out for {compound_id}")
            if output_domain == reference_domain:
                domain_matches += 1
            output_values[compound_id] = _parse_float(
                output_by_id[compound_id].get("IC50_pred", ""),
                compound_id=compound_id,
            )
            reference_values[compound_id] = _parse_float(
                reference_by_id[compound_id].get("IC50_pred", ""),
                compound_id=compound_id,
            )

        output_series = [output_values[compound_id] for compound_id in ordered_ids]
        reference_series = [reference_values[compound_id] for compound_id in ordered_ids]
        pearson_r = _pearson(output_series, reference_series)
        top_overlap = _rank_overlap(output_values, reference_values, fraction=0.30, highest=True)
        bottom_overlap = _rank_overlap(output_values, reference_values, fraction=0.30, highest=False)
        domain_agreement = domain_matches / len(ordered_ids) if ordered_ids else 0.0

        metrics = {
            "pearson_r": pearson_r,
            "top_30_overlap": top_overlap,
            "bottom_30_overlap": bottom_overlap,
            "domain_agreement": domain_agreement,
        }
        passed = (
            pearson_r >= 0.6
            and top_overlap >= 0.5
            and bottom_overlap >= 0.5
            and domain_agreement >= 0.7
        )
        return {
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "row_count": len(ordered_ids),
            **metrics,
        }
    except Exception as exc:
        return {"score": 0.0, "passed": False, "reason": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score QSAR prediction output.")
    parser.add_argument("--agent", required=True, type=Path)
    parser.add_argument("--ref", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    args = parser.parse_args()

    result = evaluate_files(output_file=args.agent, reference_file=args.ref, test_file=args.test)
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result.get("score", 0.0) == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
