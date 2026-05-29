"""Deterministic scorer for nhanes_crp_multimorbidity_interaction."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NUMERIC_TOLERANCE = 1e-9

DESCRIPTIVE_COLUMNS = [
    "group",
    "group_label",
    "n_person",
    "age_mean",
    "age_sd",
    "bmi_mean",
    "bmi_sd",
    "education_mean",
    "education_sd",
    "ulcer_rate_pct",
    "MM_count_mean",
    "MM_count_sd",
    "crp_median",
    "crp_p25",
    "crp_p75",
    "male_pct",
    "female_pct",
    "hibpe_pct",
    "diabe_pct",
    "hearte_pct",
    "lunge_pct",
    "arthritis_pct",
    "cancer_pct",
    "mm0_pct",
    "mm1_pct",
    "mm2_pct",
    "mm3plus_pct",
]

DISTRIBUTION_COLUMNS = [
    "mm_group",
    "n_person",
    "crp_mean",
    "crp_median",
    "crp_p95",
    "crp_high_pct",
]

MODEL_COLUMNS = [
    "cohort",
    "model",
    "method",
    "term",
    "beta",
    "se",
    "or",
    "ci_low",
    "ci_high",
    "p_value",
    "n_obs",
    "n_person",
    "outcome",
    "crp_cut_mgL",
    "crp_bins",
]

STRATIFIED_COLUMNS = [
    "cohort",
    "model",
    "crp_group",
    "or",
    "ci_low",
    "ci_high",
    "n_obs",
    "n_person",
]

TERTILE_COLUMNS = [
    "cohort",
    "crp_tertile",
    "or",
    "ci_low",
    "ci_high",
    "n_obs",
    "n_person",
]

FILE_SPECS = {
    "nhanes_crp_descriptives.csv": {
        "columns": DESCRIPTIVE_COLUMNS,
        "keys": ["group"],
        "numeric_columns": [column for column in DESCRIPTIVE_COLUMNS if column != "group_label"],
    },
    "nhanes_crp_distribution_by_mm.csv": {
        "columns": DISTRIBUTION_COLUMNS,
        "keys": ["mm_group"],
        "numeric_columns": [column for column in DISTRIBUTION_COLUMNS if column != "mm_group"],
    },
    "nhanes_crp_models.csv": {
        "columns": MODEL_COLUMNS,
        "keys": ["model", "term"],
        "numeric_columns": [
            "beta",
            "se",
            "or",
            "ci_low",
            "ci_high",
            "p_value",
            "n_obs",
            "n_person",
            "crp_cut_mgL",
        ],
    },
    "nhanes_crp_stratified.csv": {
        "columns": STRATIFIED_COLUMNS,
        "keys": ["crp_group"],
        "numeric_columns": ["or", "ci_low", "ci_high", "n_obs", "n_person"],
    },
    "nhanes_crp_tertile.csv": {
        "columns": TERTILE_COLUMNS,
        "keys": ["crp_tertile"],
        "numeric_columns": ["or", "ci_low", "ci_high", "n_obs", "n_person"],
    },
}


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hard_fail(reason: str, details: dict[str, Any] | None = None) -> ScoreResult:
    return ScoreResult(0.0, False, reason, reason, details or {})


def _read_csv(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if reader.fieldnames is None:
        raise ValueError("CSV has no header row")
    return list(reader.fieldnames), list(reader)


def _parse_float(value: str) -> float | None:
    stripped = value.strip()
    if stripped == "":
        return None
    number = float(stripped)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value {value!r}")
    return number


def _numeric_match(left_raw: str, right_raw: str) -> bool:
    left = _parse_float(left_raw)
    right = _parse_float(right_raw)
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, rel_tol=NUMERIC_TOLERANCE, abs_tol=NUMERIC_TOLERANCE)


def _sort_rows(rows: list[dict[str, str]], keys: list[str]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: tuple(row[key].strip() for key in keys))


def _compare_file(
    *,
    name: str,
    candidate_text: str,
    reference_text: str,
) -> ScoreResult | None:
    spec = FILE_SPECS[name]
    try:
        candidate_columns, candidate_rows = _read_csv(candidate_text)
        reference_columns, reference_rows = _read_csv(reference_text)
    except Exception as exc:
        return _hard_fail(f"{name}: csv_parse_error", {"error": str(exc)})

    if candidate_columns != spec["columns"]:
        return _hard_fail(
            f"{name}: schema_mismatch",
            {"expected": spec["columns"], "observed": candidate_columns},
        )
    if reference_columns != spec["columns"]:
        return _hard_fail(
            f"{name}: reference_schema_mismatch",
            {"expected": spec["columns"], "observed": reference_columns},
        )
    if len(candidate_rows) != len(reference_rows):
        return _hard_fail(
            f"{name}: row_count_mismatch",
            {"candidate": len(candidate_rows), "reference": len(reference_rows)},
        )

    candidate_sorted = _sort_rows(candidate_rows, spec["keys"])
    reference_sorted = _sort_rows(reference_rows, spec["keys"])
    for row_index, (candidate_row, reference_row) in enumerate(
        zip(candidate_sorted, reference_sorted),
        start=1,
    ):
        for column in spec["columns"]:
            candidate_value = candidate_row[column]
            reference_value = reference_row[column]
            if column in spec["numeric_columns"]:
                if not _numeric_match(candidate_value, reference_value):
                    return _hard_fail(
                        f"{name}: numeric_mismatch",
                        {
                            "row_index": row_index,
                            "keys": {key: candidate_row[key] for key in spec["keys"]},
                            "column": column,
                            "candidate": candidate_value,
                            "reference": reference_value,
                        },
                    )
            elif candidate_value != reference_value:
                return _hard_fail(
                    f"{name}: value_mismatch",
                    {
                        "row_index": row_index,
                        "keys": {key: candidate_row[key] for key in spec["keys"]},
                        "column": column,
                        "candidate": candidate_value,
                        "reference": reference_value,
                    },
                )
    return None


def score_output_bundle(
    *,
    candidate_descriptives_csv: str,
    candidate_distribution_csv: str,
    candidate_models_csv: str,
    candidate_stratified_csv: str,
    candidate_tertile_csv: str,
    reference_descriptives_csv: str,
    reference_distribution_csv: str,
    reference_models_csv: str,
    reference_stratified_csv: str,
    reference_tertile_csv: str,
) -> ScoreResult:
    comparisons = [
        ("nhanes_crp_descriptives.csv", candidate_descriptives_csv, reference_descriptives_csv),
        (
            "nhanes_crp_distribution_by_mm.csv",
            candidate_distribution_csv,
            reference_distribution_csv,
        ),
        ("nhanes_crp_models.csv", candidate_models_csv, reference_models_csv),
        ("nhanes_crp_stratified.csv", candidate_stratified_csv, reference_stratified_csv),
        ("nhanes_crp_tertile.csv", candidate_tertile_csv, reference_tertile_csv),
    ]
    per_file_weight = 1.0 / len(comparisons)
    passed_files: list[str] = []
    failed_files: list[dict[str, Any]] = []
    for name, candidate_text, reference_text in comparisons:
        result = _compare_file(name=name, candidate_text=candidate_text, reference_text=reference_text)
        if result is None:
            passed_files.append(name)
        else:
            failed_files.append({"file": name, "reason": result.reason, "details": result.details})
    score = round(len(passed_files) * per_file_weight, 10)
    if not failed_files:
        return ScoreResult(
            score=1.0,
            passed=True,
            reason="exact_match_within_tolerance",
            hard_gate=None,
            details={
                "files_checked": [name for name, _, _ in comparisons],
                "numeric_tolerance": NUMERIC_TOLERANCE,
            },
        )
    return ScoreResult(
        score=score,
        passed=False,
        reason=f"{len(passed_files)}/{len(comparisons)} files matched",
        hard_gate=failed_files[0]["reason"],
        details={
            "passed_files": passed_files,
            "failed_files": failed_files,
            "numeric_tolerance": NUMERIC_TOLERANCE,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidate_dir = args.candidate_dir
    reference_dir = args.reference_dir
    result = score_output_bundle(
        candidate_descriptives_csv=(candidate_dir / "nhanes_crp_descriptives.csv").read_text(encoding="utf-8"),
        candidate_distribution_csv=(
            candidate_dir / "nhanes_crp_distribution_by_mm.csv"
        ).read_text(encoding="utf-8"),
        candidate_models_csv=(candidate_dir / "nhanes_crp_models.csv").read_text(encoding="utf-8"),
        candidate_stratified_csv=(
            candidate_dir / "nhanes_crp_stratified.csv"
        ).read_text(encoding="utf-8"),
        candidate_tertile_csv=(candidate_dir / "nhanes_crp_tertile.csv").read_text(encoding="utf-8"),
        reference_descriptives_csv=(
            reference_dir / "nhanes_crp_descriptives.csv"
        ).read_text(encoding="utf-8"),
        reference_distribution_csv=(
            reference_dir / "nhanes_crp_distribution_by_mm.csv"
        ).read_text(encoding="utf-8"),
        reference_models_csv=(reference_dir / "nhanes_crp_models.csv").read_text(encoding="utf-8"),
        reference_stratified_csv=(
            reference_dir / "nhanes_crp_stratified.csv"
        ).read_text(encoding="utf-8"),
        reference_tertile_csv=(reference_dir / "nhanes_crp_tertile.csv").read_text(encoding="utf-8"),
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
