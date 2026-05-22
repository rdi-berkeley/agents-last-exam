"""Verification helpers for epidemiology_forecast outputs."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass

import pandas as pd

MODEL_REQUIRED_COLUMNS = ["model", "wis", "rel_wis", "mae", "cov50", "cov95"]
PER_CELL_REQUIRED_COLUMNS = [
    "model",
    "location",
    "target_end_date",
    "target",
    "wis",
    "ae_median",
    "cov50_hit",
    "cov95_hit",
]
CELL_KEY = ["model", "location", "target_end_date", "target"]
ABS_TOL_MODEL = {"rel_wis": 0.005, "cov50": 0.005, "cov95": 0.005}
REL_TOL_MODEL = {"wis": 0.001, "mae": 0.001}
CELL_TOL_WIS = 0.001
CELL_TOL_AE = 1e-6


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_csv(text: str, *, location_as_str: bool = False) -> pd.DataFrame:
    kwargs = {"dtype": {"location": str}} if location_as_str else {}
    return pd.read_csv(io.StringIO(text), **kwargs)


def _missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [column for column in required if column not in df.columns]


def _score_failure(reason: str, hard_gate: str, **details: object) -> ScoreResult:
    return ScoreResult(score=0.0, passed=False, reason=reason, hard_gate=hard_gate, details=details)


def verify_output_bundle(
    *,
    submission_csv: str,
    per_cell_csv: str,
    reference_submission_csv: str,
    reference_per_cell_csv: str,
) -> ScoreResult:
    try:
        submission = _load_csv(submission_csv)
        per_cell = _load_csv(per_cell_csv, location_as_str=True)
        reference_submission = _load_csv(reference_submission_csv)
        reference_per_cell = _load_csv(reference_per_cell_csv, location_as_str=True)
    except Exception as exc:
        return _score_failure("parse_error", "csv_parse_failure", error=str(exc))

    missing_model_cols = _missing_columns(submission, MODEL_REQUIRED_COLUMNS)
    if missing_model_cols:
        return _score_failure(
            "schema_mismatch",
            "submission_missing_columns",
            missing_columns=missing_model_cols,
        )

    missing_cell_cols = _missing_columns(per_cell, PER_CELL_REQUIRED_COLUMNS)
    if missing_cell_cols:
        return _score_failure(
            "schema_mismatch",
            "per_cell_missing_columns",
            missing_columns=missing_cell_cols,
        )

    if submission["model"].duplicated().any():
        return _score_failure(
            "schema_mismatch",
            "duplicate_submission_models",
            duplicate_models=sorted(
                submission.loc[submission["model"].duplicated(), "model"].unique().tolist()
            ),
        )

    reference_models = set(reference_submission["model"])
    submission_models = set(submission["model"])
    extra_models = sorted(submission_models - reference_models)
    missing_models = sorted(reference_models - submission_models)
    if extra_models or missing_models:
        return _score_failure(
            "model_set_mismatch",
            "submission_model_set",
            extra_models=extra_models,
            missing_models=missing_models,
        )

    merged_cells = reference_per_cell.merge(
        per_cell,
        on=CELL_KEY,
        how="outer",
        suffixes=("_ref", "_sub"),
        indicator=True,
    )
    only_ref = merged_cells[merged_cells["_merge"] == "left_only"]
    only_sub = merged_cells[merged_cells["_merge"] == "right_only"]
    if len(only_ref) or len(only_sub):
        return _score_failure(
            "row_set_mismatch",
            "per_cell_row_set",
            missing_rows=int(len(only_ref)),
            extra_rows=int(len(only_sub)),
            first_missing=only_ref[CELL_KEY].head(3).to_dict("records"),
            first_extra=only_sub[CELL_KEY].head(3).to_dict("records"),
        )

    cell_overlap = merged_cells[merged_cells["_merge"] == "both"].copy()
    cell_overlap["wis_err"] = (cell_overlap["wis_sub"] - cell_overlap["wis_ref"]).abs()
    cell_overlap["ae_err"] = (cell_overlap["ae_median_sub"] - cell_overlap["ae_median_ref"]).abs()
    cell_overlap["cov50_mismatch"] = cell_overlap["cov50_hit_sub"].astype(int) != cell_overlap[
        "cov50_hit_ref"
    ].astype(int)
    cell_overlap["cov95_mismatch"] = cell_overlap["cov95_hit_sub"].astype(int) != cell_overlap[
        "cov95_hit_ref"
    ].astype(int)

    bad_cells = cell_overlap[
        (cell_overlap["wis_err"] > CELL_TOL_WIS + 1e-9)
        | (cell_overlap["ae_err"] > CELL_TOL_AE + 1e-9)
        | cell_overlap["cov50_mismatch"]
        | cell_overlap["cov95_mismatch"]
    ]
    if len(bad_cells):
        sample = bad_cells[
            CELL_KEY
            + ["wis_ref", "wis_sub", "wis_err", "ae_err", "cov50_mismatch", "cov95_mismatch"]
        ].head(5)
        return _score_failure(
            "cell_value_mismatch",
            "per_cell_tolerance_failure",
            bad_rows=int(len(bad_cells)),
            max_wis_error=float(cell_overlap["wis_err"].max()),
            max_ae_error=float(cell_overlap["ae_err"].max()),
            sample_bad_rows=sample.to_dict("records"),
        )

    merged_models = reference_submission.merge(submission, on="model", suffixes=("_ref", "_sub"))
    metric_errors: dict[str, float] = {}
    for metric, tol in ABS_TOL_MODEL.items():
        err = (merged_models[f"{metric}_sub"] - merged_models[f"{metric}_ref"]).abs()
        metric_errors[f"{metric}_max_error"] = float(err.max())
        if (err > tol + 1e-9).any():
            bad_row = merged_models.loc[
                err.idxmax(), ["model", f"{metric}_ref", f"{metric}_sub"]
            ].to_dict()
            return _score_failure(
                "model_value_mismatch",
                f"{metric}_absolute_tolerance_failure",
                metric=metric,
                max_error=float(err.max()),
                offending_row=bad_row,
            )

    for metric, tol in REL_TOL_MODEL.items():
        rel_err = (
            merged_models[f"{metric}_sub"] - merged_models[f"{metric}_ref"]
        ).abs() / merged_models[f"{metric}_ref"].abs().clip(lower=1e-9)
        metric_errors[f"{metric}_max_relative_error"] = float(rel_err.max())
        if (rel_err > tol + 1e-9).any():
            bad_row = merged_models.loc[
                rel_err.idxmax(), ["model", f"{metric}_ref", f"{metric}_sub"]
            ].to_dict()
            return _score_failure(
                "model_value_mismatch",
                f"{metric}_relative_tolerance_failure",
                metric=metric,
                max_relative_error=float(rel_err.max()),
                offending_row=bad_row,
            )

    return ScoreResult(
        score=1.0,
        passed=True,
        reason="ok",
        hard_gate=None,
        details={
            "submission_rows": int(len(submission)),
            "per_cell_rows": int(len(per_cell)),
            **metric_errors,
            "max_per_cell_wis_error": float(cell_overlap["wis_err"].max()),
            "max_per_cell_ae_error": float(cell_overlap["ae_err"].max()),
        },
    )


def _read_csv_header(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def main() -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_dir)
    result = verify_output_bundle(
        submission_csv=(output_dir / "submission.csv").read_text(encoding="utf-8"),
        per_cell_csv=(output_dir / "per_cell_scores.csv").read_text(encoding="utf-8"),
        reference_submission_csv=(reference_dir / "expected_table1_2021-22.csv").read_text(
            encoding="utf-8"
        ),
        reference_per_cell_csv=(reference_dir / "expected_per_cell_2021-22.csv").read_text(
            encoding="utf-8"
        ),
    )
    payload = result.to_dict()
    payload["submission_header"] = _read_csv_header(str(output_dir / "submission.csv"))
    payload["per_cell_header"] = _read_csv_header(str(output_dir / "per_cell_scores.csv"))
    print(json.dumps(payload, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
