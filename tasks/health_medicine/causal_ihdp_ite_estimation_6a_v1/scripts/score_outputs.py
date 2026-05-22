"""Local scorer for causal_ihdp_ite_estimation_6a_v1."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_OUTPUT_COLUMNS = ["replication", "unit_id", "mu0_hat", "mu1_hat", "ite_hat"]
REQUIRED_GOLD_COLUMNS = ["replication", "unit_id", "mu0", "mu1"]
REQUIRED_MODEL_SELECTION_COLUMNS = [
    "candidate_id",
    "method_name",
    "selection_objective",
    "selected",
    "valid_rows",
]
REQUIRED_OVERLAP_COLUMNS = ["replication", "bin_id", "propensity_mean", "n_treated", "n_control"]
REQUIRED_SUBGROUP_COLUMNS = ["subgroup_name", "subgroup_value", "n", "mean_ite_hat", "std_ite_hat"]

LEADERBOARD_ANCHOR = 2.348958
PASS_THRESHOLD = 2.80
CONSISTENCY_TOL = 1e-6
CONSTANT_STD_TOL = 1e-3
ROUND_DECIMALS = 6
MIN_NOTES_CHARS = 100
HARD_FAIL_SCORE = 1_000_000.0


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score_failure(reason: str, hard_gate: str, **details: Any) -> ScoreResult:
    payload = {
        "metric_name": "mean_sqrt_epsilon_PEHE",
        "pass_threshold": PASS_THRESHOLD,
        "leaderboard_anchor": LEADERBOARD_ANCHOR,
        "lower_is_better": True,
        "computed_metric_available": False,
        **details,
    }
    return ScoreResult(
        score=HARD_FAIL_SCORE,
        passed=False,
        reason=reason,
        hard_gate=hard_gate,
        details=payload,
    )


def _load_csv(text: str | bytes) -> pd.DataFrame:
    payload = text.decode("utf-8-sig") if isinstance(text, bytes) else text
    return pd.read_csv(io.StringIO(payload))


def _read_text_if_exists(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _parse_optional_csv(text: str | bytes | None) -> pd.DataFrame | None:
    if text is None:
        return None
    return _load_csv(text)


def _has_required_columns(df: pd.DataFrame, required: list[str]) -> bool:
    return all(column in df.columns for column in required)


def _to_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    )


def _coerce_key_columns(df: pd.DataFrame, *, label: str) -> tuple[pd.DataFrame | None, ScoreResult | None]:
    numeric_keys = df[["replication", "unit_id"]].apply(pd.to_numeric, errors="coerce")
    if numeric_keys.isna().any().any():
        return None, _score_failure(f"{label}_non_numeric_keys", f"{label}_non_numeric_keys")
    if not np.isfinite(numeric_keys.to_numpy()).all():
        return None, _score_failure(f"{label}_non_finite_keys", f"{label}_non_finite_keys")
    rounded_keys = np.round(numeric_keys.to_numpy())
    if not np.array_equal(numeric_keys.to_numpy(), rounded_keys):
        return None, _score_failure(f"{label}_non_integer_keys", f"{label}_non_integer_keys")
    coerced = df.copy()
    coerced[["replication", "unit_id"]] = numeric_keys.astype(np.int64)
    return coerced, None


def _validate_artifacts(
    model_selection_csv: str | bytes | None,
    overlap_csv: str | bytes | None,
    subgroup_csv: str | bytes | None,
    run_notes_txt: str | bytes | None,
) -> list[str]:
    invalidators: list[str] = []

    if model_selection_csv is None:
        invalidators.append("Missing artifacts/model_selection.csv")
    else:
        try:
            model_df = _parse_optional_csv(model_selection_csv)
        except Exception:
            model_df = None
        if model_df is None or model_df.empty or not _has_required_columns(
            model_df, REQUIRED_MODEL_SELECTION_COLUMNS
        ):
            invalidators.append("Malformed artifacts/model_selection.csv")
        else:
            selected = _to_bool_series(model_df["selected"])
            if selected.isna().any():
                invalidators.append(
                    "Malformed artifacts/model_selection.csv: selected must be boolean-like"
                )
            if model_df["candidate_id"].astype(str).nunique() < 2:
                invalidators.append(
                    "Low-signal model_selection.csv: fewer than 2 candidate rows"
                )
            if int(selected.sum()) != 1:
                invalidators.append(
                    "Malformed artifacts/model_selection.csv: exactly 1 selected row required"
                )
            objectives = pd.to_numeric(model_df["selection_objective"], errors="coerce")
            if objectives.isna().any() or not np.isfinite(objectives.to_numpy()).all():
                invalidators.append(
                    "Malformed artifacts/model_selection.csv: non-finite selection_objective"
                )
            valid_rows = pd.to_numeric(model_df["valid_rows"], errors="coerce")
            if valid_rows.isna().any() or not (valid_rows > 0).all():
                invalidators.append(
                    "Malformed artifacts/model_selection.csv: valid_rows must be positive"
                )

    if overlap_csv is None:
        invalidators.append("Missing artifacts/overlap_diagnostics.csv")
    else:
        try:
            overlap_df = _parse_optional_csv(overlap_csv)
        except Exception:
            overlap_df = None
        if overlap_df is None or overlap_df.empty or not _has_required_columns(
            overlap_df, REQUIRED_OVERLAP_COLUMNS
        ):
            invalidators.append("Malformed artifacts/overlap_diagnostics.csv")
        else:
            if overlap_df["replication"].nunique() != 100:
                invalidators.append(
                    "Low-signal overlap_diagnostics.csv: all 100 replications must appear"
                )
            bins_per_replication = overlap_df.groupby("replication")["bin_id"].nunique()
            if not bins_per_replication.empty and (bins_per_replication < 5).any():
                invalidators.append(
                    "Low-signal overlap_diagnostics.csv: at least 5 bins per replication required"
                )
            propensity_mean = pd.to_numeric(overlap_df["propensity_mean"], errors="coerce")
            if propensity_mean.isna().any() or not (
                ((propensity_mean >= 0) & (propensity_mean <= 1)).all()
            ):
                invalidators.append(
                    "Malformed artifacts/overlap_diagnostics.csv: propensity_mean must be in [0, 1]"
                )
            n_treated = pd.to_numeric(overlap_df["n_treated"], errors="coerce")
            n_control = pd.to_numeric(overlap_df["n_control"], errors="coerce")
            if (
                n_treated.isna().any()
                or n_control.isna().any()
                or (n_treated < 0).any()
                or (n_control < 0).any()
            ):
                invalidators.append(
                    "Malformed artifacts/overlap_diagnostics.csv: counts must be non-negative"
                )

    if subgroup_csv is None:
        invalidators.append("Missing artifacts/subgroup_ite.csv")
    else:
        try:
            subgroup_df = _parse_optional_csv(subgroup_csv)
        except Exception:
            subgroup_df = None
        if subgroup_df is None or subgroup_df.empty or not _has_required_columns(
            subgroup_df, REQUIRED_SUBGROUP_COLUMNS
        ):
            invalidators.append("Malformed artifacts/subgroup_ite.csv")
        else:
            if subgroup_df["subgroup_name"].astype(str).nunique() < 2:
                invalidators.append(
                    "Low-signal subgroup_ite.csv: at least 2 distinct subgroup_name values required"
                )
            n = pd.to_numeric(subgroup_df["n"], errors="coerce")
            mean = pd.to_numeric(subgroup_df["mean_ite_hat"], errors="coerce")
            std = pd.to_numeric(subgroup_df["std_ite_hat"], errors="coerce")
            if (
                n.isna().any()
                or (n <= 0).any()
                or mean.isna().any()
                or std.isna().any()
                or not np.isfinite(np.c_[mean.to_numpy(), std.to_numpy()]).all()
            ):
                invalidators.append(
                    "Malformed artifacts/subgroup_ite.csv: invalid numeric content"
                )

    if run_notes_txt is None:
        invalidators.append("Missing artifacts/run_notes.txt")
    else:
        notes = run_notes_txt.decode("utf-8") if isinstance(run_notes_txt, bytes) else run_notes_txt
        if len(notes.strip()) < MIN_NOTES_CHARS:
            invalidators.append(
                "Low-signal artifacts/run_notes.txt: must contain at least 100 characters"
            )
        lowered = notes.lower()
        for keyword in ["method", "validation", "overlap", "subgroup"]:
            if keyword not in lowered:
                invalidators.append(
                    f"Low-signal artifacts/run_notes.txt: missing keyword '{keyword}'"
                )

    return invalidators


def score_output_bundle(
    *,
    candidate_output_csv: str | bytes,
    reference_gold_csv: str | bytes,
    model_selection_csv: str | bytes | None = None,
    overlap_csv: str | bytes | None = None,
    subgroup_csv: str | bytes | None = None,
    run_notes_txt: str | bytes | None = None,
) -> ScoreResult:
    try:
        predictions = _load_csv(candidate_output_csv)
    except Exception as exc:
        return _score_failure("parse_error", "candidate_output_unreadable", error=str(exc))

    if list(predictions.columns) != REQUIRED_OUTPUT_COLUMNS:
        return _score_failure(
            "schema_mismatch",
            "candidate_output_columns",
            expected_columns=REQUIRED_OUTPUT_COLUMNS,
            observed_columns=list(predictions.columns),
        )
    predictions, key_error = _coerce_key_columns(predictions, label="candidate_output")
    if key_error is not None:
        return key_error

    if predictions.isna().any().any():
        return _score_failure("nan_values", "candidate_output_nan_values")

    numeric_predictions = predictions[["mu0_hat", "mu1_hat", "ite_hat"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if numeric_predictions.isna().any().any():
        return _score_failure("non_numeric_values", "candidate_output_non_numeric_values")
    if not np.isfinite(numeric_predictions.to_numpy()).all():
        return _score_failure("non_finite_values", "candidate_output_non_finite_values")
    predictions = predictions.copy()
    predictions[["mu0_hat", "mu1_hat", "ite_hat"]] = numeric_predictions

    if predictions.duplicated(subset=["replication", "unit_id"]).any():
        duplicates = predictions.loc[
            predictions.duplicated(subset=["replication", "unit_id"]),
            ["replication", "unit_id"],
        ].head(5)
        return _score_failure(
            "duplicate_prediction_keys",
            "candidate_duplicate_replication_unit_id",
            sample_duplicates=duplicates.to_dict("records"),
        )

    if not np.allclose(
        predictions["ite_hat"].to_numpy(),
        (predictions["mu1_hat"] - predictions["mu0_hat"]).to_numpy(),
        atol=CONSISTENCY_TOL,
    ):
        return _score_failure(
            "potential_outcome_inconsistency",
            "candidate_ite_consistency_failure",
        )

    try:
        gold = _load_csv(reference_gold_csv)
    except Exception as exc:  # pragma: no cover - hidden reference should remain valid
        return _score_failure("reference_parse_error", "reference_gold_unreadable", error=str(exc))

    if list(gold.columns) != REQUIRED_GOLD_COLUMNS:
        return _score_failure(
            "reference_schema_mismatch",
            "reference_gold_columns",
            expected_columns=REQUIRED_GOLD_COLUMNS,
            observed_columns=list(gold.columns),
        )
    gold, gold_key_error = _coerce_key_columns(gold, label="reference_gold")
    if gold_key_error is not None:
        return gold_key_error
    gold_numeric = gold[["mu0", "mu1"]].apply(pd.to_numeric, errors="coerce")
    if gold_numeric.isna().any().any() or not np.isfinite(gold_numeric.to_numpy()).all():
        return _score_failure("reference_non_numeric", "reference_gold_non_numeric")
    gold = gold.copy()
    gold[["mu0", "mu1"]] = gold_numeric

    try:
        merged = gold.merge(predictions, on=["replication", "unit_id"], how="outer", indicator=True)
    except Exception as exc:
        return _score_failure("prediction_merge_error", "candidate_reference_merge_error", error=str(exc))
    if not (merged["_merge"] == "both").all():
        return _score_failure(
            "prediction_key_mismatch",
            "candidate_missing_or_extra_rows",
            merge_counts=merged["_merge"].value_counts().to_dict(),
        )

    replication_scores: list[dict[str, Any]] = []
    for replication, replication_df in merged.groupby("replication", sort=True):
        true_ite = replication_df["mu1"] - replication_df["mu0"]
        pred_ite = replication_df["ite_hat"]
        replication_scores.append(
            {
                "replication": int(replication),
                "sqrt_epsilon_pehe": float(np.sqrt(np.mean((true_ite - pred_ite) ** 2))),
            }
        )

    mean_score = float(np.mean([entry["sqrt_epsilon_pehe"] for entry in replication_scores]))

    invalidators = _validate_artifacts(
        model_selection_csv=model_selection_csv,
        overlap_csv=overlap_csv,
        subgroup_csv=subgroup_csv,
        run_notes_txt=run_notes_txt,
    )

    replication_std = predictions.groupby("replication")["ite_hat"].std(ddof=0).fillna(0.0)
    if float((replication_std < CONSTANT_STD_TOL).mean()) > 0.90:
        invalidators.append(
            "Constant-effect collapse: >90% of replications have std(ite_hat) < 1e-3"
        )

    rounded = predictions["ite_hat"].round(ROUND_DECIMALS)
    if not rounded.empty and float(rounded.value_counts(normalize=True).iloc[0]) > 0.95:
        invalidators.append(
            "Constant-effect collapse: >95% of ite_hat values are identical after rounding to 6 decimals"
        )

    passed = mean_score <= PASS_THRESHOLD and not invalidators
    if passed:
        reason = "ok"
    elif invalidators:
        reason = invalidators[0]
    else:
        reason = f"threshold_miss: score {mean_score:.6f} > {PASS_THRESHOLD:.2f}"

    return ScoreResult(
        score=mean_score,
        passed=passed,
        reason=reason,
        hard_gate=None,
        details={
            "metric_name": "mean_sqrt_epsilon_PEHE",
            "pass_threshold": PASS_THRESHOLD,
            "leaderboard_anchor": LEADERBOARD_ANCHOR,
            "anchor_role": "leaderboard calibration reference only; not the pass bar",
            "lower_is_better": True,
            "computed_metric_available": True,
            "replication_scores": replication_scores,
            "invalidators": invalidators,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_dir)
    try:
        result = score_output_bundle(
            candidate_output_csv=(output_dir / "output.csv").read_text(encoding="utf-8"),
            reference_gold_csv=(reference_dir / "gold_output.csv").read_text(encoding="utf-8"),
            model_selection_csv=_read_text_if_exists(output_dir / "artifacts" / "model_selection.csv"),
            overlap_csv=_read_text_if_exists(output_dir / "artifacts" / "overlap_diagnostics.csv"),
            subgroup_csv=_read_text_if_exists(output_dir / "artifacts" / "subgroup_ite.csv"),
            run_notes_txt=_read_text_if_exists(output_dir / "artifacts" / "run_notes.txt"),
        )
    except OSError as exc:
        result = _score_failure("file_read_error", "required_file_missing_or_unreadable", error=str(exc))
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
