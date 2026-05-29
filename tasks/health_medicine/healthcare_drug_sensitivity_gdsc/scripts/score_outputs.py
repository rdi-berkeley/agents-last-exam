"""Local scorer for healthcare_drug_sensitivity_gdsc."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


EXPECTED_PREDICTION_COLUMNS = ["COSMIC_ID", "DRUG_ID", "predicted_LN_IC50"]
EXPECTED_METRIC_KEYS = {"pearson_r", "rmse", "row_count"}
REQUIRED_HIDDEN_COLUMNS = {"COSMIC_ID", "DRUG_ID", "LN_IC50"}
PASS_THRESHOLD = 0.82
METRIC_TOL = 1e-6


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
    return ScoreResult(score=0.0, passed=False, reason=reason, hard_gate=reason, details=details or {})


def _load_csv(text: str | bytes, *, label: str) -> pd.DataFrame:
    payload = text.decode("utf-8-sig") if isinstance(text, bytes) else text
    try:
        return pd.read_csv(io.StringIO(payload))
    except Exception as exc:
        raise ValueError(f"{label} unreadable: {exc}") from exc


def _load_json(text: str | bytes, *, label: str) -> dict[str, Any]:
    payload = text.decode("utf-8-sig") if isinstance(text, bytes) else text
    try:
        parsed = json.loads(payload)
    except Exception as exc:
        raise ValueError(f"{label} unreadable: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to an object")
    return parsed


def _clean_id_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    cleaned = df.copy()
    for column in ["COSMIC_ID", "DRUG_ID"]:
        raw = cleaned[column].astype("string").str.strip()
        if raw.isna().any() or (raw == "").any():
            return cleaned, f"invalid_{column.lower()}"
        if not raw.str.fullmatch(r"[+-]?\d+").all():
            return cleaned, f"non_integer_{column.lower()}"
        values = pd.to_numeric(raw, errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            return cleaned, f"invalid_{column.lower()}"
        cleaned[column] = values.astype(np.int64)
    return cleaned, None


def _coerce_predictions(series: pd.Series) -> tuple[pd.Series | None, str | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        return None, "invalid_prediction_values"
    return numeric.astype(float), None


def _coerce_truth(series: pd.Series) -> tuple[pd.Series | None, str | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        return None, "invalid_truth_values"
    return numeric.astype(float), None


def _normalize_metric(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("metric_not_finite")
    return numeric


def _parse_reported_row_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("row_count_not_integer")
    return value


def _same_optional_float(left: float | None, right: float | None, tol: float) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) <= tol


def score_output_bundle(
    *,
    predictions_csv: str | bytes,
    metrics_json: str | bytes,
    hidden_test_csv: str | bytes,
    pass_threshold: float = PASS_THRESHOLD,
) -> ScoreResult:
    try:
        predictions = _load_csv(predictions_csv, label="predictions.csv")
        hidden_test = _load_csv(hidden_test_csv, label="hidden_test.csv")
        reported_metrics = _load_json(metrics_json, label="correlation.json")
    except ValueError as exc:
        return _hard_fail(str(exc))

    if list(predictions.columns) != EXPECTED_PREDICTION_COLUMNS:
        return _hard_fail(
            "wrong_prediction_columns",
            {"observed": list(predictions.columns), "expected": EXPECTED_PREDICTION_COLUMNS},
        )
    if not REQUIRED_HIDDEN_COLUMNS.issubset(hidden_test.columns):
        return _hard_fail(
            "missing_hidden_columns",
            {"observed": list(hidden_test.columns), "required": sorted(REQUIRED_HIDDEN_COLUMNS)},
        )
    if set(reported_metrics) != EXPECTED_METRIC_KEYS:
        return _hard_fail(
            "wrong_metric_keys",
            {"observed": sorted(reported_metrics), "expected": sorted(EXPECTED_METRIC_KEYS)},
        )

    predictions, prediction_id_error = _clean_id_frame(predictions[EXPECTED_PREDICTION_COLUMNS])
    if prediction_id_error is not None:
        return _hard_fail(prediction_id_error)
    hidden_test, hidden_id_error = _clean_id_frame(hidden_test[["COSMIC_ID", "DRUG_ID", "LN_IC50"]])
    if hidden_id_error is not None:
        return _hard_fail(hidden_id_error)

    if predictions[["COSMIC_ID", "DRUG_ID"]].duplicated().any():
        return _hard_fail("duplicate_prediction_rows")
    if hidden_test[["COSMIC_ID", "DRUG_ID"]].duplicated().any():
        return _hard_fail("duplicate_hidden_rows")

    pred_values, pred_error = _coerce_predictions(predictions["predicted_LN_IC50"])
    if pred_error is not None:
        return _hard_fail(pred_error)
    truth_values, truth_error = _coerce_truth(hidden_test["LN_IC50"])
    if truth_error is not None:
        return _hard_fail(truth_error)
    predictions["predicted_LN_IC50"] = pred_values
    hidden_test["LN_IC50"] = truth_values

    merged = hidden_test.merge(predictions, on=["COSMIC_ID", "DRUG_ID"], how="outer", indicator=True)
    if (merged["_merge"] != "both").any():
        missing_truth = int((merged["_merge"] == "right_only").sum())
        missing_predictions = int((merged["_merge"] == "left_only").sum())
        return _hard_fail(
            "id_mismatch",
            {
                "missing_truth_rows": missing_truth,
                "missing_prediction_rows": missing_predictions,
            },
        )

    merged = merged.drop(columns="_merge")
    y_true = merged["LN_IC50"].to_numpy(dtype=float)
    y_pred = merged["predicted_LN_IC50"].to_numpy(dtype=float)
    computed_row_count = int(len(merged))

    pearson_value = pearsonr(y_pred, y_true).statistic
    computed_pearson = None if np.isnan(pearson_value) else float(pearson_value)
    computed_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    try:
        reported_pearson = _normalize_metric(reported_metrics["pearson_r"])
        reported_rmse = _normalize_metric(reported_metrics["rmse"])
        reported_row_count = _parse_reported_row_count(reported_metrics["row_count"])
    except (TypeError, ValueError) as exc:
        return _hard_fail("reported_metric_invalid", {"error": str(exc)})

    if reported_row_count != computed_row_count:
        return _hard_fail(
            "reported_row_count_mismatch",
            {"reported": reported_row_count, "computed": computed_row_count},
        )
    if not _same_optional_float(reported_pearson, computed_pearson, METRIC_TOL) or abs(reported_rmse - computed_rmse) > METRIC_TOL:
        return _hard_fail(
            "reported_metrics_mismatch",
            {
                "reported_pearson_r": reported_pearson,
                "computed_pearson_r": computed_pearson,
                "reported_rmse": reported_rmse,
                "computed_rmse": computed_rmse,
                "metric_tol": METRIC_TOL,
            },
        )

    passed = computed_pearson is not None and computed_pearson >= pass_threshold
    reason = "passed" if passed else ("pearson_undefined" if computed_pearson is None else "pearson_below_threshold")
    return ScoreResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        reason=reason,
        hard_gate=None,
        details={
            "row_count": computed_row_count,
            "pearson_r": computed_pearson,
            "rmse": computed_rmse,
            "pass_threshold": pass_threshold,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score healthcare_drug_sensitivity_gdsc outputs.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--hidden-test", required=True)
    return parser.parse_args()


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


if __name__ == "__main__":
    args = _parse_args()
    result = score_output_bundle(
        predictions_csv=_read_text(args.predictions),
        metrics_json=_read_text(args.metrics),
        hidden_test_csv=_read_text(args.hidden_test),
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
