"""Local scorer for healthcare_readmission_diabetes."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


EXPECTED_PREDICTION_COLUMNS = ["encounter_id", "predicted_readmission_prob"]
EXPECTED_LABEL_COLUMNS = ["encounter_id", "readmitted_within_30_days"]
EXPECTED_METRIC_KEYS = {"auroc", "tpr_at_5pct_fpr"}
PASS_THRESHOLD = 0.66


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
        return pd.read_csv(io.StringIO(payload), dtype={"encounter_id": "string"})
    except Exception as exc:  # pragma: no cover - pandas raises varied parser errors
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


def _clean_id_series(series: pd.Series, *, label: str) -> tuple[pd.Series, str | None]:
    cleaned = series.astype("string").str.strip()
    if cleaned.isna().any() or (cleaned == "").any():
        return cleaned, f"{label}_empty_encounter_id"
    return cleaned, None


def _coerce_probabilities(series: pd.Series) -> tuple[pd.Series | None, str | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        return None, "invalid_probability_values"
    if ((numeric < 0.0) | (numeric > 1.0)).any():
        return None, "probability_out_of_range"
    return numeric.astype(float), None


def _coerce_binary_labels(series: pd.Series) -> tuple[pd.Series | None, str | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        return None, "invalid_label_values"
    if not numeric.isin([0, 1]).all():
        return None, "non_binary_labels"
    return numeric.astype(int), None


def _tpr_at_fixed_fpr(y_true: np.ndarray, y_prob: np.ndarray, fixed_fpr: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.interp(fixed_fpr, fpr, tpr))


def score_output_bundle(
    *,
    predictions_csv: str | bytes,
    metrics_json: str | bytes,
    labels_csv: str | bytes,
    pass_threshold: float = PASS_THRESHOLD,
) -> ScoreResult:
    try:
        predictions = _load_csv(predictions_csv, label="readmission_predictions.csv")
        labels = _load_csv(labels_csv, label="labels.csv")
        reported_metrics = _load_json(metrics_json, label="auroc_metrics.json")
    except ValueError as exc:
        return _hard_fail(str(exc))

    if list(predictions.columns) != EXPECTED_PREDICTION_COLUMNS:
        return _hard_fail(
            "wrong_prediction_columns",
            {"observed": list(predictions.columns), "expected": EXPECTED_PREDICTION_COLUMNS},
        )
    if list(labels.columns) != EXPECTED_LABEL_COLUMNS:
        return _hard_fail(
            "wrong_label_columns",
            {"observed": list(labels.columns), "expected": EXPECTED_LABEL_COLUMNS},
        )
    if set(reported_metrics) != EXPECTED_METRIC_KEYS:
        return _hard_fail(
            "wrong_metric_keys",
            {"observed": sorted(reported_metrics), "expected": sorted(EXPECTED_METRIC_KEYS)},
        )

    prediction_ids, prediction_id_error = _clean_id_series(
        predictions["encounter_id"], label="predictions"
    )
    if prediction_id_error is not None:
        return _hard_fail(prediction_id_error)
    label_ids, label_id_error = _clean_id_series(labels["encounter_id"], label="labels")
    if label_id_error is not None:
        return _hard_fail(label_id_error)

    if prediction_ids.duplicated().any():
        duplicates = sorted(prediction_ids[prediction_ids.duplicated()].unique().tolist())
        return _hard_fail("duplicate_prediction_rows", {"duplicates": duplicates[:10]})
    if label_ids.duplicated().any():
        duplicates = sorted(label_ids[label_ids.duplicated()].unique().tolist())
        return _hard_fail("duplicate_label_rows", {"duplicates": duplicates[:10]})

    prediction_probs, prob_error = _coerce_probabilities(predictions["predicted_readmission_prob"])
    if prob_error is not None:
        return _hard_fail(prob_error)
    label_values, label_error = _coerce_binary_labels(labels["readmitted_within_30_days"])
    if label_error is not None:
        return _hard_fail(label_error)

    observed = pd.DataFrame(
        {"encounter_id": prediction_ids, "predicted_readmission_prob": prediction_probs}
    )
    truth = pd.DataFrame(
        {"encounter_id": label_ids, "readmitted_within_30_days": label_values}
    )

    unknown_ids = sorted(set(observed["encounter_id"]) - set(truth["encounter_id"]))
    missing_ids = sorted(set(truth["encounter_id"]) - set(observed["encounter_id"]))
    if unknown_ids:
        return _hard_fail("unknown_encounter_ids", {"unknown_ids": unknown_ids[:10]})
    if missing_ids:
        return _hard_fail("missing_encounter_ids", {"missing_ids": missing_ids[:10]})

    merged = truth.merge(observed, on="encounter_id", how="inner")
    if merged.empty:
        return _hard_fail("empty_scored_merge")

    y_true = merged["readmitted_within_30_days"].to_numpy(dtype=int)
    y_prob = merged["predicted_readmission_prob"].to_numpy(dtype=float)

    try:
        computed_auroc = float(roc_auc_score(y_true, y_prob))
        computed_tpr = _tpr_at_fixed_fpr(y_true, y_prob)
    except Exception as exc:
        return _hard_fail("metric_computation_failed", {"error": str(exc)})

    passed = computed_auroc >= pass_threshold
    return ScoreResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        reason="passed" if passed else "auroc_below_threshold",
        hard_gate=None,
        details={
            "row_count": int(len(merged)),
            "auroc": computed_auroc,
            "tpr_at_5pct_fpr": computed_tpr,
            "pass_threshold": pass_threshold,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score healthcare_readmission_diabetes outputs.")
    parser.add_argument("--predictions", required=True, help="Path to readmission_predictions.csv")
    parser.add_argument("--metrics", required=True, help="Path to auroc_metrics.json")
    parser.add_argument("--labels", required=True, help="Path to hidden labels CSV")
    return parser.parse_args()


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


if __name__ == "__main__":
    args = _parse_args()
    result = score_output_bundle(
        predictions_csv=_read_text(args.predictions),
        metrics_json=_read_text(args.metrics),
        labels_csv=_read_text(args.labels),
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
