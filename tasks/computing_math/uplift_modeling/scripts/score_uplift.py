"""Scoring logic for the uplift modeling benchmark.

Compares agent output against reference data and returns a normalized 0-1 score.
Adapted from the submitter's evaluate_submission.py.
"""

import io
import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

REQUIRED_OUTPUT_FILES = {
    "feature_table": "feature_table.csv",
    "uplift_scores": "uplift_scores.csv",
    "feature_importance": "feature_importance_report.txt",
}

REFERENCE_FILES = {
    "feature_table": "reference_feature_table.csv",
    "uplift_scores": "reference_uplift_scores.csv",
    "metadata": "reference_metadata.json",
}

IMPUTATION_TOLERANCE = 0.01
SEGMENT_IOU_THRESHOLDS = [(0.85, 25), (0.75, 20), (0.65, 15), (0.50, 10)]
ITE_CORR_THRESHOLDS = [(0.70, 10), (0.60, 7), (0.50, 5)]
SEGMENT_SIZE_THRESHOLDS = [(0.5, 5), (1.0, 3), (2.0, 1)]

REQUIRED_FEATURE_COLS = [
    "user_id",
    "age",
    "tenure_days",
    "avg_duration",
    "total_duration",
    "avg_page_views",
    "total_page_views",
    "session_count",
]

EXPECTED_FEATURES = [
    "session_count",
    "avg_duration",
    "total_duration",
    "avg_page_views",
    "total_page_views",
]

INSIGHT_KEYWORDS = [
    "session",
    "engagement",
    "activity",
    "behavior",
    "persuadable",
    "moderate",
    "active",
    "interaction",
]


@dataclass
class ScoringReport:
    imputation_accuracy: float = 0.0
    column_correctness: float = 0.0
    split_reproducibility: float = 0.0
    segment_overlap: float = 0.0
    ite_correlation: float = 0.0
    segment_size: float = 0.0
    top5_overlap: float = 0.0
    insight_quality: float = 0.0
    details: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def raw_score(self) -> float:
        return (
            self.imputation_accuracy
            + self.column_correctness
            + self.split_reproducibility
            + self.segment_overlap
            + self.ite_correlation
            + self.segment_size
            + self.top5_overlap
            + self.insight_quality
        )

    @property
    def score(self) -> float:
        return max(0.0, min(1.0, self.raw_score / 100.0))

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "raw_score": self.raw_score,
            "max_score": 100,
            "breakdown": {
                "imputation_accuracy": self.imputation_accuracy,
                "column_correctness": self.column_correctness,
                "split_reproducibility": self.split_reproducibility,
                "segment_overlap": self.segment_overlap,
                "ite_correlation": self.ite_correlation,
                "segment_size": self.segment_size,
                "top5_overlap": self.top5_overlap,
                "insight_quality": self.insight_quality,
            },
            "details": self.details,
            "errors": self.errors,
        }


def _score_feature_table(
    sub_df: pd.DataFrame, ref_df: pd.DataFrame, report: ScoringReport
) -> None:
    if sub_df is None or ref_df is None:
        report.errors.append("Missing feature_table dataframe")
        return

    if "user_id" not in sub_df.columns:
        report.errors.append("Missing user_id column in feature_table")
        return

    sub_users = set(sub_df["user_id"].values)
    ref_users = set(ref_df["user_id"].values)
    overlap = len(sub_users & ref_users)
    report.details["user_id_overlap"] = f"{overlap}/{len(ref_users)}"

    merged = sub_df.merge(ref_df, on="user_id", suffixes=("_sub", "_ref"))

    imp_score = 0.0
    for col in ("avg_duration", "total_duration"):
        sub_col, ref_col = f"{col}_sub", f"{col}_ref"
        if sub_col in merged.columns and ref_col in merged.columns:
            diff = np.abs(merged[sub_col] - merged[ref_col])
            accuracy = (diff <= IMPUTATION_TOLERANCE).mean()
            report.details[f"{col}_accuracy"] = f"{accuracy * 100:.1f}%"
            if accuracy >= 0.99:
                imp_score += 7.5
    report.imputation_accuracy = imp_score

    missing_cols = [c for c in REQUIRED_FEATURE_COLS if c not in sub_df.columns]
    if not missing_cols:
        report.column_correctness = 15.0
        report.details["column_presence"] = "All required columns present"
    else:
        report.column_correctness = max(0.0, 15.0 - len(missing_cols) * 3.0)
        report.details["column_presence"] = f"Missing {len(missing_cols)} columns"
        report.errors.append(f"Missing columns: {missing_cols}")

    if len(sub_df) == len(ref_df):
        match_rate = len(sub_users & ref_users) / len(ref_users) if ref_users else 0
        report.details["user_id_match_rate"] = f"{match_rate * 100:.1f}%"
        if match_rate >= 0.99:
            report.split_reproducibility = 10.0
        elif match_rate >= 0.95:
            report.split_reproducibility = 7.0
        elif match_rate >= 0.90:
            report.split_reproducibility = 5.0
    else:
        report.errors.append(
            f"Row count mismatch: {len(sub_df)} vs {len(ref_df)}"
        )


def _score_uplift(
    sub_df: pd.DataFrame, ref_df: pd.DataFrame, report: ScoringReport
) -> None:
    if sub_df is None or ref_df is None:
        report.errors.append("Missing uplift_scores dataframe")
        return

    merged = sub_df.merge(ref_df, on="user_id", suffixes=("_sub", "_ref"))

    if "segment_sub" in merged.columns and "segment_ref" in merged.columns:
        sub_high = set(
            sub_df.loc[sub_df["segment"] == "high_uplift", "user_id"].values
        )
        ref_high = set(
            ref_df.loc[ref_df["segment"] == "high_uplift", "user_id"].values
        )
        intersection = len(sub_high & ref_high)
        union = len(sub_high | ref_high)
        iou = intersection / union if union > 0 else 0.0
        report.details["segment_iou"] = f"{iou:.3f}"

        report.segment_overlap = 5.0
        for threshold, pts in SEGMENT_IOU_THRESHOLDS:
            if iou >= threshold:
                report.segment_overlap = float(pts)
                break
    else:
        report.errors.append("Missing segment column in uplift_scores")

    if "ite_score_sub" in merged.columns and "ite_score_ref" in merged.columns:
        corr, p_val = spearmanr(merged["ite_score_sub"], merged["ite_score_ref"])
        report.details["ite_spearman"] = f"{corr:.3f} (p={p_val:.4f})"

        report.ite_correlation = 2.0
        for threshold, pts in ITE_CORR_THRESHOLDS:
            if corr >= threshold:
                report.ite_correlation = float(pts)
                break
    else:
        report.errors.append("Missing ite_score column in uplift_scores")

    if "segment" in sub_df.columns:
        high_count = (sub_df["segment"] == "high_uplift").sum()
        high_pct = (high_count / len(sub_df)) * 100 if len(sub_df) > 0 else 0
        deviation = abs(high_pct - 20.0)
        report.details["high_uplift_pct"] = f"{high_pct:.1f}%"

        for threshold, pts in SEGMENT_SIZE_THRESHOLDS:
            if deviation <= threshold:
                report.segment_size = float(pts)
                break


def _score_feature_importance(text: str, report: ScoringReport) -> None:
    if not text:
        report.errors.append("Missing feature importance report")
        return

    text_lower = text.lower()
    matches = [
        f
        for f in EXPECTED_FEATURES
        if f.replace("_", " ") in text_lower or f in text_lower
    ]
    report.details["matched_features"] = matches
    report.details["feature_overlap_count"] = f"{len(matches)}/5"

    if len(matches) >= 3:
        report.top5_overlap = 15.0
    elif len(matches) >= 2:
        report.top5_overlap = 10.0
    elif len(matches) >= 1:
        report.top5_overlap = 5.0

    kw_count = sum(1 for kw in INSIGHT_KEYWORDS if kw in text_lower)
    report.details["insight_keywords_found"] = kw_count
    if kw_count >= 3:
        report.insight_quality = 5.0
    elif kw_count >= 2:
        report.insight_quality = 3.0
    elif kw_count >= 1:
        report.insight_quality = 1.0


def score_submission(
    output_data: dict[str, bytes], reference_data: dict[str, bytes]
) -> ScoringReport:
    """Score agent output against reference data.

    Args:
        output_data: map of filename -> bytes for agent output files
        reference_data: map of filename -> bytes for reference files

    Returns:
        ScoringReport with normalized 0-1 score
    """
    report = ScoringReport()

    def _read_csv(data: dict, key: str) -> pd.DataFrame | None:
        raw = data.get(key)
        if raw is None:
            return None
        try:
            return pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            report.errors.append(f"Failed to parse {key}: {exc}")
            return None

    def _read_json(data: dict, key: str) -> dict | None:
        raw = data.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception as exc:
            report.errors.append(f"Failed to parse {key}: {exc}")
            return None

    def _read_text(data: dict, key: str) -> str | None:
        raw = data.get(key)
        if raw is None:
            return None
        return raw.decode("utf-8", errors="replace")

    sub_ft = _read_csv(output_data, "feature_table.csv")
    sub_us = _read_csv(output_data, "uplift_scores.csv")
    sub_fi = _read_text(output_data, "feature_importance_report.txt")

    ref_ft = _read_csv(reference_data, "reference_feature_table.csv")
    ref_us = _read_csv(reference_data, "reference_uplift_scores.csv")
    _read_json(reference_data, "reference_metadata.json")

    _score_feature_table(sub_ft, ref_ft, report)
    _score_uplift(sub_us, ref_us, report)
    _score_feature_importance(sub_fi, report)

    return report
