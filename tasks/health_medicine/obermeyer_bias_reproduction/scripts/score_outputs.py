"""Deterministic scorer for the Obermeyer bias reproduction task."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any


REQUIRED_OUTPUT_COLUMNS = ["patient_id", "baseline_score", "revised_score"]
TOP_PERCENTILE = 0.03
BASELINE_TOL = 1e-8


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


def _to_float(value: str, *, field: str, row_id: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric {field} for patient_id={row_id!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"non-finite {field} for patient_id={row_id!r}")
    return number


def _top_indices(rows: list[dict[str, Any]], score_key: str, n_top: int) -> list[int]:
    return sorted(range(len(rows)), key=lambda i: rows[i][score_key], reverse=True)[:n_top]


def _black_fraction(rows: list[dict[str, Any]], score_key: str, n_top: int) -> float:
    indices = _top_indices(rows, score_key, n_top)
    return sum(rows[i]["race"] == "black" for i in indices) / n_top


def _help_rate(rows: list[dict[str, Any]], score_key: str, n_top: int) -> float:
    sickest = set(_top_indices(rows, "gagne_sum_t", n_top))
    selected = set(_top_indices(rows, score_key, n_top))
    return len(sickest & selected) / n_top


def _mean_top_health(rows: list[dict[str, Any]], score_key: str, n_top: int) -> float:
    indices = _top_indices(rows, score_key, n_top)
    return sum(rows[i]["gagne_sum_t"] for i in indices) / n_top


def _report_component(baseline_report: str, revised_report: str) -> float:
    baseline = baseline_report.lower()
    revised = revised_report.lower()
    baseline_terms = [
        "black",
        "white",
        "risk",
        "gagne",
        "cost",
        "similar",
        "decile",
    ]
    revised_terms = [
        "counterfactual",
        "revised",
        "high-risk",
        "threshold",
        "black",
        "white",
        "health",
    ]
    baseline_hits = sum(term in baseline for term in baseline_terms)
    revised_hits = sum(term in revised for term in revised_terms)
    baseline_numbers = len(re.findall(r"\d+(?:\.\d+)?%?", baseline))
    revised_numbers = len(re.findall(r"\d+(?:\.\d+)?%?", revised))
    baseline_numeric_component = min(1.0, baseline_numbers / 6)
    revised_numeric_component = min(1.0, revised_numbers / 5)
    term_component = (baseline_hits / len(baseline_terms) + revised_hits / len(revised_terms)) / 2
    numeric_component = (baseline_numeric_component + revised_numeric_component) / 2
    return min(1.0, 0.65 * term_component + 0.35 * numeric_component)


def _validate_reference_metrics(payload: str) -> dict[str, Any]:
    try:
        metrics = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"reference_metrics_json is malformed: {exc}") from exc
    required = {
        "n_rows",
        "top_n",
        "baseline_black_fraction",
        "revised_black_fraction",
        "baseline_help_rate",
        "revised_help_rate",
        "baseline_max_abs_diff",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"reference_metrics_json missing keys: {missing}")
    return metrics


def score_output_bundle(
    *,
    predictions_csv: str,
    analysis_data_csv: str,
    reference_metrics_json: str,
    reference_predictions_csv: str,
    baseline_report_md: str = "",
    revised_report_md: str = "",
) -> ScoreResult:
    """Score the agent output against the staged cohort and hidden reference metrics."""

    try:
        output_columns, pred_rows = _read_csv(predictions_csv)
        data_columns, data_rows = _read_csv(analysis_data_csv)
        reference_columns, reference_rows = _read_csv(reference_predictions_csv)
    except Exception as exc:
        return _hard_fail(f"csv_parse_error: {exc}")

    if output_columns != REQUIRED_OUTPUT_COLUMNS:
        return _hard_fail(
            "invalid_output_columns",
            {"expected": REQUIRED_OUTPUT_COLUMNS, "observed": output_columns},
        )
    if reference_columns != REQUIRED_OUTPUT_COLUMNS:
        return _hard_fail(
            "invalid_reference_prediction_columns",
            {"expected": REQUIRED_OUTPUT_COLUMNS, "observed": reference_columns},
        )

    try:
        reference_metrics = _validate_reference_metrics(reference_metrics_json)
    except ValueError as exc:
        return _hard_fail(str(exc))

    required_data_cols = {"patient_id", "risk_score_t", "gagne_sum_t", "race"}
    missing_data_cols = sorted(required_data_cols - set(data_columns))
    if missing_data_cols:
        return _hard_fail("missing_required_input_columns", {"missing": missing_data_cols})

    if len(pred_rows) != len(data_rows):
        return _hard_fail(
            "row_count_mismatch",
            {"predictions": len(pred_rows), "analysis_data": len(data_rows)},
        )

    data_by_id: dict[str, dict[str, str]] = {}
    for idx, row in enumerate(data_rows, start=1):
        patient_id = (row.get("patient_id") or str(idx)).strip()
        if patient_id in data_by_id:
            return _hard_fail("duplicate_patient_id_in_input", {"patient_id": patient_id})
        data_by_id[patient_id] = row

    combined: list[dict[str, Any]] = []
    seen_output_ids: set[str] = set()
    try:
        for row in pred_rows:
            patient_id = (row.get("patient_id") or "").strip()
            if not patient_id:
                return _hard_fail("missing_patient_id_in_output")
            if patient_id in seen_output_ids:
                return _hard_fail("duplicate_patient_id_in_output", {"patient_id": patient_id})
            if patient_id not in data_by_id:
                return _hard_fail("unexpected_patient_id_in_output", {"patient_id": patient_id})
            seen_output_ids.add(patient_id)
            data_row = data_by_id[patient_id]
            combined.append(
                {
                    "patient_id": patient_id,
                    "baseline_score": _to_float(
                        row.get("baseline_score", ""), field="baseline_score", row_id=patient_id
                    ),
                    "revised_score": _to_float(
                        row.get("revised_score", ""), field="revised_score", row_id=patient_id
                    ),
                    "risk_score_t": _to_float(
                        data_row.get("risk_score_t", ""), field="risk_score_t", row_id=patient_id
                    ),
                    "gagne_sum_t": _to_float(
                        data_row.get("gagne_sum_t", ""), field="gagne_sum_t", row_id=patient_id
                    ),
                    "race": (data_row.get("race") or "").strip().lower(),
                }
            )
    except ValueError as exc:
        return _hard_fail(f"numeric_validation_error: {exc}")

    missing_output_ids = sorted(set(data_by_id) - seen_output_ids)
    if missing_output_ids:
        return _hard_fail("missing_patient_ids_in_output", {"count": len(missing_output_ids)})

    baseline_max_abs_diff = max(
        abs(row["baseline_score"] - row["risk_score_t"]) for row in combined
    )
    if baseline_max_abs_diff > BASELINE_TOL:
        return _hard_fail(
            "baseline_score_must_equal_observed_risk_score_t",
            {"baseline_max_abs_diff": baseline_max_abs_diff},
        )

    n_top = max(1, int(len(combined) * TOP_PERCENTILE))
    if int(reference_metrics["n_rows"]) != len(combined) or int(reference_metrics["top_n"]) != n_top:
        return _hard_fail(
            "reference_metrics_shape_mismatch",
            {
                "reference_n_rows": reference_metrics["n_rows"],
                "observed_n_rows": len(combined),
                "reference_top_n": reference_metrics["top_n"],
                "observed_top_n": n_top,
            },
        )

    baseline_black = _black_fraction(combined, "baseline_score", n_top)
    revised_black = _black_fraction(combined, "revised_score", n_top)
    baseline_help = _help_rate(combined, "baseline_score", n_top)
    revised_help = _help_rate(combined, "revised_score", n_top)
    baseline_top_health = _mean_top_health(combined, "baseline_score", n_top)
    revised_top_health = _mean_top_health(combined, "revised_score", n_top)

    if abs(baseline_black - float(reference_metrics["baseline_black_fraction"])) > 1e-9:
        return _hard_fail(
            "baseline_black_fraction_mismatch",
            {
                "observed": baseline_black,
                "reference": reference_metrics["baseline_black_fraction"],
            },
        )
    if abs(baseline_help - float(reference_metrics["baseline_help_rate"])) > 1e-9:
        return _hard_fail(
            "baseline_help_rate_mismatch",
            {"observed": baseline_help, "reference": reference_metrics["baseline_help_rate"]},
        )

    if revised_help + 1e-12 < baseline_help:
        return _hard_fail(
            "revised_help_rate_below_baseline",
            {"baseline_help_rate": baseline_help, "revised_help_rate": revised_help},
        )

    reference_by_id = {row["patient_id"].strip(): row for row in reference_rows}
    if set(reference_by_id) != {row["patient_id"] for row in combined}:
        return _hard_fail("reference_prediction_patient_ids_mismatch")
    reference_combined = []
    for row in combined:
        ref_row = reference_by_id[row["patient_id"]]
        try:
            reference_combined.append(
                {
                    "patient_id": row["patient_id"],
                    "revised_score": _to_float(
                        ref_row.get("revised_score", ""),
                        field="reference.revised_score",
                        row_id=row["patient_id"],
                    ),
                }
            )
        except ValueError as exc:
            return _hard_fail(f"reference_numeric_validation_error: {exc}")
    reference_top = set(_top_indices(reference_combined, "revised_score", n_top))
    revised_top = set(_top_indices(combined, "revised_score", n_top))
    reference_top_overlap = len(reference_top & revised_top) / n_top

    max_allowed_black = float(reference_metrics["revised_black_fraction"]) + 0.15
    if revised_black > max_allowed_black:
        return _hard_fail(
            "revised_black_fraction_implausibly_high",
            {
                "observed": revised_black,
                "max_allowed": max_allowed_black,
                "reference": reference_metrics["revised_black_fraction"],
            },
        )

    if reference_top_overlap < 0.50:
        return _hard_fail(
            "revised_top_set_too_far_from_hidden_reference",
            {"reference_top_overlap": reference_top_overlap},
        )

    black_improvement = revised_black - baseline_black
    black_component = 1.0 if revised_black >= 0.40 and black_improvement >= 0.15 else max(
        0.0, min(1.0, black_improvement / 0.15)
    )

    help_component = 1.0

    baseline_top = set(_top_indices(combined, "baseline_score", n_top))
    ranking_checks = [
        len({row["revised_score"] for row in combined}) > 1,
        revised_top != baseline_top,
        revised_top_health >= baseline_top_health,
    ]
    ranking_component = sum(ranking_checks) / len(ranking_checks)
    reference_component = min(1.0, reference_top_overlap / 0.90)

    report_component = _report_component(baseline_report_md, revised_report_md)

    score = (
        0.30 * black_component
        + 0.20 * help_component
        + 0.20 * report_component
        + 0.15 * ranking_component
        + 0.15 * reference_component
    )
    score = round(min(1.0, max(0.0, score)), 6)

    details = {
        "n_rows": len(combined),
        "top_n": n_top,
        "baseline_max_abs_diff": baseline_max_abs_diff,
        "baseline_black_fraction": baseline_black,
        "revised_black_fraction": revised_black,
        "black_fraction_improvement": black_improvement,
        "baseline_help_rate": baseline_help,
        "revised_help_rate": revised_help,
        "baseline_top_health": baseline_top_health,
        "revised_top_health": revised_top_health,
        "black_component": black_component,
        "help_component": help_component,
        "ranking_component": ranking_component,
        "reference_component": reference_component,
        "reference_top_overlap": reference_top_overlap,
        "report_component": report_component,
        "reference_metrics": reference_metrics,
    }

    passed = score >= 0.85
    reason = "passed" if passed else "score_below_threshold"
    return ScoreResult(score, passed, reason, None, details)
