#!/usr/bin/env python
"""Deterministic scorer for healthcare_tcga_luad_survival_kras."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass

REQUIRED_FILES = [
    "cohort.csv",
    "km_plot.png",
    "cox_results.json",
    "analysis.R",
]

REQUIRED_COHORT_COLUMNS = [
    "patient_id",
    "sample_id",
    "file_id",
    "kras_expression_selected",
    "kras_group",
    "time_days",
    "status",
    "age_at_diagnosis_years",
    "raw_stage",
    "stage_group",
]


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reasons: list[str]
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reasons": self.reasons,
            "details": self.details,
        }


def _load_csv(payload: bytes, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    rows = list(reader)
    return fieldnames, rows


def _load_json(payload: bytes, label: str) -> dict:
    try:
        return json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _load_reference_rows(reference_cohort_csv: bytes) -> dict[str, dict[str, str]]:
    _, rows = _load_csv(reference_cohort_csv, "reference cohort.csv")
    return {row["patient_id"]: row for row in rows}


def _as_float(value: str | float | int | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _validate_png(payload: bytes, reasons: list[str], details: dict[str, object]) -> None:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        reasons.append("km_plot.png is not a valid PNG signature")
        return
    details["km_plot_size_bytes"] = len(payload)
    if len(payload) < 10_000:
        reasons.append("km_plot.png is too small")


def _validate_analysis_script(payload: bytes, reasons: list[str]) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        reasons.append("analysis.R is not valid UTF-8")
        return
    required_tokens = [
        "GDCquery",
        "GDCdownload",
        "survfit",
        "coxph",
        "cox.zph",
    ]
    missing = [token for token in required_tokens if token not in text]
    if missing:
        reasons.append(f"analysis.R missing required workflow markers: {missing}")


def _validate_cohort(
    payload: bytes,
    reference_cohort_csv: bytes,
    evaluation_contract_json: bytes,
    reasons: list[str],
    details: dict[str, object],
) -> None:
    fieldnames, rows = _load_csv(payload, "cohort.csv")
    for column in REQUIRED_COHORT_COLUMNS:
        if column not in fieldnames:
            reasons.append(f"cohort.csv missing column {column}")
    if reasons:
        return

    contract = _load_json(evaluation_contract_json, "evaluation_contract.json")
    details["cohort_rows"] = len(rows)
    if len(rows) < int(contract["minimum_patient_count"]):
        reasons.append("cohort.csv has too few patients")

    reference = _load_reference_rows(reference_cohort_csv)
    observed = {row["patient_id"]: row for row in rows}

    if set(observed) != set(reference):
        reasons.append("cohort.csv patient set does not match the hidden reference")
        return

    for patient_id in sorted(reference):
        ref_row = reference[patient_id]
        row = observed[patient_id]
        for column in (
            "sample_id",
            "file_id",
            "raw_stage",
            "kras_group",
            "stage_group",
            "status",
        ):
            if str(row[column]).strip() != str(ref_row[column]).strip():
                reasons.append(f"cohort.csv mismatch for {patient_id} column {column}")
                return
        for column, tolerance in (
            ("kras_expression_selected", 1e-6),
            ("time_days", 1e-6),
            ("age_at_diagnosis_years", 5e-3),
        ):
            lhs = _as_float(row[column])
            rhs = _as_float(ref_row[column])
            if lhs is None or rhs is None or abs(lhs - rhs) > tolerance:
                reasons.append(f"cohort.csv mismatch for {patient_id} column {column}")
                return


def _validate_cox_results(
    payload: bytes,
    reference_cox_json: bytes,
    reasons: list[str],
    details: dict[str, object],
) -> None:
    data = _load_json(payload, "cox_results.json")
    reference = _load_json(reference_cox_json, "reference cox_results.json")

    for key in ("cohort_summary", "log_rank", "cox_model", "ph_test"):
        if key not in data:
            reasons.append(f"cox_results.json missing key {key}")
    if reasons:
        return

    details["reported_patients"] = data["cohort_summary"].get("n_patients")
    for key, tolerance in (
        ("n_patients", 0.0),
        ("n_events", 0.0),
        ("n_kras_high", 0.0),
        ("n_kras_low", 0.0),
        ("median_kras_expression", 1e-6),
    ):
        lhs = _as_float(data["cohort_summary"].get(key))
        rhs = _as_float(reference["cohort_summary"].get(key))
        if lhs is None or rhs is None or abs(lhs - rhs) > tolerance:
            reasons.append(f"cox_results.json cohort_summary mismatch for {key}")
            return

    for key, tolerance in (("test_statistic", 1e-3), ("p_value", 1e-6)):
        lhs = _as_float(data["log_rank"].get(key))
        rhs = _as_float(reference["log_rank"].get(key))
        if lhs is None or rhs is None or abs(lhs - rhs) > tolerance:
            reasons.append(f"cox_results.json log_rank mismatch for {key}")
            return

    expected_variables = {
        "kras_group_high",
        "age_at_diagnosis_years",
        "stage_group_advanced",
    }
    observed_variables = set((data["cox_model"].get("coefficients") or {}).keys())
    if observed_variables != expected_variables:
        reasons.append("cox_results.json coefficient set mismatch")
        return
    for variable in sorted(expected_variables):
        observed = data["cox_model"]["coefficients"][variable]
        ref = reference["cox_model"]["coefficients"][variable]
        for key, tolerance in (
            ("hazard_ratio", 1e-3),
            ("p_value", 1e-6),
            ("ci_lower_95", 1e-3),
            ("ci_upper_95", 1e-3),
        ):
            lhs = _as_float(observed.get(key))
            rhs = _as_float(ref.get(key))
            if lhs is None or rhs is None or abs(lhs - rhs) > tolerance:
                reasons.append(f"cox_results.json mismatch for {variable}.{key}")
                return


def score_submission(
    outputs: dict[str, bytes],
    *,
    reference_cohort_csv: bytes,
    reference_cox_json: bytes,
    evaluation_contract_json: bytes,
) -> ScoreResult:
    reasons: list[str] = []
    details: dict[str, object] = {}

    for name in REQUIRED_FILES:
        payload = outputs.get(name)
        if payload is None:
            reasons.append(f"missing required file {name}")
            continue
        if len(payload) == 0:
            reasons.append(f"empty required file {name}")
    if reasons:
        return ScoreResult(score=0.0, passed=False, reasons=reasons, details=details)

    try:
        _validate_png(outputs["km_plot.png"], reasons, details)
        _validate_analysis_script(outputs["analysis.R"], reasons)
        _validate_cohort(
            outputs["cohort.csv"],
            reference_cohort_csv,
            evaluation_contract_json,
            reasons,
            details,
        )
        _validate_cox_results(
            outputs["cox_results.json"],
            reference_cox_json,
            reasons,
            details,
        )
    except Exception as exc:
        reasons.append(str(exc))

    passed = not reasons
    return ScoreResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        reasons=reasons,
        details=details,
    )
