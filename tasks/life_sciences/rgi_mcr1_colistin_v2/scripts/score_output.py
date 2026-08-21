"""Scoring helpers for rgi_mcr1_colistin_v2."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_IDENTITY_TOLERANCE = 0.05
CRITICAL_FAILURE_SCORE_CAP = 0.5


@dataclass
class ScoreResult:
    score: float
    passed: bool
    valid: bool
    reason: str
    gene_score: float
    identity_score: float
    drug_class_score: float
    resistance_mechanism_score: float
    reported_gene: str | None
    reference_gene: str | None
    reported_identity: float | None
    reference_identity: float | None
    reported_drug_class: str | None
    reference_drug_class: str
    reported_resistance_mechanism: str | None
    reference_resistance_mechanism: str
    identity_tolerance: float
    pass_threshold: float
    critical_fields_match: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_json_object(raw_text: str, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be a JSON number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("numeric value must be finite")
    return parsed


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _reference_text(value: Any, *, label: str) -> tuple[str, str]:
    if not isinstance(value, str) or not _normalize_text(value):
        raise ValueError(f"{label} must be a non-empty string")
    return value, _normalize_text(value)


def score_output_payloads(*, output_json_text: str, reference_json_text: str) -> ScoreResult:
    reference = _coerce_json_object(reference_json_text, label="reference_json")
    grading = _coerce_json_object(json.dumps(reference.get("grading", {})), label="grading")
    score_mapping = _coerce_json_object(
        json.dumps(reference.get("score_mapping", {})), label="score_mapping"
    )

    gene_cfg = _coerce_json_object(json.dumps(grading.get("gene_name", {})), label="gene_name")
    identity_cfg = _coerce_json_object(
        json.dumps(grading.get("percent_identity", {})),
        label="percent_identity",
    )
    drug_cfg = _coerce_json_object(json.dumps(grading.get("drug_class", {})), label="drug_class")
    mechanism_cfg = _coerce_json_object(
        json.dumps(grading.get("resistance_mechanism", {})),
        label="resistance_mechanism",
    )

    reference_gene, normalized_reference_gene = _reference_text(
        gene_cfg.get("reference_value"), label="gene_name.reference_value"
    )

    reference_identity = _coerce_float(identity_cfg["target"])
    configured_identity_tolerance = _coerce_float(identity_cfg["full_credit_tolerance"])
    if configured_identity_tolerance < 0:
        raise ValueError("percent_identity.full_credit_tolerance must be non-negative")
    identity_tolerance = min(configured_identity_tolerance, MAX_IDENTITY_TOLERANCE)

    reference_drug_class, normalized_reference_drug_class = _reference_text(
        drug_cfg.get("reference_value"), label="drug_class.reference_value"
    )

    reference_mechanism, normalized_reference_mechanism = _reference_text(
        mechanism_cfg.get("reference_value"),
        label="resistance_mechanism.reference_value",
    )

    pass_threshold = _coerce_float(score_mapping["pass_threshold"])
    if not 0.0 <= pass_threshold <= 1.0:
        raise ValueError("score_mapping.pass_threshold must be between 0 and 1")

    def invalid_result(
        reason: str,
        *,
        reported_gene: str | None = None,
        gene_score: float = 0.0,
        reported_drug_class: str | None = None,
        reported_mechanism: str | None = None,
    ) -> ScoreResult:
        return ScoreResult(
            score=0.0,
            passed=False,
            valid=False,
            reason=reason,
            gene_score=gene_score,
            identity_score=0.0,
            drug_class_score=0.0,
            resistance_mechanism_score=0.0,
            reported_gene=reported_gene,
            reference_gene=reference_gene,
            reported_identity=None,
            reference_identity=reference_identity,
            reported_drug_class=reported_drug_class,
            reference_drug_class=reference_drug_class,
            reported_resistance_mechanism=reported_mechanism,
            reference_resistance_mechanism=reference_mechanism,
            identity_tolerance=identity_tolerance,
            pass_threshold=pass_threshold,
            critical_fields_match=False,
        )

    try:
        output = _coerce_json_object(output_json_text, label="output_json")
    except ValueError as exc:
        return invalid_result(str(exc))

    required_keys = (
        "best_hit_aro",
        "percent_identity",
        "drug_class",
        "resistance_mechanism",
    )
    missing_keys = [key for key in required_keys if key not in output]
    if missing_keys:
        return invalid_result("missing required keys: " + ", ".join(missing_keys))

    text_fields = ("best_hit_aro", "drug_class", "resistance_mechanism")
    invalid_text_fields = [key for key in text_fields if not isinstance(output[key], str)]
    if invalid_text_fields:
        return invalid_result("fields must be strings: " + ", ".join(invalid_text_fields))

    reported_gene = output["best_hit_aro"]
    gene_score = float(_normalize_text(reported_gene) == normalized_reference_gene)

    try:
        reported_identity = _coerce_float(output.get("percent_identity"))
    except (TypeError, ValueError) as exc:
        return invalid_result(
            f"percent_identity is not numeric: {exc}",
            reported_gene=reported_gene,
            gene_score=gene_score,
            reported_drug_class=output["drug_class"],
            reported_mechanism=output["resistance_mechanism"],
        )

    absolute_error = abs(reported_identity - reference_identity)
    identity_score = float(absolute_error <= identity_tolerance)

    reported_drug_class = output["drug_class"]
    drug_class_score = float(
        _normalize_text(reported_drug_class) == normalized_reference_drug_class
    )

    reported_mechanism = output["resistance_mechanism"]
    resistance_mechanism_score = float(
        _normalize_text(reported_mechanism) == normalized_reference_mechanism
    )

    raw_score = (gene_score + identity_score + drug_class_score + resistance_mechanism_score) / 4.0
    critical_fields_match = gene_score == 1.0 and identity_score == 1.0
    final_score = raw_score if critical_fields_match else min(raw_score, CRITICAL_FAILURE_SCORE_CAP)
    return ScoreResult(
        score=final_score,
        passed=critical_fields_match and final_score >= pass_threshold,
        valid=True,
        reason=("scored successfully" if critical_fields_match else "critical field mismatch"),
        gene_score=gene_score,
        identity_score=identity_score,
        drug_class_score=drug_class_score,
        resistance_mechanism_score=resistance_mechanism_score,
        reported_gene=reported_gene,
        reference_gene=reference_gene,
        reported_identity=reported_identity,
        reference_identity=reference_identity,
        reported_drug_class=reported_drug_class,
        reference_drug_class=reference_drug_class,
        reported_resistance_mechanism=reported_mechanism,
        reference_resistance_mechanism=reference_mechanism,
        identity_tolerance=identity_tolerance,
        pass_threshold=pass_threshold,
        critical_fields_match=critical_fields_match,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score an rgi_mcr1_colistin_v2 answer.json artifact."
    )
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--reference-file", required=True)
    args = parser.parse_args()

    result = score_output_payloads(
        output_json_text=Path(args.answer_file).read_text(encoding="utf-8"),
        reference_json_text=Path(args.reference_file).read_text(encoding="utf-8"),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
