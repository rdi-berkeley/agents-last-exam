"""Scoring helpers for rgi_mcr1_colistin_v2."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SPECIFIC_MCR_PATTERN = re.compile(r"\bMCR-[A-Z0-9]", re.IGNORECASE)


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
    required_drug_keyword: str
    reported_resistance_mechanism: str | None
    required_resistance_mechanism_keyword: str
    pass_threshold: float

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
    if isinstance(value, bool):
        raise ValueError("booleans are not valid numeric values")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("numeric value must be finite")
    return parsed


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


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

    reference_gene = _stringify(gene_cfg.get("reference_value"))
    gene_prefix = str(gene_cfg["full_credit_prefix"]).strip().upper()
    partial_gene_hint = str(gene_cfg["partial_credit_contains"]).strip().lower()

    reference_identity = _coerce_float(identity_cfg["target"])
    full_credit_tolerance = _coerce_float(identity_cfg["full_credit_tolerance"])
    partial_credit_tolerance = _coerce_float(identity_cfg["partial_credit_tolerance"])

    reference_drug_class = _stringify(drug_cfg.get("reference_value"))
    required_drug_keyword = str(drug_cfg["required_keyword"]).strip().lower()

    reference_mechanism = _stringify(mechanism_cfg.get("reference_value"))
    required_mechanism_keyword = str(mechanism_cfg["required_keyword"]).strip().lower()

    pass_threshold = _coerce_float(score_mapping["pass_threshold"])

    try:
        output = _coerce_json_object(output_json_text, label="output_json")
    except ValueError as exc:
        return ScoreResult(
            score=0.0,
            passed=False,
            valid=False,
            reason=str(exc),
            gene_score=0.0,
            identity_score=0.0,
            drug_class_score=0.0,
            resistance_mechanism_score=0.0,
            reported_gene=None,
            reference_gene=reference_gene,
            reported_identity=None,
            reference_identity=reference_identity,
            reported_drug_class=None,
            required_drug_keyword=required_drug_keyword,
            reported_resistance_mechanism=None,
            required_resistance_mechanism_keyword=required_mechanism_keyword,
            pass_threshold=pass_threshold,
        )

    required_keys = (
        "best_hit_aro",
        "percent_identity",
        "drug_class",
        "resistance_mechanism",
    )
    missing_keys = [key for key in required_keys if key not in output]
    if missing_keys:
        return ScoreResult(
            score=0.0,
            passed=False,
            valid=False,
            reason="missing required keys: " + ", ".join(missing_keys),
            gene_score=0.0,
            identity_score=0.0,
            drug_class_score=0.0,
            resistance_mechanism_score=0.0,
            reported_gene=None,
            reference_gene=reference_gene,
            reported_identity=None,
            reference_identity=reference_identity,
            reported_drug_class=None,
            required_drug_keyword=required_drug_keyword,
            reported_resistance_mechanism=None,
            required_resistance_mechanism_keyword=required_mechanism_keyword,
            pass_threshold=pass_threshold,
        )

    reported_gene = _stringify(output.get("best_hit_aro"))
    reported_gene_stripped = (reported_gene or "").strip()
    reported_gene_normalized = reported_gene_stripped.upper()
    reported_gene_lower = reported_gene_stripped.lower()
    if SPECIFIC_MCR_PATTERN.search(reported_gene_stripped):
        gene_score = 1.0
    elif partial_gene_hint and partial_gene_hint in reported_gene_lower:
        gene_score = 0.5
    else:
        gene_score = 0.0

    try:
        reported_identity = _coerce_float(output.get("percent_identity"))
    except (TypeError, ValueError) as exc:
        return ScoreResult(
            score=0.0,
            passed=False,
            valid=False,
            reason=f"percent_identity is not numeric: {exc}",
            gene_score=gene_score,
            identity_score=0.0,
            drug_class_score=0.0,
            resistance_mechanism_score=0.0,
            reported_gene=reported_gene,
            reference_gene=reference_gene,
            reported_identity=None,
            reference_identity=reference_identity,
            reported_drug_class=_stringify(output.get("drug_class")),
            required_drug_keyword=required_drug_keyword,
            reported_resistance_mechanism=_stringify(output.get("resistance_mechanism")),
            required_resistance_mechanism_keyword=required_mechanism_keyword,
            pass_threshold=pass_threshold,
        )

    absolute_error = abs(reported_identity - reference_identity)
    if absolute_error <= full_credit_tolerance:
        identity_score = 1.0
    elif absolute_error <= partial_credit_tolerance:
        identity_score = 0.5
    else:
        identity_score = 0.0

    reported_drug_class = _stringify(output.get("drug_class"))
    drug_class_score = (
        1.0 if required_drug_keyword in (reported_drug_class or "").strip().lower() else 0.0
    )

    reported_mechanism = _stringify(output.get("resistance_mechanism"))
    resistance_mechanism_score = (
        1.0
        if required_mechanism_keyword in (reported_mechanism or "").strip().lower()
        else 0.0
    )

    final_score = (
        gene_score + identity_score + drug_class_score + resistance_mechanism_score
    ) / 4.0
    return ScoreResult(
        score=final_score,
        passed=final_score >= pass_threshold,
        valid=True,
        reason="scored successfully",
        gene_score=gene_score,
        identity_score=identity_score,
        drug_class_score=drug_class_score,
        resistance_mechanism_score=resistance_mechanism_score,
        reported_gene=reported_gene,
        reference_gene=reference_gene,
        reported_identity=reported_identity,
        reference_identity=reference_identity,
        reported_drug_class=reported_drug_class,
        required_drug_keyword=required_drug_keyword,
        reported_resistance_mechanism=reported_mechanism,
        required_resistance_mechanism_keyword=required_mechanism_keyword,
        pass_threshold=pass_threshold,
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
