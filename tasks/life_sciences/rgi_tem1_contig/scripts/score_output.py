"""Scoring helpers for rgi_tem1_contig."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REQUIRED_DRUG_CLASS_SUBSTRINGS = ["cephalosporin", "penicillin", "monobactam"]


@dataclass
class ScoreResult:
    score: float
    valid: bool
    reason: str
    gene_score: float
    identity_score: float
    drug_class_score: float
    reported_gene: str | None
    reference_gene: str | None
    reported_identity: float | None
    reference_identity: float | None
    required_drug_terms: list[str]
    matched_drug_terms: list[str]

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
    return float(value)


def _normalize_gene_family(reference_gene: Any) -> tuple[str | None, str | None]:
    if not isinstance(reference_gene, str):
        return None, None
    trimmed = reference_gene.strip()
    if not trimmed:
        return None, None
    family = trimmed.split("-", 1)[0].upper()
    return trimmed, family


def _normalize_drug_terms(value: Any) -> list[str]:
    if isinstance(value, list):
        terms = [str(item).strip().lower() for item in value if str(item).strip()]
    elif isinstance(value, str):
        if ";" in value:
            terms = [item.strip().lower() for item in value.split(";") if item.strip()]
        else:
            stripped = value.strip().lower()
            terms = [stripped] if stripped else []
    else:
        terms = []
    seen: list[str] = []
    for term in terms:
        if term not in seen:
            seen.append(term)
    return seen


def _drug_text(value: Any) -> str:
    if isinstance(value, list):
        return " ; ".join(str(item) for item in value)
    return str(value)


def score_output_payloads(*, output_json_text: str, reference_json_text: str) -> ScoreResult:
    reference = _coerce_json_object(reference_json_text, label="reference_json")
    reference_gene, family_prefix = _normalize_gene_family(reference.get("best_hit_aro"))
    reference_identity = _coerce_float(reference["percent_identity"])
    reference_terms = _normalize_drug_terms(reference.get("drug_classes"))
    if not reference_terms:
        raise ValueError("reference_json missing required drug class terms")
    required_terms = list(REQUIRED_DRUG_CLASS_SUBSTRINGS)

    try:
        output = _coerce_json_object(output_json_text, label="output_json")
    except ValueError as exc:
        return ScoreResult(
            score=0.0,
            valid=False,
            reason=str(exc),
            gene_score=0.0,
            identity_score=0.0,
            drug_class_score=0.0,
            reported_gene=None,
            reference_gene=reference_gene,
            reported_identity=None,
            reference_identity=reference_identity,
            required_drug_terms=required_terms,
            matched_drug_terms=[],
        )

    missing_keys = [key for key in ("best_hit_aro", "percent_identity", "drug_classes") if key not in output]
    if missing_keys:
        return ScoreResult(
            score=0.0,
            valid=False,
            reason="missing required keys: " + ", ".join(missing_keys),
            gene_score=0.0,
            identity_score=0.0,
            drug_class_score=0.0,
            reported_gene=None,
            reference_gene=reference_gene,
            reported_identity=None,
            reference_identity=reference_identity,
            required_drug_terms=required_terms,
            matched_drug_terms=[],
        )

    reported_gene = output.get("best_hit_aro")
    gene_score = 0.0
    if isinstance(reported_gene, str) and reported_gene.strip() and family_prefix:
        if reported_gene.strip().upper().startswith(family_prefix):
            gene_score = 1.0

    try:
        reported_identity = _coerce_float(output.get("percent_identity"))
    except (TypeError, ValueError):
        return ScoreResult(
            score=0.0,
            valid=False,
            reason="percent_identity is not numeric",
            gene_score=gene_score,
            identity_score=0.0,
            drug_class_score=0.0,
            reported_gene=reported_gene if isinstance(reported_gene, str) else None,
            reference_gene=reference_gene,
            reported_identity=None,
            reference_identity=reference_identity,
            required_drug_terms=required_terms,
            matched_drug_terms=[],
        )

    absolute_error = abs(reported_identity - reference_identity)
    if absolute_error <= 0.5:
        identity_score = 1.0
    elif absolute_error <= 2.0:
        identity_score = 0.5
    else:
        identity_score = 0.0

    drug_text = _drug_text(output.get("drug_classes")).lower()
    matched_terms = [term for term in required_terms if term in drug_text]
    drug_class_score = len(matched_terms) / len(required_terms)

    final_score = (gene_score + identity_score + drug_class_score) / 3.0
    return ScoreResult(
        score=final_score,
        valid=True,
        reason="scored successfully",
        gene_score=gene_score,
        identity_score=identity_score,
        drug_class_score=drug_class_score,
        reported_gene=reported_gene if isinstance(reported_gene, str) else str(reported_gene),
        reference_gene=reference_gene,
        reported_identity=reported_identity,
        reference_identity=reference_identity,
        required_drug_terms=required_terms,
        matched_drug_terms=matched_terms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an rgi_tem1_contig answer.json artifact.")
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
