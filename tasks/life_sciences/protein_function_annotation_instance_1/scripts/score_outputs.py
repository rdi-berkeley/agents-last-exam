"""Scoring helpers for protein_function_annotation_instance_1."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REQUIRED_DOMAIN_COLUMNS = [
    "sequence_id",
    "interpro_accession",
    "interpro_name",
    "start",
    "end",
    "e_value",
]
REQUIRED_GO_COLUMNS = [
    "go_id",
    "go_name",
    "go_namespace",
    "source_interpro_accession",
]
LOCATION_PATTERN = re.compile(
    r"\b(cytoplasm|cytoplasmic|nucleus|nuclear|spindle|kinetochore|microtubule|microtubules)\b"
)


@dataclass
class ScoreReport:
    score: float
    passed: bool
    hard_fail_reason: str | None
    domains_match: bool
    go_match: bool
    summary_pass: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    replacements = {
        "\u03b3": "gamma",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _sentence_count(summary: str) -> int:
    stripped = summary.strip()
    if not stripped:
        return 0
    sentences = [part.strip() for part in re.split(r"[.!?]+", stripped) if part.strip()]
    return len(sentences) if sentences else 1


def _missing_columns(fieldnames: list[str] | None, required: list[str]) -> list[str]:
    present = fieldnames or []
    return [column for column in required if column not in present]


def _normalize_domains(text: str) -> tuple[list[tuple[Any, ...]], str | None, dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames != REQUIRED_DOMAIN_COLUMNS:
        return [], (
            "interpro_domains.tsv must use exactly these columns in order: "
            + ", ".join(REQUIRED_DOMAIN_COLUMNS)
        ), {"fieldnames": reader.fieldnames or []}
    missing = _missing_columns(reader.fieldnames, REQUIRED_DOMAIN_COLUMNS)
    if missing:
        return [], f"interpro_domains.tsv missing columns: {', '.join(missing)}", {"missing_columns": missing}

    rows: list[tuple[Any, ...]] = []
    raw_rows: list[tuple[Any, ...]] = []
    for index, row in enumerate(reader, start=2):
        try:
            sequence_id = (row.get("sequence_id") or "").strip()
            accession = (row.get("interpro_accession") or "").strip()
            name = (row.get("interpro_name") or "").strip()
            start = int((row.get("start") or "").strip())
            end = int((row.get("end") or "").strip())
            raw_e_value = (row.get("e_value") or "").strip()
            e_value = float(raw_e_value)
        except Exception as exc:
            return [], f"interpro_domains.tsv row {index} is invalid: {exc}", {"bad_row": index}
        if not sequence_id or not accession or not name:
            return [], f"interpro_domains.tsv row {index} has empty required fields", {"bad_row": index}
        if not re.fullmatch(r"[+-]?\d+\.\d{2}", raw_e_value):
            return [], (
                f"interpro_domains.tsv row {index} must format e_value with exactly 2 decimal places"
            ), {"bad_row": index, "raw_e_value": raw_e_value}
        normalized = (sequence_id, accession, name, start, end, f"{e_value:.2f}")
        rows.append(normalized)
        raw_rows.append(normalized)

    if not rows:
        return [], "interpro_domains.tsv is empty", {"row_count": 0}
    sorted_rows = sorted(rows, key=lambda item: (item[3], item[1]))
    if raw_rows != sorted_rows:
        return [], (
            "interpro_domains.tsv must already be sorted by start ascending and then interpro_accession ascending"
        ), {"row_count": len(rows)}
    return sorted_rows, None, {"row_count": len(rows)}


def _normalize_go_terms(text: str) -> tuple[list[tuple[str, str, str, str]], str | None, dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames != REQUIRED_GO_COLUMNS:
        return [], (
            "go_terms.tsv must use exactly these columns in order: "
            + ", ".join(REQUIRED_GO_COLUMNS)
        ), {"fieldnames": reader.fieldnames or []}
    missing = _missing_columns(reader.fieldnames, REQUIRED_GO_COLUMNS)
    if missing:
        return [], f"go_terms.tsv missing columns: {', '.join(missing)}", {"missing_columns": missing}

    raw_rows: list[tuple[str, str, str, str]] = []
    for index, row in enumerate(reader, start=2):
        values = tuple((row.get(column) or "").strip() for column in REQUIRED_GO_COLUMNS)
        if any(not value for value in values):
            return [], f"go_terms.tsv row {index} has empty required fields", {"bad_row": index}
        raw_rows.append(values)

    if not raw_rows:
        return [], "go_terms.tsv is empty", {"row_count": 0}
    if len(set(raw_rows)) != len(raw_rows):
        return [], "go_terms.tsv must not contain exact duplicate rows", {"row_count": len(raw_rows)}
    rows = sorted(raw_rows, key=lambda item: (item[2], item[0], item[3], item[1]))
    return rows, None, {"row_count": len(rows)}


def _summary_report(summary_text: str) -> tuple[bool, dict[str, Any]]:
    normalized = _normalize_text(summary_text)
    sentence_count = _sentence_count(summary_text)
    has_gamma_tubulin = bool(re.search(r"\bgamma[- ]tubulin\b", normalized))
    has_nucleation = bool(re.search(r"\bnucleat(?:e|es|ed|ing|ion)\b", normalized))
    has_minus_end = bool(re.search(r"\bminus[- ]ends?\b", normalized))
    has_initiation_equivalent = bool(
        re.search(r"\b(initiat(?:e|es|ed|ing|ion)|cap(?:s|ped|ping)?|template(?:s)?|seed(?:s|ed|ing)?)\b", normalized)
    )
    has_location = bool(LOCATION_PATTERN.search(normalized))

    details = {
        "sentence_count": sentence_count,
        "has_gamma_tubulin": has_gamma_tubulin,
        "has_nucleation": has_nucleation,
        "has_minus_end": has_minus_end,
        "has_initiation_equivalent": has_initiation_equivalent,
        "has_location": has_location,
    }
    passed = (
        1 <= sentence_count <= 2
        and has_gamma_tubulin
        and has_nucleation
        and has_minus_end
        and has_initiation_equivalent
        and has_location
    )
    return passed, details


def score_output_payloads(
    *,
    agent_interpro_tsv: str,
    agent_go_tsv: str,
    agent_summary_text: str,
    reference_interpro_tsv: str,
    reference_go_tsv: str,
    reference_summary_text: str,
) -> ScoreReport:
    agent_domains, domains_error, agent_domain_meta = _normalize_domains(agent_interpro_tsv)
    if domains_error:
        return ScoreReport(
            score=0.0,
            passed=False,
            hard_fail_reason=domains_error,
            domains_match=False,
            go_match=False,
            summary_pass=False,
            details={"agent_domains": agent_domain_meta},
        )

    agent_go, go_error, agent_go_meta = _normalize_go_terms(agent_go_tsv)
    if go_error:
        return ScoreReport(
            score=0.0,
            passed=False,
            hard_fail_reason=go_error,
            domains_match=False,
            go_match=False,
            summary_pass=False,
            details={"agent_domains": agent_domain_meta, "agent_go": agent_go_meta},
        )

    if not (agent_summary_text or "").strip():
        return ScoreReport(
            score=0.0,
            passed=False,
            hard_fail_reason="functional_summary.txt is empty",
            domains_match=False,
            go_match=False,
            summary_pass=False,
            details={"agent_domains": agent_domain_meta, "agent_go": agent_go_meta},
        )

    reference_domains, reference_domains_error, reference_domain_meta = _normalize_domains(reference_interpro_tsv)
    reference_go, reference_go_error, reference_go_meta = _normalize_go_terms(reference_go_tsv)
    if reference_domains_error or reference_go_error:
        raise ValueError(
            "reference artifacts are invalid: "
            + "; ".join(item for item in [reference_domains_error, reference_go_error] if item)
        )

    domains_match = agent_domains == reference_domains
    go_match = agent_go == reference_go
    summary_pass, summary_meta = _summary_report(agent_summary_text)
    reference_summary_pass, reference_summary_meta = _summary_report(reference_summary_text)

    passed = domains_match and go_match and summary_pass
    details = {
        "agent_domains": agent_domain_meta,
        "reference_domains": reference_domain_meta,
        "agent_go": agent_go_meta,
        "reference_go": reference_go_meta,
        "summary": summary_meta,
        "reference_summary": reference_summary_meta,
        "reference_summary_pass": reference_summary_pass,
    }
    if not domains_match:
        details["domain_mismatch"] = {
            "agent_rows": agent_domains,
            "reference_rows": reference_domains,
        }
    if not go_match:
        details["go_mismatch"] = {
            "agent_rows": agent_go,
            "reference_rows": reference_go,
        }
    return ScoreReport(
        score=1.0 if passed else 0.0,
        passed=passed,
        hard_fail_reason=None,
        domains_match=domains_match,
        go_match=go_match,
        summary_pass=summary_pass,
        details=details,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score protein annotation task outputs.")
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    agent_dir = Path(args.agent_dir)
    reference_dir = Path(args.reference_dir)
    report = score_output_payloads(
        agent_interpro_tsv=(agent_dir / "interpro_domains.tsv").read_text(encoding="utf-8"),
        agent_go_tsv=(agent_dir / "go_terms.tsv").read_text(encoding="utf-8"),
        agent_summary_text=(agent_dir / "functional_summary.txt").read_text(encoding="utf-8"),
        reference_interpro_tsv=(reference_dir / "expected_interpro_domains.tsv").read_text(encoding="utf-8"),
        reference_go_tsv=(reference_dir / "expected_go_terms.tsv").read_text(encoding="utf-8"),
        reference_summary_text=(reference_dir / "expected_functional_summary.txt").read_text(encoding="utf-8"),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
