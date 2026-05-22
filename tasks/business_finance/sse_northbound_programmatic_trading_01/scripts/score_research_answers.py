"""Scoring helpers for sse_northbound_programmatic_trading_01."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ALLOWED_CONCLUSIONS = {"yes", "no", "unknown"}
ASCII_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _normalize_text(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalized.replace("\u3000", " ")
    normalized = normalized.replace("：", ":")
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("《", "").replace("》", "")
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", "", normalized)


def _normalize_conclusion(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _looks_like_english_explanation(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return len(stripped) >= 20 and len(ASCII_WORD_RE.findall(stripped)) >= 3


def _build_citation_map(manifest: list[dict[str, Any]]) -> dict[str, set[str]]:
    citation_map: dict[str, set[str]] = {}
    for entry in manifest:
        doc_id = entry.get("doc_id")
        if not isinstance(doc_id, str):
            continue
        for alias in entry.get("citation_aliases", []):
            if not isinstance(alias, str):
                continue
            citation_map.setdefault(_normalize_text(alias), set()).add(doc_id)
    return citation_map


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_question(
    *,
    key: str,
    response: Any,
    spec: dict[str, Any],
    citation_map: dict[str, set[str]],
    evaluator_documents: dict[str, str],
) -> dict[str, Any]:
    breakdown = {
        "question": key,
        "score": 0.0,
        "conclusion_correct": False,
        "citation_valid": False,
        "evidence_valid": False,
        "answer_text_valid": False,
        "notes": [],
    }

    if not isinstance(response, dict):
        breakdown["notes"].append("question payload is not an object")
        return breakdown

    conclusion = response.get("conclusion")
    citation = response.get("citation_document")
    evidence = response.get("evidence_snippet")
    answer_text = response.get("answer_text")

    normalized_conclusion = _normalize_conclusion(conclusion)
    if normalized_conclusion not in ALLOWED_CONCLUSIONS:
        breakdown["notes"].append("conclusion is missing or invalid")
        return breakdown
    if normalized_conclusion != str(spec["expected_conclusion"]).lower():
        breakdown["notes"].append("conclusion does not match the canonical answer")
        return breakdown
    breakdown["conclusion_correct"] = True

    if not isinstance(citation, str) or not citation.strip():
        breakdown["notes"].append("citation_document is missing")
        return breakdown
    normalized_citation = _normalize_text(citation)
    accepted_citations = {_normalize_text(item) for item in spec["accepted_citations"]}
    if normalized_citation not in accepted_citations:
        breakdown["notes"].append("citation_document is not an accepted alias")
        return breakdown
    cited_doc_ids = citation_map.get(normalized_citation, set())
    accepted_doc_ids = set(spec.get("accepted_doc_ids", []))
    required_doc_id = spec.get("required_doc_id")
    if required_doc_id:
        accepted_doc_ids.add(required_doc_id)
    if not accepted_doc_ids:
        accepted_doc_ids = set(cited_doc_ids)
    doc_ids_to_check = cited_doc_ids & accepted_doc_ids if cited_doc_ids else accepted_doc_ids
    if not doc_ids_to_check:
        doc_ids_to_check = accepted_doc_ids
    breakdown["citation_valid"] = True

    if not isinstance(evidence, str) or not evidence.strip():
        breakdown["notes"].append("evidence_snippet is missing")
        return breakdown
    normalized_evidence = _normalize_text(evidence)
    if not normalized_evidence:
        breakdown["notes"].append("evidence_snippet is empty after normalization")
        return breakdown
    if not all(_normalize_text(anchor) in normalized_evidence for anchor in spec["evidence_anchors"]):
        breakdown["notes"].append("evidence_snippet does not contain the required anchor text")
        return breakdown

    evidence_found = False
    for doc_id in doc_ids_to_check:
        doc_text = evaluator_documents.get(doc_id, "")
        if normalized_evidence in _normalize_text(doc_text):
            evidence_found = True
            break
    if not evidence_found:
        breakdown["notes"].append("evidence_snippet is not an exact excerpt from the cited visible source")
        return breakdown
    breakdown["evidence_valid"] = True

    if not _looks_like_english_explanation(answer_text):
        breakdown["notes"].append("answer_text is missing or not a short English explanation")
        return breakdown
    breakdown["answer_text_valid"] = True

    breakdown["score"] = 1.0
    breakdown["notes"].append("accepted")
    return breakdown


def score_submission(
    *,
    submission_payload: Any,
    answer_key: dict[str, Any],
    evaluator_documents: dict[str, str],
    document_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(submission_payload, dict):
        return {"score": 0.0, "error": "submission root is not a JSON object"}

    required_keys = list(answer_key.keys())
    missing_keys = [key for key in required_keys if key not in submission_payload]
    if missing_keys:
        return {"score": 0.0, "error": f"missing required keys: {', '.join(missing_keys)}"}

    citation_map = _build_citation_map(document_manifest)
    per_question = {
        key: _score_question(
            key=key,
            response=submission_payload.get(key),
            spec=spec,
            citation_map=citation_map,
            evaluator_documents=evaluator_documents,
        )
        for key, spec in answer_key.items()
    }
    total = sum(item["score"] for item in per_question.values())
    score = total / len(answer_key) if answer_key else 0.0
    return {
        "score": score,
        "question_scores": per_question,
        "all_required_keys_present": True,
        "unexpected_keys": sorted(set(submission_payload.keys()) - set(required_keys)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-file", required=True)
    parser.add_argument("--answer-key-file", required=True)
    parser.add_argument("--documents-file", required=True)
    parser.add_argument("--manifest-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score_submission(
        submission_payload=_load_json(Path(args.submission_file)),
        answer_key=_load_json(Path(args.answer_key_file)),
        evaluator_documents=_load_json(Path(args.documents_file)),
        document_manifest=_load_json(Path(args.manifest_file)),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
