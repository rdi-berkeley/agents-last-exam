"""Deterministic scorer for AGORA governance classification outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


TECHNICAL_SCOPE_KEYS = [
    "ai_models",
    "ai_systems",
    "frontier_ai",
    "general_purpose_ai",
    "task_specific_ai",
    "foundation_models",
    "generative_ai",
    "predictive_ai",
    "compute_threshold",
    "open_weight_open_source",
]

LIFECYCLE_KEYS = [
    "plan_and_design",
    "collect_and_process_data",
    "build_and_use_model",
    "verify_and_validate",
    "deploy",
    "operate_and_monitor",
]

MIN_EVIDENCE_WORDS = 5
MIN_UNIQUE_EVIDENCE_RATIO = 0.5
MIN_MULTILABEL_PRECISION_FOR_PASS = 0.55
REQUIRED_ARTIFACT_CAP = 0.69

EVIDENCE_KEYWORDS = {
    "legislative_status:Hard Law": [
        "act",
        "criminal",
        "authority",
        "shall",
        "contract",
        "damages",
        "law",
        "enforcement",
    ],
    "legislative_status:Soft Law": [
        "voluntary",
        "commitment",
        "policy",
        "board",
        "internal",
        "reputational",
        "noncompliance",
    ],
    "legislative_status:Other": ["sandbox", "pilot", "hybrid", "experimental"],
    "technical_scope:ai_models": ["model", "models"],
    "technical_scope:ai_systems": ["system", "systems"],
    "technical_scope:frontier_ai": ["frontier", "advanced", "capable"],
    "technical_scope:general_purpose_ai": ["general", "generality", "multi-purpose", "multipurpose"],
    "technical_scope:task_specific_ai": ["task-specific", "narrow", "specific task"],
    "technical_scope:foundation_models": ["foundation", "pretrained", "pre-trained", "base model", "large language"],
    "technical_scope:generative_ai": ["generate", "generated", "generative", "forgery", "synthetic", "misinformation", "vision"],
    "technical_scope:predictive_ai": ["prediction", "predictions", "recommendations", "decisions", "inference"],
    "technical_scope:compute_threshold": ["compute", "flop", "threshold"],
    "technical_scope:open_weight_open_source": ["weight", "weights", "open source", "open-source", "open weight"],
    "lifecycle_stages:plan_and_design": ["review", "assess", "risk", "purpose", "design"],
    "lifecycle_stages:collect_and_process_data": ["data", "dataset", "collection", "label", "training data"],
    "lifecycle_stages:build_and_use_model": ["training", "developing", "created", "creation", "configuring", "inference", "build"],
    "lifecycle_stages:verify_and_validate": ["evaluation", "evaluate", "test", "testing", "audit", "review"],
    "lifecycle_stages:deploy": ["deploy", "deployment", "publish", "published", "available", "release", "operated"],
    "lifecycle_stages:operate_and_monitor": ["monitor", "monitoring", "incident", "remove", "report", "maintain", "ongoing"],
}


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalized_contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return normalize_ws(needle) in normalize_ws(haystack)


def binary_f1(pred: dict[str, bool], truth: dict[str, bool], keys: list[str]) -> float:
    tp = fp = fn = 0
    for key in keys:
        p = bool(pred.get(key, False))
        t = bool(truth.get(key, False))
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif not p and t:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def binary_metrics(pred: dict[str, bool], truth: dict[str, bool], keys: list[str]) -> dict[str, float]:
    tp = fp = fn = 0
    for key in keys:
        p = bool(pred.get(key, False))
        t = bool(truth.get(key, False))
        if p and t:
            tp += 1
        elif p and not t:
            fp += 1
        elif not p and t:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_documents(agent: Any) -> dict[str, dict[str, Any]]:
    raw_docs = agent.get("documents", agent) if isinstance(agent, dict) else agent
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw_docs, dict):
        items = raw_docs.items()
    elif isinstance(raw_docs, list):
        items = []
        for doc in raw_docs:
            if not isinstance(doc, dict):
                continue
            doc_id = str(doc.get("agora_id") or doc.get("id") or "").strip()
            items.append((doc_id, doc))
    else:
        return result

    for doc_id, doc in items:
        if not isinstance(doc, dict):
            continue
        normalized = dict(doc)
        if isinstance(doc.get("classifications"), dict):
            normalized.update(doc["classifications"])
        result[str(doc_id)] = normalized
    return result


def extract_bool_and_evidence(value: Any) -> tuple[bool, str | None]:
    if isinstance(value, dict):
        return bool(value.get("value", False)), value.get("evidence")
    return bool(value), None


def extract_bool_dict(dim: Any, keys: list[str]) -> dict[str, bool]:
    if not isinstance(dim, dict):
        return {key: False for key in keys}
    return {key: extract_bool_and_evidence(dim.get(key))[0] for key in keys}


def legislative_label(doc: dict[str, Any]) -> tuple[str, str | None]:
    value = doc.get("legislative_status")
    if isinstance(value, dict):
        return str(value.get("label", "")).strip(), value.get("evidence")
    return str(value or "").strip(), None


def evidence_word_count(evidence: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", evidence))


def evidence_supports_label(evidence: str, dim_name: str, label: str) -> bool:
    normalized = normalize_ws(evidence).lower()
    keywords = EVIDENCE_KEYWORDS.get(f"{dim_name}:{label}", [])
    return any(keyword.lower() in normalized for keyword in keywords)


def evidence_pass_rate(doc: dict[str, Any], doc_text: str) -> tuple[float, int, int, float]:
    checked = 0
    passed = 0
    seen: list[str] = []
    _, leg_evidence = legislative_label(doc)
    checked += 1
    if isinstance(leg_evidence, str):
        seen.append(normalize_ws(leg_evidence).lower())
    if (
        isinstance(leg_evidence, str)
        and evidence_word_count(leg_evidence) >= MIN_EVIDENCE_WORDS
        and normalized_contains(doc_text, leg_evidence)
        and evidence_supports_label(leg_evidence, "legislative_status", legislative_label(doc)[0])
    ):
        passed += 1

    for dim_name, keys in [
        ("technical_scope", TECHNICAL_SCOPE_KEYS),
        ("lifecycle_stages", LIFECYCLE_KEYS),
    ]:
        dim = doc.get(dim_name, {})
        if not isinstance(dim, dict):
            continue
        for key in keys:
            value, evidence = extract_bool_and_evidence(dim.get(key))
            if not value:
                continue
            checked += 1
            if isinstance(evidence, str):
                seen.append(normalize_ws(evidence).lower())
            if (
                isinstance(evidence, str)
                and evidence_word_count(evidence) >= MIN_EVIDENCE_WORDS
                and normalized_contains(doc_text, evidence)
                and evidence_supports_label(evidence, dim_name, key)
            ):
                passed += 1
    if checked == 0:
        return 0.0, checked, passed, 0.0
    unique_ratio = len(set(seen)) / checked if seen else 0.0
    pass_rate = passed / checked
    if checked >= 4 and unique_ratio < MIN_UNIQUE_EVIDENCE_RATIO:
        pass_rate = min(pass_rate, unique_ratio)
    return pass_rate, checked, passed, unique_ratio


def gap_analysis_valid(agent: dict[str, Any]) -> bool:
    text = agent.get("gap_analysis")
    if not isinstance(text, str):
        return False
    words = re.findall(r"\b[\w'-]+\b", text)
    if not 150 <= len(words) <= 300:
        return False
    unique_ratio = len({word.lower() for word in words}) / len(words)
    gap_terms = {
        "data",
        "lifecycle",
        "scope",
        "risk",
        "governance",
        "deployment",
        "monitoring",
        "model",
        "documents",
        "systems",
    }
    term_hits = {word.lower() for word in words} & gap_terms
    return unique_ratio >= 0.35 and len(term_hits) >= 4


def matrix_valid(agent: dict[str, Any], agent_docs: dict[str, dict[str, Any]]) -> bool:
    matrix = agent.get("cross_document_matrix")
    if not isinstance(matrix, dict):
        return False
    doc_ids = set(agent_docs)
    for dim_name, keys in [
        ("technical_scope", TECHNICAL_SCOPE_KEYS),
        ("lifecycle_stages", LIFECYCLE_KEYS),
    ]:
        dim_matrix = matrix.get(dim_name)
        if not isinstance(dim_matrix, dict):
            return False
        for key in keys:
            row = dim_matrix.get(key)
            if not isinstance(row, dict):
                return False
            for doc_id in doc_ids:
                predicted = extract_bool_dict(agent_docs[doc_id].get(dim_name), keys).get(key, False)
                if bool(row.get(doc_id, False)) != predicted:
                    return False

    leg_matrix = matrix.get("legislative_status")
    if not isinstance(leg_matrix, dict):
        return False
    for label in ["Hard Law", "Soft Law", "Other"]:
        row = leg_matrix.get(label)
        if not isinstance(row, dict):
            return False
        for doc_id in doc_ids:
            predicted_label, _ = legislative_label(agent_docs[doc_id])
            if bool(row.get(doc_id, False)) != (predicted_label == label):
                return False
    return True


def score_output(agent: dict[str, Any], truth: dict[str, Any], doc_texts: dict[str, str]) -> dict[str, Any]:
    truth_docs = truth["documents"]
    agent_docs = normalize_documents(agent)
    per_doc = {}
    leg_scores = []
    scope_scores = []
    lifecycle_scores = []
    total_tp = total_fp = 0

    for doc_id, gt_doc in truth_docs.items():
        candidate = agent_docs.get(doc_id)
        if not candidate:
            per_doc[doc_id] = {
                "missing": True,
                "legislative_score": 0.0,
                "scope_score": 0.0,
                "lifecycle_score": 0.0,
                "evidence_gate": 0.0,
            }
            leg_scores.append(0.0)
            scope_scores.append(0.0)
            lifecycle_scores.append(0.0)
            continue

        agent_leg, _ = legislative_label(candidate)
        leg_correct = float(agent_leg == gt_doc["legislative_status"])
        scope_pred = extract_bool_dict(candidate.get("technical_scope"), TECHNICAL_SCOPE_KEYS)
        lifecycle_pred = extract_bool_dict(candidate.get("lifecycle_stages"), LIFECYCLE_KEYS)
        scope_metrics = binary_metrics(scope_pred, gt_doc["technical_scope"], TECHNICAL_SCOPE_KEYS)
        lifecycle_metrics = binary_metrics(lifecycle_pred, gt_doc["lifecycle_stages"], LIFECYCLE_KEYS)
        scope_f1 = scope_metrics["f1"]
        lifecycle_f1 = lifecycle_metrics["f1"]
        total_tp += int(scope_metrics["tp"] + lifecycle_metrics["tp"])
        total_fp += int(scope_metrics["fp"] + lifecycle_metrics["fp"])
        pass_rate, evidence_checked, evidence_passed, unique_evidence_ratio = evidence_pass_rate(
            candidate,
            doc_texts.get(doc_id, ""),
        )
        evidence_gate = 1.0 if pass_rate >= 0.8 else 0.5
        leg_score = leg_correct * evidence_gate
        scope_score = scope_f1 * evidence_gate
        lifecycle_score = lifecycle_f1 * evidence_gate

        leg_scores.append(leg_score)
        scope_scores.append(scope_score)
        lifecycle_scores.append(lifecycle_score)
        per_doc[doc_id] = {
            "missing": False,
            "legislative_agent": agent_leg,
            "legislative_truth": gt_doc["legislative_status"],
            "legislative_score": leg_score,
            "scope_f1": scope_f1,
            "scope_precision": scope_metrics["precision"],
            "scope_score": scope_score,
            "lifecycle_f1": lifecycle_f1,
            "lifecycle_precision": lifecycle_metrics["precision"],
            "lifecycle_score": lifecycle_score,
            "evidence_pass_rate": pass_rate,
            "evidence_checked": evidence_checked,
            "evidence_passed": evidence_passed,
            "unique_evidence_ratio": unique_evidence_ratio,
            "evidence_gate": evidence_gate,
        }

    leg_avg = sum(leg_scores) / len(truth_docs)
    scope_avg = sum(scope_scores) / len(truth_docs)
    lifecycle_avg = sum(lifecycle_scores) / len(truth_docs)
    classification_score = leg_avg * 0.2 + scope_avg * 0.4 + lifecycle_avg * 0.4
    gap_ok = gap_analysis_valid(agent)
    matrix_ok = matrix_valid(agent, agent_docs)
    multilabel_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    final_score = classification_score if gap_ok else min(classification_score, 0.9)
    if not matrix_ok:
        final_score = min(final_score, 0.95)
    if classification_score >= 0.70 and multilabel_precision < MIN_MULTILABEL_PRECISION_FOR_PASS:
        final_score = min(final_score, 0.69)
    if not gap_ok or not matrix_ok:
        final_score = min(final_score, REQUIRED_ARTIFACT_CAP)
    return {
        "score": final_score,
        "classification_score": classification_score,
        "gap_analysis_valid": gap_ok,
        "cross_document_matrix_valid": matrix_ok,
        "multilabel_precision": multilabel_precision,
        "legislative_mean": leg_avg,
        "technical_scope_mean": scope_avg,
        "lifecycle_mean": lifecycle_avg,
        "per_doc": per_doc,
    }


def score_output_file(agent_path: Path, truth_path: Path, docs_path: Path) -> dict[str, Any]:
    agent = load_json(agent_path)
    if not isinstance(agent, dict):
        raise ValueError("agent output must be a JSON object")
    truth = load_json(truth_path)
    docs = load_json(docs_path)
    if not isinstance(docs, dict):
        raise ValueError("document_texts must be a JSON object")
    return score_output(agent, truth, docs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--docs", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(score_output_file(args.agent, args.truth, args.docs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
