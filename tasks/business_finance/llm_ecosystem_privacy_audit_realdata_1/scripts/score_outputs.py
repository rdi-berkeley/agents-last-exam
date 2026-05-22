"""Local deterministic scorer for llm_ecosystem_privacy_audit_realdata_1.

This module is imported by ``main.py`` during ``evaluate()`` and is also
useful standalone for ad-hoc verification.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from typing import Any

CRITICAL_RECALL_THRESHOLD = 0.85
OVERALL_PRECISION_THRESHOLD = 0.75
AMP_RELATIVE_TOLERANCE = 0.05
AMP_COVERAGE_THRESHOLD = 0.95

VIOLATION_KEY_FIELDS = (
    "severity",
    "action_domain",
    "api_name",
    "data_field_name",
    "data_type",
)


def _normalize_severity(value: Any) -> str:
    return str(value or "").strip().upper()


def _has_citations(item: dict[str, Any]) -> bool:
    return bool(str(item.get("policy_clause") or "").strip()) and bool(
        str(item.get("action_domain") or "").strip()
    )


def _violation_key(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        _normalize_severity(item.get("severity")),
        str(item.get("action_domain") or "").strip(),
        str(item.get("api_name") or "").strip(),
        str(item.get("data_field_name") or "").strip(),
        str(item.get("data_type") or "").strip(),
    )


def score_policy_violations(
    agent_doc: Any, reference_doc: Any
) -> dict[str, Any]:
    if not isinstance(reference_doc, dict):
        raise ValueError("reference policy_violations must be a JSON object")
    ref_items = reference_doc.get("violations")
    if not isinstance(ref_items, list):
        raise ValueError("reference .violations must be a list")

    if not isinstance(agent_doc, dict) or not isinstance(
        agent_doc.get("violations"), list
    ):
        return {
            "recall_critical": 0.0,
            "precision": 0.0,
            "reported_total": 0,
            "reported_valid": 0,
            "ref_total": len(ref_items),
            "ref_critical": sum(
                1
                for v in ref_items
                if _normalize_severity(v.get("severity")) == "CRITICAL"
            ),
            "passed": False,
            "reason": "agent output missing or malformed violations list",
        }

    agent_raw = agent_doc["violations"]
    agent_valid = [v for v in agent_raw if isinstance(v, dict) and _has_citations(v)]

    ref_counter: Counter[tuple[str, str, str, str, str]] = Counter(
        _violation_key(v) for v in ref_items if isinstance(v, dict)
    )
    agent_counter: Counter[tuple[str, str, str, str, str]] = Counter(
        _violation_key(v) for v in agent_valid
    )

    ref_critical = sum(c for k, c in ref_counter.items() if k[0] == "CRITICAL")

    tp_total = 0
    tp_critical = 0
    for key, count in agent_counter.items():
        tp = min(count, ref_counter.get(key, 0))
        tp_total += tp
        if key[0] == "CRITICAL":
            tp_critical += tp

    reported_valid = sum(agent_counter.values())
    recall_critical = tp_critical / ref_critical if ref_critical else 1.0
    precision = tp_total / reported_valid if reported_valid else 0.0

    passed = (
        recall_critical >= CRITICAL_RECALL_THRESHOLD
        and precision >= OVERALL_PRECISION_THRESHOLD
    )

    return {
        "recall_critical": recall_critical,
        "precision": precision,
        "reported_total": len(agent_raw),
        "reported_valid": reported_valid,
        "ref_total": sum(ref_counter.values()),
        "ref_critical": ref_critical,
        "tp_total": tp_total,
        "tp_critical": tp_critical,
        "passed": bool(passed),
    }


def _parse_amplification(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("x") or raw.endswith("X"):
        raw = raw[:-1]
    try:
        return float(raw)
    except ValueError:
        return None


def _read_csv_rows(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def score_cross_domain(agent_csv_text: str, reference_csv_text: str) -> dict[str, Any]:
    ref_rows = _read_csv_rows(reference_csv_text)
    agent_rows = _read_csv_rows(agent_csv_text)

    ref_by_domain: dict[str, float | None] = {}
    for row in ref_rows:
        domain = str(row.get("domain") or "").strip()
        if not domain:
            continue
        ref_by_domain[domain] = _parse_amplification(row.get("amplification_factor"))

    agent_by_domain: dict[str, float | None] = {}
    for row in agent_rows:
        domain = str(row.get("domain") or "").strip()
        if not domain:
            continue
        agent_by_domain[domain] = _parse_amplification(row.get("amplification_factor"))

    missing = sorted(set(ref_by_domain) - set(agent_by_domain))
    if missing:
        return {
            "hard_gate_passed": False,
            "missing_domains": missing,
            "missing_count": len(missing),
            "reported_domains": len(agent_by_domain),
            "ref_domains": len(ref_by_domain),
            "tolerance_ratio": 0.0,
            "within_tolerance": 0,
            "evaluated_overlap": 0,
            "passed": False,
        }

    evaluated = 0
    within = 0
    for domain, agent_amp in agent_by_domain.items():
        if domain not in ref_by_domain:
            continue
        ref_amp = ref_by_domain[domain]
        if ref_amp is None:
            continue
        evaluated += 1
        if agent_amp is None:
            continue
        if ref_amp == 0:
            if agent_amp == 0:
                within += 1
            continue
        if abs(agent_amp - ref_amp) / abs(ref_amp) <= AMP_RELATIVE_TOLERANCE:
            within += 1

    tolerance_ratio = (within / evaluated) if evaluated else 1.0
    passed = tolerance_ratio >= AMP_COVERAGE_THRESHOLD

    return {
        "hard_gate_passed": True,
        "missing_domains": [],
        "missing_count": 0,
        "reported_domains": len(agent_by_domain),
        "ref_domains": len(ref_by_domain),
        "within_tolerance": within,
        "evaluated_overlap": evaluated,
        "tolerance_ratio": tolerance_ratio,
        "passed": bool(passed),
    }


def score_submission(
    policy_agent_doc: Any,
    policy_reference_doc: Any,
    exposure_agent_csv: str,
    exposure_reference_csv: str,
) -> dict[str, Any]:
    sub1 = score_policy_violations(policy_agent_doc, policy_reference_doc)
    sub2 = score_cross_domain(exposure_agent_csv, exposure_reference_csv)
    sub1_score = 1.0 if sub1["passed"] else 0.0
    sub2_score = 1.0 if sub2["passed"] else 0.0
    overall = (sub1_score + sub2_score) / 2.0
    return {
        "sub_task_1_policy_violations": sub1,
        "sub_task_2_cross_domain_exposure": sub2,
        "sub_task_1_score": sub1_score,
        "sub_task_2_score": sub2_score,
        "overall_score": overall,
        "both_passed": bool(sub1["passed"] and sub2["passed"]),
    }


def zero_report(reason: str) -> dict[str, Any]:
    return {
        "sub_task_1_policy_violations": {"passed": False, "reason": reason},
        "sub_task_2_cross_domain_exposure": {"passed": False, "reason": reason},
        "sub_task_1_score": 0.0,
        "sub_task_2_score": 0.0,
        "overall_score": 0.0,
        "both_passed": False,
        "reason": reason,
    }


__all__ = [
    "CRITICAL_RECALL_THRESHOLD",
    "OVERALL_PRECISION_THRESHOLD",
    "AMP_RELATIVE_TOLERANCE",
    "AMP_COVERAGE_THRESHOLD",
    "score_policy_violations",
    "score_cross_domain",
    "score_submission",
    "zero_report",
]
