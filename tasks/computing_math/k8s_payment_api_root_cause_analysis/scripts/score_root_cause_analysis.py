"""Deterministic scorer for k8s_payment_api_root_cause_analysis."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL = {
    "incident_metadata",
    "root_causes",
    "affected_resources",
    "remediation_plan",
    "summary",
}

REFERENCE_RESOURCES = {
    "deployment/payment-api@prod",
    "horizontalpodautoscaler/payment-api@prod",
    "pod/payment-api-7c9d4b85f5-2m8xq@prod",
    "pod/payment-api-7c9d4b85f5-kz7rv@prod",
}


@dataclass
class ScoreResult:
    score: float
    passed: bool
    hard_gate: str | None
    components: dict[str, float]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hard_fail(reason: str, details: dict[str, Any] | None = None) -> ScoreResult:
    return ScoreResult(
        score=0.0,
        passed=False,
        hard_gate=reason,
        components={},
        details=details or {},
    )


def _load_json_object(payload: str | bytes, *, label: str) -> dict[str, Any]:
    text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _norm_type(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _text(value: Any) -> str:
    return str(value or "")


def _root_causes(report: dict[str, Any]) -> list[dict[str, Any]]:
    values = report.get("root_causes")
    return [value for value in values if isinstance(value, dict)] if isinstance(values, list) else []


def _evidence_list(cause: dict[str, Any]) -> list[str]:
    values = cause.get("evidence")
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str) and value.strip()]


def _primary_score(causes: list[dict[str, Any]]) -> float:
    tokens = ["oomkill", "oom", "memorylimit"]
    for cause in causes:
        type_norm = _norm_type(cause.get("type"))
        if any(token in type_norm for token in tokens) and _text(cause.get("severity")).lower() == "critical":
            return 1.0
    return 0.0


def _secondary_score(causes: list[dict[str, Any]]) -> float:
    for cause in causes:
        type_norm = _norm_type(cause.get("type"))
        if not any(token in type_norm for token in ["liveness", "probe", "startupprobe"]):
            continue
        evidence = "\n".join(_evidence_list(cause))
        has_delay = "initialDelaySeconds: 5" in evidence
        has_warmup = (
            "duration=6.67s" in evidence
            or "duration=6.54s" in evidence
            or "cache warm-up complete" in evidence
        )
        if has_delay and has_warmup:
            return 1.0
    return 0.0


def _tertiary_score(causes: list[dict[str, Any]]) -> float:
    for cause in causes:
        haystack = " ".join(
            [
                _text(cause.get("type")),
                _text(cause.get("description")),
                " ".join(_evidence_list(cause)),
            ]
        ).lower()
        if "metrics_port" not in haystack and "metrics port" not in haystack:
            continue
        if _text(cause.get("severity")).lower() in {"low", "medium"}:
            return 1.0
    return 0.0


def _normalize_resource(value: Any) -> str | None:
    if isinstance(value, dict):
        kind = _text(value.get("kind")).strip()
        name = _text(value.get("name")).strip()
        namespace = _text(value.get("namespace")).strip() or "prod"
        if kind and name:
            return f"{kind}/{name}@{namespace}".lower()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if "@" in text:
            return text.lower()
        parts = text.split()
        for part in parts:
            if "/" in part:
                return f"{part}@prod".lower()
    return None


def _affected_resources_score(report: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    values = report.get("affected_resources")
    observed: set[str] = set()
    if isinstance(values, list):
        for value in values:
            normalized = _normalize_resource(value)
            if normalized:
                observed.add(normalized)
    true_positive = len(observed & REFERENCE_RESOURCES)
    precision = true_positive / len(observed) if observed else 0.0
    recall = true_positive / len(REFERENCE_RESOURCES)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, {
        "observed": sorted(observed),
        "expected": sorted(REFERENCE_RESOURCES),
        "precision": precision,
        "recall": recall,
    }


def _remediation_score(report: dict[str, Any]) -> tuple[float, dict[str, bool]]:
    values = report.get("remediation_plan")
    action_text = ""
    if isinstance(values, list):
        action_text = "\n".join(
            _text(item.get("action") if isinstance(item, dict) else item) for item in values
        )
    patterns = {
        "memory_limit": r"memory.*(512|768|1Gi|1024)",
        "probe_relaxation": r"(initialDelaySeconds.*(20|[3-9][0-9])|startupProbe|startup probe)",
        "metrics_port": r"(METRICS_PORT|metrics port)",
    }
    matches = {
        name: bool(re.search(pattern, action_text, flags=re.IGNORECASE | re.DOTALL))
        for name, pattern in patterns.items()
    }
    return sum(matches.values()) / len(patterns), matches


def _evidence_grounding_score(causes: list[dict[str, Any]], input_texts: list[str]) -> tuple[float, dict[str, Any]]:
    if not causes:
        return 0.0, {"grounded_causes": 0, "total_causes": 0, "ungrounded": []}
    grounded = 0
    ungrounded: list[dict[str, Any]] = []
    for index, cause in enumerate(causes):
        evidence = _evidence_list(cause)
        if not evidence:
            ungrounded.append({"cause_index": index, "missing": ["<empty evidence>"]})
            continue
        missing = [item for item in evidence if not any(item in text for text in input_texts)]
        if missing:
            ungrounded.append({"cause_index": index, "missing": missing})
            continue
        grounded += 1
    return grounded / len(causes), {
        "grounded_causes": grounded,
        "total_causes": len(causes),
        "ungrounded": ungrounded[:10],
    }


def _summary_rollback_score(report: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    summary_ok = isinstance(report.get("summary"), str) and bool(report["summary"].strip())
    metadata = report.get("incident_metadata")
    rollback_value = metadata.get("safe_to_rollback") if isinstance(metadata, dict) else None
    rollback_ok = rollback_value is True
    return (0.5 if summary_ok else 0.0) + (0.5 if rollback_ok else 0.0), {
        "summary_ok": summary_ok,
        "safe_to_rollback": rollback_value,
    }


def score_report(
    agent_report_json: str | bytes,
    *,
    cluster_state: str,
    deployment_yaml: str,
    failing_pod_log: str,
) -> ScoreResult:
    try:
        report = _load_json_object(agent_report_json, label="agent report")
    except ValueError as exc:
        return _hard_fail(str(exc))

    missing = sorted(REQUIRED_TOP_LEVEL - set(report))
    if missing:
        return _hard_fail("missing required top-level keys", {"missing": missing})

    causes = _root_causes(report)
    if not causes:
        return _hard_fail("root_causes must be a non-empty array")

    components: dict[str, float] = {}
    details: dict[str, Any] = {}
    components["primary_root_cause"] = _primary_score(causes)
    components["secondary_liveness_probe"] = _secondary_score(causes)
    components["tertiary_metrics_port"] = _tertiary_score(causes)
    components["affected_resources"], details["affected_resources"] = _affected_resources_score(report)
    components["remediation_plan"], details["remediation_plan"] = _remediation_score(report)
    components["evidence_grounding"], details["evidence_grounding"] = _evidence_grounding_score(
        causes,
        [cluster_state, deployment_yaml, failing_pod_log],
    )
    components["summary_rollback"], details["summary_rollback"] = _summary_rollback_score(report)

    weights = {
        "primary_root_cause": 0.30,
        "secondary_liveness_probe": 0.20,
        "tertiary_metrics_port": 0.10,
        "affected_resources": 0.10,
        "remediation_plan": 0.15,
        "evidence_grounding": 0.10,
        "summary_rollback": 0.05,
    }
    score = sum(components[name] * weight for name, weight in weights.items())
    score = round(score, 6)
    return ScoreResult(
        score=score,
        passed=score >= 0.999999,
        hard_gate=None,
        components=components,
        details=details,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-report", required=True, type=Path)
    parser.add_argument("--cluster-state", required=True, type=Path)
    parser.add_argument("--deployment-yaml", required=True, type=Path)
    parser.add_argument("--failing-pod-log", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = score_report(
        args.agent_report.read_bytes(),
        cluster_state=args.cluster_state.read_text(encoding="utf-8"),
        deployment_yaml=args.deployment_yaml.read_text(encoding="utf-8"),
        failing_pod_log=args.failing_pod_log.read_text(encoding="utf-8"),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    raise SystemExit(0 if result.score >= 0.999999 else 1)


if __name__ == "__main__":
    main()
