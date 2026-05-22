"""Local scorer for computing_math/cost_optimization_1."""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPECTED_SUMMARY_COLUMNS = [
    "resource_id",
    "resource_type",
    "name",
    "action",
    "current_cost",
    "projected_cost",
    "monthly_savings",
]
EXECUTIVE_SUMMARY_FIELDS = [
    "total_current_cost",
    "total_projected_savings",
    "savings_percentage",
]
REQUIRED_EXECUTIVE_SUMMARY_FIELDS = [
    "total_current_cost",
    "total_projected_savings",
    "projected_cost_after_optimization",
    "savings_percentage",
    "total_resources_analyzed",
    "resources_with_recommendations",
]
REQUIRED_RECOMMENDATION_TEXT_FIELDS = [
    "resource_id",
    "resource_type",
    "name",
    "current_config",
    "action",
    "recommended_config",
    "reason",
]
REQUIRED_RECOMMENDATION_NUMERIC_FIELDS = [
    "current_monthly_cost",
    "projected_monthly_cost",
    "monthly_savings",
]
DEFAULT_ALIAS_MAP = {
    "nat-gw-prod-01": ["nat-gw-prod-01"],
    "eip-unused-3": ["eip-unused-3"],
    "cw-loggroup-old-processor": [
        "cw-loggroup-old-processor",
        "cw-loggroup-/aws/lambda/old-processor",
        "/aws/lambda/old-processor",
    ],
}


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
    return ScoreResult(score=0.0, passed=False, reason=reason, hard_gate=reason, details=details or {})


def _load_json_object(text: str | bytes, *, label: str) -> dict[str, Any]:
    payload = text.decode("utf-8-sig") if isinstance(text, bytes) else text
    try:
        parsed = json.loads(payload)
    except Exception as exc:
        raise ValueError(f"{label} unreadable: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to an object")
    return parsed


def _load_csv_rows(text: str | bytes, *, label: str) -> tuple[list[str], list[dict[str, str]]]:
    payload = text.decode("utf-8-sig") if isinstance(text, bytes) else text
    try:
        reader = csv.DictReader(io.StringIO(payload))
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("missing header")
        rows = list(reader)
    except Exception as exc:
        raise ValueError(f"{label} unreadable: {exc}") from exc
    return fieldnames, rows


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower()


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _has_required_summary_field(summary: dict[str, Any], field: str) -> bool:
    value = summary.get(field)
    if field in {
        "total_current_cost",
        "total_projected_savings",
        "projected_cost_after_optimization",
        "savings_percentage",
        "total_resources_analyzed",
        "resources_with_recommendations",
    }:
        return _coerce_float(value) is not None
    return _nonempty_text(value)


def _recommendation_has_required_field(recommendation: dict[str, Any], field: str) -> bool:
    if field in REQUIRED_RECOMMENDATION_NUMERIC_FIELDS:
        return _coerce_float(recommendation.get(field)) is not None
    return _nonempty_text(recommendation.get(field))


def _build_alias_lookup(source_manifest: dict[str, Any] | None) -> dict[str, str]:
    alias_map = dict(DEFAULT_ALIAS_MAP)
    if isinstance(source_manifest, dict):
        manifest_aliases = source_manifest.get("image_only_resource_aliases")
        if isinstance(manifest_aliases, dict):
            for canonical_id, aliases in manifest_aliases.items():
                if not isinstance(canonical_id, str):
                    continue
                merged = list(alias_map.get(canonical_id, []))
                if isinstance(aliases, list):
                    merged.extend(alias for alias in aliases if isinstance(alias, str))
                alias_map[canonical_id] = merged

    lookup: dict[str, str] = {}
    for canonical_id, aliases in alias_map.items():
        canonical_key = _normalize_token(canonical_id)
        lookup[canonical_key] = canonical_key
        for alias in aliases:
            lookup[_normalize_token(alias)] = canonical_key
    return lookup


def _canonical_resource_id(resource_id: Any, alias_lookup: dict[str, str]) -> str | None:
    if not _nonempty_text(resource_id):
        return None
    normalized = _normalize_token(resource_id)
    return alias_lookup.get(normalized, normalized)


def _extract_reference_recommendations(
    reference_report: dict[str, Any],
    alias_lookup: dict[str, str],
) -> dict[str, dict[str, Any]]:
    recommendations = reference_report.get("recommendations")
    if not isinstance(recommendations, list):
        raise ValueError("reference_report missing recommendations array")

    extracted: dict[str, dict[str, Any]] = {}
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            continue
        canonical_id = _canonical_resource_id(recommendation.get("resource_id"), alias_lookup)
        if canonical_id is None:
            continue
        extracted[canonical_id] = {
            "resource_id": canonical_id,
            "action": _normalize_token(recommendation.get("action", "")),
            "current_monthly_cost": _coerce_float(recommendation.get("current_monthly_cost")),
            "projected_monthly_cost": _coerce_float(recommendation.get("projected_monthly_cost")),
            "monthly_savings": _coerce_float(recommendation.get("monthly_savings")),
        }
    return extracted


def _extract_agent_recommendations(
    agent_report: dict[str, Any],
    alias_lookup: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    recommendations = agent_report.get("recommendations")
    if not isinstance(recommendations, list):
        raise ValueError("missing_recommendations_array")

    extracted: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            continue
        canonical_id = _canonical_resource_id(recommendation.get("resource_id"), alias_lookup)
        if canonical_id is None:
            continue
        if canonical_id in extracted:
            duplicates.append(canonical_id)
            continue
        extracted[canonical_id] = {
            "resource_id": canonical_id,
            "action": _normalize_token(recommendation.get("action", "")),
            "current_monthly_cost": _coerce_float(recommendation.get("current_monthly_cost")),
            "projected_monthly_cost": _coerce_float(recommendation.get("projected_monthly_cost")),
            "monthly_savings": _coerce_float(recommendation.get("monthly_savings")),
        }
    return extracted, duplicates


def _savings_accuracy(agent_value: float | None, reference_value: float | None) -> float:
    if agent_value is None or reference_value is None:
        return 0.0
    if reference_value == 0:
        return 1.0 if agent_value == 0 else 0.0
    relative_error = abs(agent_value - reference_value) / abs(reference_value)
    if relative_error <= 0.2:
        return 1.0
    if relative_error >= 1.0:
        return 0.0
    return 1.0 - ((relative_error - 0.2) / 0.8)


def _same_float(left: float | None, right: float | None, *, tol: float = 1e-6) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= tol


def _validate_summary_csv(
    summary_rows: list[dict[str, str]],
    *,
    agent_recommendations: dict[str, dict[str, Any]],
    alias_lookup: dict[str, str],
) -> tuple[str | None, dict[str, Any]]:
    meaningful_rows = [
        row
        for row in summary_rows
        if any((value or "").strip() for value in row.values() if isinstance(value, str))
    ]
    if not meaningful_rows:
        return "empty_savings_summary_csv", {}

    csv_recommendations: dict[str, dict[str, Any]] = {}
    total_row: dict[str, float] | None = None
    for row in meaningful_rows:
        resource_id_raw = (row.get("resource_id") or "").strip()
        if not resource_id_raw:
            return "missing_summary_resource_id", {"row": row}

        current_cost = _coerce_float(row.get("current_cost"))
        projected_cost = _coerce_float(row.get("projected_cost"))
        monthly_savings = _coerce_float(row.get("monthly_savings"))
        if current_cost is None or projected_cost is None or monthly_savings is None:
            return "non_numeric_summary_values", {"row": row}

        if _normalize_token(resource_id_raw) == "total":
            if total_row is not None:
                return "duplicate_total_row", {}
            total_row = {
                "current_cost": current_cost,
                "projected_cost": projected_cost,
                "monthly_savings": monthly_savings,
            }
            continue

        canonical_id = _canonical_resource_id(resource_id_raw, alias_lookup)
        if canonical_id is None:
            return "missing_summary_resource_id", {"row": row}
        if canonical_id in csv_recommendations:
            return "duplicate_summary_resource_id", {"resource_id": canonical_id}

        action = _normalize_token(row.get("action", ""))
        if not action:
            return "missing_summary_action", {"resource_id": canonical_id}

        csv_recommendations[canonical_id] = {
            "resource_id": canonical_id,
            "action": action,
            "current_monthly_cost": current_cost,
            "projected_monthly_cost": projected_cost,
            "monthly_savings": monthly_savings,
        }

    json_ids = set(agent_recommendations)
    csv_ids = set(csv_recommendations)
    if not csv_ids:
        return "empty_savings_summary_csv", {}

    missing_in_csv = sorted(json_ids - csv_ids)
    unexpected_in_csv = sorted(csv_ids - json_ids)
    if missing_in_csv or unexpected_in_csv:
        return (
            "summary_json_resource_mismatch",
            {
                "missing_in_csv": missing_in_csv,
                "unexpected_in_csv": unexpected_in_csv,
            },
        )

    for resource_id in sorted(json_ids):
        json_rec = agent_recommendations[resource_id]
        csv_rec = csv_recommendations[resource_id]
        if csv_rec["action"] != json_rec["action"]:
            return (
                "summary_json_action_mismatch",
                {
                    "resource_id": resource_id,
                    "json_action": json_rec["action"],
                    "csv_action": csv_rec["action"],
                },
            )
        for field in ("current_monthly_cost", "projected_monthly_cost", "monthly_savings"):
            if not _same_float(csv_rec[field], json_rec[field]):
                return (
                    "summary_json_value_mismatch",
                    {
                        "resource_id": resource_id,
                        "field": field,
                        "json_value": json_rec[field],
                        "csv_value": csv_rec[field],
                    },
                )

    return None, {"summary_row_count": len(meaningful_rows), "has_total_row": total_row is not None}


def score_output_bundle(
    *,
    agent_report_json: str | bytes,
    agent_summary_csv: str | bytes,
    reference_report_json: str | bytes,
    source_manifest_json: str | bytes | None = None,
) -> ScoreResult:
    try:
        agent_report = _load_json_object(agent_report_json, label="optimization_report.json")
        reference_report = _load_json_object(reference_report_json, label="reference optimization_report.json")
        source_manifest = (
            _load_json_object(source_manifest_json, label="source_manifest.json")
            if source_manifest_json is not None
            else {}
        )
        summary_columns, summary_rows = _load_csv_rows(agent_summary_csv, label="savings_summary.csv")
    except ValueError as exc:
        return _hard_fail(str(exc))

    if not isinstance(agent_report.get("recommendations"), list):
        return _hard_fail("missing_recommendations_array")

    summary = agent_report.get("executive_summary")
    if isinstance(summary, dict):
        total_current_cost = _coerce_float(summary.get("total_current_cost"))
        total_projected_savings = _coerce_float(summary.get("total_projected_savings"))
        if total_current_cost is not None and total_current_cost < 0:
            return _hard_fail("negative_total_current_cost")
        if total_projected_savings is not None and total_projected_savings < 0:
            return _hard_fail("negative_total_projected_savings")
        if (
            total_current_cost is not None
            and total_projected_savings is not None
            and total_projected_savings > total_current_cost
        ):
            return _hard_fail(
                "projected_savings_exceeds_current_cost",
                {
                    "total_current_cost": total_current_cost,
                    "total_projected_savings": total_projected_savings,
                },
            )

    if any(column not in summary_columns for column in EXPECTED_SUMMARY_COLUMNS):
        return _hard_fail(
            "malformed_savings_summary_csv",
            {"observed_columns": summary_columns, "expected_columns": EXPECTED_SUMMARY_COLUMNS},
        )

    alias_lookup = _build_alias_lookup(source_manifest)
    try:
        reference_recommendations = _extract_reference_recommendations(reference_report, alias_lookup)
        agent_recommendations, duplicate_ids = _extract_agent_recommendations(agent_report, alias_lookup)
    except ValueError as exc:
        return _hard_fail(str(exc))
    if duplicate_ids:
        return _hard_fail("duplicate_recommendation_resource_id", {"duplicate_resource_ids": sorted(duplicate_ids)})

    summary_error, summary_details = _validate_summary_csv(
        summary_rows,
        agent_recommendations=agent_recommendations,
        alias_lookup=alias_lookup,
    )
    if summary_error is not None:
        return _hard_fail(summary_error, summary_details)

    predicted_ids = set(agent_recommendations)
    reference_ids = set(reference_recommendations)
    matched_ids = predicted_ids & reference_ids

    precision = len(matched_ids) / len(predicted_ids) if predicted_ids else 0.0
    recall = len(matched_ids) / len(reference_ids) if reference_ids else 0.0
    if precision == 0.0 and recall == 0.0:
        coverage_score = 0.0
    else:
        coverage_score = 2 * precision * recall / (precision + recall)

    correct_actions = sum(
        1
        for resource_id in matched_ids
        if agent_recommendations[resource_id]["action"] == reference_recommendations[resource_id]["action"]
    )
    action_score = correct_actions / len(matched_ids) if matched_ids else 0.0

    savings_scores = [
        _savings_accuracy(
            agent_recommendations[resource_id]["monthly_savings"],
            reference_recommendations[resource_id]["monthly_savings"],
        )
        for resource_id in matched_ids
    ]
    savings_score = sum(savings_scores) / len(savings_scores) if savings_scores else 0.0

    recommendations = agent_report.get("recommendations", [])
    summary = agent_report.get("executive_summary", {})
    completeness_checks = {
        f"executive_summary_{field}": isinstance(summary, dict)
        and _has_required_summary_field(summary, field)
        for field in REQUIRED_EXECUTIVE_SUMMARY_FIELDS
    }
    completeness_checks.update(
        {
            f"every_recommendation_has_{field}": isinstance(recommendations, list)
            and len(recommendations) > 0
            and all(
                isinstance(rec, dict) and _recommendation_has_required_field(rec, field)
                for rec in recommendations
            )
            for field in REQUIRED_RECOMMENDATION_TEXT_FIELDS + REQUIRED_RECOMMENDATION_NUMERIC_FIELDS
        }
    )
    completeness_checks["savings_summary_has_expected_columns"] = all(
        column in summary_columns for column in EXPECTED_SUMMARY_COLUMNS
    )
    completeness_score = sum(1 for passed in completeness_checks.values() if passed) / len(
        completeness_checks
    )

    score = (
        0.30 * coverage_score
        + 0.30 * action_score
        + 0.25 * savings_score
        + 0.15 * completeness_score
    )

    details = {
        "coverage_f1": coverage_score,
        "precision": precision,
        "recall": recall,
        "action_score": action_score,
        "savings_score": savings_score,
        "completeness_score": completeness_score,
        "matched_resource_ids": sorted(matched_ids),
        "missing_resource_ids": sorted(reference_ids - predicted_ids),
        "unexpected_resource_ids": sorted(predicted_ids - reference_ids),
        "duplicate_resource_ids": sorted(duplicate_ids),
        "component_weights": {
            "coverage": 0.30,
            "action": 0.30,
            "savings": 0.25,
            "completeness": 0.15,
        },
        "completeness_checks": completeness_checks,
        "reference_resource_count": len(reference_ids),
        "predicted_resource_count": len(predicted_ids),
        **summary_details,
    }

    final_score = round(score, 6)
    passed = final_score >= 0.999999

    return ScoreResult(
        score=final_score,
        passed=passed,
        reason="passed" if passed else "partial_match",
        hard_gate=None,
        details={**details, "pass_threshold": 0.999999},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score cost_optimization_1 outputs.")
    parser.add_argument("--agent-report", required=True)
    parser.add_argument("--agent-summary", required=True)
    parser.add_argument("--reference-report", required=True)
    parser.add_argument("--source-manifest")
    return parser.parse_args()


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


if __name__ == "__main__":
    args = _parse_args()
    result = score_output_bundle(
        agent_report_json=_read_text(args.agent_report),
        agent_summary_csv=_read_text(args.agent_summary),
        reference_report_json=_read_text(args.reference_report),
        source_manifest_json=_read_text(args.source_manifest) if args.source_manifest else None,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
