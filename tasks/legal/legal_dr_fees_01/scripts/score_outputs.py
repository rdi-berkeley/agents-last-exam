"""Scoring helpers for legal/legal_dr_fees_01."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

TOLERANCE = Decimal("0.01")

COMPONENT_WEIGHTS = {
    "institution_fee_rmb": 7.0,
    "arbitrator_remuneration_rmb": 7.0,
    "total_fee_rmb": 6.0,
}

ANSWER_KEY: dict[str, dict[str, Decimal]] = {
    "case_1": {
        "amount_in_dispute_rmb": Decimal("5800000.00"),
        "institution_fee_rmb": Decimal("27100.00"),
        "arbitrator_remuneration_rmb": Decimal("59000.00"),
        "total_fee_rmb": Decimal("86100.00"),
    },
    "case_2": {
        "amount_in_dispute_rmb": Decimal("65902536.17"),
        "institution_fee_rmb": Decimal("205756.34"),
        "arbitrator_remuneration_rmb": Decimal("270297.86"),
        "total_fee_rmb": Decimal("476054.20"),
    },
    "case_3": {
        "amount_in_dispute_rmb": Decimal("882480634.38"),
        "institution_fee_rmb": Decimal("1749465.14"),
        "arbitrator_remuneration_rmb": Decimal("2353953.52"),
        "total_fee_rmb": Decimal("4103418.66"),
    },
    "case_4": {
        "amount_in_dispute_rmb": Decimal("251620139.47"),
        "institution_fee_rmb": Decimal("147269.57"),
        "arbitrator_remuneration_rmb": Decimal("197553.09"),
        "total_fee_rmb": Decimal("344822.66"),
    },
    "case_5": {
        "amount_in_dispute_rmb": Decimal("3215.00"),
        "institution_fee_rmb": Decimal("5000.00"),
        "arbitrator_remuneration_rmb": Decimal("12000.00"),
        "total_fee_rmb": Decimal("17000.00"),
    },
}

FIELD_ALIASES = {
    "institution_fee_rmb": ("institution_fee_rmb", "institution_fee", "inst", "inst_fee"),
    "arbitrator_remuneration_rmb": (
        "arbitrator_remuneration_rmb",
        "arbitrator_remuneration",
        "arb",
        "arb_fee",
        "arbitrator_fee",
    ),
    "total_fee_rmb": ("total_fee_rmb", "total_fee", "total"),
}


def _parse_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float | Decimal):
        text = str(value)
    elif isinstance(value, str):
        text = (
            value.strip()
            .replace(",", "")
            .replace("RMB", "")
            .replace("rmb", "")
            .replace("元", "")
            .strip()
        )
    else:
        return None
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _normalize_case_id(value: Any, fallback_index: int | None = None) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip().lower().replace(" ", "_").replace("-", "_")
        if cleaned.startswith("case_"):
            suffix = cleaned.removeprefix("case_")
        elif cleaned.startswith("case"):
            suffix = cleaned.removeprefix("case").strip("_")
        else:
            suffix = cleaned
        if suffix.isdigit():
            return f"case_{int(suffix)}"
    if isinstance(value, int):
        return f"case_{value}"
    if fallback_index is not None:
        return f"case_{fallback_index}"
    return None


def _extract_cases(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        iterable = payload["cases"]
    elif isinstance(payload, list):
        iterable = payload
    elif isinstance(payload, dict):
        iterable = []
        for key, value in payload.items():
            if isinstance(value, dict):
                case = dict(value)
                case.setdefault("case_id", key)
                iterable.append(case)
    else:
        iterable = []

    cases: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(iterable, start=1):
        if not isinstance(item, dict):
            continue
        case_id = _normalize_case_id(
            item.get("case_id", item.get("case", item.get("id"))),
            fallback_index=index,
        )
        if case_id:
            cases[case_id] = item
    return cases


def _value_for(case: dict[str, Any], expected_field: str) -> Decimal | None:
    for key in FIELD_ALIASES[expected_field]:
        if key in case:
            return _parse_decimal(case[key])
    return None


def score_submission(payload: Any) -> dict[str, Any]:
    """Score parsed JSON payload and return a 0.0-1.0 score with details."""
    submitted_cases = _extract_cases(payload)
    total_points = sum(COMPONENT_WEIGHTS.values()) * len(ANSWER_KEY)
    earned = 0.0
    details: dict[str, dict[str, Any]] = {}

    for case_id, expected in ANSWER_KEY.items():
        case = submitted_cases.get(case_id)
        case_details: dict[str, Any] = {"present": case is not None, "components": {}}
        if case is None:
            details[case_id] = case_details
            continue
        for field, weight in COMPONENT_WEIGHTS.items():
            submitted_value = _value_for(case, field)
            expected_value = expected[field]
            correct = submitted_value is not None and abs(submitted_value - expected_value) <= TOLERANCE
            if correct:
                earned += weight
            case_details["components"][field] = {
                "submitted": str(submitted_value) if submitted_value is not None else None,
                "expected": str(expected_value),
                "correct": bool(correct),
                "points": weight if correct else 0.0,
                "max_points": weight,
            }
        details[case_id] = case_details

    score = earned / total_points if total_points else 0.0
    return {
        "score": round(score, 10),
        "points": round(earned, 4),
        "max_points": total_points,
        "details": details,
    }
