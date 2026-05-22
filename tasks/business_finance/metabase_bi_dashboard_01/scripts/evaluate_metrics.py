"""Scoring logic for metabase_bi_dashboard_01.

Compares agent-produced dashboard_metrics.json against the reference metrics.
Each of the 11 top-level fields is scored independently with equal weight.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Relative tolerance for float comparisons (0.5%)
NUMERIC_RTOL = 0.005


def _numeric_match(actual, expected, rtol=NUMERIC_RTOL):
    """Return True if actual is within rtol of expected."""
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) <= rtol


def _score_scalar_numeric(actual, expected, rtol=NUMERIC_RTOL):
    """Score a single numeric value: 1.0 if within tolerance, else 0.0."""
    if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
        return 0.0
    return 1.0 if _numeric_match(actual, expected, rtol) else 0.0


def _score_exact_int(actual, expected):
    """Score an exact integer match."""
    try:
        return 1.0 if int(actual) == int(expected) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _score_dict_numeric(actual, expected, rtol=NUMERIC_RTOL):
    """Score a dict of numeric values. Partial credit per matching key."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return 0.0
    if not expected:
        return 1.0
    matched = 0
    for key, ref_val in expected.items():
        if key not in actual:
            continue
        if isinstance(ref_val, (int, float)) and isinstance(actual[key], (int, float)):
            if _numeric_match(actual[key], ref_val, rtol):
                matched += 1
    return matched / len(expected)


def _score_dict_int(actual, expected):
    """Score a dict of integer values. Partial credit per matching key."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return 0.0
    if not expected:
        return 1.0
    matched = 0
    for key, ref_val in expected.items():
        if key not in actual:
            continue
        try:
            if int(actual[key]) == int(ref_val):
                matched += 1
        except (TypeError, ValueError):
            pass
    return matched / len(expected)


def _score_top_customers(actual, expected, rtol=NUMERIC_RTOL):
    """Score top_10_customers array. Order-sensitive name + value match."""
    if not isinstance(actual, list) or not isinstance(expected, list):
        return 0.0
    if not expected:
        return 1.0
    matched = 0
    for i, ref_entry in enumerate(expected):
        if i >= len(actual):
            break
        act_entry = actual[i]
        ref_name = ref_entry.get("name", "").strip().lower()
        act_name = act_entry.get("name", "").strip().lower()
        name_ok = ref_name == act_name
        val_ok = _numeric_match(
            act_entry.get("lifetime_value", -1),
            ref_entry.get("lifetime_value", 0),
            rtol,
        )
        if name_ok and val_ok:
            matched += 1
    return matched / len(expected)


def score_metrics(agent_json_str, ref_json_str):
    """Compare agent output JSON against reference metrics.

    Returns dict with 'score' (float 0-1) and 'details' (per-field scores).
    """
    try:
        agent = json.loads(agent_json_str)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Failed to parse agent output JSON: %s", exc)
        return {"score": 0.0, "details": {"error": f"parse_error: {exc}"}}

    try:
        ref = json.loads(ref_json_str)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Failed to parse reference JSON: %s", exc)
        return {"score": 0.0, "details": {"error": f"ref_parse_error: {exc}"}}

    # Hard gate: all required top-level keys must be present
    required_keys = [
        "total_revenue",
        "total_completed_orders",
        "average_order_value",
        "revenue_by_category",
        "monthly_revenue",
        "top_10_customers",
        "orders_by_device",
        "orders_by_channel",
        "event_funnel",
        "conversion_rate",
        "unique_purchasing_customers",
    ]

    missing = [k for k in required_keys if k not in agent]
    if missing:
        logger.error("Agent output missing required keys: %s", missing)
        return {"score": 0.0, "details": {"missing_keys": missing}}

    # Score each field independently
    details = {}

    details["total_revenue"] = _score_scalar_numeric(
        agent["total_revenue"], ref["total_revenue"]
    )
    details["total_completed_orders"] = _score_exact_int(
        agent["total_completed_orders"], ref["total_completed_orders"]
    )
    details["average_order_value"] = _score_scalar_numeric(
        agent["average_order_value"], ref["average_order_value"]
    )
    details["revenue_by_category"] = _score_dict_numeric(
        agent["revenue_by_category"], ref["revenue_by_category"]
    )
    details["monthly_revenue"] = _score_dict_numeric(
        agent["monthly_revenue"], ref["monthly_revenue"]
    )
    details["top_10_customers"] = _score_top_customers(
        agent["top_10_customers"], ref["top_10_customers"]
    )
    details["orders_by_device"] = _score_dict_int(
        agent["orders_by_device"], ref["orders_by_device"]
    )
    details["orders_by_channel"] = _score_dict_int(
        agent["orders_by_channel"], ref["orders_by_channel"]
    )
    details["event_funnel"] = _score_dict_int(
        agent["event_funnel"], ref["event_funnel"]
    )
    details["conversion_rate"] = _score_scalar_numeric(
        agent["conversion_rate"], ref["conversion_rate"]
    )
    details["unique_purchasing_customers"] = _score_exact_int(
        agent["unique_purchasing_customers"], ref["unique_purchasing_customers"]
    )

    score = sum(details.values()) / len(details)

    return {"score": round(score, 4), "details": details}
