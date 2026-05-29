"""Scoring logic for parametric_energy_simulation task.

Compares agent-produced numeric metrics against reference values
extracted from Grasshopper/Ladybug simulation results.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Metric groups with (weight_per_field, full_credit_tolerance, zero_credit_tolerance)
SCALAR_SPEC: dict[str, tuple[float, float, float]] = {
    # Geometric metrics — 0.09 each, ±2% full, ±10% zero
    "total_floor_area": (0.09, 0.02, 0.10),
    "far": (0.09, 0.02, 0.10),
    "site_coverage": (0.09, 0.02, 0.10),
    "compactness": (0.09, 0.02, 0.10),
    "terrace_floor_ratio": (0.09, 0.02, 0.10),
    # Rectangular reference — 0.05 each
    "far_rectangular": (0.05, 0.02, 0.10),
    "site_coverage_rectangular": (0.05, 0.02, 0.10),
    # Energy metrics — 0.15 each, ±5% full, ±25% zero
    "annual_cooling_kwh": (0.15, 0.05, 0.25),
    "annual_heating_kwh": (0.15, 0.05, 0.25),
}

ARRAY_SPEC: dict[str, tuple[float, float, float]] = {
    # Irradiation arrays — 0.075 each, ±5% full, ±25% zero (per-element, averaged)
    "irradiation_8x": (0.075, 0.05, 0.25),
    "irradiation_4x": (0.075, 0.05, 0.25),
}

REQUIRED_SCALAR_FIELDS = set(SCALAR_SPEC.keys())


def _field_score(agent_val: float, ref_val: float, full_tol: float, zero_tol: float) -> float:
    """Linear score: 1.0 inside full_tol, decay to 0.0 at zero_tol."""
    if ref_val == 0:
        return 1.0 if agent_val == 0 else 0.0
    rel_err = abs(agent_val - ref_val) / abs(ref_val)
    if rel_err <= full_tol:
        return 1.0
    if rel_err >= zero_tol:
        return 0.0
    return (zero_tol - rel_err) / (zero_tol - full_tol)


def score_energy_metrics(
    agent_metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Score agent's parametric energy simulation output.

    Parameters
    ----------
    agent_metrics : parsed JSON dict from agent's output/metrics.json
    reference_metrics : dict of reference values for this variant

    Returns
    -------
    dict with "score" (0.0-1.0), "breakdown", "details".
    """
    breakdown = {}
    details = {}

    # ── Hard gates ─────────────────────────────────────────────────────
    present_scalars = REQUIRED_SCALAR_FIELDS.intersection(agent_metrics.keys())
    if len(present_scalars) < len(REQUIRED_SCALAR_FIELDS):
        missing = REQUIRED_SCALAR_FIELDS - present_scalars
        details["hard_gate"] = f"Missing scalar fields: {missing}"
        return {"score": 0.0, "breakdown": {}, "details": details}

    # ── Score scalar metrics ───────────────────────────────────────────
    total_weighted = 0.0
    field_details = {}

    for field_name, (weight, full_tol, zero_tol) in SCALAR_SPEC.items():
        ref_val = reference_metrics.get(field_name)
        agent_val = agent_metrics.get(field_name)

        if ref_val is None or agent_val is None:
            field_details[field_name] = {"score": 0.0, "reason": "missing"}
            continue

        try:
            agent_f = float(agent_val)
            ref_f = float(ref_val)
        except (ValueError, TypeError):
            field_details[field_name] = {"score": 0.0, "reason": "non-numeric"}
            continue

        fs = _field_score(agent_f, ref_f, full_tol, zero_tol)
        total_weighted += fs * weight
        field_details[field_name] = {
            "score": round(fs, 4),
            "agent": agent_f,
            "reference": ref_f,
            "rel_error": round(abs(agent_f - ref_f) / abs(ref_f), 6) if ref_f != 0 else None,
        }

    breakdown["scalar_metrics"] = round(total_weighted, 4)

    # ── Score array metrics ────────────────────────────────────────────
    array_weighted = 0.0
    for field_name, (weight, full_tol, zero_tol) in ARRAY_SPEC.items():
        ref_arr = reference_metrics.get(field_name, [])
        agent_arr = agent_metrics.get(field_name, [])

        if not ref_arr or not agent_arr:
            field_details[field_name] = {"score": 0.0, "reason": "missing or empty array"}
            continue

        if not isinstance(agent_arr, list):
            field_details[field_name] = {"score": 0.0, "reason": "not a list"}
            continue

        # Compare element by element, up to the shorter length
        n = min(len(ref_arr), len(agent_arr))
        if n == 0:
            field_details[field_name] = {"score": 0.0, "reason": "empty array"}
            continue

        elem_scores = []
        for i in range(n):
            try:
                a = float(agent_arr[i])
                r = float(ref_arr[i])
                elem_scores.append(_field_score(a, r, full_tol, zero_tol))
            except (ValueError, TypeError):
                elem_scores.append(0.0)

        # Penalty for length mismatch
        length_ratio = n / max(len(ref_arr), 1)
        arr_score = (sum(elem_scores) / len(elem_scores)) * length_ratio

        array_weighted += arr_score * weight
        field_details[field_name] = {
            "score": round(arr_score, 4),
            "matched_elements": n,
            "expected_elements": len(ref_arr),
        }

    breakdown["array_metrics"] = round(array_weighted, 4)
    details["fields"] = field_details

    # ── Final score ────────────────────────────────────────────────────
    final = total_weighted + array_weighted

    return {
        "score": round(min(max(final, 0.0), 1.0), 4),
        "breakdown": breakdown,
        "details": details,
    }
