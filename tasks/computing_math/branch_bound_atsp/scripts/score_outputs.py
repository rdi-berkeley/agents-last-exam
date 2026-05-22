#!/usr/bin/env python
"""Score branch_bound_atsp results.json submissions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


REQUIRED_OUTPUT = "results.json"
REFERENCE_FILE = "results.json"
SEED = 77777
WIND = (0.37, -0.21, 0.58)
WIND_SCALE = 0.15
COST_TOL = 1e-4
TIER3_GAP_LIMIT = 0.005


def generate_coords(n: int) -> list[tuple[float, float, float]]:
    try:
        import numpy as np

        rng = np.random.default_rng(SEED)
        return [tuple(map(float, row)) for row in rng.uniform(0.0, 100.0, size=(n, 3))]
    except Exception:
        # Deterministic fallback is intentionally absent: the benchmark runtime
        # includes NumPy, and using any different RNG would change the instance.
        raise RuntimeError("NumPy is required to regenerate the benchmark instances")


def edge_cost(coords: list[tuple[float, float, float]], i: int, j: int) -> float:
    dx = coords[j][0] - coords[i][0]
    dy = coords[j][1] - coords[i][1]
    dz = coords[j][2] - coords[i][2]
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0.0:
        return math.inf
    wind_dot = (dx / norm) * WIND[0] + (dy / norm) * WIND[1] + (dz / norm) * WIND[2]
    return norm + WIND_SCALE * wind_dot


def tour_cost(tour: list[int], n: int) -> float:
    coords = generate_coords(n)
    total = 0.0
    for idx, city in enumerate(tour):
        total += edge_cost(coords, city, tour[(idx + 1) % n])
    return total


def is_valid_tour(value: Any, n: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == n
        and value[0] == 0
        and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
        and sorted(value) == list(range(n))
    )


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def nonnegative_float(value: Any) -> float | None:
    result = as_float(value)
    if result is None or result < 0:
        return None
    return result


def close(a: float | None, b: float, tol: float = COST_TOL) -> bool:
    return a is not None and abs(a - b) <= tol


def score_exact_tier(
    observed: dict[str, Any],
    reference: dict[str, Any],
    *,
    tier_name: str,
    tour_key: str,
    cost_key: str,
) -> tuple[float, dict[str, Any]]:
    n = int(reference["n_cities"])
    details: dict[str, Any] = {"tier": tier_name, "checks": {}}
    score = 0.0

    details["checks"]["n_cities_matches_schema"] = observed.get("n_cities") == n
    if not details["checks"]["n_cities_matches_schema"]:
        details["passed"] = False
        return 0.0, details

    tour = observed.get(tour_key)
    valid_tour = is_valid_tour(tour, n)
    details["checks"]["valid_tour"] = valid_tour
    if valid_tour:
        score += 0.25
        recomputed = tour_cost(tour, n)
    else:
        recomputed = None
    details["recomputed_cost"] = recomputed

    reported_cost = as_float(observed.get(cost_key))
    cost_matches_tour = recomputed is not None and close(reported_cost, recomputed)
    details["checks"]["reported_cost_matches_tour"] = cost_matches_tour
    if cost_matches_tour:
        score += 0.25

    cost_matches_reference = cost_matches_tour and close(reported_cost, float(reference[cost_key]))
    details["checks"]["cost_matches_hidden_reference"] = cost_matches_reference
    if cost_matches_reference:
        score += 0.25

    bound = as_float(observed.get("best_bound"))
    gap = as_float(observed.get("optimality_gap"))
    bound_certifies = (
        cost_matches_reference
        and close(bound, float(reference[cost_key]))
        and close(gap, 0.0)
        and close(bound, reported_cost if reported_cost is not None else math.inf)
    )
    details["checks"]["bound_certifies_optimality"] = bound_certifies
    if bound_certifies:
        score += 0.25

    metadata_valid = nonnegative_int(observed.get("nodes_explored")) is not None and nonnegative_float(
        observed.get("runtime_seconds")
    ) is not None
    details["checks"]["required_metadata_valid"] = metadata_valid
    if not metadata_valid:
        details["passed"] = False
        return 0.0, details

    details["passed"] = score == 1.0
    return score, details


def score_tier3(observed: dict[str, Any], reference: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    n = int(reference["n_cities"])
    details: dict[str, Any] = {"tier": "tier3", "checks": {}}
    score = 0.0

    details["checks"]["n_cities_matches_schema"] = observed.get("n_cities") == n
    if not details["checks"]["n_cities_matches_schema"]:
        details["passed"] = False
        return 0.0, details

    tour = observed.get("best_tour")
    valid_tour = is_valid_tour(tour, n)
    details["checks"]["valid_tour"] = valid_tour
    if valid_tour:
        score += 0.20
        recomputed = tour_cost(tour, n)
    else:
        recomputed = None
    details["recomputed_cost"] = recomputed

    reported_cost = as_float(observed.get("best_cost"))
    cost_matches_tour = recomputed is not None and close(reported_cost, recomputed)
    details["checks"]["reported_cost_matches_tour"] = cost_matches_tour
    if cost_matches_tour:
        score += 0.20

    reference_cost = float(reference["best_cost"])
    near_reference_incumbent = cost_matches_tour and reported_cost is not None and reported_cost <= reference_cost + COST_TOL
    details["checks"]["incumbent_within_reference_gap"] = near_reference_incumbent
    if near_reference_incumbent:
        score += 0.20

    bound = as_float(observed.get("best_bound"))
    bound_sane = bound is not None and reported_cost is not None and 0.0 < bound <= reported_cost + COST_TOL
    details["checks"]["bound_sane"] = bound_sane
    if bound_sane:
        score += 0.20

    gap = as_float(observed.get("optimality_gap"))
    expected_gap = None if bound is None or reported_cost is None or bound <= 0 else (reported_cost - bound) / bound
    gap_valid = expected_gap is not None and close(gap, expected_gap, 1e-6) and gap <= TIER3_GAP_LIMIT + 1e-9
    details["checks"]["gap_valid"] = gap_valid
    details["expected_gap"] = expected_gap
    if gap_valid:
        score += 0.20

    metadata_valid = (
        nonnegative_int(observed.get("nodes_explored")) is not None
        and nonnegative_int(observed.get("lagrangian_iterations")) is not None
        and nonnegative_float(observed.get("runtime_seconds")) is not None
    )
    details["checks"]["required_metadata_valid"] = metadata_valid
    if not metadata_valid:
        details["passed"] = False
        return 0.0, details

    details["passed"] = score == 1.0
    return score, details


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        errors.append(f"failed to read JSON {path}: {exc}")
        return None


def score_payload(submission: Any, reference: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(submission, dict):
        errors.append("submission must be a JSON object")
        return {"score": 0.0, "pass_fail": False, "errors": errors}
    if not isinstance(reference, dict):
        errors.append("reference must be a JSON object")
        return {"score": 0.0, "pass_fail": False, "errors": errors}

    tier_scores: dict[str, float] = {}
    tier_details: dict[str, Any] = {}
    for tier in ("tier1", "tier2", "tier3"):
        if not isinstance(submission.get(tier), dict):
            errors.append(f"missing or invalid {tier}")
            tier_scores[tier] = 0.0
            tier_details[tier] = {"passed": False, "error": "missing tier"}
            continue
        if not isinstance(reference.get(tier), dict):
            errors.append(f"missing reference {tier}")
            tier_scores[tier] = 0.0
            tier_details[tier] = {"passed": False, "error": "missing reference tier"}
            continue
        if tier == "tier3":
            score, details = score_tier3(submission[tier], reference[tier])
        else:
            score, details = score_exact_tier(
                submission[tier],
                reference[tier],
                tier_name=tier,
                tour_key="optimal_tour",
                cost_key="optimal_cost",
            )
        tier_scores[tier] = score
        tier_details[tier] = details

    weighted = 0.40 * tier_scores.get("tier1", 0.0) + 0.35 * tier_scores.get("tier2", 0.0) + 0.25 * tier_scores.get("tier3", 0.0)
    pass_fail = not errors and tier_scores.get("tier1") == 1.0 and tier_scores.get("tier2") == 1.0 and tier_scores.get("tier3") == 1.0
    return {
        "score": weighted if not errors else 0.0,
        "pass_fail": pass_fail,
        "errors": errors,
        "tier_scores": tier_scores,
        "details": tier_details,
    }


def score(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    output_file = output_dir / REQUIRED_OUTPUT
    if not output_file.is_file():
        return {"score": 0.0, "pass_fail": False, "errors": [f"missing {REQUIRED_OUTPUT}"]}
    submission = load_json(output_file, errors)
    reference = load_json(reference_dir / REFERENCE_FILE, errors)
    if errors:
        return {"score": 0.0, "pass_fail": False, "errors": errors}
    return score_payload(submission, reference)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()
    report = score(Path(args.output), Path(args.reference))
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
