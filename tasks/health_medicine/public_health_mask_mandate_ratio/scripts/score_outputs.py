"""Deterministic scorer for public_health_mask_mandate_ratio."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EPS_ROUND3 = 0.001
INT_KEYS = ("n_pairs", "n_unique_counties")
RATIO_KEYS = ("ratio_week2", "ratio_week4", "ratio_week6", "ratio_avg_6wk")


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"bool not accepted as int: {value}")
    if isinstance(value, int):
        return value
    raise TypeError(f"non-numeric value: {value!r}")


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"bool not accepted as float: {value}")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"non-numeric value: {value!r}")


def score_submission(submission: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    needed = [*INT_KEYS, *RATIO_KEYS]
    extra = sorted(set(submission.keys()) - set(needed))
    if extra:
        return {
            "score": 0.0,
            "passed": False,
            "error": f"unexpected keys: {extra}",
            "per_key": {},
        }
    missing = [key for key in needed if key not in submission]
    if missing:
        return {
            "score": 0.0,
            "passed": False,
            "error": f"missing keys: {missing}",
            "per_key": {},
        }

    per_key: dict[str, Any] = {}
    all_match = True

    for key in INT_KEYS:
        expected = _coerce_int(reference[key])
        actual = _coerce_int(submission[key])
        ok = actual == expected
        per_key[key] = {
            "agent": actual,
            "reference": expected,
            "match": ok,
            "rule": "exact integer",
        }
        all_match = all_match and ok

    for key in RATIO_KEYS:
        expected = _coerce_float(reference[key])
        actual = _coerce_float(submission[key])
        agent_r2 = round(actual, 2)
        ref_r2 = round(expected, 2)
        agent_r3 = round(actual, 3)
        ref_r3 = round(expected, 3)
        c1 = agent_r2 == ref_r2
        c2 = abs(agent_r3 - ref_r3) <= EPS_ROUND3 + 1e-9
        ok = c1 and c2
        per_key[key] = {
            "agent": actual,
            "reference": expected,
            "round2_agent": agent_r2,
            "round2_ref": ref_r2,
            "C1_round2_match": c1,
            "round3_agent": agent_r3,
            "round3_ref": ref_r3,
            "round3_diff": round(abs(agent_r3 - ref_r3), 4),
            "C2_round3_within_0.001": c2,
            "match": ok,
        }
        all_match = all_match and ok

    n_keys = len(INT_KEYS) + len(RATIO_KEYS)
    matched = sum(1 for v in per_key.values() if v["match"])
    score = matched / n_keys

    return {
        "score": score,
        "passed": bool(all_match),
        "rule": "per-key scoring: each key contributes 1/6; ratios require round(agent,2)==round(reference,2) and |round(agent,3)-round(reference,3)|<=0.001",
        "per_key": per_key,
    }


def score_output_bundle(*, results_path: Path, reference_path: Path) -> dict[str, Any]:
    try:
        submission = json.loads(results_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"score": 0.0, "passed": False, "error": f"missing submission: {results_path}"}
    except json.JSONDecodeError as exc:
        return {"score": 0.0, "passed": False, "error": f"invalid submission JSON: {exc}"}

    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"score": 0.0, "passed": False, "error": f"missing reference: {reference_path}"}
    except json.JSONDecodeError as exc:
        return {"score": 0.0, "passed": False, "error": f"invalid reference JSON: {exc}"}

    try:
        return score_submission(submission, reference)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return {"score": 0.0, "passed": False, "error": str(exc)}
