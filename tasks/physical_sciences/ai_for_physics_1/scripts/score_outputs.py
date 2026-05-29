"""Scoring helpers for physical_sciences/ai_for_physics_1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScoreReport:
    score: float
    candidate_value: float | None
    expected_value: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "candidate_value": self.candidate_value,
            "expected_value": self.expected_value,
            "reason": self.reason,
        }


def _load_single_value(payload: bytes) -> float:
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("result.json must contain a JSON object")
    if set(decoded.keys()) != {"pass_16_acc"}:
        raise ValueError("result.json must contain exactly one key: pass_16_acc")
    value = decoded["pass_16_acc"]
    if not isinstance(value, (int, float)):
        raise ValueError("pass_16_acc must be numeric")
    value = float(value)
    if value < 0.0 or value > 100.0:
        raise ValueError("pass_16_acc must be between 0.0 and 100.0")
    return value


def score_result_payload(*, candidate_bytes: bytes, reference_bytes: bytes) -> ScoreReport:
    expected = _load_single_value(reference_bytes)
    candidate = _load_single_value(candidate_bytes)
    if candidate == expected:
        return ScoreReport(score=1.0, candidate_value=candidate, expected_value=expected, reason="exact_match")
    return ScoreReport(score=0.0, candidate_value=candidate, expected_value=expected, reason="value_mismatch")
