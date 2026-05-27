"""Local scorer for engineering/opensim_spring_pendulum_dynamics_1."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OUTPUT_KEYS = (
    "shoulder_at_0_5s_deg",
    "elbow_at_0_5s_deg",
    "shoulder_at_1_0s_deg",
    "elbow_at_1_0s_deg",
    "shoulder_at_1_5s_deg",
    "elbow_at_1_5s_deg",
)

TOLERANCES_DEG = {
    "shoulder_at_0_5s_deg": 0.5,
    "elbow_at_0_5s_deg": 0.5,
    "shoulder_at_1_0s_deg": 0.5,
    "elbow_at_1_0s_deg": 0.5,
    "shoulder_at_1_5s_deg": 1.0,
    "elbow_at_1_5s_deg": 1.0,
}


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hard_fail(reason: str, *, details: dict[str, Any] | None = None) -> ScoreResult:
    return ScoreResult(
        score=0.0,
        passed=False,
        reason=reason,
        hard_gate=reason,
        details=details or {},
    )


def _load_output_payload(text: str, *, label: str) -> tuple[dict[str, float], str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"{label}_invalid_json:{exc.msg}"

    if not isinstance(payload, dict):
        return {}, f"{label}_not_object"

    payload_keys = set(payload.keys())
    required_keys = set(OUTPUT_KEYS)
    if payload_keys != required_keys:
        return (
            {},
            f"{label}_key_set_mismatch",
        )

    normalized: dict[str, float] = {}
    for key in OUTPUT_KEYS:
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return {}, f"{label}_non_numeric_value:{key}"
        number = float(value)
        if not math.isfinite(number):
            return {}, f"{label}_non_finite_value:{key}"
        normalized[key] = number

    return normalized, None


def score_output_texts(
    *,
    candidate_output_text: str,
    reference_output_text: str,
) -> ScoreResult:
    candidate, candidate_error = _load_output_payload(candidate_output_text, label="candidate")
    if candidate_error:
        return _hard_fail(candidate_error)

    reference, reference_error = _load_output_payload(reference_output_text, label="reference")
    if reference_error:
        return _hard_fail(reference_error)

    checks: dict[str, dict[str, Any]] = {}
    passed_count = 0
    for key in OUTPUT_KEYS:
        candidate_value = candidate[key]
        reference_value = reference[key]
        tolerance = TOLERANCES_DEG[key]
        abs_error = abs(candidate_value - reference_value)
        passed = abs_error <= tolerance
        if passed:
            passed_count += 1
        checks[key] = {
            "candidate": candidate_value,
            "reference": reference_value,
            "abs_error": abs_error,
            "tolerance_deg": tolerance,
            "passed": passed,
        }

    score = passed_count / len(OUTPUT_KEYS)
    return ScoreResult(
        score=score,
        passed=passed_count == len(OUTPUT_KEYS),
        reason="ok" if passed_count == len(OUTPUT_KEYS) else "tolerance_mismatch",
        hard_gate=None,
        details={
            "passed_checks": passed_count,
            "total_checks": len(OUTPUT_KEYS),
            "checks": checks,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Candidate results.json path")
    parser.add_argument("--reference", required=True, help="Reference results.json path")
    args = parser.parse_args()

    result = score_output_texts(
        candidate_output_text=Path(args.candidate).read_text(encoding="utf-8"),
        reference_output_text=Path(args.reference).read_text(encoding="utf-8"),
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
