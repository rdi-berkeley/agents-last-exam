"""Scoring helpers for amr_panpred_logistic_regression."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    predicted_value: float | None
    target_value: float
    absolute_error: float | None
    full_credit_tolerance: float
    partial_credit_tolerance: float
    partial_credit_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_first_float(text: str) -> float:
    match = FLOAT_PATTERN.search(text or "")
    if not match:
        raise ValueError("no parseable float found in output")
    return float(match.group(0))


def score_output_text(*, output_text: str, reference_json_text: str) -> ScoreResult:
    reference = json.loads(reference_json_text)
    target_value = float(reference["target_value"])
    full_credit_tolerance = float(reference["full_credit_tolerance"])
    partial_credit_tolerance = float(reference["partial_credit_tolerance"])
    partial_credit_score = float(reference["partial_credit_score"])

    predicted_value = _parse_first_float(output_text)
    absolute_error = abs(predicted_value - target_value)

    if absolute_error <= full_credit_tolerance:
        return ScoreResult(
            score=1.0,
            passed=True,
            reason="prediction is within full-credit tolerance",
            predicted_value=predicted_value,
            target_value=target_value,
            absolute_error=absolute_error,
            full_credit_tolerance=full_credit_tolerance,
            partial_credit_tolerance=partial_credit_tolerance,
            partial_credit_score=partial_credit_score,
        )
    if absolute_error <= partial_credit_tolerance:
        return ScoreResult(
            score=partial_credit_score,
            passed=False,
            reason="prediction is within partial-credit tolerance only",
            predicted_value=predicted_value,
            target_value=target_value,
            absolute_error=absolute_error,
            full_credit_tolerance=full_credit_tolerance,
            partial_credit_tolerance=partial_credit_tolerance,
            partial_credit_score=partial_credit_score,
        )
    return ScoreResult(
        score=0.0,
        passed=False,
        reason="prediction is outside tolerance",
        predicted_value=predicted_value,
        target_value=target_value,
        absolute_error=absolute_error,
        full_credit_tolerance=full_credit_tolerance,
        partial_credit_tolerance=partial_credit_tolerance,
        partial_credit_score=partial_credit_score,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an AMR PanPred output file.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--reference-file", required=True)
    args = parser.parse_args()

    result = score_output_text(
        output_text=Path(args.output_file).read_text(encoding="utf-8"),
        reference_json_text=Path(args.reference_file).read_text(encoding="utf-8"),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
