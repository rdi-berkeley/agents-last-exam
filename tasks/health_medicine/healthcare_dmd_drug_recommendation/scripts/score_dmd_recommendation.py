"""Scorer for the DMD drug recommendation task."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPECTED_DRUG_KEY = "recommended_drug"
REQUIRED_OUTPUT_FILES = ("recommendation.json", "reasoning_trace.json")


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
    return ScoreResult(
        score=0.0,
        passed=False,
        reason=reason,
        hard_gate=reason,
        details=details or {},
    )


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{label}_missing") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}_invalid_json: {exc}") from exc


def _normalize_drug_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _validate_trace(trace: Any) -> list[str]:
    warnings: list[str] = []
    if not isinstance(trace, (list, dict)):
        return ["reasoning_trace_json_should_be_list_or_object"]
    steps = trace.get("steps") if isinstance(trace, dict) else trace
    if not isinstance(steps, list) or not steps:
        return ["reasoning_trace_missing_nonempty_steps"]
    for index, step in enumerate(steps[:3]):
        if not isinstance(step, dict):
            warnings.append(f"trace_step_{index}_not_object")
            continue
        if not any(key in step for key in ("tool", "action", "drug", "observation", "decision")):
            warnings.append(f"trace_step_{index}_missing_tool_or_decision_fields")
    return warnings


def score_output_dir(output_dir: Path, reference_dir: Path) -> ScoreResult:
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (output_dir / name).exists()]
    if missing:
        return _hard_fail("missing_required_output_files", {"missing_files": missing})

    try:
        recommendation = _load_json(output_dir / "recommendation.json", "recommendation_json")
        trace = _load_json(output_dir / "reasoning_trace.json", "reasoning_trace_json")
        answer = _load_json(reference_dir / "answer_key.json", "answer_key_json")
    except ValueError as exc:
        return _hard_fail(str(exc))

    if not isinstance(recommendation, dict):
        return _hard_fail("recommendation_json_must_be_object")
    if EXPECTED_DRUG_KEY not in recommendation:
        return _hard_fail("recommendation_json_missing_recommended_drug")

    predicted = _normalize_drug_name(recommendation.get(EXPECTED_DRUG_KEY))
    expected = _normalize_drug_name(answer.get(EXPECTED_DRUG_KEY))
    trace_warnings = _validate_trace(trace)

    if predicted != expected:
        return ScoreResult(
            score=0.0,
            passed=False,
            reason="incorrect_recommended_drug",
            hard_gate="incorrect_recommended_drug",
            details={
                "predicted": predicted,
                "expected": expected,
                "trace_warnings": trace_warnings,
            },
        )

    return ScoreResult(
        score=1.0,
        passed=True,
        reason="passed",
        hard_gate=None,
        details={
            "predicted": predicted,
            "expected": expected,
            "trace_warnings": trace_warnings,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score DMD recommendation outputs.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = score_output_dir(args.output_dir, args.reference_dir)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
