"""Shared parser and scorer for the historical_retrieval task."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GradeResult:
    """Normalized evaluation result."""

    score: float
    parsed_predictions: list[Any]
    expected: list[Any]
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None
    reason: str


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_exact_number(text: str) -> list[int]:
    """Parse a scalar numeric answer."""
    normalized = _normalize_space(text)
    if not normalized:
        return []
    if re.fullmatch(r"\d+", normalized):
        return [int(normalized)]
    numbers = re.findall(r"\b\d+\b", normalized)
    if len(numbers) == 1 and re.sub(r"\b\d+\b", "", normalized).strip() == "":
        return [int(numbers[0])]
    return []


def parse_exact_page(text: str) -> list[int]:
    """Backward-compatible alias for scalar numeric parsing."""
    return parse_exact_number(text)


def parse_exact_text(text: str) -> list[str]:
    """Parse an exact free-text answer after whitespace normalization."""
    normalized = _normalize_space(text)
    return [normalized] if normalized else []


def parse_csv_values(text: str) -> list[str]:
    """Parse comma-separated scalar values from free text."""
    normalized = _normalize_space(text)
    if not normalized:
        return []
    return [part.strip() for part in normalized.split(",") if part.strip()]


def parse_tuple2(text: str) -> list[tuple[int, int]]:
    """Parse `(page, year)` tuples from free text."""
    return [
        (int(a), int(b))
        for a, b in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text or "")
    ]


def parse_tuple3(text: str) -> list[tuple[int, int, int]]:
    """Parse `(start, end, year)` tuples from free text."""
    return [
        (int(a), int(b), int(c))
        for a, b, c in re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", text or "")
    ]


def _multiset_counts(predicted: list[Any], expected: list[Any]) -> tuple[int, int, int]:
    pred_counter = Counter(predicted)
    exp_counter = Counter(expected)
    tp = sum(min(pred_counter[item], exp_counter[item]) for item in pred_counter)
    fp = max(sum(pred_counter.values()) - tp, 0)
    fn = max(sum(exp_counter.values()) - tp, 0)
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def grade_answer(answer_text: str, scoring_mode: str, expected: list[Any] | int | str) -> GradeResult:
    """Grade one answer against one hidden expected payload."""
    if scoring_mode == "exact_number":
        parsed_predictions = parse_exact_number(answer_text)
        expected_number = int(expected)
        score = 1.0 if parsed_predictions == [expected_number] else 0.0
        return GradeResult(
            score=score,
            parsed_predictions=parsed_predictions,
            expected=[expected_number],
            tp=1 if score == 1.0 else 0,
            fp=0 if score == 1.0 else len(parsed_predictions),
            fn=0 if score == 1.0 else 1,
            precision=None,
            recall=None,
            f1=None,
            reason="exact_match" if score == 1.0 else "expected exactly one matching number",
        )

    if scoring_mode == "exact_text":
        parsed_predictions = parse_exact_text(answer_text)
        expected_text = str(expected)
        score = 1.0 if parsed_predictions == [expected_text] else 0.0
        return GradeResult(
            score=score,
            parsed_predictions=parsed_predictions,
            expected=[expected_text],
            tp=1 if score == 1.0 else 0,
            fp=0 if score == 1.0 else len(parsed_predictions),
            fn=0 if score == 1.0 else 1,
            precision=None,
            recall=None,
            f1=None,
            reason="exact_match" if score == 1.0 else "expected exact text answer",
        )

    if scoring_mode == "csv_values":
        parsed_predictions = parse_csv_values(answer_text)
        expected_values = [str(item).strip() for item in expected]
        tp, fp, fn = _multiset_counts(parsed_predictions, expected_values)
        precision, recall, f1 = _f1(tp, fp, fn)
        return GradeResult(
            score=f1,
            parsed_predictions=parsed_predictions,
            expected=expected_values,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            reason="ok" if parsed_predictions else "no parseable csv values found",
        )

    if scoring_mode == "tuple2":
        parsed_predictions = parse_tuple2(answer_text)
        expected_values = [tuple(item) for item in expected]
    elif scoring_mode == "tuple3":
        parsed_predictions = parse_tuple3(answer_text)
        expected_values = [tuple(item) for item in expected]
    else:
        raise ValueError(f"Unsupported scoring mode: {scoring_mode}")

    tp, fp, fn = _multiset_counts(parsed_predictions, expected_values)
    precision, recall, f1 = _f1(tp, fp, fn)
    return GradeResult(
        score=f1,
        parsed_predictions=parsed_predictions,
        expected=expected_values,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        reason="ok" if parsed_predictions else "no parseable tuples found",
    )


def serialize_grade_result(result: GradeResult) -> str:
    """Serialize a GradeResult as stable pretty JSON."""
    payload = {
        "score": result.score,
        "parsed_predictions": result.parsed_predictions,
        "expected": result.expected,
        "tp": result.tp,
        "fp": result.fp,
        "fn": result.fn,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "reason": result.reason,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", required=True, help="Path to the agent answer file")
    parser.add_argument("--expected-json", required=True, help="Path to reference/ground_truth.json")
    args = parser.parse_args()

    answer_text = Path(args.answer).read_text(encoding="utf-8")
    expected_payload = json.loads(Path(args.expected_json).read_text(encoding="utf-8"))
    result = grade_answer(
        answer_text=answer_text,
        scoring_mode=expected_payload["scoring_mode"],
        expected=expected_payload["expected"],
    )
    print(serialize_grade_result(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
