"""Scoring helpers for ecg_superclass_ptbxl."""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScoreResult:
    score: float
    macro_f1: float
    passed: bool
    reason: str
    per_label_f1: dict[str, float]
    row_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_csv(text: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    return fieldnames, list(reader)


def _is_binary(value: str) -> bool:
    return value in {"0", "1"}


def _f1(agent_values: list[int], reference_values: list[int]) -> float:
    tp = fp = fn = 0
    for pred, gold in zip(agent_values, reference_values):
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 1:
            fn += 1
    denom = (2 * tp) + fp + fn
    return 0.0 if denom == 0 else (2.0 * tp) / denom


def score_prediction_tables(
    *,
    agent_csv: str,
    reference_csv: str,
    expected_labels: list[str],
    threshold: float,
) -> ScoreResult:
    expected_columns = ["ecg_id", *expected_labels]

    agent_fieldnames, agent_rows = _parse_csv(agent_csv)
    reference_fieldnames, reference_rows = _parse_csv(reference_csv)

    if reference_fieldnames != expected_columns:
        return ScoreResult(
            score=0.0,
            macro_f1=0.0,
            passed=False,
            reason="reference_schema_mismatch",
            per_label_f1={},
            row_count=len(agent_rows),
        )

    if agent_fieldnames != expected_columns:
        return ScoreResult(
            score=0.0,
            macro_f1=0.0,
            passed=False,
            reason="agent_schema_mismatch",
            per_label_f1={},
            row_count=len(agent_rows),
        )

    if len(agent_rows) != len(reference_rows):
        return ScoreResult(
            score=0.0,
            macro_f1=0.0,
            passed=False,
            reason="row_count_mismatch",
            per_label_f1={},
            row_count=len(agent_rows),
        )

    reference_ids = [row["ecg_id"] for row in reference_rows]
    agent_ids = [row["ecg_id"] for row in agent_rows]

    if len(set(agent_ids)) != len(agent_ids):
        return ScoreResult(
            score=0.0,
            macro_f1=0.0,
            passed=False,
            reason="duplicate_agent_ids",
            per_label_f1={},
            row_count=len(agent_rows),
        )

    if agent_ids != reference_ids:
        return ScoreResult(
            score=0.0,
            macro_f1=0.0,
            passed=False,
            reason="ecg_id_mismatch",
            per_label_f1={},
            row_count=len(agent_rows),
        )

    for row in agent_rows:
        for label in expected_labels:
            if not _is_binary(row[label]):
                return ScoreResult(
                    score=0.0,
                    macro_f1=0.0,
                    passed=False,
                    reason=f"non_binary_value:{label}",
                    per_label_f1={},
                    row_count=len(agent_rows),
                )

    per_label_f1: dict[str, float] = {}
    for label in expected_labels:
        agent_values = [int(row[label]) for row in agent_rows]
        reference_values = [int(row[label]) for row in reference_rows]
        per_label_f1[label] = _f1(agent_values, reference_values)

    macro_f1 = sum(per_label_f1.values()) / len(expected_labels)
    passed = macro_f1 >= threshold
    score = 1.0 if passed else max(0.0, macro_f1 / threshold)

    return ScoreResult(
        score=score,
        macro_f1=macro_f1,
        passed=passed,
        reason="ok" if passed else "below_threshold",
        per_label_f1=per_label_f1,
        row_count=len(agent_rows),
    )
