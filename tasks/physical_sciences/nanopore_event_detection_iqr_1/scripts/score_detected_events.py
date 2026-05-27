"""Scoring helpers for nanopore_event_detection_iqr_1."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from typing import Any

EXPECTED_EVENT_COUNT_TOLERANCE = 4
MEAN_RELATIVE_TOLERANCE = 0.05


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "reason": self.reason,
            "hard_gate": self.hard_gate,
            **self.details,
        }


def _fail(reason: str, *, hard_gate: bool, **details: Any) -> ScoreResult:
    return ScoreResult(
        score=0.0,
        passed=False,
        reason=reason,
        hard_gate=hard_gate,
        details=details,
    )


def _safe_rel_error(value: float, target: float) -> float:
    if target == 0:
        return 0.0 if value == 0 else math.inf
    return abs(value - target) / abs(target)


def score_detected_events_csv(
    *,
    agent_csv: str,
    summary_json: str,
    expected_columns: list[str],
) -> ScoreResult:
    try:
        reader = csv.DictReader(io.StringIO(agent_csv))
    except Exception as exc:
        return _fail("csv_parse_error", hard_gate=True, error=str(exc))

    if reader.fieldnames != expected_columns:
        return _fail(
            "wrong_columns",
            hard_gate=True,
            expected_columns=expected_columns,
            actual_columns=reader.fieldnames,
        )

    try:
        summary = json.loads(summary_json)
    except json.JSONDecodeError as exc:
        return _fail("summary_json_parse_error", hard_gate=True, error=str(exc))

    rows = list(reader)
    if not rows:
        return _fail("empty_output", hard_gate=True)

    numeric_rows = []
    start_times = []

    for idx, row in enumerate(rows, start=1):
        try:
            event_id = int(row["event_id"])
            start_time_ms = float(row["start_time_ms"])
            end_time_ms = float(row["end_time_ms"])
            dwell_time_ms = float(row["dwell_time_ms"])
            blockade_amplitude_pA = float(row["blockade_amplitude_pA"])
            baseline_pA = float(row["baseline_pA"])
            fractional_blockade = float(row["fractional_blockade"])
            peak_time_ms = float(row["peak_time_ms"])
        except (TypeError, ValueError) as exc:
            return _fail("non_numeric_value", hard_gate=True, row_index=idx, error=str(exc))

        if event_id != idx:
            return _fail(
                "event_id_not_sequential",
                hard_gate=True,
                row_index=idx,
                observed_event_id=event_id,
            )

        values = [
            start_time_ms,
            end_time_ms,
            dwell_time_ms,
            blockade_amplitude_pA,
            baseline_pA,
            fractional_blockade,
            peak_time_ms,
        ]
        if not all(math.isfinite(value) for value in values):
            return _fail("non_finite_value", hard_gate=True, row_index=idx)

        if dwell_time_ms <= 0:
            return _fail("non_positive_dwell", hard_gate=True, row_index=idx)
        if end_time_ms < start_time_ms:
            return _fail("end_before_start", hard_gate=True, row_index=idx)
        if peak_time_ms < start_time_ms or peak_time_ms > end_time_ms:
            return _fail("peak_outside_event", hard_gate=True, row_index=idx)

        start_times.append(start_time_ms)
        numeric_rows.append(
            {
                "dwell_time_ms": dwell_time_ms,
                "blockade_amplitude_pA": blockade_amplitude_pA,
                "baseline_pA": baseline_pA,
                "fractional_blockade": fractional_blockade,
            }
        )

    if start_times != sorted(start_times):
        return _fail("not_sorted_by_start_time", hard_gate=True)

    mean_dwell_time_ms = sum(row["dwell_time_ms"] for row in numeric_rows) / len(numeric_rows)
    mean_blockade_amplitude_pA = sum(
        row["blockade_amplitude_pA"] for row in numeric_rows
    ) / len(numeric_rows)

    if mean_dwell_time_ms > 5:
        return _fail(
            "mean_dwell_time_too_large",
            hard_gate=True,
            mean_dwell_time_ms=mean_dwell_time_ms,
        )
    if mean_blockade_amplitude_pA > 2000:
        return _fail(
            "mean_blockade_amplitude_too_large",
            hard_gate=True,
            mean_blockade_amplitude_pA=mean_blockade_amplitude_pA,
        )

    target_event_count = int(summary["total_events"])
    target_mean_dwell_time_ms = float(summary["mean_dwell_time_ms"])
    target_mean_blockade_amplitude_pA = float(summary["mean_blockade_amplitude_pA"])

    event_count = len(rows)
    event_count_pass = abs(event_count - target_event_count) <= EXPECTED_EVENT_COUNT_TOLERANCE
    dwell_rel_error = _safe_rel_error(mean_dwell_time_ms, target_mean_dwell_time_ms)
    blockade_rel_error = _safe_rel_error(
        mean_blockade_amplitude_pA,
        target_mean_blockade_amplitude_pA,
    )
    dwell_pass = dwell_rel_error <= MEAN_RELATIVE_TOLERANCE
    blockade_pass = blockade_rel_error <= MEAN_RELATIVE_TOLERANCE

    score = 0.0
    if event_count_pass:
        score += 0.4
    if dwell_pass:
        score += 0.3
    if blockade_pass:
        score += 0.3

    return ScoreResult(
        score=score,
        passed=score >= 1.0,
        reason="ok" if score >= 1.0 else "below_threshold",
        hard_gate=False,
        details={
            "event_count": event_count,
            "target_event_count": target_event_count,
            "event_count_pass": event_count_pass,
            "mean_dwell_time_ms": mean_dwell_time_ms,
            "target_mean_dwell_time_ms": target_mean_dwell_time_ms,
            "dwell_relative_error": dwell_rel_error,
            "dwell_pass": dwell_pass,
            "mean_blockade_amplitude_pA": mean_blockade_amplitude_pA,
            "target_mean_blockade_amplitude_pA": target_mean_blockade_amplitude_pA,
            "blockade_relative_error": blockade_rel_error,
            "blockade_pass": blockade_pass,
        },
    )
