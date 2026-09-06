"""Translate a CAR ``run-task`` transcript (JSONL) into ATIF Steps.

The transcript is a stream of newline-delimited JSON records, one per event,
discriminated by ``type``:

  - ``run_start``   {goal, model, max_turns, servers, tool_count}
  - ``agent_turn``  {turn, text, tool_calls:[{id,name,arguments}], usage:{input_tokens,output_tokens}}
  - ``observation`` {turn, results:[{tool_call_id, content, is_error}]}
  - ``run_end``     {status, turns, answer, error}

Mapping to ATIF:
  agent_turn  -> source="agent"        (message=text, tool_calls, metrics)
  observation -> source="environment"  (observation.results -> ToolResult[])
  run_start / run_end -> recorded on trajectory.extra (run_end also a system step)

The framework seeds the leading ``user`` instruction step and calls
``builder.finalize`` with the grader's reward; this translator only appends steps.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ale_run.base_interface import (
    ContentPart,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

logger = logging.getLogger(__name__)


def _find_transcript(work_dir: Path) -> Path | None:
    """Canonical location first, then a shallow scan as a fallback."""
    canonical = work_dir / "car" / "transcript.jsonl"
    if canonical.is_file():
        return canonical
    matches = sorted(work_dir.rglob("transcript.jsonl"))
    return matches[0] if matches else None


def _read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("car transcript: skipping malformed line in %s", path)
    return records


def parse_transcript_into(work_dir: Path, builder: TrajectoryBuilder) -> None:
    """Append ATIF Steps to ``builder`` from the CAR transcript in ``work_dir``."""
    path = _find_transcript(work_dir)
    if path is None:
        builder.add_step(
            source="system",
            message=f"car: no transcript.jsonl under {work_dir}",
            extra={"reason": "no_transcript"},
        )
        return

    for rec in _read_records(path):
        kind = rec.get("type")
        if kind == "run_start":
            builder.trajectory.extra.setdefault("car", {})["run_start"] = {
                "model": rec.get("model"),
                "servers": rec.get("servers"),
                "tool_count": rec.get("tool_count"),
                "max_turns": rec.get("max_turns"),
            }
        elif kind == "agent_turn":
            _agent_turn(builder, rec)
        elif kind == "observation":
            _observation(builder, rec)
        elif kind == "run_end":
            builder.trajectory.extra.setdefault("car", {})["run_end"] = {
                "status": rec.get("status"),
                "turns": rec.get("turns"),
                "answer": rec.get("answer"),
                "error": rec.get("error"),
            }
            if rec.get("error"):
                builder.add_step(
                    source="system",
                    message=f"car run ended: {rec.get('status')} — {rec.get('error')}",
                    extra={"reason": "run_end_error"},
                )


def _agent_turn(builder: TrajectoryBuilder, rec: dict) -> None:
    tool_calls = [
        ToolCall(
            id=call.get("id") or f"call_{rec.get('turn')}_{i}",
            name=call.get("name", ""),
            arguments=call.get("arguments") or {},
        )
        for i, call in enumerate(rec.get("tool_calls") or [])
    ]
    usage = rec.get("usage") or {}
    metrics = StepMetrics(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )
    text = rec.get("text") or None
    builder.add_step(
        source="agent",
        message=text,
        tool_calls=tool_calls,
        metrics=metrics,
        extra={"turn": rec.get("turn")},
    )


def _observation(builder: TrajectoryBuilder, rec: dict) -> None:
    results = [
        ToolResult(
            tool_call_id=r.get("tool_call_id", ""),
            content=[ContentPart(type="text", text=str(r.get("content", "")))],
            is_error=bool(r.get("is_error")),
        )
        for r in rec.get("results") or []
    ]
    builder.add_step(
        source="environment",
        observation=Observation(results=results),
        extra={"turn": rec.get("turn")},
    )
