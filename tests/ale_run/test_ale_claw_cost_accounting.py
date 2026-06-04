"""Regression tests for ale_claw token/cost accounting in ``final_metrics``.

Covers the bug investigated in
``docs/investigations/ale-claw-token-cost-accounting.md``: the per-step
``StepMetrics`` sum that builds ``final_metrics`` drops the prompt-cache split
(the transcript never carries it) and the final/helper turns (they bypass the
transcript writer), so cost and tokens were under-reported. The fix reconciles
``final_metrics`` from the authoritative aggregate (state.json tokens + per-turn
``api_result.json`` cache/cost) via ``TrajectoryBuilder.override_final_metrics``.

These modules are pure pydantic/stdlib — no harness / MLX imports — so this file
does not depend on the top-level conftest shims.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ale_run.agents.ale_claw.transcript_to_trajectory import parse_transcripts_into
from ale_run.base_interface import TrajectoryBuilder


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _build_work_dir(tmp_path: Path) -> Path:
    """Synthetic ale_claw work_dir mirroring the demo-smoke shape.

    The transcript intentionally records only the FIRST of two assistant turns
    (the final "done" turn bypasses the transcript writer) and carries no cache
    split — exactly the lossy inputs the per-step sum sees. state.json holds the
    true token totals; the two ``api_result.json`` dumps hold the cache split and
    the true per-call cost (including the turn missing from the transcript).
    """
    wd = tmp_path / "ale-claw"
    session = wd / "openclaw_sessions" / "sess0"

    # Transcript: ONE assistant message only (missing the final turn), no cache.
    transcript = session / "transcript.jsonl"
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text(
        json.dumps(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "step 1"}],
                    "usage": {"input": 100, "output": 10, "total": 110, "cost": 0.001},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # state.json: true totals across BOTH turns (the accumulator sees every step).
    _write(
        session / "state.json",
        {"total_tokens": {"input_tokens": 300, "output_tokens": 25}},
    )

    # Per-turn api_result dumps: usage under result.usage, with the cache split
    # and authoritative cost. Turn 1 is the one missing from the transcript.
    traj = wd / "trajectories" / "traj0"
    _write(
        traj / "turn_000" / "0001_api_result.json",
        {
            "result": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "cost": 0.001,
                    "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 90},
                }
            }
        },
    )
    _write(
        traj / "turn_001" / "0002_api_result.json",
        {
            "result": {
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 15,
                    "cost": 0.004,
                    "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 95},
                }
            }
        },
    )
    return wd


def test_final_metrics_reconciled_from_aggregate(tmp_path: Path) -> None:
    wd = _build_work_dir(tmp_path)
    builder = TrajectoryBuilder(agent_name="ale_claw", task_path="t", variant_index=0)
    parse_transcripts_into(wd, builder)
    fm = builder.finalize(reward=1.0, status="completed").final_metrics

    # Totals from state.json (both turns), not the single transcript turn (input=100).
    assert fm.total_input_tokens == 300
    assert fm.total_output_tokens == 25
    # Cache split from api_result (transcript carried none → would be 0 pre-fix).
    assert fm.total_cache_read_tokens == 100
    assert fm.total_cache_creation_tokens == 185  # 90 + 95
    # Cost summed over per-call api_result, incl. the turn missing from transcript.
    # Pre-fix this was 0.001 (transcript-message cost only).
    assert fm.total_cost_usd == pytest.approx(0.005)


def test_extra_usage_cost_uses_api_result_not_transcript(tmp_path: Path) -> None:
    wd = _build_work_dir(tmp_path)
    builder = TrajectoryBuilder(agent_name="ale_claw", task_path="t", variant_index=0)
    parse_transcripts_into(wd, builder)
    usage = builder.trajectory.extra["ale_claw"]["usage"]

    assert usage["overall_input_tokens"] == 300
    assert usage["cache_read_input_tokens"] == 100
    assert usage["cache_write_input_tokens"] == 185
    # The regression that the doc's Option A missed: extra-usage cost must also
    # be the api_result total (0.005), not the transcript-message total (0.001).
    assert usage["total_cost_usd"] == pytest.approx(0.005)


def test_override_no_op_when_no_aggregate(tmp_path: Path) -> None:
    """Degraded run (no sessions/state) keeps the per-step sum, not zeros."""
    builder = TrajectoryBuilder(agent_name="ale_claw", task_path="t", variant_index=0)
    parse_transcripts_into(tmp_path / "empty", builder)  # no transcript → system step
    fm = builder.finalize(reward=None, status="failed").final_metrics
    assert fm.total_input_tokens == 0
    assert fm.total_cost_usd == 0.0


def test_override_final_metrics_rejects_unknown_field() -> None:
    builder = TrajectoryBuilder(agent_name="x", task_path="t", variant_index=0)
    with pytest.raises(ValueError, match="non-overridable"):
        builder.override_final_metrics(total_steps=5)


def test_override_final_metrics_ignores_none_and_beats_step_sum() -> None:
    builder = TrajectoryBuilder(agent_name="x", task_path="t", variant_index=0)
    from ale_run.base_interface import StepMetrics

    builder.add_step("agent", metrics=StepMetrics(input_tokens=10, cost_usd=0.01))
    # None → no-op (output stays from step sum); provided keys win.
    builder.override_final_metrics(total_input_tokens=999, total_output_tokens=None)
    fm = builder.finalize(reward=None, status="completed").final_metrics
    assert fm.total_input_tokens == 999  # overridden
    assert fm.total_cost_usd == pytest.approx(0.01)  # untouched → from step sum
