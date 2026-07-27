from __future__ import annotations

import json
from pathlib import Path

from ale_run.agents.grok_build.telemetry import recover_telemetry_artifacts


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_native_telemetry_captures_llm_and_tool_timings(tmp_path: Path) -> None:
    events = tmp_path / "session_events.jsonl"
    updates = tmp_path / "session_updates.jsonl"
    _write_jsonl(
        events,
        [
            {
                "ts": "2026-07-26T01:00:00.000Z",
                "type": "turn_started",
                "model_id": "kimi-k2.5",
            },
            {
                "ts": "2026-07-26T01:00:00.100Z",
                "type": "loop_started",
                "loop_index": 0,
            },
            {"ts": "2026-07-26T01:00:00.600Z", "type": "first_token"},
            {
                "ts": "2026-07-26T01:00:01.000Z",
                "type": "tool_started",
                "tool_name": "use_tool",
            },
            {
                "ts": "2026-07-26T01:00:01.010Z",
                "type": "mcp_tool_call_started",
                "server_name": "cua",
                "tool_name": "screenshot",
                "call_id": "cua__screenshot",
            },
            {
                "ts": "2026-07-26T01:00:01.260Z",
                "type": "mcp_tool_call_completed",
                "server_name": "cua",
                "tool_name": "screenshot",
                "call_id": "cua__screenshot",
                "duration_ms": 250,
                "success": True,
                "is_timeout": False,
            },
            {
                "ts": "2026-07-26T01:00:01.300Z",
                "type": "tool_completed",
                "tool_name": "use_tool",
                "duration_ms": 300,
                "outcome": "success",
            },
        ],
    )
    _write_jsonl(
        updates,
        [
            {
                "method": "session/update",
                "params": {
                    "sessionId": "session-1",
                    "update": {
                        "sessionUpdate": "turn_completed",
                        "usage": {
                            "inputTokens": 900,
                            "cachedReadTokens": 700,
                            "outputTokens": 50,
                            "reasoningTokens": 20,
                            "totalTokens": 950,
                            "modelCalls": 1,
                            "apiDurationMs": 800,
                        },
                    },
                },
            }
        ],
    )

    assert (
        recover_telemetry_artifacts(
            tmp_path,
            events_file=events,
            updates_file=updates,
        )
        == 7
    )

    normalized = [
        json.loads(line) for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()
    ]
    summary = json.loads((tmp_path / "telemetry_summary.json").read_text())
    assert len(normalized) == 7
    assert summary["session_id"] == "session-1"
    assert summary["usage"]["input_tokens"] == 200
    assert summary["usage"]["cache_read_tokens"] == 700
    assert summary["usage"]["api_duration_ms"] == 800
    assert summary["llm_calls"][0]["first_token_ms"] == 500
    assert summary["llm_calls"][0]["duration_ms"] == 900
    assert summary["tool_executions"][0]["tool_name"] == "cua__screenshot"
    assert summary["tool_executions"][0]["duration_ms"] == 250
    assert summary["tool_executions"][1]["tool_name"] == "use_tool"
    assert summary["tool_executions"][1]["success"] is True


def test_missing_native_logs_still_produces_recoverable_summary(tmp_path: Path) -> None:
    count = recover_telemetry_artifacts(
        tmp_path,
        events_file=None,
        updates_file=None,
        terminal_event={
            "sessionId": "session-fallback",
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 20,
                "output_tokens": 5,
            },
        },
    )

    summary = json.loads((tmp_path / "telemetry_summary.json").read_text())
    assert count == 0
    assert summary["session_id"] == "session-fallback"
    assert summary["usage"]["input_tokens"] == 10
    assert summary["event_count"] == 0
