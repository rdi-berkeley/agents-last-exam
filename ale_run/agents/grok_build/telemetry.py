"""Normalize Grok Build's native session telemetry into ALE artifacts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any


def recover_telemetry_artifacts(
    work_dir: Path,
    *,
    events_file: Path | None,
    updates_file: Path | None,
    terminal_event: dict[str, Any] | None = None,
) -> int:
    """Write normalized events and a compact run summary.

    Grok Build already records first-token, tool, MCP, and aggregate API timing
    events, so no injected HTTP instrumentation is needed.
    """
    events = list(_json_lines(events_file))
    updates = list(_json_lines(updates_file))
    _write_jsonl_atomic(work_dir / "telemetry.jsonl", events)

    terminal = terminal_event or {}
    usage, session_id = _terminal_usage(updates, terminal)
    normalized_usage = _normalize_usage(usage)
    if normalized_usage.get("model_usage") is None:
        normalized_usage["model_usage"] = terminal.get("modelUsage")
    summary = {
        "event_count": len(events),
        "events_by_type": dict(Counter(str(event.get("type") or "unknown") for event in events)),
        "session_id": session_id,
        "request_id": _string_or_none(terminal.get("requestId")),
        "stop_reason": terminal.get("stopReason"),
        "num_turns": terminal.get("num_turns"),
        "total_cost_usd": terminal.get("total_cost_usd"),
        "llm_calls": _summarize_llm_calls(events),
        "usage": normalized_usage,
        "tool_executions": _summarize_tools(events),
        "artifacts": {
            "events": str(events_file) if events_file else None,
            "updates": str(updates_file) if updates_file else None,
            "normalized_events": "telemetry.jsonl",
        },
    }
    (work_dir / "telemetry_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return len(events)


def _summarize_llm_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    model: str | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "turn_started":
            model = _string_or_none(event.get("model_id"))
            continue
        if event_type == "loop_started":
            if active is not None:
                _finish_call(active, event.get("ts"))
                calls.append(active)
            active = {
                "sequence": len(calls) + 1,
                "loop_index": event.get("loop_index"),
                "model": model,
                "started_at": event.get("ts"),
                "first_token_at": None,
                "completed_at": None,
            }
            continue
        if active is None:
            continue
        if event_type == "first_token" and active["first_token_at"] is None:
            active["first_token_at"] = event.get("ts")
        elif event_type in {"tool_started", "turn_ended"}:
            _finish_call(active, event.get("ts"))
            calls.append(active)
            active = None
    if active is not None:
        _finish_call(active, None)
        calls.append(active)
    return calls


def _finish_call(call: dict[str, Any], completed_at: Any) -> None:
    call["completed_at"] = completed_at
    started_ms = _timestamp_ms(call.get("started_at"))
    first_ms = _timestamp_ms(call.get("first_token_at"))
    completed_ms = _timestamp_ms(completed_at)
    call["first_token_ms"] = (
        round(first_ms - started_ms, 3) if first_ms is not None and started_ms is not None else None
    )
    call["stream_duration_ms"] = (
        round(completed_ms - first_ms, 3)
        if completed_ms is not None and first_ms is not None
        else None
    )
    call["duration_ms"] = (
        round(completed_ms - started_ms, 3)
        if completed_ms is not None and started_ms is not None
        else None
    )


def _summarize_tools(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    executions: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    mcp: list[dict[str, Any]] = []
    for event in events:
        event_type = event.get("type")
        if event_type == "tool_started":
            native.append(
                {
                    "kind": "native",
                    "tool_name": event.get("tool_name"),
                    "started_at": event.get("ts"),
                }
            )
        elif event_type == "tool_completed":
            tool = _pop_matching(native, "tool_name", event.get("tool_name"))
            tool.update(
                completed_at=event.get("ts"),
                duration_ms=event.get("duration_ms"),
                success=event.get("outcome") == "success",
                outcome=event.get("outcome"),
            )
            executions.append(tool)
        elif event_type == "mcp_tool_call_started":
            mcp.append(
                {
                    "kind": "mcp",
                    "server_name": event.get("server_name"),
                    "tool_name": _mcp_tool_name(event),
                    "call_id": event.get("call_id"),
                    "started_at": event.get("ts"),
                }
            )
        elif event_type == "mcp_tool_call_completed":
            tool = _pop_matching(mcp, "tool_name", _mcp_tool_name(event))
            tool.update(
                completed_at=event.get("ts"),
                duration_ms=event.get("duration_ms"),
                success=bool(event.get("success")),
                is_timeout=bool(event.get("is_timeout")),
                reconnect_attempted=bool(event.get("reconnect_attempted")),
                auth_retry_attempted=bool(event.get("auth_retry_attempted")),
            )
            executions.append(tool)
    for sequence, execution in enumerate(executions, 1):
        execution["sequence"] = sequence
    return executions


def _pop_matching(
    active: list[dict[str, Any]],
    key: str,
    value: Any,
) -> dict[str, Any]:
    for index in range(len(active) - 1, -1, -1):
        if active[index].get(key) == value:
            return active.pop(index)
    return {key: value, "started_at": None}


def _mcp_tool_name(event: dict[str, Any]) -> str:
    server = str(event.get("server_name") or "")
    tool = str(event.get("tool_name") or event.get("call_id") or "")
    return f"{server}__{tool}" if server and not tool.startswith(f"{server}__") else tool


def _terminal_usage(
    updates: list[dict[str, Any]],
    terminal_event: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    for event in reversed(updates):
        params = event.get("params") or {}
        update = params.get("update") or {}
        if update.get("sessionUpdate") == "turn_completed":
            usage = update.get("usage")
            return (
                usage if isinstance(usage, dict) else {},
                _string_or_none(params.get("sessionId")),
            )
    usage = terminal_event.get("usage")
    return (
        usage if isinstance(usage, dict) else {},
        _string_or_none(terminal_event.get("sessionId")),
    )


def _normalize_usage(usage: dict[str, Any]) -> dict[str, Any]:
    if "input_tokens" in usage:
        return {
            "input_tokens": usage.get("input_tokens"),
            "cache_read_tokens": usage.get("cache_read_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "model_calls": usage.get("model_calls"),
            "api_duration_ms": usage.get("api_duration_ms"),
            "model_usage": usage.get("modelUsage"),
        }
    full_input = usage.get("inputTokens")
    cache_read = usage.get("cachedReadTokens")
    input_tokens = (
        max(0, full_input - cache_read)
        if isinstance(full_input, int) and isinstance(cache_read, int)
        else full_input
    )
    return {
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read,
        "output_tokens": usage.get("outputTokens"),
        "reasoning_tokens": usage.get("reasoningTokens"),
        "total_tokens": usage.get("totalTokens"),
        "model_calls": usage.get("modelCalls"),
        "api_duration_ms": usage.get("apiDurationMs"),
        "model_usage": usage.get("modelUsage"),
    }


def _timestamp_ms(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp() * 1000
    except ValueError:
        return None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _json_lines(path: Path | None) -> Iterator[dict[str, Any]]:
    if path is None:
        return
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return
    with handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
