"""Extract task-level metrics from Antigravity CLI's native event stream."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TRANSPORT_RE = re.compile(
    r"[IWEF](?P<month>\d{2})(?P<day>\d{2})\s+"
    r"(?P<clock>\d{2}:\d{2}:\d{2}\.\d+)\s+.*?"
    r"URL:\s+(?P<url>\S*streamGenerateContent\S*)\s+"
    r"Trace:\s+(?P<trace>\S+)(?:\s+ResponseID:\s+(?P<response_id>\S+))?"
)
_MULTIMODAL_TOOL_NAMES = frozenset({"capture_browser_screenshot", "generate_image", "screenshot"})
_IMAGE_PATH_RE = re.compile(
    r"(?:file://)?(?P<path>/[^\s\]\)\"']+\.(?:gif|jpe?g|png|webp))",
    re.IGNORECASE,
)


def build_metrics_summary(
    work_dir: Path,
    *,
    model: str | None,
) -> dict[str, Any]:
    """Write one metrics summary derived from agy's native run artifacts."""
    transcript_path = work_dir / "transcript.jsonl"
    native_log_path = work_dir / "agy_cli.log"
    native_events = list(_iter_jsonl(transcript_path))
    summary = _summarize_native_events(native_events, model=model)
    transport_attempts = _parse_transport_attempts(native_log_path)
    summary["native_stream_present"] = transcript_path.exists()
    summary["native_log_present"] = native_log_path.exists()
    summary["native_log_bytes"] = native_log_path.stat().st_size if native_log_path.exists() else 0
    summary["physical_request_count"] = len(transport_attempts)
    summary["transport_attempts"] = transport_attempts
    summary["availability"]["per_transport_attempt"]["observed"] = bool(transport_attempts)
    summary["artifacts"] = {
        "native_stream": transcript_path.name,
        "native_log": native_log_path.name,
    }
    (work_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _summarize_native_events(
    native_events: list[dict[str, Any]],
    *,
    model: str | None,
) -> dict[str, Any]:
    model_generations: list[dict[str, Any]] = []
    tool_executions: list[dict[str, Any]] = []
    multimodal_groups: list[dict[str, Any]] = []
    multimodal_resources: set[str] = set()
    result: dict[str, Any] = {}
    observed_model: str | None = None
    terminal_tool_steps: set[int] = set()

    for native in native_events:
        if native.get("event") == "init" and isinstance(native.get("init"), dict):
            init_model = native["init"].get("model")
            if isinstance(init_model, str) and init_model:
                observed_model = init_model
        if native.get("event") != "step_update" or not isinstance(native.get("step_update"), dict):
            continue
        update = native["step_update"]
        step_index = _int_or_none(update.get("step_index"))
        if (
            update.get("step_type") == "tool"
            and update.get("state") in {"DONE", "ERROR"}
            and step_index is not None
        ):
            terminal_tool_steps.add(step_index)

    effective_model = observed_model or model

    for native in native_events:
        event_name = native.get("event")
        payload = native.get(event_name)
        if not isinstance(payload, dict):
            if not event_name and any(key in native for key in ("response", "status", "usage")):
                result = native
            continue
        if event_name == "result":
            result = payload
            continue
        if event_name != "step_update" or payload.get("state") not in {"DONE", "ERROR"}:
            continue

        step_index = _int_or_none(payload.get("step_index"))
        usage = payload.get("usage")
        if isinstance(usage, dict) and payload.get("state") == "DONE":
            input_tokens = _int_or_none(usage.get("input_tokens"))
            cache_read_tokens = _int_or_none(usage.get("cache_read_tokens"))
            model_generations.append(
                {
                    "sequence": len(model_generations) + 1,
                    "step_index": step_index,
                    "step_type": payload.get("step_type"),
                    "model": effective_model,
                    "duration_ms": _duration_ms(payload.get("duration_seconds")),
                    "input_tokens": input_tokens,
                    "uncached_input_tokens": input_tokens,
                    "cached_input_tokens": cache_read_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_write_tokens": None,
                    "output_tokens": _int_or_none(usage.get("output_tokens")),
                    "thinking_tokens": _int_or_none(usage.get("thinking_tokens")),
                    "total_tokens": _int_or_none(usage.get("total_tokens")),
                    "request_id": None,
                    "cost_usd": None,
                }
            )

        if payload.get("step_type") != "tool":
            continue
        _append_tool_execution(
            payload,
            tool_executions=tool_executions,
            multimodal_groups=multimodal_groups,
            multimodal_resources=multimodal_resources,
        )

    incomplete_tools: dict[int, dict[str, Any]] = {}
    for native in native_events:
        if native.get("event") != "step_update" or not isinstance(native.get("step_update"), dict):
            continue
        payload = native["step_update"]
        step_index = _int_or_none(payload.get("step_index"))
        if (
            payload.get("step_type") == "tool"
            and payload.get("state") == "ACTIVE"
            and step_index is not None
            and step_index not in terminal_tool_steps
        ):
            incomplete_tools[step_index] = payload
    for payload in incomplete_tools.values():
        _append_tool_execution(
            payload,
            tool_executions=tool_executions,
            multimodal_groups=multimodal_groups,
            multimodal_resources=multimodal_resources,
        )

    result_usage = result.get("usage")
    if not isinstance(result_usage, dict):
        result_usage = {}
    generation_totals = {
        key: sum(generation[key] or 0 for generation in model_generations)
        for key in (
            "input_tokens",
            "uncached_input_tokens",
            "cache_read_tokens",
            "output_tokens",
            "thinking_tokens",
            "total_tokens",
        )
    }
    generation_totals["cached_input_tokens"] = generation_totals["cache_read_tokens"]
    generation_totals["overall_input_tokens"] = (
        generation_totals["uncached_input_tokens"] + generation_totals["cached_input_tokens"]
    )
    result_totals = _usage_totals(result_usage)
    usage_reconciles = all(
        expected is None or generation_totals[key] == expected
        for key, expected in result_totals.items()
    )

    generations_observed = bool(model_generations)
    step_stream_observed = any(native.get("event") == "step_update" for native in native_events)
    availability = {
        "model_generations_observed": generations_observed,
        "request_granularity": "logical_model_generation",
        "input_token_semantics": (
            "input_tokens and uncached_input_tokens exclude cache reads; "
            "overall_input_tokens = input_tokens + cache_read_tokens"
        ),
        "per_model_generation": {
            "duration": generations_observed
            and all(item["duration_ms"] is not None for item in model_generations),
            "output_tokens": generations_observed
            and all(item["output_tokens"] is not None for item in model_generations),
            "cached_input_tokens": generations_observed
            and all(item["cache_read_tokens"] is not None for item in model_generations),
            "uncached_input_tokens": generations_observed
            and all(item["uncached_input_tokens"] is not None for item in model_generations),
            "cache_write_tokens": False,
            "cost_usd": False,
        },
        "per_transport_attempt": {
            "observed": False,
            "duration": False,
            "tokens": False,
        },
        "tool_calls": step_stream_observed,
        "tool_latency": step_stream_observed
        and all(tool["duration_ms"] is not None for tool in tool_executions),
        "multimodal_output_groups": step_stream_observed,
        "generation_usage_reconciles_with_result": usage_reconciles,
        "limitations": [
            "agy does not expose cache-write tokens or request cost",
            "transport retries are visible in agy_cli.log, but their latency and tokens are not",
            "logical model generations and physical transport requests are not one-to-one",
        ],
    }
    return {
        "schema_version": "ALE-agent-metrics-v1",
        "source": "agy-stream-json",
        # Match the Codex/Claude telemetry-summary surface. For agy these are
        # logical model generations exposed by stream-json, not necessarily
        # one-to-one with the physical HTTP attempts below.
        "api_calls": model_generations,
        "api_call_count": len(model_generations),
        "logical_model_generation_count": len(model_generations),
        "tool_executions": tool_executions,
        "tool_call_count": len(tool_executions),
        "tool_latency_ms": sum(tool["duration_ms"] or 0 for tool in tool_executions),
        "multimodal_output_groups": multimodal_groups,
        "multimodal_output_group_count": len(multimodal_groups),
        "logical_usage_totals": generation_totals,
        "result_usage": result_totals,
        "run": {
            "conversation_id": result.get("conversation_id"),
            "status": result.get("status"),
            "duration_ms": _duration_ms(result.get("duration_seconds")),
            "num_turns": _int_or_none(result.get("num_turns")),
            "model": effective_model,
            "configured_model": model,
        },
        "availability": availability,
    }


def _append_tool_execution(
    payload: dict[str, Any],
    *,
    tool_executions: list[dict[str, Any]],
    multimodal_groups: list[dict[str, Any]],
    multimodal_resources: set[str],
) -> None:
    tool_info = payload.get("tool_info")
    if not isinstance(tool_info, dict):
        tool_info = {}
    tool_name = str(payload.get("tool_name") or tool_info.get("name") or "")
    output = tool_info.get("output")
    error = tool_info.get("error")
    state = payload.get("state")
    tool = {
        "sequence": len(tool_executions) + 1,
        "step_index": _int_or_none(payload.get("step_index")),
        "tool_name": tool_name,
        "duration_ms": _duration_ms(payload.get("duration_seconds")),
        "state": state,
        "success": state == "DONE" and error in (None, "") if state != "ACTIVE" else None,
        "parameters": tool_info.get("parameters"),
        "output": output,
        "error": error,
    }
    tool_executions.append(tool)

    detection = _multimodal_detection(tool_name, tool["parameters"], output)
    if detection is not None and detection["resource"] not in multimodal_resources:
        multimodal_resources.add(detection["resource"])
        multimodal_groups.append(
            {
                "sequence": len(multimodal_groups) + 1,
                "step_index": tool["step_index"],
                "tool_sequence": tool["sequence"],
                "tool_name": tool_name,
                "detection": detection["detection"],
                "resource": detection["resource"],
            }
        )


def _usage_totals(usage: dict[str, Any]) -> dict[str, int | None]:
    totals = {
        "input_tokens": _int_or_none(usage.get("input_tokens")),
        "cache_read_tokens": _int_or_none(usage.get("cache_read_tokens")),
        "output_tokens": _int_or_none(usage.get("output_tokens")),
        "thinking_tokens": _int_or_none(usage.get("thinking_tokens")),
        "total_tokens": _int_or_none(usage.get("total_tokens")),
    }
    input_tokens = totals["input_tokens"]
    cache_read_tokens = totals["cache_read_tokens"]
    totals["uncached_input_tokens"] = input_tokens
    totals["cached_input_tokens"] = cache_read_tokens
    totals["overall_input_tokens"] = (
        input_tokens + cache_read_tokens
        if input_tokens is not None and cache_read_tokens is not None
        else None
    )
    return totals


def _parse_transport_attempts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    year = datetime.fromtimestamp(path.stat().st_mtime, UTC).year
    attempts: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _TRANSPORT_RE.search(line)
        if match is None:
            continue
        attempts.append(
            {
                "sequence": len(attempts) + 1,
                "observed_at": (
                    f"{year:04d}-{match.group('month')}-{match.group('day')}T{match.group('clock')}"
                ),
                "observed_at_timezone": None,
                "endpoint": match.group("url"),
                "trace_id": match.group("trace"),
                "response_id": match.group("response_id"),
                "duration_ms": None,
                "tokens": None,
            }
        )
    return attempts


def _multimodal_detection(
    tool_name: str,
    parameters: Any,
    output: Any,
) -> dict[str, str] | None:
    canonical_name = tool_name.rsplit("__", 1)[-1]
    parameter_text = _json_or_value(parameters)
    output_text = _json_or_value(output)
    combined = "\n".join(value for value in (parameter_text, output_text) if isinstance(value, str))
    image_path = _IMAGE_PATH_RE.search(combined)
    if image_path is not None:
        return {
            "detection": "image_resource",
            "resource": image_path.group("path"),
        }
    if canonical_name == "call_mcp_tool" and isinstance(parameters, dict):
        nested_name = str(parameters.get("ToolName") or parameters.get("tool_name") or "")
        if nested_name.rsplit("__", 1)[-1] in _MULTIMODAL_TOOL_NAMES:
            return {
                "detection": "nested_tool_name",
                "resource": f"tool-step:{nested_name}",
            }
    if canonical_name in _MULTIMODAL_TOOL_NAMES:
        return {
            "detection": "tool_name",
            "resource": f"tool-step:{canonical_name}",
        }
    lowered = combined.lower()
    if "data:image/" in lowered or '"media_type":"image/' in lowered:
        return {
            "detection": "output_payload",
            "resource": f"inline-image:{hashlib.sha256(combined.encode()).hexdigest()}",
        }
    return None


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _json_or_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _duration_ms(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(value * 1000)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
