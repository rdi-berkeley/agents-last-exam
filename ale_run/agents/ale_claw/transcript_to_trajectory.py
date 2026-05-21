"""Parse OpenClaw on-disk transcripts into ALE-v1.0 Trajectory steps.

OpenClaw (the harness in :mod:`ale.agents.ale_claw.harness`) writes per-run
artifacts under ``<work_dir>/openclaw_sessions/<task_id>/``:

  ├── transcript.jsonl      — append-only JSONL: session header, message entries,
  │                            compaction entries
  └── state.json            — running totals (token usage, step count,
                              compaction count, model)

And per-turn API payloads under
``<work_dir>/trajectories/<traj_id>/turn_NNN/<NNNN>_api_result.json``
(LiteLLM-shape OpenAI response — preserves cache token breakdown).

This module parses those into ALE Steps via :class:`TrajectoryBuilder`.
**No InteractionLog / InteractionStep intermediate** — we emit ATIF Steps
directly. Logic largely ported from agenthle's
``orchestration/agents/ale_claw/transcript_log.py``; key differences:

  - No agenthle deps — only stdlib + ``ale.agents.trajectory``.
  - Drop the ``InteractionLog.save(...)`` step. Caller's :meth:`collect`
    drives the trajectory builder directly.
  - Per-turn assistant messages collapse into ONE :class:`Step`
    (text → ``message``, thinking → ``reasoning``, function_calls →
    ``tool_calls``).
  - Per-message usage → :class:`StepMetrics` on that step.
  - Aggregated totals (state.json + per-turn cache) land in
    ``builder.trajectory.extra["ale_claw"]["usage"]`` for downstream
    consumers that want exact accounting (the default
    ``builder.finalize`` sum lacks cache_read/cache_write since the
    transcript itself doesn't carry them).
"""
from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

from ale_run.base_interface import (
    ContentPart,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

logger = logging.getLogger(__name__)

# Subdirs the harness writes under our work_dir. Mirror the agenthle layout
# so transcript shape is unchanged.
_SESSIONS_SUBDIR = "openclaw_sessions"
_TRAJECTORIES_SUBDIR = "trajectories"


# =============================================================================
# Public entry point
# =============================================================================

def parse_transcripts_into(work_dir: Path, builder: TrajectoryBuilder) -> None:
    """Walk OpenClaw artifacts under ``work_dir`` → emit Steps into ``builder``.

    Idempotent on partial / missing data: emits a single ``system`` step
    when no transcript is found, otherwise appends one Step per assistant
    turn + one per tool reply, then writes aggregate usage to
    ``builder.trajectory.extra["ale_claw"]["usage"]``.
    """
    sessions_root = work_dir / _SESSIONS_SUBDIR
    transcripts = sorted(sessions_root.glob("*/transcript.jsonl")) if sessions_root.is_dir() else []
    if not transcripts:
        builder.add_step(
            source="system",
            message="ale-claw: no transcript at "
                    f"{sessions_root}/*/transcript.jsonl",
            extra={"reason": "no_transcript", "expected_root": str(sessions_root)},
        )
        return

    for path in transcripts:
        _parse_one_transcript(path, builder)

    aggregated = _aggregate_usage(work_dir)
    builder.trajectory.extra.setdefault("ale_claw", {})["usage"] = aggregated
    builder.trajectory.extra["ale_claw"]["raw_transcript"] = str(transcripts[0])


# =============================================================================
# Transcript walker
# =============================================================================

def _parse_one_transcript(path: Path, builder: TrajectoryBuilder) -> None:
    """Walk one ``transcript.jsonl`` file → append Steps to builder."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:                      # noqa: BLE001
        logger.warning("ale-claw: failed to read %s: %s", path, exc)
        return

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "message":
            continue
        _consume_message(entry, builder)


def _consume_message(entry: dict[str, Any], builder: TrajectoryBuilder) -> None:
    """One transcript ``{"type":"message", ...}`` entry → one Step.

    Assistant messages become one ``agent`` Step that may carry text
    (``message``), thinking (``reasoning``), and tool_calls (joined
    function_call blocks). Tool messages become one ``environment`` Step
    with all tool_results under ``observation.results``.
    """
    msg = entry.get("message") or {}
    role = msg.get("role")
    content = msg.get("content")
    if not isinstance(content, list):
        return
    usage = msg.get("usage") or {}
    stop_reason = msg.get("stopReason")

    if role == "assistant":
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                t = block.get("text") or ""
                if t:
                    text_parts.append(t)
            elif btype == "thinking":
                # Upstream uses {"type":"thinking","thinking":"..."} — falls
                # back to "content" when emitted via newer SDK code paths.
                t = block.get("thinking") or block.get("content") or ""
                if t:
                    reasoning_parts.append(t)
            elif btype == "function_call":
                raw_args = block.get("arguments") or ""
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    args = {"_raw": raw_args}
                if not isinstance(args, dict):
                    args = {"value": args}
                tool_calls.append(ToolCall(
                    id=block.get("id") or "",
                    name=block.get("name") or "",
                    arguments=args,
                ))

        # Skip empty messages (no text / no thinking / no tool_call) to keep
        # the trajectory tight.
        if not (text_parts or reasoning_parts or tool_calls):
            return

        builder.add_step(
            source="agent",
            message="\n".join(text_parts) if text_parts else None,
            reasoning="\n".join(reasoning_parts) if reasoning_parts else None,
            tool_calls=tool_calls,
            metrics=_metrics_from_message_usage(usage),
            extra={"stop_reason": stop_reason} if stop_reason else None,
        )
        return

    if role == "tool":
        results: list[ToolResult] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            raw = block.get("content")
            text = _normalize_tool_result_content(raw)
            results.append(ToolResult(
                tool_call_id=block.get("tool_use_id") or "",
                content=[ContentPart(type="text", text=text)],
                is_error=bool(block.get("is_error", False)),
            ))
        if results:
            builder.add_step(
                source="environment",
                observation=Observation(results=results),
            )
        return

    # Other roles (user re-injects from compaction etc.) — log to extra and skip.
    builder.trajectory.extra.setdefault("ale_claw", {}).setdefault(
        "skipped_messages", []
    ).append({"role": role, "content_blocks": len(content)})


def _metrics_from_message_usage(usage: dict[str, Any]) -> StepMetrics | None:
    """OpenClaw transcript usage → :class:`StepMetrics`.

    OpenClaw's per-message usage is ``{"input": N, "output": N, "total": N,
    "cost": F}`` (note the unsuffixed key names). cache_read/cache_write are
    NOT in the transcript — they live in the per-turn ``api_result.json``
    and are surfaced via :func:`_aggregate_usage` into trajectory.extra.
    """
    if not usage:
        return None
    in_t = usage.get("input")
    out_t = usage.get("output")
    cost = usage.get("cost")
    if in_t is None and out_t is None and cost is None:
        return None
    return StepMetrics(
        input_tokens=int(in_t) if in_t is not None else None,
        output_tokens=int(out_t) if out_t is not None else None,
        cost_usd=float(cost) if cost is not None else None,
    )


# =============================================================================
# Tool-result content normalization
# =============================================================================

def _normalize_tool_result_content(raw: Any) -> str:
    """Best-effort convert OpenClaw's tool_result content to JSON text.

    OpenClaw stores tool_result content as ``str(dict)`` (Python repr) rather
    than ``json.dumps(dict)``, e.g. ``"{'success': True, ...}"`` instead of
    ``{"success": true, ...}``. Re-serialize as JSON for downstream
    consumers (matches the InteractionLog the agenthle wrapper used to
    write).

    Pass-through (no rewrite) when:
    - input is not a str (already structured — caller json.dumps it)
    - input doesn't look like a Python repr (treat as plain text)
    - literal_eval fails (treat as plain text — covers shell output,
      stack traces, etc.)
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        return json.dumps(raw, ensure_ascii=False)
    s = raw.strip()
    if not (s.startswith("{") or s.startswith("[")):
        return raw
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError):
        return raw
    if not isinstance(parsed, (dict, list)):
        return raw
    return json.dumps(parsed, ensure_ascii=False, indent=2)


# =============================================================================
# Aggregate usage (state.json + per-turn api_result.json)
# =============================================================================

def _aggregate_usage(work_dir: Path) -> dict[str, Any]:
    """Sum total tokens (state.json) + cache breakdown (per-turn api_result.json).

    state.json is the in-memory session_mgr accumulator and is incremented for
    EVERY yielded step out of ``agent.run()``, including helper / compaction /
    VLM calls that bypass the transcript message writer. Cache aggregation
    walks every per-turn ``api_result.json`` so it captures those same
    helper calls' cache breakdown — disjoint partition only balances when
    overall_input_tokens is sourced from state.json (not transcript).
    """
    state_in, state_out = _aggregate_state_json_tokens(work_dir)
    cache_read, cache_write = _aggregate_cache_tokens(work_dir)
    msg_in, msg_out, msg_cost = _aggregate_message_usage(work_dir)

    in_t = state_in or msg_in
    out_t = state_out or msg_out
    uncached_in = max(in_t - cache_read - cache_write, 0)

    out: dict[str, Any] = {
        "uncached_input_tokens": uncached_in,
        "output_tokens": out_t,
        "overall_input_tokens": in_t,
    }
    if msg_cost > 0:
        out["total_cost_usd"] = round(msg_cost, 6)
    if cache_read > 0:
        out["cache_read_input_tokens"] = cache_read
    if cache_write > 0:
        out["cache_write_input_tokens"] = cache_write
    return out


def _aggregate_state_json_tokens(work_dir: Path) -> tuple[int, int]:
    """Read main-agent input/output token totals from each session's ``state.json``."""
    sessions_root = work_dir / _SESSIONS_SUBDIR
    if not sessions_root.is_dir():
        return 0, 0
    in_t = out_t = 0
    for path in sorted(sessions_root.glob("*/state.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tt = data.get("total_tokens") or {}
        in_t += int(tt.get("input_tokens", 0) or 0)
        out_t += int(tt.get("output_tokens", 0) or 0)
    return in_t, out_t


def _aggregate_cache_tokens(work_dir: Path) -> tuple[int, int]:
    """Sum cached_tokens / cache_write_tokens from per-turn API result dumps."""
    trajectories_root = work_dir / _TRAJECTORIES_SUBDIR
    if not trajectories_root.is_dir():
        return 0, 0
    cache_read = cache_write = 0
    for path in trajectories_root.glob("*/turn_*/[0-9]*_api_result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        usage = (data.get("result") or {}).get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cache_read += int(details.get("cached_tokens") or 0)
        cache_write += int(details.get("cache_write_tokens") or 0)
    return cache_read, cache_write


def _aggregate_message_usage(work_dir: Path) -> tuple[int, int, float]:
    """Fallback: sum (input, output, cost) once per assistant message."""
    sessions_root = work_dir / _SESSIONS_SUBDIR
    if not sessions_root.is_dir():
        return 0, 0, 0.0
    in_t = out_t = 0
    cost = 0.0
    for path in sorted(sessions_root.glob("*/transcript.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "message":
                continue
            msg = entry.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            usage = msg.get("usage") or {}
            in_t += int(usage.get("input") or 0)
            out_t += int(usage.get("output") or 0)
            c = usage.get("cost")
            if c:
                cost += float(c)
    return in_t, out_t, cost
