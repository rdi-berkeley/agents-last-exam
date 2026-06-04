"""General subagent persistent session engine.

Replaces the flat 5-step ``run_general_subagent`` loop with a session-backed
engine that owns its own ``SessionManager``, ``ContextOverflowCallback``, and
in-place compaction pipeline. The LLM-only subagent becomes a first-class
session that can legitimately run for many turns, survive context pressure,
and persist a per-subagent transcript on disk.

Adapted from OpenClaw's ``pi-embedded-runner`` architecture, where every
subagent is a full session sharing the main agent's compaction pipeline. The
only OpenClaw difference for subagents is ``promptMode="minimal"``
(``pi-embedded-runner/compact.ts:688-691``); our subagent prompt is already
minimal by design.

Differences from OpenClaw:
- No gateway streaming, no ``previous_response_id`` chaining (no Responses
  API session in this path).
- No multi-provider auth or MCP runtime.
- Single-process asyncio model — ``asyncio.CancelledError`` plumbing is
  cooperative and handled by the wrapper in ``subagent_general``.

Mirrors ``OpenClawComputerAgent`` (``agent_loop.py``) for compaction shape,
minus Computer/Responses-API machinery; reuses the VM-agnostic
``compact_messages()`` and ``ContextOverflowCallback`` directly.
"""

from __future__ import annotations

import asyncio
import base64
import json as _json
import logging
import os
from pathlib import Path
from typing import Any

from ..model.model_config import ResolvedModel, resolve_model
from agent.tools.base import BaseTool

from ..context.context import (
    ContextOverflowCallback,
    compact_messages,
    is_context_overflow_error,
)
from ..session import SessionManager
from .subagent_registry import SubagentRegistry, SubagentUsage

DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_COMPACTIONS = 10**9

logger = logging.getLogger(__name__)


def _build_subagent_system_prompt(task: str) -> str:
    """Build a focused worker system prompt.

    Adapted from OpenClaw's ``buildSubagentSystemPrompt``
    (``openclaw/src/agents/subagent-announce.ts:47-104``).
    """
    return "\n".join([
        "# Subagent Context",
        "",
        "You are a **focused worker subagent** spawned by the main agent for a specific task.",
        "",
        "## Your Task",
        f"- {task}",
        "",
        "## Rules",
        "1. **Stay focused** - Do your assigned task, nothing else",
        "2. **Complete and return** - Your final message is automatically reported to the main agent",
        "3. **Don't initiate** - No heartbeats, no proactive actions, no side quests",
        "4. **No computer actions** - You cannot interact with the desktop; use only the tools provided",
        "5. **Be ephemeral** - You may be terminated after task completion. That's fine.",
        "",
        "## Output Format",
        "When complete, respond with:",
        "- What you accomplished or found",
        "- Any relevant details the main agent should know",
        "- Keep it concise but informative",
        "",
        "## What You DON'T Do",
        "- NO user conversations (that's the main agent's job)",
        "- NO computer/mouse/keyboard actions",
        "- NO external messages",
        "- NO pretending to be the main agent",
    ])


# Tools the general subagent CAN use.
ALLOWED_TOOL_NAMES = frozenset({
    "analyze_image",
    "memory_search",
    "memory_get",
    "memory_write",
})

# Tools explicitly excluded (even if passed in).
EXCLUDED_TOOL_NAMES = frozenset({
    "computer",
    "milestone",
    "delegate_general",
    "delegate_gui",
    "subagents",
})


def _filter_tools(tools: list) -> list[BaseTool]:
    """Keep only BaseTool instances whose name is in ``ALLOWED_TOOL_NAMES``."""
    filtered: list[BaseTool] = []
    for tool in tools:
        if not isinstance(tool, BaseTool):
            continue
        name = getattr(tool, "name", None)
        if name is None:
            continue
        if name in EXCLUDED_TOOL_NAMES:
            continue
        if name in ALLOWED_TOOL_NAMES:
            filtered.append(tool)
    return filtered


def _tools_to_litellm_schema(tools: list[BaseTool]) -> list[dict[str, Any]]:
    """Convert BaseTool instances to litellm function-calling schema."""
    return [{"type": "function", "function": tool.function} for tool in tools]


def _build_initial_user_content(
    task: str, screenshot_paths: list[str] | None
) -> str | list[dict[str, Any]]:
    """Build the subagent's initial user-message content.

    With no screenshots attached, returns the task as a plain string so the
    no-screenshot path (the original behavior) is byte-identical.
    With screenshots, returns a list-content block with the text + one
    ``image_url`` entry per path; unreadable paths become a
    ``[screenshot unavailable: <basename>]`` text fallback so a single bad
    path does not fail the whole spawn.
    """
    if not screenshot_paths:
        return task
    blocks: list[dict[str, Any]] = [{"type": "text", "text": task}]
    for path in screenshot_paths:
        block = _encode_image_url_from_path(path)
        if block is None:
            blocks.append({
                "type": "text",
                "text": f"[screenshot unavailable: {Path(path).name}]",
            })
        else:
            blocks.append(block)
    return blocks


def _encode_image_url_from_path(path: str) -> dict[str, Any] | None:
    """Read a PNG from disk and encode as an OpenAI-compatible image_url block.

    Returns ``None`` when the file is missing, unreadable, or empty so callers
    can emit a fallback text block instead of failing the whole spawn. Shape
    mirrors ``subagent_gui._encode_screenshot``.
    """
    try:
        data = Path(path).read_bytes()
    except (OSError, ValueError):
        return None
    if not data:
        return None
    b64 = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


class GeneralSubagentSession:
    """Persistent LLM-only session for a general subagent run.

    Mirrors the loop shape of ``OpenClawComputerAgent`` minus Computer tools
    and Responses API machinery. Owns its own ``SessionManager``, transcript,
    overflow callback, and compaction counter.

    Public attributes (read by the wrapper + tests):
        usage: Accumulated billing tokens across all LLM calls.
        session_mgr: Subagent-scoped transcript manager.
        overflow_cb: Per-session token budget callback.
        compaction_count: Number of compactions performed.
    """

    def __init__(
        self,
        *,
        run_id: str,
        task: str,
        model: str,
        tools: list,
        registry: SubagentRegistry,
        summary_model: str,
        parent_session_dir: Path,
        memory_store: Any | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_compactions: int = DEFAULT_MAX_COMPACTIONS,
        thinking_params: dict[str, Any] | None = None,
        summary_runtime: ResolvedModel | None = None,
        initial_screenshot_paths: list[str] | None = None,
    ) -> None:
        self._run_id = run_id
        self._task = task
        self._model = model
        self._registry = registry
        self._summary_model = summary_model
        self._memory_store = memory_store
        self._max_steps = max_steps
        self._max_compactions = max_compactions
        self._thinking_params = thinking_params or {}
        self._summary_runtime = summary_runtime

        # Public counters / handles.
        self.usage = SubagentUsage()
        self.compaction_count = 0
        self._inbox: asyncio.Queue[str] = asyncio.Queue()

        # Subagent-scoped session — transcript at
        # <parent_session_dir>/subagents/<run_id>/transcript.jsonl
        self.session_mgr = SessionManager(
            task_id=run_id,
            base_dir=Path(parent_session_dir) / "subagents",
        )
        self.session_mgr.init_session(model=model)

        # Build system prompt + initial messages.
        self._system_prompt = _build_subagent_system_prompt(task)
        initial_user_content = _build_initial_user_content(
            task, initial_screenshot_paths
        )
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": initial_user_content},
        ]

        # Transcribe the initial user task so the subagent transcript records
        # *what* was delegated, not just the model's responses.
        self.session_mgr.append_message("user", task)

        # Per-session overflow callback. Honors CONTEXT_WINDOW_OVERRIDE for
        # testing parity with the main agent (openclaw_agent.py:208-215).
        ctx_override = os.environ.get("CONTEXT_WINDOW_OVERRIDE")
        self.overflow_cb = ContextOverflowCallback(
            model=model,
            context_window=int(ctx_override) if ctx_override else None,
            instructions_tokens=len(self._system_prompt) // 4,
            tag=run_id,
        )

        # Filtered tool list + litellm schema built once; reused every turn.
        self._filtered_tools = _filter_tools(tools)
        self._tool_schemas = _tools_to_litellm_schema(self._filtered_tools)
        self._tool_map = {t.name: t for t in self._filtered_tools}

    @property
    def inbox(self) -> "asyncio.Queue[str]":
        """Steer inbox — exposed so the wrapper can attach it to the registry."""
        return self._inbox

    def _poll_inbox(self) -> None:
        """Consume at most one steer message from the inbox.

        Called once per loop iteration so the single-message-per-turn guard
        is enforced structurally. Additional queued messages are consumed on
        subsequent turns.
        """
        try:
            msg = self._inbox.get_nowait()
        except asyncio.QueueEmpty:
            return
        self._messages.append({"role": "user", "content": msg})
        self.session_mgr.append_message("user", f"[Steer] {msg}")

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    async def run(self) -> str:
        """Execute the session loop.

        Returns:
            Final assistant text once the model emits a message with no
            tool calls, or a sentinel string if ``max_steps`` is reached.

        Raises:
            Whatever ``litellm.acompletion`` raises if the error is not
            an overflow that compaction can recover from, or if the
            ``max_compactions`` budget is exhausted.
        """
        import litellm

        resolved = resolve_model(self._model)

        for _step in range(self._max_steps):
            # 1. Proactive compaction — fires if a prior turn's
            #    on_llm_start set needs_compaction=True.
            if (
                self.overflow_cb.needs_compaction
                and self.compaction_count < self._max_compactions
            ):
                await self._compact_in_place()

            # 1.5. Poll inbox for steer messages (at most one per turn).
            self._poll_inbox()

            # 2. Pre-call token estimate. Single source of truth for
            #    current_tokens / needs_compaction; also truncates oversized
            #    tool results in-place.
            self._messages = await self.overflow_cb.on_llm_start(self._messages)

            # 3. API call with reactive-overflow retry.
            try:
                response = await self._call_llm(litellm, resolved)
            except Exception as e:
                if (
                    is_context_overflow_error(str(e))
                    and self.compaction_count < self._max_compactions
                ):
                    self.overflow_cb.force_compaction()
                    await self._compact_in_place()
                    continue
                raise

            # 4. Usage accumulation.
            resp_usage = getattr(response, "usage", None)
            if resp_usage is not None:
                self.usage.input_tokens += int(getattr(resp_usage, "prompt_tokens", 0) or 0)
                self.usage.output_tokens += int(getattr(resp_usage, "completion_tokens", 0) or 0)

            choice = response.choices[0]
            assistant_content = choice.message.content or ""
            tool_calls = choice.message.tool_calls

            # 5. Append assistant message to transcript + in-memory list.
            self._append_assistant(assistant_content, tool_calls)

            # 6. Terminal: no tool calls => final text response.
            if not tool_calls:
                return assistant_content.strip()

            # 7. Execute each tool call inline.
            for tc in tool_calls:
                tool_result = self._execute_tool_call(tc)
                self._append_tool_result(tc, tool_result)

        # Loop exhausted.
        return "(subagent reached max steps without a final response)"

    async def _call_llm(self, litellm_mod, resolved: ResolvedModel):
        """Invoke litellm.acompletion with the current message + tool state."""
        kwargs: dict[str, Any] = {
            "model": resolved.model,
            "messages": self._messages,
            "temperature": 1.0,
            **self._thinking_params,
        }
        if self._tool_schemas:
            kwargs["tools"] = self._tool_schemas
        return await litellm_mod.acompletion(**kwargs)

    # ------------------------------------------------------------------
    # Tool execution + transcript helpers
    # ------------------------------------------------------------------

    def _execute_tool_call(self, tc: Any) -> str:
        """Run a single tool call inline and return its string result."""
        tool_name = tc.function.name
        tool_args = tc.function.arguments
        tool = self._tool_map.get(tool_name)

        if tool is None:
            return f"Error: tool '{tool_name}' is not available to this subagent."
        try:
            result = tool.call(tool_args)
            if not isinstance(result, str):
                result = _json.dumps(result)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def _append_assistant(
        self,
        text: str,
        tool_calls: list[Any] | None,
    ) -> None:
        """Append assistant turn to in-memory messages + transcript."""
        # In-memory chat-completions message (kept verbatim for litellm).
        msg: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        self._messages.append(msg)

        # Canonical transcript blocks (text + function_call entries).
        blocks: list[dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})
        if tool_calls:
            for tc in tool_calls:
                blocks.append({
                    "type": "function_call",
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })
        # Preserve the message even when assistant text is empty so call/result
        # pairing in the transcript stays intact.
        if not blocks:
            blocks = [{"type": "text", "text": ""}]

        usage_dict = {
            "input": self.usage.input_tokens,
            "output": self.usage.output_tokens,
            "total": self.usage.input_tokens + self.usage.output_tokens,
        }
        self.session_mgr.append_message(
            "assistant",
            blocks,
            usage=usage_dict,
            stop_reason="tool_use" if tool_calls else None,
        )

    def _append_tool_result(self, tc: Any, result: str) -> None:
        """Append tool-result turn to in-memory messages + transcript."""
        # In-memory chat-completions tool message.
        self._messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result),
        })

        # Canonical transcript block.
        self.session_mgr.append_message(
            "tool",
            [{
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": str(result),
            }],
        )

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def _compact_in_place(self) -> None:
        """Compact the in-memory message list in place.

        Excludes the system prompt from compaction, runs the shared
        ``compact_messages()`` pipeline on the rest, then rebuilds
        ``self._messages`` as ``[system, user(summary), ...kept]``.

        Persists a compaction entry to the subagent's transcript.
        """
        # System prompt is index 0; never compact or summarize it.
        system_msg = self._messages[0] if self._messages else None
        body = self._messages[1:] if self._messages else []

        if not body:
            self.overflow_cb.reset_after_compaction()
            return

        result = await compact_messages(
            body,
            self._summary_model,
            self.overflow_cb.context_window,
            instructions_tokens=len(self._system_prompt) // 4,
            thinking_params=self._thinking_params or None,
            summary_runtime=self._summary_runtime,
        )

        # Persist compaction entry to transcript with firstKeptEntryId.
        history = self.session_mgr.load_history()
        msg_entries = [e for e in history if e.type == "message"]
        if result.first_kept_message_index < len(msg_entries):
            first_kept_id = msg_entries[result.first_kept_message_index].id
        elif msg_entries:
            first_kept_id = msg_entries[-1].id
        else:
            first_kept_id = "unknown"
        self.session_mgr.append_compaction(
            result.summary,
            first_kept_id,
            result.tokens_before,
        )

        # Rebuild in-memory messages: system + summary user msg + kept body.
        kept = body[result.first_kept_message_index:]
        rebuilt: list[dict[str, Any]] = []
        if system_msg is not None:
            rebuilt.append(system_msg)
        rebuilt.append({
            "role": "user",
            "content": f"[Compaction summary]\n{result.summary}",
        })
        rebuilt.extend(kept)
        self._messages = rebuilt

        self.overflow_cb.reset_after_compaction()
        self.compaction_count += 1
        logger.info(
            "[Subagent %s] In-place compaction #%d (%d->~%d tokens)",
            self._run_id, self.compaction_count,
            result.tokens_before, result.tokens_after,
        )
