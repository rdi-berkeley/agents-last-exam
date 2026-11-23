"""GUI subagent — blocking vision-to-action relay loop via ComputerAgent.

Routes through ``ComputerAgent`` (the same agent infrastructure as the main
agent) instead of raw ``litellm.acompletion``.  This gives the GUI subagent:
  - Native computer tool registration (provider-specific schema + normalization)
  - Automatic action execution and post-action screenshots
  - ``ImageRetentionCallback`` for image history pruning
  - ``OperatorNormalizerCallback`` for hallucination fixes

The system prompt is intentionally minimal — the computer tool schema
(built by ``ComputerAgent``'s agent loop) provides the action vocabulary.
Only rules and output format guidance are needed.

Design adapted from OpenClaw:
  - subagent-announce.ts:47-104 (buildSubagentSystemPrompt — role, rules,
    output format, ephemeral framing)
  - subagent-registry.ts (register → mark_running → complete/fail lifecycle)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import ComputerAgent
from agent.callbacks.base import AsyncCallbackHandler
from agent.responses import make_tool_error_item
from agent.types import ToolError
from PIL import Image

from ..agent_loop import OpenClawComputerAgent
from ..agent_loop_helpers import _maybe_sanitize_screenshot
from ..context.transcript import group_step_output
from ..memory.memory import MemoryGetTool, MemorySearchTool, MemoryStore, MemoryWriteTool
from ..tools.computer_handler import (
    OpenClawComputerHandler,
    coordinate_space_for_model,
)
from .subagent_registry import (
    SubagentRegistry,
    SubagentUsage,
    _subagent_transcript_path,
)

DEFAULT_MAX_STEPS = 15
DEFAULT_MODEL = "openrouter/openai/gpt-5.4"
DEFAULT_IMAGE_HISTORY = 3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Steer inbox callback
# ---------------------------------------------------------------------------


class SteerInboxCallback(AsyncCallbackHandler):
    """Inject steer messages from an inbox queue between ComputerAgent turns.

    Registered as a ComputerAgent callback. ``on_run_continue`` is called at
    the top of each iteration of ``ComputerAgent.run()``'s while-loop, before
    ``predict_step()``.  It receives ``new_items`` by reference — appending a
    user-role message here makes it visible to the next LLM call.

    Single-message-per-turn guard: consumes at most one message per call;
    additional queued messages stay for subsequent turns.
    """

    def __init__(self, inbox: asyncio.Queue[str]) -> None:
        self._inbox = inbox

    async def on_run_continue(
        self,
        kwargs: dict[str, Any],
        old_items: list[dict[str, Any]],
        new_items: list[dict[str, Any]],
    ) -> bool:
        try:
            msg = self._inbox.get_nowait()
        except asyncio.QueueEmpty:
            return True
        new_items.append({"role": "user", "content": f"[Steer] {msg}"})
        return True


class _TurnLimitCallback(AsyncCallbackHandler):
    """Stop before the next model call, after the last turn's tools execute."""

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.turns = 0
        self.terminated = False

    async def on_responses(self, kwargs: dict[str, Any], responses: dict[str, Any]) -> None:
        self.turns += 1
        self.terminated = _is_terminated(responses.get("output", []))

    async def on_run_continue(self, kwargs: dict[str, Any], old_items: list[dict[str, Any]],
                              new_items: list[dict[str, Any]]) -> bool:
        return self.turns < self.max_steps and not self.terminated


class _GUIComputerHandler(OpenClawComputerHandler):
    async def screenshot(self, text: str | None = None) -> str:
        raw = await super().screenshot(text=text)
        resized, mime = _maybe_sanitize_screenshot(raw)
        with Image.open(io.BytesIO(base64.b64decode(resized))) as image:
            self.set_screenshot_dimensions(*image.size)
            if mime != "image/png":
                output = io.BytesIO()
                image.save(output, format="PNG")
                return base64.b64encode(output.getvalue()).decode("ascii")
        return resized


class _GUIAgent(ComputerAgent):
    """Dispatch the function-form computer tool used by OpenRouter models."""

    auto_screenshot = True
    _dispatch_computer_function_call = OpenClawComputerAgent._dispatch_computer_function_call

    async def _handle_item(self, item, computer=None, ignore_call_ids=None):
        if item.get("type") != "function_call" or item.get("name") != "computer":
            return await super()._handle_item(item, computer, ignore_call_ids)
        call_id = item.get("call_id")
        if ignore_call_ids and call_id in ignore_call_ids:
            return []
        try:
            if computer is None:
                raise ToolError("Computer handler is required")
            await self._on_function_call_start(item)
            result = await self._dispatch_computer_function_call(item, computer)
            await self._on_function_call_end(item, result)
            return result
        except (ToolError, ValueError, TypeError) as exc:
            return [make_tool_error_item(str(exc), call_id)]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _build_system_prompt() -> str:
    """Build the GUI subagent system prompt.

    Intentionally minimal — the computer tool schema (auto-built by
    ComputerAgent's agent loop) already provides the action vocabulary.
    Only behavioral guidance is needed here.

    Reference: openclaw/src/agents/subagent-announce.ts:47-104
    """
    return "\n".join([
        "# GUI Subagent",
        "",
        "You are a **GUI automation subagent** spawned by the main agent to "
        "perform a focused GUI task on the VM.",
        "",
        "## Rules",
        "1. **Observe before acting** — Read the screenshot carefully. "
        "Check UI state before deciding your next action.",
        "2. **Batch when predictable** — When the outcome of multiple "
        "actions is predictable (e.g. pressing Space 10 times to dismiss "
        "dialogs, or a known sequence of arrow keys for navigation), emit "
        "them as multiple `computer` tool calls in a single turn. This is "
        "faster and saves steps.",
        "3. **Observe when uncertain** — When the next screen state matters "
        "for deciding what to do (e.g. after clicking a button that opens "
        "a new dialog), emit a single action and wait for the next "
        "screenshot.",
        "4. **Stay focused** — Do only the assigned task, no side quests.",
        "5. **Record discoveries** — When you encounter useful information "
        "during VM interaction (error messages, UI state, discovered paths), "
        "use `memory_write` to record it. The main agent will see these "
        "observations in shared memory.",
        "6. **Complete and return** — When the task is visibly complete, "
        "call `computer` with `action='terminate'` and a brief summary.",
        "7. **Be ephemeral** — You will be terminated after returning.",
    ])


# ---------------------------------------------------------------------------
# Output item extraction helpers
# ---------------------------------------------------------------------------


def _extract_actions_from_output(
    output_items: list[dict[str, Any]],
) -> list[str]:
    """Extract human-readable action descriptions from Responses API items."""
    descs: list[str] = []
    for item in output_items:
        item_type = item.get("type", "")

        if item_type == "computer_call":
            action = item.get("action", {})
            if isinstance(action, dict):
                descs.append(_describe_action_dict(action))

        elif item_type == "function_call" and item.get("name") == "computer":
            try:
                args = json.loads(item.get("arguments", "{}"))
                descs.append(_describe_action_dict(args))
            except (json.JSONDecodeError, TypeError):
                descs.append("computer(?)")

    return descs


def _describe_action_dict(action: dict[str, Any]) -> str:
    """One-line human-readable description from an action dict."""
    a = action.get("action") or action.get("type", "")
    if a in ("click", "double_click", "right_click"):
        return f"{a} ({action.get('x')}, {action.get('y')})"
    if a == "type":
        return f"type {action.get('text', '')!r}"
    if a == "keypress":
        keys = action.get("keys", [])
        return f"keypress {'+'.join(keys) if keys else '?'}"
    if a == "scroll":
        return f"scroll ({action.get('scroll_x', 0)}, {action.get('scroll_y', 0)})"
    if a == "move":
        return f"move ({action.get('x')}, {action.get('y')})"
    if a == "drag":
        return (
            f"drag ({action.get('start_x')}, {action.get('start_y')}) -> "
            f"({action.get('end_x')}, {action.get('end_y')})"
        )
    if a == "screenshot":
        return "screenshot"
    if a == "wait":
        return f"wait {action.get('ms', action.get('seconds', 1000))}ms"
    if a == "terminate":
        return f"terminate ({action.get('status', '')})"
    return str(action)


def _is_terminated(output_items: list[dict[str, Any]]) -> bool:
    """Check if any output item is a terminate action or its result."""
    for item in output_items:
        item_type = item.get("type", "")

        # computer_call with terminate action
        if item_type == "computer_call":
            action = item.get("action", {})
            if isinstance(action, dict) and action.get("type") == "terminate":
                return True

        # function_call with terminate action
        if item_type == "function_call" and item.get("name") == "computer":
            try:
                args = json.loads(item.get("arguments", "{}"))
                if args.get("action") == "terminate":
                    return True
            except (json.JSONDecodeError, TypeError):
                pass

        # computer_call_output or function_call_output with terminated flag
        if item_type in ("computer_call_output", "function_call_output"):
            output = item.get("output", "")
            if isinstance(output, dict) and output.get("terminated"):
                return True
            if isinstance(output, str):
                try:
                    parsed = json.loads(output)
                    if isinstance(parsed, dict) and parsed.get("terminated"):
                        return True
                except (json.JSONDecodeError, TypeError):
                    pass

    return False


def _extract_text(output_items: list[dict[str, Any]]) -> str:
    """Extract text content from message items."""
    parts: list[str] = []
    for item in output_items:
        if item.get("type") != "message":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    text = c.get("text", "")
                    if text:
                        parts.append(text)
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Usage accumulation
# ---------------------------------------------------------------------------


def _accumulate_usage(
    usage: SubagentUsage, result_usage: Any,
) -> None:
    """Add tokens from a ComputerAgent result's usage to SubagentUsage."""
    usage.add(result_usage)


# ---------------------------------------------------------------------------
# Transcript persistence
# ---------------------------------------------------------------------------


class _TranscriptWriter:
    """Lightweight append-only JSONL writer for GUI subagent turns.

    Records assistant calls and tool outputs in the same format as the main
    session. Screenshots are stored as separate trajectory media artifacts.
    """

    def __init__(self, transcript_path: Path | None) -> None:
        self._path = transcript_path
        self._trajectory_dir = (
            transcript_path.parents[4] / "trajectories" if transcript_path else None
        )

    def _append(self, entry: dict[str, Any]) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def write_items(self, output_items: list[dict[str, Any]]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        assistant, results = group_step_output(output_items, self._trajectory_dir)
        for role, blocks in (("assistant", assistant), ("tool", results)):
            if blocks:
                self._append({"type": "message", "timestamp": ts,
                              "message": {"role": role, "content": blocks}})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def _build_gui_agent(
    session: Any,
    model: str,
    thinking_params: dict[str, Any] | None,
    memory_store: MemoryStore | None,
    inbox: "asyncio.Queue[str]",
    max_steps: int,
) -> ComputerAgent:
    """Construct the lightweight GUI ComputerAgent (computer tool + optional memory).

    The session's computer is wrapped in an
    :class:`OpenClawComputerHandler` so the subagent gets the same
    chord-vs-sequence keypress semantics, tolerant coordinate coercion, and —
    crucially for Gemini-family GUI models — normalized→pixel scaling as the
    main agent. The coordinate space is derived from *this subagent's* model,
    since delegate_gui runs may pick a different model than the main agent.
    ``make_computer_handler`` returns an already-``AsyncComputerHandler`` as-is
    without initializing it, so we ``_initialize`` here (idempotent)."""
    computer = _GUIComputerHandler(
        session.computer,
        coordinate_space=coordinate_space_for_model(model),
    )
    await computer._initialize()            # noqa: SLF001
    tools: list[Any] = [computer]
    if memory_store is not None:
        tools.extend([
            MemorySearchTool(memory_store),
            MemoryGetTool(memory_store),
            MemoryWriteTool(memory_store),
        ])
    return _GUIAgent(
        model=model,
        tools=tools,
        instructions=_build_system_prompt(),
        only_n_most_recent_images=DEFAULT_IMAGE_HISTORY,
        telemetry_enabled=False,
        callbacks=[SteerInboxCallback(inbox), _TurnLimitCallback(max_steps)],
        **(thinking_params or {}),
    )


async def _relay_until_done(
    agent: ComputerAgent,
    instruction: str,
    run_id: str,
    max_steps: int,
    usage: SubagentUsage,
    transcript: "_TranscriptWriter",
    credentials: dict[str, str] | None = None,
) -> str:
    """Drive ``ComputerAgent.run()``, recording turns/usage, and return the summary.

    Stops on a ``terminate`` action or when ``max_steps`` is reached.
    """
    step = 0
    terminated = False
    terminate_ids: set[str] = set()
    termination_failed = False
    last_text = ""

    async for result in agent.run(instruction, **(credentials or {})):
        output_items = result.get("output", [])
        _accumulate_usage(usage, result.get("usage"))

        actions = _extract_actions_from_output(output_items)
        text = _extract_text(output_items)
        if text:
            last_text = text
        for item in output_items:
            if item.get("type") == "function_call" and item.get("name") == "computer":
                try:
                    action = json.loads(item.get("arguments", "{}"))
                except (ValueError, TypeError):
                    continue
                if action.get("action") == "terminate":
                    terminate_ids.add(item.get("call_id"))
                    termination_failed |= action.get("status") in ("failed", "failure", "error")
            elif item.get("type") == "computer_call" and (item.get("action") or {}).get("type") == "terminate":
                terminate_ids.add(item.get("call_id"))
                termination_failed |= item["action"].get("status") in ("failed", "failure", "error")
            elif item.get("type") in ("function_call_output", "computer_call_output") and item.get("call_id") in terminate_ids:
                terminated = True
                output = item.get("output")
                if isinstance(output, str):
                    try:
                        output = json.loads(output)
                    except ValueError:
                        termination_failed |= output.lower().startswith("error")
                if isinstance(output, dict):
                    termination_failed |= output.get("success") is False or output.get("status") in ("failed", "failure", "error") or "error" in output

        transcript.write_items(output_items)
        if actions:
            logger.info("  [GUI %s] step %d: %s", run_id, step, actions)
            step += 1

    if terminated and not termination_failed:
        return last_text or "(no summary)"
    if termination_failed:
        raise RuntimeError("GUI subagent terminated with failure")
    raise RuntimeError(f"GUI subagent exited without terminate (max_steps={max_steps})")


async def run_gui_subagent(
    *,
    instruction: str,
    session: Any,
    registry: SubagentRegistry,
    run_id: str,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    thinking_params: dict[str, Any] | None = None,
    parent_session_dir: str | Path | None = None,
    memory_store: MemoryStore | None = None,
    credentials: dict[str, str] | None = None,
) -> str:
    """Blocking vision-to-action relay loop via ComputerAgent.

    Creates a lightweight ``ComputerAgent`` with only the computer tool and
    a minimal system prompt.  ``ComputerAgent`` handles:
      - Tool schema registration (provider-specific)
      - Action execution and post-action screenshots
      - Image history pruning (``only_n_most_recent_images``)
      - Action normalization (``OperatorNormalizerCallback``)

    The relay loop iterates ``ComputerAgent.run()`` results, tracking usage
    and writing a transcript.  Terminates when the model emits a
    ``terminate`` action or ``max_steps`` is reached.

    Returns the final summary string.  Raises the underlying exception on
    failure.
    """
    transcript_path: Path | None = None
    if parent_session_dir is not None:
        transcript_path = _subagent_transcript_path(Path(parent_session_dir), run_id)
    transcript = _TranscriptWriter(transcript_path)

    usage = SubagentUsage()
    inbox: asyncio.Queue[str] = asyncio.Queue()
    registry.mark_running(run_id)
    registry.attach_inbox(run_id, inbox)

    try:
        agent = await _build_gui_agent(session, model, thinking_params, memory_store, inbox, max_steps)
        summary = await _relay_until_done(
            agent, instruction, run_id, max_steps, usage, transcript, credentials
        )
        registry.complete(run_id, summary, usage)
        return summary

    except asyncio.CancelledError:
        registry.kill(run_id, usage)
        raise
    except Exception as e:
        registry.fail(run_id, str(e), usage)
        raise


__all__ = [
    "DEFAULT_IMAGE_HISTORY",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MODEL",
    "SteerInboxCallback",
    "run_gui_subagent",
]
