"""Delegation tools — BaseTool subclasses that spawn subagent runs.

Three tools are exposed to the main agent:
  - ``delegate_general``: async one-shot planning/analysis subagent driven by
    the ``GeneralSubagentSession`` persistent engine. Returns
    immediately with ``{"status": "accepted", "run_id": ...}``; the final
    result is delivered later as a ``[Subagent Result]`` user message via
    ``OpenClawComputerAgent._drain_completions``.
  - ``delegate_gui``: blocking vision-to-action relay driving the VM via the
    ``run_gui_subagent`` loop. Returns the final summary
    synchronously so the main agent can resume with fresh VM state.
  - ``subagents``: ``list`` active/recent runs or ``kill`` a runaway general
    subagent via ``registry.kill_run``.

Reference:
  ``openclaw/src/agents/tools/sessions-spawn-tool.ts`` (spawn-tool shape),
  ``openclaw/src/agents/tools/subagents-tool.ts`` (list/kill actions).

Design notes:
  * ``BaseTool.call()`` is synchronous. ``DelegateGeneralTool`` runs inside an
    active asyncio event loop (the CUA agent awaits tool calls), so it
    schedules a task with ``asyncio.get_running_loop().create_task`` and
    returns immediately. ``DelegateGUITool`` needs to block until the relay
    loop finishes, so it adopts the ``ThreadPoolExecutor + asyncio.run``
    pattern from ``AnalyzeImageTool.call`` (``analyze_image.py:144-168``).
  * All three tools degrade gracefully: ``DelegateGeneralTool`` returns a
    ``rejected`` payload when the registry refuses a spawn (concurrency cap);
    ``DelegateGUITool`` returns ``{"status": "error", ...}`` when the relay
    coroutine raises; ``SubagentsTool.kill`` reports ``noop``/``error`` for
    terminal/unknown targets.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from agent.tools.base import BaseTool, register_tool

from ..memory.memory import MemoryStore
from .subagent_general import DEFAULT_MAX_STEPS as GENERAL_DEFAULT_MAX_STEPS
from .subagent_general import run_general_subagent
from .subagent_gui import DEFAULT_MAX_STEPS as GUI_DEFAULT_MAX_STEPS
from .subagent_gui import DEFAULT_MODEL as GUI_DEFAULT_MODEL
from .subagent_gui import run_gui_subagent
from .subagent_registry import (
    SubagentLimitError,
    SubagentRegistry,
    SubagentRun,
    SubagentStatus,
    SubagentType,
)
from .subagent_session import _encode_image_url_from_path

DELEGATE_GENERAL_DEFAULT_MAX_STEPS = GENERAL_DEFAULT_MAX_STEPS

_POST_DELEGATION_TEXT = "[VM state after GUI delegation]"
_logger = logging.getLogger(__name__)

_ACCEPTED_NOTE = (
    "persistent session — result auto-announces when complete; do not poll"
)


def _build_model_param_schema(
    default_model: str, auxiliary_model: str | None
) -> dict[str, Any]:
    """Build the JSON Schema for a delegate tool's `model` parameter.

    The schema constrains the override to an explicit allowlist via `enum`,
    so the main agent can't hallucinate a sibling model ID. When an
    auxiliary model is configured, the description names the trade-off
    so the main agent can pick deliberately; otherwise the enum has a
    single element and the parameter is effectively a no-op.
    """
    allowed = _allowed_models(default_model, auxiliary_model)
    if auxiliary_model and auxiliary_model != default_model:
        description = (
            f"Optional model override. Pick one of:\n"
            f"- '{default_model}' (default): stronger reasoning. Use for "
            f"non-trivial planning, multi-step analysis, or anything where "
            f"you'd want gpt-5.4 in the main loop.\n"
            f"- '{auxiliary_model}': cheaper/faster sibling. Use for "
            f"simple lookups, short summarization, or one-shot extraction "
            f"where stronger reasoning isn't needed."
        )
    else:
        description = (
            f"Optional model override (default: '{default_model}'). "
            f"Only the default is currently allowed; pass nothing to use it."
        )
    return {"type": "string", "enum": allowed, "description": description}


def _allowed_models(
    default_model: str, auxiliary_model: str | None
) -> list[str]:
    """Build the ordered allowlist of valid model strings for a delegate tool.

    Order matters for schema enum stability: default first, auxiliary model second
    (when provided). Duplicates are dropped (e.g. caller passes the same
    string for both).
    """
    allowed = [default_model]
    if auxiliary_model and auxiliary_model != default_model:
        allowed.append(auxiliary_model)
    return allowed


def _sanitize_subagent_model(
    requested: str | None,
    default_model: str,
    auxiliary_model: str | None = None,
) -> tuple[str, str | None]:
    """Validate a subagent model override against an explicit allowlist.

    Earlier versions compared only the first ``/``-separated provider prefix,
    which let hallucinated variants under the same routing provider slip
    through (e.g. ``openrouter/openai/gpt-5.1-mini`` when the default is
    ``openrouter/openai/gpt-5.4``). The allowlist here is exact-match against
    ``[default_model, auxiliary_model]`` — anything else falls back to the
    default with a warning surfaced in the tool response.

    Returns ``(resolved_model, warning)``. ``warning`` is ``None`` when the
    requested model is accepted as-is.
    """
    if not requested:
        return default_model, None
    allowed = _allowed_models(default_model, auxiliary_model)
    if requested in allowed:
        return requested, None
    warning = (
        f"requested model '{requested}' is not in the allowlist {allowed}. "
        f"Using default '{default_model}' instead. Pick one of the listed "
        f"models or omit the `model` parameter to silence this warning."
    )
    return default_model, warning


# ---------------------------------------------------------------------------
# delegate_general
# ---------------------------------------------------------------------------


@register_tool("delegate_general")
class DelegateGeneralTool(BaseTool):
    """Spawn an async general subagent session (planning/analysis, no VM access)."""

    def __init__(
        self,
        registry: SubagentRegistry,
        tools: list,
        memory_store: MemoryStore,
        default_model: str,
        summary_model: str,
        parent_session_dir: Path,
        thinking_params: dict[str, Any] | None = None,
        auxiliary_model: str | None = None,
        cfg: dict | None = None,
    ) -> None:
        self._registry = registry
        self._tools = tools
        self._memory_store = memory_store
        self._default_model = default_model
        self._summary_model = summary_model
        self._parent_session_dir = Path(parent_session_dir)
        self._thinking_params = thinking_params
        self._auxiliary_model = auxiliary_model
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Spawn an asynchronous *general* subagent to work on a focused "
            "planning/analysis/memory task. The subagent has NO direct VM "
            "access — it can only use memory tools and LLM reasoning. "
            "Returns immediately with a run_id; the final result is "
            "announced later as a '[Subagent Result]' user message. "
            "DO NOT poll with `subagents(list)` — results auto-announce."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What the subagent should accomplish.",
                },
                "model": _build_model_param_schema(
                    self._default_model, self._auxiliary_model
                ),
                "max_steps": {
                    "type": "integer",
                    "description": (
                        "Safety rail for the subagent's loop (default 50). "
                        "The session compacts its own context and typically "
                        "completes well before this cap."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Optional human-readable label for observability.",
                },
                "screenshot_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional absolute paths to PNG screenshots to "
                        "attach to the subagent's initial message as vision "
                        "input. Use this to delegate 'analyze this frame' "
                        "work."
                    ),
                },
            },
            "required": ["task"],
        }

    def call(self, params: str | dict, **kwargs) -> dict:
        params_dict = self._verify_json_format_args(params)

        task = params_dict.get("task", "")
        if not isinstance(task, str) or not task.strip():
            return {
                "status": "error",
                "reason": "task must be a non-empty string",
            }

        model, model_warning = _sanitize_subagent_model(
            params_dict.get("model"),
            self._default_model,
            self._auxiliary_model,
        )
        if model_warning:
            _logger.warning("delegate_general: %s", model_warning)
        max_steps = int(
            params_dict.get("max_steps", DELEGATE_GENERAL_DEFAULT_MAX_STEPS)
        )
        label = params_dict.get("label", "") or ""
        screenshot_paths_raw = params_dict.get("screenshot_paths") or []
        screenshot_paths: list[str] | None = (
            [p for p in screenshot_paths_raw if isinstance(p, str) and p]
            if isinstance(screenshot_paths_raw, list)
            else None
        )

        try:
            run = self._registry.register(
                type=SubagentType.GENERAL,
                task=task,
                label=label,
                model=model,
            )
        except SubagentLimitError:
            return {
                "status": "rejected",
                "reason": "max concurrent subagents reached",
            }

        coro = run_general_subagent(
            task=task,
            model=model,
            tools=self._tools,
            registry=self._registry,
            run_id=run.run_id,
            summary_model=self._summary_model,
            parent_session_dir=self._parent_session_dir,
            memory_store=self._memory_store,
            max_steps=max_steps,
            thinking_params=self._thinking_params,
            initial_screenshot_paths=screenshot_paths,
        )

        loop = asyncio.get_running_loop()
        task_handle = loop.create_task(coro)
        self._registry.attach_task(run.run_id, task_handle)

        response = {
            "status": "accepted",
            "run_id": run.run_id,
            "note": _ACCEPTED_NOTE,
        }
        if model_warning:
            response["model_warning"] = model_warning
        return response


# ---------------------------------------------------------------------------
# delegate_gui
# ---------------------------------------------------------------------------


@register_tool("delegate_gui")
class DelegateGUITool(BaseTool):
    """Spawn a blocking GUI subagent (vision-to-action relay on the VM)."""

    def __init__(
        self,
        registry: SubagentRegistry,
        session: Any,
        parent_session_dir: Path,
        default_model: str = GUI_DEFAULT_MODEL,
        thinking_params: dict[str, Any] | None = None,
        memory_store: MemoryStore | None = None,
        auxiliary_model: str | None = None,
        cfg: dict | None = None,
    ) -> None:
        self._registry = registry
        self._session = session
        self._parent_session_dir = Path(parent_session_dir)
        self._default_model = default_model
        self._thinking_params = thinking_params
        self._memory_store = memory_store
        self._auxiliary_model = auxiliary_model
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Spawn a *GUI automation* subagent driven by a vision model. "
            "Returns immediately with a run_id; the subagent takes over "
            "the VM for a bounded number of steps. When finished, the "
            "result is announced as a '[Subagent Result]' user message "
            "followed by a fresh VM screenshot. DO NOT poll — results "
            "auto-announce. While the GUI subagent is running, the VM is "
            "occupied — do not call delegate_gui again or use computer "
            "directly until it completes."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Self-contained GUI task (e.g. 'open Notepad').",
                },
                "model": _build_model_param_schema(
                    self._default_model, self._auxiliary_model
                ),
                "max_steps": {
                    "type": "integer",
                    "description": (
                        f"Safety rail for the relay loop "
                        f"(default {GUI_DEFAULT_MAX_STEPS})."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Optional human-readable label for observability.",
                },
            },
            "required": ["instruction"],
        }

    def call(self, params: str | dict, **kwargs) -> dict:
        params_dict = self._verify_json_format_args(params)

        instruction = params_dict.get("instruction", "")
        if not isinstance(instruction, str) or not instruction.strip():
            return {
                "status": "error",
                "reason": "instruction must be a non-empty string",
            }

        model, model_warning = _sanitize_subagent_model(
            params_dict.get("model"),
            self._default_model,
            self._auxiliary_model,
        )
        if model_warning:
            _logger.warning("delegate_gui: %s", model_warning)
        max_steps = int(params_dict.get("max_steps", GUI_DEFAULT_MAX_STEPS))
        label = params_dict.get("label", "") or ""

        run = self._registry.register(
            type=SubagentType.GUI,
            task=instruction,
            label=label,
            model=model,
        )

        async def _drive() -> None:
            try:
                await run_gui_subagent(
                    instruction=instruction,
                    session=self._session,
                    registry=self._registry,
                    run_id=run.run_id,
                    model=model,
                    max_steps=max_steps,
                    thinking_params=self._thinking_params,
                    parent_session_dir=self._parent_session_dir,
                    memory_store=self._memory_store,
                )
                # run_gui_subagent already calls registry.complete()
            except Exception as e:
                # run_gui_subagent already calls registry.fail() before raising
                _logger.warning("GUI subagent %s failed: %s", run.run_id, e)
                return

            try:
                post_shot = await self._session.screenshot()
            except Exception as post_exc:
                _logger.warning(
                    "post-delegation screenshot failed for run %s: %s",
                    run.run_id,
                    post_exc,
                )
                post_shot = None

            if isinstance(post_shot, (bytes, bytearray)) and post_shot:
                self._enqueue_post_delegation(run.run_id, bytes(post_shot))

        loop = asyncio.get_running_loop()
        task_handle = loop.create_task(_drive())
        self._registry.attach_task(run.run_id, task_handle)

        response = {
            "status": "accepted",
            "run_id": run.run_id,
            "note": _ACCEPTED_NOTE,
        }
        if model_warning:
            response["model_warning"] = model_warning
        return response

    def _enqueue_post_delegation(self, run_id: str, png_bytes: bytes) -> None:
        """Persist the fresh screenshot and enqueue a user message for the main agent.

        Writes ``<parent_session_dir>/subagents/<run_id>/post_delegation.png``
        for trajectory inspection, then pushes a pre-built
        ``{role: user, content: [text, image_url]}`` dict to the registry's
        post-delegation queue for ``_drain_post_delegation`` to fold into the
        next main-agent turn.
        """
        try:
            target_dir = self._parent_session_dir / "subagents" / run_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / "post_delegation.png"
            target_path.write_bytes(png_bytes)
        except OSError as exc:
            _logger.warning(
                "could not persist post-delegation screenshot for %s: %s",
                run_id,
                exc,
            )
            return

        block = _encode_image_url_from_path(str(target_path))
        if block is None:
            _logger.warning(
                "could not encode post-delegation screenshot for %s", run_id
            )
            return

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": _POST_DELEGATION_TEXT},
                block,
            ],
        }
        self._registry.enqueue_post_delegation(message)


# ---------------------------------------------------------------------------
# subagents (list / kill / steer)
# ---------------------------------------------------------------------------

MAX_STEER_MESSAGE_CHARS = 4_000

_ACTIVE_STATUSES = frozenset({SubagentStatus.PENDING, SubagentStatus.RUNNING})


def _resolve_steer_target(
    registry: SubagentRegistry,
    target: str,
) -> SubagentRun | None:
    """Resolve a steer target to a SubagentRun.

    Searches ALL runs (active + terminal) so the caller can produce
    specific error messages (e.g. "already finished" vs "unknown").

    Precedence (simplified from OpenClaw's ``resolveSubagentTargetFromRuns``):
    1. Exact run_id match (any status)
    2. Exact label match (case-insensitive) — active first, then terminal
    3. Run ID prefix match — active first, then terminal
    4. ``"last"`` keyword → most recently created active general run
    """
    run = registry.get_run(target)
    if run is not None:
        return run

    all_runs = registry.list_runs()
    active = [r for r in all_runs if r.status in _ACTIVE_STATUSES]
    terminal = [r for r in all_runs if r.status not in _ACTIVE_STATUSES]

    lowered = target.lower()

    if lowered == "last":
        return max(active, key=lambda r: r.created_at) if active else None

    for pool in (active, terminal):
        for r in pool:
            if r.label and r.label.lower() == lowered:
                return r

    for pool in (active, terminal):
        for r in pool:
            if r.run_id.startswith(target):
                return r

    return None


@register_tool("subagents")
class SubagentsTool(BaseTool):
    """Inspect, cancel, or steer active subagent runs."""

    def __init__(
        self,
        registry: SubagentRegistry,
        cfg: dict | None = None,
    ) -> None:
        self._registry = registry
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Inspect, cancel, or steer subagent runs. "
            "action='list' returns active and recent runs. "
            "action='kill' cancels a runaway general subagent. "
            "action='steer' sends a follow-up message into a running "
            "subagent to refine or redirect its work mid-flight. "
            "Target can be a run_id, label, run_id prefix, or 'last'."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "kill", "steer"],
                    "description": "'list' (default), 'kill', or 'steer'.",
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Required for kill/steer. "
                        "A run_id, label, run_id prefix, or 'last'."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": (
                        "Required for steer. The follow-up message to "
                        "inject into the running subagent's conversation."
                    ),
                },
            },
            "required": [],
        }

    def call(self, params: str | dict, **kwargs) -> dict:
        params_dict = self._verify_json_format_args(params)
        action = params_dict.get("action", "list")

        if action == "list":
            active: list[dict] = []
            recent: list[dict] = []
            for run in self._registry.list_runs():
                if run.status in _ACTIVE_STATUSES:
                    active.append(run.to_dict())
                else:
                    recent.append(run.to_dict())
            return {"status": "ok", "active": active, "recent": recent}

        if action == "kill":
            target = params_dict.get("target")
            if not isinstance(target, str) or not target:
                return {
                    "status": "error",
                    "reason": "target run_id is required for action='kill'",
                }
            run = self._registry.get_run(target)
            if run is None:
                return {"status": "error", "reason": "unknown run_id"}
            if run.status in (
                SubagentStatus.COMPLETE,
                SubagentStatus.ERROR,
                SubagentStatus.KILLED,
            ):
                return {"status": "noop", "reason": "already terminal"}
            self._registry.kill_run(target)
            return {"status": "ok", "killed": target}

        if action == "steer":
            return self._handle_steer(params_dict)

        return {"status": "error", "reason": f"unknown action: {action}"}

    def _handle_steer(self, params_dict: dict) -> dict:
        target = params_dict.get("target")
        if not isinstance(target, str) or not target:
            return {
                "status": "error",
                "reason": "target is required for action='steer'",
            }

        message = params_dict.get("message")
        if not isinstance(message, str) or not message.strip():
            return {
                "status": "error",
                "reason": "message is required for action='steer'",
            }

        if len(message) > MAX_STEER_MESSAGE_CHARS:
            return {
                "status": "error",
                "reason": (
                    f"message too long ({len(message)} chars, "
                    f"max {MAX_STEER_MESSAGE_CHARS})"
                ),
            }

        run = _resolve_steer_target(self._registry, target)
        if run is None:
            return {"status": "error", "reason": f"unknown target: {target}"}

        if run.status not in _ACTIVE_STATUSES:
            return {
                "status": "error",
                "reason": (
                    f"target '{target}' ({run.run_id}) already "
                    f"{run.status.value} — cannot steer a finished run"
                ),
            }

        inbox = self._registry.get_inbox(run.run_id)
        if inbox is None:
            return {
                "status": "error",
                "reason": "no inbox for target (run may have just finished)",
            }

        inbox.put_nowait(message)
        return {"status": "ok", "steered": run.run_id}


