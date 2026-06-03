"""Tool registry, summaries, and logging callback for the OpenClaw agent harness.

Centralizes tool assembly (previously inline in openclaw_agent.py) and provides
a ToolLoggingCallback that uses CUA's AsyncCallbackHandler for observability.

Design rationale:
  - Tool assembly: extracted from inline list to build_tools() for reuse
  - Tool summaries: extracted to get_tool_summaries() for prompt builder
  - Logging callback: adapted from OpenClaw's wrapToolWithBeforeToolCallHook
    (pi-tools.before-tool-call.ts) — observe-only via CUA callbacks, no blocking/modification

Reference: openclaw/src/agents/pi-tools.ts (createOpenClawCodingTools),
           openclaw/src/agents/pi-tools.before-tool-call.ts (hook wrapping)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Literal, Optional, Union

from agent.callbacks.base import AsyncCallbackHandler
from agent.computers.base import AsyncComputerHandler
from agent.computers.cua import cuaComputerHandler
from agent.tools.base import BaseTool
from agent.types import ToolError

from .fs_backends import FilesystemRegistry, HostBackend, MCPBackend, VMBackend
from ..memory.memory import MemoryGetTool, MemorySearchTool, MemoryStore  # MemoryWriteTool retained in memory.py; intentionally not exposed to the main agent (write(target='host') covers journaling).
from ..subagent.subagent_registry import SubagentRegistry
from ..subagent.subagent_tools import DelegateGeneralTool, DelegateGUITool, SubagentsTool
from .tools_fs import EditFileTool, ReadFileTool, WriteFileTool
from .tools_shell import ExecTool, _MCPExecInterface
from .tools_web import WebFetchTool, WebSearchTool


COMPUTER_TOOL_NAME = "computer"
COMPUTER_TOOL_SUMMARY = (
    "Observe the current desktop via screenshots and interact with it using "
    "mouse and keyboard actions. Only the explicit `screenshot` action returns "
    "an image — other actions (click, type, keypress, scroll, drag, wait, etc.) "
    "return a textual result. Call `computer(action=\"screenshot\")` whenever "
    "you need to see the updated screen state after performing actions."
)


def _is_computer_tool(tool: Any) -> bool:
    """Detect the primary CUA computer handler without relying on ``tool.name``.

    The real ``computer.Computer`` object used by ``session._computer`` is not a
    BaseTool subclass and does not reliably expose ``name='computer'``. Mirror
    CUA's own class-name heuristic so prompt summaries match the actual runtime.
    """
    class_name = getattr(getattr(tool, "__class__", None), "__name__", "")
    return "computer" in class_name.lower()


class _RestrictedComputerHandler(AsyncComputerHandler):
    """Computer handler that only allows ``screenshot`` and ``wait`` actions.

    Used when ``disable_main_computer`` is set — the agent can observe the VM
    and idle (keeping the loop alive), but cannot perform interactive actions.
    Interactive GUI work must go through ``delegate_gui``.

    Wraps a ``cuaComputer`` with lazy initialization (the VM may not be ready
    at tool-build time). Passes ``is_agent_computer()`` because
    ``AsyncComputerHandler`` is a ``runtime_checkable`` Protocol.
    """

    def __init__(self, cua_computer: Any) -> None:
        self._cua_computer = cua_computer
        self._inner: cuaComputerHandler | None = None
        self.interface: Any = None

    async def _ensure_init(self) -> None:
        if self._inner is None:
            self._inner = cuaComputerHandler(self._cua_computer)
            await self._inner._initialize()
            self.interface = self._inner.interface

    def _blocked(self, name: str):
        async def _raise(**kw: Any) -> None:
            raise ToolError(
                f"Action '{name}' is not available — the main agent cannot "
                "drive the VM directly. Use delegate_gui for GUI interactions."
            )
        return _raise

    # -- Allowed actions --

    async def screenshot(self, text: Optional[str] = None) -> str:
        await self._ensure_init()
        assert self._inner is not None
        return await self._inner.screenshot(text=text)

    async def wait(self, ms: int = 1000) -> None:
        await self._ensure_init()
        assert self._inner is not None
        await self._inner.wait(ms=ms)

    async def terminate(self, **kw: Any) -> Dict[str, Any]:
        await self._ensure_init()
        assert self._inner is not None
        return await self._inner.terminate(**kw)

    async def get_environment(self) -> Literal["windows", "mac", "linux", "browser"]:
        await self._ensure_init()
        assert self._inner is not None
        return await self._inner.get_environment()

    async def get_dimensions(self) -> tuple[int, int]:
        await self._ensure_init()
        assert self._inner is not None
        return await self._inner.get_dimensions()

    async def get_current_url(self) -> str:
        await self._ensure_init()
        assert self._inner is not None
        return await self._inner.get_current_url()

    # -- Blocked actions --

    async def click(self, x: int, y: int, button: str = "left") -> None:
        await self._blocked("click")()

    async def double_click(self, x: int, y: int) -> None:
        await self._blocked("double_click")()

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        await self._blocked("scroll")()

    async def type(self, text: str) -> None:
        await self._blocked("type")()

    async def move(self, x: int, y: int) -> None:
        await self._blocked("move")()

    async def keypress(self, keys: Union[List[str], str]) -> None:
        await self._blocked("keypress")()

    async def drag(self, path: List[Dict[str, int]]) -> None:
        await self._blocked("drag")()

    async def left_mouse_down(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        await self._blocked("left_mouse_down")()

    async def left_mouse_up(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        await self._blocked("left_mouse_up")()


def build_tools(
    session,
    memory_store: MemoryStore,
    *,
    summary_model: str | None = None,
    vision_thinking_params: dict[str, Any] | None = None,
    registry: SubagentRegistry | None = None,
    parent_session_dir: Any = None,
    default_model: str | None = None,
    lightweight_model: str | None = None,
    thinking_params: dict[str, Any] | None = None,
    gui_thinking_params: dict[str, Any] | None = None,
    disable_main_computer: bool = False,
    disable_delegate_gui: bool = False,
    gui_model: str | None = None,
    workspace_root: str | None = None,
    host_workspace_root: str | None = None,
    context_window_tokens: int | None = None,
    computer_handler: Any = None,
    mcp_runtime: Any = None,
) -> list:
    """Assemble the canonical tool list for the OpenClaw agent.

    Returns [Computer, MilestoneTool, AnalyzeImageTool, ReadFileTool,
             WriteFileTool, EditFileTool, ExecTool, WebSearchTool, WebFetchTool,
             MemorySearchTool, MemoryGetTool] by default. ``MemoryWriteTool``
    is no longer exposed to the main agent — its journaling role is covered
    by ``write(target='host')``; the class lives in ``memory.py`` and is
    still consumed by ``memory_flush.py`` and the GUI subagent. When
    ``registry`` and ``parent_session_dir`` are supplied, also
    appends [DelegateGeneralTool, DelegateGUITool, SubagentsTool].
    ``DelegateGUITool`` is suppressed when ``disable_delegate_gui=True``
    (mirrors OpenClaw's filterToolsByPolicy pattern — absence is the signal).

    Args:
        session: CUA DesktopSession (provides ``_computer`` and ``interface``).
        memory_store: Initialized MemoryStore for this task.
        summary_model: Model string for VLM calls in AnalyzeImageTool (defaults
            to the agent's summary_model). Also reused as the subagent
            summary_model for in-session compaction.
        vision_thinking_params: Provider-specific helper thinking kwargs
            forwarded to AnalyzeImageTool's ``litellm.acompletion()`` call.
        registry: Per-task ``SubagentRegistry``. Required (with
            ``parent_session_dir``) to enable delegation tools.
        parent_session_dir: Main agent's session dir. Subagent transcripts
            land at ``<parent_session_dir>/subagents/<run_id>/transcript.jsonl``.
        default_model: Default model string for ``DelegateGeneralTool``
            (falls back to the model used for normal turns).
        lightweight_model: Optional cheaper/faster sibling exposed alongside
            ``default_model`` as the second enum option on
            ``DelegateGeneralTool.model`` (e.g. ``openrouter/openai/gpt-5.4-mini``
            when the default is ``openrouter/openai/gpt-5.4``). When ``None``
            the general delegate accepts only the default. Constrains main-
            agent model picks to a small allowlist so hallucinated sibling
            IDs (e.g. ``gpt-5.1-mini``) cannot reach litellm. NOT plumbed
            into ``DelegateGUITool``: the GUI default is user-controlled
            via ``gui_model`` and shouldn't be silently swapped by the
            main agent.
        thinking_params: Main-agent thinking kwargs, forwarded unchanged into
            the subagent session's ``litellm.acompletion`` call.
        disable_main_computer: When True, replace ``session._computer`` with a
            restricted handler that only allows ``screenshot`` and ``wait``
            actions. The agent can observe the VM and idle (keeping the loop
            alive) but cannot perform interactive GUI work — that must go
            through ``delegate_gui``. Useful for forcing GUI delegation in
            validation runs (Level 2 coverage).
        disable_delegate_gui: When True, omit ``DelegateGUITool`` from the
            returned list. ``DelegateGeneralTool`` and ``SubagentsTool``
            remain. GUI interactions must go through the main ``computer``
            tool. Conflicts with ``disable_main_computer`` — guarded at
            agent construction.
        workspace_root: Absolute path on the VM that bounds read/write/edit
            file access. When ``None``, path policy is permissive
            (matches MilestoneTool / AnalyzeImageTool behavior).
        host_workspace_root: Absolute path on the local host that bounds
            ``target='host'`` file access. When ``None`` (or missing/invalid),
            only ``target='vm'`` is registered and the agent never sees a
            ``host`` option in the schema enum.
        context_window_tokens: Resolved model context window, forwarded to
            ``ReadFileTool`` for adaptive byte-paging (cap = clamp(ctx * 4 *
            0.10, 32 KB, 128 KB) — matches OpenClaw
            ``resolveAdaptiveReadMaxBytes``).
    """
    from ..analyze_image import AnalyzeImageTool
    from ..milestone import MilestoneTool

    milestone_tool = MilestoneTool(session.interface)
    analyze_image_tool = AnalyzeImageTool(
        session.interface,
        model=summary_model,
        thinking_params=vision_thinking_params,
    )
    # `vm` target: route through the vm MCP bridge when an MCPRuntime is supplied,
    # else the legacy direct session RPC. Same `name="vm"` either way, so the
    # tool vocabulary the model sees is identical.
    fs_registry = FilesystemRegistry()
    if mcp_runtime is not None:
        fs_registry.register(MCPBackend(
            mcp_runtime,
            workspace_root=workspace_root,
            os_type=getattr(session, "os_type", None),
        ))
    else:
        fs_registry.register(VMBackend(session.interface, workspace_root=workspace_root))
    if host_workspace_root:
        try:
            fs_registry.register(HostBackend(host_workspace_root))
        except ValueError as e:
            # Bad host root (missing/not-a-dir): warn and skip — agent only
            # sees `target='vm'`. Better than crashing the whole tool stack.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "host backend not registered: %s", e
            )

    read_tool = ReadFileTool(
        fs_registry,
        context_window_tokens=context_window_tokens,
    )
    write_tool = WriteFileTool(fs_registry)
    edit_tool = EditFileTool(fs_registry)
    exec_interface = (
        _MCPExecInterface(mcp_runtime) if mcp_runtime is not None else session.interface
    )
    exec_tool = ExecTool(exec_interface, workspace_root=workspace_root)
    web_search = WebSearchTool()
    web_fetch = WebFetchTool()
    memory_search = MemorySearchTool(memory_store)
    memory_get = MemoryGetTool(memory_store)
    # memory_write intentionally not exposed to the main agent — superseded
    # by write(target='host'). Class lives in memory.py and is still used by
    # memory_flush.py and the GUI subagent's tool list.

    if disable_main_computer:
        computer = _RestrictedComputerHandler(session._computer)
    elif computer_handler is not None:
        # Caller pre-built and initialized an AsyncComputerHandler (e.g. an
        # OpenClawComputerHandler with custom ``keypress`` semantics). Pass it
        # through so upstream's ``make_computer_handler`` returns it as-is on
        # the ``isinstance(_, AsyncComputerHandler)`` check, avoiding the need
        # for orchestration to monkey-patch ``agent.computers.cuaComputerHandler``.
        computer = computer_handler
    else:
        computer = session._computer

    tools: list = [
        computer,
        milestone_tool,
        analyze_image_tool,
        read_tool,
        write_tool,
        edit_tool,
        exec_tool,
        web_search,
        web_fetch,
        memory_search,
        memory_get,
    ]

    if registry is not None and parent_session_dir is not None:
        delegate_general = DelegateGeneralTool(
            registry=registry,
            tools=tools,
            memory_store=memory_store,
            default_model=default_model or summary_model or "",
            summary_model=summary_model or default_model or "",
            parent_session_dir=parent_session_dir,
            thinking_params=thinking_params,
            lightweight_model=lightweight_model,
        )
        gui_tool_kwargs: dict[str, Any] = {
            "registry": registry,
            "session": session,
            "parent_session_dir": parent_session_dir,
            "thinking_params": gui_thinking_params,
            "memory_store": memory_store,
        }
        if gui_model:
            gui_tool_kwargs["default_model"] = gui_model
        subagents_tool = SubagentsTool(registry=registry)
        if disable_delegate_gui:
            tools.extend([delegate_general, subagents_tool])
        else:
            delegate_gui = DelegateGUITool(**gui_tool_kwargs)
            tools.extend([delegate_general, delegate_gui, subagents_tool])

    return tools


def get_tool_summaries(tools: list) -> dict[str, str]:
    """Extract prompt summaries for BaseTool instances plus the primary Computer tool."""
    summaries: dict[str, str] = {}
    for tool in tools:
        name = getattr(tool, "name", None)
        if isinstance(tool, BaseTool):
            summaries[name] = tool.description
            continue
        if _is_computer_tool(tool) or name == COMPUTER_TOOL_NAME:
            summaries[name or COMPUTER_TOOL_NAME] = COMPUTER_TOOL_SUMMARY
    return summaries


# ---------------------------------------------------------------------------
# Logging constants
# ---------------------------------------------------------------------------

_MAX_ARGS_LOG = 200
"""Max characters of serialized arguments to include in start log."""


def _get_action_type_label(item: dict[str, Any]) -> str:
    """Extract a human-readable action type label from a computer_call item.

    Handles both single ``action`` (computer-use-preview) and batched
    ``actions`` array (GPT 5.4).
    """
    # Single action (computer-use-preview)
    action = item.get("action")
    if isinstance(action, dict):
        return action.get("type", "unknown")
    if action is not None:
        return str(action)

    # Batched actions (GPT 5.4)
    actions = item.get("actions")
    if isinstance(actions, list) and actions:
        types = [a.get("type", "?") for a in actions if isinstance(a, dict)]
        return "+".join(types) if types else "unknown"

    return "unknown"

_MAX_RESULT_LOG = 100
"""Max characters of result output to include in end log."""


class ToolLoggingCallback(AsyncCallbackHandler):
    """Logs tool (function) calls with timing via CUA's callback system.

    Adapted from OpenClaw's ``wrapToolWithBeforeToolCallHook`` — but observe-only,
    since CUA callbacks cannot block or modify calls.

    Hooks used:
      - ``on_function_call_start``: log tool name + truncated args, record start time
      - ``on_function_call_end``: log tool name + truncated result + duration
      - ``on_computer_call_start``: log computer action type, record start time
      - ``on_computer_call_end``: log computer action completion + duration
    """

    def __init__(self) -> None:
        self._start_times: dict[str, float] = {}

    # --- Function calls (memory tools, milestone) ---

    async def on_function_call_start(self, item: dict[str, Any]) -> None:
        call_id = item.get("call_id", "unknown")
        name = item.get("name", "unknown")
        args_raw = item.get("arguments", "")

        # Truncate args for log readability
        if isinstance(args_raw, dict):
            args_str = json.dumps(args_raw, ensure_ascii=False)
        else:
            args_str = str(args_raw)
        if len(args_str) > _MAX_ARGS_LOG:
            args_str = args_str[:_MAX_ARGS_LOG] + "…"

        self._start_times[call_id] = time.monotonic()
        print(f"[Tool] {name}({args_str})")

    async def on_function_call_end(
        self, item: dict[str, Any], result: list[dict[str, Any]]
    ) -> None:
        call_id = item.get("call_id", "unknown")
        name = item.get("name", "unknown")

        # Calculate duration
        start = self._start_times.pop(call_id, None)
        duration_ms = round((time.monotonic() - start) * 1000) if start is not None else -1

        # Extract result summary
        result_summary = _extract_result_summary(result)

        duration_str = f"{duration_ms}ms" if duration_ms >= 0 else "?ms"
        print(f"[Tool] {name} -> {result_summary} ({duration_str})")

    # --- Computer calls (mouse, keyboard, screenshot) ---
    # Note: GPT 5.4 uses "actions" (array), computer-use-preview uses "action" (singular)

    async def on_computer_call_start(self, item: dict[str, Any]) -> None:
        call_id = item.get("call_id", "unknown")
        action_type = _get_action_type_label(item)
        self._start_times[call_id] = time.monotonic()
        print(f"[Computer] {action_type}")

    async def on_computer_call_end(
        self, item: dict[str, Any], result: list[dict[str, Any]]
    ) -> None:
        call_id = item.get("call_id", "unknown")
        action_type = _get_action_type_label(item)

        start = self._start_times.pop(call_id, None)
        duration_ms = round((time.monotonic() - start) * 1000) if start is not None else -1
        duration_str = f"{duration_ms}ms" if duration_ms >= 0 else "?ms"
        print(f"[Computer] {action_type} done ({duration_str})")


def _extract_result_summary(result: list[dict[str, Any]]) -> str:
    """Extract a truncated string summary from a function call result list."""
    if not result:
        return "(empty)"

    # Result is typically a list of output dicts; grab the first output string
    for item in result:
        output = item.get("output", "")
        if output:
            s = str(output)
            if len(s) > _MAX_RESULT_LOG:
                return s[:_MAX_RESULT_LOG] + "…"
            return s

    return "(no output)"
