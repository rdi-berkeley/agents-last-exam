"""Run ALE-Claw against a CUA desktop session."""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from cua_bench.computers.remote import RemoteDesktopSession

from .config import AleClawConfig
from .harness import (
    ContextFile,
    ContextOverflowCallback,
    MemoryStore,
    OpenClawComputerAgent,
    OpenClawComputerHandler,
    PromptBuilder,
    SessionManager,
    SubagentRegistry,
    ThinkingConfig,
    ThinkLevel,
    ToolLoggingCallback,
    build_system_prompt_report,
    build_tools,
    get_tool_summaries,
    resolve_thinking_default,
)
from .harness.agent_loop import has_done_signal
from .harness.context.context import (
    DEFAULT_CONTEXT_TOKENS,
    is_context_overflow_error,
    resolve_context_window,
)
from .harness.model.model_config import resolve_model

logger = logging.getLogger(__name__)

# System-prompt context file shipped with the harness; loaded fresh each launch
_HARNESS_AGENTS_MD = Path(__file__).resolve().parent / "harness" / "AGENTS.md"


@dataclass
class RunResult:
    status: str
    duration_s: float
    transcript_path: str | None
    error: str | None = None
    error_type: str | None = None
    status_code: int | None = None


async def run(
    prompt: str,
    *,
    config: AleClawConfig,
    sandbox_url: str = "http://127.0.0.1:8000",
    os_type: str = "linux",
    output_dir: str | Path | None = None,
) -> RunResult:
    """Run a prompt on an existing CUA sandbox and save the session on this machine."""
    target = urlsplit(sandbox_url)
    if (
        target.scheme != "http"
        or not target.hostname
        or target.username is not None
        or target.path not in ("", "/")
        or target.query
        or target.fragment
    ):
        raise ValueError("sandbox_url must be a CUA endpoint such as http://host:8000")
    if os_type not in ("linux", "windows"):
        raise ValueError("os_type must be linux or windows")
    session = RemoteDesktopSession(
        api_url=sandbox_url, os_type=os_type, ephemeral=False, headless=True
    )
    try:
        if not await session.check_status():
            raise RuntimeError(f"CUA sandbox unavailable at {sandbox_url}")
        return await _run_session(prompt, config, session, os_type, output_dir)
    finally:
        await session.close()


async def _run_session(
    prompt: str,
    cfg: AleClawConfig,
    session: RemoteDesktopSession,
    os_type: str,
    output_dir: str | Path | None,
) -> RunResult:
    task_id = uuid.uuid4().hex[:12]
    work_dir = Path(output_dir) if output_dir is not None else Path("ale-claw-runs") / task_id
    # Harness-internal id for memory + session keying (just a folder name).
    memory_base = work_dir / "openclaw_memory"
    session_base = work_dir / "openclaw_sessions"
    trajectory_dir = work_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ale-claw: launch — work_dir=%s task_id=%s", work_dir, task_id)

    # ---- 2. Memory + session + subagent registry ----
    memory_store = MemoryStore(task_id=task_id, base_dir=str(memory_base))
    memory_store.init_session()
    session_mgr = SessionManager(task_id=task_id, base_dir=str(session_base))
    session_mgr.init_session(model=cfg.model)
    registry = SubagentRegistry(persist_path=session_mgr.task_dir / "subagent-runs.jsonl")
    registry.restore()

    # ---- 3. Model resolution + context window ----
    resolved_model = resolve_model(cfg.model)
    summary_model = cfg.summary_model or cfg.auxiliary_model or cfg.model
    resolved_summary_model = (
        resolved_model if summary_model == cfg.model else resolve_model(summary_model)
    )
    ctx_override = os.environ.get("CONTEXT_WINDOW_OVERRIDE")
    if ctx_override:
        context_window_tokens = int(ctx_override)
    else:
        context_window_tokens = (
            resolved_model.context_window
            or resolve_context_window(cfg.model)
            or DEFAULT_CONTEXT_TOKENS
        )

    workspace_root: str | None = None  # permissive — full VM access
    host_workspace_root = str(memory_store.task_dir.resolve())

    # ---- 4. Thinking config ----
    thinking_config = _thinking_config(cfg)
    thinking_api_params = thinking_config.to_api_params(cfg.model)
    gui_model = cfg.gui_model or cfg.auxiliary_model or cfg.model
    gui_thinking_params = thinking_config.gui_params(gui_model)

    # ---- 5. Pre-build computer handler ----
    # Pointer coordinate space — most models emit screen pixels, but the
    # Gemini family grounds clicks to a normalized [0, 1000) grid. Detect
    # from the model that actually drives the handler (gui_model when the
    # main agent has no computer), unless the config forces a space.
    from .harness.tools.computer_handler import coordinate_space_for_model

    coord_driver_model = gui_model if cfg.disable_main_computer else cfg.model
    coordinate_space = cfg.coordinate_space or coordinate_space_for_model(coord_driver_model)
    logger.info(
        "ale-claw: coordinate_space=%s (driver_model=%s, override=%s)",
        coordinate_space,
        coord_driver_model,
        cfg.coordinate_space,
    )

    computer_handler = None
    if not cfg.disable_main_computer:
        computer_handler = OpenClawComputerHandler(
            session.computer, coordinate_space=coordinate_space
        )
        await computer_handler._initialize()

    # ---- 6. Tools + disabled_tools filter ----
    tool_loop = cfg.tool_loop_override
    tools = build_tools(
        session,
        memory_store,
        summary_model=summary_model,
        vision_thinking_params=thinking_config.vision_params(
            summary_model,
            runtime=resolved_summary_model,
        ),
        registry=registry,
        parent_session_dir=session_mgr.task_dir,
        default_model=cfg.model,
        auxiliary_model=cfg.auxiliary_model,
        thinking_params=thinking_api_params,
        thinking_config=thinking_config,
        gui_thinking_params=gui_thinking_params,
        generation_kwargs=cfg.extra_generation_kwargs,
        disable_main_computer=cfg.disable_main_computer,
        disable_delegate_gui=cfg.disable_delegate_gui,
        gui_model=gui_model,
        tool_loop=tool_loop,
        workspace_root=workspace_root,
        host_workspace_root=host_workspace_root,
        context_window_tokens=context_window_tokens,
        computer_handler=computer_handler,
        api_key=cfg.api_key,
        api_base=cfg.base_url,
    )
    if cfg.disabled_tools:
        tools = [t for t in tools if getattr(t, "name", "") not in cfg.disabled_tools]
        logger.info("ale-claw: disabled_tools=%s", cfg.disabled_tools)
    tool_summaries = get_tool_summaries(tools)

    # ---- 7. System prompt + AGENTS.md + TASK_MEMORY.md context ----
    agents_md = _HARNESS_AGENTS_MD.read_text(encoding="utf-8")
    context_files = [ContextFile(path="AGENTS.md", content=agents_md)]
    bootstrap = memory_store.get_bootstrap_context()
    if bootstrap:
        context_files.append(ContextFile(path="TASK_MEMORY.md", content=bootstrap))
    instructions = PromptBuilder().build(
        tool_summaries=tool_summaries,
        context_files=context_files,
        target_os=os_type,
    )
    report = build_system_prompt_report(
        system_prompt=instructions,
        context_files=context_files,
        tool_summaries=tool_summaries,
        tools=tools,
    )
    session_mgr.set_system_prompt_report(report)

    # ---- 8. Overflow callback + agent ----
    overflow_cb = ContextOverflowCallback(
        model=cfg.model,
        context_window=context_window_tokens,
        instructions_tokens=len(instructions) // 4,
        resolved_model=resolved_model,
    )
    if session_mgr._state is not None:
        session_mgr._state.context_tokens = overflow_cb.context_window
        session_mgr.save_state()

    agent = OpenClawComputerAgent(
        tool_loop=tool_loop,
        model=cfg.model,
        tools=tools,
        only_n_most_recent_images=3,
        trajectory_dir=trajectory_dir,
        instructions=instructions,
        use_prompt_caching=cfg.tool_calling != "prompt",
        callbacks=[ToolLoggingCallback()],
        context_files=context_files,
        image_retention_mode=cfg.image_retention_mode,
        auto_screenshot=False,
        overflow_cb=overflow_cb,
        session_mgr=session_mgr,
        memory_store=memory_store,
        summary_model=summary_model,
        thinking_config=thinking_config,
        resolved_model=resolved_model,
        summary_runtime=resolved_summary_model,
        registry=registry,
        **thinking_api_params,
    )

    # ---- 9. Drive loop ----
    # The episode wall budget is orchestration-owned: the executor wraps
    # launch() in asyncio.wait_for(timeout=timeout_s) (derived from the
    # task), so we drive the loop directly here; a cancellation on the
    # budget propagates cleanly (no subprocess to reap).
    # litellm reads OPENROUTER_API_KEY / ANTHROPIC_API_KEY etc straight
    # from os.environ — operator populates the shell, no patching needed.
    max_steps = cfg.max_turns
    t0 = time.monotonic()
    step = 0
    task_completed = False
    transcript_path = work_dir / "openclaw_sessions" / task_id / "transcript.jsonl"
    from .harness.model.usage import start_helper_usage, stop_helper_usage

    usage_token = start_helper_usage(work_dir / "helper_usage.jsonl")

    try:
        async for result in agent.run(
            prompt,
            api_base=cfg.base_url,
            api_key=cfg.api_key,
            max_turns=max_steps,
            **cfg.extra_generation_kwargs,
        ):
            sys.stdout.flush()
            step += 1
            if has_done_signal(result.get("output", [])):
                logger.info("ale-claw: done signal at step %d", step)
                task_completed = True
    except Exception as exc:
        logger.exception("ale-claw: agent.run threw")
        return RunResult(
            status="failed",
            duration_s=time.monotonic() - t0,
            error=f"{type(exc).__name__}: {exc}",
            error_type="context_limit" if is_context_overflow_error(str(exc)) else None,
            status_code=getattr(exc, "status_code", None),
            transcript_path=str(transcript_path) if transcript_path.exists() else None,
        )
    finally:
        try:
            await registry.close()
        finally:
            stop_helper_usage(usage_token)
    # Outcome mapping
    error_type = None
    if task_completed:
        status = "completed"
        error: str | None = None
    elif max_steps is not None and step >= max_steps:
        status = "failed"
        error_type = "step_limit"
        error = f"max_turns {max_steps} reached before DONE"
    else:
        status = "failed"
        error = "loop exited without done signal"

    return RunResult(
        status=status,
        duration_s=time.monotonic() - t0,
        transcript_path=str(transcript_path) if transcript_path.exists() else None,
        error=error,
        error_type=error_type,
    )


def _thinking_config(c: AleClawConfig) -> ThinkingConfig:
    level = ThinkLevel(c.thinking_level) if c.thinking_level else resolve_thinking_default(c.model)
    flush = ThinkLevel(c.flush_thinking_level) if c.flush_thinking_level else level
    compact = ThinkLevel(c.compaction_thinking_level) if c.compaction_thinking_level else level
    vision = ThinkLevel(c.vision_thinking_level)
    gui = ThinkLevel(c.gui_thinking_level)
    return ThinkingConfig(
        level=level,
        flush_level=flush,
        compaction_level=compact,
        vision_level=vision,
        gui_level=gui,
        flush_level_overridden=c.flush_thinking_level is not None,
        compaction_level_overridden=c.compaction_thinking_level is not None,
    )
