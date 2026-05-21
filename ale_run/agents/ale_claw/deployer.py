"""AleClawDeployer — OpenClaw harness native deployer.

Lives on the host (``runtime: local``) or in a docker container
(``runtime: docker``) — same code, framework picks where to run.

The deployer's surface (per :class:`InProcessHostDeployer`):

  __init__(runtime): stores the runtime context
  install():         sanity-check imports + at least one API key (from base)
  launch(prompt):    runs the OpenClaw harness end-to-end, writes
                     transcripts to runtime.work_dir, returns
                     AgentRunResult
  parse_artifacts():  reads work_dir's transcripts → ATIF Steps via builder

The OpenClaw harness itself is unchanged (lives at :mod:`.harness`,
copied from ``cua_bench/agents/openclaw/`` upstream).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, ClassVar

from ale_run.agents._bases import InProcessHostDeployer
from ale_run.agents.base import (
    AgentRunResult,
    BaseAgentConfig,
)
from ale_run.agents.trajectory import TrajectoryBuilder

from .config import AleClawConfig
from .transcript_to_trajectory import parse_transcripts_into

# Harness imports (all in-tree under harness/).
from .harness import (
    OpenClawComputerAgent,
    OpenClawComputerHandler,
    SessionManager,
    MemoryStore,
    SubagentRegistry,
    build_tools,
    get_tool_summaries,
    ToolLoggingCallback,
    ContextOverflowCallback,
    build_system_prompt_report,
    PromptBuilder,
    ContextFile,
    ThinkingConfig,
    ThinkLevel,
    resolve_thinking_default,
    build_replay_messages,
    sanitize_history,
    limit_history_turns,
    convert_to_responses_api_items,
)
from .harness.agent_loop import has_done_signal
from .harness.context import DEFAULT_CONTEXT_TOKENS, resolve_context_window
from .harness.model_config import resolve_model

logger = logging.getLogger(__name__)

# System-prompt context file shipped with the harness; loaded fresh each launch
_HARNESS_AGENTS_MD = Path(__file__).resolve().parent / "harness" / "AGENTS.md"


class AleClawDeployer(InProcessHostDeployer):
    """OpenClaw harness deployer. Runs on host or in docker container.

    Both ``local`` and ``docker`` runtimes are supported — same code path.
    The docker runtime adds process / fs / env isolation; the local
    runtime is faster for dev. yaml picks one explicitly when both apply
    (with default ``local`` if omitted).

    ``install`` from :class:`InProcessHostDeployer` checks the declared
    imports + API-key env, plus :meth:`_extra_install` here logs the
    resolved config.
    """

    supported_runtimes: ClassVar[frozenset[str]] = frozenset({"local", "docker"})

    required_modules: ClassVar[tuple[str, ...]] = (".harness.agent_loop",)
    api_key_alternatives: ClassVar[tuple[str, ...]] = (
        "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    )

    @property
    def version(self) -> str | None:
        return self.config.upstream_version

    # =========================================================================
    # launch / parse_artifacts
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        """Drive the OpenClaw agent end-to-end against the eval VM.

        Builds memory_store / session_mgr / tools / OpenClawComputerAgent,
        runs the async-generator loop with wall-clock timeout, returns the
        outcome. Transcripts land in ``self.runtime.work_dir`` for
        :meth:`parse_artifacts` to read later.
        """
        cfg: AleClawConfig = self.config  # type: ignore[assignment]
        # work_dir from BaseRuntime is a substrate-native str; local /
        # docker runtimes are host-visible so wrapping in Path is safe.
        work_dir = Path(self.runtime.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Harness-internal id for memory + session keying (just a folder name).
        task_id = uuid.uuid4().hex[:12]
        memory_base = work_dir / "openclaw_memory"
        session_base = work_dir / "openclaw_sessions"
        trajectory_dir = work_dir / "trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ale-claw: launch — work_dir=%s task_id=%s", work_dir, task_id)

        # ---- 1. Drive-VM session (constructed inside the runtime substrate) ----
        # In local runtime this is host → VM RPC. In docker runtime it's
        # container → VM RPC (container has --network host so endpoint reaches).
        session = await self.runtime.make_vm_session()

        # ---- 2. Memory + session + subagent registry ----
        memory_store = MemoryStore(task_id=task_id, base_dir=str(memory_base))
        memory_store.init_session()
        session_mgr = SessionManager(task_id=task_id, base_dir=str(session_base))
        session_mgr.init_session(model=cfg.model)
        registry = SubagentRegistry(persist_path=session_mgr.task_dir / "subagent-runs.jsonl")
        registry.restore()

        # ---- 3. Model resolution + context window ----
        resolved_model = resolve_model(cfg.model)
        summary_model = cfg.summary_model or cfg.lightweight_model or cfg.model
        resolved_summary_model = (
            resolved_model if summary_model == cfg.model
            else resolve_model(summary_model)
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

        workspace_root: str | None = None    # permissive — full VM access
        host_workspace_root = str(memory_store.task_dir.resolve())

        # ---- 4. Thinking config ----
        thinking_config = self._build_thinking_config()
        thinking_api_params = thinking_config.to_api_params(cfg.model)
        gui_thinking_params = thinking_config.gui_params(cfg.gui_model or cfg.model)

        # ---- 5. Pre-build OpenClaw computer handler ----
        computer_handler = None
        if not cfg.disable_main_computer:
            computer_handler = OpenClawComputerHandler(session.computer)
            await computer_handler._initialize()                # noqa: SLF001

        # ---- 6. Tools + disabled_tools filter ----
        tools = build_tools(
            session, memory_store,
            summary_model=summary_model,
            vision_thinking_params=thinking_config.vision_params(
                summary_model, runtime=resolved_summary_model,
            ),
            registry=registry,
            parent_session_dir=session_mgr.task_dir,
            default_model=cfg.model,
            lightweight_model=cfg.lightweight_model,
            thinking_params=thinking_api_params,
            gui_thinking_params=gui_thinking_params,
            disable_main_computer=cfg.disable_main_computer,
            disable_delegate_gui=cfg.disable_delegate_gui,
            gui_model=cfg.gui_model,
            workspace_root=workspace_root,
            host_workspace_root=host_workspace_root,
            context_window_tokens=context_window_tokens,
            computer_handler=computer_handler,
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
            tool_summaries=tool_summaries, context_files=context_files,
        )
        report = build_system_prompt_report(
            system_prompt=instructions, context_files=context_files,
            tool_summaries=tool_summaries, tools=tools,
        )
        session_mgr.set_system_prompt_report(report)

        # ---- 8. Overflow callback + agent ----
        overflow_cb = ContextOverflowCallback(
            model=cfg.model,
            context_window=context_window_tokens,
            instructions_tokens=len(instructions) // 4,
            resolved_model=resolved_model,
        )
        if session_mgr._state is not None:                       # noqa: SLF001
            session_mgr._state.contextTokens = overflow_cb.context_window  # noqa: SLF001
            session_mgr.save_state()

        agent = OpenClawComputerAgent(
            model=cfg.model,
            tools=tools,
            only_n_most_recent_images=3,
            trajectory_dir=trajectory_dir,
            instructions=instructions,
            use_prompt_caching=True,
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

        # ---- 9. Cross-run replay (always empty v1) ----
        prior_entries = session_mgr.load_history()
        replay_messages: list[dict[str, Any]] = []
        if prior_entries:
            replay_messages = build_replay_messages(prior_entries)
            replay_messages = sanitize_history(replay_messages)
            replay_messages = limit_history_turns(replay_messages, cfg.max_history_turns)
            replay_messages = sanitize_history(replay_messages)
            replay_messages = convert_to_responses_api_items(replay_messages)
        run_input = (
            replay_messages + [{"role": "user", "content": prompt}]
            if replay_messages else prompt
        )

        # ---- 10. Drive loop with timeout ----
        # litellm reads OPENROUTER_API_KEY / ANTHROPIC_API_KEY etc straight
        # from os.environ — operator populates the shell, no patching needed.
        max_steps = cfg.max_turns or 100
        total_usage = {
            "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "response_cost": 0.0,
        }
        t0 = time.monotonic()
        step = 0
        task_completed = False
        transcript_path = work_dir / "openclaw_sessions" / task_id / "transcript.jsonl"

        try:
            async def _drive() -> None:
                nonlocal step, task_completed
                async for result in agent.run(run_input):
                    sys.stdout.flush()
                    step += 1
                    for k in total_usage:
                        total_usage[k] += result["usage"].get(k, 0)
                    session_mgr.update_step_count(step)
                    session_mgr.update_tokens(
                        result["usage"].get("input_tokens", 0),
                        result["usage"].get("output_tokens", 0),
                    )
                    if step >= max_steps:
                        logger.info("ale-claw: max_steps %d reached", max_steps)
                        break
                    if has_done_signal(result.get("output", [])):
                        logger.info("ale-claw: done signal at step %d", step)
                        task_completed = True
                        break
            await asyncio.wait_for(_drive(), timeout=cfg.timeout_s)
        except asyncio.TimeoutError:
            logger.warning("ale-claw: wall budget %.0fs exceeded", cfg.timeout_s)
            return AgentRunResult(
                status="timeout",
                duration_s=time.monotonic() - t0,
                error=f"wall budget {cfg.timeout_s}s exceeded",
                transcript_path=str(transcript_path) if transcript_path.exists() else None,
            )
        except Exception as exc:                             # noqa: BLE001
            logger.exception("ale-claw: agent.run threw")
            return AgentRunResult(
                status="failed",
                duration_s=time.monotonic() - t0,
                error=f"{type(exc).__name__}: {exc}",
                transcript_path=str(transcript_path) if transcript_path.exists() else None,
            )

        # Outcome mapping
        if task_completed:
            status = "completed"
            error: str | None = None
        elif step >= max_steps:
            status = "completed"     # finished at step budget — not a wall-clock failure
            error = None
        else:
            status = "failed"
            error = "loop exited without done signal"

        return AgentRunResult(
            status=status,
            duration_s=time.monotonic() - t0,
            transcript_path=str(transcript_path) if transcript_path.exists() else None,
            error=error,
        )

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: BaseAgentConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Parse on-disk transcripts → ATIF Steps via the in-tree translator."""
        if not work_dir.exists():
            builder.add_step(
                source="system",
                message=f"ale-claw: work_dir missing {work_dir}",
                extra={"reason": "no_work_dir"},
            )
            return
        try:
            parse_transcripts_into(work_dir, builder)
        except Exception as exc:                                # noqa: BLE001
            logger.exception("ale-claw: parse_artifacts failed")
            builder.add_step(
                source="system",
                message=f"transcript parse failed: {type(exc).__name__}: {exc}",
                extra={"reason": "parse_error"},
            )
        # Surface useful debug info on trajectory.extra
        builder.trajectory.extra.setdefault("ale_claw", {}).update({
            "work_dir": str(work_dir),
            "version": getattr(config, "upstream_version", None),
            "transcript_path": run_result.transcript_path,
            "run_status": run_result.status,
        })

    # =========================================================================
    # helpers (private)
    # =========================================================================

    def _build_thinking_config(self) -> ThinkingConfig:
        c: AleClawConfig = self.config  # type: ignore[assignment]
        level = (
            ThinkLevel(c.thinking_level) if c.thinking_level
            else resolve_thinking_default(c.model)
        )
        flush = ThinkLevel(c.flush_thinking_level) if c.flush_thinking_level else level
        compact = (
            ThinkLevel(c.compaction_thinking_level) if c.compaction_thinking_level
            else level
        )
        vision = ThinkLevel(c.vision_thinking_level)
        gui = ThinkLevel(c.gui_thinking_level)
        return ThinkingConfig(
            level=level, flush_level=flush, compaction_level=compact,
            vision_level=vision, gui_level=gui,
        )

