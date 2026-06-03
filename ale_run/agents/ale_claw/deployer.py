"""AleClawDeployer — OpenClaw harness native deployer.

Lives on the host (``runtime: local``) or in a docker container
(``runtime: docker``) — same code, framework picks where to run.

The deployer's surface:

  __init__(executor): stores the executor (per-unit context + I/O)
  install():          import-check the harness modules + at least one
                      API key env var
  launch(prompt):     runs the OpenClaw harness end-to-end, writes
                      transcripts to executor.work_dir, returns
                      AgentRunResult
  parse_artifacts():  reads work_dir's transcripts → ATIF Steps via builder

The OpenClaw harness itself is unchanged (lives at :mod:`.harness`,
copied from ``cua_bench/agents/openclaw/`` upstream).
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentDeployer,
    TrajectoryBuilder,
)

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
from .harness.context.context import DEFAULT_CONTEXT_TOKENS, resolve_context_window
from .harness.inference.model_config import resolve_model

logger = logging.getLogger(__name__)

# System-prompt context file shipped with the harness; loaded fresh each launch
_HARNESS_AGENTS_MD = Path(__file__).resolve().parent / "harness" / "AGENTS.md"


class AleClawDeployer(BaseAgentDeployer):
    """OpenClaw harness deployer. Runs on host or in docker container.

    Both ``local`` and ``docker`` executors are supported — same code path.
    The docker executor adds process / fs / env isolation; the local
    one is faster for dev. yaml picks one explicitly when both apply
    (with default ``local`` if omitted).
    """

    default_executor: ClassVar[str] = "local"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"local", "docker"})

    # Modules ``install`` will import-fail-fast on (typo-catching).
    _required_modules: ClassVar[tuple[str, ...]] = (".harness.agent_loop",)
    # At least one of these env vars must be set or ``install`` raises.
    _api_key_alternatives: ClassVar[tuple[str, ...]] = (
        "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    )

    @property
    def version(self) -> str | None:
        return self.config.upstream_version

    # =========================================================================
    # install — fast-fail on import + API key checks, then mkdir work_dir
    # =========================================================================

    async def install(self) -> None:
        import importlib
        import os as _os

        # Relative-imports anchor: this deployer's parent package
        # (i.e. ``ale_run.agents.ale_claw``).
        deployer_pkg = type(self).__module__.rsplit(".", 1)[0]
        for mod in self._required_modules:
            try:
                if mod.startswith("."):
                    importlib.import_module(mod, package=deployer_pkg)
                else:
                    importlib.import_module(mod)
            except ImportError as e:
                raise RuntimeError(
                    f"{type(self).__name__}: failed to import {mod!r}: {e}"
                ) from e
        if not any(_os.environ.get(k) for k in self._api_key_alternatives):
            raise RuntimeError(
                f"{type(self).__name__}: no LLM API key in env — set one of "
                f"{', '.join(self._api_key_alternatives)}"
            )
        Path(self.executor.work_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "%s: install ok (model=%s, work_dir=%s, executor=%s)",
            type(self).__name__,
            getattr(self.config, "model", "?"),
            self.executor.work_dir,
            self.executor.type,
        )

    # =========================================================================
    # launch / parse_artifacts
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        """Drive the OpenClaw agent end-to-end against the eval VM.

        Builds memory_store / session_mgr / tools / OpenClawComputerAgent,
        runs the async-generator loop with wall-clock timeout, returns the
        outcome. Transcripts land in ``self.executor.work_dir`` for
        :meth:`parse_artifacts` to read later.
        """
        cfg: AleClawConfig = self.config  # type: ignore[assignment]
        # work_dir from BaseExecutor is a substrate-native str; local /
        # docker runtimes are host-visible so wrapping in Path is safe.
        work_dir = Path(self.executor.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Harness-internal id for memory + session keying (just a folder name).
        task_id = uuid.uuid4().hex[:12]
        memory_base = work_dir / "openclaw_memory"
        session_base = work_dir / "openclaw_sessions"
        trajectory_dir = work_dir / "trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ale-claw: launch — work_dir=%s task_id=%s", work_dir, task_id)

        # ---- 1. Drive-VM session (deployer-side, talks to eval cua-server) ----
        # In local executor this is host → VM RPC. In docker executor it's
        # container → VM RPC (container has --network host so endpoint reaches).
        from cua_bench.computers.remote import RemoteDesktopSession

        sb = self.executor.sandbox
        session = RemoteDesktopSession(
            api_url=sb.endpoint,
            os_type=sb.os,
            ephemeral=False,        # env lifecycle is owned by ALEEnv
            headless=True,
        )
        await session.check_status()

        # ---- 1b. MCP substrate (non-GUI tools route through the vm bridge) ----
        # Build the runtime object now (so build_tools can wire the backends to
        # it); it is *connected* later, around the drive loop, and torn down with
        # it. GUI stays on `session` in Phase 1. For the `local`/`docker`
        # executor the bridge runs on the host and points at the same cua-server
        # endpoint the harness already uses (cua_bridge_url == sb.endpoint), so
        # there is no extra network hop.
        mcp_runtime = None
        if cfg.substrate_transport == "mcp":
            from ale_run.agents._bootstrap import (
                cua_bridge_env,
                ensure_cua_mcp_server_at,
                ensure_node_npm,
                ensure_vm_mcp_server,
                vm_bridge_env,
            )
            from mcp.client.stdio import StdioServerParameters

            from .harness.tools.mcp_runtime import MCPRuntime

            node_path, _ = await ensure_node_npm()
            servers: dict[str, Any] = {}
            vm_bridge_dir = await ensure_vm_mcp_server(str(work_dir / "mcp" / "vm"))
            servers["vm"] = StdioServerParameters(
                command=node_path,
                args=[os.path.join(vm_bridge_dir, "src", "index.js")],
                env={**os.environ, **vm_bridge_env(self.executor)},
            )
            if cfg.gui_transport == "mcp":
                cua_bridge_dir = await ensure_cua_mcp_server_at(str(work_dir / "mcp" / "cua"))
                servers["cua"] = StdioServerParameters(
                    command=node_path,
                    args=[os.path.join(cua_bridge_dir, "src", "index.js")],
                    env={**os.environ, **cua_bridge_env(self.executor)},
                )
            mcp_runtime = MCPRuntime(servers)
            logger.info(
                "ale-claw: substrate_transport=mcp gui_transport=%s (servers=%s)",
                cfg.gui_transport, sorted(servers),
            )

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

        # ---- 5. Pre-build computer handler ----
        # gui_transport=mcp → drive GUI through the cua bridge; else the session
        # handler. The MCP handler inits lazily (the runtime connects later,
        # around the drive loop), so don't _initialize it here.
        computer_handler = None
        if not cfg.disable_main_computer:
            if cfg.gui_transport == "mcp" and mcp_runtime is not None:
                from .harness.computer_handler import MCPComputerHandler
                computer_handler = MCPComputerHandler(mcp_runtime, os_type=sb.os)
            else:
                computer_handler = OpenClawComputerHandler(session.computer)
                await computer_handler._initialize()            # noqa: SLF001

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
            mcp_runtime=mcp_runtime,
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

        # ---- 10. Drive loop ----
        # The episode wall budget is orchestration-owned: the executor wraps
        # launch() in asyncio.wait_for(timeout=timeout_s) (derived from the
        # task), so we drive the loop directly here; a cancellation on the
        # budget propagates cleanly (no subprocess to reap).
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

        # Connect the MCP bridge(s) for the duration of the drive loop and tear
        # them down (terminating the node children) on any exit — success,
        # exception, or wall-budget cancellation. A startup failure here is
        # caught by the except below and surfaced as a failed run.
        mcp_stack = AsyncExitStack()
        try:
            if mcp_runtime is not None:
                await mcp_stack.enter_async_context(mcp_runtime)

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
            await _drive()
        except Exception as exc:                             # noqa: BLE001
            logger.exception("ale-claw: agent.run threw")
            return AgentRunResult(
                status="failed",
                duration_s=time.monotonic() - t0,
                error=f"{type(exc).__name__}: {exc}",
                transcript_path=str(transcript_path) if transcript_path.exists() else None,
            )
        finally:
            await mcp_stack.aclose()

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
        config: AleClawConfig,
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

