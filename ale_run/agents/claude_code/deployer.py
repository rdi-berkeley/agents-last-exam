"""ClaudeCodeDeployer — drives the @anthropic-ai/claude-code CLI.

This deployer is **pure Python stdlib** — it uses ``subprocess`` /
``pathlib`` / ``os`` / ``asyncio`` directly. Whatever substrate the
framework's :class:`BaseExecutor` places the deployer in (sandbox VM /
docker container / host process), the agent code is identical: it just
spawns the local ``claude`` CLI and waits.

Responsibilities (claude-code-specific only):

* probe the ``claude`` binary is on PATH
* write the cua MCP config the CLI reads via ``--mcp-config``
* compose the OpenRouter-vs-Anthropic env var dance
* spawn the CLI with stdin from ``prompt.txt``, stdout to
  ``transcript.jsonl``, stderr to ``stderr.log``
* poll the process, time-bound it, surface failure diagnostics
* :meth:`parse_artifacts` — host-side, reads gathered transcript.jsonl
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentConfig,
    BaseAgentDeployer,
    ContentPart,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

from .config import ClaudeCodeConfig

logger = logging.getLogger(__name__)


_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0


class ClaudeCodeDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the @anthropic-ai/claude-code CLI."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    @property
    def version(self) -> str | None:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        return cfg.cli_version

    # =========================================================================
    # install
    # =========================================================================

    async def _auto_install_cli(self) -> None:
        """Install claude CLI via npm; bootstrap node+npm if missing."""
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        npm = shutil.which("npm")
        if not npm:
            from ale_run.agents._bootstrap import ensure_npm
            npm = await ensure_npm()
        home = os.path.expanduser("~")
        env = {**os.environ, "npm_config_cache": f"{home}/.npm-ale"}
        # cfg.cli_version is the full npm spec, e.g.
        # "@anthropic-ai/claude-code@2.1.85".
        pkg = cfg.cli_version or "@anthropic-ai/claude-code"
        proc = await asyncio.to_thread(
            subprocess.run,
            [npm, "install", "-g", "--prefix", f"{home}/.local", pkg],
            capture_output=True, text=True, timeout=300, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"npm install -g {pkg} failed "
                f"(rc={proc.returncode}): {(proc.stderr or '')[:500]}"
            )
        bin_dir = f"{home}/.local/bin"
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        logger.info("claude_code: auto-installed via npm — %s", (proc.stdout or "").strip()[-200:])

    async def install(self) -> None:
        sandbox = self.executor.sandbox
        is_linux = sandbox.is_linux

        # 1. Discover — or install — the claude binary.
        claude_path = shutil.which("claude")
        if not claude_path:
            logger.info("claude_code: 'claude' not on PATH, installing via npm …")
            await self._auto_install_cli()
            claude_path = shutil.which("claude")
            if not claude_path:
                raise RuntimeError(
                    "ClaudeCodeDeployer: 'claude' still not found after "
                    "npm install -g @anthropic-ai/claude-code"
                )
        self._claude_path = claude_path

        # 2. Version probe. Pass stdin=DEVNULL so the probe never blocks waiting
        # on a TTY/stdin (on Windows the freshly-uploaded claude.EXE could hang
        # the 30s probe on first exec — Defender scan + stdin check); 60s gives
        # cold-image first-exec headroom.
        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [claude_path, "--version"],
                capture_output=True, text=True, timeout=60,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"ClaudeCodeDeployer: claude --version timed out: {e}"
            )
        if probe.returncode != 0:
            raise RuntimeError(
                f"ClaudeCodeDeployer: claude --version rc={probe.returncode} "
                f"stderr={(probe.stderr or '').strip()[:300]}"
            )
        logger.info("claude_code: claude CLI ok — %s", (probe.stdout or "").strip())

        # 3. Make work_dir.
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # 4. Ensure the cua MCP bridge is installed at sandbox.mcp_server_dir
        # (idempotent: no-op when prebaked, install when missing).
        from ale_run.agents._bootstrap import cua_bridge_env, ensure_cua_mcp_server
        await ensure_cua_mcp_server(sandbox)

        # 5. MCP config. Paths reference the sandbox's baked node +
        # mcp_server_dir — these are valid because the deployer runs INSIDE
        # the sandbox (SandboxExecutor) and the cua MCP server is on the
        # same machine. CUA_SERVER_URL points the bridge at the image's
        # cua-server port (the bridge otherwise defaults to 5000).
        mcp_config = {
            "mcpServers": {
                "cua": {
                    "command": sandbox.node,
                    "args": [self._join(sandbox.mcp_server_dir, "src", "index.js",
                                        is_linux=is_linux)],
                    "env": cua_bridge_env(self.executor),
                },
            },
        }
        mcp_path = wd / "mcp_config.json"
        mcp_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
        logger.info("claude_code: mcp_config staged at %s", mcp_path)

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        prompt_file = wd / "prompt.txt"
        transcript_file = wd / "transcript.jsonl"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "claude.pid"
        mcp_config = wd / "mcp_config.json"

        # Reset prior-run files so the puller's "rotation detected" logic
        # sees a clean slate.
        for f in (transcript_file, stderr_log, pid_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        prompt_file.write_text(prompt, encoding="utf-8")

        argv = self._build_argv(
            claude_path=self._claude_path,
            cfg=cfg,
            mcp_config=str(mcp_config),
        )
        env = self._build_env(cfg)

        t0 = time.monotonic()
        # Open output files; subprocess inherits the descriptors and the
        # parent's references can close after spawn (the child keeps them).
        with open(prompt_file, "rb") as pin, \
             open(transcript_file, "wb") as tout, \
             open(stderr_log, "wb") as terr:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                argv,
                stdin=pin,
                stdout=tout,
                stderr=terr,
                env=env,
                cwd=str(wd),
                # Detach: child outlives any incidental signal sent to us.
                start_new_session=True if hasattr(os, "setsid") else False,
            )
        pid_file.write_text(str(proc.pid), encoding="ascii")
        logger.info("claude_code: spawned pid=%s argv0=%s", proc.pid, argv[0])

        # Poll until done or timeout.
        deadline = t0 + cfg.timeout_s
        while proc.poll() is None:
            if time.monotonic() > deadline:
                # TERM then KILL — give it a moment to flush.
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(proc.wait), timeout=_TERM_GRACE_S,
                    )
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                duration_s = time.monotonic() - t0
                return AgentRunResult(
                    status="timeout",
                    pid=proc.pid,
                    transcript_path=str(transcript_file),
                    stderr_path=str(stderr_log),
                    duration_s=duration_s,
                    error=f"wall budget {cfg.timeout_s}s exceeded",
                )
            await asyncio.sleep(_POLL_INTERVAL_S)

        duration_s = time.monotonic() - t0
        exit_code = proc.returncode
        status = "completed" if exit_code == 0 else "failed"
        error: str | None = None
        if status == "failed":
            error = self._diagnose_failure(
                stderr_log=stderr_log,
                transcript=transcript_file,
                exit_code=exit_code,
            )
        return AgentRunResult(
            status=status,
            pid=proc.pid,
            exit_code=exit_code,
            transcript_path=str(transcript_file),
            stderr_path=str(stderr_log),
            duration_s=duration_s,
            error=error,
        )

    # =========================================================================
    # internals
    # =========================================================================

    @staticmethod
    def _join(*parts: str, is_linux: bool) -> str:
        """OS-aware path join in the substrate convention."""
        sep = "/" if is_linux else "\\"
        head = parts[0].rstrip("/\\")
        tail = sep.join(p.strip("/\\") for p in parts[1:])
        return f"{head}{sep}{tail}" if tail else head

    @staticmethod
    def _build_argv(
        *, claude_path: str, cfg: ClaudeCodeConfig, mcp_config: str,
    ) -> list[str]:
        argv = [
            claude_path,
            "-p", "-",
            "--output-format", "stream-json", "--verbose",
            "--mcp-config", mcp_config,
            "--model", cfg.model,
        ]
        if cfg.max_turns is not None and cfg.max_turns >= 0:
            argv += ["--max-turns", str(cfg.max_turns)]
        if cfg.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(cfg.max_budget_usd)]
        if cfg.dangerously_skip_permissions:
            argv += ["--dangerously-skip-permissions"]
        for tool in cfg.disabled_tools:
            argv += ["--disallowedTools", tool]
        return argv

    def _build_env(self, cfg: ClaudeCodeConfig) -> dict[str, str]:
        """Compose the env dict subprocess will see.

        OpenRouter remap mirrors the previous shell-script logic, just in
        Python so it works identically on linux + windows.
        """
        env = os.environ.copy()
        # Inject framework-supplied env (api keys, base URLs) on top —
        # _sandbox_entry already merged these into os.environ when
        # running in sandbox; this is a belt-and-braces overwrite for
        # the local executor case where install() may not have triggered it.
        for k, v in (self.executor.env or {}).items():
            env[k] = v

        # Provider-driven routing (explicit, not key-presence heuristic).
        if cfg.provider == "openrouter":
            or_key = env.get("OPENROUTER_API_KEY")
            if not or_key:
                raise RuntimeError(
                    "claude_code: provider=openrouter but OPENROUTER_API_KEY "
                    "is not set"
                )
            env["ANTHROPIC_BASE_URL"] = cfg.base_url or "https://openrouter.ai/api"
            env["ANTHROPIC_AUTH_TOKEN"] = or_key
            env["ANTHROPIC_API_KEY"] = ""
        elif cfg.provider == "direct":
            if not env.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "claude_code: provider=direct but ANTHROPIC_API_KEY is "
                    "not set"
                )
            if cfg.base_url:
                env["ANTHROPIC_BASE_URL"] = cfg.base_url
        else:
            raise RuntimeError(
                f"claude_code: unknown provider {cfg.provider!r} "
                "(expected 'openrouter' or 'direct')"
            )
        return env

    def _diagnose_failure(
        self, *, stderr_log: Path, transcript: Path, exit_code: int | None,
    ) -> str:
        """Build a diagnostic string from log files (best-effort reads)."""
        parts = [f"agent failed (rc={exit_code})"]
        stderr_text = _read_text_tolerant(stderr_log)
        tx_text = _read_text_tolerant(transcript)
        parts.append(f"stderr={len(stderr_text)}B transcript={len(tx_text)}B")
        if stderr_text.strip():
            parts.append(f"stderr tail: ...{stderr_text[-800:]}")
        if '"authentication_failed"' in tx_text or '"User not found"' in tx_text:
            parts.append("LLM auth failed (check api keys)")
        elif '"error_status":429' in tx_text or '"rate_limit_error"' in tx_text:
            parts.append("LLM rate-limited")
        elif '"error_status":5' in tx_text:
            parts.append("LLM upstream 5xx")
        elif '"type":"result"' not in tx_text and exit_code != 0:
            parts.append("agent never produced result event")
        if tx_text.strip():
            parts.append(f"transcript tail: ...{tx_text[-800:]}")
        return " | ".join(parts)

    # =========================================================================
    # parse_artifacts — host-side, runs on gathered transcript.jsonl
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: BaseAgentConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        transcript_file = work_dir / "transcript.jsonl"
        if not transcript_file.exists():
            builder.add_step(
                source="system",
                message=f"claude-code: no transcript at {transcript_file}",
                extra={"reason": "no_transcript"},
            )
            return

        raw = transcript_file.read_text(encoding="utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            cls._consume_event(event, builder)

        builder.trajectory.extra.setdefault("claude_code", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(transcript_file),
            "stderr_path": run_result.stderr_path,
        })

    @classmethod
    def _consume_event(cls, event: dict, builder: TrajectoryBuilder) -> None:
        etype = event.get("type")
        if etype == "assistant":
            cls._consume_assistant(event, builder)
        elif etype == "user":
            cls._consume_user(event, builder)
        elif etype == "system":
            builder.trajectory.extra.setdefault("system_events", []).append(event)
        elif etype == "result":
            builder.trajectory.extra["result"] = event

    @staticmethod
    def _consume_assistant(event: dict, builder: TrajectoryBuilder) -> None:
        message = event.get("message", {}) or {}
        content_blocks = message.get("content", []) or []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id") or "",
                    name=block.get("name") or "",
                    arguments=block.get("input") or {},
                ))
        usage = message.get("usage") or {}
        metrics = StepMetrics(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cache_read_tokens=usage.get("cache_read_input_tokens"),
            cache_creation_tokens=usage.get("cache_creation_input_tokens"),
        )
        builder.add_step(
            source="agent",
            message="\n".join(p for p in text_parts if p) or None,
            tool_calls=tool_calls,
            metrics=metrics,
            extra={"stop_reason": message.get("stop_reason")},
        )

    @staticmethod
    def _consume_user(event: dict, builder: TrajectoryBuilder) -> None:
        message = event.get("message", {}) or {}
        content_blocks = message.get("content", []) or []
        results: list[ToolResult] = []
        text_parts: list[str] = []
        for block in content_blocks:
            btype = block.get("type")
            if btype == "tool_result":
                content = block.get("content")
                parts: list[ContentPart] = []
                if isinstance(content, str):
                    parts.append(ContentPart(type="text", text=content))
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            parts.append(ContentPart(type="text", text=c.get("text", "")))
                results.append(ToolResult(
                    tool_call_id=block.get("tool_use_id") or "",
                    content=parts,
                    is_error=bool(block.get("is_error")),
                ))
            elif btype == "text":
                text_parts.append(block.get("text", ""))
        builder.add_step(
            source="environment",
            message="\n".join(p for p in text_parts if p) or None,
            observation=Observation(results=results),
        )


def _read_text_tolerant(path: Path) -> str:
    """Best-effort text read; never raises."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""
