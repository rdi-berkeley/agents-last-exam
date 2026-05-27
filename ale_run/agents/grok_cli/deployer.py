"""GrokCliDeployer — drives the ``grok`` CLI from superagent-ai.

Standalone binary (no Node runtime for CLI itself).  Node IS required
for the CUA MCP Server bridge.  Supports direct xAI or OpenRouter
routing via ``GROK_BASE_URL``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
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

from .config import GrokCliConfig, native_to_openrouter_model

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0
_GROK_CLI_VERSION = "1.1.5"


class GrokCliDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the ``grok`` CLI."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    @property
    def version(self) -> str | None:
        return _GROK_CLI_VERSION

    # =========================================================================
    # install
    # =========================================================================

    async def _auto_install_cli(self) -> None:
        home = os.path.expanduser("~")
        proc = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c",
             "curl -fsSL https://raw.githubusercontent.com/superagent-ai/grok-cli/main/install.sh | bash"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "HOME": home},
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"grok-cli install failed (rc={proc.returncode}): "
                f"{(proc.stderr or '')[:500]}"
            )
        grok_bin = f"{home}/.grok/bin"
        if grok_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{grok_bin}:{os.environ.get('PATH', '')}"
        logger.info("grok_cli: installed — %s", (proc.stdout or "").strip()[-200:])

    async def install(self) -> None:
        cfg: GrokCliConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        grok_path = shutil.which("grok")
        if not grok_path:
            logger.info("grok_cli: 'grok' not on PATH, installing …")
            await self._auto_install_cli()
            grok_path = shutil.which("grok")
            if not grok_path:
                raise RuntimeError(
                    "GrokCliDeployer: 'grok' still not found after install"
                )
        self._grok_path = grok_path

        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [grok_path, "--version"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"grok --version timed out: {e}")
        logger.info("grok_cli: CLI ok — %s", (probe.stdout or "").strip())

        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # MCP + settings: ~/.grok/user-settings.json
        home = os.path.expanduser("~")
        grok_home = Path(home) / ".grok"
        grok_home.mkdir(parents=True, exist_ok=True)

        settings = {
            "mcp": {
                "servers": [
                    {
                        "id": "cua",
                        "label": "CUA MCP Server",
                        "enabled": True,
                        "transport": "stdio",
                        "command": sandbox.node,
                        "args": [self._join(sandbox.mcp_server_dir, "src", "index.js",
                                            is_linux=sandbox.is_linux)],
                    },
                ],
            },
            "disabledTools": list(cfg.disabled_tools),
        }
        (grok_home / "user-settings.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8",
        )
        logger.info("grok_cli: config staged at %s", grok_home)

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: GrokCliConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        prompt_file = wd / "prompt.txt"
        transcript_file = wd / "transcript.jsonl"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "grok.pid"

        for f in (transcript_file, stderr_log, pid_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        prompt_file.write_text(prompt, encoding="utf-8")

        argv = self._build_argv(cfg, prompt)
        env = self._build_env(cfg)

        t0 = time.monotonic()
        with open(transcript_file, "wb") as tout, \
             open(stderr_log, "wb") as terr:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                argv,
                stdin=subprocess.DEVNULL,
                stdout=tout,
                stderr=terr,
                env=env,
                cwd=str(wd),
                start_new_session=True if hasattr(os, "setsid") else False,
            )
        pid_file.write_text(str(proc.pid), encoding="ascii")
        logger.info("grok_cli: spawned pid=%s", proc.pid)

        deadline = t0 + cfg.timeout_s
        while proc.poll() is None:
            if time.monotonic() > deadline:
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
                return AgentRunResult(
                    status="timeout",
                    pid=proc.pid,
                    transcript_path=str(transcript_file),
                    stderr_path=str(stderr_log),
                    duration_s=time.monotonic() - t0,
                    error=f"wall budget {cfg.timeout_s}s exceeded",
                )
            await asyncio.sleep(_POLL_INTERVAL_S)

        duration_s = time.monotonic() - t0
        exit_code = proc.returncode
        status = "completed" if exit_code == 0 else "failed"
        error: str | None = None
        if status == "failed":
            error = _diagnose_failure(stderr_log, transcript_file, exit_code)
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
        sep = "/" if is_linux else "\\"
        head = parts[0].rstrip("/\\")
        tail = sep.join(p.strip("/\\") for p in parts[1:])
        return f"{head}{sep}{tail}" if tail else head

    def _build_argv(self, cfg: GrokCliConfig, prompt: str) -> list[str]:
        effective_model = cfg.model
        if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("GROK_API_KEY"):
            effective_model = native_to_openrouter_model(cfg.model)

        max_rounds = cfg.max_tool_rounds
        if max_rounds == -1:
            max_rounds = 100_000

        argv = [
            self._grok_path,
            "--prompt", prompt,
            "--model", effective_model,
            "--format", "json",
            "--max-tool-rounds", str(max_rounds),
        ]
        return argv

    def _build_env(self, cfg: GrokCliConfig) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env["NO_COLOR"] = "1"
        # OpenRouter routing
        if env.get("OPENROUTER_API_KEY") and not env.get("GROK_API_KEY"):
            env["GROK_API_KEY"] = env["OPENROUTER_API_KEY"]
            env["GROK_BASE_URL"] = "https://openrouter.ai/api/v1"
        return env

    # =========================================================================
    # parse_artifacts
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
                message=f"grok-cli: no transcript at {transcript_file}",
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

        builder.trajectory.extra.setdefault("grok_cli", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(transcript_file),
        })

    @classmethod
    def _consume_event(cls, event: dict, builder: TrajectoryBuilder) -> None:
        etype = event.get("type")
        if etype == "text":
            builder.add_step(source="agent", message=event.get("text", ""))
        elif etype == "tool_use":
            cls._consume_tool_use(event, builder)
        elif etype == "step_finish":
            cls._consume_step_finish(event, builder)
        elif etype == "error":
            builder.add_step(
                source="system",
                message=event.get("message", "unknown error"),
            )

    @staticmethod
    def _consume_tool_use(event: dict, builder: TrajectoryBuilder) -> None:
        tc = event.get("toolCall", {})
        func = tc.get("function", {})
        args = func.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}

        builder.add_step(
            source="agent",
            tool_calls=[ToolCall(
                id=tc.get("id", ""),
                name=func.get("name", ""),
                arguments=args,
            )],
        )

        tr = event.get("toolResult", {})
        if tr:
            output = tr.get("output", "")
            error = tr.get("error")
            text = error if error else (output if isinstance(output, str) else json.dumps(output))
            builder.add_step(
                source="environment",
                observation=Observation(results=[
                    ToolResult(
                        tool_call_id=tc.get("id", ""),
                        content=[ContentPart(type="text", text=text)],
                        is_error=not tr.get("success", True),
                    ),
                ]),
            )

    @staticmethod
    def _consume_step_finish(event: dict, builder: TrajectoryBuilder) -> None:
        usage = event.get("usage", {})
        if usage:
            builder.trajectory.extra.setdefault("grok_cli", {}).setdefault(
                "usage_steps", [],
            ).append(usage)


def _diagnose_failure(stderr_log: Path, transcript: Path, exit_code: int | None) -> str:
    parts = [f"agent failed (rc={exit_code})"]
    stderr_text = _read_text_tolerant(stderr_log)
    tx_text = _read_text_tolerant(transcript)
    if stderr_text.strip():
        parts.append(f"stderr tail: ...{stderr_text[-800:]}")
    if tx_text.strip():
        parts.append(f"transcript tail: ...{tx_text[-800:]}")
    return " | ".join(parts)


def _read_text_tolerant(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""
