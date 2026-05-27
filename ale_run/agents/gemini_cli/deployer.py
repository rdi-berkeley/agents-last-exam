"""GeminiCliDeployer — drives the Google ``gemini`` CLI.

Install via npm, configure MCP for CUA bridge, launch in yolo mode,
parse stream-json NDJSON output into trajectory steps.
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

from .config import GeminiCliConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0

_YOLO_POLICY = """\
# AgentHLE headless yolo policy — allow all tools, deny ask_user.
[[rule]]
toolName = "*"
decision = "allow"
priority = 998
modes = ["yolo"]
allowRedirection = true

[[rule]]
toolName = "ask_user"
decision = "deny"
priority = 999
modes = ["yolo"]
"""


class GeminiCliDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the Google ``gemini`` CLI."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    @property
    def version(self) -> str | None:
        cfg: GeminiCliConfig = self.config  # type: ignore[assignment]
        return cfg.npm_package

    # =========================================================================
    # install
    # =========================================================================

    async def _auto_install_cli(self, package: str) -> None:
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError(
                "GeminiCliDeployer: 'gemini' not found and 'npm' not on PATH"
            )
        home = os.path.expanduser("~")
        env = {**os.environ, "npm_config_cache": f"{home}/.npm-ale"}
        proc = await asyncio.to_thread(
            subprocess.run,
            [npm, "install", "-g", "--prefix", f"{home}/.local", package],
            capture_output=True, text=True, timeout=300, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"npm install -g {package} failed "
                f"(rc={proc.returncode}): {(proc.stderr or '')[:500]}"
            )
        bin_dir = f"{home}/.local/bin"
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        logger.info("gemini_cli: installed via npm — %s", (proc.stdout or "").strip()[-200:])

    async def install(self) -> None:
        cfg: GeminiCliConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        gemini_path = shutil.which("gemini")
        if not gemini_path:
            logger.info("gemini_cli: 'gemini' not on PATH, installing via npm …")
            await self._auto_install_cli(cfg.npm_package)
            gemini_path = shutil.which("gemini")
            if not gemini_path:
                raise RuntimeError(
                    "GeminiCliDeployer: 'gemini' still not found after npm install"
                )
        self._gemini_path = gemini_path

        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [gemini_path, "--version"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"gemini --version timed out: {e}")
        logger.info("gemini_cli: CLI ok — %s", (probe.stdout or "").strip())

        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        home = os.path.expanduser("~")
        gemini_home = Path(home) / ".gemini"
        gemini_home.mkdir(parents=True, exist_ok=True)

        # MCP + settings
        settings = {
            "mcpServers": {
                "cua": {
                    "command": sandbox.node,
                    "args": [self._join(sandbox.mcp_server_dir, "src", "index.js",
                                        is_linux=sandbox.is_linux)],
                },
            },
            "tools": {
                "exclude": list(cfg.disabled_tools),
            },
            "policyPaths": [str(gemini_home / "agenthle_policy.toml")],
        }
        (gemini_home / "settings.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8",
        )

        # Yolo policy
        (gemini_home / "agenthle_policy.toml").write_text(
            _YOLO_POLICY, encoding="utf-8",
        )
        logger.info("gemini_cli: config staged at %s", gemini_home)

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: GeminiCliConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        prompt_file = wd / "prompt.txt"
        transcript_file = wd / "transcript.jsonl"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "gemini.pid"

        for f in (transcript_file, stderr_log, pid_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        prompt_file.write_text(prompt, encoding="utf-8")

        argv = self._build_argv(cfg)
        env = self._build_env(cfg)

        t0 = time.monotonic()
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
                start_new_session=True if hasattr(os, "setsid") else False,
            )
        pid_file.write_text(str(proc.pid), encoding="ascii")
        logger.info("gemini_cli: spawned pid=%s", proc.pid)

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
        sep = "/" if is_linux else "\\"
        head = parts[0].rstrip("/\\")
        tail = sep.join(p.strip("/\\") for p in parts[1:])
        return f"{head}{sep}{tail}" if tail else head

    def _build_argv(self, cfg: GeminiCliConfig) -> list[str]:
        argv = [
            self._gemini_path,
            "-p", "-",
            "--model", cfg.model,
            "--output-format", "stream-json",
            "--approval-mode", cfg.approval_mode,
        ]
        if cfg.allowed_tools:
            argv.append(f"--allowed-tools={','.join(cfg.allowed_tools)}")
        return argv

    def _build_env(self, cfg: GeminiCliConfig) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env["NO_COLOR"] = "1"
        env["NO_BROWSER"] = "1"
        return env

    def _diagnose_failure(
        self, *, stderr_log: Path, transcript: Path, exit_code: int | None,
    ) -> str:
        parts = [f"agent failed (rc={exit_code})"]
        stderr_text = _read_text_tolerant(stderr_log)
        tx_text = _read_text_tolerant(transcript)
        if stderr_text.strip():
            parts.append(f"stderr tail: ...{stderr_text[-800:]}")
        if tx_text.strip():
            parts.append(f"transcript tail: ...{tx_text[-800:]}")
        return " | ".join(parts)

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
                message=f"gemini-cli: no transcript at {transcript_file}",
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

        builder.trajectory.extra.setdefault("gemini_cli", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(transcript_file),
        })

    @classmethod
    def _consume_event(cls, event: dict, builder: TrajectoryBuilder) -> None:
        etype = event.get("type")
        if etype == "message":
            role = event.get("role")
            if event.get("delta"):
                return
            if role == "assistant":
                cls._consume_assistant_message(event, builder)
            elif role == "user":
                cls._consume_user_message(event, builder)
        elif etype == "tool_use":
            cls._consume_tool_use(event, builder)
        elif etype == "tool_result":
            cls._consume_tool_result(event, builder)
        elif etype == "result":
            cls._consume_result(event, builder)
        elif etype == "error":
            builder.add_step(
                source="system",
                message=event.get("message", "unknown error"),
                extra={"severity": event.get("severity")},
            )

    @staticmethod
    def _consume_assistant_message(event: dict, builder: TrajectoryBuilder) -> None:
        msg_type = event.get("messageType", "text")
        text = event.get("content", "")
        if msg_type == "thinking":
            builder.add_step(source="agent", reasoning=text)
        else:
            builder.add_step(source="agent", message=text)

    @staticmethod
    def _consume_user_message(event: dict, builder: TrajectoryBuilder) -> None:
        builder.add_step(source="user", message=event.get("content", ""))

    @staticmethod
    def _consume_tool_use(event: dict, builder: TrajectoryBuilder) -> None:
        params = event.get("parameters", {})
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {"raw": params}
        builder.add_step(
            source="agent",
            tool_calls=[ToolCall(
                id=event.get("tool_id", ""),
                name=event.get("tool_name", ""),
                arguments=params,
            )],
        )

    @staticmethod
    def _consume_tool_result(event: dict, builder: TrajectoryBuilder) -> None:
        output = event.get("output", "")
        error = event.get("error")
        text = error if error else (output if isinstance(output, str) else json.dumps(output))
        builder.add_step(
            source="environment",
            observation=Observation(results=[
                ToolResult(
                    tool_call_id=event.get("tool_id", ""),
                    content=[ContentPart(type="text", text=text)],
                    is_error=bool(error),
                ),
            ]),
        )

    @staticmethod
    def _consume_result(event: dict, builder: TrajectoryBuilder) -> None:
        stats = event.get("stats", {})
        builder.trajectory.extra.setdefault("gemini_cli", {})["result"] = event
        response = event.get("response", "")
        if response:
            builder.add_step(source="agent", message=response)


def _read_text_tolerant(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""
