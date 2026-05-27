"""CursorCliDeployer — drives the ``cursor-agent`` CLI from Anysphere.

Closed-source binary, installed via ``curl -fsS https://cursor.com/install | bash``.
Bundles its own Node runtime — no separate Node install needed for the
CLI itself.  Node IS still required for the CUA MCP Server bridge.

Auth: cursor-agent reads ``~/.config/cursor/auth.json`` (on Linux),
which contains ``accessToken``, ``refreshToken``, and ``apiKey`` fields
from an OAuth login flow.  The ``CURSOR_API_KEY`` env var shown in the
stream-json ``apiKeySource: "env"`` is misleading — the actual runtime
auth comes from the auth.json file.  This deployer receives the
auth.json content via the ``CURSOR_AUTH_JSON`` env var (set by lifecycle
env passthrough) or reads it from the path in ``CURSOR_AUTH_JSON_PATH``.
OpenRouter / BYOK is NOT supported (cursor-agent rejects non-Cursor keys).

MCP auto-discovered from ``~/.cursor/mcp.json`` (no CLI flag).
Headless mode via three-layer permission system: CLI flags +
``cli-config.json`` + deny list.
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

from .config import CursorCliConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0


class CursorCliDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the ``cursor-agent`` CLI."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    # =========================================================================
    # install
    # =========================================================================

    async def _auto_install_cli(self) -> None:
        proc = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", "curl -fsS https://cursor.com/install | bash"],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"cursor-agent install failed (rc={proc.returncode}): "
                f"{(proc.stderr or '')[:500]}"
            )
        home = os.path.expanduser("~")
        bin_dir = f"{home}/.local/bin"
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        logger.info("cursor_cli: installed — %s", (proc.stdout or "").strip()[-200:])

    async def install(self) -> None:
        cfg: CursorCliConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        cursor_path = shutil.which("cursor-agent")
        if not cursor_path:
            logger.info("cursor_cli: 'cursor-agent' not on PATH, installing …")
            await self._auto_install_cli()
            cursor_path = shutil.which("cursor-agent")
            if not cursor_path:
                raise RuntimeError(
                    "CursorCliDeployer: 'cursor-agent' still not found after install"
                )
        self._cursor_path = cursor_path

        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [cursor_path, "--version"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"cursor-agent --version timed out: {e}")
        logger.info("cursor_cli: CLI ok — %s", (probe.stdout or "").strip())

        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        home = os.path.expanduser("~")
        cursor_home = Path(home) / ".cursor"
        cursor_home.mkdir(parents=True, exist_ok=True)

        # MCP config (auto-discovered, no CLI flag)
        mcp_config = {
            "mcpServers": {
                "cua": {
                    "command": sandbox.node,
                    "args": [self._join(sandbox.mcp_server_dir, "src", "index.js",
                                        is_linux=sandbox.is_linux)],
                },
            },
        }
        (cursor_home / "mcp.json").write_text(
            json.dumps(mcp_config, indent=2), encoding="utf-8",
        )

        # Headless config
        cli_config = {
            "version": 1,
            "permissions": {
                "allow": [
                    "Shell(*)",
                    "Read(**)",
                    "Write(**)",
                    "WebFetch(*)",
                    "Mcp(*:*)",
                ],
                "deny": list(cfg.disabled_tools),
            },
            "approvalMode": "unrestricted",
            "sandbox": {
                "mode": "disabled",
                "networkAccess": "user_config_with_defaults",
            },
        }
        (cursor_home / "cli-config.json").write_text(
            json.dumps(cli_config, indent=2), encoding="utf-8",
        )

        # Clean any project-scoped override that could conflict
        project_cli = wd / ".cursor" / "cli.json"
        if project_cli.exists():
            project_cli.unlink()

        # ---- auth.json passthrough ----
        # cursor-agent authenticates via ~/.config/cursor/auth.json, NOT via
        # the CURSOR_API_KEY env var.  The auth.json content arrives here
        # either as CURSOR_AUTH_JSON (inline JSON string set by lifecycle env
        # passthrough) or via CURSOR_AUTH_JSON_PATH (file path).
        self._setup_cursor_auth()

        logger.info("cursor_cli: config staged at %s", cursor_home)

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: CursorCliConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        prompt_file = wd / "prompt.txt"
        transcript_file = wd / "transcript.jsonl"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "cursor.pid"

        for f in (transcript_file, stderr_log, pid_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        prompt_file.write_text(prompt, encoding="utf-8")

        argv = [
            self._cursor_path,
            "-p",
            "--model", cfg.model,
            "--output-format", "stream-json",
            "--force", "--approve-mcps", "--trust",
            "--sandbox", "disabled",
        ]
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
        logger.info("cursor_cli: spawned pid=%s", proc.pid)

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
    # auth.json passthrough
    # =========================================================================

    def _setup_cursor_auth(self) -> None:
        """Write ``~/.config/cursor/auth.json`` from env vars.

        Resolution order:
        1. ``CURSOR_AUTH_JSON`` env var — raw JSON content (set by lifecycle).
        2. ``CURSOR_AUTH_JSON_PATH`` env var — path to a file inside the
           container.
        3. If neither is available, log a warning.

        Security: never log the auth.json content; only log byte count.
        """
        home = os.path.expanduser("~")
        auth_dir = Path(home) / ".config" / "cursor"
        auth_file = auth_dir / "auth.json"

        # Source 1: inline JSON from CURSOR_AUTH_JSON env var
        auth_content = os.environ.get("CURSOR_AUTH_JSON", "").strip()
        if auth_content:
            auth_dir.mkdir(parents=True, exist_ok=True)
            auth_file.write_text(auth_content, encoding="utf-8")
            auth_file.chmod(0o600)
            logger.info(
                "cursor_cli: wrote auth.json from CURSOR_AUTH_JSON (%d B)",
                len(auth_content),
            )
            return

        # Source 2: file path from CURSOR_AUTH_JSON_PATH env var
        auth_path_str = os.environ.get("CURSOR_AUTH_JSON_PATH", "").strip()
        if auth_path_str:
            auth_path = Path(auth_path_str)
            if auth_path.is_file():
                content = auth_path.read_text(encoding="utf-8")
                auth_dir.mkdir(parents=True, exist_ok=True)
                auth_file.write_text(content, encoding="utf-8")
                auth_file.chmod(0o600)
                logger.info(
                    "cursor_cli: wrote auth.json from CURSOR_AUTH_JSON_PATH=%s (%d B)",
                    auth_path_str, len(content),
                )
                return
            logger.warning(
                "cursor_cli: CURSOR_AUTH_JSON_PATH=%s does not exist inside container",
                auth_path_str,
            )

        # Neither source available
        logger.warning(
            "cursor_cli: no auth.json source found (checked CURSOR_AUTH_JSON, "
            "CURSOR_AUTH_JSON_PATH) — cursor-agent will likely fail to authenticate"
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

    def _build_env(self, cfg: CursorCliConfig) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env["NO_COLOR"] = "1"
        return env

    # =========================================================================
    # parse_artifacts — same stream-json format as claude_code
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
                message=f"cursor-cli: no transcript at {transcript_file}",
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

        builder.trajectory.extra.setdefault("cursor_cli", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(transcript_file),
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
            cls._consume_result(event, builder)

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
        # camelCase usage fields
        usage = event.get("usage") or message.get("usage") or {}
        metrics = StepMetrics(
            input_tokens=usage.get("inputTokens") or usage.get("input_tokens"),
            output_tokens=usage.get("outputTokens") or usage.get("output_tokens"),
            cache_read_tokens=usage.get("cacheReadTokens") or usage.get("cache_read_input_tokens"),
            cache_creation_tokens=usage.get("cacheWriteTokens") or usage.get("cache_creation_input_tokens"),
        )
        builder.add_step(
            source="agent",
            message="\n".join(p for p in text_parts if p) or None,
            tool_calls=tool_calls,
            metrics=metrics,
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

    @staticmethod
    def _consume_result(event: dict, builder: TrajectoryBuilder) -> None:
        usage = event.get("usage", {})
        builder.trajectory.extra.setdefault("cursor_cli", {})["result"] = event
        if usage:
            builder.trajectory.extra["cursor_cli"]["usage"] = usage


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
