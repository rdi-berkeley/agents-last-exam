"""ClaudeCodeDeployer — runs the ``@anthropic-ai/claude-code`` CLI on Linux or Windows VM.

``install()`` is verify-or-install: if Node / claude / MCP server are baked
into the image, it skips; otherwise it runs the shared
:mod:`ale.agents.runtime_install` helpers to stage them.

``launch()`` dispatches by ``session.os_type``:

- Linux  → bash runner + setsid launcher, PID file + done.marker
- Windows → PowerShell runner + Start-Process launcher, PID file + done.marker

``collect()`` parses stream-json transcript into ATIF Trajectory (same on
both OSes — stream-json is portable).
"""
from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from typing import TYPE_CHECKING, Any

from ale.agents.base import (
    AgentRunResult,
    BaseAgentDeployer,
)
from ale.agents.cli_flags import (
    CliFlag,
    EnvVar,
    build_cli_args,
    build_env,
    render_env_lines,
)
from ale.agents.runtime_install import (
    ensure_node,
    npm_install_global,
    upload_mcp_server,
)
from ale.core.cmd_result import cmd_stdout
from ale.agents.trajectory import (
    ContentPart,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
    Observation as TrajObservation,
)

from .config import ClaudeCodeConfig

if TYPE_CHECKING:
    import cua_bench as cb

logger = logging.getLogger(__name__)


# =============================================================================
# Per-VM path layout
# =============================================================================

class _VMPaths:
    """Resolved VM paths for one run. Built from :class:`InstallPaths` + OS."""

    def __init__(self, install_paths, os_type: str):
        self.os = os_type
        self.node_exe = install_paths.node_exe(os_type)
        self.mcp_server_dir = install_paths.mcp_server_dir(os_type)
        self.work_dir = install_paths.work_dir(os_type, "claude-code")
        self.sep = "\\" if os_type == "windows" else "/"
        # claude shim location differs by OS install convention
        if os_type == "windows":
            self.claude_cmd = rf"{install_paths.agent_bin_dir('windows')}\claude.cmd"
            self.script_ext = ".ps1"
        else:
            self.claude_cmd = install_paths.cli_path("linux", "claude")
            self.script_ext = ".sh"
        s = self.sep
        self.mcp_config = f"{self.work_dir}{s}mcp_config.json"
        self.prompt_file = f"{self.work_dir}{s}prompt.txt"
        self.runner_script = f"{self.work_dir}{s}run_claude{self.script_ext}"
        self.launcher_script = f"{self.work_dir}{s}launch{self.script_ext}"
        self.transcript_file = f"{self.work_dir}{s}transcript.jsonl"
        self.stderr_log = f"{self.work_dir}{s}stderr.log"
        self.pid_file = f"{self.work_dir}{s}claude.pid"
        self.done_marker = f"{self.work_dir}{s}done.marker"


# =============================================================================
# CLI / env descriptors
# =============================================================================

CLI_FLAGS: list[CliFlag] = [
    CliFlag("model", "--model"),
    CliFlag("max_turns", "--max-turns",
            when=lambda v: v is not None and v >= 0),
    CliFlag("max_budget_usd", "--max-budget-usd",
            when=lambda v: v is not None),
    CliFlag("dangerously_skip_permissions", "--dangerously-skip-permissions",
            kind="bool_flag"),
    CliFlag("disabled_tools", "--disallowedTools",
            kind="multi_value", when=lambda v: bool(v)),
]

ENV_VARS: list[EnvVar] = [
    EnvVar("anthropic_api_key", "ANTHROPIC_API_KEY",
           when=lambda c: not c.is_openrouter and bool(c.anthropic_api_key)),
    EnvVar("openrouter_api_key", "ANTHROPIC_AUTH_TOKEN",
           when=lambda c: c.is_openrouter),
    EnvVar("resolved_base_url", "ANTHROPIC_BASE_URL",
           when=lambda c: c.is_openrouter),
]


# Pinned in config; this is the npm package coordinate to install.
CLAUDE_NPM_PACKAGE = "@anthropic-ai/claude-code@2.1.85"


# =============================================================================
# Deployer
# =============================================================================

class ClaudeCodeDeployer(BaseAgentDeployer):
    """Three-phase deployer for the Claude Code CLI."""

    def __init__(self, config: ClaudeCodeConfig):
        self._cfg = config

    @property
    def config(self) -> ClaudeCodeConfig:
        return self._cfg

    @property
    def version(self) -> str | None:
        return self._cfg.cli_version

    # ------------------------------------------------------------------ install

    async def install(self, session: "cb.DesktopSession") -> None:
        """Verify-or-install Node / claude CLI / MCP server, then write MCP config."""
        paths = self._paths(session)

        # 1. Node (Linux verifies, Windows downloads).
        await ensure_node(session, self._cfg.install_paths)

        # 2. Claude CLI.
        if not await session.exists(paths.claude_cmd):
            logger.info("claude CLI missing at %s; installing %s",
                        paths.claude_cmd, CLAUDE_NPM_PACKAGE)
            await npm_install_global(
                session, CLAUDE_NPM_PACKAGE, self._cfg.install_paths,
            )
            if not await session.exists(paths.claude_cmd):
                raise RuntimeError(
                    f"claude CLI still missing after npm install at {paths.claude_cmd}"
                )
        else:
            logger.info("claude CLI present at %s", paths.claude_cmd)

        # 3. MCP server.
        mcp_entry = paths.mcp_server_dir + paths.sep + "src" + paths.sep + "index.js"
        if not await session.exists(mcp_entry):
            logger.info("cua-mcp-server missing; uploading + npm install")
            await upload_mcp_server(session, self._cfg.install_paths)
        else:
            logger.info("cua-mcp-server present at %s", paths.mcp_server_dir)

        # 4. Per-run work dir + MCP config.
        await self._ensure_dir(session, paths.work_dir, paths.os)
        await self._write_mcp_config(session, paths)
        logger.info("claude_code: install done (work_dir=%s)", paths.work_dir)

    async def _write_mcp_config(self, session, paths: _VMPaths) -> None:
        mcp_config = {
            "mcpServers": {
                "cua": {
                    "command": paths.node_exe,
                    "args": [paths.mcp_server_dir + paths.sep + "src" + paths.sep + "index.js"],
                },
            },
        }
        await session.write_file(paths.mcp_config, json.dumps(mcp_config, indent=2))

    async def _ensure_dir(self, session, path: str, os_type: str) -> None:
        if os_type == "windows":
            await session.run_command(
                f'powershell -NoProfile -Command "'
                f"New-Item -ItemType Directory -Force -Path '{path}' | Out-Null"
                f'"', timeout=60,
            )
        else:
            await session.makedirs(path)

    # ------------------------------------------------------------------- launch

    async def launch(
        self,
        session: "cb.DesktopSession",
        *,
        prompt: str,
        timeout_s: float,
    ) -> AgentRunResult:
        paths = self._paths(session)
        await session.write_file(paths.prompt_file, prompt)

        if paths.os == "linux":
            await self._launch_linux(session, paths)
        else:
            await self._launch_windows(session, paths)

        t0 = time.monotonic()
        pid = await self._read_pid(session, paths)
        if pid is None:
            return AgentRunResult(
                status="failed",
                error="launcher did not write a PID — see stderr.log on the VM",
                stderr_path=paths.stderr_log,
                duration_s=time.monotonic() - t0,
            )

        status, exit_code = await self._poll_until_exit(session, paths, pid, timeout_s)
        error: str | None = None
        if status != "completed":
            error = await self._diagnose_failure(session, paths, status, exit_code)
        return AgentRunResult(
            status=status,
            pid=pid,
            exit_code=exit_code,
            transcript_path=paths.transcript_file,
            stderr_path=paths.stderr_log,
            duration_s=time.monotonic() - t0,
            error=error,
        )

    # ---- Linux launch ----

    async def _launch_linux(self, session, paths: _VMPaths) -> None:
        runner = self._build_runner_script_linux(paths)
        launcher = self._build_launcher_script_linux(paths)
        await session.write_file(paths.runner_script, runner)
        await session.write_file(paths.launcher_script, launcher)
        await session.run_command(
            f"chmod +x {shlex.quote(paths.runner_script)} "
            f"{shlex.quote(paths.launcher_script)}"
        )
        await session.run_command(f"bash {shlex.quote(paths.launcher_script)}")

    def _build_runner_script_linux(self, paths: _VMPaths) -> str:
        env_lines = render_env_lines(build_env(self._cfg, ENV_VARS))
        argv_extra = build_cli_args(self._cfg, CLI_FLAGS)
        flags_str = " \\\n  ".join(shlex.quote(a) for a in argv_extra)
        return (
            "#!/bin/bash\n"
            "set -u\n"
            f"{env_lines}"
            f"cd {shlex.quote(paths.work_dir)}\n"
            f"prompt=$(cat {shlex.quote(paths.prompt_file)})\n"
            f"echo \"$prompt\" | {shlex.quote(paths.claude_cmd)} -p - \\\n"
            f"  --output-format stream-json --verbose \\\n"
            f"  --mcp-config {shlex.quote(paths.mcp_config)} \\\n"
            f"  {flags_str} \\\n"
            f"  2>{shlex.quote(paths.stderr_log)} >{shlex.quote(paths.transcript_file)}\n"
            f"echo $? > {shlex.quote(paths.done_marker)}\n"
        )

    def _build_launcher_script_linux(self, paths: _VMPaths) -> str:
        return (
            "#!/bin/bash\n"
            f"rm -f {shlex.quote(paths.done_marker)} {shlex.quote(paths.pid_file)}\n"
            f"setsid bash {shlex.quote(paths.runner_script)} </dev/null >/dev/null 2>&1 &\n"
            "CHILD=$!\n"
            f"echo \"$CHILD\" > {shlex.quote(paths.pid_file)}\n"
            "disown $CHILD 2>/dev/null || true\n"
        )

    # ---- Windows launch ----

    async def _launch_windows(self, session, paths: _VMPaths) -> None:
        runner = self._build_runner_script_windows(paths)
        await session.write_file(paths.runner_script, runner)
        # Launcher is inline: Start-Process the runner, capture PID.
        spawn_ps = (
            f"$proc = Start-Process powershell "
            f"-ArgumentList '-ExecutionPolicy','Bypass','-File','{paths.runner_script}' "
            f"-WindowStyle Hidden -PassThru; "
            f"$proc.Id | Out-File -FilePath '{paths.pid_file}' -Encoding ascii -NoNewline"
        )
        await session.run_command(
            f'powershell -NoProfile -Command "{spawn_ps}"', timeout=30,
        )

    def _build_runner_script_windows(self, paths: _VMPaths) -> str:
        # Build PowerShell env-setting + flags. Env values single-quoted (PS literal).
        env_kv = build_env(self._cfg, ENV_VARS)
        env_block = "\n".join(
            f"$env:{k} = '{v.replace(chr(39), chr(39)*2)}'"
            for k, v in env_kv.items()
        )
        argv_extra = build_cli_args(self._cfg, CLI_FLAGS)
        # Reassemble argv into a flat string with PowerShell quoting.
        flag_string = " ".join(
            f"'{a.replace(chr(39), chr(39)*2)}'" if any(c in a for c in " '\"") else a
            for a in argv_extra
        )
        npm_bin = self._cfg.install_paths.agent_bin_dir("windows")
        node_dir = self._cfg.install_paths.node_exe("windows").rsplit("\\", 1)[0]

        return (
            "$ErrorActionPreference = 'Continue'\n"
            "$utf8 = New-Object System.Text.UTF8Encoding($false)\n"
            "[Console]::InputEncoding = $utf8\n"
            "[Console]::OutputEncoding = $utf8\n"
            "$OutputEncoding = $utf8\n"
            f"{env_block}\n"
            f'$env:PATH = "$env:PATH;{npm_bin};{node_dir}"\n'
            f"$prompt = Get-Content '{paths.prompt_file}' -Encoding UTF8 -Raw\n"
            f"$prompt | & '{paths.claude_cmd}' -p - "
            f"--output-format stream-json --verbose "
            f"--mcp-config '{paths.mcp_config}' {flag_string} "
            f"2>'{paths.stderr_log}' | "
            f"Out-File -FilePath '{paths.transcript_file}' -Encoding utf8\n"
            f"$LASTEXITCODE | Out-File -FilePath '{paths.done_marker}' "
            f"-Encoding ascii -NoNewline\n"
        )

    # ---- PID + polling ----

    async def _read_pid(self, session, paths: _VMPaths) -> int | None:
        for _ in range(15):
            if await session.exists(paths.pid_file):
                try:
                    raw = (await session.read_file(paths.pid_file)).strip()
                    return int(raw) if raw else None
                except (ValueError, FileNotFoundError):
                    pass
            await asyncio.sleep(0.3)
        return None

    async def _poll_until_exit(
        self,
        session,
        paths: _VMPaths,
        pid: int,
        timeout_s: float,
        *,
        poll_interval_s: float = 10.0,
    ) -> tuple[str, int | None]:
        """Returns ``(status, exit_code)``.

        Status semantics:
            ``completed`` — done.marker present AND exit_code == 0
            ``failed``    — done.marker present AND exit_code != 0,
                            OR process disappeared without writing done.marker
            ``timeout``   — wall budget exhausted before any of the above
        """
        deadline = time.monotonic() + timeout_s
        while True:
            # 1. Done marker → process finished, classify by exit_code.
            if await session.exists(paths.done_marker):
                try:
                    raw = (await session.read_file(paths.done_marker)).strip()
                    exit_code = int(raw) if raw else None
                except (ValueError, FileNotFoundError):
                    exit_code = None
                status = "completed" if exit_code == 0 else "failed"
                return status, exit_code

            # 2. Process still alive?
            alive = await self._is_pid_alive(session, paths, pid)
            if not alive:
                return "failed", None

            # 3. Timeout.
            if time.monotonic() >= deadline:
                await self._kill_pid(session, paths, pid)
                return "timeout", None

            await asyncio.sleep(poll_interval_s)

    async def _diagnose_failure(
        self,
        session,
        paths: _VMPaths,
        status: str,
        exit_code: int | None,
    ) -> str:
        """Build a short error string from on-VM logs. Best-effort, never raises."""
        parts: list[str] = [f"agent {status} (rc={exit_code})"]

        # Tail stderr.log (most useful for crash / startup failure).
        try:
            stderr = await session.read_file(paths.stderr_log)
            tail = (stderr or "").strip()
            if tail:
                parts.append(f"stderr tail: ...{tail[-500:]}")
        except Exception:                                # noqa: BLE001
            pass

        # Scan transcript for well-known root-cause patterns.
        try:
            transcript = await session.read_file(paths.transcript_file)
        except Exception:                                # noqa: BLE001
            transcript = ""
        if '"authentication_failed"' in transcript or '"User not found"' in transcript:
            parts.append(
                "LLM auth failed (check ANTHROPIC_API_KEY / OPENROUTER_API_KEY)"
            )
        elif '"error_status":429' in transcript or '"rate_limit_error"' in transcript:
            parts.append("LLM rate-limited")
        elif '"error_status":5' in transcript:           # 5xx
            parts.append("LLM upstream 5xx")
        elif '"type":"result"' not in transcript and exit_code != 0:
            parts.append("agent never produced a result event; check stderr")

        return " | ".join(parts)

    async def _is_pid_alive(self, session, paths: _VMPaths, pid: int) -> bool:
        if paths.os == "linux":
            cr = await session.run_command(
                f"kill -0 {pid} 2>/dev/null && echo alive || echo dead"
            )
            stdout = cmd_stdout(cr).strip()
            return "alive" in stdout
        # Windows: Get-Process returns nothing if dead.
        cr = await session.run_command(
            f'powershell -NoProfile -Command "'
            f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ Write-Output alive }} "
            f"else {{ Write-Output dead }}"
            f'"', timeout=30,
        )
        stdout = cmd_stdout(cr).strip()
        return "alive" in stdout

    async def _kill_pid(self, session, paths: _VMPaths, pid: int) -> None:
        if paths.os == "linux":
            await session.run_command(f"kill -TERM {pid} 2>/dev/null || true")
            await asyncio.sleep(2.0)
            await session.run_command(f"kill -KILL {pid} 2>/dev/null || true")
        else:
            await session.run_command(
                f'powershell -NoProfile -Command "'
                f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"
                f'"', timeout=30,
            )

    # ------------------------------------------------------------------ collect

    async def collect(
        self,
        session: "cb.DesktopSession",
        run: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        paths = self._paths(session)
        try:
            raw = await session.read_file(paths.transcript_file)
        except FileNotFoundError:
            logger.warning("claude_code: no transcript at %s", paths.transcript_file)
            builder.add_step(
                source="system",
                message="claude-code transcript missing",
                extra={"reason": "no_transcript", "expected_path": paths.transcript_file},
            )
            return

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._consume_event(event, builder)

        builder.trajectory.extra.update({
            "claude_code": {
                "exit_code": run.exit_code,
                "transcript_path": run.transcript_path,
                "stderr_path": run.stderr_path,
            },
        })

    def _consume_event(self, event: dict[str, Any], builder: TrajectoryBuilder) -> None:
        etype = event.get("type")
        if etype == "system":
            builder.trajectory.extra.setdefault("system_events", []).append(event)
            return
        if etype == "assistant":
            self._consume_assistant(event, builder)
            return
        if etype == "user":
            self._consume_user(event, builder)
            return
        if etype == "result":
            builder.trajectory.extra["result"] = event
            return
        builder.trajectory.extra.setdefault("unknown_events", []).append(event)

    def _consume_assistant(self, event: dict[str, Any], builder: TrajectoryBuilder) -> None:
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

    def _consume_user(self, event: dict[str, Any], builder: TrajectoryBuilder) -> None:
        message = event.get("message", {}) or {}
        content_blocks = message.get("content", []) or []
        tool_results: list[ToolResult] = []
        text_parts: list[str] = []
        for block in content_blocks:
            btype = block.get("type")
            if btype == "tool_result":
                tr_content = block.get("content")
                parts: list[ContentPart] = []
                if isinstance(tr_content, str):
                    parts.append(ContentPart(type="text", text=tr_content))
                elif isinstance(tr_content, list):
                    for c in tr_content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            parts.append(ContentPart(type="text", text=c.get("text", "")))
                tool_results.append(ToolResult(
                    tool_call_id=block.get("tool_use_id") or "",
                    content=parts,
                    is_error=bool(block.get("is_error")),
                ))
            elif btype == "text":
                text_parts.append(block.get("text", ""))
        builder.add_step(
            source="environment",
            message="\n".join(p for p in text_parts if p) or None,
            observation=TrajObservation(results=tool_results),
        )

    # ------------------------------------------------------------------ helpers

    def _paths(self, session: "cb.DesktopSession") -> _VMPaths:
        os_type = getattr(session, "os_type", "linux") or "linux"
        return _VMPaths(self._cfg.install_paths, os_type)

    def work_dir(self, session: "cb.DesktopSession") -> str | None:
        """Directory on the VM where this deployer puts everything (mirror source)."""
        return self._paths(session).work_dir
