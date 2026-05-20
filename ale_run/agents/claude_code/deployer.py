"""ClaudeCodeDeployer — drives @anthropic-ai/claude-code CLI on a remote VM.

The deployer runs on the **framework host**. All VM I/O goes through the
HTTP primitives in :mod:`ale_run.environments.remote` (``run_remote``,
``upload_file``, ``download_file``): stateless POSTs with per-call
timeouts. We deliberately avoid the cua-bench ``session`` here — its
websocket transport can wedge mid-run (long agent loops, idle timeouts,
firewall NAT churn), and recovering a dropped WS is more work than
re-establishing per-request HTTP/SSE.

Image assumptions:

* Linux (e.g. ``agenthle-ubuntu-0505``):

  - bash present, ``setsid`` available
  - ``/usr/local/bin/claude`` baked in
  - ``/usr/local/bin/node`` + ``/home/user/cua_mcp_server`` (for MCP)

* Windows (e.g. ``agenthle-unified-v1`` / ``agenthle-dev-cpu-free-0505``):

  - PowerShell present
  - ``C:\\Users\\User\\.local\\bin\\claude.exe`` baked in
  - ``C:\\Users\\User\\node-v24.12.0-win-x64\\node.exe`` and
    ``C:\\Users\\User\\cua_mcp_server`` (for MCP)

If the binary is missing, ``install()`` raises clearly — rebuild the image
or repoint :meth:`VmRuntime.cli_path` rather than installing at runtime.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shlex
import time
from pathlib import Path
from typing import ClassVar

import requests

from ale_run.agents.base import (
    AgentRunResult,
    BaseAgentConfig,
    BaseAgentDeployer,
)
from ale_run.agents.trajectory import (
    ContentPart,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)
from ale_run.environments.remote import (
    RemoteVMConfig,
    _read_first_sse_event,
    download_file,
    run_remote,
    upload_file,
)

from .config import ClaudeCodeConfig

logger = logging.getLogger(__name__)


def _http_file_exists(vm_config: RemoteVMConfig, path: str, timeout: float = 15) -> bool:
    """Sync HTTP-only existence check; mirrors session.exists without WS."""
    try:
        with requests.post(
            f"{vm_config.server_url.rstrip('/')}/cmd",
            json={"command": "file_exists", "params": {"path": path}},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            stream=True,
        ) as resp:
            data = _read_first_sse_event(resp, read_timeout=timeout)
    except requests.RequestException:
        return False
    if not data:
        return False
    return bool(data.get("exists"))


def _http_read_text(vm_config: RemoteVMConfig, path: str, timeout: float = 30) -> str:
    """Sync HTTP read of a small text file. Empty string on any error."""
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(prefix="ale_dl_")
    os.close(fd)
    try:
        ok = download_file(vm_config, path, tmp, timeout)
        if not ok:
            return ""
        with open(tmp, "rb") as f:
            return f.read().decode("utf-8", errors="replace")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


class ClaudeCodeDeployer(BaseAgentDeployer):
    """Host-side deployer for the @anthropic-ai/claude-code CLI."""

    supported_runtimes: ClassVar[frozenset[str]] = frozenset({"vm"})

    # transcript.jsonl + stderr.log are written by the claude CLI as the
    # agent runs; the framework's IncrementalPuller tails them every
    # ~15s so a SIGTERM mid-agent doesn't lose the transcript.
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    @property
    def version(self) -> str | None:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        return cfg.cli_version

    # ------------------------------------------------------------------ helpers

    def _vm_config(self) -> RemoteVMConfig:
        rt = self.runtime
        return RemoteVMConfig(server_url=rt.vm_endpoint, os_type=rt.vm_os)

    @staticmethod
    def _join(vm_os: str, *parts: str) -> str:
        sep = "/" if vm_os == "linux" else "\\"
        head = parts[0].rstrip("/\\")
        tail = sep.join(p.strip("/\\") for p in parts[1:])
        return f"{head}{sep}{tail}" if tail else head

    @staticmethod
    def _run(vm_config: RemoteVMConfig, command: str, timeout: float = 60):
        return run_remote(vm_config, command, timeout=timeout)

    @classmethod
    def _mkdir(cls, vm_config: RemoteVMConfig, path: str) -> None:
        if vm_config.is_linux:
            cmd = f"mkdir -p {shlex.quote(path)}"
        else:
            cmd = (
                'powershell -NoProfile -Command "'
                f"New-Item -ItemType Directory -Force -Path '{path}' | Out-Null"
                '"'
            )
        cls._run(vm_config, cmd, timeout=30)

    @classmethod
    def _rm(cls, vm_config: RemoteVMConfig, paths: list[str]) -> None:
        if not paths:
            return
        if vm_config.is_linux:
            quoted = " ".join(shlex.quote(p) for p in paths)
            cls._run(vm_config, f"rm -f {quoted}", timeout=30)
        else:
            inner = "; ".join(
                f"Remove-Item -Force -ErrorAction SilentlyContinue '{p}'" for p in paths
            )
            cls._run(vm_config, f'powershell -NoProfile -Command "{inner}"', timeout=30)

    # =========================================================================
    # install — verify claude on VM, prepare work_dir + mcp_config
    # =========================================================================

    async def install(self) -> None:
        await asyncio.to_thread(self._install_sync)

    def _install_sync(self) -> None:
        runtime = self.runtime
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        vm_config = self._vm_config()
        wd = runtime.work_dir_vm
        claude_cmd = runtime.cli_path("claude")

        # 1. Probe claude binary (image-baked expectation).
        if vm_config.is_linux:
            probe = (
                f"test -x {shlex.quote(claude_cmd)} && "
                f"{shlex.quote(claude_cmd)} --version"
            )
        else:
            probe = (
                'powershell -NoProfile -Command "'
                f"if (Test-Path '{claude_cmd}') {{ & '{claude_cmd}' --version }} "
                "else { Write-Error 'claude not found'; exit 1 }"
                '"'
            )
        result = self._run(vm_config, probe, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI missing or not runnable at {claude_cmd}. "
                f"Bake `{cfg.cli_version}` into the VM image, or override "
                f"runtime.cli_path for this image. stderr: "
                f"{(result.stderr or '').strip()[:300]}"
            )
        logger.info(
            "claude_code: claude CLI ok — %s",
            (result.stdout or "").strip(),
        )

        # 2. Create work_dir.
        self._mkdir(vm_config, wd)

        # 3. Write MCP config.
        mcp_path = self._join(runtime.vm_os, wd, "mcp_config.json")
        mcp_config = {
            "mcpServers": {
                "cua": {
                    "command": runtime.node_exe,
                    "args": [f"{runtime.mcp_server_dir}/src/index.js"],
                },
            },
        }
        upload_file(vm_config, mcp_path, json.dumps(mcp_config, indent=2))
        logger.info("claude_code: install ok — work_dir=%s", wd)

    # =========================================================================
    # launch — spawn claude detached on VM, poll done.marker from host
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        runtime = self.runtime
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        vm_config = self._vm_config()
        wd = runtime.work_dir_vm
        claude_cmd = runtime.cli_path("claude")

        join = lambda *parts: self._join(runtime.vm_os, *parts)
        prompt_file = join(wd, "prompt.txt")
        transcript_file = join(wd, "transcript.jsonl")
        stderr_log = join(wd, "stderr.log")
        pid_file = join(wd, "claude.pid")
        done_marker = join(wd, "done.marker")
        mcp_config = join(wd, "mcp_config.json")

        # Stage prompt + reset markers + spawn the detached runner. All sync
        # HTTP — wrap in to_thread so we don't block the event loop.
        await asyncio.to_thread(
            self._spawn_runner_sync,
            vm_config=vm_config,
            cfg=cfg,
            env=dict(runtime.env),
            wd=wd,
            claude_cmd=claude_cmd,
            prompt=prompt,
            prompt_file=prompt_file,
            transcript_file=transcript_file,
            stderr_log=stderr_log,
            pid_file=pid_file,
            done_marker=done_marker,
            mcp_config=mcp_config,
        )

        # Resolve pid (best-effort).
        pid = await asyncio.to_thread(self._read_pid_sync, vm_config, pid_file)

        # Poll done.marker until exit or deadline.
        t0 = time.monotonic()
        deadline = t0 + cfg.timeout_s
        poll_interval = 5.0
        while True:
            if await asyncio.to_thread(_http_file_exists, vm_config, done_marker):
                raw = (await asyncio.to_thread(_http_read_text, vm_config, done_marker)).strip()
                try:
                    exit_code = int(raw) if raw else None
                except ValueError:
                    exit_code = None
                status = "completed" if exit_code == 0 else "failed"
                error = None
                if status == "failed":
                    error = await asyncio.to_thread(
                        self._diagnose_failure_sync,
                        vm_config=vm_config,
                        stderr_log=stderr_log,
                        transcript=transcript_file,
                        exit_code=exit_code,
                    )
                return AgentRunResult(
                    status=status,
                    pid=pid,
                    exit_code=exit_code,
                    transcript_path=transcript_file,
                    stderr_path=stderr_log,
                    duration_s=time.monotonic() - t0,
                    error=error,
                )

            if time.monotonic() >= deadline:
                if pid is not None:
                    await asyncio.to_thread(self._kill_pid_sync, vm_config, pid)
                return AgentRunResult(
                    status="timeout",
                    pid=pid,
                    transcript_path=transcript_file,
                    stderr_path=stderr_log,
                    duration_s=time.monotonic() - t0,
                    error=f"wall budget {cfg.timeout_s}s exceeded",
                )
            await asyncio.sleep(poll_interval)

    # ------------------------------------------------------------------ sync helpers

    def _spawn_runner_sync(
        self,
        *,
        vm_config: RemoteVMConfig,
        cfg: ClaudeCodeConfig,
        env: dict[str, str],
        wd: str,
        claude_cmd: str,
        prompt: str,
        prompt_file: str,
        transcript_file: str,
        stderr_log: str,
        pid_file: str,
        done_marker: str,
        mcp_config: str,
    ) -> None:
        # Clean old markers + outputs (best effort).
        self._rm(vm_config, [done_marker, pid_file, stderr_log, transcript_file])
        # Stage prompt.
        upload_file(vm_config, prompt_file, prompt)

        if vm_config.is_linux:
            self._launch_linux_sync(
                vm_config=vm_config, cfg=cfg, env=env, wd=wd,
                claude_cmd=claude_cmd, prompt_file=prompt_file,
                transcript_file=transcript_file, stderr_log=stderr_log,
                pid_file=pid_file, done_marker=done_marker, mcp_config=mcp_config,
            )
        else:
            self._launch_windows_sync(
                vm_config=vm_config, cfg=cfg, env=env, wd=wd,
                claude_cmd=claude_cmd, prompt_file=prompt_file,
                transcript_file=transcript_file, stderr_log=stderr_log,
                pid_file=pid_file, done_marker=done_marker, mcp_config=mcp_config,
            )

    @classmethod
    def _read_pid_sync(cls, vm_config: RemoteVMConfig, pid_file: str) -> int | None:
        # Launcher writes PID synchronously; ~15 ticks of 300ms tolerance.
        for _ in range(15):
            if _http_file_exists(vm_config, pid_file):
                raw = _http_read_text(vm_config, pid_file).strip()
                try:
                    return int(raw)
                except ValueError:
                    return None
            time.sleep(0.3)
        return None

    @classmethod
    def _kill_pid_sync(cls, vm_config: RemoteVMConfig, pid: int) -> None:
        if vm_config.is_linux:
            cls._run(vm_config, f"kill -TERM {pid}", timeout=15)
            time.sleep(2)
            cls._run(vm_config, f"kill -KILL {pid}", timeout=15)
        else:
            cls._run(
                vm_config,
                f'powershell -NoProfile -Command "Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"',
                timeout=15,
            )

    # ---- per-OS launch ----------------------------------------------------

    @staticmethod
    def _build_argv_linux(
        *, claude_cmd: str, cfg: ClaudeCodeConfig, mcp_config: str,
    ) -> str:
        argv = [
            shlex.quote(claude_cmd), "-p", "-",
            "--output-format", "stream-json", "--verbose",
            "--mcp-config", shlex.quote(mcp_config),
            "--model", shlex.quote(cfg.model),
        ]
        if cfg.max_turns is not None and cfg.max_turns >= 0:
            argv += ["--max-turns", str(cfg.max_turns)]
        if cfg.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(cfg.max_budget_usd)]
        if cfg.dangerously_skip_permissions:
            argv += ["--dangerously-skip-permissions"]
        for tool in cfg.disabled_tools:
            argv += ["--disallowedTools", shlex.quote(tool)]
        return " ".join(argv)

    @classmethod
    def _launch_linux_sync(
        cls, *, vm_config, cfg, env, wd,
        claude_cmd, prompt_file, transcript_file, stderr_log,
        pid_file, done_marker, mcp_config,
    ) -> None:
        base_url_default = cfg.base_url or "https://openrouter.ai/api"
        env_lines: list[str] = []
        if not env.get("ANTHROPIC_API_KEY") and env.get("OPENROUTER_API_KEY"):
            env_lines += [
                f"export ANTHROPIC_BASE_URL={shlex.quote(base_url_default)}",
                f'export ANTHROPIC_AUTH_TOKEN={shlex.quote(env["OPENROUTER_API_KEY"])}',
                'export ANTHROPIC_API_KEY=""',
            ]
        elif env.get("ANTHROPIC_API_KEY"):
            env_lines.append(
                f'export ANTHROPIC_API_KEY={shlex.quote(env["ANTHROPIC_API_KEY"])}'
            )
        if cfg.base_url:
            env_lines.append(f"export ANTHROPIC_BASE_URL={shlex.quote(cfg.base_url)}")

        cmd_line = cls._build_argv_linux(
            claude_cmd=claude_cmd, cfg=cfg, mcp_config=mcp_config,
        )

        runner_script = cls._join("linux", wd, "run_claude.sh")
        launcher_script = cls._join("linux", wd, "launch.sh")

        runner = (
            "#!/bin/bash\nset -u\n"
            + "\n".join(env_lines) + ("\n" if env_lines else "")
            + f"cd {shlex.quote(wd)}\n"
            + f"prompt=$(cat {shlex.quote(prompt_file)})\n"
            + f"echo \"$prompt\" | {cmd_line} "
            + f"2>{shlex.quote(stderr_log)} >{shlex.quote(transcript_file)}\n"
            + f"echo $? > {shlex.quote(done_marker)}\n"
        )
        launcher = (
            "#!/bin/bash\n"
            f"setsid bash {shlex.quote(runner_script)} </dev/null >/dev/null 2>&1 &\n"
            "CHILD=$!\n"
            f"echo \"$CHILD\" > {shlex.quote(pid_file)}\n"
            "disown $CHILD 2>/dev/null || true\n"
        )
        upload_file(vm_config, runner_script, runner)
        upload_file(vm_config, launcher_script, launcher)
        cls._run(
            vm_config,
            f"chmod +x {shlex.quote(runner_script)} {shlex.quote(launcher_script)}",
            timeout=15,
        )
        cls._run(vm_config, f"bash {shlex.quote(launcher_script)}", timeout=30)

    @staticmethod
    def _build_argv_windows(
        *, claude_cmd: str, cfg: ClaudeCodeConfig, mcp_config: str,
    ) -> str:
        argv = [
            f"& '{claude_cmd}'", "-p", "-",
            "--output-format", "stream-json", "--verbose",
            "--mcp-config", f"'{mcp_config}'",
            "--model", cfg.model,
        ]
        if cfg.max_turns is not None and cfg.max_turns >= 0:
            argv += ["--max-turns", str(cfg.max_turns)]
        if cfg.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(cfg.max_budget_usd)]
        if cfg.dangerously_skip_permissions:
            argv += ["--dangerously-skip-permissions"]
        for tool in cfg.disabled_tools:
            argv += ["--disallowedTools", f"'{tool}'"]
        return " ".join(argv)

    @classmethod
    def _launch_windows_sync(
        cls, *, vm_config, cfg, env, wd,
        claude_cmd, prompt_file, transcript_file, stderr_log,
        pid_file, done_marker, mcp_config,
    ) -> None:
        base_url_default = cfg.base_url or "https://openrouter.ai/api"
        env_lines: list[str] = []
        if not env.get("ANTHROPIC_API_KEY") and env.get("OPENROUTER_API_KEY"):
            env_lines += [
                f"$env:ANTHROPIC_BASE_URL = '{base_url_default}'",
                f"$env:ANTHROPIC_AUTH_TOKEN = '{env['OPENROUTER_API_KEY']}'",
                "$env:ANTHROPIC_API_KEY = ''",
            ]
        elif env.get("ANTHROPIC_API_KEY"):
            env_lines.append(
                f"$env:ANTHROPIC_API_KEY = '{env['ANTHROPIC_API_KEY']}'"
            )
        if cfg.base_url:
            env_lines.append(f"$env:ANTHROPIC_BASE_URL = '{cfg.base_url}'")

        cmd_line = cls._build_argv_windows(
            claude_cmd=claude_cmd, cfg=cfg, mcp_config=mcp_config,
        )

        runner_script = cls._join("windows", wd, "run_claude.ps1")

        runner = (
            "$ErrorActionPreference = 'Continue'\n"
            "$utf8 = New-Object System.Text.UTF8Encoding($false)\n"
            "[Console]::InputEncoding = $utf8\n"
            "[Console]::OutputEncoding = $utf8\n"
            "$OutputEncoding = $utf8\n"
            + "\n".join(env_lines) + ("\n" if env_lines else "")
            + f"Set-Location -LiteralPath '{wd}'\n"
            + f"$prompt = Get-Content -LiteralPath '{prompt_file}' -Encoding UTF8 -Raw\n"
            + f"$prompt | {cmd_line} 2>'{stderr_log}' | Out-File -FilePath '{transcript_file}' -Encoding utf8\n"
            + f"$LASTEXITCODE | Out-File -FilePath '{done_marker}' -Encoding ascii -NoNewline\n"
        )
        upload_file(vm_config, runner_script, runner)

        spawn_cmd = (
            'powershell -NoProfile -Command "'
            f"$proc = Start-Process powershell "
            f"-ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','{runner_script}' "
            f"-WindowStyle Hidden -PassThru; "
            f"$proc.Id | Out-File -FilePath '{pid_file}' -Encoding ascii -NoNewline"
            '"'
        )
        cls._run(vm_config, spawn_cmd, timeout=30)

    # ---- failure diagnosis -------------------------------------------------

    @staticmethod
    def _diagnose_failure_sync(
        *,
        vm_config: RemoteVMConfig,
        stderr_log: str,
        transcript: str,
        exit_code: int | None,
    ) -> str:
        parts = [f"agent failed (rc={exit_code})"]
        stderr_text = _http_read_text(vm_config, stderr_log)
        tx_text = _http_read_text(vm_config, transcript)
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
    # parse_artifacts — host-side, reads gathered transcript.jsonl
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
        """Parse stream-json transcript → ATIF Steps."""
        transcript_file = work_dir / "transcript.jsonl"
        if not transcript_file.exists():
            builder.add_step(
                source="system",
                message=f"claude-code: no transcript at {transcript_file}",
                extra={"reason": "no_transcript"},
            )
            return

        raw = transcript_file.read_text()
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
