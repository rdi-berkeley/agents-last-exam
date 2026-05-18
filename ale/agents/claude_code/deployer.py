"""ClaudeCodeDeployer — runs the @anthropic-ai/claude-code CLI INSIDE the eval VM.

Lives at ``runtime: vm`` only. When the framework invokes
``deployer.install()`` / ``deployer.launch(prompt)``, those methods are
**running inside the VM's Python process** (shipped by :class:`VmExecutor`
via ``cua.python_exec``). So:

  - ``subprocess.run("npm install -g ...")`` executes on the VM
  - ``Path("/home/user/.ale/...").write_text(...)`` writes to VM fs
  - ``self.runtime`` is a :class:`VmRuntime` constructed locally in VM

The deployer NEVER touches a cua session — it doesn't need one (it IS in
the VM, uses local stdlib). Only ``parse_artifacts`` runs on the
framework host, reading the gathered work_dir locally.

Image assumptions (defaults match ``agenthle-ubuntu-0505``):
  - Node 24.x at ``/usr/local/bin/node``
  - @anthropic-ai/claude-code at ``/usr/local/bin/claude`` (baked)
  - We do NOT install claude at runtime in v1 — the image has it.
    If the binary is missing the deployer raises clearly (rebuild image
    or move to a baked one).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar, Iterable

from ale.agents.base import (
    AgentRunResult,
    BaseAgentConfig,
    BaseAgentDeployer,
)
from ale.agents.trajectory import (
    ContentPart,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

from .config import ClaudeCodeConfig

logger = logging.getLogger(__name__)


class ClaudeCodeDeployer(BaseAgentDeployer):
    """In-VM deployer for the @anthropic-ai/claude-code CLI."""

    supported_runtimes: ClassVar[frozenset[str]] = frozenset({"vm"})

    # Files the framework keeps incrementally-pulled to the host during
    # agent run, so Ctrl-C / VM revert / network blip don't lose
    # diagnostic data. Pulled every 15s with JSONL-boundary safety.
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    @property
    def version(self) -> str | None:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        return cfg.cli_version

    # =========================================================================
    # install — runs inside the VM
    # =========================================================================

    async def install(self) -> None:
        """Verify claude CLI present, write MCP config, create work_dir.

        Runs INSIDE the VM. Uses stdlib only.
        """
        import subprocess
        from pathlib import Path as P

        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        runtime = self.runtime
        claude_cmd = runtime.cli_path("claude")    # type: ignore[attr-defined]

        # 1. Verify claude binary present (image-baked expectation)
        if not P(claude_cmd).exists():
            raise RuntimeError(
                f"claude CLI missing at {claude_cmd}. Bake "
                f"`{cfg.cli_version}` into the VM image; runtime npm-install "
                f"is intentionally not done in v1."
            )

        # 1b. Heal /home/user/.claude.json — the agenthle-ubuntu-0505 image
        # baked a 0-byte ~/.claude.json which causes claude CLI 2.1.85 to
        # fail at startup with "Unexpected end of JSON input". simprun has
        # the same workaround (deployers/claude_code.py). We rewrite
        # unconditionally — these are throwaway benchmark VMs with no
        # persistent user state worth keeping, and validate-then-skip has
        # subtle edge cases (0-byte file passes a `or "{}"` truthy check).
        cj = P("/home/user/.claude.json")
        try:
            cj.write_text("{}")
            cj.chmod(0o600)
            logger.info("claude_code: rewrote %s as `{}` (heal)", cj)
        except OSError as exc:
            logger.warning("claude_code: could not heal %s: %s", cj, exc)
        version_out = subprocess.run(
            [claude_cmd, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        logger.info("claude_code: claude CLI ok — %s", version_out.stdout.strip())

        # 2. Create work_dir
        wd = P(runtime.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # 3. Write MCP config pointing at cua mcp server (if installed; if
        #    not, claude still runs but won't have MCP tools — fine for
        #    demo/hello which uses claude's built-in exec).
        node_exe = runtime.node_exe                # type: ignore[attr-defined]
        mcp_dir = runtime.mcp_server_dir           # type: ignore[attr-defined]
        mcp_config = {
            "mcpServers": {
                "cua": {
                    "command": node_exe,
                    "args": [f"{mcp_dir}/src/index.js"],
                },
            },
        }
        (wd / "mcp_config.json").write_text(
            json.dumps(mcp_config, indent=2),
        )
        logger.info("claude_code: install ok — work_dir=%s", wd)

    # =========================================================================
    # launch — runs inside the VM
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        """Spawn claude CLI detached, poll done.marker, classify outcome.

        Pattern: setsid + bg-run + done.marker poll. Survives long agent
        runs that exceed any single RPC timeout (the launch call blocks
        in this Python process which is running ON the VM via python_exec
        — the LIMIT is python_exec's own timeout, currently long enough
        for demo/hello).
        """
        import asyncio
        import shlex
        import subprocess
        import time
        from pathlib import Path as P

        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        runtime = self.runtime
        wd = P(runtime.work_dir)

        claude_cmd = runtime.cli_path("claude")    # type: ignore[attr-defined]
        prompt_file = wd / "prompt.txt"
        runner_script = wd / "run_claude.sh"
        launcher_script = wd / "launch.sh"
        transcript_file = wd / "transcript.jsonl"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "claude.pid"
        done_marker = wd / "done.marker"
        mcp_config = wd / "mcp_config.json"

        prompt_file.write_text(prompt)

        # ---- env setup ----
        # Read keys from VM-side os.environ (populated by VmExecutor's
        # host_env passthrough → _vm_entry.os.environ.update) and INLINE
        # the values into the bash script. Matches simprun's proven
        # pattern (deployers/claude_code.py:L375-L382): explicit values
        # > env inheritance, plus an explicit ANTHROPIC_API_KEY='' reset
        # for the OpenRouter remap path — claude 2.1.85 prefers API_KEY
        # over AUTH_TOKEN whenever API_KEY is "set" (incl. empty), so an
        # ambient ANTHROPIC_API_KEY=... from host shell would still take
        # precedence and break the remap. Clearing it is the only
        # reliable fix.
        import os as _os
        or_key = _os.environ.get("OPENROUTER_API_KEY", "")
        ant_key = _os.environ.get("ANTHROPIC_API_KEY", "")
        if or_key:
            base_url = cfg.base_url or "https://openrouter.ai/api"
            env_lines = [
                f"export ANTHROPIC_BASE_URL={shlex.quote(base_url)}",
                f"export ANTHROPIC_AUTH_TOKEN={shlex.quote(or_key)}",
                "export ANTHROPIC_API_KEY=''",
            ]
        elif ant_key:
            env_lines = [f"export ANTHROPIC_API_KEY={shlex.quote(ant_key)}"]
            if cfg.base_url:
                env_lines.append(f"export ANTHROPIC_BASE_URL={shlex.quote(cfg.base_url)}")
        else:
            raise RuntimeError(
                "claude_code: neither OPENROUTER_API_KEY nor ANTHROPIC_API_KEY "
                "is set in the VM-side os.environ. Either the host shell didn't "
                "export them (check `source .env`) or VmExecutor's host_env "
                "propagation broke."
            )

        # ---- CLI args ----
        argv = [shlex.quote(claude_cmd), "-p", "-",
                "--output-format", "stream-json", "--verbose",
                "--mcp-config", shlex.quote(str(mcp_config)),
                "--model", shlex.quote(cfg.model)]
        if cfg.max_turns is not None and cfg.max_turns >= 0:
            argv += ["--max-turns", str(cfg.max_turns)]
        if cfg.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(cfg.max_budget_usd)]
        if cfg.dangerously_skip_permissions:
            argv += ["--dangerously-skip-permissions"]
        for tool in cfg.disabled_tools:
            argv += ["--disallowedTools", shlex.quote(tool)]
        cmd_line = " ".join(argv)

        # ---- runner.sh: env + run claude with prompt on stdin ----
        runner_script.write_text(
            "#!/bin/bash\nset -u\n"
            + "\n".join(env_lines) + "\n"
            f"cd {shlex.quote(str(wd))}\n"
            f"prompt=$(cat {shlex.quote(str(prompt_file))})\n"
            f"echo \"$prompt\" | {cmd_line} "
            f"2>{shlex.quote(str(stderr_log))} >{shlex.quote(str(transcript_file))}\n"
            f"echo $? > {shlex.quote(str(done_marker))}\n"
        )
        # ---- launch.sh: setsid + record PID ----
        launcher_script.write_text(
            "#!/bin/bash\n"
            f"rm -f {shlex.quote(str(done_marker))} {shlex.quote(str(pid_file))}\n"
            f"setsid bash {shlex.quote(str(runner_script))} </dev/null >/dev/null 2>&1 &\n"
            "CHILD=$!\n"
            f"echo \"$CHILD\" > {shlex.quote(str(pid_file))}\n"
            "disown $CHILD 2>/dev/null || true\n"
        )
        subprocess.run(["chmod", "+x", str(runner_script), str(launcher_script)], check=True)
        subprocess.run(["bash", str(launcher_script)], check=True, timeout=10)

        # ---- read PID ----
        pid: int | None = None
        for _ in range(15):
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text().strip())
                    break
                except ValueError:
                    pass
            await asyncio.sleep(0.3)
        if pid is None:
            return AgentRunResult(
                status="failed",
                error="launcher did not write a PID — see stderr.log",
                stderr_path=str(stderr_log),
                duration_s=0.0,
            )

        # ---- poll done.marker ----
        deadline = time.monotonic() + cfg.timeout_s
        poll_interval = 5.0
        t0 = time.monotonic()
        while True:
            if done_marker.exists():
                raw = done_marker.read_text().strip()
                try:
                    exit_code = int(raw) if raw else None
                except ValueError:
                    exit_code = None
                status = "completed" if exit_code == 0 else "failed"
                error = None
                if status == "failed":
                    error = self._diagnose_failure(
                        stderr_log=stderr_log,
                        transcript=transcript_file,
                        exit_code=exit_code,
                    )
                return AgentRunResult(
                    status=status,
                    pid=pid,
                    exit_code=exit_code,
                    transcript_path=str(transcript_file),
                    stderr_path=str(stderr_log),
                    duration_s=time.monotonic() - t0,
                    error=error,
                )
            # check process alive
            alive_check = subprocess.run(
                ["kill", "-0", str(pid)],
                capture_output=True,
            )
            if alive_check.returncode != 0:
                # process gone but no done.marker — crash
                return AgentRunResult(
                    status="failed",
                    pid=pid,
                    transcript_path=str(transcript_file),
                    stderr_path=str(stderr_log),
                    duration_s=time.monotonic() - t0,
                    error="process disappeared before writing done.marker; see stderr",
                )
            if time.monotonic() >= deadline:
                subprocess.run(["kill", "-TERM", str(pid)])
                await asyncio.sleep(2)
                subprocess.run(["kill", "-KILL", str(pid)])
                return AgentRunResult(
                    status="timeout",
                    pid=pid,
                    transcript_path=str(transcript_file),
                    stderr_path=str(stderr_log),
                    duration_s=time.monotonic() - t0,
                    error=f"wall budget {cfg.timeout_s}s exceeded",
                )
            await asyncio.sleep(poll_interval)

    @staticmethod
    def _diagnose_failure(*, stderr_log: Path, transcript: Path, exit_code: int | None) -> str:
        parts = [f"agent failed (rc={exit_code})"]
        try:
            tail = stderr_log.read_text().strip()
            if tail:
                parts.append(f"stderr tail: ...{tail[-500:]}")
        except Exception:                                       # noqa: BLE001
            pass
        try:
            tx = transcript.read_text()
        except Exception:                                       # noqa: BLE001
            tx = ""
        if '"authentication_failed"' in tx or '"User not found"' in tx:
            parts.append("LLM auth failed (check api keys)")
        elif '"error_status":429' in tx or '"rate_limit_error"' in tx:
            parts.append("LLM rate-limited")
        elif '"error_status":5' in tx:
            parts.append("LLM upstream 5xx")
        elif '"type":"result"' not in tx and exit_code != 0:
            parts.append("agent never produced result event")
        return " | ".join(parts)

    # =========================================================================
    # parse_artifacts — runs on framework HOST after gather
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
