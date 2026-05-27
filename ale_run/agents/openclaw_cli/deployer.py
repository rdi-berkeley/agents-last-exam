"""OpenClawCliDeployer — drives ``openclaw agent --local``.

Fork tarball + native CUA plugin (not MCP).  The deployer expects both
the tarball and plugin source to be available inside the sandbox at
configurable paths (baked into the image or volume-mounted).

Config files written: ``openclaw.json``, ``auth-profiles.json``,
``exec-approvals.json``, ``workspace-state.json``.

Output: JSON envelope on stderr (``--json`` flag), plus session
trajectory JSONL at ``~/.openclaw/agents/main/sessions/``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
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

from .config import CUA_TOOL_NAMES, OpenClawCliConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0
_AGENT_ID = "main"

_STDERR_PREAMBLE_PREFIXES = (
    "[agent/embedded] session file repaired",
    "[agent/embedded] embedded run agent end",
    "[agent/embedded] embedded run failover decision",
    "[diagnostic] lane task error",
    "[model-fallback/decision]",
)


class OpenClawCliDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for ``openclaw agent --local``."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("stderr.log",)

    # =========================================================================
    # install
    # =========================================================================

    async def _install_from_tarball(self, tarball: str) -> None:
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("OpenClawCliDeployer: npm not on PATH")
        home = os.path.expanduser("~")
        env = {**os.environ, "npm_config_cache": f"{home}/.npm-ale"}
        proc = await asyncio.to_thread(
            subprocess.run,
            [npm, "install", "-g", "--prefix", f"{home}/.local", tarball],
            capture_output=True, text=True, timeout=300, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"npm install -g {tarball} failed "
                f"(rc={proc.returncode}): {(proc.stderr or '')[:500]}"
            )
        bin_dir = f"{home}/.local/bin"
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        logger.info("openclaw_cli: installed from tarball — %s",
                     (proc.stdout or "").strip()[-200:])

    async def _build_cua_plugin(self, plugin_src: str) -> None:
        """Build CUA plugin from source and install to ~/.openclaw/extensions/cua/."""
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("npm not on PATH — cannot build CUA plugin")

        home = os.path.expanduser("~")
        build_dir = Path(home) / ".ale-cua-plugin-build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        shutil.copytree(plugin_src, str(build_dir))

        env = {**os.environ, "npm_config_cache": f"{home}/.npm-ale"}
        for step_name, step_cmd in [
            ("npm install", [npm, "install", "--no-audit", "--no-fund"]),
            ("npm run build", [npm, "run", "build"]),
        ]:
            proc = await asyncio.to_thread(
                subprocess.run, step_cmd,
                capture_output=True, text=True, timeout=120,
                cwd=str(build_dir), env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"CUA plugin {step_name} failed (rc={proc.returncode}): "
                    f"{(proc.stderr or '')[:500]}"
                )

        install_dir = Path(home) / ".openclaw" / "extensions" / "cua"
        install_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("package.json", "openclaw.plugin.json"):
            src = build_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(install_dir / fname))
        dist_src = build_dir / "dist" / "index.cjs"
        if dist_src.exists():
            (install_dir / "dist").mkdir(exist_ok=True)
            shutil.copy2(str(dist_src), str(install_dir / "dist" / "index.cjs"))
        logger.info("openclaw_cli: CUA plugin installed at %s", install_dir)

    def _write_config(self, cfg: OpenClawCliConfig) -> None:
        """Write openclaw.json, auth-profiles.json, exec-approvals, workspace-state."""
        home = os.path.expanduser("~")
        oc_home = Path(home) / ".openclaw"
        oc_home.mkdir(parents=True, exist_ok=True)

        # --- openclaw.json ---
        primary_model = cfg.model
        tools_also_allow = list(CUA_TOOL_NAMES)
        oc_config = {
            "agents": {
                "defaults": {
                    "model": {"primary": primary_model},
                    "timeoutSeconds": int(cfg.timeout_s),
                    "heartbeat": {"every": cfg.heartbeat_every},
                    "models": {primary_model: {}},
                },
            },
            "plugins": {
                "allow": list(cfg.plugins_allow),
                "deny": list(cfg.plugins_deny),
            },
            "tools": {
                "alsoAllow": tools_also_allow,
                "deny": list(cfg.tools_deny),
            },
            "gateway": {
                "mode": "local",
                "bind": "loopback",
            },
        }
        if cfg.vision_model:
            oc_config["tools"]["media"] = {
                "image": {"models": {"default": cfg.vision_model}},
            }
        (oc_home / "openclaw.json").write_text(
            json.dumps(oc_config, indent=2), encoding="utf-8",
        )

        # --- exec-approvals.json (yolo) ---
        approvals = {
            "version": 1,
            "defaults": {
                "security": "full",
                "ask": "off",
                "askFallback": "full",
            },
            "socket": {},
            "agents": {},
        }
        (oc_home / "exec-approvals.json").write_text(
            json.dumps(approvals, indent=2), encoding="utf-8",
        )

        # --- auth-profiles.json ---
        agent_dir = oc_home / "agents" / _AGENT_ID / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)

        env = self.executor.env or {}
        provider = "openrouter"
        api_key = env.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            provider = "openai"
            api_key = env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

        auth = {
            "profiles": {
                f"{provider}:default": {
                    "provider": provider,
                    "type": "api_key",
                    "key": api_key,
                },
            },
            "lastGood": {
                provider: f"{provider}:default",
            },
        }
        (agent_dir / "auth-profiles.json").write_text(
            json.dumps(auth, indent=2), encoding="utf-8",
        )

        # --- workspace-state.json (skip bootstrap wizard) ---
        now = datetime.now(timezone.utc).isoformat()
        state = {
            "version": 1,
            "setupCompletedAt": now,
            "bootstrapSeededAt": now,
        }
        state_dir = oc_home / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "workspace-state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8",
        )

        # Remove BOOTSTRAP.md if present
        bootstrap_md = oc_home / "BOOTSTRAP.md"
        if bootstrap_md.exists():
            bootstrap_md.unlink()

        # --- .env ---
        env_file = oc_home / ".env"
        env_file.write_text("OPENCLAW_RAW_STREAM=0\n", encoding="utf-8")

        logger.info("openclaw_cli: config staged at %s", oc_home)

    async def install(self) -> None:
        cfg: OpenClawCliConfig = self.config  # type: ignore[assignment]

        # 1. Install openclaw CLI
        openclaw_path = shutil.which("openclaw")
        if not openclaw_path:
            tarball = cfg.tarball_path
            if not Path(tarball).exists():
                raise RuntimeError(
                    f"OpenClawCliDeployer: 'openclaw' not on PATH and "
                    f"tarball not found at {tarball}. Bake the tarball into "
                    f"the sandbox image or set tarball_path in config."
                )
            logger.info("openclaw_cli: installing from tarball %s", tarball)
            await self._install_from_tarball(tarball)
            openclaw_path = shutil.which("openclaw")
            if not openclaw_path:
                raise RuntimeError(
                    "OpenClawCliDeployer: 'openclaw' still not found after install"
                )
        self._openclaw_path = openclaw_path

        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [openclaw_path, "--version"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"openclaw --version timed out: {e}")
        logger.info("openclaw_cli: CLI ok — %s", (probe.stdout or "").strip())

        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # 2. Build CUA plugin (if source available and not already installed)
        home = os.path.expanduser("~")
        plugin_entry = Path(home) / ".openclaw" / "extensions" / "cua" / "dist" / "index.cjs"
        if not plugin_entry.exists():
            plugin_src = cfg.cua_plugin_path
            if Path(plugin_src).is_dir():
                logger.info("openclaw_cli: building CUA plugin from %s", plugin_src)
                await self._build_cua_plugin(plugin_src)
            else:
                logger.warning(
                    "openclaw_cli: CUA plugin source not found at %s — "
                    "computer tools will be unavailable",
                    plugin_src,
                )
        else:
            logger.info("openclaw_cli: CUA plugin already installed")

        # 3. Write config files
        self._write_config(cfg)

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: OpenClawCliConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        prompt_file = wd / "prompt.txt"
        stdout_log = wd / "stdout.log"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "openclaw.pid"

        for f in (stdout_log, stderr_log, pid_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        prompt_file.write_text(prompt, encoding="utf-8")

        home = os.path.expanduser("~")
        env_file = Path(home) / ".openclaw" / ".env"
        argv = [
            self._openclaw_path,
            "agent", "--local",
            "--agent", _AGENT_ID,
            "--message", prompt,
            "--json",
            "--timeout", str(int(cfg.timeout_s)),
            "--thinking", cfg.thinking,
        ]
        env = self._build_env(cfg, env_file)

        t0 = time.monotonic()
        with open(stdout_log, "wb") as fout, \
             open(stderr_log, "wb") as ferr:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                argv,
                stdin=subprocess.DEVNULL,
                stdout=fout,
                stderr=ferr,
                env=env,
                cwd=str(wd),
                start_new_session=True if hasattr(os, "setsid") else False,
            )
        pid_file.write_text(str(proc.pid), encoding="ascii")
        logger.info("openclaw_cli: spawned pid=%s", proc.pid)

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
            error = _diagnose_failure(stderr_log, exit_code)

        # Copy session trajectory to work_dir for gathering
        session_id = self._extract_session_id(stderr_log)
        if session_id:
            src = Path(home) / ".openclaw" / "agents" / _AGENT_ID / "sessions" / f"{session_id}.jsonl"
            dst = wd / "transcript.jsonl"
            if src.exists():
                shutil.copy2(str(src), str(dst))
                logger.info("openclaw_cli: copied session trajectory to %s", dst)

        return AgentRunResult(
            status=status,
            pid=proc.pid,
            exit_code=exit_code,
            transcript_path=str(wd / "transcript.jsonl"),
            stderr_path=str(stderr_log),
            duration_s=duration_s,
            error=error,
        )

    # =========================================================================
    # internals
    # =========================================================================

    def _build_env(self, cfg: OpenClawCliConfig, env_file: Path) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env["NO_COLOR"] = "1"
        # Source .env file values
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
        return env

    @staticmethod
    def _extract_session_id(stderr_log: Path) -> str | None:
        """Extract session ID from the --json stderr envelope."""
        text = _read_text_tolerant(stderr_log)
        if not text:
            return None
        json_obj = _parse_stderr_json(text)
        if json_obj:
            meta = json_obj.get("meta", {})
            agent_meta = meta.get("agentMeta", {})
            return agent_meta.get("sessionId")
        return None

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
        # 1. Parse session trajectory JSONL (if available)
        transcript_file = work_dir / "transcript.jsonl"
        if transcript_file.exists():
            raw = transcript_file.read_text(encoding="utf-8", errors="replace")
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cls._consume_session_event(event, builder)

        # 2. Parse stderr JSON envelope
        stderr_file = work_dir / "stderr.log"
        if stderr_file.exists():
            stderr_text = stderr_file.read_text(encoding="utf-8", errors="replace")
            json_obj = _parse_stderr_json(stderr_text)
            if json_obj:
                builder.trajectory.extra.setdefault("openclaw_cli", {})["result_envelope"] = json_obj
                meta = json_obj.get("meta", {})
                agent_meta = meta.get("agentMeta", {})
                usage = agent_meta.get("usage", {})
                if usage:
                    builder.trajectory.extra["openclaw_cli"]["usage"] = usage

        if not transcript_file.exists():
            builder.add_step(
                source="system",
                message="openclaw-cli: no session transcript available",
                extra={"reason": "no_transcript"},
            )

        builder.trajectory.extra.setdefault("openclaw_cli", {}).update({
            "exit_code": run_result.exit_code,
        })

    @classmethod
    def _consume_session_event(cls, event: dict, builder: TrajectoryBuilder) -> None:
        etype = event.get("type")
        if etype == "message":
            cls._consume_message(event, builder)
        elif etype == "tool_result":
            cls._consume_tool_result(event, builder)

    @staticmethod
    def _consume_message(event: dict, builder: TrajectoryBuilder) -> None:
        message = event.get("message", {})
        role = message.get("role", "")
        content_blocks = message.get("content", [])
        if not isinstance(content_blocks, list):
            content_blocks = []

        usage = message.get("usage", {})
        metrics = StepMetrics(
            input_tokens=usage.get("input"),
            output_tokens=usage.get("output"),
            cache_read_tokens=usage.get("cacheRead"),
            cache_creation_tokens=usage.get("cacheWrite"),
        ) if usage else None

        if role == "assistant":
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_calls: list[ToolCall] = []

            for block in content_blocks:
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "thinking":
                    reasoning_parts.append(block.get("thinking", ""))
                elif btype in ("toolCall", "tool_use"):
                    name = block.get("name", "")
                    args = block.get("arguments") or block.get("input") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    tool_calls.append(ToolCall(
                        id=block.get("id", ""),
                        name=name,
                        arguments=args,
                    ))

            builder.add_step(
                source="agent",
                message="\n".join(p for p in text_parts if p) or None,
                reasoning="\n".join(p for p in reasoning_parts if p) or None,
                tool_calls=tool_calls,
                metrics=metrics,
            )
        elif role == "user":
            text_parts = []
            results: list[ToolResult] = []
            for block in content_blocks:
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    content = block.get("content")
                    parts: list[ContentPart] = []
                    if isinstance(content, str):
                        parts.append(ContentPart(type="text", text=content))
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                parts.append(ContentPart(type="text", text=c.get("text", "")))
                    results.append(ToolResult(
                        tool_call_id=block.get("tool_use_id") or block.get("call_id", ""),
                        content=parts,
                        is_error=bool(block.get("is_error")),
                    ))
            if results:
                builder.add_step(
                    source="environment",
                    observation=Observation(results=results),
                )
            elif text_parts:
                builder.add_step(
                    source="user",
                    message="\n".join(p for p in text_parts if p),
                )
        elif role == "toolResult":
            output = event.get("output") or message.get("output", "")
            call_id = event.get("tool_use_id") or event.get("call_id", "")
            builder.add_step(
                source="environment",
                observation=Observation(results=[
                    ToolResult(
                        tool_call_id=call_id,
                        content=[ContentPart(type="text", text=str(output))],
                        is_error=bool(event.get("is_error")),
                    ),
                ]),
            )

    @staticmethod
    def _consume_tool_result(event: dict, builder: TrajectoryBuilder) -> None:
        output = event.get("output", "")
        call_id = event.get("tool_use_id") or event.get("call_id", "")
        builder.add_step(
            source="environment",
            observation=Observation(results=[
                ToolResult(
                    tool_call_id=call_id,
                    content=[ContentPart(type="text", text=str(output))],
                    is_error=bool(event.get("is_error")),
                ),
            ]),
        )


def _parse_stderr_json(stderr: str) -> dict | None:
    """Find JSON envelope in openclaw --json stderr stream."""
    if not stderr:
        return None

    lines = stderr.splitlines()
    # Strategy 1: line-based — first line that is just "{"
    for i, line in enumerate(lines):
        if line.strip() == "{":
            payload = "\n".join(lines[i:])
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                break

    # Strategy 2: backward brace-balance from last "}"
    text = stderr.rstrip()
    if text.endswith("}"):
        end = len(text) - 1
        depth = 1
        in_str = False
        for j in range(end - 1, -1, -1):
            ch = text[j]
            if in_str:
                if ch == '"' and (j == 0 or text[j - 1] != "\\"):
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[j:end + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def _diagnose_failure(stderr_log: Path, exit_code: int | None) -> str:
    parts = [f"agent failed (rc={exit_code})"]
    text = _read_text_tolerant(stderr_log)
    if text.strip():
        parts.append(f"stderr tail: ...{text[-800:]}")
    return " | ".join(parts)


def _read_text_tolerant(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""
