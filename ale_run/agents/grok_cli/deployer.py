"""GrokCliDeployer — drives the ``grok`` CLI from superagent-ai.

Standalone binary (no Node runtime for CLI itself).  Node IS required
for the CUA MCP Server bridge.  When the fork bundle is used, Bun
runtime is required (auto-installed).  Supports direct xAI or OpenRouter
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

    async def _auto_install_cli_windows(self, binary_url: str) -> str:
        """Download the fork ``grok.exe`` from a GitHub release.

        The Linux ``install.sh`` is bash-only and the linux ``grok-bundle.js``
        can't run under the Windows Bun loader (@opentui externalization), so
        Windows uses a self-contained native ``grok.exe`` compiled from the
        same fork tree via ``bun build --compile``. It carries the identical
        OpenRouter fixes as the linux bundle, so headless ``--prompt`` over
        OpenRouter works without ZodErrors. Download it to
        ``~\\.grok\\bin\\grok.exe`` and return its path.
        """
        home = os.path.expanduser("~")
        grok_bin_dir = Path(home) / ".grok" / "bin"
        grok_bin_dir.mkdir(parents=True, exist_ok=True)
        dest = grok_bin_dir / "grok.exe"
        proc = await asyncio.to_thread(
            subprocess.run,
            ["curl", "-fSL", "--retry", "3", "--retry-all-errors",
             "--connect-timeout", "30", "--max-time", "300",
             "-o", str(dest), binary_url],
            capture_output=True, text=True, timeout=360,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"grok-cli Windows binary download failed (rc={proc.returncode}) "
                f"from {binary_url}: {(proc.stderr or '')[:500]}"
            )
        if not dest.exists() or dest.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"grok.exe at {dest} missing or too small "
                f"({dest.stat().st_size if dest.exists() else 0} bytes)"
            )
        if str(grok_bin_dir) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(grok_bin_dir) + os.pathsep + os.environ.get("PATH", "")
        logger.info("grok_cli: installed fork Windows binary at %s (%d bytes)",
                    dest, dest.stat().st_size)
        return str(dest)

    async def install(self) -> None:
        cfg: GrokCliConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox
        self._is_windows = not sandbox.is_linux

        if self._is_windows:
            # Windows uses the self-contained fork grok.exe (carries the
            # OpenRouter fixes). Always (re)download from win_binary_url so a
            # reverted/clean VM state self-heals and a stale stock grok on
            # PATH never shadows the fork build. Empty url = debug-only
            # fallback to whatever 'grok' is already on PATH.
            if cfg.win_binary_url:
                logger.info("grok_cli: installing fork Windows binary from %s",
                            cfg.win_binary_url)
                grok_path = await self._auto_install_cli_windows(cfg.win_binary_url)
            else:
                grok_path = shutil.which("grok") or shutil.which("grok.exe")
                logger.warning(
                    "grok_cli: win_binary_url empty — using stock grok on PATH "
                    "(%s); fork OpenRouter fixes NOT present", grok_path,
                )
        else:
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

        # --- Fork bundle deployment (Linux only) ---------------------------
        # On Linux, when bundle_url is set, download the pre-built JS bundle
        # and place it next to the grok binary.  At launch time _build_argv()
        # switches from `grok --prompt ...` to `bun <bundle> --prompt ...`.
        # The bundle is built with `bun build --target=bun` and uses
        # bun:sqlite internally, so Bun runtime is required (not Node).
        # On Windows the bundle can't load (@opentui externalization under the
        # Windows Bun loader); instead the fork is shipped as a self-contained
        # native grok.exe (downloaded above from win_binary_url), which is the
        # launch target directly — no bundle, no bun. Both carry the same
        # OpenRouter fixes.
        self._bundle_path: str | None = None
        if cfg.bundle_url and not self._is_windows:
            await self._ensure_bun()
            await self._deploy_bundle(cfg.bundle_url)

        # Ensure the cua MCP bridge is installed at sandbox.mcp_server_dir
        # (idempotent: no-op when prebaked, install when missing).
        from ale_run.agents._bootstrap import cua_bridge_env, ensure_cua_mcp_server
        await ensure_cua_mcp_server(sandbox)

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
                        "env": cua_bridge_env(self.executor),
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
    # bundle deployment
    # =========================================================================

    async def _ensure_bun(self) -> None:
        """Install Bun runtime if not on PATH.

        The grok-cli fork bundle uses ``bun:sqlite`` and other Bun-specific
        APIs, so it must be run via ``bun <bundle>`` rather than ``node``.
        """
        bun_path = shutil.which("bun")
        if bun_path:
            self._bun_path = bun_path
            return
        home = os.path.expanduser("~")
        from ale_run.agents._bootstrap import ensure_unzip
        await ensure_unzip()
        logger.info("grok_cli: bun not found, installing ...")
        proc = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", "curl -fsSL https://bun.sh/install | bash"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "HOME": home},
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"bun install failed (rc={proc.returncode}): "
                f"{(proc.stderr or '')[:500]}"
            )
        bun_bin = f"{home}/.bun/bin"
        if bun_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bun_bin}:{os.environ.get('PATH', '')}"
        bun_path = shutil.which("bun")
        if not bun_path:
            raise RuntimeError("bun still not found after install")
        self._bun_path = bun_path
        logger.info("grok_cli: bun installed at %s", bun_path)

    async def _deploy_bundle(self, bundle_url: str) -> None:
        """Download the fork bundle and place it next to the grok binary.

        The bundle is a self-contained JS file (built with
        ``bun build --target=bun``) that replaces the default CLI
        entry point.  When present, launch uses
        ``bun <bundle_path> --prompt ...`` instead of the stock binary.
        """
        home = os.path.expanduser("~")
        grok_bin_dir = Path(home) / ".grok" / "bin"
        grok_bin_dir.mkdir(parents=True, exist_ok=True)
        bundle_dest = grok_bin_dir / "grok-bundle.js"

        logger.info("grok_cli: downloading fork bundle from %s", bundle_url)
        proc = await asyncio.to_thread(
            subprocess.run,
            ["curl", "-fsSL", "--retry", "3", "--retry-delay", "2",
             "--connect-timeout", "30", "--max-time", "120",
             "-o", str(bundle_dest), bundle_url],
            capture_output=True, text=True, timeout=180,
            env={**os.environ, "HOME": home},
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"grok-cli bundle download failed (rc={proc.returncode}): "
                f"{(proc.stderr or '')[:500]}"
            )
        if not bundle_dest.exists() or bundle_dest.stat().st_size < 1024:
            raise RuntimeError(
                f"grok-cli bundle at {bundle_dest} is missing or too small"
            )

        self._bundle_path = str(bundle_dest)
        logger.info(
            "grok_cli: fork bundle deployed at %s (%d bytes)",
            bundle_dest, bundle_dest.stat().st_size,
        )

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

        # The episode wall budget is orchestration-owned: the executor wraps
        # launch() in asyncio.wait_for(timeout=timeout_s) (derived from the
        # task), so we just wait for the child here. If that budget fires we
        # are cancelled mid-await; reap the child before propagating so it
        # cannot outlive the run.
        try:
            while proc.poll() is None:
                await asyncio.sleep(_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(proc.wait), timeout=_TERM_GRACE_S,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            raise

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
        if cfg.provider == "openrouter":
            effective_model = native_to_openrouter_model(cfg.model)

        max_rounds = cfg.max_tool_rounds
        if max_rounds == -1:
            max_rounds = 100_000

        # When the fork bundle is deployed, run it via bun instead of the
        # stock grok binary.  The bundle uses bun:sqlite and other Bun-
        # specific APIs, so Bun runtime is required.
        bundle_path = getattr(self, "_bundle_path", None)
        if bundle_path:
            bun = getattr(self, "_bun_path", None) or shutil.which("bun")
            if not bun:
                raise RuntimeError(
                    "bun not found on PATH — required to run the fork bundle"
                )
            argv = [
                bun, bundle_path,
                "--prompt", prompt,
                "--model", effective_model,
                "--format", "json",
                "--max-tool-rounds", str(max_rounds),
            ]
        else:
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
        # Provider-driven routing (explicit, not key-presence heuristic).
        if cfg.provider == "openrouter":
            or_key = env.get("OPENROUTER_API_KEY")
            if not or_key:
                raise RuntimeError(
                    "grok_cli: provider=openrouter but OPENROUTER_API_KEY "
                    "is not set"
                )
            env["GROK_API_KEY"] = or_key
            env["GROK_BASE_URL"] = "https://openrouter.ai/api/v1"
        elif cfg.provider == "direct":
            if not env.get("GROK_API_KEY"):
                raise RuntimeError(
                    "grok_cli: provider=direct but GROK_API_KEY is not set"
                )
        else:
            raise RuntimeError(
                f"grok_cli: unknown provider {cfg.provider!r} "
                "(expected 'openrouter' or 'direct')"
            )
        return env

    # =========================================================================
    # parse_artifacts
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: GrokCliConfig,
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
        # Grok CLI reports reliable per-step usage on each `step_finish` event:
        # usage.inputTokens / usage.outputTokens, plus usage.costUsdTicks
        # (one tick == 1e-6 USD). Populate StepMetrics per step so finalize()
        # sums tokens AND the grok-priced cost.
        usage = event.get("usage", {})
        if not isinstance(usage, dict) or not usage:
            return
        builder.trajectory.extra.setdefault("grok_cli", {}).setdefault(
            "usage_steps", [],
        ).append(usage)
        cost_ticks = usage.get("costUsdTicks")
        timing = event.get("timing", {})
        duration_ms = timing.get("durationMs") if isinstance(timing, dict) else None
        builder.add_step(
            source="system",
            message=None,
            metrics=StepMetrics(
                input_tokens=usage.get("inputTokens"),
                output_tokens=usage.get("outputTokens"),
                cost_usd=(cost_ticks / 1_000_000) if cost_ticks else None,
                duration_ms=duration_ms,
            ),
            extra={"usage_step": True, "finish_reason": event.get("finishReason")},
        )


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
