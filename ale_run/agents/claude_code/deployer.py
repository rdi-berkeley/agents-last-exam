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
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentDeployer,
    ContentPart,
    ImageSource,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

from .config import ClaudeCodeConfig
from .telemetry import ClaudeCodeOtelCollector, recover_telemetry_artifacts

logger = logging.getLogger(__name__)


_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0

_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")


def _expected_version(npm_spec: str | None) -> str | None:
    """Parse the pinned version out of ``cli_version``.

    ``"@anthropic-ai/claude-code@2.1.170"`` → ``"2.1.170"``. Returns ``None``
    for unpinned specs (``"@anthropic-ai/claude-code"``), in which case
    whatever is already installed is accepted.
    """
    if not npm_spec:
        return None
    m = _VERSION_RE.search(npm_spec.rsplit("@", 1)[-1])
    return m.group(1) if m else None


def _find_claude_shim(prefix: str) -> str | None:
    """Locate the npm-installed claude shim under our install prefix.

    npm drops the shim in ``<prefix>/bin/claude`` on Linux and directly in
    ``<prefix>\\claude.cmd`` on Windows. Resolving it explicitly (instead of
    ``shutil.which``) makes the freshly-installed copy authoritative even
    when an older claude elsewhere on PATH would shadow it (e.g. a pre-baked
    copy in ``AppData\\Roaming\\npm``).
    """
    for cand in (
        os.path.join(prefix, "bin", "claude"),     # Linux
        os.path.join(prefix, "claude.cmd"),        # Windows
        os.path.join(prefix, "bin", "claude.cmd"),
    ):
        if os.path.exists(cand):
            return cand
    return None


def _installed_version(claude_path: str) -> str | None:
    """Best-effort ``claude --version`` → ``X.Y.Z``; ``None`` if unreadable.

    stdin=DEVNULL so the probe never blocks on a TTY check (on Windows the
    first exec of a freshly-baked claude.EXE can hang on Defender scan +
    stdin); 60s gives cold-image first-exec headroom.
    """
    try:
        probe = subprocess.run(
            [claude_path, "--version"],
            capture_output=True, text=True, timeout=60,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = _VERSION_RE.search((probe.stdout or "") + (probe.stderr or ""))
    return m.group(1) if m else None


class ClaudeCodeDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the @anthropic-ai/claude-code CLI."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    # otel_requests.jsonl is the raw OTLP write-ahead log: mirror it incrementally
    # off the sandbox so long tasks do not depend solely on the final gather.
    # The derived telemetry.jsonl / telemetry_summary.json are rebuilt from it in
    # parse_artifacts, so they are deliberately NOT hot.
    hot_artifacts: ClassVar[tuple[str, ...]] = (
        "transcript.jsonl",
        "stderr.log",
        "otel_requests.jsonl",
    )

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
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if not npm:
            from ale_run.agents._bootstrap import ensure_npm
            npm = await ensure_npm()
        home = os.path.expanduser("~")
        prefix = os.path.join(home, ".local")
        env = {**os.environ, "npm_config_cache": os.path.join(home, ".npm-ale")}
        # cfg.cli_version is the full npm spec, e.g.
        # "@anthropic-ai/claude-code@2.1.170". --force so a same-version
        # residue under the prefix is overwritten cleanly.
        pkg = cfg.cli_version or "@anthropic-ai/claude-code"
        # 300s is too short for a COLD npm install on Windows (no pre-baked CLI,
        # empty cache) — e.g. the ale-win-server image, which (unlike ale-win10)
        # doesn't bake the claude CLI, needs ~5-10 min. Allow 15 min.
        proc = await asyncio.to_thread(
            subprocess.run,
            [npm, "install", "-g", "--force", "--prefix", prefix, pkg],
            capture_output=True, text=True, timeout=900, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"npm install -g {pkg} failed "
                f"(rc={proc.returncode}): {(proc.stderr or '')[:500]}"
            )
        # npm drops the claude shim in <prefix>/bin on Linux and directly in
        # <prefix> on Windows. Prepend both UNCONDITIONALLY so our freshly
        # installed copy wins over any pre-baked claude elsewhere on PATH —
        # a membership check is not enough: the dir may already be on PATH
        # but BEHIND the dir holding a stale baked copy (seen on ale-win10,
        # where AppData\Roaming\npm shadowed ~/.local).
        for bin_dir in (prefix, os.path.join(prefix, "bin")):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        logger.info("claude_code: auto-installed via npm — %s", (proc.stdout or "").strip()[-200:])

    async def install(self) -> None:
        cfg: ClaudeCodeConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox
        is_linux = sandbox.is_linux

        # 1. Discover the claude binary and decide whether to (re)install.
        # Reinstall when EITHER:
        #   • claude is not found at all, or
        #   • its --version is STALE vs the version pinned in cli_version.
        # The stale check is what lets a cli_version bump ship onto images
        # with an older claude pre-baked: the next run detects the mismatch
        # and installs the pinned version under ~/.local, prepended on PATH
        # so it shadows the baked copy.
        claude_path = shutil.which("claude")
        expected = _expected_version(cfg.cli_version)
        installed = (
            await asyncio.to_thread(_installed_version, claude_path)
            if claude_path else None
        )
        stale = bool(expected and installed != expected)
        if not claude_path or stale:
            if claude_path:
                logger.info(
                    "claude_code: installed version %s != expected %s — "
                    "installing pinned version", installed, expected,
                )
            else:
                logger.info("claude_code: 'claude' not on PATH, installing via npm …")
            await self._auto_install_cli()
            # Resolve the shim under OUR prefix explicitly — which() may
            # still surface a stale baked copy from elsewhere on PATH.
            home = os.path.expanduser("~")
            claude_path = (
                _find_claude_shim(os.path.join(home, ".local"))
                or shutil.which("claude")
            )
            if not claude_path:
                raise RuntimeError(
                    "ClaudeCodeDeployer: 'claude' still not found after "
                    f"npm install -g {cfg.cli_version}"
                )
            installed = await asyncio.to_thread(_installed_version, claude_path)
            if expected and installed != expected:
                raise RuntimeError(
                    f"ClaudeCodeDeployer: post-install version {installed} "
                    f"still != expected {expected} (path {claude_path})"
                )
        else:
            logger.info(
                "claude_code: reusing installed claude %s (version %s)",
                claude_path, installed or "unknown",
            )
        self._claude_path = claude_path
        logger.info("claude_code: claude CLI ok — version %s", installed or "unknown")

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
        collector = ClaudeCodeOtelCollector(wd) if cfg.otel_enabled else None

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
        # Start the run-local OTLP receiver BEFORE composing env, so its
        # endpoint can be injected into the CLI's telemetry env vars.
        otel_endpoint: str | None = None
        if collector is not None:
            collector.start()
            otel_endpoint = collector.endpoint
            logger.info("claude_code: OTel collector listening at %s", otel_endpoint)
        env = self._build_env(cfg, otel_endpoint=otel_endpoint)

        t0 = time.monotonic()
        try:
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

            # Wait for the child to finish.
            # The episode wall budget is orchestration-owned: the executor
            # wraps launch() in asyncio.wait_for(timeout=timeout_s) (derived
            # from the task), so we just wait for the child here. If that
            # budget fires we are cancelled mid-await; reap the child before
            # propagating so it cannot outlive the run.
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
        finally:
            # Stop the collector so it drains the final batch and writes the
            # derived telemetry files, whether the run completed or was killed.
            if collector is not None:
                collector.stop()

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
        if cfg.effort_level:
            # Reasoning-effort for adaptive-thinking Claude models (→ API
            # output_config.effort). A CLI flag like --model, not an env var.
            argv += ["--effort", cfg.effort_level]
        if cfg.max_turns is not None and cfg.max_turns >= 0:
            argv += ["--max-turns", str(cfg.max_turns)]
        if cfg.max_budget_usd is not None:
            argv += ["--max-budget-usd", str(cfg.max_budget_usd)]
        if cfg.dangerously_skip_permissions:
            argv += ["--dangerously-skip-permissions"]
        for tool in cfg.disabled_tools:
            argv += ["--disallowedTools", tool]
        return argv

    def _build_env(
        self, cfg: ClaudeCodeConfig, *, otel_endpoint: str | None = None,
    ) -> dict[str, str]:
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

        # Extended-thinking budget — pin it explicitly via Claude Code's
        # documented MAX_THINKING_TOKENS env var. The CLI already defaults to
        # extended thinking (31999 cap); setting it makes the reasoning level
        # explicit + tunable (parity with codex's reasoning_effort=high).
        # None ⇒ omit the thinking param entirely via Claude Code's
        # CLAUDE_CODE_DISABLE_THINKING=1. Required for adaptive-thinking
        # models (fable-5) over OpenRouter: the CLI otherwise sends
        # thinking={type:"adaptive"}, which the gateway mangles into
        # thinking={type:"disabled"} upstream → 400 ("thinking.type.disabled
        # is not supported"). With the param absent the provider applies the
        # model's native adaptive thinking (the model still thinks).
        # Verified via request capture + e2e tool flow on 2.1.170.
        #
        # When effort_level is set, --effort is the reasoning control: skip the
        # legacy MAX_THINKING_TOKENS/budget path entirely (budget_tokens is
        # removed → 400 on Opus 4.7/4.8) and leave thinking to the model's
        # adaptive default — don't disable it.
        if cfg.effort_level:
            env.pop("MAX_THINKING_TOKENS", None)
            env.pop("CLAUDE_CODE_DISABLE_THINKING", None)
        elif cfg.max_thinking_tokens is not None:
            env["MAX_THINKING_TOKENS"] = str(cfg.max_thinking_tokens)
            env.pop("CLAUDE_CODE_DISABLE_THINKING", None)
        else:
            env.pop("MAX_THINKING_TOKENS", None)
            env["CLAUDE_CODE_DISABLE_THINKING"] = "1"

        # Raise the CLI's per-response output-token ceiling. The Claude Code CLI
        # defaults to a 32000 output-token max; a single thinking-heavy turn
        # (large `MAX_THINKING_TOKENS` + a long tool result) can exceed it and
        # the CLI aborts the whole run with "response exceeded the N output
        # token maximum". Set from cfg (the yaml) — NOT os.environ, which in the
        # sandbox is the _sandbox_entry env, not the operator's shell.
        if cfg.max_output_tokens is not None:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(cfg.max_output_tokens)

        # Provider-driven routing (explicit, not key-presence heuristic).
        if cfg.provider == "openrouter":
            # A literal cfg.api_key (travels with the serialized config) takes
            # precedence over OPENROUTER_API_KEY and avoids env collisions.
            token = cfg.api_key or env.get("OPENROUTER_API_KEY")
            if not token:
                raise RuntimeError(
                    "claude_code: provider=openrouter but neither config "
                    "api_key nor OPENROUTER_API_KEY is set"
                )
            env["ANTHROPIC_BASE_URL"] = cfg.base_url or "https://openrouter.ai/api"
            env["ANTHROPIC_AUTH_TOKEN"] = token
            env["ANTHROPIC_API_KEY"] = ""
        elif cfg.provider == "direct":
            key = cfg.api_key or env.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "claude_code: provider=direct but neither config api_key "
                    "nor ANTHROPIC_API_KEY is set"
                )
            env["ANTHROPIC_API_KEY"] = key
            if cfg.base_url:
                env["ANTHROPIC_BASE_URL"] = cfg.base_url
        elif cfg.provider == "bedrock":
            # Claude Code calls Bedrock through the AWS SDK when
            # CLAUDE_CODE_USE_BEDROCK=1; auth is the standard AWS credential
            # chain. The framework resolves the operator's AWS credentials on
            # the host and propagates them into the sandbox (see
            # _collect_env_passthrough), so they arrive here as env vars.
            # Require at least an access key OR a bearer token so we fail fast
            # with a clear message instead of letting the CLI 403 mid-run.
            has_sigv4 = bool(env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY"))
            has_bearer = bool(env.get("AWS_BEARER_TOKEN_BEDROCK"))
            if not (has_sigv4 or has_bearer):
                raise RuntimeError(
                    "claude_code: provider=bedrock but no AWS credentials in env "
                    "(need AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, or "
                    "AWS_BEARER_TOKEN_BEDROCK). Authenticate on the host (e.g. "
                    "`ada credentials update ...` / `aws sso login`) before running."
                )
            env["CLAUDE_CODE_USE_BEDROCK"] = "1"
            region = cfg.aws_region or env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
            if region:
                env["AWS_REGION"] = region
            # OpenRouter/Anthropic-direct routing vars must not leak in: a
            # stale ANTHROPIC_BASE_URL would send Bedrock traffic to the wrong
            # endpoint. Bedrock auth is AWS-SDK only.
            for k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
                env.pop(k, None)
            # Long-run credential refresh: point the AWS SDK at a
            # credential_process that re-reads an always-fresh creds file (kept
            # current by a host sidecar). Static SigV4 env creds OVERRIDE
            # credential_process, so they MUST be cleared for the refresh to
            # take effect — verified: env creds win the AWS credential chain.
            if cfg.aws_credential_process_file:
                self._setup_credential_process(env, cfg.aws_credential_process_file)
        else:
            raise RuntimeError(
                f"claude_code: unknown provider {cfg.provider!r} "
                "(expected 'openrouter', 'direct', or 'bedrock')"
            )

        # OpenTelemetry: point Claude Code's built-in exporter at the run-local
        # OTLP/HTTP receiver. Claude Code emits per-event logs (api_request,
        # tool_result, tool_decision, user_prompt, api_error, ...) and cumulative
        # metrics; we take both over http/json so the collector can parse JSON
        # directly. Short export intervals keep the raw WAL near-real-time for the
        # incremental sandbox pull. Prompt + tool-detail logging is enabled so the
        # events carry the operational context (prompt text, tool arguments) the
        # transcript alone omits. No upstream Claude Code changes are required.
        if otel_endpoint is not None:
            env.update({
                "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
                "OTEL_LOGS_EXPORTER": "otlp",
                "OTEL_METRICS_EXPORTER": "otlp",
                "OTEL_TRACES_EXPORTER": "none",
                "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
                "OTEL_EXPORTER_OTLP_ENDPOINT": otel_endpoint,
                "OTEL_LOGS_EXPORT_INTERVAL": "1000",
                "OTEL_METRIC_EXPORT_INTERVAL": "10000",
                "OTEL_LOG_USER_PROMPTS": "1",
                "OTEL_LOG_TOOL_DETAILS": "1",
            })
        return env

    @staticmethod
    def _setup_credential_process(env: dict[str, str], creds_file: str) -> None:
        """Wire the AWS SDK to a refreshable credential_process (long runs).

        Writes ``~/.aws/config`` with a profile whose ``credential_process``
        cats ``creds_file``, seeds that file with the current (still-valid) env
        creds so calls work before the host sidecar's first refresh, points the
        SDK at the profile, and CLEARS the static env creds (they would
        otherwise override credential_process and keep expiring). Runs
        in-sandbox, so all paths are sandbox-local; the host sidecar overwrites
        ``creds_file`` in the running container as the token rotates.
        """
        import json as _json

        home = os.path.expanduser("~")
        aws_dir = os.path.join(home, ".aws")
        os.makedirs(aws_dir, exist_ok=True)
        config_path = os.path.join(aws_dir, "config")
        profile = "ale_refresh"
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"[profile {profile}]\n"
                f"credential_process = cat {creds_file}\n"
            )

        # Seed the creds file with the current env creds (process format) so
        # the SDK has valid creds immediately; the sidecar refreshes it later.
        ak, sk, st = (
            env.get("AWS_ACCESS_KEY_ID"),
            env.get("AWS_SECRET_ACCESS_KEY"),
            env.get("AWS_SESSION_TOKEN"),
        )
        if ak and sk:
            seed = {
                "Version": 1,
                "AccessKeyId": ak,
                "SecretAccessKey": sk,
            }
            if st:
                seed["SessionToken"] = st
            try:
                os.makedirs(os.path.dirname(creds_file) or "/", exist_ok=True)
                with open(creds_file, "w", encoding="utf-8") as fh:
                    _json.dump(seed, fh)
            except OSError as e:
                logger.warning("claude_code: could not seed creds file %s: %s", creds_file, e)

        env["AWS_CONFIG_FILE"] = config_path
        env["AWS_PROFILE"] = profile
        env["AWS_SDK_LOAD_CONFIG"] = "1"
        # Static env creds OVERRIDE credential_process — clear them so the SDK
        # uses the refreshable profile instead of the frozen, expiring token.
        for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            env.pop(k, None)
        logger.info(
            "claude_code: bedrock credential_process wired (profile=%s, creds_file=%s)",
            profile, creds_file,
        )

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
        config: ClaudeCodeConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        # Rebuild derived telemetry (telemetry.jsonl / telemetry_metrics.jsonl /
        # telemetry_summary.json) from the incrementally-mirrored raw OTLP WAL,
        # so a task whose final directory gather was interrupted still yields
        # complete telemetry. Best-effort: never let it break artifact parsing.
        try:
            recover_telemetry_artifacts(work_dir)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("claude_code: could not recover telemetry artifacts: %s", exc)

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

        # Usage/cost reconciliation. Claude Code reports the AUTHORITATIVE
        # cumulative usage (`usage`) + `total_cost_usd` on the final `result`
        # event. Per-message `usage` is frequently all-zero over OpenRouter (the
        # CLI doesn't get token counts back from a non-Anthropic gateway), so the
        # per-step sum alone would be 0. Add a reconciliation step carrying the
        # DELTA between the result-event cumulative and whatever per-step metrics
        # already summed, plus the full cost — so the trajectory's final_metrics
        # equal the result-event total on OpenRouter AND don't double-count on a
        # direct Anthropic run where per-message usage IS populated.
        result = builder.trajectory.extra.get("result") or {}
        ru = result.get("usage") or {}
        if ru or result.get("total_cost_usd") is not None:
            si = so = scr = scc = 0
            for s in builder.trajectory.steps:
                m = s.metrics
                if m:
                    si += m.input_tokens or 0
                    so += m.output_tokens or 0
                    scr += m.cache_read_tokens or 0
                    scc += m.cache_creation_tokens or 0
            builder.add_step(
                source="system",
                message=None,
                metrics=StepMetrics(
                    input_tokens=max((ru.get("input_tokens") or 0) - si, 0),
                    output_tokens=max((ru.get("output_tokens") or 0) - so, 0),
                    cache_read_tokens=max((ru.get("cache_read_input_tokens") or 0) - scr, 0),
                    cache_creation_tokens=max((ru.get("cache_creation_input_tokens") or 0) - scc, 0),
                    cost_usd=result.get("total_cost_usd"),
                ),
                extra={"usage_reconciliation": True},
            )

        builder.trajectory.extra.setdefault("claude_code", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(transcript_file),
            "stderr_path": run_result.stderr_path,
            "telemetry_path": str(work_dir / "telemetry.jsonl"),
            "telemetry_metrics_path": str(work_dir / "telemetry_metrics.jsonl"),
            "telemetry_summary_path": str(work_dir / "telemetry_summary.json"),
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
                        if not isinstance(c, dict):
                            continue
                        ctype = c.get("type")
                        if ctype == "text":
                            parts.append(ContentPart(type="text", text=c.get("text", "")))
                        elif ctype == "image":
                            # MCP/cua tool results return screenshots as Anthropic
                            # image blocks: {"type":"image","source":{"type":
                            # "base64","media_type":"image/png","data":"..."}}.
                            # Keep them as inline ImageSource so persist_screenshots
                            # writes them to screenshots/ and rewrites to path refs.
                            src = c.get("source") or {}
                            if src.get("type") == "base64" and src.get("data"):
                                parts.append(ContentPart(
                                    type="image",
                                    image=ImageSource(
                                        type="base64",
                                        media_type=src.get("media_type", "image/png"),
                                        data=src.get("data"),
                                    ),
                                ))
                            elif src.get("type") == "url" and src.get("url"):
                                parts.append(ContentPart(
                                    type="image",
                                    image=ImageSource(type="url", url=src.get("url")),
                                ))
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
