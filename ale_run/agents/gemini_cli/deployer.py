"""GeminiCliDeployer — drives the Google ``gemini`` CLI.

Install via npm, configure MCP for CUA bridge, launch in yolo mode,
parse stream-json NDJSON output into trajectory steps.
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

from .config import GeminiCliConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0

_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")


def _expected_version(npm_package: str) -> str | None:
    """Parse the fork version from the configured tarball URL/path.

    The release asset is named ``google-gemini-cli-<version>.tgz``, so the
    version is the single source of truth in ``npm_package``. Returns ``None``
    for non-versioned specs (e.g. a ``github:owner/repo#branch`` git spec), in
    which case install falls back to location-based detection only.
    """
    m = re.search(r"-(\d+\.\d+\.\d+)\.tgz(?:$|[?#])", npm_package)
    return m.group(1) if m else None


def _installed_version(gemini_path: str) -> str | None:
    """Best-effort ``gemini --version`` → ``X.Y.Z``; ``None`` if unreadable."""
    try:
        probe = subprocess.run(
            [gemini_path, "--version"],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    m = _VERSION_RE.search((probe.stdout or "") + (probe.stderr or ""))
    return m.group(1) if m else None


def _find_gemini_shim(local_prefix: str) -> str | None:
    """Resolve the gemini executable, preferring our ``~/.local`` install.

    The sandbox entry runs WITHOUT a login shell, so ``~/.local`` may not be on
    PATH and ``shutil.which('gemini')`` misses both an image-baked fork under
    our prefix AND a copy we just installed. npm drops the shim at
    ``<prefix>/bin/gemini`` on Linux and directly in ``<prefix>`` on Windows
    (``gemini.cmd`` / ``gemini.ps1`` / ``gemini``); ``shutil.which`` does not
    reliably find the Windows ``.cmd`` shim from this process. Fall back to the
    exact shim paths so detection AND post-install resolution behave identically
    on Windows and Linux.
    """
    p = shutil.which("gemini")
    if p and p.startswith(local_prefix):
        return p
    for cand in (
        os.path.join(local_prefix, "bin", "gemini"),
        os.path.join(local_prefix, "gemini.cmd"),
        os.path.join(local_prefix, "gemini.ps1"),
        os.path.join(local_prefix, "gemini"),
    ):
        if os.path.isfile(cand):
            return cand
    return p

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
        from ale_run.agents._bootstrap import ensure_npm
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if not npm:
            npm = await ensure_npm()
        home = os.path.expanduser("~")
        prefix = os.path.join(home, ".local")
        env = {**os.environ, "npm_config_cache": os.path.join(home, ".npm-ale")}
        # --force so a stale baked tarball (old converter) is overwritten by the
        # configured fork tarball even when npm thinks the version is unchanged.
        proc = await asyncio.to_thread(
            subprocess.run,
            [npm, "install", "-g", "--force", "--prefix", prefix, package],
            capture_output=True, text=True, timeout=300, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"npm install -g {package} failed "
                f"(rc={proc.returncode}): {(proc.stderr or '')[:500]}"
            )
        # npm drops the gemini shim in <prefix>/bin on Linux and directly in
        # <prefix> on Windows. Put both on PATH (prepended) so our freshly
        # installed copy wins over any pre-baked system gemini.
        for bin_dir in (prefix, os.path.join(prefix, "bin")):
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        logger.info("gemini_cli: installed via npm — %s", (proc.stdout or "").strip()[-200:])

    async def install(self) -> None:
        cfg: GeminiCliConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        # Ensure node/npm reachable (on Windows node ships off PATH).
        from ale_run.agents._bootstrap import ensure_npm
        await ensure_npm()

        home = os.path.expanduser("~")
        local_prefix = os.path.join(home, ".local")
        gemini_path = _find_gemini_shim(local_prefix)
        # Decide whether to (re)install the configured fork tarball. Reinstall
        # when ANY of:
        #   • gemini is not found at all, or
        #   • it is baked OUTSIDE our prefix (an image-baked copy we can't trust
        #     to be the configured fork), or
        #   • its --version is STALE vs the version encoded in npm_package.
        # The stale check is what lets a fork code change ship: bump the tarball
        # version and the next run detects the image's older copy and upgrades,
        # even when the prior copy already lives under ~/.local. (Before version
        # bumping, all fork variants reported the same version and could not be
        # distinguished — hence the older location-only heuristic.)
        baked = bool(gemini_path) and not gemini_path.startswith(local_prefix)
        expected = _expected_version(cfg.npm_package)
        installed = (
            await asyncio.to_thread(_installed_version, gemini_path)
            if gemini_path else None
        )
        stale = bool(expected and installed and installed != expected)
        if not gemini_path or baked or stale:
            if stale:
                logger.info(
                    "gemini_cli: installed version %s != expected %s — "
                    "reinstalling configured fork tarball", installed, expected,
                )
            elif baked:
                logger.info(
                    "gemini_cli: pre-baked gemini at %s (outside %s) — "
                    "reinstalling configured fork tarball to override",
                    gemini_path, local_prefix,
                )
            else:
                logger.info("gemini_cli: 'gemini' not on PATH, installing via npm …")
            await self._auto_install_cli(cfg.npm_package)
            gemini_path = _find_gemini_shim(local_prefix)
            if not gemini_path:
                raise RuntimeError(
                    "GeminiCliDeployer: 'gemini' still not found after npm install"
                )
            installed = await asyncio.to_thread(_installed_version, gemini_path)
            if expected and installed and installed != expected:
                logger.warning(
                    "gemini_cli: post-install version %s still != expected %s",
                    installed, expected,
                )
        else:
            logger.info(
                "gemini_cli: reusing installed gemini %s (version %s)",
                gemini_path, installed or "unknown",
            )
        self._gemini_path = gemini_path
        logger.info("gemini_cli: CLI ok — version %s", installed or "unknown")

        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        home = os.path.expanduser("~")
        gemini_home = Path(home) / ".gemini"
        gemini_home.mkdir(parents=True, exist_ok=True)

        # Ensure the cua MCP bridge is installed at sandbox.mcp_server_dir
        # (idempotent: no-op when prebaked, install when missing).
        from ale_run.agents._bootstrap import cua_bridge_env, ensure_cua_mcp_server
        await ensure_cua_mcp_server(sandbox)

        # MCP + settings
        settings = {
            "mcpServers": {
                "cua": {
                    "command": sandbox.node,
                    "args": [self._join(sandbox.mcp_server_dir, "src", "index.js",
                                        is_linux=sandbox.is_linux)],
                    "env": cua_bridge_env(self.executor),
                },
            },
            "tools": {
                "exclude": list(cfg.disabled_tools),
            },
            "maxSessionTurns": cfg.max_session_turns,
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
        logger.info("gemini_cli: argv=%s", argv)

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

    def _resolve_model(self, cfg: GeminiCliConfig) -> str:
        # Both routes take the bare model id: the agenthle fork maps
        # "gemini-*" to "google/gemini-*" on the OpenRouter request itself,
        # and Google's native API also expects bare ids. Strip any "google/"
        # prefix so one config value works for either provider.
        if cfg.model.startswith("google/"):
            return cfg.model[len("google/"):]
        return cfg.model

    def _build_argv(self, cfg: GeminiCliConfig) -> list[str]:
        argv = [
            self._gemini_path,
            "-p", "-",
            "--model", self._resolve_model(cfg),
            "--output-format", "stream-json",
            "--approval-mode", cfg.approval_mode,
        ]
        # Tasks read/write under task_data_root, which is outside the launch
        # cwd. Gemini's file tools reject paths outside the workspace, so add
        # the task data root as an extra workspace directory.
        task_data_root = getattr(self.executor.sandbox, "task_data_root", "")
        if task_data_root:
            argv.append(f"--include-directories={task_data_root}")
        # Deny-only tool policy: no --allowed-tools allow list. Everything is
        # available except what `disabled_tools` excludes (settings.json "exclude").
        return argv

    def _build_env(self, cfg: GeminiCliConfig) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env["NO_COLOR"] = "1"
        env["NO_BROWSER"] = "1"

        provider = (cfg.provider or "openrouter").lower()
        if provider == "openrouter":
            # The agenthle fork auto-selects OpenRouter auth from
            # OPENROUTER_API_KEY and forwards tool-result content correctly,
            # so native file tools work. GEMINI_API_KEY must be cleared or the
            # CLI would prefer Google's native API over OpenRouter.
            if not env.get("OPENROUTER_API_KEY"):
                raise RuntimeError(
                    "gemini_cli provider=openrouter requires OPENROUTER_API_KEY"
                )
            env.setdefault("OPENROUTER_COMPRESSION_MODEL", cfg.compression_model)
            env.pop("GEMINI_API_KEY", None)
            env.pop("GOOGLE_API_KEY", None)
        elif provider == "google":
            # Direct Google API. The CLI reads GEMINI_API_KEY; mirror
            # GOOGLE_API_KEY into it when only the latter is set.
            if not (env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY")):
                raise RuntimeError(
                    "gemini_cli provider=google requires GEMINI_API_KEY or GOOGLE_API_KEY"
                )
            env.setdefault("GEMINI_API_KEY", env.get("GOOGLE_API_KEY", ""))
            env.pop("OPENROUTER_API_KEY", None)
        else:
            raise RuntimeError(
                f"gemini_cli: unknown provider {cfg.provider!r} "
                "(expected 'openrouter' or 'google')"
            )
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
        config: GeminiCliConfig,
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
        raw = error if error else output
        text = raw if isinstance(raw, str) else json.dumps(raw)
        content: list[ContentPart] = [ContentPart(type="text", text=text)]

        # Inline media the tool returned to the model (e.g. the CUA `screenshot`
        # PNG). The fork surfaces these as a `media` array on the TOOL_RESULT
        # event ([{mime_type, data?, uri?}]); `output` is only the
        # `[Image: image/png]` placeholder. Keep them as image ContentParts so
        # persist_screenshots() writes them to screenshots/.
        for m in event.get("media") or []:
            if not isinstance(m, dict):
                continue
            data, uri = m.get("data"), m.get("uri")
            media_type = m.get("mime_type") or "image/png"
            if data:
                img = ImageSource(type="base64", media_type=media_type, data=data)
            elif uri:
                img = ImageSource(type="url", media_type=media_type, url=uri)
            else:
                continue
            content.append(ContentPart(type="image", image=img))

        builder.add_step(
            source="environment",
            observation=Observation(results=[
                ToolResult(
                    tool_call_id=event.get("tool_id", ""),
                    content=content,
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

        # Usage. Gemini CLI reports cumulative token counts on the final
        # `result` event's `stats`. Two shapes coexist: a per-model breakdown
        # under stats.models.<model> AND flat top-level fields (input_tokens /
        # output_tokens / cached) carrying the SAME totals. Prefer the models
        # breakdown and fall back to the flat fields so we never double-count.
        # `cached` is the cache-read subset of input_tokens. Gemini CLI does
        # NOT report cost (no total_cost_usd), so cost is left unset.
        metrics = GeminiCliDeployer._stats_to_metrics(stats)
        if metrics is not None:
            builder.add_step(
                source="system",
                message=None,
                metrics=metrics,
                extra={"usage_result": True},
            )

    @staticmethod
    def _stats_to_metrics(stats: dict) -> StepMetrics | None:
        if not isinstance(stats, dict) or not stats:
            return None
        input_total = output_total = cache_total = 0
        saw_any = False

        def _toks(d: dict) -> tuple[int, int, int]:
            inp = d.get("input_tokens")
            if inp is None:
                inp = d.get("inputTokens", d.get("prompt", d.get("input", 0)))
            out = d.get("output_tokens")
            if out is None:
                out = d.get("outputTokens", d.get("candidates", 0))
            cached = d.get("cached")
            if cached is None:
                cached = d.get("cachedInputTokens", 0)
            return int(inp or 0), int(out or 0), int(cached or 0)

        models = stats.get("models", {})
        if isinstance(models, dict) and models:
            for model_data in models.values():
                if not isinstance(model_data, dict):
                    continue
                tokens = model_data.get("tokens")
                src = tokens if isinstance(tokens, dict) else model_data
                inp, out, cached = _toks(src)
                input_total += inp
                output_total += out
                cache_total += cached
                saw_any = True
        else:
            inp, out, cached = _toks(stats)
            if inp or out or cached:
                input_total, output_total, cache_total = inp, out, cached
                saw_any = True

        if not saw_any:
            return None
        return StepMetrics(
            input_tokens=max(input_total - cache_total, 0),
            output_tokens=output_total,
            cache_read_tokens=cache_total or None,
        )


def _read_text_tolerant(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""
