"""CodexDeployer — drives the OpenAI ``codex`` CLI (fork build, reports 0.0.0).

The deployer ensures the running ``codex`` is exactly the pinned fork build
(``CodexConfig.fork_version``): it compares ``codex --version`` and overlays the
fork native binary from a GitHub Release when missing/stale/stock (installing
stock from NPM first if nothing is on PATH), else skips the download. The fork =
openai/codex ``main`` + Windows ``apply_patch.exe`` fix + OpenRouter MCP
adaptation (see CodexConfig).

OpenRouter routing: ``OPENROUTER_API_KEY`` + ``config.toml`` with
``model_provider = "openrouter"`` and a custom model_providers block. Direct
routing: ``OPENAI_API_KEY`` → ``CODEX_API_KEY`` (non-default models via a supplied
model catalog).

MCP config at ``~/.codex/config.toml``.  Headless via
``--dangerously-bypass-approvals-and-sandbox`` (yolo) or
``--full-auto --sandbox <mode>``.  Output: NDJSON (one JSON object
per line).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
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

from .config import CodexConfig
from .telemetry import CodexOtelCollector, recover_telemetry_artifacts

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0

# npm-installed native binary paths (Linux).
# npm 11.x stopped hoisting platform deps so the nested copy is the one
# codex.js's require.resolve actually picks. Both paths are tried for
# replacement; whichever exists gets overwritten.
_VENDOR_BINARY_LINUX_TOPLEVEL = (
    "/usr/local/lib/node_modules/@openai/codex-linux-x64/"
    "vendor/x86_64-unknown-linux-musl/codex/codex"
)
_VENDOR_BINARY_LINUX_NESTED = (
    "/usr/local/lib/node_modules/@openai/codex/node_modules/"
    "@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/codex/codex"
)
# Windows equivalents (used when the sandbox VM is Windows).
_VENDOR_BINARY_WIN_TOPLEVEL = (
    r"C:\Users\User\AppData\Roaming\npm\node_modules\@openai\codex-win32-x64"
    r"\vendor\x86_64-pc-windows-msvc\codex\codex.exe"
)
_VENDOR_BINARY_WIN_NESTED = (
    r"C:\Users\User\AppData\Roaming\npm\node_modules\@openai\codex"
    r"\node_modules\@openai\codex-win32-x64"
    r"\vendor\x86_64-pc-windows-msvc\codex\codex.exe"
)


class CodexDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the OpenAI ``codex`` CLI."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = (
        "transcript.jsonl",
        "stderr.log",
        "otel_requests.jsonl",
    )

    # NPM stock fallback version (only relevant when nothing is on PATH and the
    # fork overlay is then applied on top). Last-resort value for ``version``.
    _PINNED_VERSION: ClassVar[str] = "0.114.0"

    @property
    def version(self) -> str | None:
        """Pinned build the deployer guarantees is running.

        Reports the fork build (``CodexConfig.fork_version`` — what install()
        ensures on PATH), not the npm stock fallback, so run metadata records
        the engine that actually executed.
        """
        cfg: CodexConfig = self.config  # type: ignore[assignment]
        return getattr(cfg, "fork_version", None) or self._PINNED_VERSION

    # =========================================================================
    # install
    # =========================================================================

    async def install(self) -> None:
        cfg: CodexConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox
        self._is_windows = not sandbox.is_linux

        # 1. Ensure node/npm are on PATH (on Windows node ships off PATH;
        # ensure_npm fixes it and also puts the npm-global bin dir on PATH so a
        # baked global ``codex`` resolves via shutil.which).
        from ale_run.agents._bootstrap import ensure_npm
        self._npm_path = await ensure_npm()

        # Ensure the running codex is exactly the pinned fork build
        # (cfg.fork_version), comparing `codex --version`:
        #   * not on PATH        -> npm install stock, then overlay the fork
        #   * present, wrong ver -> overlay the fork (download)  [e.g. a stale
        #                           baked fork, or stock]
        #   * present, matches   -> already current, skip the GitHub download
        #                           (matters at concurrency × 135 tasks)
        # We compare the FULL version string (not a 0.0.0 sentinel) so an old
        # fork build is correctly seen as stale and replaced — there is no
        # silent fall back to stock/old.
        patched_url = (
            cfg.patched_binary_url_windows if self._is_windows
            else cfg.patched_binary_url
        )
        codex_path = shutil.which("codex")
        already_current = bool(codex_path) and await self._version_matches(
            codex_path, cfg.fork_version
        )
        need_overlay = True
        if not codex_path:
            logger.info(
                "codex: not found on PATH, installing @openai/codex@%s via npm ...",
                cfg.codex_version,
            )
            await self._npm_install_codex(cfg.codex_version)
            codex_path = shutil.which("codex")
            if not codex_path:
                raise RuntimeError(
                    "CodexDeployer: 'codex' still not found after "
                    f"npm install -g @openai/codex@{cfg.codex_version}"
                )
        elif already_current:
            logger.info(
                "codex: %s already at pinned %s, skipping overlay",
                codex_path, cfg.fork_version,
            )
            need_overlay = False
        else:
            logger.info(
                "codex: %s present but not at pinned %s — overlaying fork",
                codex_path, cfg.fork_version,
            )
        self._codex_path = codex_path

        # 2. Overlay the fork native binary when needed (missing/stale/stock).
        if need_overlay:
            if not patched_url:
                raise RuntimeError(
                    "codex: need the fork binary but patched_binary_url is empty "
                    f"(running codex is not pinned {cfg.fork_version})"
                )
            await self._replace_native_binary(patched_url)

        # 3. Verify codex --version is now the pinned fork (ensure latest or fail
        # loudly — never silently run a stale build).
        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [codex_path, "--version"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"codex --version timed out: {e}")
        version_out = (probe.stdout or "").strip()
        logger.info("codex: CLI ok -- %s", version_out)
        if need_overlay and cfg.fork_version not in version_out:
            raise RuntimeError(
                f"codex: overlay did not yield pinned {cfg.fork_version!r} "
                f"(got {version_out!r}); refusing to run a stale build"
            )

        # 4. Prepare work directory
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # 4b. Ensure the cua MCP bridge is installed at sandbox.mcp_server_dir
        # (idempotent: no-op when prebaked, install when missing).
        from ale_run.agents._bootstrap import ensure_cua_mcp_server
        await ensure_cua_mcp_server(sandbox)

        # The ensure fast-path keeps prebaked bridges. Refresh the source so
        # the Unicode text-input policy is present on older images too.
        bridge_source = (
            Path(__file__).resolve().parents[1] / "_assets/cua_mcp_server/src"
        )
        bridge_target = Path(sandbox.mcp_server_dir) / "src"
        shutil.copytree(bridge_source, bridge_target, dirs_exist_ok=True)

        # 5. Write MCP config (config.toml) for CUA bridge
        await self._write_codex_config(cfg)

    async def _version_matches(self, codex_path: str, version: str) -> bool:
        """True if ``codex --version`` reports the pinned version string."""
        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [codex_path, "--version"],
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
        return version in (probe.stdout or "")

    async def _npm_install_codex(self, version: str) -> None:
        """Install Codex CLI globally via npm."""
        npm = getattr(self, "_npm_path", None) or shutil.which("npm") or "npm"
        pkg = f"@openai/codex@{version}"
        proc = await asyncio.to_thread(
            subprocess.run,
            [npm, "install", "-g", "--force", pkg],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"npm install -g {pkg} failed (rc={proc.returncode}): "
                f"{(proc.stderr or '')[:500]}"
            )
        logger.info("codex: installed %s via npm", pkg)

        # Ensure npm bin dir is on PATH
        sep = ";" if getattr(self, "_is_windows", False) else ":"
        npm_prefix_proc = await asyncio.to_thread(
            subprocess.run,
            [npm, "prefix", "-g"],
            capture_output=True, text=True, timeout=15,
        )
        if npm_prefix_proc.returncode == 0:
            prefix = npm_prefix_proc.stdout.strip()
            # Windows drops global shims directly in the prefix; Linux uses bin/
            npm_bin = prefix if getattr(self, "_is_windows", False) else os.path.join(prefix, "bin")
            if npm_bin and npm_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{npm_bin}{sep}{os.environ.get('PATH', '')}"

    async def _replace_native_binary(self, url: str) -> None:
        """Download a patched binary from URL and replace the vendor copy.

        Tries both the top-level and nested npm vendor paths. On Linux,
        vendor dirs are typically root-owned, so we stage to /tmp and
        use sudo -n mv if needed.
        """
        # Single source of OS truth: the sandbox flag set in install() (the
        # deployer runs in-VM, so this matches the running platform).
        is_linux = not self._is_windows
        if is_linux:
            vendor_paths = [_VENDOR_BINARY_LINUX_TOPLEVEL, _VENDOR_BINARY_LINUX_NESTED]
        else:
            vendor_paths = [_VENDOR_BINARY_WIN_TOPLEVEL, _VENDOR_BINARY_WIN_NESTED]

        # Download the patched binary to a temp location. mkstemp (not the
        # deprecated mktemp) creates the file atomically with a private name; we
        # close the fd and let curl -o overwrite the path.
        fd, staged = tempfile.mkstemp(prefix="codex-patched-", suffix=".bin")
        os.close(fd)
        try:
            dl = await asyncio.to_thread(
                subprocess.run,
                ["curl", "-fsSL", "-o", staged, url],
                capture_output=True, text=True, timeout=600,
            )
            if dl.returncode != 0:
                logger.warning(
                    "codex: failed to download patched binary from %s (rc=%d): %s",
                    url, dl.returncode, (dl.stderr or "")[:300],
                )
                return
            if not is_linux:
                # Windows: user-owned npm vendor dirs, no sudo/chmod needed.
                replaced = 0
                for vp in vendor_paths:
                    if not os.path.isfile(vp):
                        logger.info("codex: vendor path not present, skipping: %s", vp)
                        continue
                    try:
                        shutil.copyfile(staged, vp)
                        logger.info("codex: replaced vendor binary at %s", vp)
                        replaced += 1
                    except OSError as exc:
                        logger.warning("codex: could not replace %s: %s", vp, exc)
                if replaced == 0:
                    logger.warning("codex: no vendor binaries replaced (Windows)")
                return
            # Make executable
            os.chmod(staged, 0o755)

            replaced = 0
            for vp in vendor_paths:
                if not os.path.isfile(vp):
                    logger.info("codex: vendor path not present, skipping: %s", vp)
                    continue
                try:
                    # Try direct copy first
                    proc = await asyncio.to_thread(
                        subprocess.run,
                        ["cp", "-f", staged, vp],
                        capture_output=True, text=True, timeout=30,
                    )
                    if proc.returncode != 0:
                        # Fall back to sudo
                        proc = await asyncio.to_thread(
                            subprocess.run,
                            ["sudo", "-n", "cp", "-f", staged, vp],
                            capture_output=True, text=True, timeout=30,
                        )
                    if proc.returncode == 0:
                        # Ensure executable
                        await asyncio.to_thread(
                            subprocess.run,
                            ["chmod", "+x", vp],
                            capture_output=True, timeout=10,
                        )
                        logger.info("codex: replaced vendor binary at %s", vp)
                        replaced += 1
                    else:
                        logger.warning(
                            "codex: could not replace %s (rc=%d): %s",
                            vp, proc.returncode, (proc.stderr or "")[:200],
                        )
                except Exception as exc:
                    logger.warning("codex: error replacing %s: %s", vp, exc)

            if replaced == 0:
                logger.warning(
                    "codex: no vendor binaries were replaced -- "
                    "has npm install -g @openai/codex run?"
                )
        finally:
            try:
                os.unlink(staged)
            except OSError:
                pass

    async def _write_codex_config(
        self,
        cfg: CodexConfig,
        *,
        otel_endpoint: str | None = None,
    ) -> None:
        """Write ~/.codex/config.toml with MCP server + provider config."""
        sandbox = self.executor.sandbox

        node_exe = sandbox.node
        mcp_entry = self._join(
            sandbox.mcp_server_dir, "src", "index.js",
            is_linux=sandbox.is_linux,
        )
        # TOML basic strings interpret backslash escapes (\\U, \\n, ...), so a
        # raw Windows path like C:\Users\User\node...\node.exe breaks the
        # parser. node + Node's require() accept forward slashes on Windows,
        # so normalise to '/' to keep the TOML valid.
        if not sandbox.is_linux:
            node_exe = node_exe.replace("\\", "/")
            mcp_entry = mcp_entry.replace("\\", "/")

        # Build TOML content.
        # Top-level keys MUST appear before any [table] header in TOML.
        preamble = f'model_reasoning_effort = "{cfg.reasoning_effort}"\n'

        # Provider-driven routing (explicit, not model-name heuristic).
        is_openrouter = (cfg.provider == "openrouter")
        if is_openrouter:
            preamble += 'model_provider = "openrouter"\n'

        # Model catalog (models not in codex's bundled catalog). Write the shipped catalog
        # content to ~/.codex/model_catalog.json and point config.toml at it via
        # ``model_catalog_json`` (must be an absolute path — codex parses it as
        # AbsolutePathBuf). The content travelled here in the serialized config
        # (cfg.model_catalog_content), already sanitised host-side.
        home = os.path.expanduser("~")
        codex_config_dir = os.path.join(home, ".codex")
        os.makedirs(codex_config_dir, exist_ok=True)
        if cfg.model_catalog_content:
            catalog_path = os.path.join(codex_config_dir, "model_catalog.json")
            Path(catalog_path).write_text(cfg.model_catalog_content, encoding="utf-8")
            toml_catalog_path = (
                catalog_path.replace("\\", "/") if not sandbox.is_linux
                else catalog_path
            )
            preamble += f'model_catalog_json = "{toml_catalog_path}"\n'
            logger.info("codex: model catalog written to %s", catalog_path)

        config_toml = preamble + "\n"

        # MCP server config for CUA bridge. CUA_SERVER_URL points the bridge at
        # this image's cua-server port (it otherwise defaults to 5000, wrong on
        # ale-kasm which runs on 8000). URL is host:port only — no backslashes,
        # safe in a TOML basic string.
        cua_url = self.executor.cua_bridge_url()
        config_toml += (
            "[mcp_servers.cua]\n"
            'type = "stdio"\n'
            f'command = "{node_exe}"\n'
            f'args = ["{mcp_entry}"]\n'
            f'env = {{ CUA_SERVER_URL = "{cua_url}" }}\n'
        )

        # OpenRouter provider block
        if is_openrouter:
            # base_url override lets the openrouter routing path point at any
            # OpenAI-compatible gateway that speaks the Responses API (this
            # codex build dropped the chat wire), e.g. Volcengine Ark
            # (https://ark.cn-beijing.volces.com/api/v3). Default = OpenRouter.
            or_base = cfg.base_url or "https://openrouter.ai/api/v1"
            config_toml += (
                "\n[model_providers.openrouter]\n"
                'name = "openrouter"\n'
                f'base_url = "{or_base}"\n'
            )
            # Key source: a literal cfg.api_key (travels with the serialized
            # config — typically `api_key: ${env:ARK_API_KEY}` in the agent
            # yaml, so no extra env passthrough is needed) is written as the
            # provider's experimental_bearer_token; otherwise fall back to the
            # OPENROUTER_API_KEY env var.
            if cfg.api_key:
                config_toml += f'experimental_bearer_token = "{cfg.api_key}"\n'
            else:
                config_toml += 'env_key = "OPENROUTER_API_KEY"\n'

        # Feature overrides → codex [features] map (== tool surface). Each entry
        # force-enables (true) or force-disables (false) a codex feature; an
        # empty map leaves codex's defaults untouched. codex's features table is
        # a single bool map so both directions live here (e.g. enable
        # multi_agent_v2 + disable multi_agent). Keys are validated by codex at
        # load (unknown keys warn); see CodexConfig.feature_overrides and the
        # codex.yaml preset for the documented headless-meaningful keys.
        if cfg.feature_overrides:
            config_toml += "\n[features]\n"
            for key, val in cfg.feature_overrides.items():
                config_toml += f"{key} = {'true' if val else 'false'}\n"
            logger.info(
                "codex: feature overrides -> %s",
                ", ".join(f"{k}={'on' if v else 'off'}"
                          for k, v in cfg.feature_overrides.items()),
            )

        if otel_endpoint is not None:
            config_toml += (
                "\n[otel]\n"
                'environment = "ale"\n'
                "log_user_prompt = true\n"
                'trace_exporter = "none"\n'
                'metrics_exporter = "none"\n'
                "exporter = { otlp-http = { "
                f'endpoint = "{otel_endpoint}", protocol = "json"'
                " } }\n"
            )

        # Write config file (codex_config_dir created above).
        config_path = os.path.join(codex_config_dir, "config.toml")
        Path(config_path).write_text(config_toml, encoding="utf-8")
        logger.info("codex: config written to %s", config_path)

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: CodexConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        prompt_file = wd / "prompt.txt"
        transcript_file = wd / "transcript.jsonl"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "codex.pid"
        collector = CodexOtelCollector(wd) if cfg.otel_enabled else None

        for f in (transcript_file, stderr_log, pid_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        prompt_file.write_text(prompt, encoding="utf-8")

        # Codex requires being in a git repo
        git_dir = wd / ".git"
        if not git_dir.exists():
            await asyncio.to_thread(
                subprocess.run,
                ["git", "init"],
                capture_output=True, cwd=str(wd), timeout=15,
            )

        argv = self._build_argv(cfg)
        env = self._build_env(cfg)

        try:
            if collector is not None:
                collector.start()
                await self._write_codex_config(cfg, otel_endpoint=collector.endpoint)
                logger.info("codex: OTel collector listening at %s", collector.endpoint)

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
            logger.info("codex: spawned pid=%s", proc.pid)

            try:
                while proc.poll() is None:
                    await asyncio.sleep(_POLL_INTERVAL_S)
            except asyncio.CancelledError:
                self._terminate_proc_group(proc, force=False)
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(proc.wait), timeout=_TERM_GRACE_S,
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self._terminate_proc_group(proc, force=True)
                raise
        finally:
            if collector is not None:
                collector.stop()

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

    @staticmethod
    def _terminate_proc_group(proc: subprocess.Popen, *, force: bool) -> None:
        """Signal codex and everything it spawned (sub-agents, stdio MCP servers).

        launch() sets ``start_new_session=True`` on POSIX, so the child leads its
        own process group; signalling the whole group (``killpg``) reaps codex's
        children instead of orphaning them. Windows has no setsid, so we fall
        back to terminating the single child. ``force`` picks SIGKILL vs SIGTERM
        (POSIX) / ``kill`` vs ``terminate`` (Windows).
        """
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                import signal  # POSIX-only constants; safe under killpg guard
                os.killpg(
                    os.getpgid(proc.pid),
                    signal.SIGKILL if force else signal.SIGTERM,
                )
            elif force:
                proc.kill()
            else:
                proc.terminate()
        except (ProcessLookupError, OSError):
            pass

    # =========================================================================
    # internals
    # =========================================================================

    @staticmethod
    def _join(*parts: str, is_linux: bool) -> str:
        sep = "/" if is_linux else "\\"
        head = parts[0].rstrip("/\\")
        tail = sep.join(p.strip("/\\") for p in parts[1:])
        return f"{head}{sep}{tail}" if tail else head

    def _build_argv(self, cfg: CodexConfig) -> list[str]:
        """Build the codex exec command line.

        ``codex exec`` reads the prompt from stdin when no positional
        prompt is given; the caller wires the prompt file to the child's
        stdin. Building a plain argv (no shell) works identically on
        Linux and Windows (the win npm shim is ``codex.cmd``, which
        ``subprocess`` launches directly).
        """
        argv = [self._codex_path, "exec", "--model", cfg.model, "--json"]
        if cfg.yolo:
            argv += ["--dangerously-bypass-approvals-and-sandbox"]
        else:
            argv += ["--full-auto", "--sandbox", cfg.sandbox_mode]
        return argv

    def _build_env(self, cfg: CodexConfig) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v

        # Provider-driven routing (explicit, not model-name heuristic).
        if cfg.provider == "openrouter":
            # OpenRouter: needs OPENROUTER_API_KEY, clear OPENAI_API_KEY
            # to avoid confusion. When a literal cfg.api_key is supplied (e.g.
            # for an Ark base_url override) the key travels in config.toml as
            # experimental_bearer_token, so OPENROUTER_API_KEY is not required.
            or_key = env.get("OPENROUTER_API_KEY", "")
            if not or_key and not cfg.api_key:
                raise RuntimeError(
                    "codex: provider=openrouter but OPENROUTER_API_KEY is "
                    "not set. Export it or pass it via executor env before "
                    "launch()."
                )
            # Remove direct OpenAI keys to avoid routing confusion
            env.pop("OPENAI_API_KEY", None)
            env.pop("CODEX_API_KEY", None)
            env.pop("OPENAI_BASE_URL", None)
        elif cfg.provider == "direct":
            # Direct OpenAI routing
            oai_key = env.get("OPENAI_API_KEY", "")
            if not oai_key:
                raise RuntimeError(
                    "codex: provider=direct but OPENAI_API_KEY is not set. "
                    "Export it or pass it via executor env before launch()."
                )
            env["CODEX_API_KEY"] = oai_key
        else:
            raise RuntimeError(
                f"codex: unknown provider {cfg.provider!r} "
                "(expected 'openrouter' or 'direct')"
            )

        env["NO_COLOR"] = "1"
        return env

    # =========================================================================
    # parse_artifacts
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: CodexConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Parse Codex NDJSON transcript into trajectory steps.

        Codex ``--json`` outputs NDJSON with event types:
        - ``item.started``: initial item data (tool call args, command)
        - ``item.completed``: final item data (results, output)
        - ``turn.completed``: usage stats
        - ``thread.started``, ``error``, etc.
        """
        try:
            recover_telemetry_artifacts(work_dir)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("codex: could not recover telemetry artifacts: %s", exc)

        transcript_file = work_dir / "transcript.jsonl"
        if not transcript_file.exists():
            builder.add_step(
                source="system",
                message=f"codex: no transcript at {transcript_file}",
                extra={"reason": "no_transcript"},
            )
            return

        raw = transcript_file.read_text(encoding="utf-8", errors="replace")
        # Strip UTF-8 BOM if present (PowerShell on Windows may produce this)
        if raw.startswith("﻿"):
            raw = raw[1:]

        events: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"raw": line, "parse_error": True})

        # Track started items for merging with completed
        started_items: dict[str, dict] = {}
        completed_ids: set[str] = set()

        for event in events:
            etype = event.get("type", "")

            if etype == "item.started":
                item = event.get("item", {})
                item_id = item.get("id", "")
                if item_id:
                    started_items[item_id] = item
                continue

            if etype == "item.completed":
                cls._consume_item_completed(event, started_items, completed_ids, builder)
                continue

            if etype == "turn.completed":
                cls._consume_turn_completed(event, builder)
                continue

            if etype == "error":
                builder.add_step(
                    source="system",
                    message=event.get("message", str(event.get("error", ""))),
                )

        # Emit steps for items that started but never completed (timeout/kill)
        for item_id, item in started_items.items():
            if item_id in completed_ids:
                continue
            item_type = item.get("type", "")
            if item_type == "mcp_tool_call":
                builder.add_step(
                    source="agent",
                    tool_calls=[ToolCall(
                        id=item_id,
                        name=item.get("tool", ""),
                        arguments=item.get("arguments", {}),
                    )],
                    extra={"server": item.get("server", ""), "status": "incomplete"},
                )

        builder.trajectory.extra.setdefault("codex", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(transcript_file),
            "telemetry_path": str(work_dir / "telemetry.jsonl"),
            "telemetry_summary_path": str(work_dir / "telemetry_summary.json"),
        })

    @classmethod
    def _consume_item_completed(
        cls,
        event: dict,
        started_items: dict[str, dict],
        completed_ids: set[str],
        builder: TrajectoryBuilder,
    ) -> None:
        """Process an ``item.completed`` NDJSON event."""
        item = event.get("item", {})
        item_type = item.get("type", "")
        item_id = item.get("id", "")
        completed_ids.add(item_id)
        started = started_items.get(item_id, {})

        if item_type == "agent_message":
            builder.add_step(
                source="agent",
                message=item.get("text", ""),
                extra={"item_id": item_id},
            )

        elif item_type == "reasoning":
            builder.add_step(
                source="agent",
                reasoning=item.get("text", ""),
                extra={"item_id": item_id},
            )

        elif item_type == "command_execution":
            cmd = item.get("command", "") or started.get("command", "")
            output = item.get("aggregated_output", "") or started.get(
                "aggregated_output", ""
            )
            builder.add_step(
                source="agent",
                tool_calls=[ToolCall(
                    id=item_id,
                    name="shell",
                    arguments={"command": cmd},
                )],
            )
            builder.add_step(
                source="environment",
                observation=Observation(results=[
                    ToolResult(
                        tool_call_id=item_id,
                        content=[ContentPart(type="text", text=output)],
                        is_error=(item.get("exit_code") or 0) != 0,
                    ),
                ]),
                extra={
                    "exit_code": item.get("exit_code"),
                    "status": item.get("status", ""),
                },
            )

        elif item_type == "mcp_tool_call":
            builder.add_step(
                source="agent",
                tool_calls=[ToolCall(
                    id=item_id,
                    name=item.get("tool", ""),
                    arguments=item.get("arguments", {}),
                )],
                extra={
                    "server": item.get("server", ""),
                    "status": item.get("status", ""),
                },
            )
            # Extract tool result
            result_data = item.get("result")
            error_data = item.get("error")
            result_text = ""
            image_parts: list[ContentPart] = []
            if error_data:
                result_text = str(error_data)
            elif result_data:
                if isinstance(result_data, dict):
                    content_blocks = result_data.get("content", [])
                    text_chunks: list[str] = []
                    for block in (
                        content_blocks if isinstance(content_blocks, list) else []
                    ):
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_chunks.append(block.get("text", ""))
                            elif block.get("type") == "image" and block.get("data"):
                                # MCP/CUA screenshot block (flat shape):
                                # {"type":"image","data":"<base64>","mimeType":...}.
                                # Keep it so persist_screenshots() can extract it
                                # instead of collapsing to "[image]".
                                image_parts.append(ContentPart(
                                    type="image",
                                    image=ImageSource(
                                        type="base64",
                                        media_type=block.get("mimeType", "image/png"),
                                        data=block.get("data"),
                                    ),
                                ))
                    if text_chunks:
                        result_text = "\n".join(text_chunks)
                    elif not image_parts:
                        result_text = json.dumps(result_data)[:500]
                else:
                    result_text = str(result_data)[:500]
            if result_text or image_parts or item.get("status") == "completed":
                tr_content: list[ContentPart] = []
                if result_text:
                    tr_content.append(ContentPart(type="text", text=result_text))
                tr_content.extend(image_parts)
                builder.add_step(
                    source="environment",
                    observation=Observation(results=[
                        ToolResult(
                            tool_call_id=item_id,
                            content=tr_content,
                            is_error=bool(error_data),
                        ),
                    ]),
                )

        elif item_type == "file_change":
            builder.add_step(
                source="environment",
                message="[file_change]",
                extra={
                    "item_id": item_id,
                    "changes": item.get("changes", []),
                    "status": item.get("status", ""),
                },
            )

        elif item_type == "web_search":
            builder.add_step(
                source="agent",
                tool_calls=[ToolCall(
                    id=item_id,
                    name="web_search",
                    arguments={"query": item.get("query", "")},
                )],
                extra={"item_id": item_id},
            )

        elif item_type == "error":
            builder.add_step(
                source="system",
                message=item.get("message", ""),
                extra={"item_id": item_id},
            )

    @classmethod
    def _consume_turn_completed(
        cls,
        event: dict,
        builder: TrajectoryBuilder,
    ) -> None:
        """Extract usage from a ``turn.completed`` event and attach
        as metrics on a synthetic step."""
        usage = event.get("usage")
        if not usage:
            return

        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        cached = usage.get("cached_input_tokens")
        if cached is None:
            details = usage.get("input_tokens_details") or {}
            if isinstance(details, dict):
                cached = details.get("cached_tokens")
        cached = cached or 0

        metrics = StepMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached if cached > 0 else None,
        )
        # Attach metrics to the most recent agent step if available,
        # otherwise emit a synthetic completion step.
        steps = builder.trajectory.steps
        if steps and steps[-1].source == "agent" and steps[-1].metrics is None:
            steps[-1].metrics = metrics
        else:
            builder.add_step(
                source="agent",
                message="[turn.completed]",
                metrics=metrics,
                extra={"codex_turn_usage": usage},
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
