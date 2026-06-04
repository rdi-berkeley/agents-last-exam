"""CodexDeployer — drives the OpenAI ``codex`` CLI (v0.114.0).

Installed via ``npm install -g @openai/codex@<version>``.  An optional
patched native binary (downloaded from a GitHub Release URL) replaces
the npm-installed vendor binary to fix the Windows ``apply_patch``
corruption bug.

OpenRouter routing: ``OPENROUTER_API_KEY`` + ``config.toml`` with
``model_provider = "openrouter"`` and a custom model_providers block.

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

from .config import CodexConfig

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
# Windows equivalents (not used in sandbox/docker — kept for reference).
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
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    _PINNED_VERSION: ClassVar[str] = "0.114.0"

    @property
    def version(self) -> str | None:
        return self._PINNED_VERSION

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

        # Skip-install policy (Option B): if a ``codex`` is already present
        # (baked into the image) USE IT, regardless of its version string. Both
        # the ale-ubuntu22 and ale-win10 images bake the *fork* build, which
        # reports ``codex-cli 0.0.0`` — requiring the npm-pinned semver would
        # force a needless reinstall + GitHub re-download on every single run.
        # We only treat a baked binary as good enough to skip the patched-binary
        # overlay when it IS the fork; a stock build (or anything not the fork)
        # still gets the fork overlaid below, so the running engine is always
        # the fork — we never silently fall back to stock.
        codex_path = shutil.which("codex")
        baked_is_fork = bool(codex_path) and await self._is_fork_build(codex_path)
        if codex_path:
            logger.info(
                "codex: using pre-installed binary %s (fork=%s), skipping npm install",
                codex_path, baked_is_fork,
            )
        else:
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
        self._codex_path = codex_path

        # 2. Overlay the patched fork native binary UNLESS we already have the
        # fork. A baked fork build needs no overlay (this skips a per-run GitHub
        # download — important at concurrency × 135 tasks); a freshly
        # npm-installed stock build (or a non-fork baked one) gets the fork
        # overlaid so the engine is always the fork. The Linux and Windows
        # builds are distinct release assets, so pick the URL matching the OS.
        if baked_is_fork:
            logger.info("codex: baked fork build detected, skipping patched-binary overlay")
        else:
            patched_url = (
                cfg.patched_binary_url_windows if self._is_windows
                else cfg.patched_binary_url
            )
            if patched_url:
                await self._replace_native_binary(patched_url)

        # 3. Verify codex --version
        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [codex_path, "--version"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"codex --version timed out: {e}")
        logger.info("codex: CLI ok -- %s", (probe.stdout or "").strip())

        # 4. Prepare work directory
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # 4b. Ensure the cua MCP bridge is installed at sandbox.mcp_server_dir
        # (idempotent: no-op when prebaked, install when missing).
        from ale_run.agents._bootstrap import ensure_cua_mcp_server
        await ensure_cua_mcp_server(sandbox)

        # 5. Write MCP config (config.toml) for CUA bridge
        await self._write_codex_config(cfg)

    # The cua-verse fork carries no real semver and reports ``codex-cli
    # 0.0.0``; any stock npm release reports a normal version (e.g.
    # ``0.114.0``). We use that sentinel to recognise a baked fork build.
    _FORK_VERSION_MARKER: ClassVar[str] = "0.0.0"

    async def _is_fork_build(self, codex_path: str) -> bool:
        """True if the codex at ``codex_path`` is our cua-verse fork build.

        Used to decide whether the patched-binary overlay can be skipped (the
        baked binary already IS the fork) — see :meth:`install` step 2.
        """
        return await self._version_matches(codex_path, self._FORK_VERSION_MARKER)

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
        is_linux = platform.system() == "Linux"
        if is_linux:
            vendor_paths = [_VENDOR_BINARY_LINUX_TOPLEVEL, _VENDOR_BINARY_LINUX_NESTED]
        else:
            vendor_paths = [_VENDOR_BINARY_WIN_TOPLEVEL, _VENDOR_BINARY_WIN_NESTED]

        # Download the patched binary to a temp location
        import tempfile
        staged = tempfile.mktemp(prefix="codex-patched-", suffix=".bin")
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

    async def _write_codex_config(self, cfg: CodexConfig) -> None:
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
            config_toml += (
                "\n[model_providers.openrouter]\n"
                'name = "openrouter"\n'
                'base_url = "https://openrouter.ai/api/v1"\n'
                'env_key = "OPENROUTER_API_KEY"\n'
            )

        # Write config file
        home = os.path.expanduser("~")
        codex_config_dir = os.path.join(home, ".codex")
        os.makedirs(codex_config_dir, exist_ok=True)
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
            # to avoid confusion
            or_key = env.get("OPENROUTER_API_KEY", "")
            if not or_key:
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
            if error_data:
                result_text = str(error_data)
            elif result_data:
                if isinstance(result_data, dict):
                    content_blocks = result_data.get("content", [])
                    parts = []
                    for block in (
                        content_blocks if isinstance(content_blocks, list) else []
                    ):
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif block.get("type") == "image":
                                parts.append("[image]")
                    result_text = (
                        "\n".join(parts) if parts else json.dumps(result_data)[:500]
                    )
                else:
                    result_text = str(result_data)[:500]
            if result_text or item.get("status") == "completed":
                builder.add_step(
                    source="environment",
                    observation=Observation(results=[
                        ToolResult(
                            tool_call_id=item_id,
                            content=[ContentPart(type="text", text=result_text)],
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
