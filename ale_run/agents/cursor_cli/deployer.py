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
    ImageSource,
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

    _PINNED_VERSION: ClassVar[str] = "2026.05.28-a70ca7c"

    @property
    def version(self) -> str | None:
        cfg: CursorCliConfig = self.config  # type: ignore[assignment]
        return cfg.cursor_version or self._PINNED_VERSION

    # =========================================================================
    # install
    # =========================================================================

    @staticmethod
    def _probe_version(cursor_path: str) -> str:
        """Return the ``--version`` string of a cursor-agent binary ("" on error)."""
        try:
            probe = subprocess.run(
                [cursor_path, "--version"],
                capture_output=True, text=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError):
            return ""
        # `--version` prints just the version (e.g. "2026.05.28-a70ca7c").
        return (probe.stdout or "").strip().splitlines()[0].strip() if probe.stdout else ""

    @staticmethod
    def _download_url(version: str) -> str:
        machine = platform.machine().lower()
        arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
        sysname = platform.system()
        if sysname == "Windows":
            # Windows ships a .zip (no tar.gz published for win/x64).
            return (
                f"https://downloads.cursor.com/lab/{version}/windows/{arch}/"
                "agent-cli-package.zip"
            )
        osname = "darwin" if sysname == "Darwin" else "linux"
        return (
            f"https://downloads.cursor.com/lab/{version}/{osname}/{arch}/"
            "agent-cli-package.tar.gz"
        )

    async def _install_pinned(self, version: str) -> None:
        """Install an exact cursor-agent version from downloads.cursor.com.

        Mirrors the official installer layout: untar to
        ``~/.local/share/cursor-agent/versions/<version>/`` and symlink
        ``~/.local/bin/cursor-agent`` (and ``agent``) at it. Overrides any
        mismatched pre-baked copy. Falls back to ``cursor.com/install``
        (latest-only) only if the versioned download fails.
        """
        home = os.path.expanduser("~")
        version_dir = f"{home}/.local/share/cursor-agent/versions/{version}"
        bin_dir = f"{home}/.local/bin"
        url = self._download_url(version)
        # Atomic-ish: download+extract into the version dir, then symlink.
        script = (
            f'set -e; mkdir -p "{version_dir}" "{bin_dir}"; '
            f'curl -fSL -s "{url}" | tar --strip-components=1 -xzf - -C "{version_dir}"; '
            f'rm -f "{bin_dir}/cursor-agent" "{bin_dir}/agent"; '
            f'ln -s "{version_dir}/cursor-agent" "{bin_dir}/cursor-agent"; '
            f'ln -s "{version_dir}/cursor-agent" "{bin_dir}/agent"'
        )
        proc = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", script],
            capture_output=True, text=True, timeout=180,
        )
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        if proc.returncode != 0:
            logger.warning(
                "cursor_cli: pinned download of %s failed (rc=%d): %s — "
                "falling back to cursor.com/install (latest)",
                version, proc.returncode, (proc.stderr or "")[:300],
            )
            await self._auto_install_cli()
            return
        logger.info("cursor_cli: installed pinned version %s from %s", version, url)

    def _install_pinned_windows_sync(self, version: str) -> None:
        """Install an exact cursor-agent version on Windows.

        Downloads the published ``agent-cli-package.zip`` and extracts the
        ``dist-package/`` payload (which co-locates ``node.exe`` + ``index.js``)
        into a versioned dir. Launch invokes ``node.exe index.js`` directly —
        the same path the bundled ``cursor-agent.ps1`` takes when node.exe sits
        next to the script. Uses urllib + zipfile (no curl/tar/unzip
        dependency).
        """
        import urllib.request
        import zipfile

        home = os.path.expanduser("~")
        version_dir = Path(home) / ".local" / "share" / "cursor-agent" / "versions" / version
        if version_dir.exists():
            shutil.rmtree(version_dir, ignore_errors=True)
        version_dir.mkdir(parents=True, exist_ok=True)
        url = self._download_url(version)
        zip_path = version_dir.parent / f"cursor-agent-{version}.zip"

        # downloads.cursor.com's CDN returns 403 for the default
        # "Python-urllib/x.y" User-Agent, so send a browser-style UA.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as out:
            shutil.copyfileobj(resp, out)
        with zipfile.ZipFile(str(zip_path)) as zf:
            for member in zf.namelist():
                # Strip the leading "dist-package/" prefix so the payload lands
                # directly in version_dir (node.exe + index.js side by side).
                rel = member.split("/", 1)[1] if "/" in member else member
                if not rel:
                    continue
                target = version_dir / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        try:
            zip_path.unlink()
        except OSError:
            pass

        node_exe = version_dir / "node.exe"
        index_js = version_dir / "index.js"
        if not (node_exe.is_file() and index_js.is_file()):
            raise RuntimeError(
                f"cursor_cli: Windows package missing node.exe/index.js under "
                f"{version_dir} after extracting {url}"
            )
        self._win_node = str(node_exe)
        self._win_index_js = str(index_js)
        logger.info("cursor_cli: installed pinned Windows version %s at %s", version, version_dir)

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
        self._is_windows = not sandbox.is_linux
        pinned = cfg.cursor_version or self._PINNED_VERSION

        if self._is_windows:
            # Windows runs via `node.exe index.js` from a versioned dir. Skip the
            # ~63MB download when the pinned version is already baked at the path
            # we'd extract to (node.exe + index.js present) — verify-and-skip,
            # mirroring the Linux branch. A clean/reverted or stale-version VM has
            # no matching dir, so it falls through to a fresh pinned extract.
            home = os.path.expanduser("~")
            version_dir = (
                Path(home) / ".local" / "share" / "cursor-agent" / "versions" / pinned
            )
            node_exe = version_dir / "node.exe"
            index_js = version_dir / "index.js"
            if node_exe.is_file() and index_js.is_file():
                self._win_node = str(node_exe)
                self._win_index_js = str(index_js)
                logger.info(
                    "cursor_cli: pinned Windows version %s already installed at %s — skipping",
                    pinned, version_dir,
                )
            else:
                logger.info("cursor_cli: installing pinned Windows version %s …", pinned)
                await asyncio.to_thread(self._install_pinned_windows_sync, pinned)
            self._cursor_path = self._win_index_js
        else:
            # Verify-and-correct: keep a pre-installed binary only if it already
            # matches the pinned version; otherwise install the pinned version
            # (overriding any mismatched pre-baked copy).
            cursor_path = shutil.which("cursor-agent")
            current = self._probe_version(cursor_path) if cursor_path else ""
            if cursor_path and current == pinned:
                logger.info("cursor_cli: pinned version %s already installed — skipping", pinned)
            else:
                if cursor_path:
                    logger.info(
                        "cursor_cli: installed version %r != pinned %s — reinstalling pinned",
                        current or "unknown", pinned,
                    )
                else:
                    logger.info("cursor_cli: 'cursor-agent' not on PATH, installing pinned %s …", pinned)
                await self._install_pinned(pinned)
                cursor_path = shutil.which("cursor-agent")
                if not cursor_path:
                    raise RuntimeError(
                        "CursorCliDeployer: 'cursor-agent' still not found after install"
                    )
            self._cursor_path = cursor_path
            final = self._probe_version(cursor_path)
            logger.info("cursor_cli: CLI ok — %s (pinned target %s)", final or "unknown", pinned)

        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        home = os.path.expanduser("~")
        cursor_home = Path(home) / ".cursor"
        cursor_home.mkdir(parents=True, exist_ok=True)

        # Ensure the cua MCP bridge is installed at sandbox.mcp_server_dir
        # (idempotent: no-op when prebaked, install when missing).
        from ale_run.agents._bootstrap import cua_bridge_env, ensure_cua_mcp_server
        await ensure_cua_mcp_server(sandbox)

        # MCP config (auto-discovered, no CLI flag)
        mcp_config = {
            "mcpServers": {
                "cua": {
                    "command": sandbox.node,
                    "args": [self._join(sandbox.mcp_server_dir, "src", "index.js",
                                        is_linux=sandbox.is_linux)],
                    "env": cua_bridge_env(self.executor),
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

        if getattr(self, "_is_windows", False):
            # Launch via node.exe index.js directly (matches the bundled
            # cursor-agent.ps1's node-co-located branch) — avoids the
            # cmd→powershell→node shim chain that mangles stdin/args headless.
            argv = [self._win_node, self._win_index_js, "-p"]
        else:
            argv = [self._cursor_path, "-p"]
        # Empty/None model => omit --model so cursor-agent picks "auto".
        # Tier/variant (e.g. composer-2.5 Standard vs composer-2.5-fast) is
        # part of the model id, passed through verbatim.
        if cfg.model:
            argv += ["--model", cfg.model]
        argv += [
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
        if getattr(self, "_is_windows", False):
            # cursor-agent on Windows reads %APPDATA%\Cursor\auth.json.
            appdata = os.environ.get("APPDATA") or str(Path(home) / "AppData" / "Roaming")
            auth_dir = Path(appdata) / "Cursor"
        else:
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
            auth_path = Path(auth_path_str).expanduser()
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
        config: CursorCliConfig,
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
        elif etype == "tool_call":
            cls._consume_tool_call(event, builder)
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
                        if not isinstance(c, dict):
                            continue
                        ctype = c.get("type")
                        if ctype == "text":
                            parts.append(ContentPart(type="text", text=c.get("text", "")))
                        elif ctype == "image":
                            # cursor uses the same stream-json/Anthropic shape as
                            # claude_code: {"type":"image","source":{"type":
                            # "base64"|"url",...}}. Keep MCP/CUA screenshots so
                            # persist_screenshots() can write them out.
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

    @classmethod
    def _consume_tool_call(cls, event: dict, builder: TrajectoryBuilder) -> None:
        """Composer-style tool calls.

        Shape: ``{"type":"tool_call","subtype":"started"|"completed",
        "call_id":"…","tool_call":{"<name>ToolCall":{"args":{…},
        "result":{…}}}}``. The inner dict is keyed by the tool name
        (``editToolCall``, ``readToolCall``, ``shellToolCall``, …). On
        ``started`` we emit an agent ToolCall; on ``completed`` an
        environment ToolResult.
        """
        subtype = event.get("subtype")
        call_id = event.get("call_id") or ""
        name, body = cls._unwrap_tool_call(event.get("tool_call"))
        if name is None:
            return

        if subtype == "started":
            builder.add_step(
                source="agent",
                tool_calls=[ToolCall(
                    id=call_id,
                    name=name,
                    arguments=body.get("args") or {},
                )],
            )
        elif subtype == "completed":
            text, is_error, image_parts = cls._render_tool_result(
                body.get("result")
            )
            content: list[ContentPart] = []
            if text or not image_parts:
                content.append(ContentPart(type="text", text=text))
            content.extend(image_parts)
            builder.add_step(
                source="environment",
                observation=Observation(results=[ToolResult(
                    tool_call_id=call_id,
                    content=content,
                    is_error=is_error,
                )]),
            )

    @staticmethod
    def _unwrap_tool_call(tool_call: object) -> tuple[str | None, dict]:
        """Return ``(tool_name, body)`` from the ``tool_call`` wrapper.

        Picks the first dict-valued entry whose body carries ``args`` or
        ``result`` (skips sibling scalar keys like ``description``).
        """
        if not isinstance(tool_call, dict):
            return None, {}
        for key, val in tool_call.items():
            if isinstance(val, dict) and ("args" in val or "result" in val):
                return key, val
        # Fallback: first dict-valued entry.
        for key, val in tool_call.items():
            if isinstance(val, dict):
                return key, val
        return None, {}

    @staticmethod
    def _render_tool_result(result: object) -> tuple[str, bool, list[ContentPart]]:
        """Flatten a Composer tool ``result`` into ``(text, is_error, images)``.

        Variants observed: ``success`` (ok), ``error`` (``errorMessage``),
        ``failure`` (shell non-zero exit, has ``stderr``/``exitCode``).

        A successful MCP tool (e.g. the cua ``screenshot`` tool) returns
        ``success.content`` as a list of blocks — ``{"text":{"text":…}}`` and
        ``{"image":{"data":"<base64>","mimeType":…}}``. Without pulling the
        image blocks out, the whole base64 was JSON-dumped into a text part and
        persist_screenshots() never saw a screenshot.
        """
        image_parts: list[ContentPart] = []
        if not isinstance(result, dict):
            return ("" if result is None else json.dumps(result)), False, image_parts
        if "error" in result and isinstance(result["error"], dict):
            return (
                result["error"].get("errorMessage") or json.dumps(result["error"]),
                True,
                image_parts,
            )
        if "failure" in result and isinstance(result["failure"], dict):
            return json.dumps(result["failure"]), True, image_parts
        if "success" in result:
            succ = result["success"]
            if isinstance(succ, dict) and isinstance(succ.get("content"), list):
                text_chunks: list[str] = []
                for c in succ["content"]:
                    if not isinstance(c, dict):
                        continue
                    txt = c.get("text")
                    if isinstance(txt, dict):
                        text_chunks.append(str(txt.get("text", "")))
                    elif isinstance(txt, str):
                        text_chunks.append(txt)
                    img = c.get("image")
                    if isinstance(img, dict) and img.get("data"):
                        image_parts.append(ContentPart(
                            type="image",
                            image=ImageSource(
                                type="base64",
                                media_type=img.get("mimeType")
                                or img.get("mime_type")
                                or "image/png",
                                data=img.get("data"),
                            ),
                        ))
                return (
                    "\n".join(t for t in text_chunks if t),
                    False,
                    image_parts,
                )
            return (
                json.dumps(succ) if not isinstance(succ, str) else succ,
                False,
                image_parts,
            )
        return json.dumps(result), False, image_parts

    @staticmethod
    def _consume_result(event: dict, builder: TrajectoryBuilder) -> None:
        usage = event.get("usage", {})
        builder.trajectory.extra.setdefault("cursor_cli", {})["result"] = event
        if usage:
            builder.trajectory.extra["cursor_cli"]["usage"] = usage
            # cursor-agent reports the CUMULATIVE token usage on the final
            # `result` event (per-step assistant messages carry none), so add a
            # step carrying it into StepMetrics — otherwise the trajectory's
            # final_metrics (summed from per-step metrics) stays 0. cursor-agent
            # does NOT surface a dollar cost (Cursor's own backend prices it
            # internally), so cost_usd is left unset.
            builder.add_step(
                source="system",
                message=None,
                metrics=StepMetrics(
                    input_tokens=usage.get("inputTokens") or usage.get("input_tokens"),
                    output_tokens=usage.get("outputTokens") or usage.get("output_tokens"),
                    cache_read_tokens=usage.get("cacheReadTokens") or usage.get("cache_read_input_tokens"),
                    cache_creation_tokens=usage.get("cacheWriteTokens") or usage.get("cache_creation_input_tokens"),
                ),
                extra={"usage_reconciliation": True},
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
