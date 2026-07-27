"""Run the official xAI Grok Build CLI inside an ALE sandbox."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentDeployer,
    ContentPart,
    ImageSource,
    Observation,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

from .config import GrokBuildConfig
from .telemetry import recover_telemetry_artifacts

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0
_MAX_MCP_SPILL_BYTES = 32 * 1024 * 1024
_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")
_MIN_NODE_VERSION = (20, 0, 0)
_DATA_URL_RE = re.compile(r"data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=_-]+)")
_CUSTOM_MODEL_ALIAS = "ale-custom"
_SANDBOX_PROFILE = "off"
_HEADLESS_DISABLED_TOOLS = (
    "ask_user_question",
    "enter_plan_mode",
    "exit_plan_mode",
)
_SESSION_EXPORTS = {
    "chat_history.jsonl": "session_chat_history.jsonl",
    "updates.jsonl": "session_updates.jsonl",
    "events.jsonl": "session_events.jsonl",
    "summary.json": "session_summary.json",
}
_SESSION_MEDIA_EXPORT = "session_media.jsonl"


def _expected_version(npm_spec: str | None) -> str | None:
    if not npm_spec:
        return None
    match = _VERSION_RE.search(npm_spec.rsplit("@", 1)[-1])
    return match.group(1) if match else None


def _node_version(node_path: str) -> tuple[int, int, int] | None:
    try:
        probe = subprocess.run(
            [node_path, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = _VERSION_RE.search((probe.stdout or "") + (probe.stderr or ""))
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))  # type: ignore[return-value]


def _installed_package_version(npm_path: str, prefix: Path) -> str | None:
    try:
        probe = subprocess.run(
            [npm_path, "root", "--global", "--prefix", str(prefix)],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            return None
        package_json = Path(probe.stdout.strip()) / "@xai-official" / "grok" / "package.json"
        metadata = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    version = metadata.get("version")
    return version if isinstance(version, str) else None


def _native_binary(grok_install_home: Path, expected: str | None) -> Path | None:
    bin_dir = grok_install_home / "bin"
    candidates = [
        bin_dir / ("grok.exe" if os.name == "nt" else "grok"),
    ]
    if expected:
        candidates.append(bin_dir / f"grok-{expected}{'.exe' if os.name == 'nt' else ''}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _installed_binary_version(grok_path: Path) -> str | None:
    try:
        probe = subprocess.run(
            [str(grok_path), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = _VERSION_RE.search((probe.stdout or "") + (probe.stderr or ""))
    return match.group(1) if match else None


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _grok_cwd(work_dir: Path, *, is_linux: bool) -> Path:
    if is_linux:
        return work_dir
    digest = hashlib.sha256(str(work_dir).encode("utf-8")).hexdigest()[:12]
    return Path.home() / ".ale-grok-build" / "cwd" / digest


class GrokBuildDeployer(BaseAgentDeployer):
    """Sandbox deployer for ``@xai-official/grok``."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = (
        "transcript.jsonl",
        "stderr.log",
        "session_chat_history.jsonl",
        "session_updates.jsonl",
        "session_events.jsonl",
        "session_summary.json",
        "session_media.jsonl",
    )

    @property
    def version(self) -> str | None:
        config: GrokBuildConfig = self.config  # type: ignore[assignment]
        return config.cli_version

    async def install(self) -> None:
        config: GrokBuildConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        from ale_run.agents._bootstrap import ensure_cua_mcp_server, ensure_node_npm

        node_path, npm_path = await ensure_node_npm()
        installed_node = await asyncio.to_thread(_node_version, node_path)
        if installed_node is None or installed_node < _MIN_NODE_VERSION:
            rendered = ".".join(str(part) for part in installed_node or ())
            raise RuntimeError(
                "grok_build: Grok Build requires Node.js >=20, "
                f"found {rendered or 'an unreadable version'} at {node_path}"
            )

        work_dir = Path(self.executor.work_dir)
        grok_home = work_dir / "grok-home"
        grok_home.mkdir(parents=True, exist_ok=True)

        install_root = Path(os.path.expanduser("~")) / ".ale-grok-build"
        npm_prefix = install_root / "npm"
        grok_install_home = install_root / "home"
        npm_prefix.mkdir(parents=True, exist_ok=True)
        grok_install_home.mkdir(parents=True, exist_ok=True)

        expected = _expected_version(config.cli_version)
        installed_package = await asyncio.to_thread(
            _installed_package_version,
            npm_path,
            npm_prefix,
        )
        grok_path = _native_binary(grok_install_home, expected)
        installed_binary = (
            await asyncio.to_thread(_installed_binary_version, grok_path)
            if grok_path is not None
            else None
        )
        if grok_path is None or (
            expected is not None and (installed_package != expected or installed_binary != expected)
        ):
            logger.info(
                "grok_build: installing %s (package=%s binary=%s expected=%s)",
                config.cli_version,
                installed_package or "missing",
                installed_binary or "missing",
                expected or "unpinned",
            )
            npm_env = {
                **os.environ,
                "GROK_HOME": str(grok_install_home),
                "npm_config_cache": str(install_root / "npm-cache"),
            }
            process = await asyncio.to_thread(
                subprocess.run,
                [
                    npm_path,
                    "install",
                    "-g",
                    "--force",
                    "--prefix",
                    str(npm_prefix),
                    config.cli_version,
                ],
                capture_output=True,
                text=True,
                timeout=1200,
                env=npm_env,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    "grok_build: npm install failed "
                    f"(rc={process.returncode}, package={config.cli_version}): "
                    f"{(process.stderr or process.stdout or '')[-800:]}"
                )
            installed_package = await asyncio.to_thread(
                _installed_package_version,
                npm_path,
                npm_prefix,
            )
            grok_path = _native_binary(grok_install_home, expected)
            installed_binary = (
                await asyncio.to_thread(_installed_binary_version, grok_path)
                if grok_path is not None
                else None
            )

        if grok_path is None:
            raise RuntimeError(
                f"grok_build: official binary missing under {grok_install_home / 'bin'}"
            )
        if expected is not None and (installed_package != expected or installed_binary != expected):
            raise RuntimeError(
                "grok_build: post-install version mismatch "
                f"(package={installed_package!r}, binary={installed_binary!r}, "
                f"expected={expected!r})"
            )

        self._grok_path = str(grok_path)
        self._node_path = node_path
        logger.info(
            "grok_build: CLI ready at %s (version %s, node %s)",
            grok_path,
            installed_binary or installed_package or "unknown",
            ".".join(str(part) for part in installed_node),
        )

        await ensure_cua_mcp_server(sandbox)
        self._write_config(config, work_dir=work_dir)

    async def launch(self, prompt: str) -> AgentRunResult:
        config: GrokBuildConfig = self.config  # type: ignore[assignment]
        work_dir = Path(self.executor.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        prompt_file = work_dir / "prompt.txt"
        transcript_file = work_dir / "transcript.jsonl"
        stderr_file = work_dir / "stderr.log"
        pid_file = work_dir / "grok.pid"
        stale_paths = [
            transcript_file,
            stderr_file,
            pid_file,
            *(work_dir / export_name for export_name in _SESSION_EXPORTS.values()),
            work_dir / _SESSION_MEDIA_EXPORT,
        ]
        for path in stale_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        prompt_file.write_text(prompt, encoding="utf-8")

        env = self._build_env(config, work_dir=work_dir)
        grok_cwd = _grok_cwd(
            work_dir,
            is_linux=self.executor.sandbox.is_linux,
        )
        grok_cwd.mkdir(parents=True, exist_ok=True)
        argv = self._build_argv(config, grok_cwd=grok_cwd, prompt_file=prompt_file)
        started = time.monotonic()
        stdout = await asyncio.to_thread(transcript_file.open, "wb")
        stderr = await asyncio.to_thread(stderr_file.open, "wb")
        try:
            process = await asyncio.to_thread(
                subprocess.Popen,
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                cwd=str(grok_cwd),
                env=env,
                start_new_session=hasattr(os, "setsid"),
            )
        finally:
            await asyncio.to_thread(stdout.close)
            await asyncio.to_thread(stderr.close)
        pid_file.write_text(str(process.pid), encoding="ascii")
        logger.info("grok_build: spawned pid=%s", process.pid)

        try:
            while process.poll() is None:
                await asyncio.sleep(_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(process.wait),
                    timeout=_TERM_GRACE_S,
                )
            except (TimeoutError, asyncio.CancelledError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            raise

        await asyncio.to_thread(
            self._export_session_artifacts,
            work_dir,
            transcript_file,
        )
        duration_s = time.monotonic() - started
        status = "completed" if process.returncode == 0 else "failed"
        error = None
        if status == "failed":
            error = self._diagnose_failure(
                transcript_file=transcript_file,
                stderr_file=stderr_file,
                exit_code=process.returncode,
            )
        return AgentRunResult(
            status=status,
            pid=process.pid,
            exit_code=process.returncode,
            transcript_path=str(transcript_file),
            stderr_path=str(stderr_file),
            duration_s=duration_s,
            error=error,
        )

    def _write_config(self, config: GrokBuildConfig, *, work_dir: Path) -> None:
        sandbox = self.executor.sandbox
        grok_home = work_dir / "grok-home"
        lines = [
            "[models]",
            f"default = {_toml_string(_CUSTOM_MODEL_ALIAS if config.base_url else config.model)}",
            "",
        ]
        if config.base_url:
            lines.extend(
                [
                    f'[model."{_CUSTOM_MODEL_ALIAS}"]',
                    f"model = {_toml_string(config.model)}",
                    f"base_url = {_toml_string(config.base_url)}",
                    f"name = {_toml_string(f'ALE {config.model}')}",
                    'env_key = "ALE_GROK_BUILD_API_KEY"',
                    f"api_backend = {_toml_string(config.api_backend)}",
                ]
            )
            if config.context_window is not None:
                lines.append(f"context_window = {config.context_window}")
            if config.max_completion_tokens is not None:
                lines.append(f"max_completion_tokens = {config.max_completion_tokens}")
            lines.append("")

        lines.extend(
            [
                "[cli]",
                f"auto_update = {'false' if config.disable_auto_update else 'true'}",
                "",
                "[compat.cursor]",
                "skills = false",
                "rules = false",
                "agents = false",
                "mcps = false",
                "hooks = false",
                "",
                "[compat.claude]",
                "skills = false",
                "rules = false",
                "agents = false",
                "mcps = false",
                "hooks = false",
                "",
                "[mcp_servers.cua]",
                f"command = {_toml_string(self._node_path)}",
                (
                    "args = ["
                    + _toml_string(
                        self._join(
                            sandbox.mcp_server_dir,
                            "src",
                            "index.js",
                            is_linux=sandbox.is_linux,
                        )
                    )
                    + "]"
                ),
                ("env = { CUA_SERVER_URL = " + _toml_string(self.executor.cua_bridge_url()) + " }"),
                f"startup_timeout_sec = {config.mcp_startup_timeout_s}",
                f"tool_timeout_sec = {config.mcp_tool_timeout_s}",
                "",
            ]
        )
        (grok_home / "config.toml").write_text("\n".join(lines), encoding="utf-8")

    def _build_env(
        self,
        config: GrokBuildConfig,
        *,
        work_dir: Path,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.executor.env or {})

        api_key = config.api_key or env.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"grok_build: no API key configured; set config.api_key or {config.api_key_env}"
            )

        env.update(
            {
                "GROK_HOME": str(work_dir / "grok-home"),
                "GROK_MANAGED_BY_NPM": "1",
                # The CLI's debug log includes its resolved sampler config,
                # including API keys. Session JSONL is the trajectory source,
                # so discard this unsafe duplicate rather than gathering it.
                "GROK_LOG_FILE": os.devnull,
                "GROK_SANDBOX": _SANDBOX_PROFILE,
                "GROK_MEMORY": "0",
                "GROK_CURSOR_SKILLS_ENABLED": "0",
                "GROK_CURSOR_RULES_ENABLED": "0",
                "GROK_CURSOR_AGENTS_ENABLED": "0",
                "GROK_CURSOR_MCPS_ENABLED": "0",
                "GROK_CURSOR_HOOKS_ENABLED": "0",
                "GROK_CLAUDE_SKILLS_ENABLED": "0",
                "GROK_CLAUDE_RULES_ENABLED": "0",
                "GROK_CLAUDE_AGENTS_ENABLED": "0",
                "GROK_CLAUDE_MCPS_ENABLED": "0",
                "GROK_CLAUDE_HOOKS_ENABLED": "0",
            }
        )
        if config.disable_auto_update:
            env["GROK_DISABLE_AUTOUPDATER"] = "1"
        else:
            env.pop("GROK_DISABLE_AUTOUPDATER", None)
        if config.base_url:
            env["ALE_GROK_BUILD_API_KEY"] = api_key
        else:
            env["XAI_API_KEY"] = api_key
        return env

    def _build_argv(
        self,
        config: GrokBuildConfig,
        *,
        grok_cwd: Path,
        prompt_file: Path,
    ) -> list[str]:
        runtime_model = _CUSTOM_MODEL_ALIAS if config.base_url else config.model
        argv = [
            self._grok_path,
            "--prompt-file",
            str(prompt_file),
            "--cwd",
            str(grok_cwd),
            "--model",
            runtime_model,
            "--output-format",
            "streaming-json",
            "--sandbox",
            _SANDBOX_PROFILE,
            "--no-plan",
        ]
        if config.always_approve:
            argv.append("--always-approve")
        if config.disable_auto_update:
            argv.append("--no-auto-update")
        if config.disable_web_search:
            argv.append("--disable-web-search")
        disabled_tools = tuple(dict.fromkeys((*_HEADLESS_DISABLED_TOOLS, *config.disabled_tools)))
        argv.extend(
            [
                "--disallowed-tools",
                ",".join(disabled_tools),
            ]
        )
        if config.reasoning_effort:
            argv.extend(["--reasoning-effort", config.reasoning_effort])
        if config.max_turns is not None:
            argv.extend(["--max-turns", str(config.max_turns)])
        return argv

    @staticmethod
    def _join(*parts: str, is_linux: bool) -> str:
        separator = "/" if is_linux else "\\"
        head = parts[0].rstrip("/\\")
        tail = separator.join(part.strip("/\\") for part in parts[1:])
        return f"{head}{separator}{tail}" if tail else head

    @classmethod
    def _export_session_artifacts(
        cls,
        work_dir: Path,
        transcript_file: Path,
    ) -> None:
        transcript_events = list(cls._json_lines(transcript_file))
        terminal_event = next(
            (
                event
                for event in reversed(transcript_events)
                if event.get("type") in {"end", "error"}
            ),
            {},
        )
        session_dir = cls._find_session_dir(
            work_dir / "grok-home",
            terminal_event.get("sessionId"),
        )
        if session_dir is None:
            return

        for source_name, export_name in _SESSION_EXPORTS.items():
            source = session_dir / source_name
            destination = work_dir / export_name
            try:
                shutil.copyfile(source, destination)
            except (FileNotFoundError, OSError) as exc:
                logger.debug(
                    "grok_build: could not export %s from %s: %s",
                    source_name,
                    session_dir,
                    exc,
                )

        media_file = work_dir / _SESSION_MEDIA_EXPORT
        media_file.unlink(missing_ok=True)
        media_records: list[dict[str, str]] = []
        for spill_file in sorted((session_dir / "mcp").glob("*.txt")):
            tool_call_id = spill_file.stem
            if not tool_call_id or Path(tool_call_id).name != tool_call_id:
                continue
            try:
                if spill_file.stat().st_size > _MAX_MCP_SPILL_BYTES:
                    continue
                raw = spill_file.read_text(encoding="utf-8", errors="replace")
            except (FileNotFoundError, OSError):
                continue
            if _DATA_URL_RE.search(raw):
                media_records.append(
                    {
                        "tool_call_id": tool_call_id,
                        "content": raw,
                    }
                )
        if media_records:
            with media_file.open("w", encoding="utf-8") as handle:
                for record in media_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _session_artifact(
        work_dir: Path,
        session_dir: Path | None,
        *,
        source_name: str,
    ) -> Path | None:
        if session_dir is not None:
            session_path = session_dir / source_name
            if session_path.is_file():
                return session_path
        export_name = _SESSION_EXPORTS[source_name]
        export_path = work_dir / export_name
        return export_path if export_path.is_file() else None

    @classmethod
    def _session_media(cls, path: Path) -> dict[str, list[ContentPart]]:
        results: dict[str, list[ContentPart]] = {}
        for event in cls._json_lines(path):
            tool_call_id = event.get("tool_call_id")
            content = event.get("content")
            if not isinstance(tool_call_id, str) or not isinstance(content, str):
                continue
            images = [part for part in cls._content_parts(content) if part.type == "image"]
            if images:
                results[tool_call_id] = images
        return results

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: GrokBuildConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        transcript_file = work_dir / "transcript.jsonl"
        transcript_events = list(cls._json_lines(transcript_file))
        terminal_event = next(
            (
                event
                for event in reversed(transcript_events)
                if event.get("type") in {"end", "error"}
            ),
            {},
        )
        session_id = terminal_event.get("sessionId")
        session_dir = cls._find_session_dir(work_dir / "grok-home", session_id)
        chat_file = cls._session_artifact(
            work_dir,
            session_dir,
            source_name="chat_history.jsonl",
        )
        updates_file = cls._session_artifact(
            work_dir,
            session_dir,
            source_name="updates.jsonl",
        )
        events_file = cls._session_artifact(
            work_dir,
            session_dir,
            source_name="events.jsonl",
        )
        media_results = cls._session_media(work_dir / _SESSION_MEDIA_EXPORT)

        tool_errors: dict[str, bool] = {}
        update_terminal: dict[str, Any] = {}
        if updates_file is not None:
            tool_errors, update_terminal = cls._updates_metadata(updates_file)

        parsed_chat = False
        if chat_file is not None and chat_file.exists():
            parsed_chat = cls._parse_chat_history(
                chat_file,
                builder,
                tool_errors=tool_errors,
                session_dir=session_dir,
                media_results=media_results,
            )
        if not parsed_chat:
            cls._parse_stream(transcript_events, transcript_file, builder)

        usage_source = terminal_event if terminal_event.get("usage") else update_terminal
        cls._apply_usage(usage_source, builder)
        try:
            recover_telemetry_artifacts(
                work_dir,
                events_file=events_file,
                updates_file=updates_file,
                terminal_event=terminal_event,
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("grok_build: telemetry recovery failed: %s", exc)

        metadata = {
            "exit_code": run_result.exit_code,
            "transcript_path": str(transcript_file),
            "stderr_path": run_result.stderr_path,
            "session_id": session_id or update_terminal.get("sessionId"),
            "session_dir": str(session_dir) if session_dir else None,
            "chat_history_path": str(chat_file) if chat_file else None,
            "updates_path": str(updates_file) if updates_file else None,
            "events_path": str(events_file) if events_file else None,
            "telemetry_path": str(work_dir / "telemetry.jsonl"),
            "telemetry_summary_path": str(work_dir / "telemetry_summary.json"),
            "stop_reason": terminal_event.get("stopReason") or update_terminal.get("stop_reason"),
            "num_turns": terminal_event.get("num_turns") or update_terminal.get("numTurns"),
            "model_usage": terminal_event.get("modelUsage")
            or (update_terminal.get("usage") or {}).get("modelUsage"),
        }
        builder.trajectory.extra.setdefault("grok_build", {}).update(metadata)

    @classmethod
    def _parse_chat_history(
        cls,
        chat_file: Path,
        builder: TrajectoryBuilder,
        *,
        tool_errors: dict[str, bool],
        session_dir: Path | None,
        media_results: dict[str, list[ContentPart]],
    ) -> bool:
        pending_reasoning: list[str] = []
        added = False
        for event in cls._json_lines(chat_file):
            event_type = event.get("type")
            if event_type == "reasoning":
                for item in event.get("summary") or []:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        pending_reasoning.append(item["text"])
                continue
            if event_type == "assistant":
                tool_calls: list[ToolCall] = []
                for raw_call in event.get("tool_calls") or []:
                    if not isinstance(raw_call, dict):
                        continue
                    arguments = cls._tool_arguments(raw_call.get("arguments"))
                    name = str(raw_call.get("name") or "")
                    if name == "use_tool" and isinstance(arguments.get("tool_name"), str):
                        name = arguments["tool_name"]
                        tool_input = arguments.get("tool_input")
                        arguments = tool_input if isinstance(tool_input, dict) else {}
                    tool_calls.append(
                        ToolCall(
                            id=str(raw_call.get("id") or ""),
                            name=name,
                            arguments=arguments,
                        )
                    )
                message = cls._message_text(event.get("content"))
                reasoning = "\n".join(pending_reasoning).strip() or None
                pending_reasoning.clear()
                if message is not None or reasoning is not None or tool_calls:
                    builder.add_step(
                        source="agent",
                        message=message,
                        reasoning=reasoning,
                        tool_calls=tool_calls,
                    )
                    added = True
                continue
            if event_type == "tool_result":
                tool_call_id = str(event.get("tool_call_id") or "")
                content = cls._content_parts(event.get("content"))
                if not any(part.type == "image" for part in content):
                    if session_dir is not None:
                        content.extend(
                            cls._images_from_mcp_spill(
                                session_dir=session_dir,
                                tool_call_id=tool_call_id,
                            )
                        )
                    if not any(part.type == "image" for part in content):
                        content.extend(media_results.get(tool_call_id, []))
                builder.add_step(
                    source="environment",
                    observation=Observation(
                        results=[
                            ToolResult(
                                tool_call_id=tool_call_id,
                                content=content,
                                is_error=tool_errors.get(tool_call_id, False),
                            )
                        ]
                    ),
                )
                added = True
        if pending_reasoning:
            builder.add_step(
                source="agent",
                reasoning="\n".join(pending_reasoning).strip(),
            )
            added = True
        return added

    @classmethod
    def _images_from_mcp_spill(
        cls,
        *,
        session_dir: Path,
        tool_call_id: str,
    ) -> list[ContentPart]:
        if not tool_call_id or Path(tool_call_id).name != tool_call_id:
            return []
        spill_file = session_dir / "mcp" / f"{tool_call_id}.txt"
        try:
            if spill_file.stat().st_size > _MAX_MCP_SPILL_BYTES:
                return []
            raw = spill_file.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, OSError):
            return []
        return [part for part in cls._content_parts(raw) if part.type == "image"]

    @classmethod
    def _parse_stream(
        cls,
        events: list[dict[str, Any]],
        transcript_file: Path,
        builder: TrajectoryBuilder,
    ) -> None:
        if not transcript_file.exists():
            builder.add_step(
                source="system",
                message=f"grok-build: no transcript at {transcript_file}",
                extra={"reason": "no_transcript"},
            )
            return
        text: list[str] = []
        reasoning: list[str] = []
        errors: list[str] = []
        for event in events:
            event_type = event.get("type")
            if event_type == "text" and isinstance(event.get("data"), str):
                text.append(event["data"])
            elif event_type == "thought" and isinstance(event.get("data"), str):
                reasoning.append(event["data"])
            elif event_type == "error" and isinstance(event.get("message"), str):
                errors.append(event["message"])
        message = "".join(text).strip() or None
        thought = "".join(reasoning).strip() or None
        if message is not None or thought is not None:
            builder.add_step(source="agent", message=message, reasoning=thought)
        for error in errors:
            builder.add_step(
                source="system",
                message=error,
                extra={"grok_build_error": True},
            )
        if message is None and thought is None and not errors:
            builder.add_step(
                source="system",
                message=f"grok-build: transcript at {transcript_file} contained no events",
                extra={"reason": "empty_transcript"},
            )

    @classmethod
    def _updates_metadata(
        cls,
        updates_file: Path,
    ) -> tuple[dict[str, bool], dict[str, Any]]:
        tool_errors: dict[str, bool] = {}
        terminal: dict[str, Any] = {}
        for event in cls._json_lines(updates_file):
            params = event.get("params") or {}
            update = params.get("update") or {}
            update_type = update.get("sessionUpdate")
            if update_type == "tool_call_update":
                tool_call_id = update.get("toolCallId")
                status = str(update.get("status") or "").lower()
                if isinstance(tool_call_id, str) and status:
                    tool_errors[tool_call_id] = status in {
                        "failed",
                        "error",
                        "cancelled",
                    }
            elif update_type == "turn_completed":
                terminal = {
                    **update,
                    "sessionId": params.get("sessionId"),
                }
        return tool_errors, terminal

    @staticmethod
    def _tool_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                return {"raw": raw}
            if isinstance(decoded, dict):
                return decoded
            return {"value": decoded}
        return {"value": raw}

    @staticmethod
    def _message_text(content: Any) -> str | None:
        if isinstance(content, str):
            return content.strip() or None
        if not isinstance(content, list):
            return None
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "output_text"}
            and isinstance(item.get("text"), str)
        ]
        return "".join(parts).strip() or None

    @classmethod
    def _content_parts(cls, content: Any) -> list[ContentPart]:
        if isinstance(content, list):
            parts: list[ContentPart] = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(ContentPart(type="text", text=str(item)))
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    parts.extend(cls._content_parts(item.get("text", "")))
                elif item_type == "image" and isinstance(item.get("data"), str):
                    parts.append(
                        ContentPart(
                            type="image",
                            image=ImageSource(
                                type="base64",
                                data=item["data"],
                                media_type=str(item.get("mimeType") or "image/png"),
                            ),
                        )
                    )
            return parts
        if not isinstance(content, str):
            return [
                ContentPart(
                    type="text",
                    text=json.dumps(content, ensure_ascii=False, default=str),
                )
            ]

        parts = []
        cursor = 0
        for match in _DATA_URL_RE.finditer(content):
            text = content[cursor : match.start()]
            if text:
                parts.append(ContentPart(type="text", text=text))
            parts.append(
                ContentPart(
                    type="image",
                    image=ImageSource(
                        type="base64",
                        data=match.group(2),
                        media_type=match.group(1),
                    ),
                )
            )
            cursor = match.end()
        tail = content[cursor:]
        if tail:
            parts.append(ContentPart(type="text", text=tail))
        return parts or [ContentPart(type="text", text=content)]

    @classmethod
    def _apply_usage(
        cls,
        terminal_event: dict[str, Any],
        builder: TrajectoryBuilder,
    ) -> None:
        usage = terminal_event.get("usage")
        if not isinstance(usage, dict):
            return
        if "input_tokens" in usage:
            input_tokens = usage.get("input_tokens")
            cache_read_tokens = usage.get("cache_read_input_tokens")
            output_tokens = usage.get("output_tokens")
        else:
            full_input = usage.get("inputTokens")
            cache_read_tokens = usage.get("cachedReadTokens")
            if isinstance(full_input, int) and isinstance(cache_read_tokens, int):
                input_tokens = max(0, full_input - cache_read_tokens)
            else:
                input_tokens = full_input
            output_tokens = usage.get("outputTokens")
        builder.override_final_metrics(
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cache_read_tokens=cache_read_tokens,
            total_cost_usd=terminal_event.get("total_cost_usd"),
        )

    @staticmethod
    def _find_session_dir(grok_home: Path, session_id: Any) -> Path | None:
        sessions_root = grok_home / "sessions"
        if isinstance(session_id, str) and session_id:
            matches = [path for path in sessions_root.glob(f"**/{session_id}") if path.is_dir()]
            if matches:
                return max(matches, key=lambda path: path.stat().st_mtime)
        summaries = [path for path in sessions_root.glob("**/summary.json") if path.is_file()]
        if not summaries:
            return None
        return max(summaries, key=lambda path: path.stat().st_mtime).parent

    @staticmethod
    def _json_lines(path: Path):
        try:
            handle = path.open(encoding="utf-8", errors="replace")
        except (FileNotFoundError, OSError):
            return
        with handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event

    @staticmethod
    def _diagnose_failure(
        *,
        transcript_file: Path,
        stderr_file: Path,
        exit_code: int | None,
    ) -> str:
        transcript = (
            transcript_file.read_text(encoding="utf-8", errors="replace")
            if transcript_file.exists()
            else ""
        )
        stderr = (
            stderr_file.read_text(encoding="utf-8", errors="replace")
            if stderr_file.exists()
            else ""
        )
        parts = [
            f"agent failed (rc={exit_code})",
            f"stderr={len(stderr)}B transcript={len(transcript)}B",
        ]
        if '"http_status": 429' in transcript or "429" in stderr:
            parts.append("LLM rate-limited")
        if stderr.strip():
            parts.append(f"stderr tail: ...{stderr[-800:]}")
        if transcript.strip():
            parts.append(f"transcript tail: ...{transcript[-800:]}")
        return " | ".join(parts)
