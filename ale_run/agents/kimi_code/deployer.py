"""Run the official Kimi Code CLI inside an ALE sandbox."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar, Iterator

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

from .config import KimiCodeConfig
from .telemetry import (
    OTEL_NPM_PACKAGE_VERSIONS,
    OTEL_NPM_PACKAGES,
    KimiCodeOtelCollector,
    recover_telemetry_artifacts,
    write_otel_bootstrap,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0
_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")
_MIN_NODE_VERSION = (22, 19, 0)
_DATA_URL_RE = re.compile(r"^data:([^;,]+);base64,(.+)$", re.DOTALL)


def _expected_version(npm_spec: str | None) -> str | None:
    if not npm_spec:
        return None
    match = _VERSION_RE.search(npm_spec.rsplit("@", 1)[-1])
    return match.group(1) if match else None


def _command_version(path: str, timeout: int = 60) -> str | None:
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = _VERSION_RE.search((result.stdout or "") + (result.stderr or ""))
    return match.group(1) if match else None


def _node_version(path: str) -> tuple[int, int, int] | None:
    version = _command_version(path, timeout=30)
    return tuple(map(int, version.split("."))) if version else None


def _find_kimi_shim(prefix: str) -> str | None:
    for candidate in (
        os.path.join(prefix, "bin", "kimi"),
        os.path.join(prefix, "kimi.cmd"),
        os.path.join(prefix, "bin", "kimi.cmd"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _npm_global_root(npm_path: str, prefix: str) -> Path:
    result = subprocess.run(
        [npm_path, "root", "--global", "--prefix", prefix],
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            "kimi_code: could not resolve npm global root: "
            f"{(result.stderr or result.stdout or '').strip()[-400:]}"
        )
    return Path(result.stdout.strip()).resolve()


class KimiCodeDeployer(BaseAgentDeployer):
    """Sandbox deployer for ``@moonshot-ai/kimi-code``."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = (
        "transcript.jsonl",
        "stderr.log",
        "otel_requests.jsonl",
    )

    @property
    def version(self) -> str | None:
        config: KimiCodeConfig = self.config  # type: ignore[assignment]
        return config.cli_version

    async def install(self) -> None:
        config: KimiCodeConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox
        from ale_run.agents._bootstrap import ensure_cua_mcp_server, ensure_node_npm

        node_path, npm_path = await ensure_node_npm()
        node_version = await asyncio.to_thread(_node_version, node_path)
        if node_version is None or node_version < _MIN_NODE_VERSION:
            rendered = ".".join(map(str, node_version or ()))
            raise RuntimeError(
                "kimi_code: Kimi Code requires Node.js >=22.19.0, "
                f"found {rendered or 'an unreadable version'} at {node_path}"
            )

        home = os.path.expanduser("~")
        prefix = os.path.join(home, ".local")
        npm_env = {**os.environ, "npm_config_cache": os.path.join(home, ".npm-ale")}
        node_modules = await asyncio.to_thread(_npm_global_root, npm_path, prefix)
        kimi_path = shutil.which("kimi")
        expected = _expected_version(config.cli_version)
        installed = await asyncio.to_thread(_command_version, kimi_path) if kimi_path else None

        packages: list[str] = []
        if not kimi_path or (expected is not None and installed != expected):
            packages.append(config.cli_version or "@moonshot-ai/kimi-code")
        if config.otel_enabled and any(
            self._package_version(node_modules, name) != version
            for name, version in OTEL_NPM_PACKAGE_VERSIONS.items()
        ):
            packages.extend(OTEL_NPM_PACKAGES)
        if packages:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    npm_path,
                    "install",
                    "-g",
                    "--force",
                    "--prefix",
                    prefix,
                    *packages,
                ],
                capture_output=True,
                text=True,
                timeout=900,
                env=npm_env,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"kimi_code: npm install failed for {packages}: "
                    f"{(result.stderr or result.stdout or '')[-800:]}"
                )

        if not kimi_path or (expected is not None and installed != expected):
            for bin_dir in (prefix, os.path.join(prefix, "bin")):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            kimi_path = _find_kimi_shim(prefix) or shutil.which("kimi")
            if not kimi_path:
                raise RuntimeError("kimi_code: 'kimi' not found after npm installation")
            installed = await asyncio.to_thread(_command_version, kimi_path)
            if expected is not None and installed != expected:
                raise RuntimeError(
                    f"kimi_code: installed version {installed!r} != expected {expected!r}"
                )

        self._kimi_path = kimi_path
        self._otel_bootstrap_path: Path | None = None
        if config.otel_enabled:
            node_modules = await asyncio.to_thread(_npm_global_root, npm_path, prefix)
            self._otel_bootstrap_path = node_modules / "ale-kimi-otel" / "bootstrap.mjs"
            await asyncio.to_thread(write_otel_bootstrap, self._otel_bootstrap_path)

        work_dir = Path(self.executor.work_dir)
        kimi_home = work_dir / "kimi-home"
        kimi_home.mkdir(parents=True, exist_ok=True)
        await ensure_cua_mcp_server(sandbox)
        from ale_run.agents._bootstrap import cua_bridge_env

        mcp_config = {
            "mcpServers": {
                "cua": {
                    "command": sandbox.node,
                    "args": [
                        self._join(
                            sandbox.mcp_server_dir,
                            "src",
                            "index.js",
                            is_linux=sandbox.is_linux,
                        )
                    ],
                    "env": cua_bridge_env(self.executor),
                }
            }
        }
        (kimi_home / "mcp.json").write_text(
            json.dumps(mcp_config, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "kimi_code: CLI ready at %s (version %s, node %s)",
            kimi_path,
            installed or "unknown",
            ".".join(map(str, node_version)),
        )

    async def launch(self, prompt: str) -> AgentRunResult:
        config: KimiCodeConfig = self.config  # type: ignore[assignment]
        work_dir = Path(self.executor.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = work_dir / "prompt.txt"
        transcript_file = work_dir / "transcript.jsonl"
        stderr_file = work_dir / "stderr.log"
        pid_file = work_dir / "kimi.pid"
        for path in (transcript_file, stderr_file, pid_file):
            path.unlink(missing_ok=True)
        prompt_file.write_text(prompt, encoding="utf-8")

        collector = KimiCodeOtelCollector(work_dir) if config.otel_enabled else None
        process: subprocess.Popen[bytes] | None = None
        started = time.monotonic()
        try:
            if collector is not None:
                collector.start()
            env = self._build_env(
                config,
                work_dir=work_dir,
                otel_endpoint=collector.endpoint if collector else None,
                otel_bootstrap=self._otel_bootstrap_path,
            )
            with transcript_file.open("wb") as stdout, stderr_file.open("wb") as stderr:
                process = await asyncio.to_thread(
                    subprocess.Popen,
                    [
                        self._kimi_path,
                        "--prompt",
                        prompt,
                        "--output-format",
                        "stream-json",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    cwd=str(work_dir),
                    env=env,
                    start_new_session=hasattr(os, "setsid"),
                )
            pid_file.write_text(str(process.pid), encoding="ascii")
            while process.poll() is None:
                await asyncio.sleep(_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            if process is not None:
                await self._terminate(process)
            raise
        finally:
            if collector is not None:
                collector.stop()

        if process is None:
            raise RuntimeError("kimi_code: process was not started")
        duration_s = time.monotonic() - started
        status = "completed" if process.returncode == 0 else "failed"
        return AgentRunResult(
            status=status,
            pid=process.pid,
            exit_code=process.returncode,
            transcript_path=str(transcript_file),
            stderr_path=str(stderr_file),
            duration_s=duration_s,
            error=None
            if status == "completed"
            else self._diagnose_failure(transcript_file, stderr_file, process.returncode),
        )

    def _build_env(
        self,
        config: KimiCodeConfig,
        *,
        work_dir: Path,
        otel_endpoint: str | None = None,
        otel_bootstrap: Path | None = None,
    ) -> dict[str, str]:
        env = {**os.environ, **(self.executor.env or {})}
        api_key = config.api_key or env.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"kimi_code: set config.api_key or environment {config.api_key_env}")
        env.update(
            {
                "KIMI_CODE_HOME": str(work_dir / "kimi-home"),
                "KIMI_MODEL_NAME": config.model,
                "KIMI_MODEL_API_KEY": api_key,
                "KIMI_MODEL_PROVIDER_TYPE": config.provider_type,
                "KIMI_MODEL_BASE_URL": config.base_url,
                "KIMI_MODEL_MAX_CONTEXT_SIZE": str(config.max_context_size),
                "KIMI_MODEL_CAPABILITIES": ",".join(config.capabilities),
                "KIMI_MODEL_OUTPUT_FORMAT": "stream-json",
            }
        )
        self._set_optional_env(env, "KIMI_MODEL_THINKING_EFFORT", config.thinking_effort)
        self._set_optional_env(
            env,
            "KIMI_MODEL_MAX_COMPLETION_TOKENS",
            config.max_completion_tokens,
        )
        if config.disable_telemetry:
            env["KIMI_DISABLE_TELEMETRY"] = "1"
        if config.disable_auto_update:
            env["KIMI_CODE_NO_AUTO_UPDATE"] = "1"
        if otel_endpoint is not None:
            if otel_bootstrap is None:
                raise RuntimeError("kimi_code: OTel enabled without bootstrap")
            import_option = f"--import={otel_bootstrap.resolve().as_uri()}"
            env["ALE_KIMI_OTEL_ENDPOINT"] = otel_endpoint
            env["ALE_KIMI_OTEL_IMPORT_OPTION"] = import_option
            env["NODE_OPTIONS"] = " ".join(
                value for value in (env.get("NODE_OPTIONS", "").strip(), import_option) if value
            )
        return env

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: KimiCodeConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        transcript_file = work_dir / "transcript.jsonl"
        wire_files = sorted(
            (work_dir / "kimi-home" / "sessions").glob("**/agents/main/wire.jsonl"),
            key=lambda path: path.stat().st_mtime,
        )
        wire_file = wire_files[-1] if wire_files else None
        try:
            recover_telemetry_artifacts(work_dir, wire_file=wire_file)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("kimi_code: telemetry recovery failed: %s", exc)

        if wire_file is not None:
            cls._parse_wire(
                wire_file,
                builder,
                transcript_results=cls._transcript_tool_results(transcript_file),
            )
        else:
            cls._parse_transcript(transcript_file, builder)
        builder.trajectory.extra.setdefault("kimi_code", {}).update(
            {
                "exit_code": run_result.exit_code,
                "transcript_path": str(transcript_file),
                "wire_path": str(wire_file) if wire_file else None,
                "stderr_path": run_result.stderr_path,
                "telemetry_path": str(work_dir / "telemetry.jsonl"),
                "telemetry_summary_path": str(work_dir / "telemetry_summary.json"),
            }
        )

    @classmethod
    def _parse_wire(
        cls,
        wire_file: Path,
        builder: TrajectoryBuilder,
        *,
        transcript_results: dict[str, list[ContentPart]],
    ) -> None:
        active: dict[str, Any] | None = None
        requests: list[dict[str, Any]] = []
        blobs_dir = wire_file.parent / "blobs"
        for record in cls._json_lines(wire_file):
            if record.get("type") == "llm.request":
                requests.append(record)
                continue
            if record.get("type") != "context.append_loop_event":
                continue
            event = record.get("event") or {}
            event_type = event.get("type")
            if event_type == "step.begin":
                if active is not None:
                    cls._flush_step(active, builder)
                active = cls._new_step()
            elif event_type == "content.part":
                active = active or cls._new_step()
                part = event.get("part") or {}
                if part.get("type") == "text":
                    active["text"].append(part.get("text", ""))
                elif part.get("type") == "think":
                    active["reasoning"].append(part.get("think", ""))
            elif event_type == "tool.call" and active is not None:
                arguments = event.get("args")
                active["tool_calls"].append(
                    ToolCall(
                        id=event.get("toolCallId") or event.get("uuid") or "",
                        name=event.get("name") or "",
                        arguments=arguments
                        if isinstance(arguments, dict)
                        else {"value": arguments},
                    )
                )
            elif event_type == "tool.result" and active is not None:
                tool_id = event.get("toolCallId") or event.get("parentUuid") or ""
                active["tool_results"].append(
                    cls._wire_tool_result(
                        event,
                        blobs_dir=blobs_dir,
                        transcript_content=transcript_results.get(tool_id),
                    )
                )
            elif event_type in {"step.retrying", "turn.interrupted"}:
                builder.add_step(
                    source="system",
                    message=event.get("errorMessage")
                    or event.get("message")
                    or f"Kimi Code {event_type}",
                    extra={"kimi_code_event": event},
                )
            elif event_type == "step.end" and active is not None:
                usage = event.get("usage") or {}
                active["metrics"] = StepMetrics(
                    input_tokens=usage.get("inputOther"),
                    output_tokens=usage.get("output"),
                    cache_read_tokens=usage.get("inputCacheRead"),
                    cache_creation_tokens=usage.get("inputCacheCreation"),
                    duration_ms=(
                        (event.get("llmFirstTokenLatencyMs") or 0)
                        + (event.get("llmStreamDurationMs") or 0)
                    )
                    or None,
                )
                active["extra"] = {
                    key: value
                    for key, value in event.items()
                    if key not in {"type", "uuid", "turnId", "step", "usage"}
                }
                cls._flush_step(active, builder)
                active = None
        if active is not None:
            cls._flush_step(active, builder)
        if requests:
            builder.trajectory.extra.setdefault("kimi_code", {})["llm_requests"] = requests

    @classmethod
    def _parse_transcript(cls, path: Path, builder: TrajectoryBuilder) -> None:
        if not path.exists():
            builder.add_step(
                source="system",
                message=f"kimi-code: no transcript at {path}",
                extra={"reason": "no_transcript"},
            )
            return
        for event in cls._json_lines(path):
            role = event.get("role")
            if role == "assistant":
                tool_calls: list[ToolCall] = []
                for raw in event.get("tool_calls") or []:
                    function = raw.get("function") or {}
                    arguments = function.get("arguments") or "{}"
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"raw": arguments}
                    tool_calls.append(
                        ToolCall(
                            id=raw.get("id") or "",
                            name=function.get("name") or "",
                            arguments=arguments,
                        )
                    )
                builder.add_step(
                    source="agent",
                    message=event.get("content"),
                    tool_calls=tool_calls,
                )
            elif role == "tool":
                builder.add_step(
                    source="environment",
                    observation=Observation(
                        results=[
                            ToolResult(
                                tool_call_id=event.get("tool_call_id") or "",
                                content=cls._content_parts(event.get("content", "")),
                            )
                        ]
                    ),
                )
            elif role == "meta" and event.get("type") == "turn.step.retrying":
                builder.add_step(
                    source="system",
                    message=event.get("error_message") or "Kimi Code retrying",
                    extra={"kimi_code_retry": event},
                )

    @classmethod
    def _wire_tool_result(
        cls,
        event: dict[str, Any],
        *,
        blobs_dir: Path,
        transcript_content: list[ContentPart] | None,
    ) -> ToolResult:
        result = event.get("result") or {}
        content = cls._content_parts(result.get("output"), blobs_dir=blobs_dir)
        if transcript_content and any(part.type == "image" for part in transcript_content):
            content = transcript_content
        for key in ("message", "note"):
            if isinstance(result.get(key), str) and result[key]:
                content.append(ContentPart(type="text", text=result[key]))
        return ToolResult(
            tool_call_id=event.get("toolCallId") or event.get("parentUuid") or "",
            content=content,
            is_error=bool(result.get("isError")),
        )

    @classmethod
    def _content_parts(
        cls,
        output: Any,
        *,
        blobs_dir: Path | None = None,
    ) -> list[ContentPart]:
        if isinstance(output, str):
            try:
                decoded = json.loads(output)
            except json.JSONDecodeError:
                return [ContentPart(type="text", text=output)]
            if isinstance(decoded, (dict, list)):
                return cls._content_parts(decoded, blobs_dir=blobs_dir)
            return [ContentPart(type="text", text=output)]
        if isinstance(output, dict):
            output = [output]
        if not isinstance(output, list):
            return [ContentPart(type="text", text=json.dumps(output, default=str))]

        content: list[ContentPart] = []
        for part in output:
            if not isinstance(part, dict):
                content.append(ContentPart(type="text", text=str(part)))
                continue
            if part.get("type") == "text":
                content.append(ContentPart(type="text", text=part.get("text", "")))
                continue
            if part.get("type") != "image_url":
                continue
            image_url = part.get("imageUrl") or part.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str):
                continue
            image = cls._image_source(url, blobs_dir)
            if image is not None:
                content.append(ContentPart(type="image", image=image))
        return content

    @staticmethod
    def _image_source(url: str, blobs_dir: Path | None) -> ImageSource | None:
        if url.startswith("blobref:"):
            if blobs_dir is None:
                return None
            media_type, separator, digest = url.removeprefix("blobref:").partition(";")
            blob = blobs_dir / digest
            if separator and digest and blob.is_file():
                return ImageSource(
                    type="base64",
                    media_type=media_type or "image/png",
                    data=base64.b64encode(blob.read_bytes()).decode("ascii"),
                )
            return None
        match = _DATA_URL_RE.match(url)
        if match:
            return ImageSource(type="base64", media_type=match.group(1), data=match.group(2))
        return ImageSource(type="url", url=url)

    @classmethod
    def _transcript_tool_results(cls, path: Path) -> dict[str, list[ContentPart]]:
        results: dict[str, list[ContentPart]] = {}
        for event in cls._json_lines(path):
            tool_id = event.get("tool_call_id")
            if event.get("role") == "tool" and isinstance(tool_id, str) and tool_id:
                results[tool_id] = cls._content_parts(event.get("content", ""))
        return results

    @staticmethod
    def _new_step() -> dict[str, Any]:
        return {
            "text": [],
            "reasoning": [],
            "tool_calls": [],
            "tool_results": [],
            "extra": {},
        }

    @staticmethod
    def _flush_step(active: dict[str, Any], builder: TrajectoryBuilder) -> None:
        text = "".join(active["text"]).strip() or None
        reasoning = "".join(active["reasoning"]).strip() or None
        if text or reasoning or active["tool_calls"] or active.get("metrics"):
            builder.add_step(
                source="agent",
                message=text,
                reasoning=reasoning,
                tool_calls=active["tool_calls"],
                metrics=active.get("metrics"),
                extra=active.get("extra") or {},
            )
        if active["tool_results"]:
            builder.add_step(
                source="environment",
                observation=Observation(results=active["tool_results"]),
            )

    @staticmethod
    def _json_lines(path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event

    @staticmethod
    def _join(*parts: str, is_linux: bool) -> str:
        separator = "/" if is_linux else "\\"
        return (
            parts[0].rstrip("/\\")
            + separator
            + separator.join(part.strip("/\\") for part in parts[1:])
        )

    @staticmethod
    def _set_optional_env(env: dict[str, str], key: str, value: Any) -> None:
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)

    @staticmethod
    def _package_version(root: Path, package: str) -> str | None:
        try:
            return json.loads((root / package / "package.json").read_text())["version"]
        except (FileNotFoundError, OSError, KeyError, json.JSONDecodeError):
            return None

    @staticmethod
    async def _terminate(process: subprocess.Popen[bytes]) -> None:
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=_TERM_GRACE_S)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            try:
                process.kill()
            except ProcessLookupError:
                pass

    @staticmethod
    def _diagnose_failure(transcript: Path, stderr: Path, exit_code: int | None) -> str:
        transcript_text = transcript.read_text(errors="replace") if transcript.exists() else ""
        stderr_text = stderr.read_text(errors="replace") if stderr.exists() else ""
        parts = [
            f"agent failed (rc={exit_code})",
            f"stderr={len(stderr_text)}B transcript={len(transcript_text)}B",
        ]
        if "429" in stderr_text or "status_code=429" in transcript_text:
            parts.append("LLM rate-limited")
        if stderr_text.strip():
            parts.append(f"stderr tail: ...{stderr_text[-800:]}")
        if transcript_text.strip():
            parts.append(f"transcript tail: ...{transcript_text[-800:]}")
        return " | ".join(parts)
