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
from typing import Any, ClassVar

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


def _installed_version(kimi_path: str) -> str | None:
    try:
        probe = subprocess.run(
            [kimi_path, "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = _VERSION_RE.search((probe.stdout or "") + (probe.stderr or ""))
    return match.group(1) if match else None


def _node_version(node_path: str) -> tuple[int, int, int] | None:
    try:
        probe = subprocess.run(
            [node_path, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = _VERSION_RE.search((probe.stdout or "") + (probe.stderr or ""))
    if not match:
        return None
    major, minor, patch = (int(part) for part in match.group(1).split("."))
    return major, minor, patch


def _find_kimi_shim(prefix: str) -> str | None:
    for candidate in (
        os.path.join(prefix, "bin", "kimi"),
        os.path.join(prefix, "kimi.cmd"),
        os.path.join(prefix, "bin", "kimi.cmd"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


class KimiCodeDeployer(BaseAgentDeployer):
    """Sandbox deployer for ``@moonshot-ai/kimi-code``."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    @property
    def version(self) -> str | None:
        config: KimiCodeConfig = self.config  # type: ignore[assignment]
        return config.cli_version

    async def install(self) -> None:
        config: KimiCodeConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        from ale_run.agents._bootstrap import ensure_cua_mcp_server, ensure_node_npm

        node_path, npm_path = await ensure_node_npm()
        installed_node = await asyncio.to_thread(_node_version, node_path)
        if installed_node is None or installed_node < _MIN_NODE_VERSION:
            rendered = ".".join(str(part) for part in installed_node or ())
            raise RuntimeError(
                "kimi_code: Kimi Code requires Node.js >=22.19.0, "
                f"found {rendered or 'an unreadable version'} at {node_path}"
            )

        kimi_path = shutil.which("kimi")
        expected = _expected_version(config.cli_version)
        installed = await asyncio.to_thread(_installed_version, kimi_path) if kimi_path else None
        if not kimi_path or (expected is not None and installed != expected):
            home = os.path.expanduser("~")
            prefix = os.path.join(home, ".local")
            package = config.cli_version or "@moonshot-ai/kimi-code"
            logger.info(
                "kimi_code: installing %s via npm (installed=%s expected=%s)",
                package,
                installed or "missing",
                expected or "unpinned",
            )
            env = {
                **os.environ,
                "npm_config_cache": os.path.join(home, ".npm-ale"),
            }
            process = await asyncio.to_thread(
                subprocess.run,
                [
                    npm_path,
                    "install",
                    "-g",
                    "--force",
                    "--prefix",
                    prefix,
                    package,
                ],
                capture_output=True,
                text=True,
                timeout=900,
                env=env,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"npm install -g {package} failed (rc={process.returncode}): "
                    f"{(process.stderr or process.stdout or '')[-800:]}"
                )
            for bin_dir in (prefix, os.path.join(prefix, "bin")):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            kimi_path = _find_kimi_shim(prefix) or shutil.which("kimi")
            if not kimi_path:
                raise RuntimeError("kimi_code: 'kimi' not found after dynamic npm installation")
            installed = await asyncio.to_thread(_installed_version, kimi_path)
            if expected is not None and installed != expected:
                raise RuntimeError(
                    f"kimi_code: post-install version {installed!r} does not "
                    f"match expected {expected!r}"
                )
            logger.info("kimi_code: dynamic npm installation completed")

        self._kimi_path = kimi_path
        logger.info(
            "kimi_code: CLI ready at %s (version %s, node %s)",
            kimi_path,
            installed or "unknown",
            ".".join(str(part) for part in installed_node),
        )

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

    async def launch(self, prompt: str) -> AgentRunResult:
        config: KimiCodeConfig = self.config  # type: ignore[assignment]
        work_dir = Path(self.executor.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        prompt_file = work_dir / "prompt.txt"
        transcript_file = work_dir / "transcript.jsonl"
        stderr_file = work_dir / "stderr.log"
        pid_file = work_dir / "kimi.pid"
        for path in (transcript_file, stderr_file, pid_file):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        prompt_file.write_text(prompt, encoding="utf-8")
        argv = [
            self._kimi_path,
            "--prompt",
            prompt,
            "--output-format",
            "stream-json",
        ]
        env = self._build_env(config, work_dir=work_dir)

        started = time.monotonic()
        with open(transcript_file, "wb") as stdout, open(stderr_file, "wb") as stderr:
            process = await asyncio.to_thread(
                subprocess.Popen,
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                cwd=str(work_dir),
                env=env,
                start_new_session=True if hasattr(os, "setsid") else False,
            )
        pid_file.write_text(str(process.pid), encoding="ascii")
        logger.info("kimi_code: spawned pid=%s", process.pid)

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
            except (asyncio.TimeoutError, asyncio.CancelledError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            raise

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

    def _build_env(
        self,
        config: KimiCodeConfig,
        *,
        work_dir: Path,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.executor.env or {})

        api_key = config.api_key or env.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"kimi_code: no API key configured; set config.api_key or {config.api_key_env}"
            )

        env.update(
            {
                "KIMI_CODE_HOME": str(work_dir / "kimi-home"),
                "KIMI_MODEL_NAME": config.model,
                "KIMI_MODEL_API_KEY": api_key,
                "KIMI_MODEL_PROVIDER_TYPE": config.provider_type,
                "KIMI_MODEL_BASE_URL": config.base_url,
                "KIMI_MODEL_MAX_CONTEXT_SIZE": str(config.max_context_size),
                "KIMI_MODEL_CAPABILITIES": ",".join(config.capabilities),
            }
        )
        if config.thinking_effort is None:
            env.pop("KIMI_MODEL_THINKING_EFFORT", None)
        else:
            env["KIMI_MODEL_THINKING_EFFORT"] = config.thinking_effort
        if config.max_completion_tokens is None:
            env.pop("KIMI_MODEL_MAX_COMPLETION_TOKENS", None)
        else:
            env["KIMI_MODEL_MAX_COMPLETION_TOKENS"] = str(config.max_completion_tokens)
        if config.disable_telemetry:
            env["KIMI_DISABLE_TELEMETRY"] = "1"
        if config.disable_auto_update:
            env["KIMI_CODE_NO_AUTO_UPDATE"] = "1"
        return env

    @staticmethod
    def _join(*parts: str, is_linux: bool) -> str:
        separator = "/" if is_linux else "\\"
        head = parts[0].rstrip("/\\")
        tail = separator.join(part.strip("/\\") for part in parts[1:])
        return f"{head}{separator}{tail}" if tail else head

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: KimiCodeConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        wire_files = sorted(
            (work_dir / "kimi-home" / "sessions").glob("**/agents/main/wire.jsonl"),
            key=lambda path: path.stat().st_mtime,
        )
        if wire_files:
            wire_file = wire_files[-1]
            cls._parse_wire(wire_file, builder)
            source_path = wire_file
        else:
            transcript_file = work_dir / "transcript.jsonl"
            cls._parse_transcript(transcript_file, builder)
            source_path = transcript_file

        builder.trajectory.extra.setdefault("kimi_code", {}).update(
            {
                "exit_code": run_result.exit_code,
                "transcript_path": str(work_dir / "transcript.jsonl"),
                "wire_path": str(source_path) if wire_files else None,
                "stderr_path": run_result.stderr_path,
            }
        )

    @classmethod
    def _parse_wire(cls, wire_file: Path, builder: TrajectoryBuilder) -> None:
        active: dict[str, Any] | None = None
        requests: list[dict[str, Any]] = []
        blobs_dir = wire_file.parent / "blobs"
        for record in cls._json_lines(wire_file):
            record_type = record.get("type")
            if record_type == "llm.request":
                requests.append(record)
                continue
            if record_type != "context.append_loop_event":
                continue
            event = record.get("event") or {}
            event_type = event.get("type")
            if event_type == "step.begin":
                if active is not None:
                    cls._flush_wire_step(active, builder)
                active = {
                    "text": [],
                    "reasoning": [],
                    "tool_calls": [],
                    "tool_results": [],
                    "extra": {},
                }
            elif event_type == "content.part":
                if active is None:
                    active = {
                        "text": [],
                        "reasoning": [],
                        "tool_calls": [],
                        "tool_results": [],
                        "extra": {},
                    }
                part = event.get("part") or {}
                if part.get("type") == "text":
                    active["text"].append(part.get("text", ""))
                elif part.get("type") == "think":
                    active["reasoning"].append(part.get("think", ""))
            elif event_type == "tool.call":
                if active is None:
                    continue
                arguments = event.get("args")
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
                active["tool_calls"].append(
                    ToolCall(
                        id=event.get("toolCallId") or event.get("uuid") or "",
                        name=event.get("name") or "",
                        arguments=arguments,
                    )
                )
            elif event_type == "tool.result":
                if active is None:
                    continue
                active["tool_results"].append(cls._wire_tool_result(event, blobs_dir=blobs_dir))
            elif event_type == "step.retrying":
                builder.add_step(
                    source="system",
                    message=event.get("errorMessage") or "Kimi Code retrying",
                    extra={"kimi_code_retry": event},
                )
            elif event_type == "turn.interrupted":
                builder.add_step(
                    source="system",
                    message=event.get("message") or "Kimi Code turn interrupted",
                    extra={"kimi_code_interrupt": event},
                )
            elif event_type == "step.end":
                if active is None:
                    continue
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
                    if key
                    not in {
                        "type",
                        "uuid",
                        "turnId",
                        "step",
                        "usage",
                    }
                }
                cls._flush_wire_step(active, builder)
                active = None

        if active is not None:
            cls._flush_wire_step(active, builder)
        if requests:
            builder.trajectory.extra.setdefault("kimi_code", {})["llm_requests"] = requests

    @staticmethod
    def _flush_wire_step(active: dict[str, Any], builder: TrajectoryBuilder) -> None:
        text = "".join(active["text"]).strip() or None
        reasoning = "".join(active["reasoning"]).strip() or None
        tool_calls = active["tool_calls"]
        metrics = active.get("metrics")
        if text is not None or reasoning is not None or tool_calls or metrics is not None:
            builder.add_step(
                source="agent",
                message=text,
                reasoning=reasoning,
                tool_calls=tool_calls,
                metrics=metrics,
                extra=active.get("extra") or {},
            )
        tool_results = active["tool_results"]
        if tool_results:
            builder.add_step(
                source="environment",
                observation=Observation(results=tool_results),
            )

    @classmethod
    def _wire_tool_result(
        cls,
        event: dict[str, Any],
        *,
        blobs_dir: Path,
    ) -> ToolResult:
        result = event.get("result") or {}
        content = cls._content_parts(result.get("output"), blobs_dir=blobs_dir)
        for key in ("message", "note"):
            value = result.get(key)
            if isinstance(value, str) and value:
                content.append(ContentPart(type="text", text=value))
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
            return [ContentPart(type="text", text=output)]
        if not isinstance(output, list):
            return [
                ContentPart(
                    type="text",
                    text=json.dumps(output, ensure_ascii=False, default=str),
                )
            ]

        content: list[ContentPart] = []
        for part in output:
            if not isinstance(part, dict):
                content.append(ContentPart(type="text", text=str(part)))
                continue
            part_type = part.get("type")
            if part_type == "text":
                content.append(ContentPart(type="text", text=part.get("text", "")))
                continue
            if part_type != "image_url":
                continue
            image_url = part.get("imageUrl") or part.get("image_url") or {}
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if not isinstance(url, str):
                continue
            if url.startswith("blobref:") and blobs_dir is not None:
                reference = url.removeprefix("blobref:")
                media_type, separator, digest = reference.partition(";")
                blob_path = blobs_dir / digest
                if separator and digest and blob_path.is_file():
                    content.append(
                        ContentPart(
                            type="image",
                            image=ImageSource(
                                type="base64",
                                media_type=media_type or "image/png",
                                data=base64.b64encode(blob_path.read_bytes()).decode("ascii"),
                            ),
                        )
                    )
                    continue
            match = _DATA_URL_RE.match(url)
            if match:
                content.append(
                    ContentPart(
                        type="image",
                        image=ImageSource(
                            type="base64",
                            media_type=match.group(1),
                            data=match.group(2),
                        ),
                    )
                )
            else:
                content.append(
                    ContentPart(
                        type="image",
                        image=ImageSource(type="url", url=url),
                    )
                )
        return content

    @classmethod
    def _parse_transcript(
        cls,
        transcript_file: Path,
        builder: TrajectoryBuilder,
    ) -> None:
        if not transcript_file.exists():
            builder.add_step(
                source="system",
                message=f"kimi-code: no transcript at {transcript_file}",
                extra={"reason": "no_transcript"},
            )
            return
        for event in cls._json_lines(transcript_file):
            role = event.get("role")
            if role == "assistant":
                tool_calls: list[ToolCall] = []
                for raw_call in event.get("tool_calls") or []:
                    function = raw_call.get("function") or {}
                    raw_arguments = function.get("arguments") or "{}"
                    try:
                        arguments = json.loads(raw_arguments)
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"raw": raw_arguments}
                    tool_calls.append(
                        ToolCall(
                            id=raw_call.get("id") or "",
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

    @staticmethod
    def _json_lines(path: Path):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, OSError):
            return
        for line in raw.splitlines():
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
        if "status_code=429" in transcript or "429" in stderr:
            parts.append("LLM rate-limited")
        if stderr.strip():
            parts.append(f"stderr tail: ...{stderr[-800:]}")
        if transcript.strip():
            parts.append(f"transcript tail: ...{transcript[-800:]}")
        return " | ".join(parts)
