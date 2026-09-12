"""Run DeepSeek Harness's official automation SDK inside an ALE sandbox."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import logging
import os
import signal
import subprocess
import sys
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

from .config import DeepSeekHarnessConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 1.0
_TERM_GRACE_S = 2.0
_SDK_DISTRIBUTION = "deepseek-harness-sdk"
_RUNTIME_DISTRIBUTION = "deepseek-harness-runtime-bin"


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _bundled_runtime_path() -> Path | None:
    try:
        runtime = importlib.import_module("deepseek_harness_runtime")
        path = Path(runtime.bundled_runtime_path())
    except (ImportError, FileNotFoundError, OSError, AttributeError):
        return None
    return path if path.is_file() else None


class DeepSeekHarnessDeployer(BaseAgentDeployer):
    """Linux sandbox deployer for ``deepseek-harness-sdk``."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    @property
    def version(self) -> str | None:
        config: DeepSeekHarnessConfig = self.config  # type: ignore[assignment]
        return config.sdk_version

    async def install(self) -> None:
        config: DeepSeekHarnessConfig = self.config  # type: ignore[assignment]
        if not self.executor.sandbox.is_linux:
            raise RuntimeError(
                "deepseek_harness: the official bundled SDK runtime has no Windows wheel; "
                "use a Linux task image"
            )

        installed = await asyncio.to_thread(_installed_version, _SDK_DISTRIBUTION)
        installed_runtime = await asyncio.to_thread(
            _installed_version,
            _RUNTIME_DISTRIBUTION,
        )
        runtime_path = await asyncio.to_thread(_bundled_runtime_path)
        if (
            installed == config.sdk_version
            and installed_runtime == config.sdk_version
            and runtime_path is not None
        ):
            logger.info(
                "deepseek_harness: SDK/runtime %s already installed (%s)",
                installed,
                runtime_path,
            )
            return

        package = f"{_SDK_DISTRIBUTION}=={config.sdk_version}"
        pip_probe = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            stdin=subprocess.DEVNULL,
        )
        if pip_probe.returncode != 0:
            ensurepip = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-m", "ensurepip", "--upgrade"],
                capture_output=True,
                text=True,
                timeout=120,
                stdin=subprocess.DEVNULL,
            )
            if ensurepip.returncode != 0:
                raise RuntimeError(
                    "deepseek_harness: could not bootstrap pip: "
                    f"{(ensurepip.stderr or ensurepip.stdout or '')[-1000:]}"
                )
        argv = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--upgrade",
        ]
        if installed == config.sdk_version:
            argv.append("--force-reinstall")
        argv.append(package)
        logger.info(
            "deepseek_harness: installing %s "
            "(installed_sdk=%s installed_runtime=%s runtime_path=%s)",
            package,
            installed or "missing",
            installed_runtime or "missing",
            runtime_path or "missing",
        )
        process = await asyncio.to_thread(
            subprocess.run,
            argv,
            capture_output=True,
            text=True,
            timeout=1200,
            stdin=subprocess.DEVNULL,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "deepseek_harness: pip install failed "
                f"(rc={process.returncode}, package={package}): "
                f"{(process.stderr or process.stdout or '')[-1000:]}"
            )

        importlib.invalidate_caches()
        sys.modules.pop("deepseek_harness_runtime", None)
        installed = await asyncio.to_thread(_installed_version, _SDK_DISTRIBUTION)
        installed_runtime = await asyncio.to_thread(
            _installed_version,
            _RUNTIME_DISTRIBUTION,
        )
        runtime_path = await asyncio.to_thread(_bundled_runtime_path)
        if (
            installed != config.sdk_version
            or installed_runtime != config.sdk_version
            or runtime_path is None
        ):
            raise RuntimeError(
                "deepseek_harness: SDK installation verification failed "
                f"(expected={config.sdk_version}, installed_sdk={installed}, "
                f"installed_runtime={installed_runtime}, "
                f"runtime_path={runtime_path or 'missing'})"
            )

    async def launch(self, prompt: str) -> AgentRunResult:
        config: DeepSeekHarnessConfig = self.config  # type: ignore[assignment]
        work_dir = Path(self.executor.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        prompt_file = work_dir / "prompt.txt"
        transcript_file = work_dir / "transcript.jsonl"
        stderr_file = work_dir / "stderr.log"
        pid_file = work_dir / "deepseek-harness.pid"
        session_root = work_dir / "sessions"
        for path in (transcript_file, stderr_file, pid_file):
            path.unlink(missing_ok=True)
        prompt_file.write_text(prompt, encoding="utf-8")

        argv = self._build_argv(
            config,
            prompt_file=prompt_file,
            work_dir=work_dir,
            session_root=session_root,
        )
        env = self._build_env(config, session_root=session_root)
        started = time.monotonic()
        with transcript_file.open("wb") as transcript, stderr_file.open("wb") as stderr:
            process = await asyncio.to_thread(
                subprocess.Popen,
                argv,
                stdin=subprocess.DEVNULL,
                stdout=transcript,
                stderr=stderr,
                env=env,
                cwd=str(work_dir),
                start_new_session=True,
            )
        pid_file.write_text(str(process.pid), encoding="ascii")
        logger.info("deepseek_harness: spawned driver pid=%s", process.pid)

        try:
            while process.poll() is None:
                await asyncio.sleep(_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            self._signal_process_group(process, signal.SIGTERM)
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(process.wait),
                    timeout=_TERM_GRACE_S,
                )
            except (TimeoutError, asyncio.CancelledError):
                self._signal_process_group(process, signal.SIGKILL)
                await asyncio.shield(asyncio.to_thread(process.wait))
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

    @staticmethod
    def _signal_process_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    @staticmethod
    def _build_argv(
        config: DeepSeekHarnessConfig,
        *,
        prompt_file: Path,
        work_dir: Path,
        session_root: Path,
    ) -> list[str]:
        argv = [
            sys.executable,
            "-m",
            "ale_run.agents.deepseek_harness.driver",
            "--prompt-file",
            str(prompt_file),
            "--cwd",
            str(work_dir),
            "--session-root",
            str(session_root),
            "--provider",
            config.provider,
            "--model",
            config.model,
        ]
        if config.max_tokens is not None:
            argv.extend(["--max-tokens", str(config.max_tokens)])
        return argv

    def _build_env(
        self,
        config: DeepSeekHarnessConfig,
        *,
        session_root: Path,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.executor.env or {})
        api_key = config.api_key or env.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                "deepseek_harness: no API key configured; set config.api_key or "
                f"{config.api_key_env}"
            )

        env["DEEPSEEK_API_KEY"] = api_key
        if config.base_url is None:
            env.pop("DEEPSEEK_BASE_URL", None)
        else:
            env["DEEPSEEK_BASE_URL"] = config.base_url

        env["DSH_SESSION_ROOT"] = str(session_root)
        env["DSH_TELEMETRY_DISABLED"] = "1"
        env.pop("DSH_CORDIS_CONFIG", None)
        env.pop("DSH_RUNTIME_MODE", None)
        if config.system_prompt is None:
            env.pop("DSH_SYSTEM_PROMPT", None)
        else:
            env["DSH_SYSTEM_PROMPT"] = config.system_prompt
        env["NO_COLOR"] = "1"
        return env

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: DeepSeekHarnessConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        transcript_file = work_dir / "transcript.jsonl"
        records = list(cls._json_lines(transcript_file))
        if not records:
            builder.add_step(
                source="system",
                message=f"deepseek_harness: no transcript at {transcript_file}",
                extra={"reason": "no_transcript"},
            )
            return

        result = next(
            (record for record in reversed(records) if record.get("type") == "result"),
            {},
        )
        root_session_id = result.get("session_id")
        if not isinstance(root_session_id, str):
            root_session_id = cls._first_session_id(records)

        notification_count = 0
        subagent_count = 0
        for record in records:
            record_type = record.get("type")
            if record_type == "driver_error":
                builder.add_step(
                    source="system",
                    message=str(
                        record.get("message") or record.get("error_type") or "driver error"
                    ),
                    extra={
                        "deepseek_harness_error": True,
                        "error_type": record.get("error_type"),
                    },
                )
                continue
            if record_type != "notification":
                continue
            notification_count += 1
            method = record.get("method")
            params = record.get("params")
            if method == "subagent.started":
                subagent_count += 1
                continue
            if method != "session.event" or not isinstance(params, dict):
                continue
            if root_session_id is not None and params.get("sessionId") != root_session_id:
                continue
            event = params.get("event")
            if isinstance(event, dict):
                cls._consume_session_event(event, builder)

        builder.trajectory.extra.setdefault("deepseek_harness", {}).update(
            {
                "exit_code": run_result.exit_code,
                "transcript_path": str(transcript_file),
                "stderr_path": run_result.stderr_path,
                "session_id": root_session_id,
                "finish_reason": result.get("finish_reason"),
                "notification_count": notification_count,
                "subagent_count": subagent_count,
                "runtime": "python-sdk-bundled-jsonrpc",
            }
        )

    @classmethod
    def _consume_session_event(
        cls,
        event: dict[str, Any],
        builder: TrajectoryBuilder,
    ) -> None:
        event_type = event.get("type")
        data = event.get("data")
        if not isinstance(data, dict):
            return

        if event_type == "assistant/message":
            message = data.get("message")
            owner = message if isinstance(message, dict) else data
            content = owner.get("content")
            text: list[str] = []
            reasoning: list[str] = []
            tool_calls: list[ToolCall] = []
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text" and isinstance(block.get("text"), str):
                    text.append(block["text"])
                elif block_type == "reasoning" and isinstance(block.get("text"), str):
                    reasoning.append(block["text"])
                elif block_type == "tool-call":
                    tool_calls.append(
                        ToolCall(
                            id=str(block.get("id") or ""),
                            name=str(block.get("name") or ""),
                            arguments=cls._tool_arguments(block.get("arguments")),
                        )
                    )

            usage = data.get("usage")
            metrics = None
            extra: dict[str, Any] = {
                "turn": data.get("turn"),
                "step": data.get("step"),
                "event_seq": event.get("seq"),
            }
            if isinstance(usage, dict):
                metrics = StepMetrics(
                    input_tokens=cls._integer_or_none(usage.get("inputTokens")),
                    output_tokens=cls._integer_or_none(usage.get("outputTokens")),
                    cache_read_tokens=cls._integer_or_none(usage.get("cacheReadTokens")),
                    cache_creation_tokens=cls._integer_or_none(usage.get("cacheWriteTokens")),
                )
                extra["usage"] = usage
            if text or reasoning or tool_calls or metrics is not None:
                builder.add_step(
                    source="agent",
                    message="".join(text) or None,
                    reasoning="".join(reasoning) or None,
                    tool_calls=tool_calls,
                    metrics=metrics,
                    extra=extra,
                )
            return

        if event_type == "tool/result":
            message = data.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            results: list[ToolResult] = []
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict) or block.get("type") != "tool-result":
                    continue
                results.append(
                    ToolResult(
                        tool_call_id=str(block.get("toolCallId") or ""),
                        content=cls._content_parts(block.get("content")),
                        is_error=bool(block.get("isError")),
                    )
                )
            if results:
                builder.add_step(
                    source="environment",
                    observation=Observation(results=results),
                    extra={
                        "turn": data.get("turn"),
                        "step": data.get("step"),
                        "event_seq": event.get("seq"),
                    },
                )
            return

        if event_type == "turn/end":
            reason = data.get("reason")
            if not isinstance(reason, dict) or reason.get("kind") != "error":
                return
            error = reason.get("error")
            error_data = error if isinstance(error, dict) else {}
            builder.add_step(
                source="system",
                message=str(error_data.get("message") or "DeepSeek Harness turn failed"),
                extra={
                    "deepseek_harness_error": True,
                    "code": error_data.get("code"),
                    "turn": data.get("turn"),
                    "event_seq": event.get("seq"),
                },
            )

    @staticmethod
    def _first_session_id(records: list[dict[str, Any]]) -> str | None:
        for record in records:
            params = record.get("params")
            if (
                record.get("type") == "notification"
                and record.get("method") == "session.event"
                and isinstance(params, dict)
                and isinstance(params.get("sessionId"), str)
            ):
                return params["sessionId"]
        return None

    @staticmethod
    def _tool_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                return {"raw": raw}
            return decoded if isinstance(decoded, dict) else {"value": decoded}
        return {"value": raw}

    @classmethod
    def _content_parts(cls, content: Any) -> list[ContentPart]:
        if not isinstance(content, list):
            return [ContentPart(type="text", text=str(content))]
        parts: list[ContentPart] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(ContentPart(type="text", text=str(block)))
                continue
            if block.get("type") == "text":
                parts.append(ContentPart(type="text", text=str(block.get("text") or "")))
            elif block.get("type") == "image" and isinstance(block.get("data"), str):
                parts.append(
                    ContentPart(
                        type="image",
                        image=ImageSource(
                            type="base64",
                            data=block["data"],
                            media_type=str(block.get("mimeType") or "image/png"),
                        ),
                    )
                )
            else:
                parts.append(ContentPart(type="text", text=json.dumps(block, ensure_ascii=False)))
        return parts

    @staticmethod
    def _integer_or_none(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _json_lines(path: Path):
        try:
            handle = path.open(encoding="utf-8", errors="replace")
        except (FileNotFoundError, OSError):
            return
        with handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record

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
            f"deepseek_harness failed (rc={exit_code})",
            f"stderr={len(stderr)}B transcript={len(transcript)}B",
        ]
        if stderr.strip():
            parts.append(f"stderr tail: ...{stderr[-1000:]}")
        if transcript.strip():
            parts.append(f"transcript tail: ...{transcript[-1000:]}")
        return " | ".join(parts)
