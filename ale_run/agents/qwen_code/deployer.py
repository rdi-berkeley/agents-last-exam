"""Run the pinned Qwen Code CLI with OpenAI-compatible auth and the CUA MCP bridge."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import tempfile
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

from ale_run.agents._cli_runtime import (
    redact,
    ensure_cli_node,
    run_command,
    secret_values,
    wait_process,
)

from .config import QwenCodeConfig

logger = logging.getLogger(__name__)

_TERM_GRACE_S = 3.0


class QwenCodeDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the ``@qwen-code/qwen-code`` npm package."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("stdout.log", "stderr.log")

    @property
    def version(self) -> str | None:
        cfg: QwenCodeConfig = self.config  # type: ignore[assignment]
        return cfg.cli_version

    async def install(self) -> None:
        cfg: QwenCodeConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        if not sandbox.is_linux:
            raise NotImplementedError("qwen_code is Linux-only")

        if not cfg.api_key:
            raise RuntimeError(
                "qwen_code: config.api_key is required (e.g. "
                "api_key: ${env:HF_INFERENCE_TOKEN} in the agent yaml) -- there is no "
                "provider-name env-var fallback to read a key from."
            )

        from ale_run.agents._bootstrap import ensure_node_npm

        node, npm = await ensure_node_npm()
        node = await ensure_cli_node(node, npm, (22, 0, 0))
        self._node_path = node

        qwen_path = shutil.which("qwen")
        installed = None
        if qwen_path:
            probe = await run_command([qwen_path, "--version"], timeout=30)
            installed = probe.stdout.strip() if probe.returncode == 0 else None
        if not qwen_path or installed != cfg.cli_version:
            logger.info("qwen_code: 'qwen' not on PATH, installing v%s ...", cfg.cli_version)
            await self._install_cli(npm, cfg.cli_version)
            qwen_path = shutil.which("qwen")
            if not qwen_path:
                raise RuntimeError("QwenCodeDeployer: 'qwen' still not found after install")
        self._qwen_path = qwen_path

        try:
            probe = await run_command(
                [qwen_path, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except TimeoutError as e:
            raise RuntimeError(f"qwen --version timed out: {e}")
        if probe.returncode != 0 or probe.stdout.strip() != cfg.cli_version:
            raise RuntimeError("qwen_code: installed CLI does not match cli_version")
        logger.info("qwen_code: CLI ok -- %s", (probe.stdout or "").strip())

        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        self._home_dir = tempfile.mkdtemp(prefix=".ale-qwen-", dir=wd.parent)

        from ale_run.agents._bootstrap import ensure_cua_mcp_server

        await ensure_cua_mcp_server(sandbox)
        mcp_config = self._build_mcp_config(sandbox, self.executor.cua_bridge_url())
        mcp_index = mcp_config["mcpServers"]["cua"]["args"][0]
        self._mcp_config_json = json.dumps(mcp_config) if os.path.exists(mcp_index) else None
        if self._mcp_config_json is None:
            logger.warning(
                "qwen_code: CUA MCP server not found at %s -- skipping "
                "--mcp-config; qwen will use built-in tools only",
                mcp_index,
            )

        logger.info("qwen_code: install complete (model=%s base_url=%s)", cfg.model, cfg.base_url)

    async def _install_cli(self, npm: str, version: str) -> None:
        pkg = f"@qwen-code/qwen-code@{version}"
        proc = await run_command(
            [npm, "install", "-g", pkg],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"qwen_code: npm install -g {pkg} failed (rc={proc.returncode}): "
                f"stdout: {(proc.stdout or '')[-800:]} "
                f"stderr: {(proc.stderr or '')[-800:]}"
            )
        logger.info("qwen_code: installed via npm -- %s", (proc.stdout or "").strip()[-200:])

    @staticmethod
    def _build_mcp_config(sandbox: Any, cua_url: str) -> dict:
        """Build the ``--mcp-config`` payload for the CUA MCP bridge."""
        node_exe = sandbox.node
        mcp_server_dir = sandbox.mcp_server_dir
        is_linux = sandbox.is_linux

        sep = "/" if is_linux else "\\"
        mcp_index = f"{mcp_server_dir.rstrip('/\\ ')}{sep}src{sep}index.js"

        return {
            "mcpServers": {
                "cua": {
                    "command": node_exe,
                    "args": [mcp_index],
                    "env": {"CUA_SERVER_URL": cua_url},
                },
            },
        }

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: QwenCodeConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        stdout_log = wd / "stdout.log"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "qwen_code.pid"
        for f in (stdout_log, stderr_log, pid_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        argv = self._build_argv(prompt)
        env = self._build_env(cfg)

        t0 = time.monotonic()
        with open(stdout_log, "wb") as fout, open(stderr_log, "wb") as ferr:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=fout,
                stderr=ferr,
                env=env,
                cwd=str(wd),
                start_new_session=True if hasattr(os, "setsid") else False,
            )
        pid_file.write_text(str(proc.pid), encoding="ascii")
        logger.info(
            "qwen_code: spawned pid=%s (model=%s base_url=%s)", proc.pid, cfg.model, cfg.base_url
        )

        await wait_process(proc, grace_s=_TERM_GRACE_S)

        duration_s = time.monotonic() - t0
        exit_code = proc.returncode
        status = "completed" if exit_code == 0 else "failed"
        error: str | None = None
        if status == "failed":
            error = _diagnose_failure(stderr_log, stdout_log, exit_code)

        return AgentRunResult(
            status=status,
            pid=proc.pid,
            exit_code=exit_code,
            transcript_path=str(stdout_log),
            stderr_path=str(stderr_log),
            duration_s=duration_s,
            error=redact(error, secret_values(cfg, self.executor.env)),
        )

    _BROKEN_SCHEMA_TOOLS = ("get_goal", "list_agents")

    def _build_argv(self, prompt: str) -> list[str]:
        argv = [
            self._qwen_path,
            "-y",
            "--exclude-tools",
            ",".join(self._BROKEN_SCHEMA_TOOLS),
            "--output-format",
            "stream-json",
        ]
        if getattr(self, "_mcp_config_json", None):
            argv += ["--mcp-config", self._mcp_config_json]
        argv += [f"--prompt={prompt}"]
        return argv

    def _build_env(self, cfg: QwenCodeConfig) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env["OPENAI_API_KEY"] = cfg.api_key or ""
        env["OPENAI_BASE_URL"] = cfg.base_url
        env["OPENAI_MODEL"] = cfg.model
        env["QWEN_CODE_SUPPRESS_YOLO_WARNING"] = "1"
        env["NO_COLOR"] = "1"
        env["HOME"] = getattr(self, "_home_dir", os.path.expanduser("~"))
        for k, v in cfg.extra_envs.items():
            env[k] = v
        if getattr(self, "_node_path", None):
            env["PATH"] = str(Path(self._node_path).parent) + os.pathsep + env.get("PATH", "")
        return env

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: QwenCodeConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Parse NDJSON messages and authoritative cumulative usage."""
        stdout_log = work_dir / "stdout.log"
        if not stdout_log.exists():
            builder.add_step(
                source="system",
                message=f"qwen_code: no stdout at {stdout_log}",
                extra={"reason": "no_transcript"},
            )
            return

        raw = redact(
            stdout_log.read_text(encoding="utf-8", errors="replace"), secret_values(config)
        )
        events, result_summary = cls._parse_stream_json(raw)

        transcript_jsonl = work_dir / "transcript.jsonl"
        transcript_jsonl.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + ("\n" if events else ""),
            encoding="utf-8",
        )

        if not events and not result_summary:
            builder.add_step(
                source="system",
                message="qwen_code: no JSON events parsed from stdout",
                extra={"reason": "no_events"},
            )
            builder.trajectory.extra.setdefault("qwen_code", {}).update(
                {
                    "exit_code": run_result.exit_code,
                    "transcript_path": str(stdout_log),
                }
            )
            return

        for ev in events:
            cls._consume_event(ev, builder)

        if result_summary:
            usage = (
                result_summary.get("usage")
                if isinstance(result_summary.get("usage"), dict)
                else None
            )
            metrics = None
            if usage:
                metrics = StepMetrics(
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    cache_read_tokens=usage.get("cache_read_input_tokens"),
                    cache_creation_tokens=usage.get("cache_creation_input_tokens"),
                    cost_usd=result_summary.get("total_cost_usd"),
                )
            if metrics:
                builder.override_final_metrics(
                    total_input_tokens=metrics.input_tokens,
                    total_output_tokens=metrics.output_tokens,
                    total_cache_read_tokens=metrics.cache_read_tokens,
                    total_cache_creation_tokens=metrics.cache_creation_tokens,
                    total_cost_usd=metrics.cost_usd,
                )
            builder.add_step(
                source="system",
                message=None,
                extra={
                    "result_subtype": result_summary.get("subtype"),
                    "num_turns": result_summary.get("num_turns"),
                    "duration_ms": result_summary.get("duration_ms"),
                    "is_error": result_summary.get("is_error"),
                },
            )

            response = result_summary.get("result")
            if (
                isinstance(response, str)
                and response
                and not any(
                    step.source == "agent" and step.message == response
                    for step in builder.trajectory.steps
                )
            ):
                builder.add_step(source="agent", message=response)

        builder.trajectory.extra.setdefault("qwen_code", {}).update(
            {
                "exit_code": run_result.exit_code,
                "transcript_path": str(stdout_log),
                "event_count": len(events),
                "result_summary": result_summary,
            }
        )

    @staticmethod
    def _parse_stream_json(stdout_text: str) -> tuple[list[dict], dict | None]:
        """Separate stream events from the terminal result summary."""
        events: list[dict] = []
        result_summary: dict | None = None
        for line in stdout_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            if parsed.get("type") == "result":
                result_summary = parsed
                continue
            events.append(parsed)
        return events, result_summary

    @classmethod
    def _consume_event(cls, event: dict, builder: TrajectoryBuilder) -> None:
        top_type = str(event.get("type") or "").strip()

        if top_type in ("system", "stream_event"):
            return

        message = event.get("message") if isinstance(event.get("message"), dict) else None
        if top_type == "assistant" and message:
            cls._consume_assistant_message(message, builder)
            return

        if top_type == "user" and message:
            cls._consume_user_message(message, builder)
            return

        builder.add_step(
            source="system",
            message="",
            extra={"kind": top_type or "unknown", "raw": event},
        )

    @classmethod
    def _consume_assistant_message(cls, message: dict, builder: TrajectoryBuilder) -> None:
        content = message.get("content")
        if not isinstance(content, list):
            return
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else None
        metrics = None
        if usage and (usage.get("input_tokens") or usage.get("output_tokens")):
            metrics = StepMetrics(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_read_tokens=usage.get("cache_read_input_tokens") or None,
            )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            elif btype == "thinking" and isinstance(block.get("thinking"), str):
                reasoning_parts.append(block["thinking"])
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id") or ""),
                        name=str(block.get("name") or "unknown"),
                        arguments=block.get("input")
                        if isinstance(block.get("input"), dict)
                        else {},
                    )
                )

        text = "\n".join(text_parts).strip()
        reasoning = "\n".join(reasoning_parts).strip()
        if not (text or reasoning or tool_calls):
            return
        builder.add_step(
            source="agent",
            message=text or None,
            reasoning=reasoning or None,
            tool_calls=tool_calls or None,
            metrics=metrics,
        )

    @classmethod
    def _consume_user_message(cls, message: dict, builder: TrajectoryBuilder) -> None:
        content = message.get("content")
        if not isinstance(content, list):
            return
        results: list[ToolResult] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            raw_content = block.get("content")
            text = (
                raw_content
                if isinstance(raw_content, str)
                else (
                    json.dumps(raw_content, ensure_ascii=False) if raw_content is not None else ""
                )
            )
            parts = []
            if isinstance(raw_content, list):
                for item in raw_content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(ContentPart(type="text", text=item["text"]))
                    elif item.get("type") == "image" and isinstance(item.get("source"), dict):
                        source = item["source"]
                        if source.get("type") == "base64" and isinstance(source.get("data"), str):
                            parts.append(
                                ContentPart(
                                    type="image",
                                    image=ImageSource(
                                        type="base64",
                                        data=source["data"],
                                        media_type=source.get("media_type"),
                                    ),
                                )
                            )
            elif text:
                parts = [ContentPart(type="text", text=text)]
            results.append(
                ToolResult(
                    tool_call_id=str(block.get("tool_use_id") or ""),
                    content=parts,
                    is_error=bool(block.get("is_error")),
                )
            )
        if not results:
            return
        builder.add_step(
            source="environment",
            observation=Observation(results=results),
        )


def _diagnose_failure(stderr_log: Path, stdout_log: Path, exit_code: int | None) -> str:
    parts = [f"agent failed (rc={exit_code})"]
    stderr_text = _read_text_tolerant(stderr_log)
    stdout_text = _read_text_tolerant(stdout_log)
    if stderr_text.strip():
        parts.append(f"stderr tail: ...{stderr_text[-800:]}")
    if stdout_text.strip():
        parts.append(f"stdout tail: ...{stdout_text[-800:]}")
    return " | ".join(parts)


def _read_text_tolerant(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""
