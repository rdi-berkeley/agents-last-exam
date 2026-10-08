"""QwenCodeDeployer -- drives the official ``@qwen-code/qwen-code`` npm package.

Installed via ``npm install -g @qwen-code/qwen-code@<version>``. Published
package, no from-source build needed (unlike zcode).

Headless invocation::

    qwen -y --exclude-tools get_goal,list_agents --output-format stream-json \
        [--mcp-config <json>] "<prompt>"

* Positional prompt (NOT ``-p``/``--prompt`` -- that flag is deprecated in
  favour of the positional form, per the CLI's own ``--help``).
* ``-y``/``--yolo`` auto-approves every tool call.
* Deliberately NOT passing ``--bare``: an earlier version of this deployer
  did, believing it only trimmed ambient noise from a contaminated-looking
  local test. Root-caused (via ``--openai-logging`` raw-traffic capture)
  that it ALSO silently drops core built-in tools -- ``write_file``,
  ``glob``, ``grep_search``, ``skill``, ``tool_search`` -- that the CLI's own
  system-prompt few-shot examples (``packages/core/src/core/prompts.ts``)
  reference regardless of ``--bare``. That mismatch was the real cause of a
  near-total 0-score result on a full 22-task run: with no ``write_file``
  tool available, the model (which still "knows" the tool name from those
  examples) emitted a textual pseudo-tool-call
  (``<function=write_file>\n<parameter=file_path>...``) that nothing parses
  back into a real call, and the turn dead-ends. Confirmed by direct A/B:
  the identical failing prompt succeeds cleanly (real ``write_file`` tool_use,
  real file written) the moment ``--bare`` is dropped. (The larger,
  ~40-tool set seen once ``--bare`` is off -- cron/worktree/goal/monitor/
  send_message/etc, alongside the plain built-ins -- is qwen-code's own
  bundled extension set, auto-installed to ``~/.qwen/extensions`` on
  startup; it is NOT contamination from any host-side IDE/agent process,
  and it appears the same way inside a real, freshly-provisioned ALE
  sandbox as it did locally.)
* ``--exclude-tools get_goal,list_agents``: both ship with a malformed
  OpenAI function schema in v0.24.4 -- ``{"type":"function","function":
  {"name":...,"description":...}}`` with NO ``parameters`` key at all (every
  other built-in tool includes one correctly, even when empty). The HF
  router's strict request deserializer rejects the ENTIRE request with a
  422 (``tools[N].function: missing field 'parameters'``) the instant it
  reaches either tool's entry, killing every turn from that point on --
  confirmed via ``--openai-logging`` (4/4 identical local repeats hit it
  with both tools present; 4/4 clean with them excluded). This is a genuine
  qwen-code bug (a zero-argument-tool schema-generation gap), not a server
  or deployer issue. Excluding them costs nothing for ALE tasks, which never
  use qwen-code's interactive Goal-tracking or multi-agent-delegation
  features.
* ``--output-format stream-json`` emits one JSON object per line in the same
  wire shape as Claude Code's own ``--output-format stream-json`` (the
  project's README states explicit parity-with-Claude-Code as a design
  goal, and the wire schema confirms it): ``{"type":"system","subtype":
  "init",...}`` once at start, ``{"type":"assistant",...}``/``{"type":
  "user",...}`` per turn (content blocks: ``text``/``thinking``/
  ``tool_use``/``tool_result``), and a final ``{"type":"result",
  "subtype":"success", result, usage, ...}`` line.

Auth is entirely env-var driven -- no login step, no on-disk provider
registry (unlike zcode): ``OPENAI_API_KEY`` + ``OPENAI_MODEL`` +
``OPENAI_BASE_URL`` set together make the CLI auto-infer
``AuthType.USE_OPENAI`` (confirmed by reading ``getAuthTypeFromEnv`` in
``packages/cli/src/utils/modelConfigUtils.ts``).

Bridge: the agent talks to the sandbox through the CUA MCP Server bridge
baked into sandbox images, wired via ``--mcp-config`` (inline JSON, no file
write needed -- the flag accepts either a path or an inline JSON string).

Trajectory recovery: stdout is the source of truth -- one JSON object per
line, parsed directly (no delimiter-splitting needed, unlike openhands_cli's
older CLI versions).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar

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

from .config import QwenCodeConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 5.0
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

    # =========================================================================
    # install
    # =========================================================================

    async def install(self) -> None:
        cfg: QwenCodeConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        if not sandbox.is_linux:
            raise NotImplementedError("qwen_code is Linux-only")

        if not cfg.api_key:
            raise RuntimeError(
                "qwen_code: config.api_key is required (e.g. "
                "api_key: ${env:HF_TOKEN} in the agent yaml) -- there is no "
                "provider-name env-var fallback to read a key from."
            )

        from ale_run.agents._bootstrap import ensure_node_npm
        node, npm = await ensure_node_npm()
        self._node_path = node

        qwen_path = shutil.which("qwen")
        if not qwen_path:
            logger.info("qwen_code: 'qwen' not on PATH, installing v%s ...",
                        cfg.cli_version)
            await self._install_cli(npm, cfg.cli_version)
            qwen_path = shutil.which("qwen")
            if not qwen_path:
                raise RuntimeError(
                    "QwenCodeDeployer: 'qwen' still not found after install"
                )
        self._qwen_path = qwen_path

        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [qwen_path, "--version"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"qwen --version timed out: {e}")
        logger.info("qwen_code: CLI ok -- %s", (probe.stdout or "").strip())

        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # CUA MCP bridge -- only if the stdio server actually exists in this
        # image (see openhands_cli's identical guard: pointing --mcp-config
        # at a missing script blocks MCP client init instead of falling
        # back to built-in tools).
        from ale_run.agents._bootstrap import ensure_cua_mcp_server
        await ensure_cua_mcp_server(sandbox)
        mcp_config = self._build_mcp_config(sandbox, self.executor.cua_bridge_url())
        mcp_index = mcp_config["mcpServers"]["cua"]["args"][0]
        self._mcp_config_json = (
            json.dumps(mcp_config) if os.path.exists(mcp_index) else None
        )
        if self._mcp_config_json is None:
            logger.warning(
                "qwen_code: CUA MCP server not found at %s -- skipping "
                "--mcp-config; qwen will use built-in tools only",
                mcp_index,
            )

        logger.info("qwen_code: install complete (model=%s base_url=%s)",
                     cfg.model, cfg.base_url)

    async def _install_cli(self, npm: str, version: str) -> None:
        pkg = f"@qwen-code/qwen-code@{version}"
        proc = await asyncio.to_thread(
            subprocess.run,
            [npm, "install", "-g", pkg],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"qwen_code: npm install -g {pkg} failed (rc={proc.returncode}): "
                f"stdout: {(proc.stdout or '')[-800:]} "
                f"stderr: {(proc.stderr or '')[-800:]}"
            )
        logger.info("qwen_code: installed via npm -- %s",
                     (proc.stdout or "").strip()[-200:])

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

    # =========================================================================
    # launch
    # =========================================================================

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
            proc = await asyncio.to_thread(
                subprocess.Popen,
                argv,
                stdin=subprocess.DEVNULL,
                stdout=fout,
                stderr=ferr,
                env=env,
                cwd=str(wd),
                start_new_session=True if hasattr(os, "setsid") else False,
            )
        pid_file.write_text(str(proc.pid), encoding="ascii")
        logger.info("qwen_code: spawned pid=%s (model=%s base_url=%s)",
                     proc.pid, cfg.model, cfg.base_url)

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
            error = _diagnose_failure(stderr_log, stdout_log, exit_code)

        return AgentRunResult(
            status=status,
            pid=proc.pid,
            exit_code=exit_code,
            transcript_path=str(stdout_log),
            stderr_path=str(stderr_log),
            duration_s=duration_s,
            error=error,
        )

    # Tool names confirmed (via --openai-logging raw-traffic capture, 4/4
    # repeat local runs) to ship with a malformed OpenAI function schema --
    # {"type":"function","function":{"name":...,"description":...}} with NO
    # "parameters" key at all, not even an empty {"type":"object"} one. Every
    # OTHER built-in tool (read_file, write_file, edit, ...) includes
    # "parameters" correctly; this only affects genuinely zero-argument
    # tools. The HF router's strict request deserializer rejects the ENTIRE
    # request with a 422 the moment it hits one of these, killing the whole
    # turn -- confirmed this is a real qwen-code v0.24.4 bug, not a server or
    # deployer issue (identical tool list, minus these two, sends cleanly).
    # Excluded via --exclude-tools since ALE tasks never use qwen-code's
    # interactive Goal-tracking or multi-agent-delegation features anyway.
    _BROKEN_SCHEMA_TOOLS = ("get_goal", "list_agents")

    def _build_argv(self, prompt: str) -> list[str]:
        argv = [
            self._qwen_path,
            "-y",
            "--exclude-tools", ",".join(self._BROKEN_SCHEMA_TOOLS),
            "--output-format", "stream-json",
        ]
        if getattr(self, "_mcp_config_json", None):
            argv += ["--mcp-config", self._mcp_config_json]
        argv.append(prompt)
        return argv

    def _build_env(self, cfg: QwenCodeConfig) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env["OPENAI_API_KEY"] = cfg.api_key or ""
        env["OPENAI_BASE_URL"] = cfg.base_url
        env["OPENAI_MODEL"] = cfg.model
        # Silences a stderr warning line about running --yolo without a
        # sandbox -- expected and intentional in ALE's own sandboxed
        # execution, not something worth polluting stderr diagnostics with.
        env["QWEN_CODE_SUPPRESS_YOLO_WARNING"] = "1"
        env["NO_COLOR"] = "1"
        env["HOME"] = os.path.expanduser("~")
        for k, v in cfg.extra_envs.items():
            env[k] = v
        return env

    # =========================================================================
    # parse_artifacts
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: QwenCodeConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Parse Qwen Code's ``--output-format stream-json`` NDJSON stdout.

        Wire shape confirmed via real local runs (Claude-Code-SDK-parity
        schema, not guessed): ``system``/``stream_event`` are framework
        bookkeeping (dropped); ``assistant``/``user`` messages carry content
        blocks (``text``, ``thinking``, ``tool_use``, ``tool_result``); the
        final ``result`` line carries the run summary + cumulative usage.
        """
        stdout_log = work_dir / "stdout.log"
        if not stdout_log.exists():
            builder.add_step(
                source="system",
                message=f"qwen_code: no stdout at {stdout_log}",
                extra={"reason": "no_transcript"},
            )
            return

        raw = stdout_log.read_text(encoding="utf-8", errors="replace")
        events, result_summary = cls._parse_stream_json(raw)

        transcript_jsonl = work_dir / "transcript.jsonl"
        transcript_jsonl.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in events)
            + ("\n" if events else ""),
            encoding="utf-8",
        )

        if not events and not result_summary:
            builder.add_step(
                source="system",
                message="qwen_code: no JSON events parsed from stdout",
                extra={"reason": "no_events"},
            )
            builder.trajectory.extra.setdefault("qwen_code", {}).update({
                "exit_code": run_result.exit_code,
                "transcript_path": str(stdout_log),
            })
            return

        for ev in events:
            cls._consume_event(ev, builder)

        if result_summary:
            usage = result_summary.get("usage") if isinstance(result_summary.get("usage"), dict) else None
            metrics = None
            if usage:
                metrics = StepMetrics(
                    prompt_tokens=usage.get("input_tokens"),
                    completion_tokens=usage.get("output_tokens"),
                    cached_tokens=usage.get("cache_read_input_tokens") or None,
                )
            builder.add_step(
                source="system",
                message=None,
                metrics=metrics,
                extra={
                    "result_subtype": result_summary.get("subtype"),
                    "num_turns": result_summary.get("num_turns"),
                    "duration_ms": result_summary.get("duration_ms"),
                    "is_error": result_summary.get("is_error"),
                },
            )

        builder.trajectory.extra.setdefault("qwen_code", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(stdout_log),
            "event_count": len(events),
            "result_summary": result_summary,
        })

    @staticmethod
    def _parse_stream_json(stdout_text: str) -> tuple[list[dict], dict | None]:
        """Split ``--output-format stream-json`` NDJSON stdout.

        Each non-blank line is one JSON object; the final
        ``{"type":"result",...}`` line is the run summary and is pulled out
        separately rather than treated as a generic event.
        """
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
            # "system"/"init" is a one-time capability manifest (tool/MCP/
            # slash-command list) and "stream_event" wraps internal
            # bookkeeping (e.g. goal_state) -- neither carries
            # trajectory-relevant content.
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
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                cached_tokens=usage.get("cache_read_input_tokens") or None,
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
                tool_calls.append(ToolCall(
                    tool_call_id=str(block.get("id") or ""),
                    function_name=str(block.get("name") or "unknown"),
                    arguments=block.get("input") if isinstance(block.get("input"), dict) else {},
                ))

        text = "\n".join(text_parts).strip()
        reasoning = "\n".join(reasoning_parts).strip()
        if not (text or reasoning or tool_calls):
            return
        builder.add_step(
            source="agent",
            message=text or None,
            reasoning_content=reasoning or None,
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
            text = raw_content if isinstance(raw_content, str) else (
                json.dumps(raw_content, ensure_ascii=False) if raw_content is not None else ""
            )
            parts = [ContentPart(type="text", text=text)] if text else []
            results.append(ToolResult(
                source_call_id=str(block.get("tool_use_id") or ""),
                content=parts,
                is_error=bool(block.get("is_error")),
            ))
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
