"""Run the pi (earendil-works/pi) CLI inside an ALE sandbox."""

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
    ImageSource,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

from .config import PiCliConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0
_NPM_PACKAGE = "@earendil-works/pi-coding-agent"

_DIRECT_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "huggingface": "HF_TOKEN",
}


class PiCliDeployer(BaseAgentDeployer):
    """Sandbox deployer for ``@earendil-works/pi-coding-agent``."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    @property
    def version(self) -> str | None:
        cfg: PiCliConfig = self.config  # type: ignore[assignment]
        return cfg.cli_version

    async def install(self) -> None:
        cfg: PiCliConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        if not sandbox.is_linux:
            raise NotImplementedError("pi_cli is Linux-only")

        from ale_run.agents._bootstrap import ensure_node_npm
        _node, npm = await ensure_node_npm()

        pi_path = shutil.which("pi")
        installed_version = _installed_version(pi_path) if pi_path else None
        if not pi_path or installed_version != cfg.cli_version:
            proc = await asyncio.to_thread(
                subprocess.run,
                [npm, "install", "-g", "--ignore-scripts",
                 f"{_NPM_PACKAGE}@{cfg.cli_version}"],
                capture_output=True, text=True, timeout=300,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"pi_cli: npm install failed (rc={proc.returncode}): "
                    f"{(proc.stderr or '')[:500]}"
                )
            pi_path = shutil.which("pi")
            if not pi_path:
                raise RuntimeError("PiCliDeployer: 'pi' still not found after install")
        self._pi_path = pi_path

        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        home = os.path.expanduser("~")
        pi_agent_dir = Path(home) / ".pi" / "agent"
        pi_agent_dir.mkdir(parents=True, exist_ok=True)

        auth_file = pi_agent_dir / "auth.json"
        if auth_file.exists():
            try:
                auth_file.unlink()
            except OSError:
                pass

        if cfg.provider == "custom":
            if not cfg.base_url:
                raise RuntimeError("pi_cli: provider=custom requires base_url")
            exec_env = dict(self.executor.env or {})
            key = cfg.api_key or exec_env.get("OPENROUTER_API_KEY") or exec_env.get("HF_TOKEN")
            if not key:
                raise RuntimeError(
                    "pi_cli: provider=custom but no api_key/OPENROUTER_API_KEY/HF_TOKEN set"
                )
            models_json = {
                "providers": {
                    "custom": {
                        "baseUrl": cfg.base_url,
                        "api": "openai-completions",
                        "apiKey": key,
                        "models": [{"id": cfg.model}],
                    },
                },
            }
            (pi_agent_dir / "models.json").write_text(
                json.dumps(models_json, indent=2), encoding="utf-8",
            )
        else:
            models_json_file = pi_agent_dir / "models.json"
            if models_json_file.exists():
                try:
                    models_json_file.unlink()
                except OSError:
                    pass

        from ale_run.agents._bootstrap import cua_bridge_env, ensure_cua_mcp_server
        await ensure_cua_mcp_server(sandbox)
        mcp_index = f"{sandbox.mcp_server_dir.rstrip('/')}/src/index.js"
        mcp_dir = Path(home) / ".config" / "mcp"
        mcp_path = mcp_dir / "mcp.json"
        if os.path.exists(mcp_index):
            mcp_dir.mkdir(parents=True, exist_ok=True)
            mcp_config = {
                "mcpServers": {
                    "cua": {
                        "command": sandbox.node,
                        "args": [mcp_index],
                        "env": cua_bridge_env(self.executor),
                    },
                },
            }
            mcp_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")

            adapter_probe = await asyncio.to_thread(
                subprocess.run,
                [self._pi_path, "list"],
                capture_output=True, text=True, timeout=30,
            )
            if "pi-mcp-adapter" not in (adapter_probe.stdout or ""):
                install_proc = await asyncio.to_thread(
                    subprocess.run,
                    [self._pi_path, "install", "npm:pi-mcp-adapter"],
                    capture_output=True, text=True, timeout=120,
                )
                if install_proc.returncode != 0:
                    logger.warning(
                        "pi_cli: pi-mcp-adapter install failed (rc=%d): %s",
                        install_proc.returncode, (install_proc.stderr or "")[:400],
                    )
        else:
            if mcp_path.exists():
                try:
                    mcp_path.unlink()
                except OSError:
                    pass

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: PiCliConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        prompt_file = wd / "prompt.txt"
        transcript_file = wd / "transcript.jsonl"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "pi.pid"
        exit_code_file = wd / "exit_code.txt"

        for f in (transcript_file, stderr_log, pid_file, exit_code_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        prompt_file.write_text(prompt, encoding="utf-8")

        argv = self._build_argv(cfg, str(prompt_file))
        env = self._build_env(cfg)

        t0 = time.monotonic()
        with open(transcript_file, "wb") as tout, \
             open(stderr_log, "wb") as terr:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                argv,
                stdin=subprocess.DEVNULL,
                stdout=tout,
                stderr=terr,
                env=env,
                cwd=str(wd),
                start_new_session=True if hasattr(os, "setsid") else False,
            )
        pid_file.write_text(str(proc.pid), encoding="ascii")

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
        exit_code_file.write_text(str(exit_code), encoding="utf-8")

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

    def _build_argv(self, cfg: PiCliConfig, prompt_file: str) -> list[str]:
        pi_bin = getattr(self, "_pi_path", "pi")

        flags: list[str] = ["-p", "--mode", "json", "--no-session", "-na"]

        if cfg.provider == "openrouter":
            flags += ["--provider", "openrouter", "--model", cfg.model]
        elif cfg.provider == "direct":
            vendor, _, model_id = cfg.model.partition("/")
            if not model_id:
                raise RuntimeError(
                    f"pi_cli: provider=direct requires '<vendor>/<id>', got {cfg.model!r}"
                )
            flags += ["--provider", vendor, "--model", model_id]
        elif cfg.provider == "custom":
            flags += ["--provider", "custom", "--model", cfg.model]
        else:
            raise RuntimeError(f"pi_cli: unknown provider {cfg.provider!r}")

        if cfg.thinking_level:
            flags += ["--thinking", cfg.thinking_level]
        if cfg.disabled_tools:
            flags += ["--exclude-tools", ",".join(cfg.disabled_tools)]
        flags += list(cfg.extra_args)

        flag_str = " ".join(_quote(f) for f in flags)
        return [
            "bash", "-c",
            f'PROMPT="$(cat {_quote(prompt_file)})"; '
            f'exec {_quote(pi_bin)} {flag_str} -- "$PROMPT"',
        ]

    def _build_env(self, cfg: PiCliConfig) -> dict[str, str]:
        env = os.environ.copy()
        exec_env = dict(self.executor.env or {})
        for k, v in exec_env.items():
            env[k] = v

        def _key(name: str) -> str:
            return exec_env.get(name, "")

        if cfg.provider == "openrouter":
            key = cfg.api_key or _key("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("pi_cli: provider=openrouter but no api_key/OPENROUTER_API_KEY")
            env["OPENROUTER_API_KEY"] = key
        elif cfg.provider == "direct":
            vendor = cfg.model.split("/", 1)[0].lower()
            env_var = _DIRECT_ENV_VAR.get(vendor)
            if env_var is None:
                raise RuntimeError(f"pi_cli: unrecognized direct vendor {vendor!r}")
            key = cfg.api_key or _key(env_var)
            if not key:
                raise RuntimeError(f"pi_cli: provider=direct but no api_key/{env_var}")
            env[env_var] = key

        env["PI_OFFLINE"] = "1"
        env["PI_SKIP_VERSION_CHECK"] = "1"
        env["NO_COLOR"] = "1"
        env["HOME"] = os.path.expanduser("~")

        home = os.path.expanduser("~")
        path = env.get("PATH", "")
        for extra in (f"{home}/.npm-global/bin", f"{home}/.local/bin"):
            if extra not in path:
                path = f"{extra}:{path}"
        env["PATH"] = path

        return env

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: PiCliConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        transcript_file = work_dir / "transcript.jsonl"
        if not transcript_file.exists():
            builder.add_step(
                source="system",
                message=f"pi_cli: no transcript at {transcript_file}",
                extra={"reason": "no_transcript"},
            )
            return

        raw = transcript_file.read_text(encoding="utf-8", errors="replace")
        events: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)

        if not events:
            builder.add_step(
                source="system",
                message="pi_cli: no JSON events parsed from transcript",
                extra={"reason": "no_events"},
            )
            builder.trajectory.extra.setdefault("pi_cli", {}).update({
                "exit_code": run_result.exit_code,
                "transcript_path": str(transcript_file),
            })
            return

        total_input = total_output = total_cache_read = total_cache_write = 0
        total_cost = 0.0
        cost_seen = usage_seen = False

        for ev in events:
            etype = ev.get("type")

            if etype == "message_end":
                usage = cls._consume_message(ev.get("message"), builder)
                if usage:
                    usage_seen = True
                    total_input += usage.get("input_tokens") or 0
                    total_output += usage.get("output_tokens") or 0
                    total_cache_read += usage.get("cache_read_tokens") or 0
                    total_cache_write += usage.get("cache_write_tokens") or 0
                    if usage.get("cost_usd") is not None:
                        cost_seen = True
                        total_cost += usage["cost_usd"]

            elif etype == "tool_execution_start":
                builder.add_step(
                    source="agent",
                    tool_calls=[ToolCall(
                        id=str(ev.get("toolCallId") or ""),
                        name=str(ev.get("toolName") or ""),
                        arguments=_as_dict(ev.get("args")),
                    )],
                )

            elif etype == "tool_execution_end":
                content = cls._result_to_content(ev.get("result"))
                builder.add_step(
                    source="environment",
                    observation=Observation(results=[
                        ToolResult(
                            tool_call_id=str(ev.get("toolCallId") or ""),
                            content=content,
                            is_error=bool(ev.get("isError")),
                        ),
                    ]),
                )

        if usage_seen or cost_seen:
            builder.add_step(
                source="system",
                message=None,
                metrics=StepMetrics(
                    input_tokens=total_input or None,
                    output_tokens=total_output or None,
                    cache_read_tokens=total_cache_read or None,
                    cache_creation_tokens=total_cache_write or None,
                    cost_usd=total_cost if cost_seen else None,
                ),
                extra={"usage_summary": True},
            )

        builder.trajectory.extra.setdefault("pi_cli", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(transcript_file),
            "event_count": len(events),
        })

    @classmethod
    def _consume_message(
        cls, message: Any, builder: TrajectoryBuilder,
    ) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return None
        role = message.get("role") or "assistant"
        source = {"user": "user", "assistant": "agent"}.get(role, "system")

        text_parts: list[str] = []
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype in ("text", "text_delta") and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                elif btype in ("reasoning", "thinking") and isinstance(block.get("text"), str):
                    text_parts.append(f"[reasoning] {block['text']}")

        if text_parts:
            builder.add_step(source=source, message="\n".join(t for t in text_parts if t))

        usage = message.get("usage")
        if not isinstance(usage, dict):
            return None
        cost = usage.get("cost")
        cost_total = cost.get("total") if isinstance(cost, dict) else cost
        return {
            "input_tokens": _num(usage.get("input") or usage.get("inputTokens")),
            "output_tokens": _num(usage.get("output") or usage.get("outputTokens")),
            "cache_read_tokens": _num(usage.get("cacheRead") or usage.get("cacheReadTokens")),
            "cache_write_tokens": _num(usage.get("cacheWrite") or usage.get("cacheCreationTokens")),
            "cost_usd": _opt_float(cost_total if cost_total is not None else usage.get("costUsd")),
        }

    @staticmethod
    def _result_to_content(result: Any) -> list[ContentPart]:
        if result is None:
            return []
        if isinstance(result, str):
            return [ContentPart(type="text", text=result)]
        if isinstance(result, list):
            parts: list[ContentPart] = []
            texts: list[str] = []
            for item in result:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    texts.append(item["text"])
                elif isinstance(item, dict) and item.get("type") == "image":
                    img = _image_part(item)
                    if img:
                        parts.append(img)
            if texts:
                parts.insert(0, ContentPart(type="text", text="\n".join(texts)))
            return parts
        if isinstance(result, dict):
            if isinstance(result.get("text"), str):
                return [ContentPart(type="text", text=result["text"])]
            img = _image_part(result)
            if img:
                return [img]
            return [ContentPart(type="text", text=json.dumps(result, ensure_ascii=False)[:2000])]
        return [ContentPart(type="text", text=str(result)[:2000])]


def _image_part(block: dict[str, Any]) -> ContentPart | None:
    url = block.get("url") or block.get("data") or block.get("image")
    if not isinstance(url, str) or "base64," not in url:
        return None
    marker = "base64,"
    idx = url.find(marker)
    data = url[idx + len(marker):]
    media_type = "image/png"
    header = url[:idx]
    if header.startswith("data:") and ";" in header:
        media_type = header[5:].split(";", 1)[0] or media_type
    return ContentPart(
        type="image",
        image=ImageSource(type="base64", media_type=media_type, data=data),
    )


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"_raw": value}


def _num(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _opt_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _installed_version(pi_path: str) -> str | None:
    try:
        probe = subprocess.run(
            [pi_path, "-v"], capture_output=True, text=True, timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    out = (probe.stdout or "") + (probe.stderr or "")
    for token in out.split():
        token = token.strip().lstrip("v")
        if token and token[0].isdigit() and "." in token:
            return token
    return None


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


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
