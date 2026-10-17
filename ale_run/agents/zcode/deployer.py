"""Build ZCode from source and run its headless CLI with a private provider registry."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import time
import tempfile
import urllib.request
import zipfile
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

from ale_run.agents._cli_runtime import (
    redact,
    ensure_cli_node,
    run_command,
    secret_values,
    wait_process,
    write_private_json,
)

from .config import ZCodeConfig

logger = logging.getLogger(__name__)

_TERM_GRACE_S = 3.0

_REPO_URL = "https://github.com/zai-org/ZCode.git"
_PNPM_VERSION = "10.33.2"
_CLI_DIST_RELPATH = "apps/zcode-cli/packages/cli/dist/zcode.cjs"
_NODE_VERSION = "24.14.0"
_WINDOWS_PROMPT_LOADER = (
    "const fs = require('node:fs'); const cli = process.argv[1]; "
    "process.argv = [process.argv[0], cli, '-p', "
    "fs.readFileSync(process.env.ALE_ZCODE_PROMPT_FILE, 'utf8'), ...process.argv.slice(2)]; "
    "require(cli);"
)

_BUILTIN_PROVIDER_CONFIG_FILE_ENV = "ZCODE_BUILTIN_PROVIDER_CONFIG_FILE"
_PERSONAL_PROVIDER_CONFIG_FILE_ENV = "ZCODE_PERSONAL_PROVIDER_CONFIG_FILE"


class ZCodeDeployer(BaseAgentDeployer):
    """Stdlib-only deployer that builds ``zcode`` from source and drives it."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("stdout.log", "stderr.log")

    @property
    def version(self) -> str | None:
        cfg: ZCodeConfig = self.config  # type: ignore[assignment]
        return f"src:{getattr(self, '_resolved_ref', cfg.cli_ref)}"

    async def install(self) -> None:
        cfg: ZCodeConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        if not cfg.api_key:
            raise RuntimeError(
                "zcode: config.api_key is required (e.g. api_key: ${env:HF_INFERENCE_TOKEN} "
                "in the agent yaml) -- ZCode's custom-provider auth has no env-var "
                "fallback to read a key from."
            )

        from ale_run.agents._bootstrap import ensure_node_npm

        node, npm = await ensure_node_npm()
        if sandbox.is_linux:
            node = await ensure_cli_node(node, npm, (24, 14, 0))
        else:
            node = await self._ensure_windows_node(node)
            npm = str(Path(node).parent / "node_modules/npm/bin/npm-cli.js")
            if not Path(npm).is_file():
                raise RuntimeError("zcode: Windows Node runtime must include npm-cli.js")
            if not shutil.which("git"):
                for directory in (
                    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/cmd",
                    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
                    / "Programs/Git/cmd",
                ):
                    if (directory / "git.exe").is_file():
                        os.environ["PATH"] = (
                            str(directory) + os.pathsep + os.environ.get("PATH", "")
                        )
                        break
                if not shutil.which("git"):
                    raise RuntimeError(
                        "zcode: Git for Windows is required to build the pinned source"
                    )
        self._node_path = node

        pnpm = await self._ensure_pnpm(npm)

        repo_dir = Path(os.path.expanduser("~")) / ".zcode_build" / "ZCode"
        t0 = time.monotonic()
        await self._clone_or_update(repo_dir, cfg.cli_ref)
        clone_s = time.monotonic() - t0

        t0 = time.monotonic()
        await self._pnpm_install(pnpm, repo_dir)
        install_s = time.monotonic() - t0

        t0 = time.monotonic()
        await self._pnpm_build(pnpm, repo_dir)
        build_s = time.monotonic() - t0

        cli_path = repo_dir / _CLI_DIST_RELPATH
        if not cli_path.is_file():
            raise RuntimeError(f"zcode: build finished but {cli_path} does not exist")
        self._cli_path = cli_path
        logger.info(
            "zcode: built %s (clone=%.0fs install=%.0fs build=%.0fs)",
            cli_path,
            clone_s,
            install_s,
            build_s,
        )

        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)
        cfg_dir = Path(tempfile.mkdtemp(prefix=".ale-zcode-", dir=wd.parent))
        self._home_dir = str(cfg_dir)
        builtin_path = cfg_dir / "builtin_provider_config.json"
        personal_path = cfg_dir / "personal_provider_config.json"
        upstream_builtin = json.loads(
            (cli_path.parent / "provider/zcode-builtin.json").read_text(encoding="utf-8")
        )
        for path, value in (
            (builtin_path, self._build_builtin_config(cfg, upstream_builtin)),
            (personal_path, self._build_personal_config(cfg)),
        ):
            if sandbox.is_linux:
                write_private_json(path, value)
            else:
                with path.open("x", encoding="utf-8") as handle:
                    json.dump(value, handle, indent=2)
        self._provider_env = {
            _BUILTIN_PROVIDER_CONFIG_FILE_ENV: str(builtin_path),
            _PERSONAL_PROVIDER_CONFIG_FILE_ENV: str(personal_path),
        }
        logger.info(
            "zcode: provider config staged at %s (provider_id=%s model=%s base_url=%s)",
            cfg_dir,
            cfg.provider_id,
            cfg.model,
            cfg.base_url,
        )

    async def _ensure_pnpm(self, npm: str) -> str:
        if not self.executor.sandbox.is_linux:
            prefix = Path.home() / ".zcode_build/tools"
            pnpm = prefix / "node_modules/pnpm/bin/pnpm.cjs"
            if pnpm.is_file():
                probe = await self._run_command([str(pnpm), "--version"], timeout=30)
                if probe.returncode == 0 and probe.stdout.strip() == _PNPM_VERSION:
                    return str(pnpm)
            result = await self._run_command(
                [
                    npm,
                    "install",
                    "--prefix",
                    str(prefix),
                    "--no-save",
                    "--no-audit",
                    "--no-fund",
                    "--ignore-scripts",
                    f"pnpm@{_PNPM_VERSION}",
                ],
                timeout=180,
            )
            if result.returncode or not pnpm.is_file():
                raise RuntimeError("zcode: could not install private Windows pnpm")
            probe = await self._run_command([str(pnpm), "--version"], timeout=30)
            if probe.returncode or probe.stdout.strip() != _PNPM_VERSION:
                raise RuntimeError("zcode: private Windows pnpm version mismatch")
            return str(pnpm)

        pnpm = shutil.which("pnpm")
        if pnpm:
            probe = await self._run_command([pnpm, "--version"], timeout=30)
            if probe.returncode == 0 and probe.stdout.strip() == _PNPM_VERSION:
                return pnpm

        corepack = shutil.which("corepack")
        if corepack:
            proc = await self._run_command(
                [corepack, "prepare", f"pnpm@{_PNPM_VERSION}", "--activate"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode == 0:
                pnpm = shutil.which("pnpm")
                if pnpm:
                    probe = await self._run_command([pnpm, "--version"], timeout=30)
                    if probe.returncode == 0 and probe.stdout.strip() == _PNPM_VERSION:
                        return pnpm
            logger.warning(
                "zcode: corepack prepare pnpm failed (rc=%s), falling back to npm -g",
                proc.returncode,
            )

        proc = await self._run_command(
            [npm, "install", "-g", f"pnpm@{_PNPM_VERSION}"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"zcode: could not install pnpm (corepack and npm -g both failed): "
                f"{(proc.stderr or '')[-500:]}"
            )
        pnpm = shutil.which("pnpm")
        if not pnpm:
            raise RuntimeError("zcode: pnpm installed via npm -g but still not on PATH")
        probe = await self._run_command([pnpm, "--version"], timeout=30)
        if probe.returncode != 0 or probe.stdout.strip() != _PNPM_VERSION:
            raise RuntimeError("zcode: pnpm on PATH does not match the required version")
        return pnpm

    async def _clone_or_update(self, repo_dir: Path, ref: str) -> None:
        repo_dir.mkdir(parents=True, exist_ok=True)
        if not (repo_dir / ".git").is_dir():
            for command in (
                ["git", "init", str(repo_dir)],
                ["git", "-C", str(repo_dir), "remote", "add", "origin", _REPO_URL],
            ):
                result = await self._run_command(command, timeout=60)
                if result.returncode:
                    raise RuntimeError("zcode: could not initialize build source")
        if not self.executor.sandbox.is_linux:
            result = await self._run_command(
                ["git", "-C", str(repo_dir), "config", "core.longpaths", "true"], timeout=30
            )
            if result.returncode:
                raise RuntimeError("zcode: could not enable Windows Git long paths")
        for command in (
            ["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", ref],
            ["git", "-C", str(repo_dir), "checkout", "--detach", "FETCH_HEAD"],
        ):
            result = await self._run_command(command, timeout=300)
            if result.returncode:
                raise RuntimeError("zcode: could not resolve requested cli_ref")
        result = await self._run_command(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], timeout=30
        )
        if result.returncode:
            raise RuntimeError("zcode: could not record source revision")
        self._resolved_ref = result.stdout.strip()

    async def _pnpm_install(self, pnpm: str, repo_dir: Path) -> None:
        env = {**os.environ, "ELECTRON_SKIP_BINARY_DOWNLOAD": "1"}
        proc = await self._run_command(
            [pnpm, "install", "--filter", "@zcode/cli..."],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"zcode: pnpm install failed (rc={proc.returncode}): "
                f"stdout: {(proc.stdout or '')[-1000:]} "
                f"stderr: {(proc.stderr or '')[-1000:]}"
            )

    async def _pnpm_build(self, pnpm: str, repo_dir: Path) -> None:
        env = {**os.environ, "ELECTRON_SKIP_BINARY_DOWNLOAD": "1"}
        proc = await self._run_command(
            [pnpm, "--filter", "@zcode/cli...", "build"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"zcode: pnpm build failed (rc={proc.returncode}): "
                f"stdout: {(proc.stdout or '')[-1000:]} "
                f"stderr: {(proc.stderr or '')[-1000:]}"
            )

    async def _ensure_windows_node(self, node: str) -> str:
        probe = await self._run_command([node, "--version"], timeout=30)
        if probe.returncode == 0 and probe.stdout.strip() == f"v{_NODE_VERSION}":
            runtime = Path(node)
        else:
            architecture = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64"}.get(
                platform.machine().lower()
            )
            if architecture is None:
                raise RuntimeError("zcode: unsupported Windows Node architecture")
            directory = f"node-v{_NODE_VERSION}-win-{architecture}"
            cache = Path.home() / ".zcode_build/node"
            runtime = cache / directory / "node.exe"
            if not runtime.is_file():
                await asyncio.to_thread(_download_windows_node, cache, directory)
            probe = await self._run_command([str(runtime), "--version"], timeout=30)
            if probe.returncode or probe.stdout.strip() != f"v{_NODE_VERSION}":
                raise RuntimeError("zcode: private Windows Node version mismatch")
        os.environ["PATH"] = str(runtime.parent) + os.pathsep + os.environ.get("PATH", "")
        return str(runtime)

    async def _run_command(self, argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        if self.executor.sandbox.is_linux:
            return await run_command(argv, **kwargs)
        if Path(argv[0]).suffix.lower() in {".js", ".cjs"}:
            argv = [self._node_path, *argv]
        timeout = kwargs.get("timeout", 300)
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=kwargs.get("cwd"),
                env=kwargs.get("env"),
            )
            await self._wait_process(proc, timeout=timeout)
            stdout_file.seek(0)
            stderr_file.seek(0)
            return subprocess.CompletedProcess(
                argv,
                proc.returncode,
                stdout_file.read().decode("utf-8", errors="replace"),
                stderr_file.read().decode("utf-8", errors="replace"),
            )

    async def _wait_process(self, proc: subprocess.Popen, *, timeout: float | None = None) -> int:
        """Reap the CLI; Windows descendants remain in the sandbox supervisor's Job Object."""
        if self.executor.sandbox.is_linux:
            return await wait_process(proc, timeout=timeout, grace_s=_TERM_GRACE_S)
        try:
            async with asyncio.timeout(timeout):
                while proc.poll() is None:
                    await asyncio.sleep(0.05)
                return proc.returncode
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=_TERM_GRACE_S)

    @staticmethod
    def _build_builtin_config(cfg: ZCodeConfig, upstream_builtin: dict) -> dict:
        """Preserve CLI-owned defaults and scope explicit overrides to this model."""
        builtin = copy.deepcopy(upstream_builtin)
        options = {
            "reasoningLevel": {
                "values": ["disabled", "enabled"],
                "map": json.dumps(cfg.reasoning_request),
            },
        }
        if cfg.max_output_tokens is not None:
            options["maxOutputTokens"] = {
                "max": cfg.max_output_tokens,
                "map": "{'max_completion_tokens': maxOutputTokens}",
            }
        builtin["config"]["modelConfigRules"]["builtinProviderModelRules"].append(
            {
                "providerId": cfg.provider_id,
                "modelId": cfg.model,
                "config": {
                    "properties": {"contextWindow": cfg.context_window},
                    "optionSpecs": options,
                },
            }
        )
        return builtin

    @staticmethod
    def _build_personal_config(cfg: ZCodeConfig) -> dict:
        """Register a custom provider and its default model selection."""
        return {
            "schemaVersion": 1,
            "config": {
                "providerOrder": [cfg.provider_id],
                "providerConfigRules": {
                    "providerRules": [
                        {
                            "providerId": cfg.provider_id,
                            "providerName": cfg.provider_id,
                            "config": {
                                "group": "standard-personal",
                                "access": {"type": "api-key", "apiKey": cfg.api_key},
                                "api": {
                                    "type": "openai-chat-completions",
                                    "baseUrl": cfg.base_url,
                                },
                                "personalModelIds": [cfg.model],
                            },
                        },
                    ],
                },
                "modelConfigRules": {"providerModelRules": [], "manualProviderModelRules": []},
                "defaultModelSelection": {
                    "providerId": cfg.provider_id,
                    "modelId": cfg.model,
                },
            },
        }

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: ZCodeConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        stdout_log = wd / "stdout.log"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "zcode.pid"
        for f in (stdout_log, stderr_log, pid_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        argv = [
            self._node_path,
            str(self._cli_path),
            "-p",
            prompt,
            "--mode",
            "yolo",
            "--no-browser",
            "--output-format",
            "stream-json",
            "--cwd",
            str(wd),
        ]
        env = self._build_env(cfg)
        if not self.executor.sandbox.is_linux:
            prompt_path = wd / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            env["ALE_ZCODE_PROMPT_FILE"] = str(prompt_path)
            argv = [self._node_path, "-e", _WINDOWS_PROMPT_LOADER, str(self._cli_path), *argv[4:]]

        t0 = time.monotonic()
        with open(stdout_log, "wb") as fout, open(stderr_log, "wb") as ferr:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=fout,
                stderr=ferr,
                env=env,
                cwd=str(wd),
                start_new_session=self.executor.sandbox.is_linux,
            )
        pid_file.write_text(str(proc.pid), encoding="ascii")
        logger.info(
            "zcode: spawned pid=%s (model=%s base_url=%s)", proc.pid, cfg.model, cfg.base_url
        )

        await self._wait_process(proc)

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

    def _build_env(self, cfg: ZCodeConfig) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env.update(self._provider_env)
        env["NO_COLOR"] = "1"
        env["HOME"] = getattr(self, "_home_dir", os.path.expanduser("~"))
        if not self.executor.sandbox.is_linux:
            env["USERPROFILE"] = env["HOME"]
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
        config: ZCodeConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        stdout_log = work_dir / "stdout.log"
        if not stdout_log.exists():
            builder.add_step(
                source="system",
                message=f"zcode: no stdout at {stdout_log}",
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
                message="zcode: no JSON events parsed from stdout",
                extra={"reason": "no_events"},
            )
            builder.trajectory.extra.setdefault("zcode", {}).update(
                {
                    "exit_code": run_result.exit_code,
                    "transcript_path": str(stdout_log),
                }
            )
            return

        text_buffers: dict[str, dict[str, list[str]]] = {}
        for ev in events:
            cls._consume_event(ev, builder, text_buffers)

        for buffer in text_buffers.values():
            if buffer["text"] or buffer["reasoning"]:
                builder.add_step(
                    source="agent",
                    message="".join(buffer["text"]) or None,
                    reasoning="".join(buffer["reasoning"]) or None,
                    extra={"partial": True},
                )

        if result_summary:
            usage = (
                result_summary.get("usage")
                if isinstance(result_summary.get("usage"), dict)
                else None
            )
            metrics = None
            if usage:
                metrics = StepMetrics(
                    input_tokens=usage.get(
                        "inputTokens", usage.get("input_tokens", usage.get("prompt_tokens"))
                    ),
                    output_tokens=usage.get(
                        "outputTokens", usage.get("output_tokens", usage.get("completion_tokens"))
                    ),
                    cache_read_tokens=usage.get("cacheReadTokens", usage.get("cache_read_tokens")),
                    cache_creation_tokens=usage.get(
                        "cacheWriteTokens", usage.get("cache_creation_tokens")
                    ),
                    cost_usd=usage.get("cost_usd", usage.get("cost")),
                )
            if metrics:
                builder.override_final_metrics(
                    total_input_tokens=metrics.input_tokens,
                    total_output_tokens=metrics.output_tokens,
                    total_cache_read_tokens=metrics.cache_read_tokens,
                    total_cache_creation_tokens=metrics.cache_creation_tokens,
                    total_cost_usd=metrics.cost_usd,
                )
            response_text = result_summary.get("response")
            if (
                isinstance(response_text, str)
                and response_text.strip()
                and not any(
                    step.source == "agent" and step.message == response_text
                    for step in builder.trajectory.steps
                )
            ):
                builder.add_step(source="agent", message=response_text)

        builder.trajectory.extra.setdefault("zcode", {}).update(
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
    def _consume_event(
        cls,
        event: dict,
        builder: TrajectoryBuilder,
        text_buffers: dict[str, dict[str, list[str]]],
    ) -> None:
        """Append tool results, streamed text, and per-request usage to the trajectory."""
        top_type = str(event.get("type") or "").strip()
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}

        if top_type == "tool.updated":
            pkind = payload.get("kind")
            tool_call_id = str(payload.get("toolCallId") or "")
            if pkind == "scheduled":
                name = payload.get("toolName") or "unknown"
                args = payload.get("input")
                builder.add_step(
                    source="agent",
                    tool_calls=[
                        ToolCall(
                            id=tool_call_id or f"zc_{name}",
                            name=str(name),
                            arguments=args if isinstance(args, dict) else {"_raw": str(args)},
                        )
                    ],
                )
            elif pkind == "result":
                result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
                content_text = result.get("content")
                parts = (
                    [ContentPart(type="text", text=content_text)]
                    if isinstance(content_text, str) and content_text
                    else []
                )
                builder.add_step(
                    source="environment",
                    observation=Observation(
                        results=[
                            ToolResult(
                                tool_call_id=tool_call_id,
                                content=parts,
                                is_error=result.get("success") is False,
                            ),
                        ]
                    ),
                )
            # started/batch/progress: pure scheduling telemetry, dropped.
            return

        if top_type == "model.streaming":
            skind = payload.get("kind")
            msg_id = str(payload.get("assistantMessageId") or "")
            buf = text_buffers.setdefault(msg_id, {"text": [], "reasoning": []})
            if skind == "text_delta":
                delta = payload.get("delta")
                if isinstance(delta, str):
                    buf["text"].append(delta)
            elif skind == "reasoning_delta":
                delta = payload.get("delta")
                if isinstance(delta, str):
                    buf["reasoning"].append(delta)
            elif skind == "finish":
                text = "".join(buf["text"]).strip()
                reasoning = "".join(buf["reasoning"]).strip()
                text_buffers.pop(msg_id, None)
                if text or reasoning:
                    builder.add_step(
                        source="agent",
                        message=text or None,
                        reasoning=reasoning or None,
                    )
            return

        if top_type == "session.updated":
            nested_type = payload.get("type")
            if nested_type == "model_request_completed":
                usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
                if usage:
                    builder.add_step(
                        source="system",
                        message=None,
                        metrics=StepMetrics(
                            input_tokens=usage.get("inputTokens"),
                            output_tokens=usage.get("outputTokens"),
                            cache_read_tokens=usage.get("cacheReadTokens") or None,
                            cache_creation_tokens=usage.get("cacheWriteTokens") or None,
                        ),
                        extra={"provider_request_id": payload.get("providerRequestId")},
                    )
            # model_request_started and the None-typed session.updated
            # variants (title/context bookkeeping) carry nothing new.
            return

        if top_type in (
            "turn.started",
            "turn.completed",
            "session.titleUpdated",
            "checkpoint.created",
            "streamRecovery.updated",
        ):
            # turn.completed duplicates the final "result" line (handled
            # separately in parse_artifacts); the rest is pure bookkeeping.
            return

        builder.add_step(
            source="system",
            message="",
            extra={"kind": top_type or "unknown", "raw": event},
        )


def _download_windows_node(cache: Path, directory: str) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    filename = f"{directory}.zip"
    base_url = f"https://nodejs.org/dist/v{_NODE_VERSION}/"
    with urllib.request.urlopen(base_url + "SHASUMS256.txt", timeout=60) as response:
        checksums = dict(
            line.split()[::-1] for line in response.read().decode("ascii").splitlines()
        )
    expected = checksums.get(filename)
    if not expected:
        raise RuntimeError("zcode: Windows Node archive has no official checksum")
    with tempfile.TemporaryDirectory(prefix=".download-", dir=cache) as temporary:
        staging = Path(temporary)
        archive = staging / filename
        with urllib.request.urlopen(base_url + filename, timeout=120) as response:
            with archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        with archive.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError("zcode: Windows Node archive checksum mismatch")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
        extracted = staging / directory
        if (
            not (extracted / "node.exe").is_file()
            or not (extracted / "node_modules/npm/bin/npm-cli.js").is_file()
        ):
            raise RuntimeError("zcode: Windows Node archive is missing Node or npm")
        extracted.rename(cache / directory)


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
