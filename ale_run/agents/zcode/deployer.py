"""ZCodeDeployer -- builds and drives Z.ai's ``zcode`` CLI from source.

No published package exists (``zcode-cli``/``@zcode/cli`` are absent from
the npm registry), so every ``install()`` clones
https://github.com/zai-org/ZCode and builds the CLI bundle:

    git clone --depth 1 --branch <cli_ref> https://github.com/zai-org/ZCode
    corepack prepare pnpm@10.33.2 --activate   # (or: npm install -g pnpm)
    pnpm install --filter @zcode/cli...        # from the repo ROOT
    pnpm --filter @zcode/cli build             # -> apps/zcode-cli/packages/cli/dist/zcode.cjs

The repo root matters: ``apps/zcode-cli`` has its own nested
``pnpm-workspace.yaml`` (packages/*, tools/*), but its packages import
``@zcode/provider`` / ``@zcode/provider-node`` / ``@zcode/shared`` /
``@zcode/services`` etc. from the *root*-level ``packages/*`` -- the root
``pnpm-workspace.yaml`` is the one that stitches both trees together
(``packages/*`` + ``apps/zcode-cli`` + ``apps/zcode-cli/packages/*`` +
``apps/zcode-cli/tools/*``). Installing/building from inside
``apps/zcode-cli`` alone cannot resolve those workspace deps.

Headless invocation (confirmed by reading
``apps/zcode-cli/packages/cli/src/{arguments.ts,run.ts,prompt-command.ts}``)::

    node dist/zcode.cjs -p "<prompt>" --mode yolo \
        --output-format stream-json --cwd <task_workdir>

* ``-p``/``--prompt`` is the one-shot headless entry point.
* ``--mode yolo`` auto-approves every tool call (no interactive permission
  prompts). It is also ``run.ts``'s own ``DEFAULT_HEADLESS_PROMPT_MODE`` for
  ``-p``, so this flag is technically redundant -- passed explicitly anyway.
* ``--output-format stream-json`` prints one JSON object per line as the
  run progresses, ending with a final ``{"type":"result", sessionId,
  traceId, response, usage, eventCount, projection:{...}}`` summary line.
* There is no ``--model`` flag -- model selection is entirely file-based,
  see below.

Auth / model routing is entirely file-based -- no interactive ``zcode
login`` is ever run. Two JSON files, set via env vars
``ZCODE_BUILTIN_PROVIDER_CONFIG_FILE`` / ``ZCODE_PERSONAL_PROVIDER_CONFIG_FILE``
(both-or-neither: ``prepareCliProviderRuntimeEnv`` in
``provider-runtime-env.ts`` only takes the fast, no-CDN-refresh path when
both are set explicitly):

* builtin -- a *minimal empty* registry (we don't need Z.ai's real one
  since our provider is injected via the personal layer). Shape validated
  against ``zcode-builtin-release.ts``'s ``releaseSchema``: top-level
  ``{schemaVersion: 1, revision: <int>, config: {providerConfigRules,
  modelConfigRules}}`` -- note ``revision`` here is a *number*, unlike the
  personal file below.
* personal -- one custom ``openai-chat-completions`` provider pointed at
  ``base_url`` with a literal API key, plus ``defaultModelSelection``.
  Shape validated against ``provider-config-file-codec.ts``'s
  ``storedProviderConfigSchema`` (``.strict()``, only ``schemaVersion`` +
  ``config`` -- NO ``revision`` key here, a different codec from the
  builtin file above) and ``rule-data-schema.ts``'s
  ``personalProviderConfigRuleSchema`` (``config.access`` is
  ``apiKeyAccessDataSchema``, ``config.api`` is
  ``personalProviderApiDataSchema``, ``config.group`` must be
  ``"standard-personal"`` or omitted for a non-``account:``-prefixed
  provider id).

Trajectory recovery: the NDJSON stream-json stdout is the source of truth.
The exact per-line *intermediate* event shape (as opposed to the final
``result`` line, which is confirmed) was not known ahead of a real run, so
``parse_artifacts`` degrades gracefully -- known shapes get typed steps,
everything else becomes a generic ``system`` step carrying the raw event.
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

from .config import ZCodeConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 5.0
_TERM_GRACE_S = 3.0

_REPO_URL = "https://github.com/zai-org/ZCode.git"
_PNPM_VERSION = "10.33.2"
_CLI_DIST_RELPATH = "apps/zcode-cli/packages/cli/dist/zcode.cjs"

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
        return f"src:{cfg.cli_ref}"

    # =========================================================================
    # install
    # =========================================================================

    async def install(self) -> None:
        cfg: ZCodeConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        if not sandbox.is_linux:
            raise NotImplementedError("zcode is Linux-only")

        if not cfg.api_key:
            raise RuntimeError(
                "zcode: config.api_key is required (e.g. api_key: ${env:HF_TOKEN} "
                "in the agent yaml) -- ZCode's custom-provider auth has no env-var "
                "fallback to read a key from."
            )

        from ale_run.agents._bootstrap import ensure_node_npm
        node, npm = await ensure_node_npm()
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
            raise RuntimeError(
                f"zcode: build finished but {cli_path} does not exist"
            )
        self._cli_path = cli_path
        logger.info(
            "zcode: built %s (clone=%.0fs install=%.0fs build=%.0fs)",
            cli_path, clone_s, install_s, build_s,
        )

        # Provider config: two JSON files, env-var-addressed (see module
        # docstring). Written under a stable per-run dir.
        cfg_dir = Path(os.path.expanduser("~")) / ".zcode_ale"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        builtin_path = cfg_dir / "builtin_provider_config.json"
        personal_path = cfg_dir / "personal_provider_config.json"
        builtin_path.write_text(
            json.dumps(self._build_builtin_config(), indent=2), encoding="utf-8",
        )
        personal_path.write_text(
            json.dumps(self._build_personal_config(cfg), indent=2), encoding="utf-8",
        )
        self._provider_env = {
            _BUILTIN_PROVIDER_CONFIG_FILE_ENV: str(builtin_path),
            _PERSONAL_PROVIDER_CONFIG_FILE_ENV: str(personal_path),
        }
        logger.info(
            "zcode: provider config staged at %s (provider_id=%s model=%s base_url=%s)",
            cfg_dir, cfg.provider_id, cfg.model, cfg.base_url,
        )

    async def _ensure_pnpm(self, npm: str) -> str:
        pnpm = shutil.which("pnpm")
        if pnpm:
            return pnpm

        corepack = shutil.which("corepack")
        if corepack:
            proc = await asyncio.to_thread(
                subprocess.run,
                [corepack, "prepare", f"pnpm@{_PNPM_VERSION}", "--activate"],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode == 0:
                pnpm = shutil.which("pnpm")
                if pnpm:
                    return pnpm
            logger.warning(
                "zcode: corepack prepare pnpm failed (rc=%s), falling back to npm -g",
                proc.returncode,
            )

        proc = await asyncio.to_thread(
            subprocess.run,
            [npm, "install", "-g", f"pnpm@{_PNPM_VERSION}"],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"zcode: could not install pnpm (corepack and npm -g both failed): "
                f"{(proc.stderr or '')[-500:]}"
            )
        pnpm = shutil.which("pnpm")
        if not pnpm:
            raise RuntimeError("zcode: pnpm installed via npm -g but still not on PATH")
        return pnpm

    async def _clone_or_update(self, repo_dir: Path, ref: str) -> None:
        if (repo_dir / ".git").is_dir():
            logger.info("zcode: repo already present at %s, reusing", repo_dir)
            return
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.to_thread(
            subprocess.run,
            ["git", "clone", "--depth", "1", "--branch", ref, _REPO_URL, str(repo_dir)],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"zcode: git clone failed (rc={proc.returncode}): {(proc.stderr or '')[-800:]}"
            )

    async def _pnpm_install(self, pnpm: str, repo_dir: Path) -> None:
        # @zcode/cli's headless-browser feature (headless-browser.ts,
        # sea-playwright-runtime.ts) pulls in ``electron`` as a transitive
        # dependency. Its postinstall downloads a large prebuilt binary from
        # an external CDN, which reliably ETIMEDOUTs from sandbox network
        # conditions and fails the *entire* install (even though we never
        # invoke the browser feature -- headless CLI runs pass --no-browser).
        # ELECTRON_SKIP_BINARY_DOWNLOAD=1 makes electron's install script
        # skip the download and exit 0; a handful of optional native
        # bindings (e.g. ssh2's crypto binding) also fail to compile under
        # newer V8/Node headers, but pnpm treats those as non-fatal
        # ("Failed to build optional ... binding") and still exits 0 --
        # confirmed via a real local ``pnpm install`` run.
        env = {**os.environ, "ELECTRON_SKIP_BINARY_DOWNLOAD": "1"}
        proc = await asyncio.to_thread(
            subprocess.run,
            [pnpm, "install", "--filter", "@zcode/cli..."],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=1800,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"zcode: pnpm install failed (rc={proc.returncode}): "
                f"stdout: {(proc.stdout or '')[-1000:]} "
                f"stderr: {(proc.stderr or '')[-1000:]}"
            )

    async def _pnpm_build(self, pnpm: str, repo_dir: Path) -> None:
        # NOTE the trailing "..." -- ``--filter @zcode/cli`` (bare) only runs
        # @zcode/cli's own build script and fails immediately with "Could
        # not resolve @zcode/i18n" etc: turbo/pnpm don't build a package's
        # workspace deps unless the filter says to. "..." tells pnpm to
        # build @zcode/cli AND its whole workspace dependency graph, in
        # topological order -- confirmed via a real local build (17 of 33
        # workspace projects built, ending in dist/zcode.cjs).
        env = {**os.environ, "ELECTRON_SKIP_BINARY_DOWNLOAD": "1"}
        proc = await asyncio.to_thread(
            subprocess.run,
            [pnpm, "--filter", "@zcode/cli...", "build"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=900,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"zcode: pnpm build failed (rc={proc.returncode}): "
                f"stdout: {(proc.stdout or '')[-1000:]} "
                f"stderr: {(proc.stderr or '')[-1000:]}"
            )

    @staticmethod
    def _build_builtin_config() -> dict:
        """Empty provider registry + two generic catch-all model-config
        rules copied verbatim from Z.ai's real ``config/provider/
        zcode-builtin.json`` (not Z.ai-brand-specific -- they key off
        ``modelMatch: ".*"`` / ``apiTypeMatch: "openai-chat-completions"``,
        i.e. "any model, any OpenAI-compatible provider").

        Without these, a personal ``personalModelIds`` entry resolves to
        an *empty* model config (no rule anywhere supplies contextWindow /
        supportsToolCall / the reasoningLevel+maxOutputTokens option maps),
        which fails the registry's completeness validation silently --
        the model gets marked ``enabled: false`` / non-executable, and the
        CLI's only symptom is ``CONFIGURATION_ERROR: Select a model before
        continuing`` on every turn. Confirmed via a real local ``-p`` run:
        adding just these two rules (no per-model manual rule needed) was
        enough for the HF-router Qwen model to resolve and execute.
        """
        return {
            "schemaVersion": 1,
            "revision": 1,
            "config": {
                "providerConfigRules": {"templateRules": [], "providerRules": []},
                "modelConfigRules": {
                    "modelRules": [
                        {
                            "modelMatch": ".*",
                            "config": {
                                "enabled": True,
                                "properties": {
                                    "contextWindow": 200000,
                                    "inputFormat": {
                                        "supportsText": True,
                                        "supportsImage": False,
                                        "supportsVideo": False,
                                        "supportsAudio": False,
                                        "supportsPdf": False,
                                    },
                                    "outputFormat": {"supportsText": True},
                                    "supportsToolCall": True,
                                    "supportsJsonSchemaOutput": False,
                                    "supportsNativeWebSearch": False,
                                    "supportsMidConversationSystem": False,
                                    "requiresMfjsToolSchema": False,
                                },
                                "optionSpecs": {
                                    # Raised from Z.ai's own original 32000 --
                                    # confirmed too small for other reasoning-
                                    # heavy models tested on this endpoint
                                    # (pi_cli hit a 16384-token cap mid-
                                    # calculation with GLM-5.3's verbose
                                    # chain-of-thought; being generous here
                                    # avoids the same class of truncation).
                                    "maxOutputTokens": {"max": 65536},
                                    "reasoningLevel": {
                                        "values": ["disabled", "enabled"],
                                        "map": "{}",
                                    },
                                },
                            },
                        },
                    ],
                    "modelApiRules": [
                        {
                            "modelMatch": ".*",
                            "apiTypeMatch": "openai-chat-completions",
                            "config": {
                                "optionSpecs": {
                                    "reasoningLevel": {
                                        "map": (
                                            '{\n  "thinking": {\n    "type": '
                                            'reasoningLevel == "disabled" || '
                                            'reasoningLevel == "none" ? "disabled" '
                                            ': "enabled"\n  },\n  '
                                            '"enable_thinking": reasoningLevel != '
                                            '"disabled" && reasoningLevel != "none",'
                                            '\n  "reasoning_effort": reasoningLevel '
                                            '== "disabled" ? "none" : reasoningLevel '
                                            '== "enabled" ? "high" : reasoningLevel,'
                                            '\n  "reasoning": {\n    "effort": '
                                            'reasoningLevel == "disabled" ? "none" : '
                                            'reasoningLevel == "enabled" ? "high" : '
                                            "reasoningLevel\n  }\n}"
                                        ),
                                    },
                                    "maxOutputTokens": {
                                        "map": "{'max_completion_tokens': maxOutputTokens}",
                                    },
                                },
                            },
                        },
                    ],
                    "providerSiteRules": [],
                    "templateModelRules": [],
                    "builtinProviderModelRules": [],
                },
            },
        }

    @staticmethod
    def _build_personal_config(cfg: ZCodeConfig) -> dict:
        """One custom provider -- see module docstring for schema source."""
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
                                # "group" is optional at *parse* time (personal
                                # rule schema) but required at *registry*
                                # validation time -- omitting it produces a
                                # silent "required-field-missing" provider
                                # issue that filters the whole provider out of
                                # the registry, with no visible error beyond
                                # the CLI's generic "Select a model before
                                # continuing" (confirmed via a standalone
                                # resolver probe against the built package).
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

    # =========================================================================
    # launch
    # =========================================================================

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
            self._node_path, str(self._cli_path),
            "-p", prompt,
            "--mode", "yolo",
            "--output-format", "stream-json",
            "--cwd", str(wd),
        ]
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
        logger.info("zcode: spawned pid=%s (model=%s base_url=%s)",
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

    def _build_env(self, cfg: ZCodeConfig) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env.update(self._provider_env)
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

        raw = stdout_log.read_text(encoding="utf-8", errors="replace")
        events, result_summary = cls._parse_stream_json(raw)

        transcript_jsonl = work_dir / "transcript.jsonl"
        transcript_jsonl.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in events)
            + ("\n" if events else ""),
            encoding="utf-8",
        )

        if not events:
            builder.add_step(
                source="system",
                message="zcode: no JSON events parsed from stdout",
                extra={"reason": "no_events"},
            )
            builder.trajectory.extra.setdefault("zcode", {}).update({
                "exit_code": run_result.exit_code,
                "transcript_path": str(stdout_log),
            })
            return

        text_buffers: dict[str, dict[str, list[str]]] = {}
        for ev in events:
            cls._consume_event(ev, builder, text_buffers)

        if result_summary:
            usage = result_summary.get("usage") if isinstance(result_summary.get("usage"), dict) else None
            metrics = None
            if usage:
                metrics = StepMetrics(
                    prompt_tokens=usage.get("input_tokens") or usage.get("prompt_tokens"),
                    completion_tokens=usage.get("output_tokens") or usage.get("completion_tokens"),
                    cached_tokens=usage.get("cache_read_tokens"),
                    cache_creation_tokens=usage.get("cache_creation_tokens"),
                    cost_usd=usage.get("cost_usd") or usage.get("cost"),
                )
            response_text = result_summary.get("response")
            if isinstance(response_text, str) and response_text.strip():
                builder.add_step(source="agent", message=response_text, metrics=metrics)
            elif metrics:
                builder.add_step(source="system", message=None, metrics=metrics)

        builder.trajectory.extra.setdefault("zcode", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(stdout_log),
            "event_count": len(events),
            "result_summary": result_summary,
        })

    @staticmethod
    def _parse_stream_json(stdout_text: str) -> tuple[list[dict], dict | None]:
        """Split ``--output-format stream-json`` NDJSON stdout.

        Each non-blank line is expected to be one JSON object; the final
        ``{"type": "result", ...}`` line is the run summary (see
        ``prompt-command.ts``'s ``streamsEvents`` branch) and is pulled out
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
    def _consume_event(
        cls,
        event: dict,
        builder: TrajectoryBuilder,
        text_buffers: dict[str, dict[str, list[str]]],
    ) -> None:
        """Route one ``--output-format stream-json`` event to a trajectory step.

        Shapes below are confirmed from a real sandbox run's raw NDJSON (not
        guessed): ``tool.updated`` (payload.kind: scheduled/started/result/
        batch/progress), ``model.streaming`` (payload.kind: start/
        reasoning_start/reasoning_delta/reasoning_end/text_start/text_delta/
        text_end/tool_input_start/tool_input_delta/tool_input_end/tool_call/
        finish), and ``session.updated`` wrapping a nested
        ``payload.type`` of ``model_request_started``/``model_request_completed``
        (the latter carries per-request ``usage``). Purely structural/noise
        events (turn.started, session.titleUpdated, checkpoint.created,
        streamRecovery.updated, tool.updated kind=started/batch/progress,
        most model.streaming kinds) are dropped rather than emitted as
        1-per-line system steps -- on a real run these accounted for ~4200 of
        ~4270 raw events with no trajectory-relevant content.
        """
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
                    tool_calls=[ToolCall(
                        tool_call_id=tool_call_id or f"zc_{name}",
                        function_name=str(name),
                        arguments=args if isinstance(args, dict) else {"_raw": str(args)},
                    )],
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
                    observation=Observation(results=[
                        ToolResult(
                            source_call_id=tool_call_id,
                            content=parts,
                            is_error=result.get("success") is False,
                        ),
                    ]),
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
                        reasoning_content=reasoning or None,
                    )
            # start/*_start/*_end/tool_input_*/tool_call: superseded by the
            # tool.updated events above (which carry the resolved arguments,
            # not the raw streamed JSON fragments) or pure framing, dropped.
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
                            prompt_tokens=usage.get("inputTokens"),
                            completion_tokens=usage.get("outputTokens"),
                            cached_tokens=usage.get("cacheReadTokens") or None,
                            cache_creation_tokens=usage.get("cacheWriteTokens") or None,
                        ),
                        extra={"provider_request_id": payload.get("providerRequestId")},
                    )
            # model_request_started and the None-typed session.updated
            # variants (title/context bookkeeping) carry nothing new.
            return

        if top_type in (
            "turn.started", "turn.completed", "session.titleUpdated",
            "checkpoint.created", "streamRecovery.updated",
        ):
            # turn.completed duplicates the final "result" line (handled
            # separately in parse_artifacts); the rest is pure bookkeeping.
            return

        # Anything else is a genuinely unrecognized shape -- keep it visible
        # rather than silently dropping, since coverage above was derived
        # from one real run and may not be exhaustive.
        builder.add_step(
            source="system",
            message="",
            extra={"kind": top_type or "unknown", "raw": event},
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
