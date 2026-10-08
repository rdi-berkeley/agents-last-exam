"""OpenHandsCliDeployer — drives the official ``openhands-cli`` pip package.

Installed via ``uv pip install openhands-cli==<version>`` (or ``pip``
fallback).  No fork needed; the official package supports all required
headless flags.

Headless invocation::

    openhands --headless --json --yolo --override-with-envs \
              --exit-without-confirmation -t "<prompt>"

* ``--headless`` forces ``exit_without_confirmation = True`` and
  disables the LLM critic (the only path that asks for human input).
* ``--json`` (only valid with ``--headless``) installs ``json_callback``
  which prints ``--JSON Event--`` delimiter lines followed by
  pretty-printed ``Event.model_dump()`` for each event other than
  ``SystemPromptEvent``.
* ``--yolo`` (alias ``--always-approve``) auto-approves every tool call.
* ``--override-with-envs`` makes the CLI honour ``LLM_API_KEY`` /
  ``LLM_MODEL`` / ``LLM_BASE_URL`` from the process environment.

Bridge: the agent talks to the sandbox through the CUA MCP Server
bridge baked into sandbox images.  The deployer writes
``~/.openhands/mcp.json`` so the OpenHands runner picks it up at
start-up.

Trajectory recovery: stdout is the source of truth.  The runner script
tees stdout/stderr into files; the parser splits on the literal
``--JSON Event--`` delimiter, ``json.loads`` each block, and converts
the events into trajectory steps.
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

from .config import OpenHandsCliConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 5.0
_TERM_GRACE_S = 3.0

JSON_EVENT_DELIMITER = "--JSON Event--"


class OpenHandsCliDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the ``openhands-cli`` pip package."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    @property
    def version(self) -> str | None:
        cfg: OpenHandsCliConfig = self.config  # type: ignore[assignment]
        return cfg.cli_version

    # =========================================================================
    # install
    # =========================================================================

    async def install(self) -> None:
        cfg: OpenHandsCliConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        if not sandbox.is_linux:
            raise NotImplementedError("openhands_cli is Linux-only")

        # 1. Check if openhands binary already on PATH
        openhands_path = shutil.which("openhands")
        if not openhands_path:
            logger.info("openhands_cli: 'openhands' not on PATH, installing v%s ...",
                        cfg.cli_version)
            await self._install_cli(cfg.cli_version)
            openhands_path = shutil.which("openhands")
            if not openhands_path:
                raise RuntimeError(
                    "OpenHandsCliDeployer: 'openhands' still not found after install"
                )
        self._openhands_path = openhands_path

        # 2. Verify version
        try:
            probe = await asyncio.to_thread(
                subprocess.run,
                [openhands_path, "--version"],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"openhands --version timed out: {e}")
        logger.info("openhands_cli: CLI ok -- %s", (probe.stdout or "").strip())

        # 3. Create work directory
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # 4. Write ~/.openhands/.env with LLM credentials
        home = os.path.expanduser("~")
        openhands_home = Path(home) / ".openhands"
        openhands_home.mkdir(parents=True, exist_ok=True)

        env_content = self._build_env_file(cfg)
        env_file = openhands_home / ".env"
        env_file.write_text(env_content, encoding="utf-8")
        env_file.chmod(0o600)

        # 4b. Ensure the cua MCP bridge is installed at sandbox.mcp_server_dir
        #     (idempotent: no-op when prebaked, install when missing).
        from ale_run.agents._bootstrap import ensure_cua_mcp_server
        await ensure_cua_mcp_server(sandbox)

        # 5. Write ~/.openhands/mcp.json with CUA MCP server config -- but
        #    only if the CUA stdio server actually exists in this image.
        #    The ensure-step above installs it, but keep the existence guard:
        #    pointing mcp.json at a missing script makes OpenHands block during
        #    MCP client init and then die with a ``McpError: Connection
        #    closed`` before doing any work.  Without the file OpenHands falls
        #    back to its built-in terminal/file tools, which is sufficient for
        #    non-GUI tasks.
        mcp_path = openhands_home / "mcp.json"
        mcp_config = self._build_mcp_config(sandbox, self.executor.cua_bridge_url())
        mcp_index = mcp_config["mcpServers"]["cua"]["args"][0]
        if os.path.exists(mcp_index):
            mcp_path.write_text(
                json.dumps(mcp_config, indent=2), encoding="utf-8",
            )
        else:
            logger.warning(
                "openhands_cli: CUA MCP server not found at %s -- skipping "
                "mcp.json; OpenHands will use built-in tools only",
                mcp_index,
            )
            if mcp_path.exists():
                try:
                    mcp_path.unlink()
                except OSError:
                    pass

        # 6. Create conversations and work dirs
        (openhands_home / "conversations").mkdir(parents=True, exist_ok=True)

        logger.info("openhands_cli: config staged at %s (model=%s)",
                     openhands_home, cfg.model)

    async def _install_cli(self, version: str) -> None:
        """Install openhands-cli via uv pip (with pip fallback)."""
        home = os.path.expanduser("~")
        bin_dir = f"{home}/.local/bin"
        os.makedirs(bin_dir, exist_ok=True)

        pkg = f"openhands=={version}"

        # Try uv pip first
        uv_path = shutil.which("uv")
        if uv_path:
            proc = await asyncio.to_thread(
                subprocess.run,
                [uv_path, "pip", "install", pkg],
                capture_output=True, text=True, timeout=300,
            )
            if proc.returncode == 0:
                logger.info("openhands_cli: installed via uv pip -- %s",
                            (proc.stdout or "").strip()[-200:])
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
                return

            logger.warning(
                "openhands_cli: uv pip install failed (rc=%d), trying pip ...",
                proc.returncode,
            )

        # Fallback to pip
        pip_path = shutil.which("pip") or shutil.which("pip3")
        if not pip_path:
            raise RuntimeError(
                "OpenHandsCliDeployer: neither uv nor pip found on PATH"
            )
        proc = await asyncio.to_thread(
            subprocess.run,
            [pip_path, "install", pkg],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"openhands-cli install failed: rc={proc.returncode}, "
                f"stderr={( proc.stderr or '')[:400]}"
            )
        logger.info("openhands_cli: installed via pip -- %s",
                     (proc.stdout or "").strip()[-200:])

        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

    def _build_env_file(self, cfg: OpenHandsCliConfig) -> str:
        """Build the ~/.openhands/.env content."""
        # Secrets flow via ``self.executor.env`` (a sidecar writes the keys
        # into the executor env); resolve from there only, never from
        # ``os.environ``.
        exec_env = dict(self.executor.env or {})

        def _key(name: str) -> str:
            return exec_env.get(name, "")

        or_key = _key("OPENROUTER_API_KEY")
        anthropic_key = _key("ANTHROPIC_API_KEY")

        # Provider-driven routing (explicit, not a model-name heuristic).
        if cfg.provider == "openrouter":
            # A literal cfg.api_key (travels with the serialized config) takes
            # precedence over OPENROUTER_API_KEY and avoids env collisions.
            api_key = cfg.api_key or or_key
            if not api_key:
                raise RuntimeError(
                    "openhands_cli: provider=openrouter but neither config "
                    "api_key nor OPENROUTER_API_KEY is set"
                )
            base_url = cfg.base_url or "https://openrouter.ai/api/v1"
            # LiteLLM routes via OpenRouter only when the model id carries
            # the ``openrouter/`` prefix; add it if the operator left a bare
            # model name.
            model = cfg.model if cfg.model.startswith("openrouter/") else f"openrouter/{cfg.model}"
        elif cfg.provider == "direct":
            api_key = cfg.api_key or anthropic_key
            if not api_key:
                raise RuntimeError(
                    "openhands_cli: provider=direct but neither config api_key "
                    "nor ANTHROPIC_API_KEY is set"
                )
            base_url = cfg.base_url or ""
            model = cfg.model
        else:
            raise RuntimeError(
                f"openhands_cli: unknown provider {cfg.provider!r} "
                "(expected 'openrouter' or 'direct')"
            )

        lines = [
            "# Written by OpenHandsCliDeployer.install (do not edit by hand)",
            f"LLM_API_KEY={api_key}",
            f"LLM_MODEL={model}",
        ]
        if base_url:
            lines.append(f"LLM_BASE_URL={base_url}")
        if cfg.disable_condenser:
            lines.append("OPENHANDS_DISABLE_CONDENSER=1")

        home = os.path.expanduser("~")
        lines.append(f"OPENHANDS_PERSISTENCE_DIR={home}/.openhands")

        for k, v in cfg.extra_envs.items():
            lines.append(f"{k}={v}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _build_mcp_config(sandbox: Any, cua_url: str) -> dict:
        """Build the ~/.openhands/mcp.json config for CUA MCP bridge."""
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
        cfg: OpenHandsCliConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        prompt_file = wd / "prompt.txt"
        stdout_log = wd / "stdout.log"
        stderr_log = wd / "stderr.log"
        transcript_file = wd / "transcript.jsonl"
        pid_file = wd / "openhands.pid"

        for f in (stdout_log, stderr_log, transcript_file, pid_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        prompt_file.write_text(prompt, encoding="utf-8")

        argv = self._build_argv(prompt_file)
        env = self._build_env(cfg)

        t0 = time.monotonic()
        with open(stdout_log, "wb") as fout, \
             open(stderr_log, "wb") as ferr:
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
        logger.info("openhands_cli: spawned pid=%s (model=%s)",
                     proc.pid, cfg.model)

        # Wait for the child to finish.
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

        # Token/cost usage is NOT in the JSON event stream — it lives only in
        # the persisted conversation's ``base_state.json``. Copy the freshest
        # one into work_dir so it gets gathered and parse_artifacts can read it.
        self._copy_base_state(wd)

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

    def _build_argv(self, prompt_file: Path) -> list[str]:
        """Build the openhands CLI invocation."""
        return [
            self._openhands_path,
            "--headless",
            "--json",
            "--yolo",
            "--override-with-envs",
            "--exit-without-confirmation",
            "-t", prompt_file.read_text(encoding="utf-8"),
        ]

    def _build_env(self, cfg: OpenHandsCliConfig) -> dict[str, str]:
        """Build the process env for the openhands subprocess."""
        env = os.environ.copy()
        # Fold in executor env (API keys etc.)
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        # Source the .env file into the env dict
        home = os.path.expanduser("~")
        env_file = Path(home) / ".openhands" / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    env[key] = val
        env["NO_COLOR"] = "1"
        env["HOME"] = home
        return env

    def _copy_base_state(self, wd: Path) -> None:
        """Copy the freshest conversation base_state.json into work_dir.

        OpenHands persists per-run token/cost accounting only in
        ``$OPENHANDS_PERSISTENCE_DIR/conversations/<id>/base_state.json``.
        The framework gathers files from work_dir, so we mirror that file
        in as ``base_state.json`` for parse_artifacts to consume.
        """
        home = os.path.expanduser("~")
        conv_root = Path(home) / ".openhands" / "conversations"
        if not conv_root.is_dir():
            return
        candidates = list(conv_root.glob("*/base_state.json"))
        if not candidates:
            return
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            shutil.copy2(newest, wd / "base_state.json")
        except OSError as e:
            logger.warning("openhands_cli: could not copy base_state.json: %s", e)

    # =========================================================================
    # parse_artifacts
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: OpenHandsCliConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Parse OpenHands JSON event stream into trajectory steps."""
        stdout_log = work_dir / "stdout.log"
        if not stdout_log.exists():
            builder.add_step(
                source="system",
                message=f"openhands_cli: no stdout at {stdout_log}",
                extra={"reason": "no_transcript"},
            )
            return

        raw = stdout_log.read_text(encoding="utf-8", errors="replace")
        events = cls._parse_json_event_stream(raw)

        # Write clean JSONL transcript for downstream tooling
        transcript_jsonl = work_dir / "transcript.jsonl"
        transcript_jsonl.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in events)
            + ("\n" if events else ""),
            encoding="utf-8",
        )

        if not events:
            builder.add_step(
                source="system",
                message="openhands_cli: no JSON events parsed from stdout",
                extra={"reason": "no_events"},
            )
            builder.trajectory.extra.setdefault("openhands_cli", {}).update({
                "exit_code": run_result.exit_code,
                "transcript_path": str(stdout_log),
            })
            return

        for ev in events:
            cls._consume_event(ev, builder)

        # Token/cost usage is not in the event stream — it lives in the
        # persisted base_state.json (copied into work_dir by launch()).
        # Emit one StepMetrics carrying the cumulative totals. OpenHands
        # reports cost via accumulated_cost.
        usage = cls._extract_usage_from_base_state(work_dir / "base_state.json")
        if usage:
            builder.add_step(
                source="system",
                message=None,
                metrics=StepMetrics(
                    prompt_tokens=usage.get("input_tokens"),
                    completion_tokens=usage.get("output_tokens"),
                    cached_tokens=usage.get("cache_read_tokens"),
                    cache_creation_tokens=usage.get("cache_write_tokens"),
                    cost_usd=usage.get("cost_usd"),
                ),
                extra={"usage_base_state": True},
            )

        builder.trajectory.extra.setdefault("openhands_cli", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(stdout_log),
            "event_count": len(events),
        })

    @staticmethod
    def _extract_usage_from_base_state(path: Path) -> dict[str, Any] | None:
        """Aggregate token/cost from OpenHands base_state.json.

        Usage lives under state["stats"]["usage_to_metrics"][<id>] with
        accumulated_token_usage (prompt/completion/cache_read/cache_write)
        and accumulated_cost. The CLI default agent uses two usage ids
        ("agent" and "condenser"); sum across all so totals reflect the run.
        prompt_tokens is the full input (cache_read inclusive), so the
        uncached input is prompt_tokens - cache_read_tokens.
        """
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        u2m = (state.get("stats") or {}).get("usage_to_metrics")
        if not isinstance(u2m, dict):
            return None
        total_input = total_output = total_cache_read = total_cache_write = 0
        total_cost = 0.0
        any_usage = False
        saw_cost = False
        for entry in u2m.values():
            if not isinstance(entry, dict):
                continue
            tu = entry.get("accumulated_token_usage")
            if isinstance(tu, dict):
                total_input += int(tu.get("prompt_tokens") or 0)
                total_output += int(tu.get("completion_tokens") or 0)
                total_cache_read += int(tu.get("cache_read_tokens") or 0)
                total_cache_write += int(tu.get("cache_write_tokens") or 0)
                any_usage = True
            cost = entry.get("accumulated_cost")
            if cost is not None:
                try:
                    total_cost += float(cost)
                    saw_cost = True
                except (TypeError, ValueError):
                    pass
        if not any_usage:
            return None
        return {
            "input_tokens": max(total_input - total_cache_read, 0),
            "output_tokens": total_output,
            "cache_read_tokens": total_cache_read or None,
            "cache_write_tokens": total_cache_write or None,
            "cost_usd": total_cost if saw_cost else None,
        }

    # ------------------------------------------------------------------
    # JSON-Event stream parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_event_stream(stdout_text: str) -> list[dict]:
        """Split the OpenHands CLI ``--json`` stdout into event dicts.

        Two emission formats are supported:

        * Newer CLIs (>= 1.16 / SDK >= 1.21) print each Event as a single
          compact JSON object on its own line, with no delimiter, and
          interleave plain status text (``Agent is working``, ``Goodbye!``,
          ``Initializing agent...``) on other lines.
        * Older CLIs prefixed every Event with a ``--JSON Event--``
          delimiter line followed by a pretty-printed (indent=2) object
          spanning multiple lines.

        We handle both by scanning the whole stream with
        ``JSONDecoder.raw_decode``: at each position that looks like the
        start of an object (``{``) we try to decode one JSON value and skip
        ahead past it; anything else is treated as status text and skipped.
        The ``--JSON Event--`` delimiter, when present, is just non-JSON
        text and is harmlessly ignored.
        """
        if not stdout_text:
            return []

        decoder = json.JSONDecoder()
        events: list[dict] = []
        text = stdout_text
        n = len(text)
        i = 0
        while i < n:
            ch = text[i]
            if ch != "{":
                i += 1
                continue
            try:
                parsed, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
            i = end
        return events

    # ------------------------------------------------------------------
    # Event consumption -> trajectory steps
    # ------------------------------------------------------------------

    @classmethod
    def _consume_event(cls, event: dict, builder: TrajectoryBuilder) -> None:
        kind = cls._event_kind(event)

        if kind in ("MessageEvent", "message"):
            cls._consume_message(event, builder)
        elif kind in ("ActionEvent", "action"):
            cls._consume_action(event, builder)
        elif kind in ("ObservationEvent", "observation", "UserRejectObservation"):
            cls._consume_observation(event, builder)
        elif kind in ("AgentErrorEvent", "agent_error", "error"):
            builder.add_step(
                source="system",
                message=str(event.get("error") or event.get("message") or str(event)),
            )
        elif kind in ("Condensation", "CondensationRequest"):
            builder.add_step(
                source="system",
                message=f"<{kind}>",
            )
        else:
            # Anything else (PauseEvent, ConversationStateUpdateEvent, etc.)
            builder.add_step(
                source="system",
                message="",
                extra={"kind": kind, "raw": event},
            )

    @staticmethod
    def _event_kind(ev: dict) -> str:
        return str(ev.get("kind") or ev.get("type") or "").strip()

    @classmethod
    def _consume_message(cls, event: dict, builder: TrajectoryBuilder) -> None:
        llm_msg = event.get("llm_message") if isinstance(event.get("llm_message"), dict) else None
        if not isinstance(llm_msg, dict):
            return
        role = llm_msg.get("role") or "assistant"
        text = cls._content_to_text(llm_msg.get("content"))
        if not text:
            return

        # Extract usage if available
        usage = cls._extract_event_usage(event)
        metrics: StepMetrics | None = None
        if usage:
            metrics = StepMetrics(
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                cached_tokens=usage.get("cache_read_tokens"),
                cache_creation_tokens=usage.get("cache_write_tokens"),
            )

        if role == "user":
            builder.add_step(source="user", message=text)
        elif role == "assistant":
            builder.add_step(source="agent", message=text, metrics=metrics)
        else:
            builder.add_step(source="system", message=text)

    @classmethod
    def _consume_action(cls, event: dict, builder: TrajectoryBuilder) -> None:
        tool_name = event.get("tool_name") or ""
        tool_call_id = event.get("tool_call_id") or ""
        action = event.get("action") if isinstance(event.get("action"), dict) else {}

        # Some action events carry a thought/summary
        summary = event.get("summary") or event.get("thought")
        if isinstance(summary, str) and summary.strip():
            builder.add_step(
                source="agent",
                reasoning_content=summary,
            )

        builder.add_step(
            source="agent",
            tool_calls=[ToolCall(
                tool_call_id=tool_call_id or f"oh_{tool_name}",
                function_name=tool_name,
                arguments=action if isinstance(action, dict) else {"_raw": str(action)},
            )],
        )

    @classmethod
    def _consume_observation(cls, event: dict, builder: TrajectoryBuilder) -> None:
        tool_name = event.get("tool_name") or ""
        tool_call_id = event.get("tool_call_id") or ""
        obs = event.get("observation") if isinstance(event.get("observation"), dict) else {}

        content_text, screenshot_b64 = cls._observation_to_text_and_image(obs)
        content_parts: list[ContentPart] = []
        if content_text:
            content_parts.append(ContentPart(type="text", text=content_text))

        extra: dict[str, Any] = {}
        if screenshot_b64:
            extra["_screenshot_b64"] = screenshot_b64

        builder.add_step(
            source="environment",
            observation=Observation(results=[
                ToolResult(
                    source_call_id=tool_call_id or f"oh_{tool_name}",
                    content=content_parts,
                    is_error=False,
                ),
            ]),
            extra=extra if extra else None,
        )

    # ------------------------------------------------------------------
    # Content helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    if isinstance(block.get("text"), str):
                        parts.append(block["text"])
                    elif block.get("type") == "image_url":
                        parts.append("[image]")
            return "\n".join(p for p in parts if p)
        return ""

    @staticmethod
    def _observation_to_text_and_image(obs: dict) -> tuple[str, str | None]:
        """Render an OpenHands observation as (text, optional screenshot).

        OpenHands observations are typed.  Common shapes:

        * Terminal / file editor: ``content`` is a list of
          ``{"type": "text", "text": "..."}`` blocks.
        * MCP tool: ``content`` mixes text with ``{"type": "image",
          "image_urls": ["data:image/png;base64,..."]}`` blocks.
        """
        if not isinstance(obs, dict):
            return (str(obs) if obs is not None else "", None)

        screenshot: str | None = None
        text_parts: list[str] = []

        def _consume_block(block: Any) -> None:
            nonlocal screenshot
            if not isinstance(block, dict):
                return
            kind = block.get("type")
            if kind == "text":
                t = block.get("text")
                if isinstance(t, str) and t:
                    text_parts.append(t)
                return
            if kind == "image":
                if screenshot is not None:
                    return
                urls = block.get("image_urls") or []
                if isinstance(urls, list):
                    for u in urls:
                        b64 = _parse_data_url(u)
                        if b64:
                            screenshot = b64
                            return
                src = block.get("source")
                if isinstance(src, dict):
                    data = src.get("data")
                    if isinstance(data, str) and data:
                        screenshot = data
                return

        for key in ("content", "output", "stdout", "result"):
            val = obs.get(key)
            if isinstance(val, list):
                for block in val:
                    _consume_block(block)
            elif isinstance(val, str) and val and not text_parts:
                text_parts.append(val)

        if not text_parts and not screenshot:
            text_parts.append(json.dumps(obs, ensure_ascii=False))

        return ("\n".join(text_parts), screenshot)

    @staticmethod
    def _extract_event_usage(ev: dict) -> dict[str, Any] | None:
        """Pull token / cost numbers out of any event that carries them."""
        candidates = []
        for k in ("usage", "token_metrics"):
            v = ev.get(k)
            if isinstance(v, dict):
                candidates.append(v)
        llm_resp = ev.get("llm_response") if isinstance(ev.get("llm_response"), dict) else None
        if llm_resp:
            for k in ("usage", "token_metrics"):
                v = llm_resp.get(k)
                if isinstance(v, dict):
                    candidates.append(v)
        if not candidates:
            return None
        merged: dict[str, Any] = {}
        for u in candidates:
            for src_key, dst_key in (
                ("input_tokens", "input_tokens"),
                ("prompt_tokens", "input_tokens"),
                ("output_tokens", "output_tokens"),
                ("completion_tokens", "output_tokens"),
                ("cache_read_input_tokens", "cache_read_tokens"),
                ("cache_creation_input_tokens", "cache_write_tokens"),
                ("cost", "cost"),
                ("total_cost", "cost"),
            ):
                if u.get(src_key) is not None and merged.get(dst_key) is None:
                    merged[dst_key] = u[src_key]
        return merged or None

    @staticmethod
    def _extract_final_output(events: list[dict]) -> str:
        """Return the last assistant message text from the event stream."""
        for ev in reversed(events):
            llm_msg = ev.get("llm_message") if isinstance(ev, dict) else None
            if not isinstance(llm_msg, dict):
                continue
            if llm_msg.get("role") != "assistant":
                continue
            content = llm_msg.get("content") or []
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            texts.append(text)
                if texts:
                    return "\n".join(texts)
        return ""


def _parse_data_url(url: Any) -> str | None:
    """Return base64 payload from ``data:image/<...>;base64,<payload>``."""
    if not isinstance(url, str):
        return None
    marker = "base64,"
    idx = url.find(marker)
    if idx < 0 or "data:image/" not in url[:idx]:
        return None
    b64 = url[idx + len(marker):].strip()
    return b64 or None


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
