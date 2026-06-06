"""ForgecodeDeployer — drives the ``forge`` Rust CLI (tailcallhq/forgecode).

Pre-built binary downloaded from GitHub releases.  The deployer downloads
it in ``install()`` if not already on PATH.

forge runs headlessly via ``forge -p "<prompt>" --conversation-id <UUID>``.
All built-in tools (Read / Write / Shell / FsSearch / Patch / Fetch / ...)
execute directly inside the sandbox via Rust syscalls.  There is no Docker
layer and no CUA MCP bridge.

Provider routing: for OpenRouter, set ``ANTHROPIC_API_KEY`` to the
OpenRouter key and ``ANTHROPIC_BASE_URL`` to
``https://openrouter.ai/api/v1``.  For direct providers, export the
standard env var.  ``forge.toml`` pins the ``[session]`` provider/model.

Output: ``forge conversation dump <id>`` produces a ``ConversationDump``
JSON.  ``auto_dump = "json"`` in forge.toml also fires on TaskComplete.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
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

from .config import ForgecodeConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 2.0

# Vendored copy of crates/forge_services/permissions.default.yaml.
# Pre-writing this avoids forge's first-run init code path and matches the
# YOLO requirement: forge has full read/write/exec/url access.
PERMISSIONS_YAML_FULLY_OPEN = """\
policies:
  - permission: allow
    rule:
      read: "**/*"
  - permission: allow
    rule:
      write: "**/*"
  - permission: allow
    rule:
      command: "*"
  - permission: allow
    rule:
      url: "*"
"""


class ForgecodeDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the ``forge`` CLI (tailcallhq/forgecode)."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    @property
    def version(self) -> str | None:
        return self._forge_version if hasattr(self, "_forge_version") else None

    # =========================================================================
    # install
    # =========================================================================

    async def _auto_install_cli(self) -> None:
        """Download the forge binary from GitHub releases.

        Downloads a PINNED version (``cfg.forge_version``) from
        ``releases/download/v<version>/`` — never ``releases/latest`` — so
        every environment runs the same validated binary. forge's ``-p`` mode
        self-updates to latest on startup otherwise; the ``[updates]`` block
        in forge.toml disables that, and this pin governs fresh installs.
        Falls back to cargo install from source if the binary download fails.
        """
        cfg: ForgecodeConfig = self.config  # type: ignore[assignment]
        version = cfg.forge_version
        home = os.path.expanduser("~")
        bin_dir = f"{home}/.local/bin"
        os.makedirs(bin_dir, exist_ok=True)

        # Try downloading pre-built binary from GitHub releases.
        # Asset naming: forge-x86_64-unknown-linux-musl (statically linked,
        # works on any glibc version including Ubuntu 22.04's 2.35).
        # Download to a temp file first, then mv — avoids ETXTBSY when the
        # destination path is immediately executed.
        proc = await asyncio.to_thread(
            subprocess.run,
            [
                "bash", "-c",
                f'curl -fsSL '
                f'"https://github.com/tailcallhq/forgecode/releases/download/v{version}/forge-x86_64-unknown-linux-musl" '
                f'-o "{bin_dir}/forge.tmp" && chmod +x "{bin_dir}/forge.tmp" '
                f'&& mv -f "{bin_dir}/forge.tmp" "{bin_dir}/forge"',
            ],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode == 0:
            logger.info(
                "forgecode: installed via GitHub releases download — %s",
                (proc.stdout or "").strip()[-200:],
            )
        else:
            logger.warning(
                "forgecode: GitHub releases download failed (rc=%d), "
                "trying cargo install ...",
                proc.returncode,
            )
            # Fallback: cargo install from source (requires Rust toolchain).
            cargo_proc = await asyncio.to_thread(
                subprocess.run,
                [
                    "bash", "-c",
                    "cargo install "
                    "--git https://github.com/tailcallhq/forgecode "
                    "--bin forge forge_main",
                ],
                capture_output=True, text=True, timeout=600,
            )
            if cargo_proc.returncode != 0:
                raise RuntimeError(
                    f"forgecode install failed — both GitHub releases download "
                    f"(rc={proc.returncode}: {(proc.stderr or '')[:300]}) and "
                    f"cargo install (rc={cargo_proc.returncode}: "
                    f"{(cargo_proc.stderr or '')[:300]}) failed"
                )
            logger.info("forgecode: installed via cargo install")

        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

    async def install(self) -> None:
        cfg: ForgecodeConfig = self.config  # type: ignore[assignment]

        if not self.executor.sandbox.is_linux:
            raise NotImplementedError("forgecode is Linux-only")

        # ----------------------------------------------------------
        # 1. Ensure forge binary is available
        # ----------------------------------------------------------
        # IMPORTANT: prefer a PRE-BAKED forge before downloading. The sandbox
        # entry runs WITHOUT a login shell, so ~/.local/bin is not on PATH and
        # shutil.which("forge") misses an image-baked binary — which would make
        # us re-download forge `latest`. Newer forge (>=2.13.3) DROPPED reading
        # API keys from env vars and hangs on "Migrating credentials" headless,
        # so re-downloading silently breaks the agent. Check the baked location
        # explicitly first.
        forge_path = shutil.which("forge")
        if not forge_path:
            baked = os.path.join(os.path.expanduser("~"), ".local", "bin", "forge")
            if os.path.isfile(baked) and os.access(baked, os.X_OK):
                forge_path = baked
                logger.info("forgecode: using pre-baked forge at %s", baked)
        if not forge_path:
            logger.info("forgecode: 'forge' not on PATH or baked dir, installing ...")
            await self._auto_install_cli()
            forge_path = shutil.which("forge") or (
                os.path.join(os.path.expanduser("~"), ".local", "bin", "forge")
                if os.path.isfile(os.path.join(os.path.expanduser("~"), ".local", "bin", "forge"))
                else None
            )
            if not forge_path:
                raise RuntimeError(
                    "ForgecodeDeployer: 'forge' still not found after install"
                )
        self._forge_path = forge_path
        self._forge_version = "(skipped — overlay2 ETXTBSY)"
        logger.info("forgecode: binary at %s, skipping version probe", forge_path)

        # ----------------------------------------------------------
        # 2. Stage work dir
        # ----------------------------------------------------------
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------
        # 3. Write forge config files
        # ----------------------------------------------------------
        home = os.path.expanduser("~")
        forge_home = Path(home) / ".forge"
        forge_home.mkdir(parents=True, exist_ok=True)
        (forge_home / "logs").mkdir(parents=True, exist_ok=True)

        # permissions.yaml — fully permissive, no interactive prompts
        (forge_home / "permissions.yaml").write_text(
            PERMISSIONS_YAML_FULLY_OPEN, encoding="utf-8",
        )

        # forge.toml — model routing, auto_dump, session config
        (forge_home / ".forge.toml").write_text(
            cfg.render_forge_toml(), encoding="utf-8",
        )

        # Wipe .credentials.json so forge's env-var -> file migration
        # re-runs from the current env every time.  Without this, a stale
        # migrated key from a prior run would silently win.
        creds_file = forge_home / ".credentials.json"
        if creds_file.exists():
            creds_file.unlink()

        logger.info(
            "forgecode: config staged (model=%s, provider=%s)",
            cfg.model, cfg.provider,
        )

        # ----------------------------------------------------------
        # 4. CUA MCP bridge. forge supports MCP via a `.mcp.json` read from the
        #    project-local cwd AND the global ~/forge/.mcp.json (Anthropic MCP
        #    schema: mcpServers.<name>.{command,args,env}). Wire the cua server
        #    the same way every other MCP-capable agent does; CUA_SERVER_URL
        #    points the bridge at the image's cua-server port. The project-local
        #    copy (forge's cwd = dump_dir) is written in launch().
        # ----------------------------------------------------------
        sandbox = self.executor.sandbox
        from ale_run.agents._bootstrap import cua_bridge_env, ensure_cua_mcp_server
        await ensure_cua_mcp_server(sandbox)
        mcp_index = f"{sandbox.mcp_server_dir.rstrip('/')}/src/index.js"
        self._mcp_config = {
            "mcpServers": {
                "cua": {
                    "command": sandbox.node,
                    "args": [mcp_index],
                    "env": cua_bridge_env(self.executor),
                },
            },
        }
        # Global config home is ~/.forge (WITH dot) — same dir as .forge.toml.
        # NB: the forge docs say "~/forge/.mcp.json" but creating a no-dot
        # ~/forge dir derails forge's config resolution (it then thinks no
        # provider is set and drops into an interactive provider picker, which
        # ENXIOs on the missing TTY in headless `-p` mode). The real location is
        # ~/.forge/.mcp.json.
        (forge_home / ".mcp.json").write_text(
            json.dumps(self._mcp_config, indent=2), encoding="utf-8",
        )
        logger.info("forgecode: cua MCP bridge wired (~/.forge/.mcp.json)")

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: ForgecodeConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        prompt_file = wd / "prompt.txt"
        transcript_file = wd / "transcript.jsonl"
        stderr_log = wd / "stderr.log"
        pid_file = wd / "forge.pid"
        exit_code_file = wd / "exit_code.txt"
        dump_dir = wd / "dump_dir"
        dump_dir.mkdir(parents=True, exist_ok=True)

        # Project-local .mcp.json in forge's cwd (dump_dir). Project-local takes
        # precedence over the global ~/forge/.mcp.json written in install().
        mcp_cfg = getattr(self, "_mcp_config", None)
        if mcp_cfg:
            (dump_dir / ".mcp.json").write_text(
                json.dumps(mcp_cfg, indent=2), encoding="utf-8",
            )

        # Clean up previous run artifacts
        for f in (transcript_file, stderr_log, pid_file, exit_code_file):
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass

        prompt_file.write_text(prompt, encoding="utf-8")

        # Generate a conversation id for this run
        conversation_id = str(uuid.uuid4())

        argv = self._build_argv(cfg, str(prompt_file), conversation_id, str(dump_dir))
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
                cwd=str(dump_dir),
                start_new_session=True if hasattr(os, "setsid") else False,
            )
        pid_file.write_text(str(proc.pid), encoding="ascii")
        logger.info("forgecode: spawned pid=%s, conversation_id=%s", proc.pid, conversation_id)

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
        exit_code_file.write_text(str(exit_code), encoding="utf-8")

        # Post-run: materialise dump.json via forge conversation dump
        self._materialise_dump(cfg, conversation_id, wd, dump_dir)

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

    # =========================================================================
    # internals
    # =========================================================================

    def _build_argv(
        self,
        cfg: ForgecodeConfig,
        prompt_file: str,
        conversation_id: str,
        dump_dir: str,
    ) -> list[str]:
        """Build the forge CLI argv.

        Uses a shell wrapper to read the prompt from file (avoiding
        word-splitting issues with direct -p argument).
        """
        # forge -p expects the prompt as a string argument.  We use bash
        # to read the prompt file into a variable and pass it properly
        # quoted to avoid word-splitting.
        forge_bin = getattr(self, "_forge_path", "forge")
        return [
            "bash", "-c",
            f'PROMPT="$(cat {_quote(prompt_file)})"; '
            f'exec {_quote(forge_bin)} -p "$PROMPT" '
            f'--conversation-id {_quote(conversation_id)} '
            f'-C {_quote(dump_dir)}',
        ]

    def _build_env(self, cfg: ForgecodeConfig) -> dict[str, str]:
        """Build the environment for the forge process.

        For OpenRouter: sets ``ANTHROPIC_API_KEY`` to the OpenRouter key
        and ``ANTHROPIC_BASE_URL`` to ``https://openrouter.ai/api/v1``.
        For direct providers: exports the appropriate API key env var.
        """
        env = os.environ.copy()
        exec_env = dict(self.executor.env or {})
        for k, v in exec_env.items():
            env[k] = v

        # Secrets flow via ``self.executor.env`` (a sidecar writes the keys
        # into the executor env); resolve from there only, never from
        # ``os.environ``.
        def _key(name: str) -> str:
            return exec_env.get(name, "")

        if cfg.is_openrouter:
            # OpenRouter routing: forge >=2.13 DROPPED the old behaviour of
            # reading ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL to tunnel the
            # Anthropic protocol through OpenRouter. It now migrates known
            # provider env vars into a credentials store on first run; the
            # native ``open_router`` provider is keyed off OPENROUTER_API_KEY.
            # So pass the OpenRouter key under its own name and let forge's
            # built-in open_router provider (forge.toml provider_id) own it.
            # ANTHROPIC_* must NOT be set here, or forge migrates an
            # "anthropic" provider that shadows the requested open_router one.
            or_key = _key("OPENROUTER_API_KEY")
            if not or_key:
                raise RuntimeError(
                    "ForgecodeDeployer: OPENROUTER_API_KEY is not set in the "
                    "executor env."
                )
            env["OPENROUTER_API_KEY"] = or_key
            env.pop("ANTHROPIC_API_KEY", None)
            env.pop("ANTHROPIC_BASE_URL", None)
        else:
            # Direct provider: infer from model prefix
            prefix = cfg.model.split("/", 1)[0].lower()
            if prefix == "anthropic":
                key = _key("ANTHROPIC_API_KEY")
                if not key:
                    raise RuntimeError(
                        "ForgecodeDeployer: ANTHROPIC_API_KEY not set for "
                        "direct Anthropic provider."
                    )
                env["ANTHROPIC_API_KEY"] = key
            elif prefix == "openai" or prefix.startswith("gpt"):
                key = _key("OPENAI_API_KEY")
                if not key:
                    raise RuntimeError(
                        "ForgecodeDeployer: OPENAI_API_KEY not set for "
                        "direct OpenAI provider."
                    )
                env["OPENAI_API_KEY"] = key

        # Ensure cargo bin and local bin are on PATH
        home = os.path.expanduser("~")
        path = env.get("PATH", "")
        for extra in (f"{home}/.cargo/bin", f"{home}/.local/bin"):
            if extra not in path:
                path = f"{extra}:{path}"
        env["PATH"] = path
        env["NO_COLOR"] = "1"

        return env

    def _materialise_dump(
        self,
        cfg: ForgecodeConfig,
        conversation_id: str,
        work_dir: Path,
        dump_dir: Path,
    ) -> None:
        """Run ``forge conversation dump <id>`` and promote to dump.json.

        Best-effort: if the dump command fails we log and continue.
        The run still produced transcript.jsonl, stderr.log, exit_code.txt.
        """
        forge_bin = getattr(self, "_forge_path", "forge")
        env = self._build_env(cfg)
        dump_file = work_dir / "dump.json"

        try:
            subprocess.run(
                [forge_bin, "conversation", "dump", conversation_id],
                capture_output=True, text=True, timeout=120,
                cwd=str(dump_dir), env=env,
            )
        except Exception as exc:
            logger.warning("forgecode: conversation dump failed: %s", exc)

        # Pick the newest *-dump.json and copy to dump.json
        try:
            candidates = sorted(
                dump_dir.glob("*-dump.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                import shutil as _shutil
                _shutil.copy2(str(candidates[0]), str(dump_file))
                logger.info("forgecode: materialised dump.json from %s", candidates[0].name)
            else:
                logger.warning("forgecode: no *-dump.json found in %s", dump_dir)
        except Exception as exc:
            logger.warning("forgecode: dump materialisation failed: %s", exc)

    # =========================================================================
    # parse_artifacts
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: ForgecodeConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Parse forge's ConversationDump JSON into trajectory steps.

        Forge writes dumps to ``~/.forge/conversations/<id>/`` and the
        deployer materialises them as ``dump.json`` in ``work_dir``.
        Falls back to ``transcript.jsonl`` (stdout capture) if dump.json
        is missing.
        """
        dump_file = work_dir / "dump.json"
        if dump_file.exists():
            cls._parse_dump_json(dump_file, builder)
        else:
            # Fallback: parse transcript.jsonl (raw stdout)
            transcript_file = work_dir / "transcript.jsonl"
            if transcript_file.exists():
                cls._parse_transcript(transcript_file, builder)
            else:
                builder.add_step(
                    source="system",
                    message="forgecode: no dump.json or transcript found",
                    extra={"reason": "no_artifacts"},
                )

        builder.trajectory.extra.setdefault("forgecode", {}).update({
            "exit_code": run_result.exit_code,
            "dump_path": str(dump_file) if dump_file.exists() else None,
            "transcript_path": run_result.transcript_path,
        })

    @classmethod
    def _parse_dump_json(cls, dump_file: Path, builder: TrajectoryBuilder) -> None:
        """Parse the ConversationDump JSON format.

        The dump file structure (from ``crates/forge_main/src/ui.rs:60``):

        ::

            {
              "conversation": {
                "context": {
                  "messages": [
                    { "text": { "role": "...", "content": "...",
                                "tool_calls": [...], "reasoning_details": [...] },
                      "usage": { ... } },
                    { "tool": { "name": "...", "call_id": "...",
                                "output": { "values": [...] } } },
                    ...
                  ]
                }
              },
              "related_conversations": [...]
            }
        """
        try:
            dump = json.loads(dump_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            builder.add_step(
                source="system",
                message=f"forgecode: dump.json parse error: {exc}",
            )
            return

        conversation = dump.get("conversation") or {}
        context = conversation.get("context") or {}
        messages = context.get("messages") or []

        total_input_tokens = 0
        total_output_tokens = 0
        total_cached_tokens = 0
        total_cost = 0.0
        cost_seen = False

        for entry in messages:
            # Externally-tagged enum variants from Rust serde.
            if "text" in entry:
                cls._consume_text_message(entry["text"], builder)
            elif "tool" in entry:
                cls._consume_tool_result(entry["tool"], builder)
            elif "image" in entry:
                builder.add_step(source="user", message="[image]")

            # Accumulate usage
            usage = entry.get("usage") or {}
            prompt_tokens = _token_count(usage.get("prompt_tokens"))
            cached_tokens = _token_count(usage.get("cached_tokens"))
            completion_tokens = _token_count(usage.get("completion_tokens"))
            total_input_tokens += prompt_tokens
            total_cached_tokens += cached_tokens
            total_output_tokens += completion_tokens
            cost = usage.get("cost")
            if cost is not None:
                try:
                    total_cost += float(cost)
                    cost_seen = True
                except (TypeError, ValueError):
                    pass

        # Store aggregated usage in trajectory extra
        usage_summary: dict[str, Any] = {
            "uncached_input_tokens": max(total_input_tokens - total_cached_tokens, 0),
            "cache_read_input_tokens": total_cached_tokens,
            "output_tokens": total_output_tokens,
            "overall_input_tokens": total_input_tokens,
        }
        if cost_seen:
            usage_summary["total_cost_usd"] = total_cost
        builder.trajectory.extra.setdefault("forgecode", {})["usage"] = usage_summary

        # Route the dump-aggregated usage into a StepMetrics so finalize()
        # sums it. forge's per-message ``usage.prompt_tokens`` is the full
        # input (cache_read inclusive), so uncached = prompt - cached. Forge
        # reports cost via usage.cost.
        if total_input_tokens or total_output_tokens or cost_seen:
            builder.add_step(
                source="system",
                message=None,
                metrics=StepMetrics(
                    input_tokens=max(total_input_tokens - total_cached_tokens, 0),
                    output_tokens=total_output_tokens,
                    cache_read_tokens=total_cached_tokens or None,
                    cost_usd=total_cost if cost_seen else None,
                ),
                extra={"usage_dump": True},
            )

    @classmethod
    def _consume_text_message(
        cls, msg: dict[str, Any], builder: TrajectoryBuilder,
    ) -> None:
        """Process a text message entry from the dump."""
        role_raw = (msg.get("role") or "").strip().lower()
        # forge's Role enum serialises as PascalCase; normalise.
        role_map = {"system": "system", "user": "user", "assistant": "assistant"}
        role = role_map.get(role_raw, role_raw or "assistant")

        # Reasoning blocks (emitted before assistant text)
        for rd in msg.get("reasoning_details") or []:
            text = rd.get("text") if isinstance(rd, dict) else None
            if text:
                builder.add_step(source="agent", message=f"[reasoning] {text}")

        # Main content
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            source = "agent" if role == "assistant" else "user" if role == "user" else "system"
            builder.add_step(source=source, message=content)

        # Tool calls
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            tool_input = tc.get("arguments")
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except (TypeError, ValueError):
                    tool_input = {"raw": tool_input}
            if not isinstance(tool_input, dict):
                tool_input = {"raw": str(tool_input)}
            builder.add_step(
                source="agent",
                tool_calls=[ToolCall(
                    id=tc.get("call_id") or tc.get("id") or "",
                    name=str(tc.get("name") or ""),
                    arguments=tool_input,
                )],
            )

    @staticmethod
    def _image_part_from_forge(img: dict[str, Any]) -> ContentPart | None:
        """Build an image ContentPart from a forgecode image value.

        Shape: ``{"url": "data:image/png;base64,<payload>", "mime_type": ...}``
        (CUA screenshot) or a plain ``{"url": "https://..."}``. Returns ``None``
        when no usable url is present (caller falls back to a text placeholder).
        persist_screenshots() later moves the inline base64 to screenshots/.
        """
        url = img.get("url")
        if not isinstance(url, str) or not url:
            return None
        media_type = img.get("mime_type") or "image/png"
        if url.startswith("data:"):
            marker = "base64,"
            idx = url.find(marker)
            if idx == -1:
                return None
            data = url[idx + len(marker):]
            # Prefer the media type embedded in the data URL when present.
            header = url[5:idx]
            if header and ";" in header:
                media_type = header.split(";", 1)[0] or media_type
            return ContentPart(
                type="image",
                image=ImageSource(type="base64", media_type=media_type, data=data),
            )
        return ContentPart(type="image", image=ImageSource(type="url", url=url))

    @classmethod
    def _consume_tool_result(
        cls, result: dict[str, Any], builder: TrajectoryBuilder,
    ) -> None:
        """Process a tool result entry from the dump."""
        output = result.get("output") or {}
        chunks: list[str] = []
        image_parts: list[ContentPart] = []
        for v in output.get("values") or []:
            if not isinstance(v, dict):
                continue
            if "text" in v and isinstance(v["text"], str):
                chunks.append(v["text"])
            elif "image" in v and isinstance(v["image"], dict):
                # forgecode/CUA stores screenshots as a data: URL:
                # {"image": {"url": "data:image/png;base64,...", "mime_type": ...}}.
                # Keep them so persist_screenshots() can extract them instead of
                # collapsing to "[image]".
                img = v["image"]
                part = cls._image_part_from_forge(img)
                if part is not None:
                    image_parts.append(part)
                else:
                    chunks.append("[image]")
            else:
                chunks.append(json.dumps(v)[:200])

        content: list[ContentPart] = []
        if chunks:
            content.append(ContentPart(type="text", text="\n".join(chunks)))
        content.extend(image_parts)
        builder.add_step(
            source="environment",
            observation=Observation(results=[
                ToolResult(
                    tool_call_id=result.get("call_id") or result.get("id") or "",
                    content=content,
                    is_error=bool(output.get("is_error")),
                ),
            ]),
        )

    @classmethod
    def _parse_transcript(cls, transcript_file: Path, builder: TrajectoryBuilder) -> None:
        """Fallback: parse raw stdout as line-delimited text."""
        raw = transcript_file.read_text(encoding="utf-8", errors="replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                # If it's valid JSON, try to extract useful info
                if isinstance(event, dict):
                    msg = event.get("message") or event.get("content") or json.dumps(event)
                    builder.add_step(source="agent", message=str(msg))
                    continue
            except json.JSONDecodeError:
                pass
            # Plain text line
            builder.add_step(source="agent", message=line)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _quote(value: str) -> str:
    """POSIX-quote ``value`` for embedding in a bash command string."""
    return "'" + value.replace("'", "'\\''") + "'"


def _token_count(value: Any) -> int:
    """Coerce a forge ``TokenCount`` field into a Python int.

    forge serializes ``TokenCount`` as ``{"actual": <int>}``.  Older
    snapshots may emit a bare integer.  Accept both.
    """
    if isinstance(value, dict):
        inner = value.get("actual")
        if isinstance(inner, (int, float)):
            return int(inner)
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


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
