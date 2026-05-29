"""HermesDeployer — drives the Hermes Agent CLI (cua-verse fork).

Hermes is a Python CLI agent from NousResearch.  ALE uses the
``cua-verse/hermes-agent`` fork on branch ``agenthle`` which carries
vision patches (MCP ImageContent -> multimodal follow-up, tool-result
truncation guards).

Install: git clone the fork, ``uv pip install -e ".[all]"``, download
Playwright Chromium, write ``~/.hermes/config.yaml`` with model config,
compression settings, and MCP server config (CUA bridge).

Launch: ``hermes chat -q "<prompt>" -Q --provider openrouter --model <m>
--toolsets <csv> --yolo --accept-hooks --ignore-rules --max-turns N
--pass-session-id``.

Transcript recovery: Hermes persists sessions in SQLite at
``~/.hermes/state.db``.  After the run the deployer finds the session id
(preferring the printed ``session_id: ...`` from ``--pass-session-id``,
falling back to the latest row whose ``started_at`` is at-or-after the
run start marker) and exports it with ``hermes sessions export``.
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

import yaml

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentConfig,
    BaseAgentDeployer,
    ContentPart,
    Observation,
    StepMetrics,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

from .config import HermesConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 5.0
_TERM_GRACE_S = 3.0

# Prepended to every user prompt so the chat process does not exit early
# when the agent schedules cron jobs or other background work.
KEEPALIVE_PREAMBLE = (
    "[runtime contract]\n"
    "You are running under ALE's external runner. The harness detects "
    "task completion by waiting for THIS hermes chat process to exit. Returning "
    "a final assistant message ends your session immediately and stops the run. "
    "Implications:\n"
    "  - If you schedule cron jobs, delegate work to subagents that report back "
    "asynchronously, or kick off any background process whose result you need, "
    "you MUST keep this session alive until that work is reflected in your "
    "context. Use the `terminal` tool with a foreground `sleep` (or polling "
    "loop) to wait. The run's wall-clock timeout will stop you if you wait too "
    "long; you cannot end early.\n"
    "  - Do not produce a 'final' answer that depends on data not yet observed. "
    "Wait, observe, then answer.\n"
    "  - When the task IS truly complete and all evidence is in your context, "
    "return your final answer normally -- that's the correct exit signal.\n"
    "[/runtime contract]\n\n"
)


class HermesDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the Hermes Agent CLI (cua-verse fork)."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = ("transcript.jsonl", "stderr.log")

    @property
    def version(self) -> str | None:
        return None  # discovered at install time

    # =========================================================================
    # install
    # =========================================================================

    async def install(self) -> None:
        """Install Hermes from the cua-verse fork and configure it.

        Uses subprocess.run() directly (not sandbox.run_command()) because
        install() executes inside the container via _sandbox_entry.py.
        """
        cfg: HermesConfig = self.config  # type: ignore[assignment]
        sandbox = self.executor.sandbox

        if not sandbox.is_linux:
            raise NotImplementedError("hermes is Linux-only")

        home = os.path.expanduser("~")
        hermes_home = f"{home}/.hermes"
        install_dir = f"{hermes_home}/hermes-agent"
        hermes_bin = f"{home}/.local/bin/hermes"

        async def _sh(cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
            return await asyncio.to_thread(
                subprocess.run,
                ["bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout,
            )

        # 1. Check if hermes already installed
        already_installed = os.path.isfile(hermes_bin) and os.access(hermes_bin, os.X_OK)

        if not already_installed:
            logger.info("hermes: not installed, cloning fork ...")

            # Ensure uv is available
            if not shutil.which("uv"):
                logger.info("hermes: bootstrapping uv ...")
                await _sh("curl -LsSf https://astral.sh/uv/install.sh | sh", timeout=120)
                uv_bin = f"{home}/.local/bin"
                if uv_bin not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{uv_bin}:{os.environ.get('PATH', '')}"

            # Clone the fork (remove stale dir from prior failed installs)
            os.makedirs(hermes_home, exist_ok=True)
            if os.path.exists(install_dir):
                shutil.rmtree(install_dir, ignore_errors=True)
            clone = await _sh(
                f"git clone --depth 1 -b agenthle "
                f"https://github.com/cua-verse/hermes-agent.git "
                f"'{install_dir}'",
                timeout=180,
            )
            if clone.returncode != 0:
                raise RuntimeError(
                    f"hermes: git clone failed (rc={clone.returncode}): "
                    f"{(clone.stderr or '')[:500]}"
                )

            # Install editable with all extras into a dedicated venv.
            #
            # We must NOT use `uv pip install --system`: the container's
            # system site-packages (/usr/local/lib/python3.x/dist-packages)
            # is root-owned and read-only for the run user, so any dep that
            # needs upgrading (e.g. opentelemetry) fails with
            # "Permission denied (os error 13)" — which previously surfaced
            # as an empty-stderr rc=1.  A venv under the install dir is
            # writable, isolates hermes' heavy [all] deps, and produces a
            # `.venv/bin/hermes` console script that the symlink step below
            # already knows how to find.
            uv = shutil.which("uv") or f"{home}/.local/bin/uv"
            venv_path = f"{install_dir}/.venv"
            install_cmd = (
                f"cd '{install_dir}' && "
                f"export PATH=\"{home}/.local/bin:$PATH\" && "
                f"'{uv}' venv '{venv_path}' --python 3.12 2>&1 && "
                f"VIRTUAL_ENV='{venv_path}' '{uv}' pip install -e '.[all]' 2>&1"
            )
            pip_result = await _sh(install_cmd, timeout=600)
            if pip_result.returncode != 0:
                # The install command uses `2>&1`, so the real error lands in
                # stdout, not stderr.  Surface both, preferring the tail of
                # stdout where pip prints the failing build output.
                combined = (
                    (pip_result.stdout or "") + (pip_result.stderr or "")
                ).strip()
                raise RuntimeError(
                    f"hermes: pip install failed (rc={pip_result.returncode}): "
                    f"...{combined[-1500:]}"
                )

            # Create symlink for the hermes binary
            os.makedirs(f"{home}/.local/bin", exist_ok=True)
            for venv_dir in ("venv", ".venv"):
                candidate = f"{install_dir}/{venv_dir}/bin/hermes"
                if os.path.isfile(candidate):
                    if os.path.exists(hermes_bin):
                        os.remove(hermes_bin)
                    os.symlink(candidate, hermes_bin)
                    break
            else:
                hermes_which = shutil.which("hermes")
                if hermes_which and not os.path.exists(hermes_bin):
                    os.symlink(hermes_which, hermes_bin)

            # Install Playwright Chromium (skip if already cached)
            pw_cached = any(
                d.startswith("chromium")
                for d in os.listdir(f"{home}/.cache/ms-playwright")
            ) if os.path.isdir(f"{home}/.cache/ms-playwright") else False
            if not pw_cached:
                logger.info("hermes: installing Playwright Chromium ...")
                await _sh(
                    f"cd '{install_dir}' && "
                    "npx --yes playwright install --with-deps chromium 2>&1",
                    timeout=300,
                )

        # Verify hermes is accessible
        verify = await _sh(
            f"export PATH=\"{home}/.local/bin:$PATH\" && "
            f"'{hermes_bin}' --version 2>&1 || "
            f"'{hermes_bin}' --help 2>&1 | head -3",
            timeout=30,
        )
        logger.info("hermes: CLI check -- %s", (verify.stdout or "").strip()[:200])

        # Write ~/.hermes/.env
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        for k, v in (self.executor.env or {}).items():
            if k == "OPENROUTER_API_KEY":
                or_key = v
        if not or_key and cfg.provider == "openrouter":
            raise RuntimeError(
                "HermesDeployer: OPENROUTER_API_KEY is not set. "
                "Export it or pass it via executor env before install()."
            )

        env_lines = [
            "# Written by HermesDeployer.install (do not edit by hand)",
            "HERMES_YOLO_MODE=1",
            "HERMES_ACCEPT_HOOKS=1",
            "HERMES_IGNORE_USER_CONFIG=0",
            "HERMES_DUMP_REQUESTS=1",
            f"HERMES_INFERENCE_PROVIDER={cfg.provider}",
            f"HERMES_INFERENCE_MODEL={cfg.model}",
        ]
        if or_key:
            env_lines.append(f"OPENROUTER_API_KEY={or_key}")
        env_lines.append("OPENROUTER_BASE_URL=https://openrouter.ai/api/v1")
        env_content = "\n".join(env_lines) + "\n"

        env_path = Path(hermes_home) / ".env"
        env_path.write_text(env_content)
        os.chmod(env_path, 0o600)

        # Ensure the cua MCP bridge is installed at sandbox.mcp_server_dir
        # (idempotent: no-op when prebaked, install when missing). This makes
        # _cua_bridge_available() below pass on a thin image instead of
        # silently degrading to no-MCP. A failed install raises here rather
        # than leaving a half-installed bridge that would wedge the MCP stdio
        # handshake — the orphaned-MCP-child hardening is handled separately.
        from ale_run.agents._bootstrap import ensure_cua_mcp_server
        await ensure_cua_mcp_server(sandbox)

        # Write ~/.hermes/config.yaml
        config_yaml = self._build_config_yaml(cfg, sandbox, self.executor.cua_bridge_url())
        (Path(hermes_home) / "config.yaml").write_text(config_yaml)

        # Create working directories
        wd = self.executor.work_dir
        for d in [f"{hermes_home}/logs", f"{hermes_home}/sessions", wd]:
            os.makedirs(d, exist_ok=True)

        logger.info("hermes: install complete")

    @staticmethod
    def _cua_bridge_index(sandbox: Any) -> str:
        """Path to the cua MCP bridge entry script in the substrate."""
        return sandbox.mcp_server_dir.rstrip("/") + "/src/index.js"

    @classmethod
    def _cua_bridge_available(cls, sandbox: Any) -> bool:
        """Whether the cua MCP bridge is present AND actually runnable.

        Gating on ``isfile(index.js)`` alone is not enough: a bridge dir
        that has ``src/index.js`` but no installed ``node_modules`` (e.g. an
        image where the bridge source was copied in but ``npm install`` never
        ran, or a partially-shipped tree) makes the node child die instantly
        with ``MODULE_NOT_FOUND`` on its first ``import`` of
        ``@modelcontextprotocol/sdk``. On a host whose PID 1 is a non-reaping
        init (the static-provider container runs ``bash -c "sleep infinity"``
        as PID 1) that dead child is reparented to PID 1 and never reaped, so
        its stdio pipe never closes and hermes' MCP stdio transport blocks on
        the connect handshake read FOREVER — wedging the whole run to the
        wall budget even though the task is done (observed: 600s+ timeouts on
        ale-kasm-dev with exit_code already 0). claude_code tolerates a dead
        MCP server (its CLI treats the failed connect as non-fatal); hermes'
        stdio transport does not, so we must only register a server we can
        actually launch to completion.

        We therefore require the SDK module the bridge imports first
        (``node_modules/@modelcontextprotocol/sdk``) to exist alongside the
        entry script. If either is missing the bridge is treated as
        unavailable and hermes runs without it (degrading like claude_code),
        which is correct on every substrate. The real "ship + npm install the
        bridge so cua MCP actually works" fix is tracked separately.
        See [[static-kasm-pid1-zombie-reaping]].
        """
        index = cls._cua_bridge_index(sandbox)
        if not os.path.isfile(index):
            return False
        # index.js lives at <bridge>/src/index.js; node_modules sits at the
        # bridge root (<bridge>/node_modules), i.e. two levels up from index.
        bridge_root = os.path.dirname(os.path.dirname(index))
        sdk_dir = os.path.join(bridge_root, "node_modules", "@modelcontextprotocol", "sdk")
        return os.path.isdir(sdk_dir)

    @classmethod
    def _build_config_yaml(cls, cfg: HermesConfig, sandbox: Any, cua_url: str) -> str:
        """Build the ~/.hermes/config.yaml content."""
        mcp_index = cls._cua_bridge_index(sandbox)
        hermes_work = f"{os.path.expanduser('~')}/hermes_work"

        mcp_available = cls._cua_bridge_available(sandbox)
        if not mcp_available:
            logger.warning(
                "hermes: cua MCP bridge unavailable (missing index.js or its "
                "node_modules) at %s — disabling the cua MCP bridge and the "
                "'mcp-cua' toolset for this run so a broken/absent bridge "
                "cannot wedge the MCP stdio handshake under a non-reaping "
                "PID 1",
                mcp_index,
            )

        config = {
            "model": {
                "default": cfg.model,
                "provider": cfg.provider,
                "context_length": cfg.context_length,
            },
            "agent": {
                "max_turns": int(cfg.max_turns or 100_000),
                "verbose": False,
                "reasoning_effort": "medium",
            },
            "terminal": {
                "backend": "local",
                "cwd": hermes_work,
                "timeout": 180,
                "lifetime_seconds": 600,
                "sudo_password": "",
            },
            "skills": {
                "creation_nudge_interval": 0,
            },
            "compression": {
                "enabled": True,
                "threshold": cfg.compression_threshold,
                "target_ratio": cfg.compression_target_ratio,
                "protect_last_n": cfg.compression_protect_last_n,
            },
            "delegation": {
                "max_iterations": 50,
                "subagent_auto_approve": True,
                "inherit_mcp_toolsets": True,
            },
            "code_execution": {
                "timeout": 300,
                "max_tool_calls": 50,
            },
            "platform_toolsets": {
                "cli": [
                    t for t in cfg.toolsets_enabled
                    if mcp_available or t != "mcp-cua"
                ],
            },
            "display": {
                "compact": True,
                "tool_progress": "off",
                "streaming": False,
                "bell_on_complete": False,
                "show_reasoning": False,
            },
            "stt": {"enabled": False},
            "mcp_servers": (
                {
                    "cua": {
                        "command": sandbox.node,
                        "args": [mcp_index],
                        "env": {"CUA_SERVER_URL": cua_url},
                        "timeout": 120,
                        "connect_timeout": 60,
                    },
                }
                if mcp_available
                else {}
            ),
            "hooks_auto_accept": True,
        }

        if cfg.provider == "openrouter":
            config["model"]["base_url"] = "https://openrouter.ai/api/v1"

        return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        """Spawn hermes chat and poll until completion or timeout.

        Runs inside the container via ``_sandbox_entry`` -- like
        :meth:`install`, every filesystem/process op uses local I/O and
        ``subprocess`` rather than the CUA ``sandbox.*`` bridge.  The bridge
        routes through the in-container HTTP server, which drops connections
        under host load and previously made ``launch`` die with
        ``write_file(...) failed after retries: transport error`` even when
        the agent itself was healthy.
        """
        cfg: HermesConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        home = os.path.expanduser("~")
        hermes_bin = f"{home}/.local/bin/hermes"
        hermes_home = f"{home}/.hermes"

        prompt_file = f"{wd}/prompt.txt"
        stdout_log = f"{wd}/stdout.log"
        stderr_log = f"{wd}/stderr.log"
        exit_file = f"{wd}/exit_code.txt"
        pid_file = f"{wd}/hermes.pid"
        start_marker = f"{wd}/start_ts.txt"
        session_id_file = f"{wd}/session_id.txt"
        session_file = f"{wd}/transcript.jsonl"
        runner_script = f"{wd}/hermes_runner.sh"
        launcher_script = f"{wd}/hermes_launcher.sh"
        work_dir = f"{home}/hermes_work"

        def _sh_sync(cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True, text=True, timeout=timeout,
            )

        # Clean previous run files
        for f in (
            stdout_log, stderr_log, exit_file, pid_file,
            start_marker, session_id_file, session_file,
        ):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
        os.makedirs(work_dir, exist_ok=True)

        # Prepare prompt
        wire_prompt = (KEEPALIVE_PREAMBLE + prompt) if cfg.keepalive_preamble else prompt
        Path(prompt_file).write_text(wire_prompt, encoding="utf-8")

        # Build the runner script. Drop the ``mcp-cua`` toolset from the
        # ``--toolsets`` flag when the cua MCP server isn't installed — it is
        # gated out of config.yaml's mcp_servers for the same reason (see
        # _build_config_yaml), and passing a toolset whose backing server was
        # never registered would otherwise wedge the run on this image's
        # non-reaping init. Recomputed here (not threaded from install)
        # because launch() runs in-sandbox and sees the real container FS.
        mcp_available = self._cua_bridge_available(self.executor.sandbox)
        mcp_index = self._cua_bridge_index(self.executor.sandbox)
        enabled_toolsets = [
            t for t in cfg.toolsets_enabled
            if mcp_available or t != "mcp-cua"
        ]
        toolsets_csv = ",".join(enabled_toolsets) if enabled_toolsets else ""
        provider_arg = cfg.provider or "openrouter"

        cli_parts = [
            f"'{hermes_bin}'",
            "chat",
            '-q "$(cat "$PROMPT")"',
            "-Q",
            f"--provider {provider_arg}",
            f'--model "{cfg.model}"',
            f"--max-turns {int(cfg.max_turns or 100_000)}",
            "--yolo",
            "--accept-hooks",
            "--ignore-rules",
            "--pass-session-id",
        ]
        if toolsets_csv:
            cli_parts.append(f'--toolsets "{toolsets_csv}"')

        cli_cmd = " \\\n  ".join(cli_parts)

        runner = f"""#!/bin/bash
set -u
PROMPT='{prompt_file}'
export HOME='{home}'
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
set -a
. '{hermes_home}/.env'
set +a

cd '{work_dir}'

date +%s > '{start_marker}'

{cli_cmd} \\
  > '{stdout_log}' 2> '{stderr_log}'
echo $? > '{exit_file}'

# Identify the session created by this run.
START_TS=$(cat '{start_marker}' 2>/dev/null || echo 0)
PRINTED_SID=$(grep -oE 'session_id:[[:space:]]*[A-Za-z0-9_]+' '{stderr_log}' '{stdout_log}' 2>/dev/null \\
  | head -1 | awk '{{print $NF}}')
SID=""
STATE_DB='{hermes_home}/state.db'
if [ -f "$STATE_DB" ]; then
  SID=$(START_TS_VAL="$START_TS" PRINTED_SID="$PRINTED_SID" python3 - <<'PYEOF' 2>/dev/null
import os, sqlite3, sys
db = os.environ.get('STATE_DB', '{hermes_home}/state.db')
start_ts = float(os.environ.get('START_TS_VAL', '0') or 0)
printed = (os.environ.get('PRINTED_SID') or '').strip()
try:
    conn = sqlite3.connect('{hermes_home}/state.db', timeout=5)
    if printed:
        row = conn.execute(
            'SELECT id FROM sessions WHERE id = ?',
            (printed,),
        ).fetchone()
        if row:
            sys.stdout.write(row[0])
            sys.exit(0)
    row = conn.execute(
        'SELECT id FROM sessions WHERE started_at >= ? '
        'ORDER BY started_at DESC LIMIT 1',
        (start_ts,),
    ).fetchone()
    if row is None:
        row = conn.execute(
            'SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1'
        ).fetchone()
    if row:
        sys.stdout.write(row[0])
finally:
    try: conn.close()
    except Exception: pass
PYEOF
)
fi
echo "$SID" > '{session_id_file}'
if [ -n "$SID" ]; then
  '{hermes_bin}' sessions export '{session_file}' --session-id "$SID" \\
    >>'{stderr_log}' 2>&1 || true
fi

# Deterministic teardown: hermes spawns a cua MCP stdio child (node
# index.js) plus helper python processes. On a host whose PID 1 is a
# non-reaping init (e.g. the static-provider container runs
# `bash -c "sleep infinity"` as PID 1), an MCP/agent child that escapes
# hermes' own shutdown is reparented to PID 1 and never reaped, leaving
# this runner's setsid session with a live straggler. The deployer polls
# this script's PID for liveness, so a straggler that keeps the session
# from tearing down wedges the whole run until the wall budget — even
# though the task finished and exit_code was already written. Under a
# real init (docker vnc_startup.sh / systemd) the child is reaped
# instantly and this never bites, which is why docker/gcloud pass clean.
#
# We are the session leader (launched via `setsid`), so our PID == our
# PGID. Kill every other member of our process group, then exit. This is
# post-task cleanup of our OWN descendants only (not a global pkill), so
# it cannot disrupt concurrent runs or unrelated MCP servers.
SELF_PID=$$
MY_PGID=$(ps -o pgid= -p "$SELF_PID" 2>/dev/null | tr -d ' ')
if [ -n "$MY_PGID" ]; then
  for victim in $(pgrep -g "$MY_PGID" 2>/dev/null); do
    [ "$victim" = "$SELF_PID" ] && continue
    kill -TERM "$victim" 2>/dev/null || true
  done
  sleep 1
  for victim in $(pgrep -g "$MY_PGID" 2>/dev/null); do
    [ "$victim" = "$SELF_PID" ] && continue
    kill -KILL "$victim" 2>/dev/null || true
  done
fi

# Defense-in-depth: the MCP stdio child is a `node <bridge>/src/index.js`
# process. If the MCP SDK spawned it in its own session (setsid) it leaves
# OUR process group and the pgid sweep above misses it; under a non-reaping
# PID 1 it would then linger and (via its still-open stdio pipe) could keep
# a hermes straggler alive. Match by the FULL bridge index path — which is
# unique to this substrate and is ours — so this targets only the cua MCP
# child we caused to exist and never a concurrent run or unrelated node
# process. ``pgrep -f`` matches against the whole command line.
BRIDGE_INDEX='{mcp_index}'
if [ -n "$BRIDGE_INDEX" ]; then
  for victim in $(pgrep -f "$BRIDGE_INDEX" 2>/dev/null); do
    [ "$victim" = "$SELF_PID" ] && continue
    kill -KILL "$victim" 2>/dev/null || true
  done
fi
exit 0
"""

        launcher = f"""#!/bin/bash
setsid bash '{runner_script}' </dev/null >/dev/null 2>&1 &
CHILD=$!
echo "$CHILD" > '{pid_file}'
disown $CHILD 2>/dev/null || true
"""

        Path(runner_script).write_text(runner, encoding="utf-8")
        Path(launcher_script).write_text(launcher, encoding="utf-8")
        os.chmod(runner_script, 0o755)
        os.chmod(launcher_script, 0o755)

        logger.info(
            "hermes: launching (model=%s, provider=%s, timeout=%.0fs)",
            cfg.model, cfg.provider, cfg.timeout_s,
        )
        await asyncio.to_thread(_sh_sync, f"bash '{launcher_script}'", 20)

        # Poll until done or timeout
        t0 = time.monotonic()
        deadline = t0 + cfg.timeout_s

        while True:
            if time.monotonic() > deadline:
                # Kill the agent process group, then the process.
                await asyncio.to_thread(
                    _sh_sync,
                    f"PID=$(cat '{pid_file}' 2>/dev/null) && "
                    f'[ -n "$PID" ] && kill -9 -$PID 2>/dev/null; '
                    f'[ -n "$PID" ] && kill -9 $PID 2>/dev/null; true',
                    15,
                )
                return AgentRunResult(
                    status="timeout",
                    transcript_path=str(session_file),
                    stderr_path=str(stderr_log),
                    duration_s=time.monotonic() - t0,
                    error=f"wall budget {cfg.timeout_s}s exceeded",
                )

            await asyncio.sleep(_POLL_INTERVAL_S)

            # Liveness check. NOTE: ``kill -0 $PID`` is NOT sufficient on a
            # host whose PID 1 does not reap (the static-provider container
            # runs `bash -c "sleep infinity"` as PID 1). When the runner bash
            # finishes, its parent shell has already `disown`ed it, so it is
            # reparented to PID 1; with no reaper it lingers as a <defunct>
            # ZOMBIE — which still owns its PID slot, so `kill -0` reports
            # success and we would poll "running" forever, hanging the run to
            # the wall budget even though exit_code was already written. We
            # therefore also inspect the process state via /proc and treat a
            # zombie ('Z') as done. (Under a real reaping init the PID is gone
            # immediately and the `kill -0` path alone already reports done.)
            pid_check = await asyncio.to_thread(
                _sh_sync,
                f"PID=$(cat '{pid_file}' 2>/dev/null); "
                f'if [ -z "$PID" ]; then echo no_pid; '
                f"elif ! kill -0 $PID 2>/dev/null; then echo done; "
                f"else "
                f"  ST=$(awk '{{print $3}}' /proc/$PID/stat 2>/dev/null); "
                f'  if [ "$ST" = "Z" ]; then echo done; else echo running; fi; '
                f"fi",
                60,
            )
            proc_status = (pid_check.stdout or "").strip()
            if proc_status in ("done", "no_pid", ""):
                # Give post-run session export a moment to finish
                await asyncio.sleep(_TERM_GRACE_S)
                break

        duration_s = time.monotonic() - t0

        # Read exit code
        exit_code: int | None = None
        try:
            exit_text = Path(exit_file).read_text(encoding="utf-8")
            exit_code = int(exit_text.strip())
        except (FileNotFoundError, ValueError):
            pass

        status = "completed" if (exit_code is not None and exit_code == 0) else "failed"
        error: str | None = None
        if status == "failed":
            error = self._diagnose_failure(stderr_log, stdout_log, exit_code)

        return AgentRunResult(
            status=status,
            pid=None,
            exit_code=exit_code,
            transcript_path=str(session_file),
            stderr_path=str(stderr_log),
            duration_s=duration_s,
            error=error,
        )

    @staticmethod
    def _diagnose_failure(
        stderr_log: str, stdout_log: str, exit_code: int | None,
    ) -> str:
        parts = [f"hermes failed (rc={exit_code})"]
        try:
            stderr_text = Path(stderr_log).read_text(encoding="utf-8", errors="replace")
            if stderr_text.strip():
                parts.append(f"stderr tail: ...{stderr_text.strip()[-800:]}")
        except (FileNotFoundError, OSError):
            pass
        try:
            stdout_text = Path(stdout_log).read_text(encoding="utf-8", errors="replace")
            if stdout_text.strip():
                parts.append(f"stdout tail: ...{stdout_text.strip()[-800:]}")
        except (FileNotFoundError, OSError):
            pass
        return " | ".join(parts)

    # =========================================================================
    # parse_artifacts
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: BaseAgentConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        """Parse the hermes session export (transcript.jsonl) into trajectory steps."""
        session_file = work_dir / "transcript.jsonl"
        if not session_file.exists():
            builder.add_step(
                source="system",
                message=f"hermes: no transcript at {session_file}",
                extra={"reason": "no_transcript"},
            )
            return

        # Read single-line JSONL session export
        session_data = cls._read_session_jsonl(session_file)
        if session_data is None:
            builder.add_step(
                source="system",
                message="hermes: transcript file exists but could not be parsed",
                extra={"reason": "parse_error"},
            )
            return

        # Parse messages into trajectory steps
        for msg in session_data.get("messages", []) or []:
            if not isinstance(msg, dict):
                continue
            cls._consume_message(msg, builder)

        # Extract usage and attach as extra
        usage = cls._extract_usage(session_data)
        builder.trajectory.extra.setdefault("hermes", {}).update({
            "exit_code": run_result.exit_code,
            "transcript_path": str(session_file),
            "usage": usage,
        })

    @classmethod
    def _consume_message(cls, msg: dict, builder: TrajectoryBuilder) -> None:
        """Convert a single hermes session message into trajectory step(s)."""
        role = msg.get("role") or ""
        content = msg.get("content") or ""
        tool_calls_raw = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")
        tool_name = msg.get("tool_name")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""

        if role == "system":
            builder.add_step(source="system", message=str(content))
            return

        if role == "user":
            builder.add_step(source="user", message=str(content))
            return

        if role == "assistant":
            # Reasoning step
            if reasoning:
                builder.add_step(
                    source="agent",
                    reasoning=str(reasoning),
                )
            # Text content
            if isinstance(content, str) and content.strip():
                tool_calls = cls._normalize_tool_calls(tool_calls_raw)
                tc_list = []
                for tc in tool_calls:
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    name = fn.get("name") or tc.get("name") or ""
                    args = cls._decode_tool_args(
                        fn.get("arguments") if fn else tc.get("arguments")
                    )
                    tc_list.append(ToolCall(
                        id=tc.get("id", ""),
                        name=name,
                        arguments=args,
                    ))
                builder.add_step(
                    source="agent",
                    message=content,
                    tool_calls=tc_list if tc_list else None,
                )
            elif tool_calls_raw:
                # Tool calls without text content
                tool_calls = cls._normalize_tool_calls(tool_calls_raw)
                tc_list = []
                for tc in tool_calls:
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    name = fn.get("name") or tc.get("name") or ""
                    args = cls._decode_tool_args(
                        fn.get("arguments") if fn else tc.get("arguments")
                    )
                    tc_list.append(ToolCall(
                        id=tc.get("id", ""),
                        name=name,
                        arguments=args,
                    ))
                if tc_list:
                    builder.add_step(
                        source="agent",
                        tool_calls=tc_list,
                    )
            return

        if role == "tool":
            builder.add_step(
                source="environment",
                observation=Observation(results=[
                    ToolResult(
                        tool_call_id=tool_call_id or "",
                        content=[ContentPart(
                            type="text",
                            text=str(content),
                        )],
                        is_error=False,
                    ),
                ]),
                extra={"tool_name": tool_name} if tool_name else {},
            )
            return

        # Unknown role -- keep as system note
        builder.add_step(
            source="system",
            message=f"[{role}] {content}",
            extra={"original_role": role},
        )

    # =========================================================================
    # helpers
    # =========================================================================

    @staticmethod
    def _read_session_jsonl(path: Path) -> dict | None:
        """Read a single-line JSONL session export.

        ``hermes sessions export ... --session-id <id>`` writes one JSON
        object per line; with ``--session-id`` only one line is emitted.
        Tolerates BOM and stray blank lines.
        """
        try:
            raw = path.read_bytes()
            # Strip BOM
            if raw.startswith(b"\xef\xbb\xbf"):
                raw = raw[3:]
            text = raw.decode("utf-8", errors="replace").strip()
        except OSError:
            return None
        if not text:
            return None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    @staticmethod
    def _normalize_tool_calls(raw: Any) -> list[dict]:
        """Normalize a message's ``tool_calls`` field to a list of dicts."""
        if raw is None:
            return []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        if not isinstance(raw, list):
            return []
        return [tc for tc in raw if isinstance(tc, dict)]

    @staticmethod
    def _decode_tool_args(raw: Any) -> dict:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return {"_raw": raw}
            return parsed if isinstance(parsed, dict) else {"_raw": parsed}
        return {}

    @staticmethod
    def _extract_usage(session_data: dict | None) -> dict:
        """Extract usage/cost info from a hermes session export."""
        if not session_data:
            return {}

        def _int(name: str) -> int:
            v = session_data.get(name)
            return int(v) if isinstance(v, (int, float)) else 0

        uncached = _int("input_tokens")
        cache_read = _int("cache_read_tokens")
        cache_write = _int("cache_write_tokens")
        output = _int("output_tokens")
        cost = session_data.get("actual_cost_usd")
        if not isinstance(cost, (int, float)):
            cost = session_data.get("estimated_cost_usd")
        if not isinstance(cost, (int, float)):
            cost = None

        return {
            "uncached_input_tokens": uncached,
            "output_tokens": output,
            "cache_read_input_tokens": cache_read,
            "cache_write_input_tokens": cache_write,
            "overall_input_tokens": uncached + cache_read + cache_write,
            "total_cost_usd": cost,
            "num_turns": session_data.get("api_call_count"),
        }
