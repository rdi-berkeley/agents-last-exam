"""OctavusCliDeployer - drives the public ``@octavus/agent`` (``octoagent``) CLI.

The agent under test is a cloud Octavus agent; the CLI runs on the machine that *is*
the ALE eval box and drives that machine's own desktop / filesystem / shell,
while the brain (model, prompt, tools, workers, skills, memory) runs in the
Octavus cloud. This is the same shape ALE expects from a ``sandbox``-executor
agent: read everything from the executor/sandbox handle, spawn a local CLI,
poll it under the orchestration wall budget.

Pure stdlib (``subprocess`` / ``pathlib`` / ``json`` / ``urllib``) plus the
shared node bootstrap. It uses only the public CLI and the consumer (agent-key)
API - no Octavus-internal or admin surface - so the integration relies only on
surfaces any user (including the ALE authors) can access.

CLI reference: https://octavus.ai/docs/workforce-agents/cli

Responsibilities:

* :meth:`install` - ensure node, the computer's display prerequisites, and the
  ``octoagent`` CLI (via npm); optionally fetch Chrome for Testing.
* :meth:`launch` - ``octoagent run --json --workdir <task variant dir>`` on the
  graded desktop, spawned detached and polled so the wall budget can reap it.
* :meth:`parse_artifacts` - host-side: read the ``--json`` result, then read the
  observable thread with the same agent key for the transcript + per-run cost.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import ClassVar

from ale_run.base_interface import (
    AgentRunResult,
    BaseAgentDeployer,
    ContentPart,
    Observation,
    ToolCall,
    ToolResult,
    TrajectoryBuilder,
)

from .config import OctavusCliConfig

logger = logging.getLogger(__name__)


_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 3.0

# Terminal thread statuses on the consumer read surface.
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

# The computer's display + accessibility stack the CLI's browser / computer-use
# tools need. Mirrors the public install script's apt set; a shell/filesystem-
# only run needs none of it, so a failure here is a warning, not fatal.
# `fluxbox` is a lightweight window manager the CLI starts on a bare display so the
# computer-use `label` driver can find the foreground window; a no-op on the
# standard ALE `:0`, which already runs a WM.
_PREREQ_APT_PACKAGES = (
    "xvfb", "fluxbox", "dbus-x11", "at-spi2-core", "x11-utils", "xdotool", "wmctrl", "scrot", "ffmpeg",
    "python3", "python3-gi", "gir1.2-atspi-2.0", "python3-pil",
    "libnss3", "libatk-bridge2.0-0", "libgtk-3-0", "libgbm1",
    "fonts-liberation",
)
# The ALSA runtime lib is resolved at install time, not listed above: libasound2
# became a virtual package on Ubuntu 24.04 (renamed libasound2t64 in the 64-bit
# time_t transition), so the literal name has no install candidate there.

# The computer-use MCP runs its AT-SPI `label` driver (linux-driver.py) as
# `python3`, resolved from PATH. That driver imports the distro PyGObject
# bindings (python3-gi), which exist ONLY for the system interpreter - never in
# the uv/conda venvs some images prepend to PATH. The ALE image, for one, puts
# /opt/cua-server/.venv/bin (a gi-less uv Python 3.14) first, so a bare `python3`
# lands there and `label` dies with "AT-SPI2 not available" even though the
# packages are installed for /usr/bin/python3. Pin the driver to the system
# interpreter that carries the bindings (cloud/desktop/VM never hit this: their
# `python3` is already that interpreter).
_SYSTEM_PYTHON = "/usr/bin/python3"
_SYSTEM_BIN = os.path.dirname(_SYSTEM_PYTHON)

# Absolute path tokens the task prompt renders (backtick spans + bare POSIX
# paths), used to locate the task's output dir -> variant dir.
_BACKTICK = re.compile(r"`([^`]+)`")
_UNIXPATH = re.compile(r"/(?:[\w.\-]+/)+[\w.\-]+")
# octoagent --json threadUrl is ``<platform>/agents/<agentId>/chat/<threadId>``.
_AGENT_URL = re.compile(r"/agents/([^/]+)/chat/")


def _variant_dir_from_prompt(prompt: str, fallback: str) -> str:
    """Best-effort absolute variant directory (the parent of ``output/``).

    ALE task descriptions render absolute paths; the graded output directory is
    the segment ending at ``output`` (either the dir itself or a file under it),
    and the variant dir is its parent. Rooting the CLI's ``--workdir`` there puts
    the filesystem/shell tools where ``input/`` and ``output/`` live and match
    the prompt's paths. Falls back to the whole task-data root (the only staged
    variant lives under it) when no output path is present.
    """
    candidates: set[str] = set()
    for span in _BACKTICK.findall(prompt):
        for token in span.split():
            token = token.strip().strip("\"'").rstrip("/")
            if token.startswith("/"):
                candidates.add(token)
    for match in _UNIXPATH.findall(prompt):
        candidates.add(match.rstrip("/"))

    variant_dirs: list[str] = []
    for path in candidates:
        segments = path.split("/")
        output_idx = [i for i, seg in enumerate(segments) if seg == "output"]
        if output_idx:
            variant_dirs.append("/".join(segments[: output_idx[-1]]))
    # A task renders one output tree in practice; ``min`` is only a deterministic
    # tie-break for the rare prompt that references several output paths.
    return min(variant_dirs) if variant_dirs else fallback


def _read_text_tolerant(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""


class OctavusCliDeployer(BaseAgentDeployer):
    """Stdlib-only deployer for the ``@octavus/agent`` (``octoagent``) CLI."""

    default_executor: ClassVar[str] = "sandbox"
    supported_executors: ClassVar[frozenset[str]] = frozenset({"sandbox"})
    hot_artifacts: ClassVar[tuple[str, ...]] = (
        "octoagent.result.json",
        "octoagent.stdout.log",
        "octoagent.stderr.log",
    )

    _octoagent_path: str
    _chrome_path: str | None = None

    @property
    def version(self) -> str | None:
        cfg: OctavusCliConfig = self.config  # type: ignore[assignment]
        return cfg.cli_version

    # =========================================================================
    # install
    # =========================================================================

    async def install(self) -> None:
        cfg: OctavusCliConfig = self.config  # type: ignore[assignment]

        from ale_run.agents._bootstrap import ensure_node_npm
        _, npm = await ensure_node_npm()

        if cfg.install_prereqs:
            await self._ensure_computer_prereqs()

        pkg = cfg.cli_version or "@octavus/agent"
        env = {**os.environ, "npm_config_cache": os.path.join(os.path.expanduser("~"), ".npm-ale")}
        proc = await asyncio.to_thread(
            subprocess.run,
            [npm, "install", "-g", "--force", pkg],
            capture_output=True, text=True, timeout=600, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"octavus_cli: npm install -g {pkg} failed "
                f"(rc={proc.returncode}): {(proc.stderr or '')[-500:]}"
            )

        octoagent = self._resolve_octoagent()
        if not octoagent:
            raise RuntimeError(
                f"octavus_cli: 'octoagent' not found after npm install -g {pkg}"
            )
        self._octoagent_path = octoagent
        logger.info("octavus_cli: octoagent CLI ready at %s", octoagent)

        self._chrome_path = cfg.chrome_path or await self._ensure_chrome()

        Path(self.executor.work_dir).mkdir(parents=True, exist_ok=True)

    async def _ensure_computer_prereqs(self) -> None:
        """Best-effort ``apt-get`` of the display + accessibility stack.

        Skipped only when the box already has BOTH the screenshot/automation
        tools AND the AT-SPI2 python bindings the computer-use ``label`` driver
        imports. The screenshot/automation binaries do not imply those bindings:
        an image can bake a desktop plus scrot/xdotool/ffmpeg (for other agents)
        yet omit python3-gi/gir1.2-atspi-2.0, which leaves ``computer-use__label``
        dead with "AT-SPI2 not available" while screenshots still work. Non-fatal:
        a shell/filesystem-only task still runs, and a browser/computer-use task
        surfaces a clear diagnostic itself.
        """
        def _atspi_bindings_importable() -> bool:
            # The `label` driver reaches AT-SPI2 through GObject introspection.
            # Probe the SYSTEM interpreter the MCP is pinned to (see _SYSTEM_PYTHON
            # / _build_env), not a bare `python3` that a venv on PATH would shadow -
            # otherwise the check reflects the wrong interpreter and apt runs every
            # time even though the bindings are present for /usr/bin/python3.
            try:
                return subprocess.run(
                    [_SYSTEM_PYTHON, "-c",
                     "import gi; gi.require_version('Atspi', '2.0'); from gi.repository import Atspi"],
                    capture_output=True, text=True, timeout=30, check=False,
                ).returncode == 0
            except (OSError, subprocess.SubprocessError):
                return False

        have_tools = bool(
            shutil.which("scrot") and shutil.which("xdotool") and shutil.which("ffmpeg")
        )
        if have_tools and await asyncio.to_thread(_atspi_bindings_importable):
            logger.info("octavus_cli: computer prerequisites already present")
            return
        packages = " ".join(_PREREQ_APT_PACKAGES)
        # Pick the real ALSA package the distro ships (libasound2t64 on Ubuntu 24.04+,
        # libasound2 on older Debian/Ubuntu), mirroring the public installer.
        cmd = (
            "sudo apt-get update -y; "
            "alsa_pkg=libasound2t64; "
            "apt-cache show libasound2t64 >/dev/null 2>&1 || alsa_pkg=libasound2; "
            f'sudo apt-get install -y --no-install-recommends {packages} "$alsa_pkg"'
        )
        proc = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            logger.warning(
                "octavus_cli: prerequisite apt-get failed (rc=%s); browser/"
                "computer-use tools may be unavailable: %s",
                proc.returncode, (proc.stderr or "")[-400:],
            )
        else:
            logger.info("octavus_cli: installed computer prerequisites")

    async def _ensure_chrome(self) -> str | None:
        """Install Chrome for Testing to ``~/.octavus/browsers`` and return its path.

        The browser tools load the automation extension with ``--load-extension``,
        which branded Chrome 137+ removed and the Chromium snap's confinement blocks;
        only Chrome for Testing (or a real Chromium) still supports it. The ALE images
        bake branded Google Chrome, which the CLI refuses. The deployer therefore
        provisions Chrome for Testing to ``~/.octavus/browsers`` - the exact location
        the CLI auto-detects (matching the public installer) - so it is the browser
        that comes up during ``computer-ensure-ready``, and also points the CLI at the
        binary explicitly via ``--chrome-path``. Set ``chrome_path`` in the agent
        config to pin your own CfT.
        """
        proc = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", 'npx --yes @puppeteer/browsers install chrome@stable --path "$HOME/.octavus/browsers"'],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            logger.warning(
                "octavus_cli: could not install Chrome for Testing; browser tasks "
                "may fail to load the extension: %s",
                (proc.stderr or proc.stdout or "")[-400:],
            )
            return None
        # @puppeteer/browsers lays the build out as
        # ~/.octavus/browsers/chrome/<platform>-<buildId>/chrome-linux64/chrome (the
        # layout the CLI's resolver scans); pick the newest install.
        browsers_dir = Path(os.path.expanduser("~")) / ".octavus" / "browsers"
        binaries = sorted(
            browsers_dir.glob("chrome/*/chrome-linux64/chrome"),
            key=lambda p: p.stat().st_mtime,
        )
        if binaries:
            chrome = str(binaries[-1])
            logger.info("octavus_cli: using Chrome for Testing at %s", chrome)
            return chrome
        logger.warning(
            "octavus_cli: Chrome for Testing not found under %s after install", browsers_dir
        )
        return None

    @staticmethod
    def _resolve_octoagent() -> str | None:
        home = os.path.expanduser("~")
        for candidate in (
            os.path.join(home, ".npm-global", "bin", "octoagent"),
            os.path.join(home, ".local", "bin", "octoagent"),
        ):
            if os.path.exists(candidate):
                return candidate
        return shutil.which("octoagent")

    # =========================================================================
    # launch
    # =========================================================================

    async def launch(self, prompt: str) -> AgentRunResult:
        cfg: OctavusCliConfig = self.config  # type: ignore[assignment]
        wd = Path(self.executor.work_dir)
        wd.mkdir(parents=True, exist_ok=True)

        result_file = wd / "octoagent.result.json"
        stdout_log = wd / "octoagent.stdout.log"
        stderr_log = wd / "octoagent.stderr.log"
        pid_file = wd / "octoagent.pid"
        for stale in (result_file, stdout_log, stderr_log, pid_file):
            if stale.exists():
                try:
                    stale.unlink()
                except OSError:
                    pass

        workdir = _variant_dir_from_prompt(prompt, self.executor.sandbox.task_data_root)
        argv = self._build_argv(cfg, workdir=workdir, prompt=prompt)
        env = self._build_env(cfg)

        t0 = time.monotonic()
        with open(stdout_log, "wb") as out, open(stderr_log, "wb") as err:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                argv,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                env=env,
                cwd=str(wd),
                start_new_session=hasattr(os, "setsid"),
            )
        pid_file.write_text(str(proc.pid), encoding="ascii")
        logger.info("octavus_cli: spawned pid=%s workdir=%s", proc.pid, workdir)

        try:
            while proc.poll() is None:
                await asyncio.sleep(_POLL_INTERVAL_S)
        except asyncio.CancelledError:
            self._reap(proc, force=False)
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(proc.wait), timeout=_TERM_GRACE_S,
                )
            except (TimeoutError, asyncio.CancelledError):
                self._reap(proc, force=True)
            raise

        duration_s = time.monotonic() - t0
        exit_code = proc.returncode
        result = self._extract_result_json(stdout_log, result_file)
        status = "completed" if exit_code == 0 else "failed"
        error: str | None = None
        if status == "failed":
            error = self._diagnose_failure(exit_code, result, stderr_log)

        return AgentRunResult(
            status=status,
            pid=proc.pid,
            exit_code=exit_code,
            transcript_path=str(result_file),
            stderr_path=str(stderr_log),
            duration_s=duration_s,
            error=error,
        )

    def _build_argv(self, cfg: OctavusCliConfig, *, workdir: str, prompt: str) -> list[str]:
        # Use the CLI's built-in default platform unless the caller pins a
        # platform_url to point the run at a different platform deployment.
        argv = [self._octoagent_path, "run", "--json", "--workdir", workdir]
        if cfg.platform_url:
            argv += ["--platform-url", cfg.platform_url]
        if cfg.operator_url:
            argv += ["--operator-url", cfg.operator_url]
        if cfg.model:
            argv += ["--model", cfg.model]
        if cfg.backup_model:
            argv += ["--backup-model", cfg.backup_model]
        if cfg.thinking:
            argv += ["--thinking", cfg.thinking]
        for slug, enabled in cfg.capabilities.items():
            argv += ["--capability", f"{slug}={'on' if enabled else 'off'}"]
        if cfg.record:
            # --record-public implies recording and stores at a permanent, shareable
            # URL; --record alone keeps the recording private (signed playback).
            argv.append("--record-public" if cfg.record_visibility == "public" else "--record")
        chrome = cfg.chrome_path or self._chrome_path
        if chrome:
            argv += ["--chrome-path", chrome]
        argv.append(prompt)
        return argv

    def _build_env(self, cfg: OctavusCliConfig) -> dict[str, str]:
        env = os.environ.copy()
        for k, v in (self.executor.env or {}).items():
            env[k] = v
        env["DISPLAY"] = cfg.display
        # Put the system bin first so the MCP-spawned `python3` (the AT-SPI
        # `label` driver) is the gi-carrying system interpreter, not a gi-less
        # venv the image prepends to PATH. The MCP child inherits this PATH, so
        # this is what makes computer-use labeling work. See _SYSTEM_PYTHON.
        if os.path.isfile(_SYSTEM_PYTHON):
            env["PATH"] = _SYSTEM_BIN + os.pathsep + env.get("PATH", "")
        if cfg.api_key:
            # Pass the key by env, never argv, so it stays out of the process
            # list and any gathered log.
            env["OCTAVUS_API_KEY"] = cfg.api_key
        return env

    @staticmethod
    def _reap(proc: subprocess.Popen, *, force: bool) -> None:
        """Signal the CLI and its whole process group on wall-budget cancellation.

        ``launch()`` sets ``start_new_session=True`` on POSIX, so the child leads
        its own process group; signalling the group (``killpg``) reaps the CLI's
        children (browser, MCP drivers) instead of orphaning them, falling back to
        the single child where the group is gone. ``force`` picks SIGKILL over
        SIGTERM. The caller does the bounded wait between the two signals.
        """
        sig = signal.SIGKILL if force else signal.SIGTERM
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

    @staticmethod
    def _extract_result_json(stdout_log: Path, result_file: Path) -> dict | None:
        """Pull the single ``--json`` object off stdout and persist it."""
        text = _read_text_tolerant(stdout_log)
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                result_file.write_text(json.dumps(obj, indent=2), encoding="utf-8")
                return obj
        return None

    @staticmethod
    def _diagnose_failure(
        exit_code: int | None, result: dict | None, stderr_log: Path,
    ) -> str:
        parts = [f"octoagent exited rc={exit_code}"]
        if exit_code == 2:
            parts.append("bad usage (check --platform-url / --api-key)")
        elif exit_code == 3:
            parts.append("CLI update required (platform minimum version)")
        if result and result.get("error"):
            parts.append(f"error: {result['error']}")
        stderr_text = _read_text_tolerant(stderr_log)
        if stderr_text.strip():
            parts.append(f"stderr tail: ...{stderr_text[-600:]}")
        return " | ".join(parts)

    # =========================================================================
    # parse_artifacts - host-side, runs on the gathered work_dir
    # =========================================================================

    @classmethod
    def parse_artifacts(
        cls,
        *,
        work_dir: Path,
        config: OctavusCliConfig,
        run_result: AgentRunResult,
        builder: TrajectoryBuilder,
    ) -> None:
        result_file = work_dir / "octoagent.result.json"
        if not result_file.exists():
            builder.add_step(
                source="system",
                message=f"octavus_cli: no result json at {result_file}",
                extra={"reason": "no_result"},
            )
            return

        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            builder.add_step(
                source="system",
                message=f"octavus_cli: unreadable result json: {exc}",
                extra={"reason": "bad_result"},
            )
            return

        thread_url = str(result.get("threadUrl") or "")
        thread_id = str(result.get("threadId") or "")
        builder.trajectory.extra.setdefault("octavus_cli", {}).update({
            "thread_id": thread_id,
            "thread_url": thread_url,
            "cli_status": result.get("status"),
            "exit_code": run_result.exit_code,
        })

        agent_match = _AGENT_URL.search(thread_url)
        # The sandbox strips config.api_key from the gathered spec (it is a
        # secret), so host-side reads fall back to the env the CLI key lives in.
        api_key = (
            config.api_key
            or os.environ.get("OCTAVUS_AGENT_API_KEY")
            or os.environ.get("OCTAVUS_API_KEY")
        )
        platform_match = re.match(r"(https?://[^/]+)", thread_url)
        platform_url = platform_match.group(1) if platform_match else None
        if not (agent_match and thread_id and api_key and platform_url):
            builder.add_step(
                source="system",
                message=(
                    "octavus_cli: cannot read thread "
                    f"(url={thread_url or 'missing'}); recorded CLI result only"
                ),
                extra={"reason": "thread_unreadable"},
            )
            return

        octo_agent_id = agent_match.group(1)
        thread = cls._read_thread(platform_url, octo_agent_id, thread_id, api_key)
        if thread is None:
            builder.add_step(
                source="system",
                message=f"octavus_cli: thread read failed; view: {thread_url}",
                extra={"reason": "thread_read_error"},
            )
            return

        cls._map_messages(thread.get("messages") or [], builder)

        usage = thread.get("usage") or {}
        if usage:
            builder.override_final_metrics(
                total_cost_usd=usage.get("costUsd"),
                total_input_tokens=usage.get("inputTokens"),
                total_output_tokens=usage.get("outputTokens"),
            )
        run_config = thread.get("runConfig") or {}
        model = run_config.get("model") or config.model
        if model:
            builder.trajectory.agent.model = model

        recording = thread.get("recording") or {}
        builder.trajectory.extra["octavus_cli"].update({
            "thread_status": thread.get("status"),
            "failure_reason": thread.get("failureReason"),
            "usage": usage or None,
            "run_config": run_config or None,
            "recording": recording or None,
        })

    @classmethod
    def _read_thread(
        cls, platform_url: str, agent_id: str, thread_id: str, api_key: str,
    ) -> dict | None:
        """Read the observable thread, polling briefly for terminal status/usage.

        The CLI already waited for the run to finish, so the thread is normally
        terminal by the time this host-side pass runs; a short poll absorbs the
        few seconds usage aggregation can lag.
        """
        url = (
            f"{platform_url}/api/v1/workforce/agents/{agent_id}"
            f"/threads/{thread_id}"
        )
        deadline = time.monotonic() + 60.0
        last: dict | None = None
        while True:
            try:
                request = urllib.request.Request(
                    url,
                    headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    last = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                logger.warning("octavus_cli: thread read error: %s", exc)
                if time.monotonic() >= deadline:
                    return last
                time.sleep(5.0)
                continue
            status = str((last or {}).get("status") or "")
            if status in _TERMINAL_STATUSES and (last or {}).get("usage"):
                return last
            if time.monotonic() >= deadline:
                return last
            time.sleep(5.0)

    @staticmethod
    def _map_messages(messages: list[dict], builder: TrajectoryBuilder) -> None:
        """Map consumer UIMessages to trajectory steps (best-effort).

        The instruction ``user`` step is already seeded by the framework, so only
        assistant turns are mapped: text/reasoning -> an ``agent`` step (with any
        tool calls), and each tool call's result -> an ``environment`` step.
        """
        for message in messages:
            if message.get("role") != "assistant":
                continue
            texts: list[str] = []
            reasonings: list[str] = []
            tool_calls: list[ToolCall] = []
            results: list[ToolResult] = []
            for part in message.get("parts") or []:
                ptype = part.get("type")
                if ptype == "text":
                    texts.append(part.get("text") or "")
                elif ptype == "reasoning":
                    reasonings.append(part.get("text") or "")
                elif ptype == "tool-call":
                    call_id = part.get("toolCallId") or ""
                    tool_calls.append(ToolCall(
                        id=call_id,
                        name=part.get("toolName") or "",
                        arguments=part.get("args") or {},
                    ))
                    payload = part.get("result")
                    if payload is None:
                        payload = part.get("error")
                    if payload is not None:
                        content = payload if isinstance(payload, str) else json.dumps(payload)
                        results.append(ToolResult(
                            tool_call_id=call_id,
                            content=[ContentPart(type="text", text=content)],
                            is_error=part.get("status") == "error" or bool(part.get("error")),
                        ))
            builder.add_step(
                source="agent",
                message="\n".join(t for t in texts if t) or None,
                reasoning="\n".join(r for r in reasonings if r) or None,
                tool_calls=tool_calls or None,
            )
            if results:
                builder.add_step(source="environment", observation=Observation(results=results))
