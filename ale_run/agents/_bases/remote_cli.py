"""Base classes for deployers that drive a CLI baked into / downloaded onto
a remote substrate (VM or container).

Three classes:

* :class:`RemoteCliDeployer` — common helpers for any deployer that
  spawns a long-running CLI in the substrate and polls a done-marker
  for completion. OS-aware (Linux ``bash`` + ``setsid`` vs Windows
  PowerShell ``Start-Process``). Concrete agents subclass one of the
  two specializations below.

* :class:`PrebakedRemoteCliDeployer` — install() verifies a baked-in
  binary at ``runtime.cli_path(<name>)``. Used by ClaudeCode.

* :class:`DownloadedRemoteCliDeployer` — install() fetches the CLI into
  the substrate from a small DSL (``"npm:<pkg>@<ver>"`` /
  ``"pip:<pkg>"`` / ``"url:<url>"``). **Shell** — concrete dispatch is
  left for the first caller.
"""
from __future__ import annotations

import asyncio
import logging
import shlex
import time
from typing import ClassVar

from ..base import BaseAgentDeployer

logger = logging.getLogger(__name__)


class RemoteCliDeployer(BaseAgentDeployer):
    """Shared infrastructure for VM-side CLI agents.

    Provides:

    * :meth:`_probe_cli` — version-probe an absolute path, raise on failure.
    * :meth:`_spawn_detached` — upload a runner script + spawn detached
      with PID + done-marker. OS-branched internally.
    * :meth:`_poll_until_done` — wait on the done-marker on a 5s tick,
      with an absolute deadline; returns ``(exit_code, status, duration_s)``.
    * :meth:`_kill_pid` — TERM+KILL on Linux, Stop-Process on Windows.
    * :meth:`_join` — substrate-native path joining.

    Subclasses MUST implement ``parse_artifacts``; ``install`` and
    ``launch`` come from the two specializations below.
    """

    # Subclasses still declare their supported runtimes.
    supported_runtimes: ClassVar[frozenset[str]] = frozenset({"vm"})

    # ------------------------------------------------------------------ helpers

    def _join(self, *parts: str) -> str:
        """Substrate-native path join. Linux ``/`` vs Windows ``\\``."""
        sep = "/" if self.runtime.vm_os == "linux" else "\\"
        head = parts[0].rstrip("/\\")
        tail = sep.join(p.strip("/\\") for p in parts[1:])
        return f"{head}{sep}{tail}" if tail else head

    # ---- CLI probing -------------------------------------------------------

    async def _probe_cli(
        self, cli_path: str, version_args: tuple[str, ...] = ("--version",),
    ) -> str:
        """Run ``cli_path <version_args>`` in the substrate. Raise if the
        binary isn't present or exits non-zero. Return the trimmed stdout
        (useful for the install-ok log line).
        """
        if self.runtime.vm_os == "linux":
            args = " ".join(shlex.quote(a) for a in version_args)
            cmd = f"test -x {shlex.quote(cli_path)} && {shlex.quote(cli_path)} {args}"
        else:
            args = " ".join(version_args)
            cmd = (
                'powershell -NoProfile -Command "'
                f"if (Test-Path '{cli_path}') {{ & '{cli_path}' {args} }} "
                f"else {{ Write-Error 'cli not found at {cli_path}'; exit 1 }}"
                '"'
            )
        result = await self.runtime.run_command(cmd, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"CLI missing or unrunnable at {cli_path}: "
                f"stderr={(result.stderr or '').strip()[:300]}"
            )
        return (result.stdout or "").strip()

    # ---- detached spawn + poll -------------------------------------------

    async def _spawn_detached(
        self,
        *,
        runner_body: str,
        runner_script_path: str,
        pid_file: str,
        done_marker: str,
        reset_files: list[str] | None = None,
    ) -> None:
        """Upload ``runner_body`` as a script + spawn it detached.

        ``runner_body`` is the agent-specific shell/PS script. It MUST
        end by writing its exit code to ``done_marker`` — we don't wrap
        that, because the agent may need to redirect stdout/stderr in
        ways that wrappers would interfere with.

        ``reset_files`` (optional) is a list of paths to remove before
        spawn — typically the prior run's done.marker / pid / logs.
        """
        if reset_files:
            await self.runtime.rm(reset_files)
        await self.runtime.write_file(runner_script_path, runner_body)

        if self.runtime.vm_os == "linux":
            await self.runtime.run_command(
                f"chmod +x {shlex.quote(runner_script_path)}", timeout=15,
            )
            launcher = (
                "#!/bin/bash\n"
                f"setsid bash {shlex.quote(runner_script_path)} "
                "</dev/null >/dev/null 2>&1 &\n"
                "CHILD=$!\n"
                f"echo \"$CHILD\" > {shlex.quote(pid_file)}\n"
                "disown $CHILD 2>/dev/null || true\n"
            )
            launcher_path = runner_script_path + ".launch"
            await self.runtime.write_file(launcher_path, launcher)
            await self.runtime.run_command(
                f"chmod +x {shlex.quote(launcher_path)}", timeout=15,
            )
            result = await self.runtime.run_command(
                f"bash {shlex.quote(launcher_path)}", timeout=30,
            )
        else:
            spawn_cmd = (
                'powershell -NoProfile -Command "'
                f"$proc = Start-Process powershell "
                f"-ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','{runner_script_path}' "
                f"-WindowStyle Hidden -PassThru; "
                f"$proc.Id | Out-File -FilePath '{pid_file}' -Encoding ascii -NoNewline"
                '"'
            )
            result = await self.runtime.run_command(spawn_cmd, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"spawn detached failed rc={result.returncode}: "
                f"{(result.stderr or '').strip()[:300]}"
            )

    async def _read_pid(self, pid_file: str, max_wait_s: float = 4.5) -> int | None:
        """Read pid_file the launcher writes synchronously. Polls every
        300ms up to ``max_wait_s`` to absorb the tiny gap between
        Start-Process returning and the file being flushed."""
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            if await self.runtime.exists(pid_file):
                raw = (await self.runtime.read_text(pid_file)).strip()
                try:
                    return int(raw)
                except ValueError:
                    return None
            await asyncio.sleep(0.3)
        return None

    async def _kill_pid(self, pid: int) -> None:
        if self.runtime.vm_os == "linux":
            await self.runtime.run_command(f"kill -TERM {pid}", timeout=15)
            await asyncio.sleep(2)
            await self.runtime.run_command(f"kill -KILL {pid}", timeout=15)
        else:
            await self.runtime.run_command(
                f'powershell -NoProfile -Command "Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"',
                timeout=15,
            )

    async def _poll_until_done(
        self,
        *,
        done_marker: str,
        timeout_s: float,
        poll_interval_s: float = 5.0,
    ) -> tuple[int | None, str, float]:
        """Return ``(exit_code, status, duration_s)``.

        ``status`` is ``"completed"`` (rc == 0), ``"failed"`` (rc != 0)
        or ``"timeout"`` (deadline hit before marker appeared).
        Caller decides whether to kill the PID on timeout.
        """
        t0 = time.monotonic()
        deadline = t0 + timeout_s
        while True:
            if await self.runtime.exists(done_marker):
                raw = (await self.runtime.read_text(done_marker)).strip()
                try:
                    exit_code: int | None = int(raw) if raw else None
                except ValueError:
                    exit_code = None
                status = "completed" if exit_code == 0 else "failed"
                return exit_code, status, time.monotonic() - t0
            if time.monotonic() >= deadline:
                return None, "timeout", time.monotonic() - t0
            await asyncio.sleep(poll_interval_s)


class PrebakedRemoteCliDeployer(RemoteCliDeployer):
    """Specialization for image-baked CLI agents.

    The image carries the agent's CLI; install() just probes it and
    prepares ``work_dir``. Subclasses set :attr:`cli_name` (resolved via
    ``runtime.cli_path``) and override :meth:`_post_install` to write
    any agent-specific config (e.g. MCP server config).
    """

    cli_name: ClassVar[str] = ""
    """Tool name passed to ``runtime.cli_path`` (e.g. ``"claude"``)."""

    version_probe_args: ClassVar[tuple[str, ...]] = ("--version",)

    async def install(self) -> None:
        if not self.cli_name:
            raise RuntimeError(
                f"{type(self).__name__}: cli_name class attribute must be set"
            )
        cli_path = self.runtime.cli_path(self.cli_name)
        version_text = await self._probe_cli(cli_path, self.version_probe_args)
        logger.info(
            "%s: %s CLI ok — %s",
            type(self).__name__, self.cli_name, version_text,
        )
        await self.runtime.mkdir(self.runtime.work_dir)
        await self._post_install()

    async def _post_install(self) -> None:
        """Hook — write agent-specific config files into ``runtime.work_dir``.

        Default: no-op. ClaudeCode writes the MCP server config here.
        """
        return None


class DownloadedRemoteCliDeployer(RemoteCliDeployer):
    """Specialization for CLIs installed into the substrate at install
    time (no pre-baking in the image).

    **Shell.** The DSL parsing and per-scheme dispatch are deliberately
    not implemented — they'll be designed alongside the first concrete
    caller (likely an agent that ``npm i -g`` its CLI fresh into the
    VM, or pulls a binary tarball).

    Sketch of the intended shape::

        class CodexDeployer(DownloadedRemoteCliDeployer):
            install_spec = "npm:@openai/codex@1.2.3"
            cli_name = "codex"

            async def _post_install(self): ...
    """

    install_spec: ClassVar[str] = ""
    """One of ``"npm:<pkg>@<ver>"``, ``"pip:<pkg>"``, ``"url:<url>"``.
    Concrete subclass sets this; the base parses + dispatches."""

    cli_name: ClassVar[str] = ""
    """Tool name passed to ``runtime.cli_path`` after install completes."""

    async def install(self) -> None:
        if not self.install_spec:
            raise RuntimeError(
                f"{type(self).__name__}: install_spec class attribute must be set"
            )
        raise NotImplementedError(
            "DownloadedRemoteCliDeployer.install: per-scheme dispatch "
            "(npm / pip / url) will be wired alongside the first concrete "
            "subclass. Override install() directly in the meantime."
        )

    async def _post_install(self) -> None:
        return None
