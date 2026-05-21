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

* :class:`FetchingRemoteCliDeployer` — install() fetches the CLI into
  the substrate from a small DSL (``"npm:<pkg>@<ver>"`` /
  ``"pip:<pkg>"`` / ``"url:<url>"``), then probes it.
"""
from __future__ import annotations

import asyncio
import logging
import shlex
import time
from typing import ClassVar

from ...base_interface import BaseAgentDeployer

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


class FetchingRemoteCliDeployer(RemoteCliDeployer):
    """Specialization for CLIs fetched into the substrate at install time
    (no pre-baking in the image).

    Subclass declares :attr:`install_spec` — a small DSL that dispatches
    to a package manager or URL fetch:

    * ``"npm:<pkg>@<ver>"`` → ``npm install -g <pkg>@<ver>`` (substrate
      must have npm on PATH).
    * ``"pip:<pkg>"`` → ``pip install <pkg>`` (same caveat for pip).
    * ``"url:<url>"`` → ``runtime.fetch_url_to(url, runtime.cli_path(cli_name))``
      then ``chmod +x`` on Linux.

    After dispatch, install probes the binary at ``runtime.cli_path(cli_name)``
    to confirm it landed where expected. Subclasses override
    :meth:`_post_install` for agent-specific config files (mirrors the
    :class:`PrebakedRemoteCliDeployer` hook).

    Example::

        class CodexDeployer(FetchingRemoteCliDeployer):
            install_spec = "npm:@openai/codex@1.2.3"
            cli_name = "codex"

            async def _post_install(self): ...

    For more elaborate install shapes (multi-step builds, conditional
    fetches, registry auth), override :meth:`install` directly.
    """

    install_spec: ClassVar[str] = ""
    """One of ``"npm:<pkg>@<ver>"``, ``"pip:<pkg>"``, ``"url:<url>"``.
    Concrete subclass sets this; the base parses + dispatches."""

    cli_name: ClassVar[str] = ""
    """Tool name passed to ``runtime.cli_path`` after install completes."""

    version_probe_args: ClassVar[tuple[str, ...]] = ("--version",)

    fetch_timeout_s: ClassVar[float] = 600.0
    """Per-scheme install command timeout. Bump for large binaries."""

    async def install(self) -> None:
        if not self.install_spec:
            raise RuntimeError(
                f"{type(self).__name__}: install_spec class attribute must be set"
            )
        if not self.cli_name:
            raise RuntimeError(
                f"{type(self).__name__}: cli_name class attribute must be set"
            )

        scheme, _, payload = self.install_spec.partition(":")
        if not payload:
            raise ValueError(
                f"install_spec missing payload after scheme: {self.install_spec!r}"
            )

        if scheme == "npm":
            await self._run_install_cmd(
                f"npm install -g {shlex.quote(payload)}", scheme=scheme,
            )
        elif scheme == "pip":
            await self._run_install_cmd(
                f"pip install {shlex.quote(payload)}", scheme=scheme,
            )
        elif scheme == "url":
            target = self.runtime.cli_path(self.cli_name)
            await self.runtime.fetch_url_to(payload, target)
            if self.runtime.vm_os == "linux":
                await self.runtime.run_command(
                    f"chmod +x {shlex.quote(target)}", timeout=30,
                )
        else:
            raise NotImplementedError(
                f"{type(self).__name__}: install_spec scheme {scheme!r} "
                f"not supported — known schemes: npm, pip, url. Override "
                f"install() directly for custom dispatch."
            )

        # Confirm the binary landed at the expected substrate path.
        cli_path = self.runtime.cli_path(self.cli_name)
        version_text = await self._probe_cli(cli_path, self.version_probe_args)
        logger.info(
            "%s: %s installed via %s — %s",
            type(self).__name__, self.cli_name, scheme, version_text,
        )

        await self.runtime.mkdir(self.runtime.work_dir)
        await self._post_install()

    async def _run_install_cmd(self, cmd: str, *, scheme: str) -> None:
        result = await self.runtime.run_command(cmd, timeout=self.fetch_timeout_s)
        if result.returncode != 0:
            raise RuntimeError(
                f"{type(self).__name__}: {scheme} install failed rc={result.returncode}: "
                f"{(result.stderr or result.stdout or '')[:400]}"
            )

    async def _post_install(self) -> None:
        """Hook — agent-specific config writes after the CLI is on disk.

        Default: no-op. Override to e.g. write MCP config files into
        ``runtime.work_dir`` (mirrors :class:`PrebakedRemoteCliDeployer`).
        """
        return None
