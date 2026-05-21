"""VmRuntime — deployer runs on host, drives a remote cua-server VM.

All I/O primitives go through HTTP against cua-server's ``/cmd`` endpoint
(see :mod:`ale_run.environments.remote`). The earlier ``VmExecutor`` is
collapsed in — the lifecycle now just constructs a VmRuntime and calls
``install_deployer`` / ``launch_deployer`` on it directly.

We deliberately avoid the cua-bench ``DesktopSession`` for raw I/O: its
WebSocket transport can wedge mid-run (long agent loops, idle timeouts,
firewall NAT churn) and recovering a dropped WS is more work than
re-establishing per-request HTTP/SSE. Host-side harness deployers
(AleClaw style) that DO want a session can still call
:meth:`make_vm_session` on this runtime.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import shlex
import subprocess
from dataclasses import dataclass
from typing import ClassVar, Iterable

from ..remote import (
    RemoteVMConfig,
    download_file,
    run_remote,
    upload_binary_file,
    upload_file,
)
from .base import BaseRuntime

logger = logging.getLogger(__name__)


@dataclass
class VmRuntime(BaseRuntime):
    """Substrate adapter for a remote cua-server VM.

    The ``work_dir`` field holds the VM-side absolute path (Linux POSIX
    or Windows-style depending on :attr:`vm_os`). ``host_artifacts_dir``
    is the host-side mirror the lifecycle gathers into for parsing.
    """

    kind: ClassVar[str] = "vm"

    # --- internal cache ---

    def _vm_config(self) -> RemoteVMConfig:
        return RemoteVMConfig(server_url=self.vm_endpoint, os_type=self.vm_os)

    # ======================================================================
    # I/O primitives — wrap the HTTP helpers from environments.remote so
    # deployers don't import that module directly.
    # ======================================================================

    async def run_command(
        self, command: str, *, timeout: float = 60,
    ) -> subprocess.CompletedProcess:
        return await asyncio.to_thread(
            run_remote, self._vm_config(), command, timeout,
        )

    async def write_file(self, path: str, content: str | bytes) -> None:
        if isinstance(content, bytes):
            # No straight binary path in remote.py for in-memory bytes;
            # upload via base64 + decode on the VM, mirroring
            # ``upload_binary_file`` but with content already in memory.
            await asyncio.to_thread(self._write_binary_sync, path, content)
            return
        await asyncio.to_thread(
            upload_file, self._vm_config(), path, content,
        )

    def _write_binary_sync(self, path: str, content: bytes) -> None:
        vm_config = self._vm_config()
        encoded = base64.b64encode(content).decode("ascii")
        b64_path = f"{path}.b64"
        upload_file(vm_config, b64_path, encoded)
        if vm_config.is_linux:
            decode = (
                f"base64 -d {shlex.quote(b64_path)} > {shlex.quote(path)} && "
                f"rm -f {shlex.quote(b64_path)}"
            )
        else:
            decode = (
                'powershell -NoProfile -Command "'
                f"$b=[IO.File]::ReadAllText('{b64_path}').Trim();"
                f"[IO.File]::WriteAllBytes('{path}',[Convert]::FromBase64String($b));"
                f"Remove-Item -Path '{b64_path}' -Force"
                '"'
            )
        result = run_remote(vm_config, decode, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"write_file binary decode failed for {path}: "
                f"rc={result.returncode} stderr={(result.stderr or '')[:200]}"
            )

    async def read_file(self, path: str) -> bytes:
        import os
        import tempfile

        def _read() -> bytes:
            fd, tmp = tempfile.mkstemp(prefix="ale_dl_")
            os.close(fd)
            try:
                ok = download_file(self._vm_config(), path, tmp)
                if not ok:
                    return b""
                with open(tmp, "rb") as f:
                    return f.read()
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

        return await asyncio.to_thread(_read)

    async def exists(self, path: str) -> bool:
        import requests

        from ..remote import _read_first_sse_event, _cua_url

        def _check() -> bool:
            try:
                with requests.post(
                    f"{_cua_url(self._vm_config())}/cmd",
                    json={"command": "file_exists", "params": {"path": path}},
                    headers={"Content-Type": "application/json"},
                    timeout=15,
                    stream=True,
                ) as resp:
                    data = _read_first_sse_event(resp, read_timeout=15)
            except requests.RequestException:
                return False
            return bool(data and data.get("exists"))

        return await asyncio.to_thread(_check)

    async def mkdir(self, path: str) -> None:
        if self._is_linux():
            cmd = f"mkdir -p {shlex.quote(path)}"
        else:
            cmd = (
                'powershell -NoProfile -Command "'
                f"New-Item -ItemType Directory -Force -Path '{path}' | Out-Null"
                '"'
            )
        result = await self.run_command(cmd, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"mkdir {path} failed rc={result.returncode}: "
                f"{(result.stderr or '')[:200]}"
            )

    async def rm(self, paths: Iterable[str]) -> None:
        paths = list(paths)
        if not paths:
            return
        if self._is_linux():
            quoted = " ".join(shlex.quote(p) for p in paths)
            cmd = f"rm -rf {quoted}"
        else:
            inner = "; ".join(
                f"Remove-Item -Recurse -Force -ErrorAction SilentlyContinue '{p}'"
                for p in paths
            )
            cmd = f'powershell -NoProfile -Command "{inner}"'
        await self.run_command(cmd, timeout=30)

    async def upload_local_file(self, local_path: str, remote_path: str) -> None:
        """Convenience: upload a HOST-side file into the VM. Used by
        ``DownloadedRemoteCliDeployer`` and any agent that stages binaries
        from the framework host."""
        await asyncio.to_thread(
            upload_binary_file, self._vm_config(), local_path, remote_path,
        )

    # ======================================================================
    # Image conventions — match the published ``agenthle-ubuntu-0505`` /
    # ``agenthle-unified-v1`` images.
    # ======================================================================

    def cli_path(self, name: str) -> str:
        if self.vm_os == "linux":
            return f"/usr/local/bin/{name}"
        # Per-tool overrides for the unified Windows image (each CLI
        # under its tool-specific dir rather than a shared C:\Tools tree).
        windows_overrides = {
            "claude": r"C:\Users\User\.local\bin\claude.exe",
        }
        return windows_overrides.get(name, rf"C:\Tools\{name}.exe")

    @property
    def node_exe(self) -> str:
        if self.vm_os == "linux":
            return "/usr/local/bin/node"
        return r"C:\Users\User\node-v24.12.0-win-x64\node.exe"

    @property
    def mcp_server_dir(self) -> str:
        if self.vm_os == "linux":
            return "/home/user/cua_mcp_server"
        return r"C:\Users\User\cua_mcp_server"
