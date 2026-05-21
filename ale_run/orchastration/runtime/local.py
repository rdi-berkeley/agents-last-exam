"""LocalRuntime — deployer runs in the framework's Python process.

The deployer code is in-process Python (think AleClaw's OpenClaw harness):
``run_command`` shells out via :mod:`asyncio.subprocess`; ``write_file`` /
``read_file`` are direct filesystem ops. ``work_dir`` is a host path.

The eval VM is a SEPARATE remote machine from ``self.vm_endpoint`` — the
deployer drives it via :meth:`make_vm_session` (host-side cua session).
``run_command`` etc. do NOT touch the eval VM; they target the framework
host.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable

from .base import BaseRuntime

logger = logging.getLogger(__name__)


@dataclass
class LocalRuntime(BaseRuntime):
    """In-process host runtime — deployer is framework-host Python.

    ``work_dir`` is a host filesystem path. ``host_artifacts_dir`` equals
    ``work_dir`` since the deployer writes directly to host-visible
    locations (no gather step needed).
    """

    kind: ClassVar[str] = "local"

    def _is_linux(self) -> bool:
        # Local substrate's shell is the host's, not the eval VM's.
        # All current deployers target a POSIX dev host; if Windows-host
        # support is ever needed, branch on ``platform.system()`` here.
        import platform
        return platform.system() != "Windows"

    # ======================================================================
    # I/O primitives — operate on the framework host's filesystem.
    # ======================================================================

    async def run_command(
        self, command: str, *, timeout: float = 60,
    ) -> subprocess.CompletedProcess:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return subprocess.CompletedProcess(
                args=command, returncode=-1, stdout="", stderr="timeout",
            )
        return subprocess.CompletedProcess(
            args=command,
            returncode=proc.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    async def write_file(self, path: str, content: str | bytes) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            await asyncio.to_thread(p.write_bytes, content)
        else:
            await asyncio.to_thread(p.write_text, content, encoding="utf-8")

    async def read_file(self, path: str) -> bytes:
        p = Path(path)
        if not p.exists():
            return b""
        return await asyncio.to_thread(p.read_bytes)

    async def exists(self, path: str) -> bool:
        return await asyncio.to_thread(Path(path).exists)

    async def mkdir(self, path: str) -> None:
        await asyncio.to_thread(
            lambda: Path(path).mkdir(parents=True, exist_ok=True),
        )

    async def rm(self, paths: Iterable[str]) -> None:
        def _rm() -> None:
            for p in paths:
                pth = Path(p)
                if pth.is_dir() and not pth.is_symlink():
                    shutil.rmtree(pth, ignore_errors=True)
                else:
                    try:
                        pth.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as e:
                        logger.debug("rm %s: %s", p, e)
        await asyncio.to_thread(_rm)

    # ======================================================================
    # Host conventions — minimal. The framework's pyproject and the dev
    # venv supply binaries; deployers usually don't need ``cli_path``.
    # ======================================================================

    def cli_path(self, name: str) -> str:
        """Resolve ``name`` to an absolute path on the host using $PATH.
        Falls back to the bare name so the OS resolves at exec time."""
        found = shutil.which(name)
        return found or name
