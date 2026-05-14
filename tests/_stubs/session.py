"""StubDesktopSession: duck-typed minimal ``DesktopSession`` for tests.

Implements only the methods the demo tasks + AgenthleEnv pass-through
actions actually call. VM paths are mapped under a host temp directory.
Not a production replacement for cua-bench's RemoteDesktopSession.
"""
from __future__ import annotations

import asyncio
import dataclasses
import shutil
import time
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class _CmdResult:
    stdout: str
    stderr: str
    exit_code: int


class StubDesktopSession:
    """Maps ``/abs/path`` and ``C:\\path`` onto a host temp dir; subprocesses on host."""

    def __init__(self, host_root: Path):
        self._host_root = host_root
        self._host_root.mkdir(parents=True, exist_ok=True)

    # ---- path mapping ----
    def _local(self, vm_path: str) -> Path:
        s = str(vm_path).replace("\\", "/")
        if len(s) >= 2 and s[1] == ":":
            s = s[2:]                # drop drive letter
        s = s.lstrip("/")
        return self._host_root / s

    # ---- DesktopSession surface (only what we need) ----
    async def makedirs(self, path: str) -> None:
        self._local(path).mkdir(parents=True, exist_ok=True)

    async def exists(self, path: str) -> bool:
        return self._local(path).exists()

    async def read_file(self, path: str) -> str:
        p = self._local(path)
        if not p.exists():
            raise FileNotFoundError(str(path))
        return p.read_text(encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        p = self._local(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    async def read_bytes(self, path: str) -> bytes:
        p = self._local(path)
        if not p.exists():
            raise FileNotFoundError(str(path))
        return p.read_bytes()

    async def write_bytes(self, path: str, data: bytes) -> None:
        p = self._local(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    async def remove_file(self, path: str) -> None:
        p = self._local(path)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()

    async def list_dir(self, path: str) -> list[str]:
        return sorted(p.name for p in self._local(path).iterdir())

    async def run_command(
        self,
        cmd: str | list[str],
        *,
        check: bool = False,
        timeout: float | None = None,
        **kwargs,
    ) -> _CmdResult:
        if isinstance(cmd, list):
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._host_root),
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._host_root),
            )
        stdout, stderr = await proc.communicate()
        return _CmdResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            exit_code=proc.returncode if proc.returncode is not None else -1,
        )

    async def screenshot(self) -> bytes:
        # Minimal 1×1 PNG; tests don't introspect pixels.
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
            b"\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9c"
            b"c\xfc\xcf\xc0\xc0\xc0\x00\x00\x00\x04\x00\x01]\xcc\xdb\xd0\x00"
            b"\x00\x00\x00IEND\xaeB`\x82"
        )

    # ---- lifecycle stubs ----
    async def start(self) -> None: return None
    async def close(self) -> None: return None
    async def __aenter__(self): return self
    async def __aexit__(self, *exc): await self.close()

    # ---- access for tests ----
    @property
    def host_root(self) -> Path:
        return self._host_root
