"""StubProvider: in-process Provider for tests. Backs each VM with a temp dir."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ale.core.provider import EnvSpec, Provider, ReleaseMode, VMHandle

from .session import StubDesktopSession


class StubProvider(Provider):
    """Returns a temp dir per ``acquire`` and a :class:`StubDesktopSession` per ``open_session``."""

    def __init__(self, *, root: Path | None = None):
        self._fixed_root = root
        self._sessions: dict[str, StubDesktopSession] = {}

    async def acquire(self, spec: EnvSpec) -> VMHandle:
        if self._fixed_root is not None:
            self._fixed_root.mkdir(parents=True, exist_ok=True)
            host_root = self._fixed_root
        else:
            host_root = Path(tempfile.mkdtemp(prefix="ale-stub-"))
        return VMHandle(
            id=host_root.name,
            endpoint=str(host_root),
            os=spec.os,
            metadata={"backend": "stub"},
        )

    async def release(self, vm: VMHandle, *, mode: ReleaseMode = "delete") -> None:
        self._sessions.pop(vm.id, None)
        if mode == "delete" and self._fixed_root is None:
            shutil.rmtree(vm.endpoint, ignore_errors=True)

    def open_session(self, vm: VMHandle):
        sess = self._sessions.get(vm.id)
        if sess is None:
            sess = StubDesktopSession(Path(vm.endpoint))
            self._sessions[vm.id] = sess
        return sess
