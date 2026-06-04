"""Filesystem backends for the read/write/edit tools.

Tools dispatch through a small registry that maps a string ``target`` (an
enum value today; a URI scheme tomorrow) to a backend object that knows how
to resolve paths and perform I/O on its surface.

Two backends today:
  - ``VMBackend``: wraps ``BaseComputerInterface`` RPCs to the Windows VM.
    Path policy: lexical prefix check against an optional VM workspace root
    (``_assert_within_workspace`` from ``tools_fs``).
  - ``HostBackend``: plain Python file I/O against a configured host root.
    Path policy: realpath-resolved prefix check that defeats symlink escapes
    (mirrors OpenClaw ``writeFileWithinRoot`` / ``openFileWithinRoot`` from
    ``openclaw/src/agents/pi-tools.read.ts``).

The registry's ``names()`` and ``describe()`` views feed the JSON-schema
``enum`` and the tool description text, so adding a backend updates both
the agent's vocabulary and the API-level constraint from one place.
"""

from __future__ import annotations

import base64
import logging
import os
import shlex
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ._paths import _assert_within_workspace

if TYPE_CHECKING:
    from computer.interface import BaseComputerInterface

    from .mcp_runtime import MCPRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------


class FilesystemBackend(ABC):
    """Surface-agnostic file I/O port for the read/write/edit tools."""

    name: str
    capabilities: frozenset[str]   # subset of {"read", "write", "edit"}
    description: str

    @abstractmethod
    def resolve(self, path: str) -> str:
        """Validate and normalize ``path`` against this backend's root.

        Returns the resolved path string the I/O methods should use.
        Raises ``ValueError`` if the path escapes the backend's allowed
        region.
        """

    @abstractmethod
    async def read_bytes(self, resolved_path: str) -> bytes: ...

    @abstractmethod
    async def write_text(
        self,
        resolved_path: str,
        content: str,
        *,
        append: bool,
    ) -> None: ...

    @abstractmethod
    async def create_dir(self, resolved_path: str) -> None: ...


# ---------------------------------------------------------------------------
# VMBackend — wraps the existing CUA SDK
# ---------------------------------------------------------------------------


class VMBackend(FilesystemBackend):
    """Routes I/O through ``BaseComputerInterface`` RPCs.

    Path policy: the existing lexical ``_assert_within_workspace`` check.
    The VM is the model's threat surface (the agent operates inside it),
    not the harness's — symlink-aware resolution would buy nothing here
    because the VM filesystem is owned by the agent's own actions.
    """

    name = "vm"
    capabilities = frozenset({"read", "write", "edit"})

    def __init__(
        self,
        interface: "BaseComputerInterface",
        workspace_root: Optional[str] = None,
    ) -> None:
        self.interface = interface
        self.workspace_root = workspace_root
        if workspace_root is None:
            logger.info("VMBackend: workspace_root is None — permissive path policy")
        if workspace_root:
            self.description = (
                f"Windows VM filesystem (paths resolve under {workspace_root})."
            )
        else:
            self.description = (
                "Windows VM filesystem (no workspace bound — permissive)."
            )

    def resolve(self, path: str) -> str:
        _assert_within_workspace(path, self.workspace_root)
        return path

    async def read_bytes(self, resolved_path: str) -> bytes:
        return await self.interface.read_bytes(resolved_path)

    async def write_text(
        self,
        resolved_path: str,
        content: str,
        *,
        append: bool,
    ) -> None:
        await self.interface.write_text(resolved_path, content, append=append)

    async def create_dir(self, resolved_path: str) -> None:
        await self.interface.create_dir(resolved_path)


# ---------------------------------------------------------------------------
# MCPBackend — same VM surface as VMBackend, routed through the vm MCP bridge
# ---------------------------------------------------------------------------


class MCPBackend(FilesystemBackend):
    """VM filesystem I/O routed through the ``vm_mcp_server`` bridge.

    Drop-in replacement for :class:`VMBackend`: registered under the same
    ``name = "vm"`` so the tool ``target`` enum, descriptions, and the agent's
    vocabulary are byte-identical — only the transport changes (``RemoteDesktop
    Session`` RPC → MCP stdio bridge → the same cua-server). Path policy is the
    existing lexical ``_assert_within_workspace`` check; the VM is the model's
    own surface, so symlink-aware resolution buys nothing here (same rationale as
    ``VMBackend``).

    The three substrate ops map onto vm-bridge primitives:
      - ``read_bytes``  → ``read_bytes`` (returns base64; decoded here).
      - ``write_text``  → ``write_text`` (overwrite) / ``write_bytes`` (append;
        the bridge's ``write_text`` overwrites only).
      - ``create_dir``  → ``run_command("mkdir -p …")`` (the bridge omits a mkdir
        primitive by design — it is reducible to ``run_command``).
    """

    name = "vm"
    capabilities = frozenset({"read", "write", "edit"})

    def __init__(
        self,
        runtime: "MCPRuntime",
        workspace_root: Optional[str] = None,
        os_type: Optional[str] = None,
    ) -> None:
        self.runtime = runtime
        self.workspace_root = workspace_root
        # The vm bridge has no mkdir primitive, so create_dir is synthesized as a
        # run_command — which is shell/OS-specific. ``os_type`` (from the session)
        # selects the right form; default to POSIX when unknown.
        self._is_windows = (os_type or "").lower().startswith("win")
        if workspace_root is None:
            logger.info("MCPBackend: workspace_root is None — permissive path policy")
        if workspace_root:
            self.description = (
                f"VM filesystem via the vm MCP bridge (paths resolve under {workspace_root})."
            )
        else:
            self.description = (
                "VM filesystem via the vm MCP bridge (no workspace bound — permissive)."
            )

    def resolve(self, path: str) -> str:
        _assert_within_workspace(path, self.workspace_root)
        return path

    async def read_bytes(self, resolved_path: str) -> bytes:
        # vm bridge read_bytes returns base64 of the (full, no-range) file body.
        from .mcp_runtime import result_text
        res = await self.runtime.call("vm", "read_bytes", {"path": resolved_path})
        return base64.b64decode(result_text(res))

    async def write_text(
        self,
        resolved_path: str,
        content: str,
        *,
        append: bool,
    ) -> None:
        if append:
            # write_text overwrites; append goes through write_bytes(append=true).
            b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
            await self.runtime.call(
                "vm", "write_bytes",
                {"path": resolved_path, "content_b64": b64, "append": True},
            )
        else:
            await self.runtime.call(
                "vm", "write_text",
                {"path": resolved_path, "content": content},
            )

    async def create_dir(self, resolved_path: str) -> None:
        # The bridge has no mkdir primitive (reducible to run_command, by design),
        # so synthesize one. cmd.exe and POSIX sh disagree on flags AND quoting:
        #   - Windows: cmd `mkdir` has no -p and creates intermediates by default;
        #     guard with `if not exist` so an existing dir isn't an error. cmd
        #     uses double quotes (it does not understand POSIX single-quoting).
        #   - POSIX: `mkdir -p` is idempotent; shlex.quote is the correct quoting.
        if self._is_windows:
            p = resolved_path.replace('"', "")  # drop stray quotes; cmd paths can't contain them
            command = f'if not exist "{p}" mkdir "{p}"'
        else:
            command = f"mkdir -p {shlex.quote(resolved_path)}"
        await self.runtime.call("vm", "run_command", {"command": command})


# ---------------------------------------------------------------------------
# HostBackend — plain Python I/O, realpath-checked
# ---------------------------------------------------------------------------


class HostBackend(FilesystemBackend):
    """Plain Python file I/O scoped to a host root directory.

    Path policy: realpath-resolved prefix check. After ``Path.resolve()``
    follows any symlinks, the candidate must equal the root or sit under
    ``root + os.sep``. This defeats the symlink-escape pattern that pure
    lexical checks miss (mirrors OpenClaw's safe-fs primitives in
    ``pi-tools.read.ts``).
    """

    name = "host"
    capabilities = frozenset({"read", "write", "edit"})

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(
                f"HostBackend root does not exist or is not a directory: {self.root}"
            )
        self.description = (
            f"Local host filesystem rooted at {self.root} "
            f"(read/write/edit any file under this directory)."
        )

    def resolve(self, path: str) -> str:
        if not path:
            raise ValueError("host: path must be non-empty")
        candidate_input = Path(path)
        if candidate_input.is_absolute():
            candidate = candidate_input.resolve()
        else:
            candidate = (self.root / candidate_input).resolve()

        root_str = str(self.root)
        cand_str = str(candidate)
        if cand_str != root_str and not cand_str.startswith(root_str + os.sep):
            raise ValueError(
                f"path '{path}' resolves outside host workspace ({self.root})"
            )
        return cand_str

    async def read_bytes(self, resolved_path: str) -> bytes:
        return await _to_thread(lambda: Path(resolved_path).read_bytes())

    async def write_text(
        self,
        resolved_path: str,
        content: str,
        *,
        append: bool,
    ) -> None:
        def _do_write() -> None:
            mode = "a" if append else "w"
            with open(resolved_path, mode, encoding="utf-8") as f:
                f.write(content)

        await _to_thread(_do_write)

    async def create_dir(self, resolved_path: str) -> None:
        await _to_thread(
            lambda: Path(resolved_path).mkdir(parents=True, exist_ok=True)
        )


async def _to_thread(fn):
    """Run a sync callable in a thread to keep the event loop responsive."""
    import asyncio
    return await asyncio.to_thread(fn)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class FilesystemRegistry:
    """Maps target names (today: enum values; tomorrow: URI schemes) to backends.

    Tool descriptions and JSON-schema enums are derived from this registry,
    so adding a backend updates both surfaces without manual edits.
    """

    def __init__(self) -> None:
        self._backends: dict[str, FilesystemBackend] = {}

    def register(self, backend: FilesystemBackend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"backend '{backend.name}' already registered")
        self._backends[backend.name] = backend

    def get(self, name: str) -> FilesystemBackend:
        if name not in self._backends:
            valid = ", ".join(sorted(self._backends)) or "<none>"
            raise ValueError(f"unknown target '{name}'; valid: {valid}")
        return self._backends[name]

    def names(self) -> list[str]:
        # Stable order: vm first if present, then alphabetical for the rest.
        ordered = []
        if "vm" in self._backends:
            ordered.append("vm")
        for n in sorted(self._backends):
            if n != "vm":
                ordered.append(n)
        return ordered

    def describe(self) -> str:
        """One-line summary suitable for tool descriptions."""
        if not self._backends:
            return "No filesystems registered."
        parts = [f"{n}: {self._backends[n].description}" for n in self.names()]
        return " ".join(parts)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._backends


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def detect_host_workspace_root(start: Path | str | None = None) -> Optional[Path]:
    """Walk up from ``start`` (or cwd) looking for a ``.git`` directory.

    Submodules use a ``.git`` *file*, not a directory, so this skips submodule
    boundaries and only stops at the top-level repo root. Returns ``None``
    when no ``.git`` directory is found in any ancestor.
    """
    cur = (Path(start) if start else Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        git_path = candidate / ".git"
        if git_path.is_dir():
            return candidate
    return None
