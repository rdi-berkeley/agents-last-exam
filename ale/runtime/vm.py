"""VmRuntime — deployer code runs INSIDE the eval VM.

work_dir is a VM-side path (e.g. ``/home/user/.ale/claude-code/<run_id>``).
The deployer constructs and runs inside the VM's Python process (shipped
via cua's ``python_exec`` by :class:`VmExecutor`). All ``subprocess.run`` /
``pathlib`` ops are VM-local.

Image-baked path conventions live here (defaults match
``agenthle-ubuntu-0505``). Per-image overrides go on yaml provider
config later if needed — for now in-tree defaults are enough.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import AgentRuntime, RuntimeKind


@dataclass
class VmRuntime(AgentRuntime):
    """In-VM runtime — agent lives in the test VM.

    The deployer's ``install`` / ``launch`` run inside the VM's Python
    process (cua-server's venv) via :func:`session.python_exec`. They
    use stdlib (``subprocess``, ``pathlib``, ``json``) to operate on the
    VM's local filesystem and processes — no ``session`` RPC needed
    (the deployer IS in the VM).

    The path defaults below match the baked ``agenthle-ubuntu-0505`` image:
      - Node 24.12 at ``/usr/local/bin/node`` (system install)
      - ``@anthropic-ai/claude-code@2.1.85`` at ``/usr/local/bin/claude``
      - cua-server's venv: ``/opt/cua-server/.venv`` (python 3.14, pydantic preinstalled)
      - cua MCP server (vendored at agenthle bake): ``/home/user/cua_mcp_server``
    """

    kind: ClassVar[RuntimeKind] = "vm"

    # ---- VM-image conventions ----
    node_exe: str = "/usr/local/bin/node"
    user_home: str = "/home/user"
    mcp_server_dir: str = "/home/user/cua_mcp_server"
    agent_bin_dir: str = "/usr/local/bin"
    work_dir_root: str = "/home/user/.ale"
    python_exe: str = "/opt/cua-server/.venv/bin/python3"
    """The cua-server venv's Python — what session.python_exec actually uses.
    Carries pydantic + cua deps used to ship our trajectory module."""

    # ---- helpers for deployers ----

    def cli_path(self, cli_name: str) -> str:
        """Absolute VM path to a CLI binary in agent_bin_dir."""
        return f"{self.agent_bin_dir}/{cli_name}"
