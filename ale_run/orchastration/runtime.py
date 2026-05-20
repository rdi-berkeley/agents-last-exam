"""AgentRuntime — passive context handed to deployers.

Deployers are **host-side**: ``install()`` / ``launch()`` run in the same
Python process that drives the orchestrator. They reach into the VM via
the attached :class:`cua_bench.DesktopSession` (``runtime.session.run_command``,
``runtime.session.write_file``, etc). Nothing on the deployer side is
shipped into the VM. Image-baked things still apply (claude CLI, node, MCP
server) — the deployer just *invokes* them via the session.

For substrates other than ``vm`` (not implemented in this pass) the same
``BaseAgentDeployer`` contract would receive a different runtime kind
(LocalRuntime, DockerRuntime, …) — currently the factory raises
``NotImplementedError`` for those.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class VmRuntime:
    """Passive context for host-side deployers driving a remote VM.

    Built fresh per unit. The framework hands it to
    :class:`BaseAgentDeployer`'s constructor; the deployer reads
    ``self.runtime.work_dir_vm`` etc. and uses ``self.runtime.session``
    (a ``cua_bench.DesktopSession``) for any VM-side I/O.

    Attributes:
        kind          Always ``"vm"`` in this pass; future runtime kinds
                      keep the same field for the deployer to branch on.
        vm_endpoint   ``http://<external_ip>:5000`` — the CUA server URL.
        vm_os         ``"linux"`` or ``"windows"``.
        work_dir_vm   Absolute path INSIDE the VM where the deployer
                      writes artifacts (``/home/user/.ale/<name>/<run_id>``
                      or ``C:\\Users\\User\\.ale\\<name>\\<run_id>``).
        work_dir_host Where the framework gathers ``work_dir_vm`` to after
                      the agent exits (``<run_dir>/origin_log/<name>/``).
        config        The deployer's resolved ``BaseAgentConfig``.
        env           Extra env vars the framework wants the agent to see
                      (api keys, base URLs, …). Inject these into the
                      command shell the deployer spawns on the VM.
        session       Open ``cua_bench.DesktopSession`` to the VM. Set by
                      :class:`ale_run.orchastration.vm_executor.VmExecutor`
                      right before ``install`` / ``launch`` is called.
    """

    kind: str
    vm_endpoint: str
    vm_os: str
    work_dir_vm: str
    work_dir_host: Path
    config: Any
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    session: Any = None  # cua_bench.DesktopSession; populated by VmExecutor

    @property
    def work_dir(self) -> Path:
        return Path(self.work_dir_vm)

    def cli_path(self, name: str) -> str:
        if self.vm_os == "linux":
            return f"/usr/local/bin/{name}"
        # Per-tool overrides for the published agenthle-unified-v1 image,
        # which installs each CLI under its tool-specific location rather
        # than a shared C:\Tools tree.
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
