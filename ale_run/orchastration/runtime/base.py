"""BaseRuntime — substrate adapter that exposes a uniform I/O surface.

Three concrete subclasses: :class:`VmRuntime` (deployer code runs on host,
drives a remote cua-server VM), :class:`LocalRuntime` (deployer runs
in this Python process), :class:`DockerRuntime` (deployer runs in a host
docker container — shell only at this point).

The runtime is BOTH the "where" (data: endpoint, work_dir, vm_os, env vars,
config) AND the dispatcher (behaviour: ``install_deployer`` /
``launch_deployer``). The earlier ``VmExecutor`` indirection collapsed
into the runtime now that the deployer is always host-side and there's
no per-substrate code-placement work left for an executor to do.

Deployers reach into the substrate ONLY through this API surface:

  - :meth:`run_command` / :meth:`write_file` / :meth:`read_file` /
    :meth:`exists` / :meth:`mkdir` / :meth:`rm`
  - :meth:`fetch_url_to` (default uses curl on the substrate)
  - :meth:`make_vm_session` (open a cua DesktopSession against the
    eval VM — used by host-side harness deployers like AleClaw)

Substrate-specific path conventions (cli_path, node_exe, ...) live on
the concrete subclass.
"""
from __future__ import annotations

import abc
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable

if TYPE_CHECKING:
    from ...agents.base import AgentRunResult, BaseAgentConfig, BaseAgentDeployer


@dataclass
class BaseRuntime(abc.ABC):
    """Universal data shape + I/O contract.

    Constructed per unit by the lifecycle. The deployer reads
    ``self.runtime.<field>`` and calls ``self.runtime.<io_method>(...)``;
    it never imports substrate-specific transport directly.
    """

    config: "BaseAgentConfig"
    """The deployer's resolved config — read via ``self.runtime.config``
    or the convenience alias ``self.config`` on the deployer."""

    work_dir: str
    """Substrate-native scratch dir the deployer owns for this run.
    VM-side path for VmRuntime (may be Windows), host path for Local /
    Docker. Always a string — wrap in :class:`Path` on the host side
    when needed (``Path(self.runtime.work_dir)``)."""

    host_artifacts_dir: Path
    """Host-side path where artifacts end up after the lifecycle's gather
    step (or directly, on host-visible substrates). What ``parse_artifacts``
    reads from. For Local/Docker this is the same as ``work_dir`` (host
    bind-mount in the docker case); for Vm it's a separate directory the
    lifecycle gathers into."""

    vm_endpoint: str
    """cua-server URL of the eval VM, e.g. ``http://...:5000``. Always
    set — every benchmark target is a VM. Local/Docker deployers reach
    it via :meth:`make_vm_session`; Vm deployers reach it via the I/O
    methods on this runtime."""

    vm_os: str
    """``"linux"`` or ``"windows"`` — the eval VM's OS. Branched on by
    OS-aware deployer helpers (script generation, kill semantics, ...)."""

    env: dict[str, str] = field(default_factory=dict)
    """Env vars the framework wants injected into the agent process
    (api keys, base URLs). Deployers fold these into the launch shell."""

    kind: ClassVar[str] = ""
    """Subclass-supplied. Matches yaml ``runtime: <kind>`` values."""

    # ======================================================================
    # Dispatcher — was ``VmExecutor`` before
    # ======================================================================

    async def install_deployer(
        self, deployer_cls: "type[BaseAgentDeployer]",
    ) -> "BaseAgentDeployer":
        """Construct + install a deployer in this runtime. Returns the
        live instance so caller can ``await runtime.launch_deployer(d, prompt)``
        next. Split from launch because the lifecycle does work in between
        (incremental puller setup, rate-limit monitor)."""
        deployer = deployer_cls(self)
        await deployer.install()
        return deployer

    async def launch_deployer(
        self, deployer: "BaseAgentDeployer", prompt: str,
    ) -> "AgentRunResult":
        return await deployer.launch(prompt)

    # ======================================================================
    # I/O primitives — every deployer goes through these.
    #
    # Semantics: operate on the deployer-local substrate. For VmRuntime
    # that's the eval VM (via cua HTTP). For LocalRuntime that's the host
    # machine. For DockerRuntime that's the running container.
    # ======================================================================

    @abc.abstractmethod
    async def run_command(
        self, command: str, *, timeout: float = 60,
    ) -> subprocess.CompletedProcess:
        """Run a shell command in the substrate. Always returns a
        ``CompletedProcess`` (never raises on non-zero rc — caller checks
        ``.returncode``). On transport failure: rc=-1, stderr describes."""

    @abc.abstractmethod
    async def write_file(self, path: str, content: str | bytes) -> None:
        """Write ``content`` to ``path`` in the substrate. Overwrites.
        Binary-safe (base64 path on Windows VMs)."""

    @abc.abstractmethod
    async def read_file(self, path: str) -> bytes:
        """Read ``path`` as bytes. Empty bytes on missing file or transport
        error — caller checks :meth:`exists` first if the distinction
        matters."""

    @abc.abstractmethod
    async def exists(self, path: str) -> bool: ...

    @abc.abstractmethod
    async def mkdir(self, path: str) -> None:
        """Create ``path`` and any missing parents. Idempotent."""

    @abc.abstractmethod
    async def rm(self, paths: Iterable[str]) -> None:
        """Best-effort remove. Never raises on missing files."""

    async def read_text(self, path: str) -> str:
        return (await self.read_file(path)).decode("utf-8", errors="replace")

    # ======================================================================
    # Optional — used by ``DownloadedRemoteCliDeployer``
    # ======================================================================

    async def fetch_url_to(self, url: str, dst: str) -> None:
        """Fetch ``url`` onto the substrate at ``dst``. Default: shell out
        to curl (linux) / Invoke-WebRequest (windows). Subclasses override
        if the substrate has a more direct path."""
        if self._is_linux():
            cmd = f"curl -fsSL '{url}' -o '{dst}'"
        else:
            cmd = (
                'powershell -NoProfile -Command "'
                f"Invoke-WebRequest -Uri '{url}' -OutFile '{dst}' -UseBasicParsing"
                '"'
            )
        result = await self.run_command(cmd, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"fetch_url_to({url} -> {dst}) failed rc={result.returncode}: "
                f"{(result.stderr or '')[:300]}"
            )

    # ======================================================================
    # Eval VM session (cua DesktopSession) — used by host-side harness
    # deployers (AleClaw) that drive the eval VM via cua's high-level API
    # rather than raw HTTP.
    # ======================================================================

    async def make_vm_session(self) -> Any:
        """Open a fresh cua DesktopSession against :attr:`vm_endpoint`.

        VM-side and host-side deployers BOTH have an eval VM at the same
        endpoint — this method just wraps it in a cua session. Multiple
        concurrent sessions are safe (the cua-server is stateless for
        our usage).
        """
        from cua_bench.computers.remote import RemoteDesktopSession

        session = RemoteDesktopSession(
            api_url=self.vm_endpoint,
            os_type=self.vm_os,
            ephemeral=False,        # VM lifecycle is owned by ALEEnv
            headless=True,
        )
        await session.check_status()
        return session

    # ======================================================================
    # Substrate-image conventions (overridden per subclass)
    # ======================================================================

    def cli_path(self, name: str) -> str:
        """Absolute path of a baked-in CLI in this substrate (by tool
        name). Subclasses encode image conventions."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define cli_path"
        )

    @property
    def node_exe(self) -> str:
        raise NotImplementedError

    @property
    def mcp_server_dir(self) -> str:
        raise NotImplementedError

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _is_linux(self) -> bool:
        """True when the substrate-local shell is Linux. Default: based on
        vm_os. LocalRuntime / DockerRuntime override since they're host-OS
        bound, not eval-VM-bound."""
        return self.vm_os == "linux"
