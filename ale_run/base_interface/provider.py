"""Provider ABC: VM lifecycle + session opening.

A Provider is the single object that knows where VMs come from. The
in-VM RPC surface is cua-bench's :class:`DesktopSession` Protocol — ale
doesn't re-invent it.

Concrete providers (``GcloudProvider``, ``StaticProvider``) live in
:mod:`ale_run.environments.providers`. This module defines only the
contract: the ABC + the dataclasses describing what gets passed in and
out.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal


OS = Literal["linux", "windows"]
ReleaseMode = Literal["delete", "stop", "keep"]


@dataclass(frozen=True)
class EnvSpec:
    """Declarative description of the VM the task wants. Built from ``task_card.json``.

    Identity fields (``task_id`` / ``harness`` / ``model_tag``) are
    optional and only flow into the VM's GCP name + hash seed so a
    leftover VM is greppable. Empty strings preserve the snapshot-only
    fallback when the caller doesn't know.
    """

    snapshot: str
    os: OS = "linux"
    vcpus: int = 4
    memory_gb: int = 16
    disk_gb: int = 200
    gpu: str | None = None
    task_id: str = ""
    harness: str = ""
    model_tag: str = ""


@dataclass
class VMHandle:
    """Reference to an acquired VM. ``endpoint`` and ``metadata`` are provider-defined."""

    id: str
    endpoint: str
    os: OS
    metadata: dict[str, Any] = field(default_factory=dict)


class Provider(abc.ABC):
    """ABC for VM lifecycle. The framework consumes Provider; nothing else."""

    @abc.abstractmethod
    async def acquire(
        self,
        spec: EnvSpec,
        *,
        exclude_profiles: set[str] | None = None,
    ) -> VMHandle:
        """Allocate a VM matching ``spec``. May block on cold boot.

        ``exclude_profiles`` lets the lifecycle ask the provider to skip
        capacity profiles it knows are bad — used by the mount-fallback
        retry where a particular capacity profile boots fine but the
        data disk fails to mount.
        """

    @abc.abstractmethod
    async def release(self, vm: VMHandle, *, mode: ReleaseMode = "delete") -> None:
        """Release the VM. ``mode``: ``delete`` (default), ``stop``, or ``keep``."""

    @abc.abstractmethod
    def open_session(self, vm: VMHandle) -> Any:
        """Return a cua-bench DesktopSession talking to ``vm``.

        For real providers, this constructs cua-bench's ``RemoteDesktopSession``
        pointing at ``vm.endpoint``. For stubs, returns whatever duck-typed
        object satisfies the parts of DesktopSession that tasks actually use.
        """

    # ------------------------------------------------------------------ optional

    async def heartbeat(self, vm: VMHandle) -> None:
        """Send a keep-alive. Default: no-op."""
        return None

    async def cancel_external(self, vm: VMHandle) -> None:
        """Tell the provider's backend to stop the task on ``vm``."""
        return None
