"""Compute-environment contracts: pre-provision spec, post-provision handle,
and the Provider ABC that turns one into the other.

The contracts here are deliberately substrate-neutral. Today's only
backend is GCE VMs, but the abstraction covers any cua-server-speaking
target (containers, pods, bare-metal hosts) — nothing here mentions
``VM`` except in the field names that describe machine sizing (vcpus,
memory_gb), which are the standard knobs even when the substrate is a
container.

Lifecycle:

  EnvSpec  ── Provider.acquire ──▶  EnvHandle
                                       │
                                       │  (used to address the env via
                                       │   HTTP helpers in environments/
                                       │   remote.py, and via cua sessions)
                                       │
                                       ▼
                                 Provider.release

(There used to be a separate ``RemoteVMConfig`` class that the HTTP
helpers consumed. It was a near-duplicate of ``VMHandle`` with two
piggy-back fields (run_id / task_id) used by exactly one optional
feature; folded into ``EnvHandle`` and ``VMHandle`` renamed for
substrate neutrality.)
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal


OS = Literal["linux", "windows"]
ReleaseMode = Literal["delete", "stop", "keep"]


@dataclass(frozen=True)
class EnvSpec:
    """Pre-provision: what kind of compute env the task wants.

    Built from ``task_card.json``. Handed to ``Provider.acquire``. The
    machine-sizing fields (vcpus / memory_gb / disk_gb / gpu) are
    expressed in the GCE vocabulary today but mean the same thing
    on any IaaS or container substrate.

    Identity fields (``task_id`` / ``harness`` / ``model_tag``) are
    optional and only flow into the env's name + hash seed so a
    leftover env is greppable.
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
class EnvHandle:
    """Post-provision: a reference to a live compute env.

    Returned by ``Provider.acquire`` and carried through the rest of the
    lifecycle — into BaseRuntime (as ``runtime.env_handle``), into HTTP
    helpers in environments/remote.py (as ``env``), into the cua session
    constructor in BaseRuntime.make_session.

    ``endpoint`` is the cua-server URL. ``os`` is the env's OS (linux /
    windows) — used both to pick HTTP request shapes (PowerShell vs bash)
    and to dispatch substrate-aware code in deployers. ``metadata`` is
    provider-defined: the GCloud provider stuffs capacity profile / VM
    type details here; tests use it for assertions.
    """

    id: str
    endpoint: str
    os: OS
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_linux(self) -> bool:
        return self.os == "linux"


class Provider(abc.ABC):
    """ABC for compute-env lifecycle. The framework consumes Provider;
    nothing else."""

    @abc.abstractmethod
    async def acquire(
        self,
        spec: EnvSpec,
        *,
        exclude_profiles: set[str] | None = None,
    ) -> EnvHandle:
        """Allocate a compute env matching ``spec``. May block on cold
        boot.

        ``exclude_profiles`` lets the lifecycle ask the provider to skip
        capacity profiles it knows are bad — used by the mount-fallback
        retry where a particular profile boots fine but the data disk
        fails to mount.
        """

    @abc.abstractmethod
    async def release(self, env: EnvHandle, *, mode: ReleaseMode = "delete") -> None:
        """Release the env. ``mode``: ``delete`` (default), ``stop``, or
        ``keep``."""

    @abc.abstractmethod
    def open_session(self, env: EnvHandle) -> Any:
        """Return a cua-bench DesktopSession talking to ``env``."""

    # ------------------------------------------------------------------ optional

    async def heartbeat(self, env: EnvHandle) -> None:
        """Send a keep-alive. Default: no-op."""
        return None

    async def cancel_external(self, env: EnvHandle) -> None:
        """Tell the provider's backend to stop the task on ``env``."""
        return None
