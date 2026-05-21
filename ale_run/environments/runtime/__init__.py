"""Runtime substrate adapters — VmRuntime / LocalRuntime / DockerRuntime.

Each runtime is BOTH the per-unit context the deployer reads (work_dir,
vm_endpoint, vm_os, env, config) AND the dispatcher that drives the
deployer through ``install_deployer`` / ``launch_deployer``. The earlier
``VmExecutor`` indirection is gone — substrate-specific code lives on
the runtime subclass that owns it.

Concrete deployers should depend on :class:`BaseRuntime` (the API surface)
rather than a specific subclass, unless they need substrate-specific
helpers (e.g. ``VmRuntime.upload_local_file``).
"""
from __future__ import annotations

from ...base_interface import BaseRuntime
from .docker import DockerRuntime
from .local import LocalRuntime
from .vm import VmRuntime

# yaml ``runtime: <kind>`` → concrete class
RUNTIME_REGISTRY: dict[str, type[BaseRuntime]] = {
    VmRuntime.kind: VmRuntime,
    LocalRuntime.kind: LocalRuntime,
    DockerRuntime.kind: DockerRuntime,
}

__all__ = [
    "BaseRuntime",
    "DockerRuntime",
    "LocalRuntime",
    "RUNTIME_REGISTRY",
    "VmRuntime",
]
