"""Executor — substrate adapter that hosts the deployer code and exposes
uniform I/O.

Three concrete subclasses:

* :class:`SandboxExecutor` — deployer code runs on the framework host; all
  substrate I/O is dispatched to a remote cua-server via HTTP RPC.
* :class:`LocalExecutor` — deployer + I/O both run in this Python
  process. Used by host-side harness deployers (AleClaw).
* :class:`DockerExecutor` — deployer runs on the framework host; I/O is
  dispatched into a container via ``docker exec``. (Shell-only stub.)

Each Executor is the per-unit context the deployer reads (work_dir,
sandbox, env, config) AND the I/O + process-lifecycle surface it
acts through. The lifecycle constructs one Executor per run and
threads it into the deployer's ``__init__``.

Note: "Executor" here means **where the deployer's substrate-touching
calls land**, NOT the OpenEnv ``Environment`` (the task world the
agent acts on, i.e. the cua-server VM with reset/step semantics). The
two coincide physically for ``vm`` mode but are conceptually distinct.

Concrete deployers should depend on :class:`BaseExecutor` (the API
surface) rather than a specific subclass.
"""
from __future__ import annotations

from ..base_interface import BaseExecutor
from .docker import DockerExecutor
from .local import LocalExecutor
from .sandbox import SandboxExecutor

# yaml ``executor: <type>`` → concrete Executor class.
EXECUTOR_REGISTRY: dict[str, type[BaseExecutor]] = {
    SandboxExecutor.type: SandboxExecutor,
    LocalExecutor.type: LocalExecutor,
    DockerExecutor.type: DockerExecutor,
}

__all__ = [
    "BaseExecutor",
    "DockerExecutor",
    "EXECUTOR_REGISTRY",
    "LocalExecutor",
    "SandboxExecutor",
]
