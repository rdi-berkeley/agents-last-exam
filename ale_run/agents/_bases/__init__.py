"""Shared deployer base classes — one per (substrate × install-strategy)
combination we actually see. Concrete agents under
``ale_run.agents.<name>`` subclass the matching base to inherit the
install / spawn / poll boilerplate.

Four bases:

* :class:`PrebakedRemoteCliDeployer` — CLI baked into the VM image
  (e.g. ClaudeCode). Implemented.
* :class:`DownloadedRemoteCliDeployer` — CLI fetched into the VM at
  install time. **Shell.**
* :class:`InProcessHostDeployer` — Python harness running in the
  framework process (e.g. AleClaw on ``local`` or ``docker`` runtime).
  Implemented.
* :class:`DockerContainerDeployer` — agent owns its own image. **Shell.**

Plus :class:`RemoteCliDeployer` (the common parent of the two remote-CLI
bases) for tests that want to assert shared behaviour.
"""
from __future__ import annotations

from .docker_container import DockerContainerDeployer
from .in_process import InProcessHostDeployer
from .remote_cli import (
    DownloadedRemoteCliDeployer,
    PrebakedRemoteCliDeployer,
    RemoteCliDeployer,
)

__all__ = [
    "DockerContainerDeployer",
    "DownloadedRemoteCliDeployer",
    "InProcessHostDeployer",
    "PrebakedRemoteCliDeployer",
    "RemoteCliDeployer",
]
