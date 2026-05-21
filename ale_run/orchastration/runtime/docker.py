"""DockerRuntime — deployer runs inside a host docker container (shell).

Same shape as :class:`LocalRuntime` but I/O dispatches through ``docker
exec``; ``work_dir`` is a container-local path (typically ``/work``) bind-
mounted to ``host_artifacts_dir`` so artifacts flow back without an
explicit gather step.

The container provides process / fs / env isolation for in-process
harness deployers (AleClaw-style) that don't want to share state with
the framework host. The container shares the host's network
(``--network host``) so it can reach the eval VM's cua-server on its
public IP via :meth:`make_vm_session`.

**This is a shell.** Concrete container lifecycle (``docker run`` /
``exec`` / ``rm``) is left for the first agent that actually needs it.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import ClassVar, Iterable

from .base import BaseRuntime

logger = logging.getLogger(__name__)


@dataclass
class DockerRuntime(BaseRuntime):
    """In-container host runtime.

    Carries the container's image tag (set by lifecycle when starting the
    container) and the bind-mount mapping. The container's ID is not
    modelled at this stage — the first concrete implementation will add
    it as a field set after ``docker run`` returns.
    """

    image: str = ""
    """Docker image tag the deployer's container is built from. Set by
    lifecycle from yaml ``runtime: docker; image: <tag>`` or per-deployer
    default."""

    kind: ClassVar[str] = "docker"

    def _is_linux(self) -> bool:
        # Containers we run are linux-based; tracks the container's shell,
        # not the eval VM.
        return True

    # ======================================================================
    # I/O primitives — NOT IMPLEMENTED. Shell only.
    # ======================================================================

    async def run_command(
        self, command: str, *, timeout: float = 60,
    ) -> subprocess.CompletedProcess:
        raise NotImplementedError(
            "DockerRuntime.run_command: container dispatch not yet wired. "
            "Implement via ``docker exec <container_id> sh -c '<command>'`` "
            "when the first agent needs the docker runtime."
        )

    async def write_file(self, path: str, content: str | bytes) -> None:
        raise NotImplementedError(
            "DockerRuntime.write_file: use ``docker cp`` from a host tmp "
            "file, or shell out to ``docker exec ... tee``."
        )

    async def read_file(self, path: str) -> bytes:
        raise NotImplementedError("DockerRuntime.read_file: see write_file.")

    async def exists(self, path: str) -> bool:
        raise NotImplementedError("DockerRuntime.exists")

    async def mkdir(self, path: str) -> None:
        raise NotImplementedError("DockerRuntime.mkdir")

    async def rm(self, paths: Iterable[str]) -> None:
        raise NotImplementedError("DockerRuntime.rm")

    def cli_path(self, name: str) -> str:
        # Containers are built with deployer-specific image conventions;
        # subclassing this runtime per agent isn't expected. Default to
        # bare name so the container's $PATH resolves at exec time.
        return name
