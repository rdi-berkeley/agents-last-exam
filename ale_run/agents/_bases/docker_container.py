"""DockerContainerDeployer — base for agents that own their own container
image (shell).

**Different** from :class:`InProcessHostDeployer` running on
:class:`DockerRuntime`: that one lets the framework's process be
docker-isolated; this one is for agents that bring a **bespoke image**
with their own Python deps, tools, and entrypoint — the deployer
manages the container lifecycle itself rather than relying on a
framework-managed runtime container.

Use cases (none today):

* A research agent that ships its own pinned PyTorch + CUDA setup,
  doesn't want to share the framework's venv.
* A CLI agent distributed only as a docker image, where the deployer's
  job is essentially ``docker run`` + log capture + exit-code mapping.

**Shell.** Concrete docker lifecycle (``pull`` / ``run`` / ``stop`` /
``rm`` / log streaming) is left for the first concrete subclass.
Until then, instantiating raises.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from ..base import AgentRunResult, BaseAgentDeployer

logger = logging.getLogger(__name__)


class DockerContainerDeployer(BaseAgentDeployer):
    """Base for agents that own a self-contained docker image.

    Subclass declares :attr:`image` and :attr:`entrypoint`; the base
    handles pull / run / log capture / exit-code → AgentRunResult.
    Today this is **shell** — both methods raise NotImplementedError.
    """

    supported_runtimes: ClassVar[frozenset[str]] = frozenset({"local"})
    """Runs on the framework host's docker daemon, not inside a VM."""

    image: ClassVar[str] = ""
    """``"<repo>:<tag>"`` of the agent's image. Subclass overrides."""

    entrypoint: ClassVar[tuple[str, ...]] = ()
    """``docker run --entrypoint``-equivalent argv. Empty = image default."""

    async def install(self) -> None:
        if not self.image:
            raise RuntimeError(
                f"{type(self).__name__}: image class attribute must be set"
            )
        raise NotImplementedError(
            "DockerContainerDeployer.install: ``docker pull`` + image probe "
            "will be implemented alongside the first concrete caller."
        )

    async def launch(self, prompt: str) -> AgentRunResult:
        raise NotImplementedError(
            "DockerContainerDeployer.launch: ``docker run`` + log streaming "
            "+ exit-code mapping will be implemented alongside the first "
            "concrete caller."
        )
