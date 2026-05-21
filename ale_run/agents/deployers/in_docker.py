"""InDockerDeployer — base for agents distributed AS a docker image.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from ...base_interface import AgentRunResult, BaseAgentDeployer

logger = logging.getLogger(__name__)


class InDockerDeployer(BaseAgentDeployer):
    """Base for agents whose distribution medium is a docker image."""

    supported_runtimes: ClassVar[frozenset[str]] = frozenset({"local"})
    """Runs on the framework host's docker daemon. (The agent's container
    is separate from any framework-isolation container in
    :class:`DockerRuntime`.)"""

    image: ClassVar[str] = ""
    """``"<repo>:<tag>"`` of the agent's image. Subclass overrides.
    Must already be present locally — see module docstring."""

    entrypoint: ClassVar[tuple[str, ...]] = ()
    """``docker run --entrypoint``-equivalent argv. Empty = image default."""

    async def install(self) -> None:
        if not self.image:
            raise RuntimeError(
                f"{type(self).__name__}: image class attribute must be set"
            )
        raise NotImplementedError(
            "InDockerDeployer.install: verify image present "
            "(``docker image inspect <image>``) will be wired alongside "
            "the first concrete caller."
        )

    async def launch(self, prompt: str) -> AgentRunResult:
        raise NotImplementedError(
            "InDockerDeployer.launch: ``docker run`` + log "
            "streaming + exit-code mapping will be wired alongside the "
            "first concrete caller."
        )
