"""DockerContainerDeployer — base for agents distributed AS a docker image.

**Distinct from** :class:`InProcessHostDeployer` running on
:class:`DockerRuntime`:

  - ``InProcessHostDeployer + DockerRuntime``: the **framework** runs in
    a container. Agent code is still our Python imports (e.g. AleClaw's
    OpenClaw harness). Same code as the local runtime; docker only
    provides isolation for the deployer process.

  - ``DockerContainerDeployer``: the **agent itself** is shipped as a
    docker image (e.g. a research agent with pinned PyTorch + CUDA that
    can't easily be installed as a Python package). The deployer's job
    is ``docker run <image>`` against the agent's prompt and capture the
    exit code + logs — there's no shared Python state.

The axis here is "what is the agent's unit of distribution":

  - in-process Python module  → :class:`InProcessHostDeployer`
  - VM-baked CLI               → :class:`PrebakedRemoteCliDeployer`
  - Fetched-into-VM CLI        → :class:`FetchingRemoteCliDeployer`
  - Pre-built docker image     → :class:`DockerContainerDeployer`  ← this

The image is provisioned **outside** the deployer (CI build / registry
pull on the operator's machine). ``install()`` confirms the image is
present locally; it does NOT do ``docker pull`` — that would conflate
fetching the agent with installing it (use
:class:`FetchingRemoteCliDeployer` if your agent is genuinely fetched
at install time; a self-shipped docker image is the operator's
infrastructure concern).

**Shell.** Concrete container lifecycle (``docker run`` + log capture +
exit-code mapping) is left for the first concrete subclass; until then
``install`` / ``launch`` raise.
"""
from __future__ import annotations

import logging
from typing import ClassVar

from ..base import AgentRunResult, BaseAgentDeployer

logger = logging.getLogger(__name__)


class DockerContainerDeployer(BaseAgentDeployer):
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
            "DockerContainerDeployer.install: verify image present "
            "(``docker image inspect <image>``) will be wired alongside "
            "the first concrete caller."
        )

    async def launch(self, prompt: str) -> AgentRunResult:
        raise NotImplementedError(
            "DockerContainerDeployer.launch: ``docker run`` + log "
            "streaming + exit-code mapping will be wired alongside the "
            "first concrete caller."
        )
