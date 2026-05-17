"""Executor ABC — strategy for placing + running a deployer in a runtime.

One Executor per runtime kind (``local`` / ``vm`` / ``docker``). The
lifecycle picks the executor by spec.runtime, hands it the deployer
class + runtime context, and calls :meth:`run_deployer` to install +
launch the agent in the chosen substrate.

Gathering the agent's work_dir back to host (for parsing) is the
lifecycle's job (uses :class:`ArtifactMirror` for vm runtime; no-op
for local/docker since work_dir is already host-visible).
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ale.agents.base import AgentRunResult, BaseAgentDeployer

    from .base import AgentRuntime


RuntimeKind = Literal["vm", "local", "docker"]


class Executor(abc.ABC):
    """One executor per runtime kind. Stateless — instance per run unit."""

    kind: RuntimeKind

    @abc.abstractmethod
    async def run_deployer(
        self,
        *,
        deployer_cls: type["BaseAgentDeployer"],
        runtime: "AgentRuntime",
        prompt: str,
        timeout_s: float,
    ) -> "AgentRunResult":
        """Place the deployer in the runtime's substrate, await install + launch.

        - LocalExecutor: ``deployer = deployer_cls(runtime); await install();
                         await launch(prompt)`` in this process.
        - VmExecutor:    scp the agent subtree to the VM, then ``cua.python_exec``
                         a bootstrap that constructs deployer + awaits lifecycle.
        - DockerExecutor: ``docker run`` with bind mounts; container entrypoint
                          does the same construct + await.

        Returns the :class:`AgentRunResult` from the deployer's launch.
        """


# =============================================================================
# Registry — yaml `runtime: <key>` resolves to an Executor instance here
# =============================================================================

EXECUTORS: dict[str, Executor] = {}
"""Registry populated by each executor module's import-time
``EXECUTORS[<kind>] = <Executor>()``. Lifecycle does
``EXECUTORS[spec.runtime].run_deployer(...)``."""
