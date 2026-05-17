"""LocalExecutor — runs the deployer in this Python process.

Minimal. Constructs the deployer with the given runtime, awaits install
then launch, returns the result. No gather is needed — lifecycle creates
work_dir at ``<run_dir>/origin_log/<agent>/`` upfront, so the deployer
writes directly to the final destination.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .executor import EXECUTORS, Executor

if TYPE_CHECKING:
    from ale.agents.base import AgentRunResult, BaseAgentDeployer

    from .base import AgentRuntime

logger = logging.getLogger(__name__)


class LocalExecutor(Executor):
    """In-process executor."""

    kind = "local"

    async def run_deployer(
        self,
        *,
        deployer_cls: type["BaseAgentDeployer"],
        runtime: "AgentRuntime",
        prompt: str,
        timeout_s: float,
    ) -> "AgentRunResult":
        deployer = deployer_cls(runtime)
        logger.info(
            "local: %s.install (work_dir=%s)",
            deployer_cls.__name__, runtime.work_dir,
        )
        await deployer.install()
        logger.info(
            "local: %s.launch (timeout=%.0fs)",
            deployer_cls.__name__, timeout_s,
        )
        result = await deployer.launch(prompt)
        logger.info(
            "local: %s.launch returned status=%s",
            deployer_cls.__name__, result.status,
        )
        return result


EXECUTORS["local"] = LocalExecutor()
