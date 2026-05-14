"""StubInstalledAgentDeployer: pretends to be an in-VM agent for tests.

``solver(session)`` is the test-injected "agent behavior" — called inside
``launch``. The deployer appends one synthetic ``agent`` Step to the
trajectory in ``collect``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, ClassVar

from ale.agents.installed.base import (
    AgentRunResult,
    InstalledAgentConfig,
    InstalledAgentDeployer,
)
from ale.agents.trajectory import TrajectoryBuilder

SolverFn = Callable[[object], Awaitable[None]]


@dataclass
class StubInstalledAgentConfig(InstalledAgentConfig):
    name: ClassVar[str] = "stub-installed"
    model: str = "stub-model"


class StubInstalledAgentDeployer(InstalledAgentDeployer):
    def __init__(
        self,
        solver: SolverFn,
        *,
        config: StubInstalledAgentConfig | None = None,
    ):
        self._solver = solver
        self._cfg = config or StubInstalledAgentConfig()
        self.install_calls = 0
        self.launch_calls = 0
        self.collect_calls = 0

    @property
    def config(self) -> StubInstalledAgentConfig:
        return self._cfg

    @property
    def version(self) -> str | None:
        return "stub-0.1"

    async def install(self, session) -> None:
        self.install_calls += 1

    async def launch(self, session, *, prompt: str, timeout_s: float) -> AgentRunResult:
        self.launch_calls += 1
        await self._solver(session)
        return AgentRunResult(status="completed", exit_code=0, duration_s=0.01)

    async def collect(self, session, run: AgentRunResult, builder: TrajectoryBuilder) -> None:
        self.collect_calls += 1
        builder.add_step(
            source="agent",
            message="stub agent finished",
            extra={"stub": True, "exit_code": run.exit_code},
        )

    def work_dir(self, session) -> str | None:
        return None      # stub doesn't write anything on a "VM"
