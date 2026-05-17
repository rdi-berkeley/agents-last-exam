"""ALE Runtime abstraction — where + how a deployer's code executes.

Three runtimes, three executors. Deployer code is **identical** across
all three; the runtime is the framework's execution-substrate decision,
not an agent author's concern.

The deployer's contract (in :mod:`ale.agents.base`):

    class BaseAgentDeployer:
        supported_runtimes: ClassVar[frozenset[str]]   # e.g. {"vm"} or {"local", "docker"}

        def __init__(self, runtime: AgentRuntime): ...
        async def install(self) -> None: ...
        async def launch(self, prompt: str) -> AgentRunResult: ...
        @classmethod
        def parse_artifacts(cls, *, work_dir, config, run_result) -> Iterable[Step]: ...

The runtime (this module):

  - :class:`AgentRuntime`: passive context (work_dir, vm_endpoint, vm_os, config).
                           One ``make_vm_session()`` convenience helper.
  - :class:`LocalRuntime` / :class:`VmRuntime` / :class:`DockerRuntime`:
                           subclass-of dataclass-of-AgentRuntime with
                           per-runtime conventions (work_dir parent,
                           image-baked paths, etc.).

The executor (also this module):

  - :class:`Executor` ABC + :data:`EXECUTORS` registry.
  - :class:`LocalExecutor`: in-process construct + await.
  - :class:`VmExecutor`: scp ale subtree + cua python_exec bootstrap (Phase 3).
  - :class:`DockerExecutor`: docker run + bind mount + sentinel poll (Phase 4).
"""

from .base import AgentRuntime
from .executor import EXECUTORS, Executor, RuntimeKind
from .local import LocalRuntime
from .local_executor import LocalExecutor
from .vm import VmRuntime
from .vm_executor import VmExecutor

# DockerRuntime + DockerExecutor land in Phase 4.

__all__ = [
    "AgentRuntime",
    "LocalRuntime",
    "VmRuntime",
    "Executor",
    "LocalExecutor",
    "VmExecutor",
    "EXECUTORS",
    "RuntimeKind",
]
