"""Thin host-side dispatcher: attach the open CUA session to the runtime and
let the deployer drive the VM from the orchestrator process.

The earlier design serialized deployer code into the VM via
``cua.python_exec`` and required the ``ale_run`` package + its deps to be
pip-installed on the VM. That added a runtime staging step (zip upload +
unpack + pydantic pip install) and a sys.path hack to work around cua's
source-extractor hoisting ``from x import y`` to module level.

The current design moves deployers back to the host. Each deployer owns
its own HTTP shape against the CUA server via the session API
(``run_command`` / ``write_file`` / ``read_file`` / ``exists``). No code
is shipped into the VM; no per-image Python-dep prerequisites apply.

Contract::

    executor = VmExecutor(env.session)
    await executor.install(deployer_class, runtime)
    run_result = await executor.launch(deployer_class, runtime, prompt)
"""

from __future__ import annotations

import logging
from typing import Any

from ..agents.base import AgentRunResult
from .runtime import VmRuntime

logger = logging.getLogger(__name__)


class VmExecutor:
    """Attach the session to the runtime and dispatch deployer methods.

    Construction takes the open CUA session from the env layer; ``install``
    / ``launch`` build a deployer instance with the runtime (session
    injected) and await the corresponding method. The deployer is
    responsible for all VM I/O.
    """

    def __init__(self, session: Any):
        self._session = session

    async def install(self, deployer_cls: type, runtime: VmRuntime) -> None:
        runtime.session = self._session
        deployer = deployer_cls(runtime)
        await deployer.install()

    async def launch(
        self, deployer_cls: type, runtime: VmRuntime, prompt: str
    ) -> AgentRunResult:
        runtime.session = self._session
        deployer = deployer_cls(runtime)
        return await deployer.launch(prompt)
