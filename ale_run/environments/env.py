"""ALEEnv — OpenEnv-shaped wrapper around a Provider + VMHandle + DesktopSession.

The benchmark drives the tested agent natively inside the VM, not via
(action, observation) pairs from the orchestrator. ``step()`` / ``step_async()``
intentionally raise ``NotImplementedError``: the wrapper exists so the
configured env can be handed to OpenEnv-compatible consumers in a standard
shape; orchastration accesses ``.session`` / ``.vm`` / ``.current_phase``
directly.

Lifecycle::

    env = ALEEnv(provider=p, spec=env_spec)
    obs = await env.reset_async()           # acquires VM + opens session
    # ... orchestrator uses env.session, env.vm, env.current_phase ...
    await env.close_async(mode="delete")    # releases VM
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from .providers.provider import EnvSpec, Provider, ReleaseMode, VMHandle

logger = logging.getLogger(__name__)


# LOG_SPEC §4 termination.phase values — the lifecycle reads
# ALEEnv.current_phase to populate run.json on failure paths.
PHASE_ENV_START = "env_start"
PHASE_STAGE_INPUTS = "stage_inputs"
PHASE_TASK_SETUP = "task_setup"
PHASE_AGENT_RUN = "agent_run"
PHASE_STAGE_REFERENCE = "stage_reference"
PHASE_EVALUATION = "evaluation"
PHASE_CLEANUP = "cleanup"


class ALEEnv(Environment[Action, Observation, State]):
    """One env per (task × variant × agent). VM is owned for the env's lifetime."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, *, provider: Provider, spec: EnvSpec):
        super().__init__()
        self._provider = provider
        self._spec = spec
        self._vm: VMHandle | None = None
        self._session: Any | None = None
        self._current_phase: str | None = None

    # -------------------------------------------------------------------- props

    @property
    def spec(self) -> EnvSpec:
        return self._spec

    @property
    def vm(self) -> VMHandle:
        if self._vm is None:
            raise RuntimeError("env.vm accessed before reset_async()")
        return self._vm

    @property
    def session(self) -> Any:
        if self._session is None:
            raise RuntimeError("env.session accessed before reset_async()")
        return self._session

    @property
    def current_phase(self) -> str | None:
        return self._current_phase

    def set_phase(self, phase: str) -> None:
        self._current_phase = phase

    # -------------------------------------------------------------- OpenEnv API

    async def reset_async(
        self,
        *,
        exclude_profiles: set[str] | None = None,
    ) -> Observation:
        self._current_phase = PHASE_ENV_START
        self._vm = await self._provider.acquire(
            self._spec, exclude_profiles=exclude_profiles,
        )
        self._session = self._provider.open_session(self._vm)
        return Observation()

    async def reset_session(self) -> Any:
        """Force-close the current session + reopen against the same VM.

        Used by :class:`TaskEnv` to recover from transient CUA-transport
        failures during ``setup()`` / ``evaluate()`` (simprun parity:
        ``TaskEnv.close(force=True)`` + ``TaskEnv.connect()``). The VM
        handle is unchanged — only the session/computer wrapper is
        rebuilt; ``env.session`` is updated in place so deployer/runtime
        references picked up after this call see the new session.
        """
        if self._vm is None:
            raise RuntimeError("reset_session() before reset_async()")
        if self._session is not None:
            await _force_close_session(self._session)
        self._session = self._provider.open_session(self._vm)
        return self._session

    async def close_async(self, mode: ReleaseMode = "delete") -> None:
        self._current_phase = PHASE_CLEANUP
        if self._session is not None:
            await _force_close_session(self._session)
        self._session = None
        if self._vm is not None:
            try:
                await self._provider.release(self._vm, mode=mode)
            except Exception as e:
                logger.warning("provider.release failed for %s: %s", self._vm.id, e)
        self._vm = None

    async def step_async(self, *args: Any, **kwargs: Any) -> Observation:
        raise NotImplementedError(
            "ALEEnv.step_async() is intentionally absent: this benchmark "
            "drives the tested agent natively inside the VM, not via "
            "(action, observation) pairs from the orchestrator."
        )

    def step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ALEEnv is async-only.")

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ALEEnv is async-only — use reset_async().")

    def close(self) -> None:
        raise NotImplementedError("ALEEnv is async-only — use close_async().")

    def state(self) -> State:
        raise NotImplementedError(
            "ALEEnv.state() is intentionally absent: this benchmark does not "
            "expose an OpenEnv-style mid-run State snapshot. Use env.vm / "
            "env.session / env.current_phase from the orchastration layer."
        )


# Internal helpers (module-level so reset_session and close_async share them).


async def _force_close_session(session: Any) -> None:
    """Best-effort hard shutdown of a CUA session.

    Mirrors simprun TaskEnv.close(force=True): call interface.force_close()
    to break any in-flight WebSocket / HTTP, then session.close() with a
    bounded timeout so the runner doesn't wedge on a dead VM.
    """
    iface = None
    try:
        iface = session.computer.interface
    except Exception:
        iface = None
    if iface is not None:
        force_close = getattr(iface, "force_close", None)
        if callable(force_close):
            try:
                force_close()
            except Exception as e:
                logger.debug("interface.force_close failed: %s", e)
    try:
        await asyncio.wait_for(session.close(), timeout=10)
    except (asyncio.TimeoutError, Exception) as e:
        logger.debug("session.close failed/timed out: %s", e)
