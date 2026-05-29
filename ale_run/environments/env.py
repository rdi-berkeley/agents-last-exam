"""ALEEnv — OpenEnv-shaped wrapper around a Provider + SandboxHandle + DesktopSession.

The benchmark drives the tested agent natively inside the environment, not via
(action, observation) pairs from the orchestrator. ``step()`` / ``step_async()``
intentionally raise ``NotImplementedError``: the wrapper exists so the
configured env can be handed to OpenEnv-compatible consumers in a standard
shape; orchestration accesses ``.session`` / ``.handle`` / ``.current_phase``
directly.

Lifecycle::

    env = ALEEnv(provider=p, spec=env_spec)
    obs = await env.reset_async()           # acquires env + opens session
    # ... orchestrator uses env.session, env.sandbox, env.current_phase ...
    await env.close_async(mode="delete")    # releases env
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from ..base_interface import SandboxSpec, Provider, ReleaseMode, SandboxHandle

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
    """One env per (task × variant × agent). SandboxHandle is owned for the env's lifetime."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, *, provider: Provider, spec: SandboxSpec):
        super().__init__()
        self._provider = provider
        self._spec = spec
        self._sandbox: SandboxHandle | None = None
        self._session: Any | None = None
        self._current_phase: str | None = None

    # -------------------------------------------------------------------- props

    @property
    def spec(self) -> SandboxSpec:
        return self._spec

    @property
    def sandbox(self) -> SandboxHandle:
        if self._sandbox is None:
            raise RuntimeError("env.sandbox accessed before reset_async()")
        return self._sandbox

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
    ) -> Observation:
        self._current_phase = PHASE_ENV_START
        self._sandbox = await self._provider.acquire(self._spec)
        self._session = self._provider.open_session(self._sandbox)
        # Windows-only: force the framebuffer to a known size so GUI tasks
        # see a deterministic screen. Linux X server picks its own size.
        if self._sandbox.os == "windows":
            await _set_windows_resolution(
                self._sandbox, has_gpu=self._spec.gpu is not None,
            )
        return Observation()

    async def reset_session(self) -> Any:
        """Force-close the current session + reopen against the same env.

        Used by :class:`TaskDriver` to recover from transient CUA-transport
        failures during ``setup()`` / ``evaluate()`` (simprun parity:
        ``TaskEnv.close(force=True)`` + ``TaskEnv.connect()``). The env
        handle is unchanged — only the session/computer wrapper is
        rebuilt; ``env.session`` is updated in place so deployer/runtime
        references picked up after this call see the new session.
        """
        if self._sandbox is None:
            raise RuntimeError("reset_session() before reset_async()")
        if self._session is not None:
            await _force_close_session(self._session)
        self._session = self._provider.open_session(self._sandbox)
        return self._session

    async def close_async(self, mode: ReleaseMode = "delete") -> None:
        self._current_phase = PHASE_CLEANUP
        if self._session is not None:
            await _force_close_session(self._session)
        self._session = None
        if self._sandbox is not None:
            try:
                await self._provider.release(self._sandbox, mode=mode)
            except Exception as e:
                logger.warning("provider.release failed for %s: %s", self._sandbox.id, e)
        self._sandbox = None

    async def step_async(self, *args: Any, **kwargs: Any) -> Observation:
        raise NotImplementedError(
            "ALEEnv.step_async() is intentionally absent: this benchmark "
            "drives the tested agent natively inside the environment, not via "
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
            "expose an OpenEnv-style mid-run State snapshot. Use env.sandbox / "
            "env.session / env.current_phase from the orchestration layer."
        )


# Internal helpers (module-level so reset_session and close_async share them).


async def _force_close_session(session: Any) -> None:
    """Best-effort hard shutdown of a CUA session.

    Mirrors simprun TaskEnv.close(force=True): call interface.force_close()
    to break any in-flight WebSocket / HTTP, then session.close() with a
    bounded timeout so the runner doesn't wedge on a dead env.
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


# ============================================================================
# Windows framebuffer prep
# ============================================================================

_EXPECTED_RESOLUTION = {True: (1920, 1080), False: (1024, 768)}

_SET_RES_PY = """\
import ctypes, ctypes.wintypes as wt, sys
u = ctypes.windll.user32
cur_w, cur_h = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
tw, th = int(sys.argv[1]), int(sys.argv[2])
if (cur_w, cur_h) == (tw, th):
    print("already_ok"); sys.exit(0)
fields = [
    ("a",ctypes.c_wchar*32),("b",wt.WORD),("c",wt.WORD),
    ("d",wt.WORD),("e",wt.WORD),("f",wt.DWORD),
    ("g",ctypes.c_long),("h",ctypes.c_long),
    ("i",wt.DWORD),("j",wt.DWORD),
    ("k",ctypes.c_short),("l",ctypes.c_short),
    ("m",ctypes.c_short),("n",ctypes.c_short),("o",ctypes.c_short),
    ("p",ctypes.c_wchar*32),("q",wt.WORD),("r",wt.DWORD),
    ("w",wt.DWORD),("ht",wt.DWORD),("fl",wt.DWORD),("fr",wt.DWORD),
]
DM = type("DM", (ctypes.Structure,), {"_fields_": fields})
dm = DM(); dm.d = ctypes.sizeof(dm)
u.EnumDisplaySettingsW(None, -1, ctypes.byref(dm))
dm.w = tw; dm.ht = th; dm.f = 0x80000 | 0x100000
r = u.ChangeDisplaySettingsW(ctypes.byref(dm), 0)
print("set_ok" if r == 0 else f"failed:{r}")
"""


async def _set_windows_resolution(sandbox, *, has_gpu: bool) -> None:
    """Force the Windows sandbox framebuffer to a known size.

    Best-effort: failure logs a warning but doesn't fail reset_async —
    the tested agent can usually solve at default resolution too.
    """
    target_w, target_h = _EXPECTED_RESOLUTION[has_gpu]
    remote_path = r"C:\Users\User\_set_resolution.py"
    try:
        await sandbox.write_file(remote_path, _SET_RES_PY)
        result = await sandbox.run_command(
            f'python "{remote_path}" {target_w} {target_h}', timeout=20,
        )
        out = (result.stdout or "").strip()
        if "set_ok" in out:
            logger.info("display resolution set to %dx%d", target_w, target_h)
        elif "already_ok" in out:
            logger.info("display resolution already %dx%d", target_w, target_h)
        else:
            logger.warning("display resolution change result: %s", out)
    except Exception as e:
        logger.warning("failed to set display resolution: %s", e)
