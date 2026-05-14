"""AgenthleEnv: the single Environment for the whole benchmark.

Inherits OpenEnv's :class:`Environment`. Tasks stay in agenthle's existing
format (``main.py`` with ``@cb.tasks_config`` / ``@cb.setup_task`` /
``@cb.evaluate_task``). No Rubric — the task's ``evaluate()`` already
returns a score, which becomes ``observation.reward`` directly.

Lifecycle::

    env = AgenthleEnv(provider=p)
    obs = await env.reset_async(task_path="demo/hello", variant_index=0)
    # obs.instruction == cb_task.description
    # ... agent acts ...
    obs = await env.step_async(Submit())
    # obs.reward = float(evaluate()[0]); obs.done = True
    await env.close_async()
"""
from __future__ import annotations

import asyncio
import time
import traceback
from typing import TYPE_CHECKING, Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action

from .loader import LoadedTask, load_task
from .provider import Provider, VMHandle
from .types import (
    AgenthleObservation,
    AgenthleState,
    ReadFile,
    RunCommand,
    Screenshot,
    Submit,
    WriteFile,
)

if TYPE_CHECKING:
    import cua_bench as cb


class AgenthleEnv(Environment[Action, AgenthleObservation, AgenthleState]):
    """One Env class, one task per instance.

    Following gym/gymnasium convention: an env is bound to a specific task
    at construction time. Use :func:`ale.make` / :func:`ale.register` to
    create instances::

        env = ale.make("demo/hello", provider=GCSDirectProvider(...))
        obs = await env.reset_async(variant_index=0)

    Variants of the same task are independent — each ``reset_async`` call
    releases the prior VM and acquires a fresh one.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, *, provider: Provider, task_path: str):
        super().__init__()                 # no rubric, no transform
        self._provider = provider
        self._task_path = task_path
        self._lt: Optional[LoadedTask] = None
        self._session: Optional["cb.DesktopSession"] = None
        self._vm: Optional[VMHandle] = None
        self._st = AgenthleState(task_path=task_path)

    @property
    def task_path(self) -> str:
        """The task this env was bound to at construction time."""
        return self._task_path

    # -------------------------------------------------------------------------
    # OpenEnv API surface
    # -------------------------------------------------------------------------

    def reset(self, *args: Any, **kwargs: Any) -> AgenthleObservation:
        raise NotImplementedError(
            "AgenthleEnv is async-only — use reset_async(). VM acquisition "
            "is inherently IO-bound."
        )

    async def reset_async(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        *,
        variant_index: int = 0,
        **_: Any,
    ) -> AgenthleObservation:
        # Release any prior VM — variants are independent.
        if self._vm is not None:
            await self._provider.release(self._vm)
            self._vm = None
            self._session = None

        # Re-load picked variant of our bound task module.
        # (Python's sys.modules caches the module import, so this is cheap.)
        self._lt = load_task(self._task_path, variant_index)

        # Acquire VM + open session.
        self._vm = await self._provider.acquire(self._lt.env_spec)
        self._session = self._provider.open_session(self._vm)

        # Run the task's start function.
        await self._lt.start_fn(self._lt.cb_task, self._session)

        # State + initial observation.
        self._st = AgenthleState(
            task_path=self._task_path,
            variant_index=variant_index,
            vm_id=self._vm.id,
            episode_id=episode_id or self._st.episode_id,
        )
        return AgenthleObservation(
            instruction=self._lt.description,
            done=False,
            reward=None,
        )

    def step(self, *args: Any, **kwargs: Any) -> AgenthleObservation:
        raise NotImplementedError("AgenthleEnv is async-only — use step_async().")

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **_: Any,
    ) -> AgenthleObservation:
        if self._lt is None or self._session is None:
            raise RuntimeError("step_async() before reset_async()")

        # Final submission — call the task's evaluate() directly. Agenthle's
        # convention returns `[float]` with exactly one element (legacy list
        # shape); we take that element as the reward.
        if isinstance(action, Submit):
            t0 = time.monotonic()
            reward: float | None
            eval_status: str
            eval_error: dict[str, Any] | None = None
            try:
                scores = await self._lt.evaluate_fn(self._lt.cb_task, self._session)
                reward = float(scores[0]) if scores else 0.0
                eval_status = "success"
            except Exception as exc:
                reward = None
                eval_status = "failed"
                eval_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            eval_duration_s = time.monotonic() - t0
            self._st.step_count += 1
            return AgenthleObservation(
                done=True,
                reward=reward,
                eval_status=eval_status,
                eval_duration_s=eval_duration_s,
                eval_error=eval_error,
            )

        # Pass-through actions for NativeAgent loops. The DesktopSession
        # surface is what's available; we forward to its named methods.
        s = self._session
        if isinstance(action, RunCommand):
            cr = await s.run_command(action.cmd)  # type: ignore[attr-defined]
            self._st.step_count += 1
            return AgenthleObservation(
                stdout=cmd_stdout(cr),
                stderr=cmd_stderr(cr),
                exit_code=cmd_rc(cr),
                done=False,
            )
        if isinstance(action, ReadFile):
            data = await s.read_file(action.path)  # type: ignore[attr-defined]
            self._st.step_count += 1
            blob = data.encode("utf-8") if isinstance(data, str) else data
            return AgenthleObservation(file_data=blob, done=False)
        if isinstance(action, WriteFile):
            text = action.data.decode("utf-8") if isinstance(action.data, bytes) else action.data
            await s.write_file(action.path, text)  # type: ignore[attr-defined]
            self._st.step_count += 1
            return AgenthleObservation(done=False)
        if isinstance(action, Screenshot):
            png = await s.screenshot()  # type: ignore[attr-defined]
            self._st.step_count += 1
            return AgenthleObservation(screenshot_png=png, done=False)

        return AgenthleObservation(
            metadata={"error": f"Unknown action type: {type(action).__name__}"},
            done=False,
        )

    @property
    def state(self) -> AgenthleState:
        return self._st

    # -------------------------------------------------------------------------
    # Public handles for Agent / Deployer co-location
    # -------------------------------------------------------------------------
    # InstalledAgent's deployer needs the session to install/launch/collect the
    # in-VM CLI. We expose these read-only after reset_async(); before that
    # they're None. Agents/runners are the intended callers, not arbitrary
    # task code (tasks see the session via their setup/evaluate signatures).

    @property
    def session(self) -> "cb.DesktopSession":
        if self._session is None:
            raise RuntimeError("env.session accessed before reset_async()")
        return self._session

    @property
    def vm(self) -> VMHandle:
        if self._vm is None:
            raise RuntimeError("env.vm accessed before reset_async()")
        return self._vm

    # -------------------------------------------------------------------------
    # Resource management
    # -------------------------------------------------------------------------

    async def close_async(self) -> None:
        """Release the VM and drop refs. Idempotent."""
        if self._vm is not None:
            await self._provider.release(self._vm)
        self._vm = None
        self._session = None
        self._lt = None

    def close(self) -> None:
        if self._vm is None:
            return
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            asyncio.run(self.close_async())
            return
        if loop.is_running():
            loop.create_task(self.close_async())
        else:
            loop.run_until_complete(self.close_async())
