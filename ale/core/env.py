"""AgenthleEnv: the single Environment for the whole benchmark.

Inherits OpenEnv's :class:`Environment`. Tasks stay in agenthle's existing
format (``main.py`` with ``@cb.tasks_config`` / ``@cb.setup_task`` /
``@cb.evaluate_task``). No Rubric — the task's ``evaluate()`` already
returns a score, which becomes ``observation.reward`` directly.

Lifecycle (framework-driven data staging happens INSIDE reset/step):

    env = AgenthleEnv(provider=p)
    obs = await env.reset_async(task_path="demo/hello", variant_index=0)
    # internally:
    #   1. provider.acquire → VM up + cua ready
    #   2. ensure_data_disk + ensure_gcs_auth
    #   3. stage_input + stage_eval         ← agent-visible data
    #   4. task.setup_fn                    ← task's own setup
    # obs.instruction == cb_task.description
    # ... agent acts ...
    obs = await env.step_async(Submit())
    # internally:
    #   1. stage_reference                  ← hidden-until-eval data
    #   2. task.evaluate_fn
    # obs.reward = float(evaluate()[0]); obs.done = True
    await env.close_async()
    # internally: upload_output (best-effort) + provider.release
"""
from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import TYPE_CHECKING, Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action

from ale.io import data_staging

from .loader import LoadedTask, load_task
from .provider import Provider, VMHandle
from .types import (
    AgenthleObservation,
    AgenthleState,
    Phase,
    ReadFile,
    RunCommand,
    Screenshot,
    Submit,
    WriteFile,
)

_log = logging.getLogger(__name__)

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

    def __init__(
        self,
        *,
        provider: Provider,
        task_path: str,
        eval_timeout_s: float = 3600.0,
    ):
        super().__init__()                 # no rubric, no transform
        self._provider = provider
        self._task_path = task_path
        self._eval_timeout_s = float(eval_timeout_s)
        self._lt: Optional[LoadedTask] = None
        self._session: Optional["cb.DesktopSession"] = None
        self._vm: Optional[VMHandle] = None
        self._st = AgenthleState(task_path=task_path)
        # Run id is set by the lifecycle (RunWriter.run_id) so close_async
        # can name the GCS upload destination. None → upload_output skipped.
        self._run_id: Optional[str] = None
        # Sub-phase tracker. Lifecycle reads this in its except handler so
        # ``run.json.termination.phase`` shows the exact stage that failed
        # (env_start vs stage_inputs vs task_setup vs stage_reference vs
        # evaluation vs cleanup). Single source of truth so the lifecycle
        # doesn't have to mirror env's internal step-through.
        self._current_phase: Phase = "unknown"

    @property
    def current_phase(self) -> Phase:
        """The most-recently entered sub-phase. Useful in lifecycle's
        except handler to tag :class:`run.json` ``termination.phase``."""
        return self._current_phase

    def set_run_id(self, run_id: str | None) -> None:
        """Lifecycle hook — called between reset_async and step_async(Submit)
        to bind the run id (RunWriter.run_id) for output upload addressing."""
        self._run_id = run_id

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

        # === phase: env_start === provider.acquire + cua ready + (data disk)
        # Data disk + GCS auth ONLY when the task declares requires_task_data.
        # demo/hello and other in-VM-only tasks (typically smoke/test scaffolds
        # built on StubProvider) opt out by leaving requires_task_data=False,
        # in which case the framework runs zero VM shell commands here.
        self._current_phase = "env_start"
        self._vm = await self._provider.acquire(self._lt.env_spec)
        self._session = self._provider.open_session(self._vm)
        os_type = self._lt.env_spec.os
        if self._lt.task_data.requires_task_data:
            await data_staging.ensure_data_disk(self._session, os_type)
            await data_staging.ensure_gcs_auth(self._session, os_type)

            # === phase: stage_inputs === input/ + software/ + output/ from GCS.
            self._current_phase = "stage_inputs"
            await data_staging.stage_input(
                self._session, self._lt.task_data, os_type,
            )
            await data_staging.stage_eval(
                self._session, self._lt.task_data, os_type,
            )

        # === phase: task_setup === task's @setup_task user code.
        # By now input/software/output dirs are populated on the VM, so
        # task setup can `chmod`/`chown`/`stat` whatever it needs.
        # Reference data is INTENTIONALLY not yet on the VM here —
        # task setup must not depend on it (visibility rule).
        self._current_phase = "task_setup"
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
        # convention is ``list[float]`` (legacy shape); we surface element 0
        # as the reward. A scan of 398 agenthle tasks found ~13 that lack
        # the ``list[float]`` annotation and a few that might return a bare
        # number/None — :func:`_coerce_reward` accepts list, tuple, bare
        # numeric, or None gracefully so downstream sees a clean float.
        if isinstance(action, Submit):
            t0 = time.monotonic()
            reward: float | None
            eval_status: str
            eval_error: dict[str, Any] | None = None

            # === phase: stage_reference === reference/ from GCS, ONLY now.
            # The agent never sees reference data during solve. Failure here
            # surfaces as eval_status="failed" with category=stage_reference
            # so triage shows the staging step rather than the evaluator.
            self._current_phase = "stage_reference"
            try:
                await data_staging.stage_reference(
                    self._session, self._lt.task_data, self._lt.env_spec.os,
                )
            except Exception as exc:
                self._st.step_count += 1
                return AgenthleObservation(
                    done=True,
                    reward=None,
                    eval_status="failed",
                    eval_duration_s=time.monotonic() - t0,
                    eval_error={
                        "type": type(exc).__name__,
                        "message": f"stage_reference: {exc}",
                        "traceback": traceback.format_exc(),
                        "category": "stage_reference",
                    },
                )

            # === phase: evaluation === task's @evaluate_task user code.
            # Per-evaluate wall budget — independent of the agent's timeout_s
            # (which gates the agent's solve loop). Eval can be heavy on its
            # own (large reference compares, remote DB lookups) so we give it
            # its own knob; default 1h, override via ExperimentSpec.eval_timeout_s.
            self._current_phase = "evaluation"
            try:
                scores = await asyncio.wait_for(
                    self._lt.evaluate_fn(self._lt.cb_task, self._session),
                    timeout=self._eval_timeout_s,
                )
                reward = _coerce_reward(scores)
                eval_status = "success"
            except asyncio.TimeoutError:
                reward = None
                eval_status = "failed"
                eval_error = {
                    "type": "TimeoutError",
                    "message": f"evaluate exceeded eval_timeout_s={self._eval_timeout_s}",
                    "traceback": None,
                }
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

        # Pass-through actions (RunCommand / ReadFile / WriteFile / Screenshot).
        # The DesktopSession surface is what's available; we forward to its
        # named methods. Useful for ad-hoc env stepping in tests / probes.
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
    # Public handles for Deployer co-location
    # -------------------------------------------------------------------------
    # BaseAgentDeployer subclasses need the session to install/launch/collect.
    # We expose these read-only after reset_async(); before that they're None.
    # Deployers/runners are the intended callers, not arbitrary task code
    # (tasks see the session via their setup/evaluate signatures).

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
        """Best-effort upload_output, then release VM. Idempotent.

        Upload runs only when:
          - session is still alive (cua RPC reachable)
          - task_data.requires_task_data (no-op otherwise)
          - run_id was injected via ``set_run_id`` (lifecycle does this)

        Failures are logged, never raised — close must always release the VM.
        """
        self._current_phase = "cleanup"
        if self._session is not None and self._lt is not None and self._run_id:
            try:
                await data_staging.upload_output(
                    self._session, self._lt.task_data, self._lt.env_spec.os,
                    run_id=self._run_id,
                )
            except Exception as exc:                       # noqa: BLE001
                _log.warning("upload_output failed (best-effort): %s", exc)

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


def _coerce_reward(scores: Any) -> float:
    """Normalize agenthle's evaluate() return into a single reward float.

    Accepts:
      - ``list[float]`` / ``tuple[float, ...]`` — canonical agenthle shape; take [0]
      - bare ``int`` / ``float``                — return as-is
      - ``None`` / empty list                   — 0.0
      - anything else                           — 0.0 with a log line

    Defensive because a 398-task survey turned up ~13 tasks whose
    ``evaluate`` lacks a ``list[float]`` annotation; if any of them return
    a bare scalar we don't want a TypeError to crash the run.
    """
    if scores is None:
        return 0.0
    if isinstance(scores, bool):
        return float(scores)
    if isinstance(scores, (int, float)):
        return float(scores)
    if isinstance(scores, (list, tuple)):
        if not scores:
            return 0.0
        try:
            return float(scores[0])
        except (TypeError, ValueError):
            return 0.0
    # Unknown shape — log + 0.0 (don't crash the run)
    import logging
    logging.getLogger(__name__).warning(
        "evaluate() returned unsupported shape %r; treating as 0.0", type(scores).__name__,
    )
    return 0.0
