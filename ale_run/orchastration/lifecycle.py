"""Per-unit lifecycle: the 4-phase port of simprun's SimpRunTaskRunner.run().

Reads YAML-built ``RunUnit`` + a Provider; produces a ``UnitResult`` and the
LOG_SPEC-shaped on-disk artifacts under ``output_root/<slugs>/v<i>/<ts>/``.

Phase mapping (preserved from simprun, surface renamed to LOG_SPEC events):

  - Phase 0  provision      env.reset_async() → VM + open session
  - Phase 1  start          data_staging + TaskEnv(session).setup()
  - Phase 2  agent          runtime.install_deployer() + .launch_deployer()
  - Phase 2b fanout         pull origin_log/ (vm-only), emit output_gather_skipped
  - Phase 3  evaluate       TaskEnv.evaluate(), then deployer.parse_artifacts
  - Phase 4  cleanup        TaskEnv.close + env.close_async(mode=delete)
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from pathlib import Path
from typing import Any

from ..base_interface import (
    AgentRunResult,
    BaseRuntime,
    EnvSpec,
    Provider,
    Trajectory,
    TrajectoryBuilder,
)
from ..environments.env import ALEEnv
from ..environments.runtime import DockerRuntime, LocalRuntime, VmRuntime
from ..tasks.loader import TaskLoader
from ..tasks.task_env import TaskEnv
from . import gather
from .factory import build_config, resolve_agent
from .monitor import RateLimitDetector
from .run_writer import RunWriter
from .spec import ArtifactsSpec, RunUnit, UnitResult
from .termination import classify_error, err_dict, redact_config

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT_S = 7200

# Mount-fallback: how many provision attempts before we give up. Matches
# simprun's single retry — the original VM + one alternate profile.
_MOUNT_FALLBACK_MAX_ATTEMPTS = 2


def _is_mount_failure(exc: Exception) -> bool:
    """Detect ``ensure_data_disk`` mount failures so we can swap profiles.

    Exact match against simprun/runner.py:217:
    ``if "Failed to mount" in str(mount_err) and self._vm:``
    The only call site that raises this string is ``_ensure_linux_data_disk``
    in ``environments/data_staging.py``.
    """
    return "Failed to mount" in str(exc)


# ======================================================================
# Public entry
# ======================================================================


# Process-wide shutdown event: set by SIGINT/SIGTERM handlers, read by the
# rate-limit monitor + per-unit cleanup so signal arrival flips every
# in-flight unit to "cancelled" without losing the events.jsonl tail.
_shutdown_event: asyncio.Event | None = None


def get_shutdown_event() -> asyncio.Event:
    """Lazy singleton — needs a running loop to construct."""
    global _shutdown_event
    if _shutdown_event is None:
        _shutdown_event = asyncio.Event()
    return _shutdown_event


def install_signal_handlers() -> None:
    """Wire SIGINT / SIGTERM → shutdown event (simprun runner.py:765-780).

    Idempotent. Skipped silently when not running on the main thread (the
    asyncio loop policy refuses ``add_signal_handler`` from worker threads).
    """
    import threading

    if threading.current_thread() is not threading.main_thread():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    event = get_shutdown_event()

    def _on_signal(sig: signal.Signals) -> None:
        logger.warning("Signal %s received — flagging shutdown", sig.name)
        event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig)
        except (NotImplementedError, RuntimeError):
            # Windows / nested-loop edge cases: best-effort.
            pass


async def run_one_unit(
    *,
    unit: RunUnit,
    provider: Provider,
    output_root: Path,
    artifacts: ArtifactsSpec | None = None,
    sem: asyncio.Semaphore | None = None,
    cleanup_mode: str = "delete",
) -> UnitResult:
    started = time.monotonic()
    effective_cleanup_mode = cleanup_mode

    # ---- 1. Resolve agent + config from the AgentSpec ----
    try:
        deployer_cls, config_cls = resolve_agent(unit.agent_spec)
    except Exception as e:
        # We can't even open a RunWriter without knowing the model — bail.
        logger.exception("resolve_agent failed for %s: %s", unit.slug, e)
        return UnitResult(
            unit=unit,
            status="not_executed",
            eval_status="not_executed",
            duration_s=round(time.monotonic() - started, 2),
            error=str(e),
        )

    config = build_config(config_cls, unit.agent_spec.config)
    runtime_kind = unit.agent_spec.runtime or "vm"

    # ---- 2. Open RunWriter (creates the run dir + events.jsonl) ----
    writer = RunWriter(
        output_root=output_root,
        agent_id=unit.agent_id,
        model=config.model,
        task_path=unit.task_path,
        variant_index=unit.variant_index,
    )
    writer.emit_event(
        "run_started",
        agent=unit.agent_id,
        agent_class=unit.agent_spec.class_,
        model=config.model,
        task=unit.task_path,
        variant_index=unit.variant_index,
        runtime=runtime_kind,
    )

    env: ALEEnv | None = None
    task_env_obj: TaskEnv | None = None
    builder: TrajectoryBuilder | None = None
    trajectory: Trajectory | None = None
    run_result: AgentRunResult | None = None
    runtime: BaseRuntime | None = None

    status = "completed"
    score: float | None = None
    error_str: str | None = None
    error_obj: dict[str, Any] | None = None
    phase: str | None = None
    eval_status = "not_executed"
    eval_duration_s: float | None = None
    eval_error: dict[str, Any] | None = None

    try:
        # Single-knob concurrency: holding sem for the whole unit caps both
        # VM count and concurrent agent runs in one number.
        if sem is not None:
            writer.emit_event("provision_wait")
            await sem.acquire()
        try:
            task_path = Path("tasks") / unit.task_path
            task_meta = TaskLoader(str(task_path)).load(unit.variant_index)
            env_spec = _build_env_spec(task_meta, unit=unit)
            timeout_s = int(task_meta.get("timeout_s") or _DEFAULT_TIMEOUT_S)

            # ============================================================
            # Phase 0 + Phase 1 — provision + data staging.
            # Mount-fallback retry (simprun parity): if the data disk fails
            # to mount on the chosen capacity profile, delete the VM,
            # exclude that profile, and retry once with the next one.
            # ============================================================
            excluded_profiles: set[str] = set()
            for attempt in range(_MOUNT_FALLBACK_MAX_ATTEMPTS):
                writer.emit_event("provision_started")
                env = ALEEnv(provider=provider, spec=env_spec)
                await env.reset_async(exclude_profiles=excluded_profiles or None)
                writer.emit_event(
                    "provision_done",
                    vm_id=env.vm.id,
                    capacity_profile=env.vm.metadata.get("capacity_profile"),
                )

                # Build the trajectory the first time around (instruction is stable).
                if builder is None:
                    builder = TrajectoryBuilder(
                        agent_name=getattr(config, "name", unit.agent_spec.class_),
                        agent_version=None,
                        model=config.model,
                        task_path=unit.task_path,
                        variant_index=unit.variant_index,
                        instruction=task_meta["description"],
                    )
                    builder.add_step(source="user", message=task_meta["description"])

                try:
                    # ── Matches simprun's _phase1_start: data staging + task setup
                    # are wrapped together, so a transient mount failure during
                    # ensure_data_disk triggers the profile swap, but any error
                    # raised by task_env.setup() that happens to contain the
                    # "Failed to mount" substring would also be retried (matching
                    # simprun's behaviour even though that doesn't occur today).
                    env.set_phase("stage_inputs")
                    await _stage_data(
                        env=env,
                        provider=provider,
                        task_meta=task_meta,
                        run_id=writer.run_id,
                        task_id=unit.task_path,
                    )
                    env.set_phase("task_setup")
                    task_env_obj = TaskEnv(
                        task_path=str(task_path),
                        session=env.session,
                        variant=unit.variant_index,
                        os_type=env.vm.os,
                        # Closure capture is OK — `env` is the loop's
                        # current iteration; on the next mount-fallback
                        # retry we build a new TaskEnv with a new closure.
                        session_rebuilder=env.reset_session,
                    )
                    await task_env_obj.setup()
                    break  # phase 1 succeeded
                except RuntimeError as e:
                    if not _is_mount_failure(e):
                        raise
                    failed_profile = env.vm.metadata.get("capacity_profile")
                    logger.warning(
                        "Disk mount failed on profile %s for %s — deleting VM and "
                        "retrying with a different profile",
                        failed_profile, unit.slug,
                    )
                    writer.emit_event(
                        "mount_fallback",
                        failed_profile=failed_profile,
                        excluded_so_far=sorted(excluded_profiles),
                    )
                    if failed_profile:
                        excluded_profiles.add(failed_profile)
                    try:
                        await env.close_async(mode="delete")
                    except Exception as cleanup_e:
                        logger.debug("close after mount fail: %s", cleanup_e)
                    env = None
                    task_env_obj = None
                    if attempt == _MOUNT_FALLBACK_MAX_ATTEMPTS - 1:
                        raise

            assert env is not None  # loop exits with env+task_env set, or raises
            assert task_env_obj is not None

            # ============================================================
            # Phase 2 — agent
            # ============================================================
            env.set_phase("agent_run")
            agent_name = getattr(config, "name", "agent")
            host_artifacts_dir = writer.run_dir / "origin_log" / agent_name
            host_artifacts_dir.mkdir(parents=True, exist_ok=True)
            runtime = _build_runtime(
                runtime_kind=runtime_kind,
                env=env,
                config=config,
                agent_name=agent_name,
                run_id=writer.run_id,
                host_artifacts_dir=host_artifacts_dir,
            )
            writer.emit_event(
                "agent_run_started",
                runtime=runtime_kind,
                work_dir=runtime.work_dir,
            )
            deployer = await runtime.install_deployer(deployer_cls)

            # ─── Incremental sync (simprun parity) ─────────────────────
            # Tail each ``hot_artifacts`` file on the VM into its host
            # mirror via download_file_range. Boundary-safe slicing at
            # the last `\n` (jsonl) is in sync_helpers.apply_range_step.
            # State (per-path byte offset) is held by the puller; on a
            # transport / parse failure the offset is NOT advanced, so
            # the next tick re-pulls the same bytes — matching simprun's
            # _sync_incremental invariant.
            puller, puller_targets = _build_incremental_puller(
                deployer_cls=deployer_cls,
                runtime=runtime,
                run_id=writer.run_id,
                task_id=unit.task_path,
            )
            if puller is not None:
                writer.emit_event(
                    "incremental_pull_started",
                    targets=[t.vm_path for t in puller_targets],
                    interval_s=puller.interval_s,
                )
                puller.start()

            # ─── Rate-limit monitor (simprun parity) ────────────────────
            # Tails the locally-mirrored stderr.log (which IncrementalPuller
            # is writing at its own tick cadence) every 30s; on hit-rate threshold,
            # cancels the agent and flags the run for cleanup_mode=keep so
            # we don't waste the VM boot on the retry. Matches simprun
            # runner.py:_monitor_rate_limits + the paused-state cleanup.
            rate_detector = RateLimitDetector()
            stderr_host = host_artifacts_dir / "stderr.log"
            launch_task = asyncio.create_task(
                runtime.launch_deployer(deployer, task_meta["description"]),
                name="agent_launch",
            )
            monitor_task = asyncio.create_task(
                _monitor_rate_limits(rate_detector, stderr_host, writer),
                name="rate_limit_monitor",
            )

            rate_limited = False
            try:
                try:
                    done, _pending = await asyncio.wait(
                        {launch_task, monitor_task},
                        timeout=timeout_s,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except asyncio.CancelledError:
                    launch_task.cancel()
                    monitor_task.cancel()
                    raise

                if monitor_task in done and not launch_task.done():
                    # Rate-limit detected mid-run.
                    rate_limited = True
                    launch_task.cancel()
                    try:
                        await launch_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    run_result = AgentRunResult(
                        status="failed",
                        error="rate limit triggered during agent run",
                        duration_s=None,
                    )
                elif launch_task in done:
                    monitor_task.cancel()
                    try:
                        await monitor_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    try:
                        run_result = launch_task.result()
                    except Exception as launch_exc:
                        run_result = AgentRunResult(
                            status="failed",
                            error=str(launch_exc),
                            duration_s=None,
                        )
                else:
                    # Neither completed → wall-clock timeout.
                    launch_task.cancel()
                    monitor_task.cancel()
                    for t in (launch_task, monitor_task):
                        try:
                            await t
                        except (asyncio.CancelledError, Exception):
                            pass
                    run_result = AgentRunResult(
                        status="timeout",
                        error=f"agent wall-budget exceeded after {timeout_s}s",
                        duration_s=float(timeout_s),
                    )
            finally:
                if puller is not None:
                    reconcile_err = await puller.stop()
                    if reconcile_err:
                        writer.emit_event(
                            "incremental_pull_final_failed", error=reconcile_err,
                        )

            if rate_limited:
                # Force the VM to stay alive — operator retries the run
                # after the rate-limit window without paying for a re-boot.
                effective_cleanup_mode = "keep"
                writer.emit_event("rate_limit_paused", cleanup_mode="keep")
            writer.emit_event(
                "agent_finished",
                status=run_result.status,
                error=run_result.error,
            )

            # ============================================================
            # Phase 2b — post-launch fanout (origin_log gather + output stub)
            #
            # Gather is only meaningful for VmRuntime, where work_dir lives
            # in the VM. Local / docker runtimes write directly into
            # host_artifacts_dir, so the gather step is a no-op.
            # ============================================================
            writer.emit_event("post_launch_fanout_started", gcs_bucket="(cua direct)")
            if isinstance(runtime, VmRuntime):
                try:
                    report = await gather.pull_dir(
                        env.session,
                        src=runtime.work_dir,
                        dst=runtime.host_artifacts_dir,
                        os_type=env.vm.os,
                    )
                    if report.get("error"):
                        writer.emit_event("origin_log_gather_failed", report=report)
                    else:
                        writer.emit_event("origin_log_gather_done", report=report)
                except Exception as e:
                    writer.emit_event(
                        "origin_log_gather_failed",
                        report={"transport": "cua", "files": 0, "error": str(e)},
                    )
            else:
                writer.emit_event(
                    "origin_log_gather_skipped",
                    reason=f"runtime={runtime.kind} writes to host directly",
                )
            writer.emit_event(
                "output_gather_skipped",
                reason="not_implemented_yet",
                checked_keys=["remote_output_dir", "output_path", "runtime_output_dir"],
            )

            # ============================================================
            # Phase 3 — upload VM output → GCS, stage reference, evaluate
            # ============================================================
            # 3a. UPLOADING_OUTPUT (simprun runner.py:_phase3_evaluate first half)
            #     Push the VM's output dir to GCS so it's preserved across
            #     VM teardown. Best-effort: failure logs a warning but
            #     doesn't fail the run (eval still runs on the live VM).
            await _upload_output_best_effort(
                env=env, provider=provider, task_meta=task_meta,
                run_id=writer.run_id, task_id=unit.task_path, writer=writer,
            )

            # 3b. STAGING_EVAL (simprun runner.py:_phase3_evaluate second half)
            #     Pull reference data from GCS to the VM if the task needs
            #     it for scoring. Best-effort: many tasks don't have a
            #     reference/ prefix and we just log + continue.
            env.set_phase("stage_reference")
            await _stage_reference_best_effort(
                env=env, provider=provider, task_meta=task_meta,
                run_id=writer.run_id, task_id=unit.task_path, writer=writer,
            )

            # 3c. RUNNING_EVAL. Raw eval output is funnelled into
            #     eval_result.json + run.json + trajectory.json — there is
            #     no debug/ folder in the new spec, so simprun's
            #     `debug/eval/result.json` raw dump has no destination here.
            env.set_phase("evaluation")
            eval_start = time.monotonic()
            try:
                eval_out = await task_env_obj.evaluate()
                eval_duration_s = round(time.monotonic() - eval_start, 4)
                if eval_out is None or eval_out.get("error"):
                    eval_status = "failed"
                    eval_error = (
                        {"type": "Exception", "message": str(eval_out.get("error")),
                         "traceback": str(eval_out.get("error"))}
                        if eval_out
                        else None
                    )
                else:
                    eval_status = "success"
                    score = _extract_score(eval_out)
            except Exception as e:
                eval_duration_s = round(time.monotonic() - eval_start, 4)
                eval_status = "failed"
                eval_error = err_dict(e)
                logger.exception("evaluate raised for %s", unit.slug)

            # ============================================================
            # Trajectory finalize via deployer.parse_artifacts (LOG_SPEC §5)
            # ============================================================
            if builder is not None and run_result is not None:
                try:
                    deployer_cls.parse_artifacts(
                        work_dir=runtime.host_artifacts_dir,
                        config=config,
                        run_result=run_result,
                        builder=builder,
                    )
                except Exception as e:
                    logger.warning("parse_artifacts failed for %s: %s", unit.slug, e)
                    builder.add_step(
                        source="system",
                        message=f"parse_artifacts failed: {e}",
                        extra={"reason": "parse_error"},
                    )

                traj_status = (
                    "completed"
                    if run_result.status == "completed"
                    else "timeout"
                    if run_result.status == "timeout"
                    else "failed"
                )
                trajectory = builder.finalize(reward=score, status=traj_status)
                writer.write_trajectory(trajectory)

            # ============================================================
            # Lifecycle status promotion (LOG_SPEC §4)
            # ============================================================
            if run_result is not None and run_result.status == "timeout":
                status = "timeout"
                error_str = run_result.error
            elif run_result is not None and run_result.status == "failed":
                status = "failed"
                error_str = run_result.error or "agent failed"
                error_obj = {
                    "type": "Exception",
                    "message": error_str,
                    "traceback": error_str,
                }
            elif eval_status == "failed":
                status = "failed"
                error_str = (eval_error or {}).get("message", "evaluation failed")
                error_obj = eval_error
            else:
                status = "completed"

            phase = env.current_phase if status != "completed" else None

        finally:
            if sem is not None:
                sem.release()

    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        status = "cancelled"
        phase = env.current_phase if env is not None else None
        error_str = type(e).__name__
        writer.emit_event(
            "run_cancelled",
            phase=phase,
            reason="keyboard_interrupt" if isinstance(e, KeyboardInterrupt) else "cancelled",
        )
        if not isinstance(e, KeyboardInterrupt):
            raise

    except Exception as e:
        status = "failed"
        error_str = str(e)
        error_obj = err_dict(e)
        phase = env.current_phase if env is not None else None
        writer.emit_event(
            "run_failed",
            error_type=type(e).__name__,
            message=error_str,
            phase=phase,
            category=classify_error(e),
        )
        logger.exception("run_one_unit failed for %s", unit.slug)

    finally:
        # Phase 4 — cleanup. Both branches are best-effort; never raise here.
        if task_env_obj is not None:
            try:
                await task_env_obj.close()
            except Exception as e:
                logger.debug("TaskEnv.close failed: %s", e)
        if env is not None:
            try:
                await env.close_async(mode=effective_cleanup_mode)
            except Exception as e:
                logger.debug("ALEEnv.close_async failed: %s", e)

    total_s = round(time.monotonic() - started, 2)

    # ---- Finalize the three terminal files + the run_completed event ----
    writer.write_eval_result(
        eval_status=eval_status,
        score=score,
        eval_duration_s=eval_duration_s,
        error=eval_error,
    )

    run_meta = _build_run_meta(
        run_id=writer.run_id,
        unit=unit,
        config=config,
        runtime_kind=runtime_kind,
        status=status,
        score=score,
        phase=phase,
        error_obj=error_obj,
        error_str=error_str,
        total_s=total_s,
        trajectory=trajectory,
        category=_category_from_error(error_str),
    )
    writer.write_run_json(run_meta)

    writer.emit_event(
        "run_completed",
        status=status,
        score=score,
        total_duration_s=total_s,
    )
    writer.close()

    return UnitResult(
        unit=unit,
        status=status,
        score=score,
        eval_status=eval_status,
        duration_s=total_s,
        run_dir=writer.run_dir,
        error=error_str,
    )


# ======================================================================
# Helpers
# ======================================================================


def _extract_score(eval_output: Any) -> float | None:
    """Return a numeric score while preserving valid falsy values like 0.0."""
    if eval_output is None:
        return None
    if isinstance(eval_output, dict):
        for key in ("score", "final_score"):
            if key in eval_output and eval_output[key] is not None:
                try:
                    return float(eval_output[key])
                except (ValueError, TypeError):
                    pass
        return None
    if isinstance(eval_output, list) and eval_output:
        try:
            return float(eval_output[0])
        except (ValueError, TypeError):
            return None
    if isinstance(eval_output, (int, float)):
        return float(eval_output)
    return None


def _build_env_spec(task_meta: dict[str, Any], *, unit: RunUnit | None = None) -> EnvSpec:
    snapshot = task_meta.get("image_category") or task_meta.get("snapshot_name")
    if not snapshot:
        raise RuntimeError(
            "task_card.json is missing vm.snapshot (required for env spec)"
        )
    os_type = task_meta.get("os_type") or "linux"
    task_id = unit.task_path if unit is not None else ""
    harness = unit.agent_spec.class_ if unit is not None else ""
    model_tag = (
        str(unit.agent_spec.config.get("model") or "")
        if unit is not None
        else ""
    )
    return EnvSpec(
        snapshot=snapshot,
        os=os_type,
        vcpus=int(task_meta.get("vcpus") or 4),
        memory_gb=int(task_meta.get("memory_gb") or 16),
        disk_gb=int(task_meta.get("disk_gb") or 200),
        gpu=task_meta.get("gpu"),
        task_id=task_id,
        harness=harness,
        model_tag=model_tag,
    )


def _work_dir_vm(os_type: str, agent_name: str, run_id: str) -> str:
    """VM-side scratch dir for VmRuntime deployers."""
    if os_type == "linux":
        return f"/home/user/.ale/{agent_name}/{run_id}"
    return rf"C:\Users\User\.ale\{agent_name}\{run_id}"


def _build_runtime(
    *,
    runtime_kind: str,
    env: ALEEnv,
    config: Any,
    agent_name: str,
    run_id: str,
    host_artifacts_dir: Path,
) -> BaseRuntime:
    """Dispatch yaml ``runtime: <kind>`` to the concrete substrate adapter.

    ``host_artifacts_dir`` is always a host path the lifecycle owns; for
    Local/Docker it's also the deployer's work_dir (no gather step). For
    Vm the deployer's work_dir is a VM-side path and gather copies into
    ``host_artifacts_dir`` after launch.
    """
    env_passthrough = _collect_env_passthrough()
    if runtime_kind == "vm":
        work_dir_vm = _work_dir_vm(env.vm.os, agent_name, run_id)
        return VmRuntime(
            config=config,
            work_dir=work_dir_vm,
            host_artifacts_dir=host_artifacts_dir,
            vm_endpoint=env.vm.endpoint,
            vm_os=env.vm.os,
            env=env_passthrough,
        )
    if runtime_kind == "local":
        return LocalRuntime(
            config=config,
            work_dir=str(host_artifacts_dir),
            host_artifacts_dir=host_artifacts_dir,
            vm_endpoint=env.vm.endpoint,
            vm_os=env.vm.os,
            env=env_passthrough,
        )
    if runtime_kind == "docker":
        # The container's image is left empty here — the first concrete
        # docker deployer will plumb that through yaml.
        return DockerRuntime(
            config=config,
            work_dir="/work",
            host_artifacts_dir=host_artifacts_dir,
            vm_endpoint=env.vm.endpoint,
            vm_os=env.vm.os,
            env=env_passthrough,
        )
    raise NotImplementedError(f"runtime kind {runtime_kind!r} not wired in lifecycle")


def _collect_env_passthrough() -> dict[str, str]:
    """Env vars to propagate into the VM Python process so deployers see API keys.

    Keep this list small — only well-known LLM keys we know deployers need.
    """
    import os

    keys = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "BRAVE_API_KEY",
    )
    return {k: os.environ[k] for k in keys if k in os.environ}


async def _monitor_rate_limits(
    detector: RateLimitDetector,
    stderr_local: Path,
    writer,
    interval: float = 30.0,
) -> None:
    """Tail the host-mirrored stderr.log; return when ``detector.is_triggered``.

    Verbatim port of simprun runner.py:_monitor_rate_limits, swapping the
    VM-side path for the host-mirrored copy (IncrementalPuller writes that
    file at its own tick cadence — see ``_TICK_INTERVAL_S``). The CALLER's
    `asyncio.wait` sees this task complete only on detection — on a clean
    agent finish the caller cancels this task so it returns mid-sleep.
    """
    while True:
        await asyncio.sleep(interval)
        if not stderr_local.exists():
            continue
        try:
            text = stderr_local.read_text(errors="replace")
        except OSError:
            continue
        if detector.check(text, time.monotonic()):
            logger.warning("Rate limiting detected — flagging run paused")
            writer.emit_event("rate_limit_detected")
            return


async def _upload_output_best_effort(
    *,
    env: ALEEnv,
    provider: Provider,
    task_meta: dict[str, Any],
    run_id: str,
    task_id: str,
    writer,
) -> None:
    """Push the VM's task output dir to GCS (simprun parity).

    Mirrors simprun runner.py:_phase3_evaluate phase ``output_upload``.
    Best-effort: failure logs + emits an event but does NOT abort the
    run — the live VM still has the output for the eval that follows.
    """
    task_data = task_meta.get("task_data")
    if task_data is None or not task_data.requires_task_data:
        return
    from ..environments import data_staging
    from ..base_interface import RemoteVMConfig

    vm_cfg = RemoteVMConfig(
        server_url=env.vm.endpoint, os_type=env.vm.os,
        run_id=run_id, task_id=task_id,
    )
    bucket = _results_bucket_for(provider)
    try:
        report = await asyncio.to_thread(
            data_staging.upload_output,
            vm_cfg, task_data, env.vm.os, run_id,
            gcs_results_bucket=bucket,
        )
        if report.get("uploaded"):
            writer.emit_event("vm_output_uploaded", gcs_path=report.get("gcs_path"))
        else:
            writer.emit_event(
                "vm_output_upload_skipped",
                reason=report.get("error", "no_task_data") or "no_task_data",
            )
    except Exception as e:
        logger.warning("upload_output failed (best-effort): %s", e)
        writer.emit_event("vm_output_upload_failed", error=str(e))


async def _stage_reference_best_effort(
    *,
    env: ALEEnv,
    provider: Provider,
    task_meta: dict[str, Any],
    run_id: str,
    task_id: str,
    writer,
) -> None:
    """Pull reference data from GCS onto the VM for eval (simprun parity).

    Mirrors simprun runner.py:_phase3_evaluate phase ``eval_stage``.
    Best-effort: many tasks don't ship a reference/ prefix.
    """
    task_data = task_meta.get("task_data")
    if task_data is None or not task_data.requires_task_data:
        return
    from ..environments import data_staging
    from ..base_interface import RemoteVMConfig

    vm_cfg = RemoteVMConfig(
        server_url=env.vm.endpoint, os_type=env.vm.os,
        run_id=run_id, task_id=task_id,
    )
    bucket = _stage_bucket_for(provider)
    try:
        report = await asyncio.to_thread(
            data_staging.stage_reference,
            vm_cfg, task_data, env.vm.os,
            gcs_bucket=bucket,
        )
        if report.get("skipped"):
            writer.emit_event("reference_stage_skipped", reason="no_task_data")
        else:
            writer.emit_event(
                "reference_stage_completed",
                staged_dirs=report.get("staged_dirs"),
            )
    except RuntimeError as e:
        # Reference data is optional — many tasks don't have it.
        logger.info("Reference staging skipped (may not exist): %s", e)
        writer.emit_event("reference_stage_skipped", reason=str(e)[:200])


def _build_incremental_puller(
    *,
    deployer_cls: type,
    runtime: BaseRuntime,
    run_id: str,
    task_id: str,
):
    """Return ``(puller, targets)`` for the deployer's hot artifacts.

    Returns ``(None, [])`` when the deployer declares no ``hot_artifacts``
    OR the runtime isn't VM-backed (Local/Docker write straight to host).
    When non-empty, builds a puller targeting ``<work_dir>/<file>`` →
    ``<host_artifacts_dir>/<file>`` for each entry.
    """
    from ..base_interface import RemoteVMConfig
    from .incremental_puller import IncrementalPuller, PullTarget

    if not isinstance(runtime, VmRuntime):
        return None, []
    hot = tuple(getattr(deployer_cls, "hot_artifacts", ()) or ())
    if not hot:
        return None, []

    sep = "/" if runtime.vm_os == "linux" else "\\"
    targets = [
        PullTarget(
            vm_path=f"{runtime.work_dir.rstrip(sep)}{sep}{name}",
            host_path=runtime.host_artifacts_dir / name,
        )
        for name in hot
    ]
    vm_cfg = RemoteVMConfig(
        server_url=runtime.vm_endpoint,
        os_type=runtime.vm_os,
        run_id=run_id,
        task_id=task_id,
    )
    puller = IncrementalPuller(vm_config=vm_cfg, targets=targets)
    return puller, targets


async def _stage_data(
    *,
    env: ALEEnv,
    provider: Provider,
    task_meta: dict[str, Any],
    run_id: str,
    task_id: str,
) -> None:
    """Data-staging block formerly inlined in phase 1.

    Raises ``RuntimeError("Failed to mount...")`` when the VM's data disk
    can't be brought online — caller catches this for mount-fallback.
    Returns silently when the task has no data-staging requirements.
    """
    from ..environments import data_staging
    from ..environments.providers.gcloud import (
        GcloudProvider,
        gcloud_sa_key_path,
    )
    from ..base_interface import RemoteVMConfig

    vm_cfg = RemoteVMConfig(
        server_url=env.vm.endpoint,
        os_type=env.vm.os,
        run_id=run_id,
        task_id=task_id,
    )

    # ensure_data_disk runs for every Linux VM (formats + mounts the data
    # volume) and for every Windows VM (brings E: online); this is what
    # raises "Failed to mount" on certain c4-/hyperdisk capacity profiles
    # where the disk doesn't surface in time.
    await asyncio.to_thread(data_staging.ensure_data_disk, vm_cfg, env.vm.os)

    # Windows-only: force the framebuffer to the expected size before any
    # task-specific GUI setup. has_gpu is inferred from the env spec's gpu
    # field (None → CPU profile → 1024x768; non-None → GPU → 1920x1080).
    if env.vm.os == "windows":
        has_gpu = env.spec.gpu is not None
        await asyncio.to_thread(data_staging.set_windows_resolution, vm_cfg, has_gpu)

    task_data = task_meta.get("task_data")
    if task_data is None or not task_data.requires_task_data:
        return

    sa_key = (
        gcloud_sa_key_path(provider.config)
        if isinstance(provider, GcloudProvider)
        else None
    )
    bucket = _stage_bucket_for(provider)
    await asyncio.to_thread(
        data_staging.stage_input,
        vm_cfg, task_data, env.vm.os,
        gcs_bucket=bucket,
        gcs_local_key_path=sa_key,
    )
    await asyncio.to_thread(
        data_staging.stage_eval, vm_cfg, task_data, env.vm.os,
    )


def _stage_bucket_for(provider: Provider) -> str:
    """Best-effort GCS bucket lookup for task-data staging.

    Until ArtifactsSpec carries a dedicated ``task_data_bucket``, fall back
    to the agenthle convention.
    """
    return "gs://agenthle"


def _results_bucket_for(provider: Provider) -> str:
    """GCS bucket for VM output upload (simprun's ``GCS_RESULTS_BUCKET``)."""
    return "gs://agenthle-run-results"


def _build_run_meta(
    *,
    run_id: str,
    unit: RunUnit,
    config: Any,
    runtime_kind: str,
    status: str,
    score: float | None,
    phase: str | None,
    error_obj: dict[str, Any] | None,
    error_str: str | None,
    total_s: float,
    trajectory: Trajectory | None,
    category: str | None,
) -> dict[str, Any]:
    """Construct the LOG_SPEC §4 run.json payload."""
    usage = (
        trajectory.final_metrics.model_dump()
        if trajectory is not None and trajectory.final_metrics is not None
        else None
    )
    cfg_repr = redact_config(dict(unit.agent_spec.config))
    return {
        "schema_version": 2,
        "run_id": run_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": {
            "id": unit.agent_id,
            "class": unit.agent_spec.class_,
            "name": getattr(config, "name", unit.agent_spec.class_),
            "version": None,
            "model": config.model,
            "runtime": runtime_kind,
            "config_repr": cfg_repr,
        },
        "task": {
            "slug": _task_slug(unit.task_path),
            "path": f"tasks/{unit.task_path}",
            "variant_index": unit.variant_index,
        },
        "status": status,
        "score": score,
        "termination": {
            "reason": status if status != "completed" else "completed",
            "phase": phase,
            "category": category,
            "error": error_obj,
        },
        "timings": {"duration_s": round(total_s, 2)},
        "usage": usage,
    }


def _task_slug(task_path: str) -> str:
    from .slug import slug_task

    return slug_task(task_path)


def _category_from_error(error_str: str | None) -> str | None:
    if not error_str:
        return None
    return classify_error(Exception(error_str))


# Keep an unused signal import — placeholder for future SIGTERM handling.
_ = signal
