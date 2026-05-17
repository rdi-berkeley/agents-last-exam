"""Per-unit run lifecycle in the Runtime-refactored world.

Pipeline (mirrors the old shape but dispatches install/launch through
EXECUTORS instead of calling deployer.run(env) directly):

  1. resolve_agent(spec)          — pick deployer cls + config + runtime kind
  2. env.reset_async              — task.setup on VM (framework session)
  3. make_runtime(kind, env, ...) — passive AgentRuntime context
  4. EXECUTORS[kind].run_deployer — place + run install + launch
  5. POST-LAUNCH FAN-OUT — three concurrent pipelines:
       (a) origin_log_pipeline — gather work_dir (vm: via ArtifactMirror;
           local/docker: no-op — work_dir is already at run_dir/origin_log/),
           then run deployer_cls.parse_artifacts → builder
       (b) output_pipeline — pull task.metadata['remote_output_dir'] from
           VM to run_dir/output/ via ArtifactMirror (GCS bridge if configured)
       (c) eval_pipeline — env.step_async(Submit()) runs task.evaluate on VM
     All three awaited via asyncio.gather; each pipeline that needs a
     VM session creates a fresh one (cb.DesktopSession is not task-safe).
  6. builder.finalize + RunWriter — disk-side finalization

The 3-tier try/except/finally pattern is preserved so SIGTERM mid-flight
still finalizes run.json / eval_result.json / events.jsonl.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import ale
from ale.agents.base import EpisodeResult
from ale.agents.trajectory import TrajectoryBuilder
from ale.core.provider import Provider
from ale.core.types import Submit
from ale.io import RunWriter, slug_task
from ale.io.artifact_mirror import ArtifactMirror, ArtifactMirrorConfig
from ale.runtime import EXECUTORS, AgentRuntime
from ale.runtime.docker import DockerRuntime
from ale.runtime.local import LocalRuntime
from ale.runtime.vm import VmRuntime

from .factory import resolve_agent
from .spec import ArtifactsSpec, RunUnit, UnitResult

logger = logging.getLogger(__name__)


_SIGNAL_HANDLERS_INSTALLED = False


def install_signal_handlers() -> None:
    """Idempotent. Convert SIGTERM/SIGHUP/SIGINT into KeyboardInterrupt."""
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return

    def _on_signal(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass
    _SIGNAL_HANDLERS_INSTALLED = True


# =============================================================================
# Runtime construction
# =============================================================================

def make_runtime(
    *,
    kind: str,
    config: Any,
    env: Any,
    agent_name: str,
    run_id: str,
    host_origin_dir: Path,
) -> AgentRuntime:
    """Build the AgentRuntime instance for the given kind.

    ``host_origin_dir`` is ``<run_dir>/origin_log/<agent_name>/`` — the
    framework's "this is where this agent's artifacts should land on
    host". For local runtime, this IS the work_dir (no copy needed
    later). For docker runtime, this is the bind-mount source for
    ``/work`` in the container (Phase 4). For vm runtime, this is the
    destination for mirror.pull_dir (Phase 3 — work_dir lives in VM
    fs, gather pulls here).
    """
    vm_endpoint = _vm_endpoint(env)
    vm_os = _vm_os(env)

    if kind == "local":
        # Local runtime: work_dir IS the final origin_log dir. No copy needed.
        host_origin_dir.mkdir(parents=True, exist_ok=True)
        return LocalRuntime(
            work_dir=host_origin_dir,
            vm_endpoint=vm_endpoint,
            vm_os=vm_os,
            config=config,
        )
    if kind == "vm":
        # work_dir on VM under /home/user/.ale/<agent>/<run_id>/
        # (host_origin_dir is the gather destination — VmExecutor pulls into it)
        vm_work_dir = Path(
            f"/home/user/.ale/{agent_name}/{run_id}"
        )
        # Ensure host gather dest exists ahead of post-launch fan-out
        host_origin_dir.mkdir(parents=True, exist_ok=True)
        return VmRuntime(
            work_dir=vm_work_dir,
            vm_endpoint=vm_endpoint,
            vm_os=vm_os,
            config=config,
        )
    if kind == "docker":
        # work_dir is the HOST bind-mount source; container sees it as /work.
        # We use host_origin_dir so artifacts land directly in the run_dir.
        host_origin_dir.mkdir(parents=True, exist_ok=True)
        return DockerRuntime(
            work_dir=host_origin_dir,
            vm_endpoint=vm_endpoint,
            vm_os=vm_os,
            config=config,
        )
    raise ValueError(f"unknown runtime kind: {kind!r}")


# =============================================================================
# Post-launch concurrent pipelines (called via asyncio.gather in run_one_unit)
# =============================================================================

async def _origin_log_pipeline(
    *,
    runtime: AgentRuntime,
    mirror: ArtifactMirror,
    deployer_cls,
    cfg,
    run_result,
    builder: TrajectoryBuilder,
    origin_dest: Path,
    rw,
    slug: str,
) -> None:
    """Gather deployer work_dir → host, then parse_artifacts into builder.

    For local/docker runtimes the work_dir was already created at
    `<run_dir>/origin_log/<agent>/` so the gather is a no-op. For vm
    runtime, mirror.pull_dir does the VM → host copy (GCS bridge if
    configured, cua-direct otherwise).
    """
    # 1. Materialize artifacts on host
    if runtime.kind == "vm":
        # work_dir is a VM path; pull to <run_dir>/origin_log/<agent>/
        session = await runtime.make_vm_session()
        try:
            report = await mirror.pull_dir(
                session, str(runtime.work_dir), f"origin_log/{cfg.name}",
            )
            rw.emit_event("origin_log_gather_done", report=report)
        except Exception as exc:                                # noqa: BLE001
            rw.emit_event("origin_log_gather_failed", error=str(exc))
            raise
    # else: local + docker — work_dir is already at origin_dest on host

    # 2. parse_artifacts (pure-fn classmethod)
    try:
        deployer_cls.parse_artifacts(
            work_dir=origin_dest,
            config=cfg,
            run_result=run_result,
            builder=builder,
        )
    except Exception as parse_exc:                              # noqa: BLE001
        logger.exception("[%s] parse_artifacts threw", slug)
        builder.add_step(
            source="system",
            message=f"parse_artifacts failed: "
                    f"{type(parse_exc).__name__}: {parse_exc}",
            extra={"reason": "parse_error"},
        )


async def _output_pipeline(
    *,
    env,
    runtime: AgentRuntime,
    mirror: ArtifactMirror,
    rw,
    slug: str,
) -> None:
    """Pull task.metadata['remote_output_dir'] → <run_dir>/output/.

    Always on VM regardless of agent runtime — the task wrote there on
    the VM. We use a fresh session (via runtime.make_vm_session) so we
    don't race with env.session (which is being used concurrently by
    env.step(Submit())).
    """
    lt = getattr(env, "_lt", None)
    if lt is None or lt.cb_task is None or not lt.cb_task.metadata:
        rw.emit_event("output_gather_skipped", reason="no_metadata")
        return
    output_dir = lt.cb_task.metadata.get("remote_output_dir")
    if not output_dir:
        rw.emit_event("output_gather_skipped",
                      reason="no_remote_output_dir_in_metadata")
        return
    try:
        session = await runtime.make_vm_session()
        report = await mirror.pull_dir(session, output_dir, "output")
        rw.emit_event("output_gather_done",
                      vm_path=output_dir, report=report)
    except Exception as exc:                                    # noqa: BLE001
        rw.emit_event("output_gather_failed",
                      vm_path=output_dir, error=str(exc))
        # don't raise — output gather is best-effort; eval result is what matters


def _vm_endpoint(env) -> str:
    """Extract ``http://<host>:<port>`` from env.session.computer."""
    sess = env.session
    c = getattr(sess, "computer", None) or getattr(sess, "_computer", None)
    if c is None:
        return "stub://no-vm"
    host = getattr(c, "api_host", None) or "127.0.0.1"
    port = getattr(c, "api_port", None) or 5000
    return f"http://{host}:{port}"


def _vm_os(env) -> str:
    return getattr(env.session, "os_type", "linux") or "linux"


# =============================================================================
# Per-unit run
# =============================================================================

async def run_one_unit(
    *,
    unit: RunUnit,
    provider: Provider,
    output_root: Path,
    artifacts: ArtifactsSpec,
) -> UnitResult:
    """Run one unit end-to-end via runtime-dispatched executors.

    Always returns a UnitResult — never raises. SIGTERM mid-flight still
    finalizes the run dir.
    """
    # 1. Resolve agent: deployer cls + config + runtime kind (validated)
    resolved = resolve_agent(unit.agent_spec)
    deployer_cls = resolved.deployer_cls
    cfg = resolved.config
    runtime_kind = resolved.runtime_kind
    executor = EXECUTORS.get(runtime_kind)
    if executor is None:
        raise RuntimeError(
            f"no Executor registered for runtime kind {runtime_kind!r} "
            f"(registered: {sorted(EXECUTORS)})"
        )

    # 2. Build env
    env = ale.make(unit.task_path, provider=provider)

    # 3. Open RunWriter
    rw = RunWriter.create(
        output_root=output_root,
        agent_name=unit.agent_id,
        model=cfg.model,
        task_path=unit.task_path,
        variant_index=unit.variant_index,
    )
    rw.emit_event(
        "run_started",
        agent=unit.agent_id,
        agent_class=unit.agent_spec.class_,
        model=cfg.model,
        task=unit.task_path,
        variant_index=unit.variant_index,
        runtime=runtime_kind,
    )

    # 4. Build trajectory builder (seeded later with instruction)
    builder = TrajectoryBuilder(
        agent_name=cfg.name,
        agent_version=None,  # filled below after deployer ctor
        model=cfg.model or None,
        task_path=unit.task_path,
        variant_index=unit.variant_index,
    )

    # State that may be partially populated when finalize runs
    t0 = time.monotonic()
    status = "not_executed"
    score: float | None = None
    error: str | None = None
    eval_status = "not_executed"
    eval_duration_s: float | None = None
    eval_error: dict[str, Any] | None = None
    runtime: AgentRuntime | None = None
    run_result = None

    try:
        try:
            # a. env reset (task.setup runs on VM)
            obs = await env.reset_async(variant_index=unit.variant_index)
            instruction = obs.instruction or ""
            builder.trajectory.instruction = instruction
            builder.add_step(source="user", message=instruction)

            # b. Build runtime + record agent.version
            origin_dest = rw.run_dir / "origin_log" / cfg.name
            runtime = make_runtime(
                kind=runtime_kind,
                config=cfg,
                env=env,
                agent_name=cfg.name,
                run_id=rw.run_id,
                host_origin_dir=origin_dest,
            )
            # Construct a transient deployer just to read .version (cheap).
            try:
                transient_deployer = deployer_cls(runtime)
                builder.trajectory.agent.version = transient_deployer.version
            except Exception:                                   # noqa: BLE001
                pass

            # c. install + launch via executor
            rw.emit_event(
                "agent_run_started",
                runtime=runtime_kind, work_dir=str(runtime.work_dir),
            )
            run_result = await executor.run_deployer(
                deployer_cls=deployer_cls,
                runtime=runtime,
                prompt=instruction,
                timeout_s=cfg.timeout_s,
            )
            rw.emit_event(
                "agent_finished",
                status=run_result.status, error=run_result.error,
            )
            status = run_result.status
            error = run_result.error

            # d-f. POST-LAUNCH FAN-OUT — three things run concurrently:
            #   (1) origin_log: gather deployer work_dir from substrate +
            #       parse_artifacts → builder. Sequential within (1) (parse
            #       needs gather), but runs in parallel with (2) and (3).
            #   (2) output:     pull task.remote_output_dir from VM → run_dir/output/
            #   (3) evaluate:   env.step(Submit()) — task.evaluate runs on VM
            #
            # All three need the VM still alive (env.close_async runs in
            # `finally`). To avoid concurrent-RPC race on a single session
            # (cua.DesktopSession is not task-safe), each pipeline that
            # needs a session creates a fresh one. Evaluate uses env.session
            # (framework's existing).
            mirror = ArtifactMirror(ArtifactMirrorConfig(
                local_root=rw.run_dir,
                run_id=rw.run_id,
                gcs_bucket=artifacts.gcs_bucket,
                gcs_local_key_file=artifacts.gcs_local_key_file,
                gcs_vm_key_file=artifacts.gcs_vm_key_file,
                fallback_to_cua=artifacts.fallback_to_cua,
            ))
            rw.emit_event(
                "post_launch_fanout_started",
                gcs_bucket=mirror._cfg.gcs_bucket or "(cua direct)",   # noqa: SLF001
            )

            origin_co = _origin_log_pipeline(
                runtime=runtime, mirror=mirror, deployer_cls=deployer_cls,
                cfg=cfg, run_result=run_result, builder=builder,
                origin_dest=origin_dest, rw=rw, slug=unit.slug,
            )
            output_co = _output_pipeline(
                env=env, runtime=runtime, mirror=mirror, rw=rw, slug=unit.slug,
            )
            eval_co = env.step_async(Submit())

            origin_outcome, output_outcome, eval_outcome = await asyncio.gather(
                origin_co, output_co, eval_co, return_exceptions=True,
            )

            # ---- handle eval outcome ----
            if isinstance(eval_outcome, BaseException):
                if isinstance(eval_outcome, (KeyboardInterrupt, asyncio.CancelledError)):
                    raise eval_outcome
                logger.exception("[%s] evaluate threw", unit.slug,
                                 exc_info=eval_outcome)
                error = error or f"{type(eval_outcome).__name__}: {eval_outcome}"
                eval_status = "failed"
            else:
                final_obs = eval_outcome
                score = final_obs.reward
                eval_status = final_obs.eval_status or "not_executed"
                eval_duration_s = final_obs.eval_duration_s
                eval_error = final_obs.eval_error

            # ---- log origin/output outcomes (already added system steps if failed) ----
            if isinstance(origin_outcome, BaseException) and not isinstance(
                origin_outcome, (KeyboardInterrupt, asyncio.CancelledError)
            ):
                logger.warning("[%s] origin_log pipeline failed: %s",
                               unit.slug, origin_outcome)
            if isinstance(output_outcome, BaseException) and not isinstance(
                output_outcome, (KeyboardInterrupt, asyncio.CancelledError)
            ):
                logger.warning("[%s] output pipeline failed: %s",
                               unit.slug, output_outcome)

        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            status = "cancelled"
            error = f"{type(exc).__name__}: external signal / cancel"
            rw.emit_event("run_cancelled", reason=str(exc) or type(exc).__name__)
            logger.warning("[%s] cancelled by signal", unit.slug)
        except Exception as exc:                                # noqa: BLE001
            status = "failed"
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            rw.emit_event(
                "run_failed",
                error_type=type(exc).__name__, message=str(exc),
            )
            logger.exception("[%s] run threw", unit.slug)
    finally:
        try:
            await env.close_async()
        except Exception as exc:                                # noqa: BLE001
            logger.warning("[%s] env.close_async failed: %s", unit.slug, exc)

    # Final disk writes (wrapped so failure here never raises)
    total_s = time.monotonic() - t0
    trajectory = builder.finalize(
        reward=score,
        status=status if status in ("completed", "timeout", "failed") else "failed",
    )
    try:
        rw.write_trajectory(trajectory)
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("[%s] write_trajectory failed: %s", unit.slug, exc)
    try:
        rw.write_eval_result(
            eval_status=eval_status,
            score=score,
            eval_duration_s=eval_duration_s,
            error=eval_error,
        )
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("[%s] write_eval_result failed: %s", unit.slug, exc)

    try:
        rw.write_run_json(_build_run_json(
            unit=unit,
            cfg=cfg,
            runtime_kind=runtime_kind,
            agent_version=trajectory.agent.version if trajectory else None,
            status=status,
            score=score,
            error=error,
            total_s=total_s,
            trajectory=trajectory,
        ))
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("[%s] write_run_json failed: %s", unit.slug, exc)

    rw.emit_event(
        "run_completed",
        status=status, score=score, total_duration_s=round(total_s, 2),
    )
    rw.close()

    return UnitResult(
        unit=unit,
        status=status,
        score=score,
        eval_status=eval_status,
        duration_s=total_s,
        run_dir=rw.run_dir,
        error=error,
    )


# =============================================================================
# run.json builder
# =============================================================================

def _build_run_json(
    *,
    unit: RunUnit,
    cfg: Any,
    runtime_kind: str,
    agent_version: str | None,
    status: str,
    score: float | None,
    error: str | None,
    total_s: float,
    trajectory: Any,
) -> dict[str, Any]:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": {
            "id": unit.agent_id,
            "class": unit.agent_spec.class_,
            "name": cfg.name,
            "version": agent_version,
            "model": cfg.model,
            "runtime": runtime_kind,
            "config_repr": _safe_repr(unit.agent_spec.config),
        },
        "task": {
            "slug": slug_task(unit.task_path),
            "path": f"tasks/{unit.task_path}",
            "variant_index": unit.variant_index,
        },
        "status": status,
        "score": score,
        "termination": {
            "reason": status if status != "completed" else "completed",
            "error": (
                {"type": "Exception", "message": str(error), "traceback": error}
                if error else None
            ),
        },
        "timings": {"duration_s": round(total_s, 2)},
        "usage": (
            trajectory.final_metrics.model_dump()
            if trajectory is not None and trajectory.final_metrics is not None
            else None
        ),
    }


def _safe_repr(config: dict[str, Any]) -> dict[str, Any]:
    """Drop / redact obvious secrets from the user's agent config before logging."""
    redacted_keys = {
        "anthropic_api_key", "openrouter_api_key", "openai_api_key",
        "brave_api_key", "api_key",
    }
    out = {}
    for k, v in config.items():
        if k.lower() in redacted_keys and isinstance(v, str) and v:
            out[k] = f"***{v[-4:]}" if len(v) >= 4 else "***"
        else:
            out[k] = v
    return out
