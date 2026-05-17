"""Per-unit run lifecycle in the Runtime-refactored world.

Pipeline (mirrors the old shape but dispatches install/launch through
EXECUTORS instead of calling deployer.run(env) directly):

  1. resolve_agent(spec)          — pick deployer cls + config + runtime kind
  2. env.reset_async              — task.setup on VM (framework session)
  3. make_runtime(kind, env, ...) — passive AgentRuntime context
  4. EXECUTORS[kind].run_deployer — place + run install + launch
  5. EXECUTORS[kind].gather_to_host — materialize work_dir locally
  6. deployer_cls.parse_artifacts — pure-fn host-side parse → builder
  7. env.step_async(Submit())     — task.evaluate on VM
  8. builder.finalize + RunWriter — disk-side finalization

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
        # Ensure host gather dest exists before executor runs gather_to_host
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


async def _gather_task_output(env, *, dest: Path, rw, slug: str) -> None:
    """Pull task.metadata['remote_output_dir'] from VM → host dest.

    The task's setup() typically writes a path string to its metadata so the
    framework knows where on the VM the eval evaluator will read from. We
    mirror that dir to host for debug + record-keeping. Best-effort: if no
    such metadata or pull fails, we log + skip (the eval already ran on VM).
    """
    lt = getattr(env, "_lt", None)
    if lt is None or lt.cb_task is None or not lt.cb_task.metadata:
        rw.emit_event("task_output_gather_skipped", reason="no_metadata")
        return
    output_dir = lt.cb_task.metadata.get("remote_output_dir")
    if not output_dir:
        rw.emit_event("task_output_gather_skipped",
                      reason="no_remote_output_dir_in_metadata")
        return
    try:
        dest.mkdir(parents=True, exist_ok=True)
        session = env.session
        n = await _pull_vm_dir(session, output_dir, dest)
        rw.emit_event("task_output_gather_done",
                      vm_path=output_dir, host_dest=str(dest), files=n)
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("[%s] task output gather failed: %s", slug, exc)
        rw.emit_event("task_output_gather_failed",
                      vm_path=output_dir, error=str(exc))


async def _pull_vm_dir(session, vm_dir: str, host_dir: Path) -> int:
    """Recursive walk of VM dir → host. Same pattern as VmExecutor's
    _pull_dir_recursive but defined locally to avoid coupling lifecycle to
    a specific runtime kind."""
    count = 0
    try:
        entries = await session.list_dir(vm_dir)
    except Exception:                                           # noqa: BLE001
        return 0
    for name in entries:
        vm_sub = f"{vm_dir}/{name}"
        host_sub = host_dir / name
        try:
            data = await session.read_bytes(vm_sub)
            host_sub.parent.mkdir(parents=True, exist_ok=True)
            host_sub.write_bytes(data)
            count += 1
        except Exception:                                       # noqa: BLE001
            host_sub.mkdir(parents=True, exist_ok=True)
            count += await _pull_vm_dir(session, vm_sub, host_sub)
    return count


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

            # d. gather work_dir → host (vm: pull, docker: bind, local: already there)
            local_work_dir = await executor.gather_to_host(runtime, dest=origin_dest)
            rw.emit_event(
                "artifact_gather_done",
                work_dir=str(local_work_dir),
            )

            # d2. gather task remote_output_dir → <run_dir>/output/
            #     (Agent writes its solution there on the VM; eval reads from
            #     it. We mirror it back to host for inspection/debug. The
            #     task-side path is set by the task's setup() via
            #     task.metadata["remote_output_dir"]. Always VM-side regardless
            #     of runtime kind, so we use env.session directly.)
            await _gather_task_output(env, dest=rw.run_dir / "output", rw=rw,
                                       slug=unit.slug)

            # e. parse_artifacts on host (pure fn)
            try:
                deployer_cls.parse_artifacts(
                    work_dir=local_work_dir,
                    config=cfg,
                    run_result=run_result,
                    builder=builder,
                )
            except Exception as parse_exc:                      # noqa: BLE001
                logger.exception("[%s] parse_artifacts threw", unit.slug)
                builder.add_step(
                    source="system",
                    message=f"parse_artifacts failed: {type(parse_exc).__name__}: {parse_exc}",
                    extra={"reason": "parse_error"},
                )

            # f. evaluate (task.evaluate on VM via framework session)
            final_obs = await env.step_async(Submit())
            score = final_obs.reward
            eval_status = final_obs.eval_status or "not_executed"
            eval_duration_s = final_obs.eval_duration_s
            eval_error = final_obs.eval_error

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
