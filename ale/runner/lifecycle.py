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
from ale.core.types import Phase, Submit
from ale.io import RunWriter, slug_task
from ale.io.artifact_mirror import ArtifactMirror, ArtifactMirrorConfig
from ale.io.incremental_pull import (
    DEFAULT_INTERVAL_S,
    IncrementalPuller,
    PullTarget,
    incremental_pull_loop,
)
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
    md = lt.cb_task.metadata
    # Primary key (used by 226+ LinuxTaskConfig-based tasks + most Windows).
    # Fallbacks cover a small tail surveyed in agenthle/tasks/:
    #   - `output_path`        — 10 tasks (CTF/forensics in computing_math)
    #   - `runtime_output_dir` — 2 tasks (transport_safety)
    # Without these fallbacks, eval still scores correctly (eval reads on VM),
    # but our <run_dir>/output/ would be empty for those tasks.
    output_dir = (
        md.get("remote_output_dir")
        or md.get("output_path")
        or md.get("runtime_output_dir")
    )
    if not output_dir:
        rw.emit_event("output_gather_skipped",
                      reason="no_output_dir_in_metadata",
                      checked_keys=["remote_output_dir", "output_path", "runtime_output_dir"])
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
# Best-effort full gather (cancel/fail path)
# =============================================================================

async def _best_effort_full_gather(
    runtime,
    runtime_kind: str,
    env,
    cfg,
    rw,
    artifacts: ArtifactsSpec,
    slug: str,
    *,
    timeout_s: float = 60.0,
) -> None:
    """Tries one full ``mirror.pull_dir`` of the deployer's work_dir.

    Complements the incremental pull (which only got hot files like
    transcript / stderr) by grabbing whatever else the agent dropped in
    work_dir — intermediate scratch files, screenshots, partial outputs.
    Bounded so a dead VM can't pin us; failure is logged only.

    No-op for local/docker — those work_dirs are already host-visible.
    """
    if runtime is None or runtime_kind != "vm":
        return
    try:
        mirror = ArtifactMirror(ArtifactMirrorConfig(
            local_root=rw.run_dir,
            run_id=rw.run_id,
            gcs_bucket=artifacts.gcs_bucket,
            gcs_local_key_file=artifacts.gcs_local_key_file,
            gcs_vm_key_file=artifacts.gcs_vm_key_file,
            fallback_to_cua=artifacts.fallback_to_cua,
        ))
        session = await runtime.make_vm_session()
        rw.emit_event("best_effort_gather_started")
        await asyncio.wait_for(
            mirror.pull_dir(session, str(runtime.work_dir), f"origin_log/{cfg.name}"),
            timeout=timeout_s,
        )
        rw.emit_event("best_effort_gather_done")
    except asyncio.TimeoutError:
        rw.emit_event("best_effort_gather_timeout", timeout_s=timeout_s)
        logger.warning("[%s] best-effort gather timed out (%.0fs)", slug, timeout_s)
    except Exception as exc:                            # noqa: BLE001
        rw.emit_event("best_effort_gather_failed", error=str(exc))
        logger.warning("[%s] best-effort gather failed: %s", slug, exc)


# =============================================================================
# Phase resolver
# =============================================================================

def _resolve_phase(env, lifecycle_phase: Phase) -> Phase:
    """Prefer env's sub-phase (more granular: env_start / stage_inputs /
    task_setup / stage_reference / evaluation / cleanup) when the env was
    active. Falls back to the lifecycle's own coarse tracker (which
    covers agent_run + custodial work outside env.reset/step) when env's
    is ``unknown``.
    """
    env_phase = getattr(env, "current_phase", "unknown")
    if env_phase and env_phase != "unknown":
        return env_phase     # type: ignore[return-value]
    return lifecycle_phase


# =============================================================================
# Error categorization
# =============================================================================

# Substring patterns → short category tag. Checked against ``str(exc)``
# lowercased. First match wins. Coarse on purpose — only the categories
# that drive operator action (provisioning vs API vs auth vs network).
_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rate_limited", ("rate limit", "ratelimit", "429", "too many requests")),
    ("vm_quota_exhausted", ("quota", "stockout", "resource_exhausted",
                            "does not have enough resources", "cpus_per_vm_family")),
    ("auth_failed", ("401", "403", "authentication_failed", "permission denied",
                     "unauthorized", "forbidden", "llm auth failed",
                     "user not found", "invalid api key")),
    ("gcs_missing", ("matched no objects", "no urls matched",
                     "bucketnotfoundexception", "no such object")),
    ("transport_error", ("connection reset", "connection refused", "503",
                         "service unavailable", "deadline exceeded",
                         "broken pipe", "remote end closed connection")),
    ("rpc_timeout", ("timeout", "timed out")),
)


def _classify_error(exc: BaseException | None) -> str | None:
    """Map an exception to a short category tag for ``termination.category``.

    Returns ``None`` for cancels (no category — phase is enough) or
    when no pattern matches (phase already says enough for triage).

    Coarse on purpose: 6 buckets that drive operator action. Don't add
    finer ones here — push finer detail into the original error message.
    """
    if exc is None:
        return None
    if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
        return None
    # TimeoutError before string-match (more specific).
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return "rpc_timeout"
    msg = str(exc).lower()
    if not msg:
        return None
    for cat, patterns in _CATEGORY_PATTERNS:
        if any(p in msg for p in patterns):
            return cat
    return None


# =============================================================================
# Per-unit run
# =============================================================================

async def run_one_unit(
    *,
    unit: RunUnit,
    provider: Provider,
    output_root: Path,
    artifacts: ArtifactsSpec,
    eval_timeout_s: float = 3600.0,
    provision_sem: asyncio.Semaphore | None = None,
    run_sem: asyncio.Semaphore | None = None,
) -> UnitResult:
    """Run one unit end-to-end via runtime-dispatched executors.

    Always returns a UnitResult — never raises. SIGTERM mid-flight still
    finalizes the run dir.

    Concurrency: when ``provision_sem`` / ``run_sem`` are passed, the unit
    holds ``provision_sem`` during ``env.reset_async`` (VM acquire) and
    releases it BEFORE entering ``run_sem`` for launch / fanout / eval.
    When omitted (single-unit smoke), both sems default to no-op (size=1
    so re-entry is fine for a single caller).
    """
    # No-op sems for direct callers (smoke tests, one-off invocations).
    if provision_sem is None:
        provision_sem = asyncio.Semaphore(1)
    if run_sem is None:
        run_sem = asyncio.Semaphore(1)
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

    # 2. Build env (per-task eval_timeout_s injected from ExperimentSpec)
    env = ale.make(
        unit.task_path,
        provider=provider,
        eval_timeout_s=eval_timeout_s,
    )

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
    # Phase tracker — surfaces in run.json.termination.phase so a failure
    # in a 700-task batch can be triaged by which stage broke. The granular
    # phases inside env.reset_async (env_start → stage_inputs → task_setup)
    # all get tagged "env_start" from the lifecycle's POV because they're a
    # single awaitable; finer split would require splitting reset_async,
    # which we intentionally don't (clean OpenEnv contract).
    current_phase: Phase = "unknown"
    error_category: str | None = None

    try:
        try:
            # === provision_sem block === all VM-acquire-side work.
            # env.reset_async does: provider.acquire + cua ready +
            # ensure_data_disk + ensure_gcs_auth + stage_input + stage_eval +
            # task.setup_fn. We hold provision_sem across all of it because
            # the slow parts (gcloud + gsutil) are GCP-quota-bound, same
            # bucket as VM acquire. Released BEFORE run_sem to keep the
            # API pipeline saturated.
            current_phase = "env_start"
            rw.emit_event("provision_wait")
            async with provision_sem:
                rw.emit_event("provision_started")
                obs = await env.reset_async(variant_index=unit.variant_index)
                rw.emit_event("provision_done", vm_id=env.vm.id if env.vm else None)
            # Bind run_id so close_async's upload_output knows the GCS path.
            env.set_run_id(rw.run_id)
            instruction = obs.instruction or ""
            builder.trajectory.instruction = instruction
            builder.add_step(source="user", message=instruction)

            # === run_sem block === agent run + post-launch fanout.
            # The VM stays acquired across the wait — that's the trade-off:
            # slightly idle VMs vs starved API pipeline. With a fat-enough
            # provision pipeline this keeps the run pipeline saturated.
            current_phase = "agent_run"
            rw.emit_event("run_wait")
            async with run_sem:
                rw.emit_event("run_started")

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

                # c. install + launch via executor.
                #
                # For runtime=vm only, start an incremental pull task that
                # tails the deployer's `hot_artifacts` (transcript / stderr)
                # to host disk every 15s. This makes Ctrl-C / VM revert /
                # network blip survivable — at worst we lose the last ~15s
                # of agent output. local/docker work_dirs are already
                # host-visible, so no pull task.
                puller: IncrementalPuller | None = None
                pull_task: asyncio.Task | None = None
                if runtime_kind == "vm" and deployer_cls.hot_artifacts:
                    targets = [
                        PullTarget(
                            remote_path=str(runtime.work_dir) + "/" + name,
                            local_path=origin_dest / name,
                            boundary="newline" if name.endswith(".jsonl") else "none",
                        )
                        for name in deployer_cls.hot_artifacts
                    ]
                    origin_dest.mkdir(parents=True, exist_ok=True)
                    puller = IncrementalPuller(
                        session_factory=runtime.make_vm_session,
                        targets=targets,
                        os_type=env.session.os_type or "linux",
                    )
                    pull_task = asyncio.create_task(
                        incremental_pull_loop(puller, interval_s=DEFAULT_INTERVAL_S),
                    )
                    rw.emit_event(
                        "incremental_pull_started",
                        targets=[t.remote_path for t in targets],
                        interval_s=DEFAULT_INTERVAL_S,
                    )

                rw.emit_event(
                    "agent_run_started",
                    runtime=runtime_kind, work_dir=str(runtime.work_dir),
                )
                try:
                    run_result = await executor.run_deployer(
                        deployer_cls=deployer_cls,
                        runtime=runtime,
                        prompt=instruction,
                        timeout_s=cfg.timeout_s,
                    )
                finally:
                    # Cancel the pull loop; then do ONE final reconcile so
                    # we catch the last bytes the agent flushed between
                    # the previous tick and now. Bounded so a bad VM
                    # can't pin us here.
                    if pull_task is not None:
                        pull_task.cancel()
                        try:
                            await pull_task
                        except (asyncio.CancelledError, Exception):  # noqa: BLE001
                            pass
                    if puller is not None:
                        try:
                            await asyncio.wait_for(
                                puller.reconcile_final(), timeout=60.0,
                            )
                        except Exception as exc:                # noqa: BLE001
                            rw.emit_event(
                                "incremental_pull_final_failed", error=str(exc),
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
            current_phase = _resolve_phase(env, current_phase)
            rw.emit_event(
                "run_cancelled",
                reason=str(exc) or type(exc).__name__,
                phase=current_phase,
            )
            logger.warning("[%s] cancelled by signal in phase=%s", unit.slug, current_phase)
            await _best_effort_full_gather(
                runtime, runtime_kind, env, cfg, rw, artifacts, unit.slug,
            )
        except Exception as exc:                                # noqa: BLE001
            status = "failed"
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            current_phase = _resolve_phase(env, current_phase)
            error_category = _classify_error(exc)
            rw.emit_event(
                "run_failed",
                error_type=type(exc).__name__, message=str(exc),
                phase=current_phase, category=error_category,
            )
            logger.exception(
                "[%s] run threw in phase=%s category=%s",
                unit.slug, current_phase, error_category,
            )
            await _best_effort_full_gather(
                runtime, runtime_kind, env, cfg, rw, artifacts, unit.slug,
            )
    finally:
        # Bounded close: a dead VM (network partition, cua-server crashed)
        # can hang close indefinitely, which would pin the asyncio.gather
        # in Runner._bounded and block batch progress. 60s is plenty for
        # gcloud delete; if it slips, we log + move on (provider-side
        # reconciliation / dangling-VM sweep is a separate concern).
        try:
            await asyncio.wait_for(env.close_async(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] env.close_async exceeded 60s; VM may be dangling — "
                "check provider inventory", unit.slug,
            )
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
            # Use the phase we captured in the except handler — env.current_phase
            # is now "cleanup" because close_async() ran in the finally block.
            phase=current_phase if status != "completed" else None,
            category=error_category if status != "completed" else None,
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
    phase: Phase | None = None,
    category: str | None = None,
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
            # Which lifecycle phase the failure happened in. None on success.
            # Maps to ale.core.types.Phase: env_start | stage_inputs |
            # task_setup | agent_run | stage_reference | evaluation | cleanup
            # | unknown. Use this first for triage in big batches.
            "phase": phase,
            # Coarse error category: rate_limited | vm_quota_exhausted |
            # auth_failed | gcs_missing | transport_error | rpc_timeout |
            # None. Drives operator action (rotate keys, switch zones,
            # check creds, wait + retry). None on success or when no
            # known pattern matched (phase alone is enough).
            "category": category,
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
