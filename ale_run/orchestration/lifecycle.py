"""Per-unit lifecycle: the 4-phase port of simprun's SimpRunTaskRunner.run().

Reads YAML-built ``RunUnit`` + a Provider; produces a ``UnitResult`` and the
LOG_SPEC-shaped on-disk artifacts under ``output_root/<slugs>/v<i>/<ts>/``.

Phase mapping (refined from simprun, surface renamed to LOG_SPEC events):

  - Phase 0  provision      env.reset_async() — single-shot. Task data
                            now ships baked into the image, so there's no
                            mount-fallback to retry.
  - Phase 1  start          _stage_task_data + TaskDriver(session).setup().
                            Single-shot — failures here are task-level, not
                            env-level, so re-provisioning wouldn't help.
  - Phase 2  agent          deployer = deployer_cls(executor); install + launch
  - Phase 2b fanout         pull origin_log/ (remote-only), emit output_gather_skipped
  - Phase 3  evaluate       TaskDriver.evaluate(), then deployer.parse_artifacts
  - Phase 4  cleanup        TaskDriver.close + env.close_async(mode=delete)
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
    BaseExecutor,
    SandboxSpec,
    Provider,
    Trajectory,
    TrajectoryBuilder,
)
from ..environments.env import ALEEnv
from ..executors import DockerExecutor, LocalExecutor, SandboxExecutor
from ..tasks.loader import TaskLoader
from ..tasks.driver import TaskDriver
from .factory import EnvironmentRouter, build_config, resolve_agent
from .run_writer import RunWriter, slug_task
from .experiment_spec import ArtifactsSpec, RunUnit, UnitResult
from .termination import classify_error, err_dict, redact_config

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT_S = 7200
# Wall-clock ceiling for the evaluation phase. Without it, a wedged cua RPC
# inside a task's evaluate() (e.g. a long eval whose result never returns)
# hangs the whole unit until the episode budget — minutes-to-hours of a held
# VM. Bound it so a stuck eval ends cleanly instead. Hitting this bound is
# treated as a unit ``timeout`` (not ``failed``), so resume won't re-run it
# (it would just time out again) — same semantics as an agent wall-clock timeout.
_EVAL_TIMEOUT_S = 7200


def _append_prompt_suffix(task_meta: dict[str, Any], prompt_suffix: str) -> None:
    """Append the experiment-wide ``prompt_suffix`` to the task description.

    No-op when the suffix is empty/whitespace. The suffix is added as its
    own paragraph (blank-line separator) after the task prompt, mutating
    *task_meta* in place so every downstream consumer — the recorded
    trajectory and the prompt handed to the agent — sees the same text.
    """
    if not prompt_suffix or not prompt_suffix.strip():
        return
    description = task_meta.get("description", "") or ""
    task_meta["description"] = f"{description.rstrip()}\n\n{prompt_suffix.strip()}"


# Mount-fallback: how many provision attempts before we give up. Matches
# simprun's single retry — the original env + one alternate profile.


# ======================================================================
# Public entry
# ======================================================================


# Process-wide shutdown event: set by SIGINT/SIGTERM handlers, read by
# per-unit cleanup so signal arrival flips every in-flight unit to
# "cancelled" without losing the events.jsonl tail.
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
    router: "EnvironmentRouter",
    output_root: Path,
    artifacts: ArtifactsSpec | None = None,
    sem: asyncio.Semaphore | None = None,
    cleanup_mode: str = "delete",
    prompt_suffix: str = "",
    wall_time_s: int | None = None,
) -> UnitResult:
    started = time.monotonic()
    effective_cleanup_mode = cleanup_mode
    # `provider` is resolved per unit from the task's snapshot (below), once the
    # task card has been loaded.
    provider: Provider | None = None

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
    # Executor kind: yaml override > deployer's default_executor. resolve_agent
    # has already validated this against the deployer's supported_executors.
    executor_type = (
        unit.agent_spec.executor
        or getattr(deployer_cls, "default_executor", "")
        or "sandbox"
    )

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
        executor=executor_type,
    )

    env: ALEEnv | None = None
    task_driver: TaskDriver | None = None
    builder: TrajectoryBuilder | None = None
    trajectory: Trajectory | None = None
    run_result: AgentRunResult | None = None
    executor: BaseExecutor | None = None

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
        # env count and concurrent agent runs in one number.
        if sem is not None:
            writer.emit_event("provision_wait")
            await sem.acquire()
        try:
            task_path = Path("tasks") / unit.task_path
            task_meta = TaskLoader(str(task_path)).load(unit.variant_index)
            env_spec = _build_env_spec(task_meta, unit=unit)
            timeout_s = int(wall_time_s or task_meta.get("timeout_s") or _DEFAULT_TIMEOUT_S)

            # Resolve the provider for THIS task's snapshot (per-snapshot
            # routing: an environment can mix backends across snapshots).
            provider = router.provider_for(env_spec.snapshot)

            # ============================================================
            # Phase 0 — provision the sandbox. Single-shot: with task data
            # baked into the image (no separate data disk to mount), there's
            # no mount-fallback to retry against. Provisioning failures
            # propagate to the per-unit error handler.
            # ============================================================
            writer.emit_event("provision_started")
            env = ALEEnv(provider=provider, spec=env_spec)
            await env.reset_async()
            writer.emit_event(
                "provision_done",
                env_id=env.sandbox.id,
                machine_type=env.sandbox.metadata.get("machine_type"),
            )

            _append_prompt_suffix(task_meta, prompt_suffix)

            builder = TrajectoryBuilder(
                agent_name=getattr(config, "name", unit.agent_spec.class_),
                agent_version=None,
                model=config.model,
                task_path=unit.task_path,
                variant_index=unit.variant_index,
                instruction=task_meta["description"],
            )
            builder.add_step(source="user", message=task_meta["description"])

            # ============================================================
            # Phase 1 — task-specific setup (single-shot, no retry).
            # Failures here are about the task / its data, not the env —
            # re-provisioning wouldn't help, so we let them propagate.
            # ============================================================
            env.set_phase("stage_inputs")
            await _stage_task_data(
                env=env, provider=provider, artifacts=artifacts, task_meta=task_meta,
            )
            env.set_phase("task_setup")
            task_driver = TaskDriver(
                task_path=str(task_path),
                session=env.session,
                variant=unit.variant_index,
                os_type=env.sandbox.os,
                session_rebuilder=env.reset_session,
            )
            await task_driver.setup()

            # ============================================================
            # Phase 2 — agent
            # ============================================================
            env.set_phase("agent_run")
            agent_name = getattr(config, "name", "agent")
            host_artifacts_dir = writer.run_dir / "origin_log" / agent_name
            host_artifacts_dir.mkdir(parents=True, exist_ok=True)
            executor = _build_executor(
                executor_type=executor_type,
                env=env,
                config=config,
                agent_name=agent_name,
                run_id=writer.run_id,
                host_artifacts_dir=host_artifacts_dir,
            )
            writer.emit_event(
                "agent_run_started",
                executor=executor_type,
                work_dir=executor.work_dir,
            )

            # ─── Hot-artifact incremental tail (sandbox executor only) ───
            # For sandbox runs, work_dir lives on the remote VM; we tail
            # the deployer's declared hot files into host_artifacts_dir so
            # a SIGTERM mid-agent doesn't lose the transcript.
            # Local / docker executors keep work_dir = host_artifacts_dir,
            # so the tail is unnecessary.
            stop_event = asyncio.Event()
            tail_task: asyncio.Task | None = None
            tail_targets: list[tuple[str, Path]] = []
            if isinstance(executor, SandboxExecutor):
                hot = tuple(getattr(deployer_cls, "hot_artifacts", ()) or ())
                if hot:
                    sep = "/" if env.sandbox.is_linux else "\\"
                    tail_targets = [
                        (
                            f"{executor.work_dir.rstrip(sep)}{sep}{name}",
                            host_artifacts_dir / name,
                        )
                        for name in hot
                    ]
                    from ..executors.sandbox import tail_hot_artifacts
                    writer.emit_event(
                        "incremental_pull_started",
                        targets=[t[0] for t in tail_targets],
                    )
                    tail_task = asyncio.create_task(
                        tail_hot_artifacts(
                            executor=executor,
                            targets=tail_targets,
                            stop_event=stop_event,
                        ),
                        name="tail_hot_artifacts",
                    )

            try:
                run_result = await asyncio.wait_for(
                    executor.run_deployer(
                        deployer_cls=deployer_cls,
                        prompt=task_meta["description"],
                        timeout_s=float(timeout_s),
                    ),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                run_result = AgentRunResult(
                    status="timeout",
                    error=f"agent wall-budget exceeded after {timeout_s}s",
                    duration_s=float(timeout_s),
                )
            except asyncio.CancelledError:
                raise
            except Exception as launch_exc:
                import traceback as _tb
                run_result = AgentRunResult(
                    status="failed",
                    error=f"{type(launch_exc).__name__}: {launch_exc}\n"
                          f"{_tb.format_exc()}",
                    duration_s=None,
                )
            finally:
                if tail_task is not None:
                    stop_event.set()
                    try:
                        reconcile_err = await asyncio.wait_for(tail_task, timeout=120)
                    except asyncio.TimeoutError:
                        tail_task.cancel()
                        reconcile_err = "tail reconcile wait timed out"
                    if reconcile_err:
                        writer.emit_event(
                            "incremental_pull_final_failed", error=reconcile_err,
                        )
            writer.emit_event(
                "agent_finished",
                status=run_result.status,
                error=run_result.error,
            )

            # ============================================================
            # Phase 2b — post-launch gather (origin_log) via executor.gather_dir
            # ============================================================
            writer.emit_event("post_launch_fanout_started", gcs_bucket="(cua direct)")
            try:
                gather_report = await executor.gather_dir(
                    src=executor.work_dir, dst=host_artifacts_dir,
                )
                if gather_report.error:
                    writer.emit_event(
                        "origin_log_gather_failed",
                        report={
                            "transport": gather_report.transport,
                            "files": gather_report.files,
                            "error": gather_report.error,
                        },
                    )
                else:
                    writer.emit_event(
                        "origin_log_gather_done",
                        report={
                            "transport": gather_report.transport,
                            "files": gather_report.files,
                            "bytes": gather_report.bytes,
                        },
                    )
            except Exception as e:
                writer.emit_event(
                    "origin_log_gather_failed",
                    report={"transport": "?", "files": 0, "error": str(e)},
                )
            # ============================================================
            # Phase 3 — gather env output, stage reference, evaluate
            # ============================================================
            # 3a. OUTPUT HANDLING — dispatch on artifacts_path.output_path:
            #     null       → skip (output stays on VM, lost on teardown)
            #     "local"    → cua-direct pull → <run_dir>/output/
            #     "gs://..." → vm-side gsutil push → user bucket
            #     Best-effort: failure logs + emits event but doesn't abort.
            await pull_agent_output(
                env=env, provider=provider, artifacts=artifacts, task_meta=task_meta,
                run_id=writer.run_id, task_id=unit.task_path, writer=writer,
                run_dir=writer.run_dir,
            )

            # 3b. STAGING_EVAL (simprun runner.py:_phase3_evaluate second half)
            #     Pull reference data from GCS to the env if the task needs
            #     it for scoring. Best-effort: many tasks don't have a
            #     reference/ prefix and we just log + continue.
            env.set_phase("stage_reference")
            await stage_reference(
                env=env, provider=provider, artifacts=artifacts, task_meta=task_meta,
                run_id=writer.run_id, task_id=unit.task_path, writer=writer,
            )

            # 3c. RUNNING_EVAL. Raw eval output is funnelled into
            #     eval_result.json + run.json + trajectory.json — there is
            #     no debug/ folder in the new spec, so simprun's
            #     `debug/eval/result.json` raw dump has no destination here.
            env.set_phase("evaluation")
            eval_start = time.monotonic()
            try:
                eval_out = await asyncio.wait_for(
                    task_driver.evaluate(), timeout=_EVAL_TIMEOUT_S,
                )
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
            except asyncio.TimeoutError:
                eval_duration_s = round(time.monotonic() - eval_start, 4)
                # Eval ran out of wall-clock — treat as a timeout (not a failure)
                # so the unit status is "timeout" and resume skips it.
                eval_status = "timeout"
                eval_error = {
                    "type": "TimeoutError",
                    "message": f"evaluate() exceeded {_EVAL_TIMEOUT_S}s wall-clock",
                }
                logger.error("evaluate timed out after %ds for %s", _EVAL_TIMEOUT_S, unit.slug)
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
                        work_dir=host_artifacts_dir,
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
            elif eval_status == "timeout":
                status = "timeout"
                error_str = (eval_error or {}).get("message", "evaluation timed out")
                error_obj = eval_error
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
        if task_driver is not None:
            try:
                await task_driver.close()
            except Exception as e:
                logger.debug("TaskDriver.close failed: %s", e)
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
        executor_type=executor_type,
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


def _build_env_spec(task_meta: dict[str, Any], *, unit: RunUnit | None = None) -> SandboxSpec:
    snapshot = task_meta.get("image_category") or task_meta.get("snapshot_name")
    if not snapshot:
        raise RuntimeError(
            "task_card.json is missing snapshot (required for env spec)"
        )
    os_type = task_meta.get("os_type") or "linux"
    task_id = unit.task_path if unit is not None else ""
    harness = unit.agent_spec.class_ if unit is not None else ""
    model_tag = (
        str(unit.agent_spec.config.get("model") or "")
        if unit is not None
        else ""
    )
    return SandboxSpec(
        snapshot=snapshot,
        os=os_type,
        machine_type=task_meta.get("machine_type"),
        gpu=task_meta.get("gpu"),
        task_id=task_id,
        harness=harness,
        model_tag=model_tag,
    )


def _build_executor(
    *,
    executor_type: str,
    env: ALEEnv,
    config: Any,
    agent_name: str,
    run_id: str,
    host_artifacts_dir: Path,
) -> BaseExecutor:
    """Dispatch yaml ``executor: <type>`` to the concrete substrate adapter.

    ``host_artifacts_dir`` is owned by the lifecycle (not the executor); for
    Local/Docker it's also the deployer's work_dir (no gather step). For
    Sandbox the deployer's work_dir is a remote-side path and gather copies
    into ``host_artifacts_dir`` after launch.
    """
    env_passthrough = _collect_env_passthrough()
    if executor_type == "sandbox":
        sb = env.sandbox
        sep = "/" if sb.is_linux else "\\"
        remote_work_dir = f"{sb.work_dir_base.rstrip(sep)}{sep}{agent_name}{sep}{run_id}"
        return SandboxExecutor(
            config=config,
            work_dir=remote_work_dir,
            sandbox=env.sandbox,
            env=env_passthrough,
        )
    if executor_type == "local":
        return LocalExecutor(
            config=config,
            work_dir=str(host_artifacts_dir),
            sandbox=env.sandbox,
            env=env_passthrough,
        )
    if executor_type == "docker":
        # work_dir is the *host* path that will be bind-mounted into the
        # container at /work. DockerExecutor's run_deployer creates it on
        # host, writes spec.json there, then bind-mounts host_work → /work.
        # The in-container LocalExecutor sees work_dir="/work" via the spec.
        return DockerExecutor(
            config=config,
            work_dir=str(host_artifacts_dir),
            sandbox=env.sandbox,
            env=env_passthrough,
        )
    raise NotImplementedError(f"executor type {executor_type!r} not wired in lifecycle")


def _collect_env_passthrough() -> dict[str, str]:
    """Env vars to propagate into the substrate so deployers see API keys.

    Keep this list small — only well-known LLM keys we know deployers need.

    Special handling for ``CURSOR_AUTH_JSON_PATH``: if it points to an
    existing file on the host, the file's content is also passed as
    ``CURSOR_AUTH_JSON`` so the auth.json payload reaches the container
    even when the file path itself isn't accessible from inside the sandbox.
    """
    import os
    from pathlib import Path as _Path

    keys = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "BRAVE_API_KEY",
        "CURSOR_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROK_API_KEY",
        "FACTORY_API_KEY",
        "CURSOR_AUTH_JSON_PATH",
        "CURSOR_AUTH_JSON",
    )
    result = {k: os.environ[k] for k in keys if k in os.environ}

    # Materialise CURSOR_AUTH_JSON from the file when the env var itself
    # is not already set but CURSOR_AUTH_JSON_PATH points to a readable file.
    if "CURSOR_AUTH_JSON" not in result:
        auth_path_str = os.environ.get("CURSOR_AUTH_JSON_PATH", "").strip()
        if auth_path_str:
            auth_path = _Path(auth_path_str)
            if auth_path.is_file():
                try:
                    content = auth_path.read_text(encoding="utf-8")
                    if content.strip():
                        result["CURSOR_AUTH_JSON"] = content
                        logger.info(
                            "env_passthrough: materialised CURSOR_AUTH_JSON "
                            "from CURSOR_AUTH_JSON_PATH (%d B)",
                            len(content),
                        )
                except OSError as e:
                    logger.warning(
                        "env_passthrough: failed to read CURSOR_AUTH_JSON_PATH=%s: %s",
                        auth_path_str, e,
                    )

    return result


async def pull_agent_output(
    *,
    env: ALEEnv,
    provider: Provider,
    artifacts: ArtifactsSpec | None,
    task_meta: dict[str, Any],
    run_id: str,
    task_id: str,
    writer,
    run_dir: Path,
) -> None:
    """Phase 3a output dispatcher.

    Reads :attr:`ArtifactsSpec.output_path` and routes the env's output
    dir to one of three destinations:

      None         → emit ``output_gather_skipped`` reason=unconfigured
      ``"local"``  → pull via cua HTTP → ``<run_dir>/output/``
      ``"gs://X"`` → push from VM to ``X/<run_id>/output/`` via gsutil

    Best-effort throughout: any failure emits an event + logs a warning
    but never aborts the run (eval still runs on the live env regardless).
    """
    task_data = task_meta.get("task_data")
    # Output-pull is gated on the task IDENTITY (domain/task/variant), not on
    # input-data staging: a task can produce output to gather without consuming
    # any staged input (e.g. tool_smoke, REQUIRES_TASK_DATA=False). Coupling
    # this to requires_task_data is what silently dropped tool_report.json.
    if task_data is None or not (
        task_data.domain_name and task_data.task_name and task_data.variant_name
    ):
        writer.emit_event("output_gather_skipped", reason="no_task_identity")
        return
    from ..environments import output_pull

    output_path = artifacts.output_path if artifacts is not None else None

    if output_path is None:
        writer.emit_event("output_gather_skipped", reason="output_path_unconfigured")
        return

    if output_path == "local":
        dest_dir = run_dir / "output"
        try:
            report = await output_pull.pull_to_host(
                env.sandbox, task_data, dest_dir=dest_dir,
            )
            if report.get("skipped"):
                writer.emit_event(
                    "output_gather_skipped",
                    reason=report.get("reason", "unknown"),
                )
            else:
                writer.emit_event(
                    "output_gather_done",
                    transport="cua",
                    vm_path=report.get("vm_path"),
                    files=report.get("files"),
                    bytes=report.get("bytes"),
                    errors=len(report.get("errors") or []),
                )
        except Exception as e:
            logger.warning("pull_to_host failed (best-effort): %s", e)
            writer.emit_event("output_gather_failed", transport="cua", error=str(e))
        return

    # gs:// case
    if not output_path.startswith("gs://"):
        # loader validates this, so reaching here would mean a bypassed path.
        writer.emit_event(
            "output_gather_skipped",
            reason=f"output_path_unrecognised:{output_path!r}",
        )
        return
    try:
        report = await output_pull.push_to_gcs(
            env.sandbox, task_data, run_id=run_id, bucket=output_path,
        )
        writer.emit_event(
            "output_gather_done",
            transport="gcs",
            gcs_path=report.get("gcs_path"),
        )
    except Exception as e:
        logger.warning("push_to_gcs failed (best-effort): %s", e)
        writer.emit_event("output_gather_failed", transport="gcs", error=str(e))


async def stage_reference(
    *,
    env: ALEEnv,
    provider: Provider,
    artifacts: ArtifactsSpec | None,
    task_meta: dict[str, Any],
    run_id: str,
    task_id: str,
    writer,
) -> None:
    """Pull reference data from GCS onto the env for eval (simprun parity).

    Mirrors simprun runner.py:_phase3_evaluate phase ``eval_stage``.
    Best-effort: many tasks don't ship a reference/ prefix.
    """
    task_data = task_meta.get("task_data")
    if task_data is None or not task_data.requires_task_data:
        return
    from ..environments import task_data as task_data_pkg

    source = _task_data_source(artifacts)
    try:
        backend = task_data_pkg.select(source)
        report = await backend.stage_reference(env.sandbox, task_data, source=source)
        if report.get("skipped"):
            writer.emit_event(
                "reference_stage_skipped",
                reason=report.get("reason", "unknown"),
            )
        else:
            writer.emit_event(
                "reference_stage_completed",
                staged=report.get("staged"),
                source=report.get("source"),
            )
    except RuntimeError as e:
        # Reference data is optional — many tasks don't have it.
        logger.info("Reference staging skipped (may not exist): %s", e)
        writer.emit_event("reference_stage_skipped", reason=str(e)[:200])


async def _stage_task_data(
    *,
    env: ALEEnv,
    provider: Provider,
    artifacts: ArtifactsSpec | None,
    task_meta: dict[str, Any],
) -> None:
    """Stage input/software onto the sandbox (Phase 1).

    Dispatches on ``artifacts_path.task_data_source``:
      ``"baked_in_sandbox"``  — image already has data; sanity-check only
      ``"gs://..."``          — gsutil rsync from a GCS bucket
      ``"hf://..."``          — HuggingFace (stub)

    Returns silently when the task declares no data-staging requirements.
    Any failure here is task-level — not wrapped by the mount-fallback
    retry.
    """
    from ..environments import task_data as task_data_pkg

    task_data = task_meta.get("task_data")
    if task_data is None or not task_data.requires_task_data:
        return

    source = _task_data_source(artifacts)
    backend = task_data_pkg.select(source)
    await backend.stage_input(env.sandbox, task_data, source=source)


def _task_data_source(artifacts: ArtifactsSpec | None) -> str:
    """Where to source task data from (yaml ``artifacts_path.task_data_source``).

    One of ``"baked_in_sandbox"`` / ``"gs://<bucket>"`` / ``"hf://<dataset>"``.
    Defaults to the dataclass default when no ArtifactsSpec was built.
    """
    if artifacts is None:
        return ArtifactsSpec().task_data_source
    return artifacts.task_data_source


def _build_run_meta(
    *,
    run_id: str,
    unit: RunUnit,
    config: Any,
    executor_type: str,
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
            "executor": executor_type,
            "config_repr": cfg_repr,
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
            "phase": phase,
            "category": category,
            "error": error_obj,
        },
        "timings": {"duration_s": round(total_s, 2)},
        "usage": usage,
    }


def _category_from_error(error_str: str | None) -> str | None:
    if not error_str:
        return None
    return classify_error(Exception(error_str))
