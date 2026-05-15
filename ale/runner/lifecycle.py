"""Per-unit run lifecycle: signal handling + finalize-on-cancel guarantees.

This is the surgical extraction of the pattern that gcp_smoke.py proved
in the wild. The 3-tier try/except/finally pattern ensures that even on
SIGTERM mid-flight, the run dir gets ``run.json`` / ``eval_result.json``
/ ``events.jsonl`` finalized — only ``trajectory.json`` / mirror dirs
may be partial.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import time
import traceback
from pathlib import Path
from typing import Any

import ale
from ale.core.provider import Provider
from ale.io import RunWriter, slug_task
from ale.io.artifact_mirror import ArtifactMirror, ArtifactMirrorConfig

from .factory import build_agent
from .spec import ArtifactsSpec, RunUnit, UnitResult

logger = logging.getLogger(__name__)


_SIGNAL_HANDLERS_INSTALLED = False


def install_signal_handlers() -> None:
    """Idempotent. Convert SIGTERM/SIGHUP/SIGINT into KeyboardInterrupt.

    Python's default SIGTERM handler exits without running ``finally``;
    that would corrupt our run dirs. Raising into the main thread
    instead lets every in-flight ``run_one_unit`` finalize cleanly.
    """
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return

    def _on_signal(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass                                   # some platforms / non-main threads
    _SIGNAL_HANDLERS_INSTALLED = True


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
    """Run one unit end-to-end. Always returns a UnitResult — never raises.

    Even on SIGTERM/Ctrl-C, the run dir is finalized with the partial
    state we have (status=cancelled, eval_status=not_executed, etc.)
    before returning.
    """
    # 1. Construct deployer + env.
    deployer = build_agent(unit.agent_spec)
    env = ale.make(unit.task_path, provider=provider)

    # 2. Open RunWriter (creates the run dir + events.jsonl).
    rw = RunWriter.create(
        output_root=output_root,
        agent_name=unit.agent_id,                  # user-chosen label, not class
        model=deployer.config.model,
        task_path=unit.task_path,
        variant_index=unit.variant_index,
    )
    rw.emit_event(
        "run_started",
        agent=unit.agent_id,
        agent_class=unit.agent_spec.class_,
        model=deployer.config.model,
        task=unit.task_path,
        variant_index=unit.variant_index,
    )

    # State that may be partially populated when finalize runs.
    t0 = time.monotonic()
    status = "not_executed"
    score: float | None = None
    error: str | None = None
    trajectory = None
    eval_status = "not_executed"
    eval_duration_s: float | None = None
    eval_error: dict[str, Any] | None = None

    try:
        try:
            rw.emit_event("agent_run_started")
            result = await deployer.run(env, variant_index=unit.variant_index)
            rw.emit_event(
                "agent_finished", status=result.status, score=result.reward,
            )
            status = result.status
            score = result.reward
            error = result.error
            trajectory = result.trajectory
            eval_status = result.eval_status
            eval_duration_s = result.eval_duration_s
            eval_error = result.eval_error

            # Mirror VM-side artifacts BEFORE the env releases the VM.
            mirror = ArtifactMirror(ArtifactMirrorConfig(
                local_root=rw.run_dir,
                run_id=rw.run_id,
                gcs_bucket=artifacts.gcs_bucket,
                gcs_local_key_file=artifacts.gcs_local_key_file,
                gcs_vm_key_file=artifacts.gcs_vm_key_file,
                fallback_to_cua=artifacts.fallback_to_cua,
            ))
            rw.emit_event("artifact_mirror_started",
                          gcs_bucket=mirror._cfg.gcs_bucket or "(cua direct)")
            report = await deployer.mirror_artifacts(env, mirror)
            rw.emit_event("artifact_mirror_done", report=report)
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            status = "cancelled"
            error = f"{type(exc).__name__}: external signal / cancel"
            rw.emit_event("run_cancelled", reason=str(exc) or type(exc).__name__)
            logger.warning("[%s] cancelled by signal", unit.slug)
        except Exception as exc:                   # noqa: BLE001
            status = "failed"
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            rw.emit_event("run_failed",
                          error_type=type(exc).__name__, message=str(exc))
            logger.exception("[%s] run threw", unit.slug)
    finally:
        # Always release the VM (best-effort).
        try:
            await env.close_async()
        except Exception as exc:                   # noqa: BLE001
            logger.warning("[%s] env.close_async failed: %s", unit.slug, exc)

    # Final disk writes. Wrapped so a failure here never raises.
    total_s = time.monotonic() - t0
    if trajectory is not None:
        try:
            rw.write_trajectory(trajectory)
        except Exception as exc:                   # noqa: BLE001
            logger.warning("[%s] write_trajectory failed: %s", unit.slug, exc)
    try:
        rw.write_eval_result(
            eval_status=eval_status,
            score=score,
            eval_duration_s=eval_duration_s,
            error=eval_error,
        )
    except Exception as exc:                       # noqa: BLE001
        logger.warning("[%s] write_eval_result failed: %s", unit.slug, exc)

    try:
        rw.write_run_json(_build_run_json(
            unit=unit,
            deployer=deployer,
            status=status,
            score=score,
            error=error,
            total_s=total_s,
            trajectory=trajectory,
        ))
    except Exception as exc:                       # noqa: BLE001
        logger.warning("[%s] write_run_json failed: %s", unit.slug, exc)

    rw.emit_event("run_completed",
                  status=status, score=score, total_duration_s=round(total_s, 2))
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
    deployer: Any,
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
            "name": deployer.config.name,
            "version": getattr(deployer, "version", None),
            "model": deployer.config.model,
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
    redacted_keys = {"anthropic_api_key", "openrouter_api_key", "api_key"}
    out = {}
    for k, v in config.items():
        if k.lower() in redacted_keys and isinstance(v, str) and v:
            out[k] = f"***{v[-4:]}" if len(v) >= 4 else "***"
        else:
            out[k] = v
    return out
