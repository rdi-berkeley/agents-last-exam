"""Real-GCP end-to-end smoke for ClaudeCodeDeployer × demo/hello.

Boots a fresh VM from a non-prebaked image, runs full pipeline
``acquire → install (runtime) → task.start → agent.run → evaluate → release``,
writes the unified log spec layout under .logs/.

Usage::

    # Linux (image without claude-code pre-installed):
    uv run python tests/integration/gcp_smoke.py linux

    # Windows:
    uv run python tests/integration/gcp_smoke.py windows

Required env:
    GOOGLE_CLOUD_PROJECT       (default: sunblaze-4)
    GCP_ZONE                   (default: us-west1-b)
    OPENROUTER_API_KEY         (or ANTHROPIC_API_KEY)
    OPENROUTER_MODEL           (default: anthropic/claude-sonnet-4-6)

Cost note: real GCP VM ~$0.20/hour + ~$0.05 in tokens for one run.
Cold boot ~3 min + runtime install ~10-25 min + agent run ~1-3 min.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ale
from ale.agents.claude_code import ClaudeCodeConfig, ClaudeCodeDeployer
from ale.io import RunWriter, slug_task
from ale.io.artifact_mirror import ArtifactMirror, ArtifactMirrorConfig
from ale.providers.gcs_direct import GCSDirectConfig, GCSDirectProvider
from ale.providers.static import StaticProvider, StaticProviderConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("gcp_smoke")


# ---- per-OS image + machine knobs ----

PROFILES = {
    "linux": {
        # 2026-05-13: switched to the -0505 image to validate the pipeline.
        # The agents-baked-20260408 image's cua-computer-server.service didn't
        # come up after 10 min — likely predates the systemd config of -0505.
        # Re-test runtime install separately by deleting /usr/local/bin/claude
        # on this image before the smoke runs.
        "image": "agenthle-ubuntu-0505",
        "machine_type": "e2-standard-4",
        "snapshot_label": "cpu-free-ubuntu",
        "boot_disk_gb": 600,    # image is 600GB, boot disk must be >= image size
    },
    "windows": {
        # win10-base-0210 is 64 GB; -0505 is 200 GB. We use the base (no claude)
        # to exercise runtime install. If cua-server doesn't auto-start on this
        # image either, swap to "agenthle-dev-cpu-free-0505" (200 GB) and bump
        # boot_disk_gb to 200.
        "image": "agenthle-win10-base-0210",
        "machine_type": "e2-standard-4",
        "snapshot_label": "cpu-free",
        "boot_disk_gb": 100,
    },
}


def build_config_for_os(os_kind: str) -> ClaudeCodeConfig:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get(
        "OPENROUTER_MODEL",
        "anthropic/claude-sonnet-4-6" if openrouter_key else "claude-sonnet-4-6",
    )
    return ClaudeCodeConfig(
        model=model,
        openrouter_api_key=openrouter_key,
        anthropic_api_key=anthropic_key,
        max_turns=20,
        timeout_s=900.0,
        dangerously_skip_permissions=True,
    )


def build_provider(profile: dict, os_kind: str):
    """Pick provider:

    - ``ALE_STATIC_VM_ENDPOINT`` set → :class:`StaticProvider` (skip gcloud)
    - else                         → :class:`GCSDirectProvider`
    """
    static = os.environ.get("ALE_STATIC_VM_ENDPOINT")
    if static:
        logger.info("static mode: pinning to %s", static)
        return StaticProvider(StaticProviderConfig(
            endpoint=static, os=os_kind,
        ))
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "sunblaze-4")
    zone = os.environ.get("GCP_ZONE", "us-west1-b")
    return GCSDirectProvider(GCSDirectConfig(
        project=project,
        zone=zone,
        machine_type=profile["machine_type"],
        boot_disk_gb=profile["boot_disk_gb"],
        # Map our snapshot tag to the user-specified image.
        images={profile["snapshot_label"]: profile["image"]},
    ))


def _install_signal_handlers() -> None:
    """Convert SIGTERM/SIGHUP into KeyboardInterrupt so our try/finally fires.

    Python's default SIGTERM handler exits immediately without unwinding —
    that would leave a half-written run dir. By raising into the main
    thread instead, we get a chance to finalize run.json / trajectory.json
    / eval_result.json before the process dies. SIGINT already raises
    KeyboardInterrupt natively, but we re-register for symmetry.
    """
    import signal

    def _on_signal(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass  # not all signals available on all platforms


async def run_once(os_kind: str, variant_index: int = 0) -> int:
    _install_signal_handlers()
    if os_kind not in PROFILES:
        raise SystemExit(f"unknown os_kind: {os_kind!r} (expected 'linux' or 'windows')")
    profile = PROFILES[os_kind]

    cfg = build_config_for_os(os_kind)
    if not cfg.anthropic_api_key and not cfg.openrouter_api_key:
        raise SystemExit(
            "Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY before running this smoke."
        )

    output_root = Path(f".logs/gcp_smoke_{os_kind}")
    rw = RunWriter.create(
        output_root=output_root,
        agent_name="claude-code",
        model=cfg.model,
        task_path="demo/hello",
        variant_index=variant_index,
    )
    rw.emit_event(
        "run_started",
        agent="claude-code",
        model=cfg.model,
        task="demo/hello",
        variant_index=variant_index,
        image=profile["image"],
    )

    provider = build_provider(profile, os_kind)
    env = ale.make("demo/hello", provider=provider)
    deployer = ClaudeCodeDeployer(cfg)

    t0 = time.monotonic()
    status = "not_executed"
    error = None
    reward = None
    trajectory = None
    eval_status = "not_executed"
    eval_duration_s: float | None = None
    eval_error: dict | None = None
    mirror_report: dict = {}

    # Three-tier exception handling so that SIGTERM/Ctrl-C ALWAYS produces
    # a finalized run dir (run.json + eval_result.json present, partial
    # trajectory if any). The outer ``finally`` is the persistence barrier.
    try:
        try:
            rw.emit_event("vm_acquire_started", spec={
                "snapshot": profile["snapshot_label"],
                "image": profile["image"],
            })
            result = await deployer.run(env, variant_index=variant_index)
            rw.emit_event("agent_finished",
                          status=result.status, reward=result.reward)
            status = result.status
            error = result.error
            reward = result.reward
            trajectory = result.trajectory
            eval_status = result.eval_status
            eval_duration_s = result.eval_duration_s
            eval_error = result.eval_error

            # Mirror VM-side artifacts BEFORE releasing the VM.
            mirror = ArtifactMirror(ArtifactMirrorConfig.from_env(
                local_root=rw.run_dir, run_id=rw.run_id,
            ))
            rw.emit_event("artifact_mirror_started",
                          gcs_bucket=mirror._cfg.gcs_bucket or "(none, cua direct)")
            mirror_report = await deployer.mirror_artifacts(env, mirror)
            rw.emit_event("artifact_mirror_done", report=mirror_report)
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            status = "cancelled"
            error = f"{type(exc).__name__}: external signal / cancel"
            rw.emit_event("run_cancelled", reason=str(exc) or type(exc).__name__)
            logger.warning("run cancelled by signal")
            # do NOT re-raise — we want the outer finally to finalize
        except Exception as exc:                # noqa: BLE001
            status = "failed"
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            rw.emit_event("run_failed", error_type=type(exc).__name__, message=str(exc))
            logger.exception("smoke run threw")
    finally:
        # Best-effort VM release (also wrapped in try so we always finalize logs)
        try:
            await env.close_async()
        except Exception as exc:                # noqa: BLE001
            logger.warning("env.close_async failed: %s", exc)

    total_s = time.monotonic() - t0
    if trajectory is not None:
        try:
            rw.write_trajectory(trajectory)
        except Exception as exc:                # noqa: BLE001
            logger.warning("write_trajectory failed: %s", exc)
    rw.write_eval_result(
        eval_status=eval_status,
        score=reward,
        eval_duration_s=eval_duration_s,
        error=eval_error,
    )
    rw.write_run_json({
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": {
            "name": "claude-code",
            "version": deployer.version,
            "model": cfg.model,
            "config_repr": {
                "max_turns": cfg.max_turns,
                "timeout_s": cfg.timeout_s,
                "is_openrouter": cfg.is_openrouter,
            },
        },
        "task": {
            "slug": slug_task("demo/hello"),
            "path": "tasks/demo/hello",
            "variant_index": variant_index,
            "os_type": os_kind,
        },
        "env": {
            "provider": "gcs_direct",
            "snapshot": profile["snapshot_label"],
            "image": profile["image"],
        },
        "status": status,
        "score": reward,
        "termination": {
            "reason": status if status != "completed" else "completed",
            "error": (
                {"type": "Exception", "message": str(error), "traceback": error}
                if error else None
            ),
        },
        "timings": {"duration_s": round(total_s, 2)},
    })
    rw.emit_event("run_completed", status=status, score=reward,
                  total_duration_s=round(total_s, 2))
    rw.close()

    logger.info(
        "gcp_smoke %s done: status=%s reward=%s duration=%.1fs  →  %s",
        os_kind, status, reward, total_s, rw.run_dir,
    )
    return 0 if status == "completed" and (reward or 0) > 0 else 1


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("linux", "windows"):
        print(__doc__, file=sys.stderr)
        return 2
    os_kind = sys.argv[1]
    variant = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    return asyncio.run(run_once(os_kind, variant))


if __name__ == "__main__":
    raise SystemExit(main())
