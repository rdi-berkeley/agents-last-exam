"""End-to-end smoke: ale_claw × local runtime × demo/hello on a dev VM.

Exercises the post-Runtime-refactor lifecycle:
  - resolve_agent picks `local` runtime (ale_claw default when no
    `runtime:` in spec)
  - LocalExecutor constructs LocalRuntime + AleClawDeployer in-process
  - deployer.launch builds the session via runtime.make_vm_session()
    against the dev VM, runs OpenClaw harness, writes transcripts
  - parse_artifacts on host populates ATIF Trajectory
  - env.step(Submit) on framework session evaluates → reward
  - run.json / trajectory.json finalized under .logs/

Pre-flight (one-time per session):
    curl -X POST http://34.94.212.100:5000/cmd -H 'content-type: application/json' \\
      -d '{"command":"run_command","params":{"command":"echo ok"}}'

Run:
    OPENROUTER_API_KEY=... uv run python tests/integration/runtime_smoke_ale_claw_local.py

Cost: ~$0.05 (Sonnet 4.6 × ~8-10 turns); dev VM is already running.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ale.providers.static import StaticProvider, StaticProviderConfig
from ale.runner.lifecycle import install_signal_handlers, run_one_unit
from ale.runner.spec import AgentSpec, ArtifactsSpec, RunUnit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
logger = logging.getLogger("runtime_smoke_ale_claw_local")


VM_ENDPOINT = "http://34.94.212.100:5000"
VM_OS = "linux"


async def main() -> int:
    install_signal_handlers()
    # API key from shell env — litellm reads it from os.environ in-process.
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY required for this smoke")
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/anthropic/claude-sonnet-4.6")

    provider = StaticProvider(StaticProviderConfig(endpoint=VM_ENDPOINT, os=VM_OS))

    spec = AgentSpec(
        id="aleclaw_local",
        class_="ale_claw",
        # runtime omitted → default "local" (per validation policy)
        config={
            "model": model,
            "max_turns": 20,
            "timeout_s": 900,
            "disabled_tools": ["web_search"],
        },
    )

    unit = RunUnit(
        agent_id=spec.id,
        agent_spec=spec,
        task_path="demo/hello",
        variant_index=0,
    )

    output_root = Path(".logs/runtime_smoke_ale_claw_local")
    output_root.mkdir(parents=True, exist_ok=True)

    result = await run_one_unit(
        unit=unit,
        provider=provider,
        output_root=output_root,
        artifacts=ArtifactsSpec(),  # default — local copy, no GCS
    )

    logger.info(
        "smoke done: status=%s score=%s eval_status=%s duration=%.1fs  →  %s",
        result.status, result.score, result.eval_status, result.duration_s or 0,
        result.run_dir,
    )

    # Sanity: artifacts on disk
    if result.run_dir:
        origin = result.run_dir / "origin_log" / "ale-claw"
        if origin.exists():
            n = sum(1 for p in origin.rglob("*") if p.is_file())
            logger.info("origin_log: %d files at %s", n, origin)

    return 0 if result.status == "completed" and (result.score or 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
