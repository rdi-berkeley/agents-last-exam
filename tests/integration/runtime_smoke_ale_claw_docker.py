"""End-to-end smoke: ale_claw × docker runtime × demo/hello on a dev VM.

Exercises Phase 4:
  - resolve_agent picks `docker` runtime (explicit yaml override of
    ale_claw's default `local`)
  - DockerExecutor stages spec.json + env file into <run_dir>/origin_log/ale-claw/
  - `docker run --network host` with bind-mounts of /projects + /work +
    uv cache; container entrypoint:
      cd /projects/agents-last-exam && uv sync --all-packages --quiet
      uv run python -m ale.runtime._docker_entry
  - _docker_entry constructs DockerRuntime + AleClawDeployer in-container,
    runs install + launch (harness drives VM via session built from
    runtime.vm_endpoint, container has --network host so VM IP reachable)
  - work_dir bind-mounted ⇒ artifacts already on host; gather no-op
  - parse_artifacts on host populates ATIF Trajectory
  - env.step(Submit) on framework session evaluates → reward

Pre-flight:
    docker images | grep ale/native-base   # expect: 0.1.0
    curl -X POST http://34.94.212.100:5000/cmd -d ... # VM alive

Run:
    OPENROUTER_API_KEY=... uv run python tests/integration/runtime_smoke_ale_claw_docker.py

First-run cost: ~30-60s for container startup + uv sync. Subsequent runs
~5-10s thanks to ~/.cache/uv mount.
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
logger = logging.getLogger("runtime_smoke_ale_claw_docker")


VM_ENDPOINT = "http://34.94.212.100:5000"
VM_OS = "linux"


async def main() -> int:
    install_signal_handlers()
    # API key from shell env (host) — docker_executor's --env-file
    # propagates it from host os.environ → container.
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY required for this smoke")
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/anthropic/claude-sonnet-4.6")

    provider = StaticProvider(StaticProviderConfig(endpoint=VM_ENDPOINT, os=VM_OS))

    spec = AgentSpec(
        id="aleclaw_docker",
        class_="ale_claw",
        runtime="docker",                       # explicit override (default is "local")
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

    output_root = Path(".logs/runtime_smoke_ale_claw_docker")
    output_root.mkdir(parents=True, exist_ok=True)

    result = await run_one_unit(
        unit=unit,
        provider=provider,
        output_root=output_root,
        artifacts=ArtifactsSpec(),
    )

    logger.info(
        "smoke done: status=%s score=%s eval_status=%s duration=%.1fs  →  %s",
        result.status, result.score, result.eval_status, result.duration_s or 0,
        result.run_dir,
    )

    if result.run_dir:
        origin = result.run_dir / "origin_log" / "ale-claw"
        if origin.exists():
            n = sum(1 for p in origin.rglob("*") if p.is_file())
            logger.info("origin_log: %d files at %s", n, origin)

    return 0 if result.status == "completed" and (result.score or 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
