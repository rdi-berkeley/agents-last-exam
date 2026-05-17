"""End-to-end smoke: claude_code × vm runtime × demo/hello on a dev VM.

Exercises Phase 3:
  - resolve_agent picks `vm` runtime (claude_code's sole supported runtime)
  - VmExecutor scp's ale subtree to /home/user/.ale-src/
  - cua python_exec ships run_deployer_in_vm bootstrap to VM
  - bootstrap constructs VmRuntime + ClaudeCodeDeployer in-VM, awaits
    install + launch (claude CLI runs in VM via setsid + done.marker)
  - VmExecutor.gather_to_host pulls VM work_dir → host origin_log/
  - parse_artifacts on host populates ATIF Trajectory
  - env.step(Submit) on framework session evaluates → reward

Pre-flight:
    curl -X POST http://34.94.212.100:5000/cmd -H 'content-type: application/json' \\
      -d '{"command":"run_command","params":{"command":"echo ok"}}'

Run:
    OPENROUTER_API_KEY=... uv run python tests/integration/runtime_smoke_claude_code_vm.py
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
logger = logging.getLogger("runtime_smoke_claude_code_vm")


VM_ENDPOINT = "http://34.94.212.100:5000"
VM_OS = "linux"


async def main() -> int:
    install_signal_handlers()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY required for this smoke")
    model = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-6")

    provider = StaticProvider(StaticProviderConfig(endpoint=VM_ENDPOINT, os=VM_OS))

    spec = AgentSpec(
        id="cc_vm",
        class_="claude_code",
        # runtime omitted → resolves to "vm" (sole supported)
        config={
            "model": model,
            "openrouter_api_key": key,
            "max_turns": 20,
            "timeout_s": 900,
            "dangerously_skip_permissions": True,
        },
    )

    unit = RunUnit(
        agent_id=spec.id,
        agent_spec=spec,
        task_path="demo/hello",
        variant_index=0,
    )

    output_root = Path(".logs/runtime_smoke_claude_code_vm")
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
        origin = result.run_dir / "origin_log" / "claude-code"
        if origin.exists():
            n = sum(1 for p in origin.rglob("*") if p.is_file())
            logger.info("origin_log: %d files at %s", n, origin)

    return 0 if result.status == "completed" and (result.score or 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
