"""End-to-end smoke for the NativeAgent framework loop (stub agent + ale.make)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ale
from ale.core.types import RunCommand, WriteFile
from tests._stubs.native import StubNativeAgent, StubNativeAgentConfig
from tests._stubs.provider import StubProvider


async def main() -> int:
    provider = StubProvider()

    # Fish out the answer path.
    probe = ale.make("demo/hello", provider=provider)
    await probe.reset_async(variant_index=0)
    answer_path = probe._lt.cb_task.metadata["answer_path"]               # noqa: SLF001
    await probe.close_async()

    # --- Case 1: agent writes correct answer, then signals done ---
    agent = StubNativeAgent(actions_per_turn=[
        [WriteFile(path=answer_path, data="hello world\n")],
        [],
    ])
    env = ale.make("demo/hello", provider=provider)
    result = await agent.run(env, variant_index=0)
    print(
        f"[correct]   reward={result.reward}  status={result.status}  "
        f"steps={len(result.trajectory.steps)}  "
        f"(agent={sum(1 for s in result.trajectory.steps if s.source == 'agent')}, "
        f"env={sum(1 for s in result.trajectory.steps if s.source == 'environment')})"
    )
    assert result.reward == 1.0
    assert result.status == "completed"
    sources = [s.source for s in result.trajectory.steps]
    assert sources[0] == "user"
    assert "agent" in sources and "environment" in sources
    await env.close_async()

    # --- Case 2: parallel actions (multi tool call in one turn) ---
    agent = StubNativeAgent(actions_per_turn=[
        [
            RunCommand(cmd="echo hi"),
            WriteFile(path=answer_path, data="hello world\n"),
        ],
        [],
    ])
    env = ale.make("demo/hello", provider=provider)
    result = await agent.run(env, variant_index=0)
    print(
        f"[parallel]  reward={result.reward}  status={result.status}  "
        f"steps={len(result.trajectory.steps)}"
    )
    assert result.reward == 1.0
    n_env = sum(1 for s in result.trajectory.steps if s.source == "environment")
    assert n_env >= 2, f"expected ≥2 env steps (parallel actions), got {n_env}"
    await env.close_async()

    # --- Case 3: max_turns enforcement ---
    cfg = StubNativeAgentConfig(max_turns=2)
    looper = StubNativeAgent(
        actions_per_turn=[
            [RunCommand(cmd="echo turn1")],
            [RunCommand(cmd="echo turn2")],
            [RunCommand(cmd="echo turn3")],
        ],
        config=cfg,
    )
    env = ale.make("demo/hello", provider=provider)
    result = await looper.run(env, variant_index=0)
    print(f"[max_turns] reward={result.reward}  status={result.status}  error={result.error!r}")
    assert result.status == "timeout"
    assert "max_turns" in (result.error or "")
    await env.close_async()

    print("\nsmoke OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
