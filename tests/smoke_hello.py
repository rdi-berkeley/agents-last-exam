"""End-to-end smoke for ``tasks/demo/hello`` via ale.make().

Boots AgenthleEnv with StubProvider, runs the agenthle-format demo task,
verifies reward = 1.0 when answer is correct, 0.0 when missing.

Run from repo root:
    uv run python tests/smoke_hello.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ale
from ale.core.types import Submit, WriteFile
from tests._stubs.provider import StubProvider


async def main() -> int:
    provider = StubProvider()

    # --- Case 1: agent does NOT write the answer → reward 0.0 ---
    env = ale.make("demo/hello", provider=provider)
    obs = await env.reset_async(variant_index=0)
    print("instruction:", obs.instruction[:80] + ("..." if len(obs.instruction) > 80 else ""))
    obs = await env.step_async(Submit())
    print(f"[no write]  reward={obs.reward}  done={obs.done}")
    assert obs.reward == 0.0
    assert obs.done is True
    await env.close_async()

    # --- Case 2: write correct answer, then submit → reward 1.0 ---
    env = ale.make("demo/hello", provider=provider)
    obs = await env.reset_async(variant_index=0)
    answer_path = env._lt.cb_task.metadata["answer_path"]                # noqa: SLF001
    await env.step_async(WriteFile(path=answer_path, data="hello world\n"))
    obs = await env.step_async(Submit())
    print(f"[correct]   reward={obs.reward}  done={obs.done}")
    assert obs.reward == 1.0
    await env.close_async()

    # --- Case 3: write WRONG content → reward 0.0 ---
    env = ale.make("demo/hello", provider=provider)
    obs = await env.reset_async(variant_index=0)
    answer_path = env._lt.cb_task.metadata["answer_path"]                # noqa: SLF001
    await env.step_async(WriteFile(path=answer_path, data="goodbye world\n"))
    obs = await env.step_async(Submit())
    print(f"[wrong]     reward={obs.reward}  done={obs.done}")
    assert obs.reward == 0.0
    await env.close_async()

    print(f"\nregistered envs (auto-discovered): {ale.list_envs()}")
    print("smoke OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
