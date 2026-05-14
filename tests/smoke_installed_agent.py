"""End-to-end smoke for the InstalledAgent pipeline (stub deployer + ale.make)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ale
from ale.agents.installed.base import InstalledAgent
from tests._stubs.deployer import StubInstalledAgentDeployer
from tests._stubs.provider import StubProvider


async def main() -> int:
    provider = StubProvider()
    answer_path_holder: dict[str, str] = {}

    async def solver_correct(session) -> None:
        await session.write_file(answer_path_holder["path"], "hello world\n")

    async def solver_wrong(session) -> None:
        await session.write_file(answer_path_holder["path"], "goodbye\n")

    # Fish the answer path out via a throwaway env.
    probe = ale.make("demo/hello", provider=provider)
    await probe.reset_async(variant_index=0)
    answer_path_holder["path"] = probe._lt.cb_task.metadata["answer_path"]   # noqa: SLF001
    await probe.close_async()

    # --- Case 1: correct solver → reward 1.0 ---
    deployer = StubInstalledAgentDeployer(solver=solver_correct)
    agent = InstalledAgent(deployer=deployer)
    env = ale.make("demo/hello", provider=provider)
    result = await agent.run(env, variant_index=0)
    print(
        f"[correct]   reward={result.reward}  status={result.status}  "
        f"steps={len(result.trajectory.steps)} "
        f"(install={deployer.install_calls} launch={deployer.launch_calls} "
        f"collect={deployer.collect_calls})"
    )
    assert result.reward == 1.0
    assert result.status == "completed"
    assert [s.source for s in result.trajectory.steps] == ["user", "agent"]
    assert result.trajectory.agent.name == "stub-installed"
    assert result.trajectory.final_metrics.reward == 1.0
    await env.close_async()

    # --- Case 2: wrong solver → reward 0.0 ---
    deployer = StubInstalledAgentDeployer(solver=solver_wrong)
    agent = InstalledAgent(deployer=deployer)
    env = ale.make("demo/hello", provider=provider)
    result = await agent.run(env, variant_index=0)
    print(f"[wrong]     reward={result.reward}  status={result.status}")
    assert result.reward == 0.0
    await env.close_async()

    print("\nsmoke OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
