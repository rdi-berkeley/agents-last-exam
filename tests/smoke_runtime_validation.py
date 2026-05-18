"""Unit smoke for runtime validation in :func:`ale.runner.factory.resolve_agent`.

No live VM, no LLM. Tests that:
  - default-pick when single supported runtime (claude_code → vm)
  - default-pick local when multiple + "local" supported (ale_claw → local)
  - explicit yaml runtime rejected when not in supported set
  - explicit yaml runtime accepted when in supported set

Run from repo root:
    uv run python tests/smoke_runtime_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ale.runner.factory import resolve_agent
from ale.runner.spec import AgentSpec


def _spec(class_: str, *, runtime: str | None = None, **cfg) -> AgentSpec:
    return AgentSpec(id="t", class_=class_, config=cfg, runtime=runtime)


def test_default_picks_sole_supported_runtime() -> None:
    r = resolve_agent(_spec("claude_code"))
    assert r.runtime_kind == "vm", f"expected vm, got {r.runtime_kind!r}"
    print("[default-sole] claude_code → vm ok")


def test_default_picks_local_when_multiple_supported() -> None:
    r = resolve_agent(_spec("ale_claw"))
    assert r.runtime_kind == "local", f"expected local, got {r.runtime_kind!r}"
    print("[default-multi] ale_claw → local ok (prefers local over docker)")


def test_explicit_runtime_accepted_when_supported() -> None:
    r = resolve_agent(_spec("ale_claw", runtime="docker"))
    assert r.runtime_kind == "docker", f"expected docker, got {r.runtime_kind!r}"
    print("[explicit-supported] ale_claw + runtime=docker ok")


def test_explicit_runtime_rejected_when_unsupported() -> None:
    try:
        resolve_agent(_spec("ale_claw", runtime="vm"))
    except ValueError as e:
        msg = str(e)
        assert "ale_claw" in msg.lower() or "AleClawDeployer" in msg
        assert "vm" in msg
        assert "local" in msg and "docker" in msg, "error should list allowed runtimes"
        print(f"[rejected] ale_claw + runtime=vm correctly raised: {msg[:90]}")
        return
    raise AssertionError("expected ValueError for ale_claw + runtime=vm")


def test_explicit_runtime_rejected_for_inverse() -> None:
    try:
        resolve_agent(_spec("claude_code", runtime="docker"))
    except ValueError as e:
        msg = str(e)
        assert "claude_code" in msg.lower() or "ClaudeCodeDeployer" in msg
        assert "docker" in msg
        assert "vm" in msg, "error should list 'vm' as the supported runtime"
        print(f"[rejected] claude_code + runtime=docker correctly raised: {msg[:90]}")
        return
    raise AssertionError("expected ValueError for claude_code + runtime=docker")


def test_resolved_agent_carries_deployer_cls_and_config() -> None:
    from ale.agents.ale_claw.config import AleClawConfig
    from ale.agents.ale_claw.deployer import AleClawDeployer
    r = resolve_agent(_spec("ale_claw"))
    assert r.deployer_cls is AleClawDeployer
    assert isinstance(r.config, AleClawConfig)
    assert r.runtime_kind == "local"
    print("[shape] ResolvedAgent fields: (deployer_cls, config, runtime_kind) ok")


def main() -> int:
    test_default_picks_sole_supported_runtime()
    test_default_picks_local_when_multiple_supported()
    test_explicit_runtime_accepted_when_supported()
    test_explicit_runtime_rejected_when_unsupported()
    test_explicit_runtime_rejected_for_inverse()
    test_resolved_agent_carries_deployer_cls_and_config()
    print("\nsmoke OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
