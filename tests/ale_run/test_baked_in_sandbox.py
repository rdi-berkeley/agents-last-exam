from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.base_interface import TaskDataSpec
from ale_run.environments.task_data import baked_in_sandbox
from ale_run.orchestration.lifecycle import (
    _collect_env_passthrough,
    stage_reference as lifecycle_stage_reference,
)


class _Sandbox:
    is_linux = True
    task_data_root = "/data"

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.removed: list[list[str]] = []

    async def exists(self, path: str) -> bool:
        return path.endswith("/reference.7z")

    async def rm(self, paths: list[str]) -> None:
        self.removed.append(paths)

    async def mkdir(self, path: str) -> None:
        _ = path

    async def run_command(
        self,
        command: str,
        *,
        timeout: float = 60,
    ) -> SimpleNamespace:
        _ = timeout
        self.commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    async def list_dir(self, path: str) -> list[dict[str, str]]:
        _ = path
        return [{"relpath": "result.txt"}]


def _task_data() -> TaskDataSpec:
    return TaskDataSpec(
        requires_task_data=True,
        domain_name="demo",
        task_name="hello",
        variant_name="base",
    )


@pytest.mark.asyncio
async def test_stage_reference_reads_password_from_host_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _Sandbox()
    monkeypatch.setenv(
        baked_in_sandbox.REFERENCE_ARCHIVE_PASSWORD_ENV,
        "host-only-password",
    )

    report = await baked_in_sandbox.stage_reference(
        sandbox,
        _task_data(),
        source="baked_in_sandbox",
    )

    assert report["staged"] == ["reference"]
    assert "host-only-password" in sandbox.commands[0]


@pytest.mark.asyncio
async def test_stage_reference_requires_host_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _Sandbox()
    monkeypatch.delenv(
        baked_in_sandbox.REFERENCE_ARCHIVE_PASSWORD_ENV,
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match=baked_in_sandbox.REFERENCE_ARCHIVE_PASSWORD_ENV,
    ):
        await baked_in_sandbox.stage_reference(
            sandbox,
            _task_data(),
            source="baked_in_sandbox",
        )

    assert sandbox.commands == []
    assert sandbox.removed == []


@pytest.mark.asyncio
async def test_lifecycle_does_not_suppress_missing_host_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _Sandbox()
    monkeypatch.delenv(
        baked_in_sandbox.REFERENCE_ARCHIVE_PASSWORD_ENV,
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match=baked_in_sandbox.REFERENCE_ARCHIVE_PASSWORD_ENV,
    ):
        await lifecycle_stage_reference(
            env=SimpleNamespace(sandbox=sandbox),
            provider=None,
            artifacts=None,
            task_meta={"task_data": _task_data()},
            run_id="run",
            task_id="demo/hello",
            writer=SimpleNamespace(),
        )


def test_reference_password_is_not_forwarded_to_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "agent-key")
    monkeypatch.setenv(
        baked_in_sandbox.REFERENCE_ARCHIVE_PASSWORD_ENV,
        "host-only-password",
    )

    env = _collect_env_passthrough()

    assert env["OPENAI_API_KEY"] == "agent-key"
    assert baked_in_sandbox.REFERENCE_ARCHIVE_PASSWORD_ENV not in env


def test_reference_password_is_not_embedded_in_shipped_python() -> None:
    repo_root = Path(__file__).parents[2]
    env_example = (repo_root / "secret" / ".env.example").read_text(
        encoding="utf-8",
    )
    prefix = f"{baked_in_sandbox.REFERENCE_ARCHIVE_PASSWORD_ENV}="
    password = next(
        line.removeprefix(prefix) for line in env_example.splitlines() if line.startswith(prefix)
    )

    leaked_paths = [
        path.relative_to(repo_root)
        for path in (repo_root / "ale_run").rglob("*.py")
        if password in path.read_text(encoding="utf-8")
    ]

    assert password
    assert leaked_paths == []
