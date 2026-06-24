from __future__ import annotations

from types import SimpleNamespace

import pytest

from ale_run.base_interface import TaskDataSpec
from ale_run.environments.task_data import baked_in_sandbox


class _Sandbox:
    is_linux = True
    task_data_root = "/data"

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.removed: list[list[str]] = []

    async def exists(self, path: str) -> bool:
        return path.endswith("reference.7z")

    async def rm(self, paths: list[str]) -> None:
        self.removed.append(paths)

    async def mkdir(self, path: str) -> None:
        _ = path

    async def run_command(self, command: str, timeout: int):
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
async def test_stage_reference_reads_password_from_host_env(monkeypatch) -> None:
    sandbox = _Sandbox()
    monkeypatch.setenv("ALE_REFERENCE_ARCHIVE_PASSWORD", "host-only-password")

    report = await baked_in_sandbox.stage_reference(
        sandbox,
        _task_data(),
        source="baked_in_sandbox",
    )

    assert report["staged"] == ["reference"]
    assert "host-only-password" in sandbox.commands[0]


@pytest.mark.asyncio
async def test_stage_reference_requires_password(monkeypatch) -> None:
    sandbox = _Sandbox()
    monkeypatch.delenv("ALE_REFERENCE_ARCHIVE_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="ALE_REFERENCE_ARCHIVE_PASSWORD"):
        await baked_in_sandbox.stage_reference(
            sandbox,
            _task_data(),
            source="baked_in_sandbox",
        )

    assert sandbox.commands == []
