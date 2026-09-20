import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.environments.task_data import local_host


def process(returncode=0, stderr=b""):
    return SimpleNamespace(returncode=returncode, communicate=AsyncMock(return_value=(b"", stderr)))


@pytest.mark.asyncio
async def test_copy_assigns_actual_sandbox_user_without_following_links(monkeypatch):
    spawn = AsyncMock(side_effect=[process(), process()])
    monkeypatch.setattr(local_host.asyncio, "create_subprocess_exec", spawn)
    sandbox = SimpleNamespace(id="sandbox", run_command=AsyncMock(side_effect=[
        subprocess.CompletedProcess([], 0, "1000\n", ""),
        subprocess.CompletedProcess([], 0, "1001\n", ""),
    ]))
    await local_host._docker_cp("/host/input/.", sandbox, "/data/input")
    assert spawn.await_args_list[0].args == ("docker", "cp", "/host/input/.", "sandbox:/data/input")
    assert spawn.await_args_list[1].args == (
        "docker", "exec", "--user", "0", "sandbox", "chown", "-R", "-h", "--",
        "1000:1001", "/data/input",
    )


@pytest.mark.asyncio
async def test_failed_copy_does_not_attempt_chown(monkeypatch):
    spawn = AsyncMock(return_value=process(1, b"copy failed"))
    monkeypatch.setattr(local_host.asyncio, "create_subprocess_exec", spawn)
    sandbox = SimpleNamespace(id="sandbox", run_command=AsyncMock())
    with pytest.raises(RuntimeError, match="copy failed"):
        await local_host._docker_cp("/input", sandbox, "/data/input")
    sandbox.run_command.assert_not_awaited()
    assert spawn.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("returncode,value", [(1, "1000"), (0, ""), (0, "user"), (0, "0;id")])
async def test_invalid_owner_fails_closed(monkeypatch, returncode, value):
    spawn = AsyncMock(return_value=process())
    monkeypatch.setattr(local_host.asyncio, "create_subprocess_exec", spawn)
    sandbox = SimpleNamespace(id="sandbox", run_command=AsyncMock(return_value=
        subprocess.CompletedProcess([], returncode, value, "")))
    with pytest.raises(RuntimeError, match="user/group"):
        await local_host._docker_cp("/input", sandbox, "/data/input")
    assert spawn.await_count == 1


@pytest.mark.asyncio
async def test_failed_chown_is_not_silent(monkeypatch):
    spawn = AsyncMock(side_effect=[process(), process(1, b"denied")])
    monkeypatch.setattr(local_host.asyncio, "create_subprocess_exec", spawn)
    sandbox = SimpleNamespace(id="sandbox", run_command=AsyncMock(return_value=
        subprocess.CompletedProcess([], 0, "1000\n", "")))
    with pytest.raises(RuntimeError, match="ownership repair failed"):
        await local_host._docker_cp("/input", sandbox, "/data/input")
