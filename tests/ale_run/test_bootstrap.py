from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import AsyncMock, Mock, call

import pytest

from ale_run.agents import _bootstrap as bootstrap


@pytest.fixture
def npm_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    return tmp_path


@pytest.mark.parametrize("retry", [False, True])
async def test_npm_prefix_succeeds_with_at_most_one_timeout_retry(npm_home, monkeypatch, retry):
    command = f"npm config set prefix '{npm_home}/.npm-global'"
    success = subprocess.CompletedProcess(command, 0, "", "")
    outcomes = [subprocess.TimeoutExpired(command, 60), success] if retry else [success]
    shell = AsyncMock(side_effect=outcomes)
    monkeypatch.setattr(bootstrap, "_sh", shell)

    await bootstrap._configure_npm_prefix()

    assert (npm_home / ".npm-global").is_dir()
    assert shell.await_args_list == [call(command, timeout=60)] * len(outcomes)
    assert bootstrap.os.environ["PATH"] == f"{npm_home}/.npm-global/bin:/usr/bin:/bin"


async def test_npm_prefix_does_not_duplicate_path(npm_home, monkeypatch):
    original_path = f"{npm_home}/.npm-global/bin:/usr/bin:/bin"
    monkeypatch.setenv("PATH", original_path)
    shell = AsyncMock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(bootstrap, "_sh", shell)

    await bootstrap._configure_npm_prefix()

    shell.assert_awaited_once()
    assert bootstrap.os.environ["PATH"] == original_path


async def test_npm_prefix_exhausted_timeouts_preserve_exception_and_path(npm_home, monkeypatch):
    command = f"npm config set prefix '{npm_home}/.npm-global'"
    first = subprocess.TimeoutExpired(command, 60)
    last = subprocess.TimeoutExpired(command, 60, output=b"partial", stderr=b"diagnostic")
    shell = AsyncMock(side_effect=[first, last])
    monkeypatch.setattr(bootstrap, "_sh", shell)

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        await bootstrap._configure_npm_prefix()

    assert raised.value is last
    assert raised.value.output == b"partial"
    assert raised.value.stderr == b"diagnostic"
    assert shell.await_args_list == [call(command, timeout=60)] * 2
    assert bootstrap.os.environ["PATH"] == "/usr/bin:/bin"


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("stderr", ["npm ERR! EACCES", ""])
async def test_npm_prefix_nonzero_exit_is_not_retried(npm_home, monkeypatch, stderr, retry):
    failure = subprocess.CompletedProcess([], 1, "config write failed", stderr)
    outcomes = [subprocess.TimeoutExpired("npm", 60), failure] if retry else [failure]
    shell = AsyncMock(side_effect=outcomes)
    monkeypatch.setattr(bootstrap, "_sh", shell)

    with pytest.raises(RuntimeError, match="npm config set prefix failed.*rc=1") as raised:
        await bootstrap._configure_npm_prefix()

    assert (stderr or failure.stdout) in str(raised.value)
    assert shell.await_count == len(outcomes)
    assert bootstrap.os.environ["PATH"] == "/usr/bin:/bin"


@pytest.mark.parametrize("error", [OSError("cannot launch"), asyncio.CancelledError()])
async def test_npm_prefix_other_errors_propagate_without_retry(npm_home, monkeypatch, error):
    shell = AsyncMock(side_effect=error)
    monkeypatch.setattr(bootstrap, "_sh", shell)

    with pytest.raises(type(error)) as raised:
        await bootstrap._configure_npm_prefix()

    assert raised.value is error
    shell.assert_awaited_once()
    assert bootstrap.os.environ["PATH"] == "/usr/bin:/bin"


async def test_existing_node_bootstrap_passes_larger_timeout_to_subprocess(npm_home, monkeypatch):
    monkeypatch.setattr(bootstrap, "_IS_WINDOWS", False)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: f"/usr/bin/{name}")
    prefix_command = f"npm config set prefix '{npm_home}/.npm-global'"

    def run(command, *, capture_output, text, timeout):
        assert capture_output and text
        if command == ["bash", "-c", "'/usr/bin/node' --version"]:
            assert timeout == 10
            return subprocess.CompletedProcess(command, 0, "v22.0.0\n", "")
        assert command == ["bash", "-c", prefix_command]
        if timeout < 20:
            raise subprocess.TimeoutExpired(command, timeout)
        assert timeout == 60
        return subprocess.CompletedProcess(command, 0, "", "")

    process = Mock(side_effect=run)
    monkeypatch.setattr(bootstrap.subprocess, "run", process)

    assert await bootstrap.ensure_node_npm() == ("/usr/bin/node", "/usr/bin/npm")
    assert process.call_count == 2
    assert bootstrap.os.environ["PATH"] == f"{npm_home}/.npm-global/bin:/usr/bin:/bin"
