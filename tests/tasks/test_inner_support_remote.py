import asyncio
from types import SimpleNamespace

import pytest
from computer.interface.models import CommandResult
from cua_bench.computers.remote import RemoteDesktopSession

from tasks.engineering.inner_support_elevation_optimization.remote_commands import run_command
from tasks.engineering.inner_support_elevation_optimization.verification import (
    EvaluationUnavailableError,
)


class LocalInterface:
    async def run_command(self, command):
        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return CommandResult(stdout.decode(), stderr.decode(), process.returncode)


@pytest.fixture
def session():
    actual = RemoteDesktopSession(api_url="http://127.0.0.1:1", os_type="linux")
    actual._computer = SimpleNamespace(interface=LocalInterface())
    actual._initialized = True
    return actual


@pytest.mark.asyncio
async def test_installed_session_keyword_and_exit_code_defects_are_reproduced(session):
    with pytest.raises(TypeError, match="timeout"):
        await session.run_command("true", timeout=1)
    raw = await session.run_command("exit 17", check=False)
    assert raw["return_code"] == 0
    result = await run_command(session, "printf receipt; exit 17", timeout=2, check=False)
    assert result == {"return_code": 17, "stdout": "receipt", "stderr": ""}


@pytest.mark.asyncio
async def test_actual_session_preserves_stdout_stderr_and_raises_checked_failure(session):
    result = await run_command(session, "printf stdout; printf stderr >&2", timeout=2)
    assert result == {"return_code": 0, "stdout": "stdout", "stderr": "stderr"}
    with pytest.raises(EvaluationUnavailableError, match="exited 7"):
        await run_command(session, "exit 7", timeout=2)


@pytest.mark.asyncio
async def test_remote_deadline_is_unavailable_not_success(session):
    with pytest.raises(EvaluationUnavailableError, match="exited 124"):
        await run_command(session, "exec sleep 1", timeout=0.01, check=False)
