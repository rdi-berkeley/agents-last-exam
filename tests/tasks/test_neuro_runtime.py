from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys

import pytest
from computer.interface.models import CommandResult

from tasks.psychology_neuro._shared.cognitive_science import neuro_runtime


class _Session:
    def __init__(self, *, object_result: bool = False) -> None:
        self.object_result = object_result
        self.calls: list[tuple[str, float | None, bool]] = []

    async def run_command(
        self, command: str, *, timeout: float | None = None, check: bool = True
    ) -> dict[str, object] | CommandResult:
        self.calls.append((command, timeout, check))
        result = await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
        if self.object_result:
            return CommandResult(result.stdout, result.stderr, result.returncode)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }


class _SessionWithoutTimeout(_Session):
    async def run_command(
        self, command: str, *, check: bool = True
    ) -> dict[str, object] | CommandResult:
        return await super().run_command(command, check=check)


@pytest.mark.parametrize("object_result", [False, True], ids=["dict", "CommandResult"])
@pytest.mark.parametrize("timeout", [None, 5.0])
async def test_run_command_returns_nonzero_for_caller_to_handle(object_result, timeout):
    session = _Session(object_result=object_result)
    command = f"{shlex.quote(sys.executable)} -c 'import sys; print(\"failed\"); sys.exit(1)'"

    result = await neuro_runtime._run_command(session, command, timeout=timeout)

    assert result == {"return_code": 1, "stdout": "failed\n", "stderr": ""}
    assert session.calls == [(command, timeout, False)]


@pytest.mark.parametrize("object_result", [False, True], ids=["dict", "CommandResult"])
async def test_run_command_without_timeout_support_keeps_check_disabled(object_result):
    session = _SessionWithoutTimeout(object_result=object_result)

    result = await neuro_runtime._run_command(session, "exit 1", timeout=5.0)

    assert result["return_code"] == 1
    assert session.calls == [("exit 1", None, False)]


class _DependencySession:
    def __init__(self, *, probe_code: int, install_code: int = 0) -> None:
        self.probe_code = probe_code
        self.install_code = install_code
        self.commands: list[str] = []

    async def run_command(
        self, command: str, *, timeout: float | None = None, check: bool = True
    ) -> dict[str, object]:
        self.commands.append(command)
        installing = command == neuro_runtime.START_DEPS
        code = self.install_code if installing else self.probe_code
        stderr = "package installation failed" if installing and code else ""
        if check and code:
            raise RuntimeError(f"command failed: {code}")
        return {"return_code": code, "stdout": "", "stderr": stderr}


async def test_missing_dependency_triggers_installation():
    session = _DependencySession(probe_code=1)

    await neuro_runtime.ensure_runtime_deps(session)

    assert len(session.commands) == 2
    assert "from PIL import Image" in session.commands[0]
    assert session.commands[1] == neuro_runtime.START_DEPS


async def test_available_dependencies_do_not_trigger_installation():
    session = _DependencySession(probe_code=0)

    await neuro_runtime.ensure_runtime_deps(session)

    assert len(session.commands) == 1
    assert "from PIL import Image" in session.commands[0]


async def test_dependency_install_failure_uses_runtime_error():
    session = _DependencySession(probe_code=1, install_code=2)

    with pytest.raises(
        RuntimeError, match="failed to install runtime deps: package installation failed"
    ):
        await neuro_runtime.ensure_runtime_deps(session)

    assert session.commands[-1] == neuro_runtime.START_DEPS
