from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from computer.interface.models import CommandResult
from cua_bench.computers.remote import RemoteDesktopSession

from ale_run.tasks import driver
from ale_run.tasks.driver import _detached_run_command


class _Interface:
    def __init__(self, rc: bytes) -> None:
        self.rc = rc
        self.written: dict[str, str] = {}

    async def write_text(self, path: str, content: str) -> None:
        self.written[path] = content

    async def read_bytes(self, path: str) -> bytes:
        if path.endswith("\\out"):
            return b"stdout"
        if path.endswith("\\err"):
            return b"stderr"
        if path.endswith("\\rc"):
            return self.rc
        raise AssertionError(path)


class _RawRunner:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def __call__(self, command: str) -> SimpleNamespace:
        self.commands.append(command)
        stdout = "__DONE__" if "__DONE__" in command else ""
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)


@pytest.mark.asyncio
async def test_windows_detached_command_captures_delayed_errorlevel() -> None:
    interface = _Interface(b"7\r\n")
    raw_runner = _RawRunner()

    result = await _detached_run_command(
        interface, raw_runner, "powershell -File run.ps1", os_type="windows"
    )

    assert result.returncode == 7
    assert result.stdout == "stdout"
    assert result.stderr == "stderr"
    wrapper = next(
        content for path, content in interface.written.items() if path.endswith("wrap.bat")
    )
    assert "EnableDelayedExpansion" in wrapper
    assert 'set "ALE_RC=!ERRORLEVEL!"' in wrapper
    assert "echo !ALE_RC!" in wrapper
    assert any("rmdir /s /q" in command for command in raw_runner.commands)


@pytest.mark.asyncio
@pytest.mark.parametrize("returncode", [0, 1, 7, 127])
@pytest.mark.parametrize("fallback", [False, True])
async def test_resilient_command_preserves_exit_code_through_session(
    monkeypatch: pytest.MonkeyPatch, returncode: int, fallback: bool
) -> None:
    result = CommandResult(stdout="real output", stderr="real diagnostic", returncode=returncode)
    raw_runner = AsyncMock(return_value=result)
    interface = SimpleNamespace(_send_command=AsyncMock(), run_command=raw_runner)
    detached = AsyncMock(return_value=result)
    if fallback:
        detached.side_effect = driver._DetachedSetupError("setup failed")
    monkeypatch.setattr(driver, "_detached_run_command", detached)
    driver.install_resilient_cua_commands(interface)
    session = object.__new__(RemoteDesktopSession)
    session._computer = SimpleNamespace(interface=interface)
    session._initialized = True

    actual = await session.run_command("exit-code-probe", check=False)

    assert actual == {
        "success": returncode == 0,
        "stdout": "real output",
        "stderr": "real diagnostic",
        "return_code": returncode,
    }
    assert result.returncode == returncode
    if returncode:
        with pytest.raises(RuntimeError, match=f"return code {returncode}"):
            await session.run_command("exit-code-probe", check=True)
    else:
        assert await session.run_command("exit-code-probe", check=True) == actual
    assert raw_runner.await_count == (2 if fallback else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError("poll timeout"), RuntimeError("read failure")])
async def test_resilient_command_does_not_repeat_already_started_command(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    raw_runner = AsyncMock()
    interface = SimpleNamespace(_send_command=AsyncMock(), run_command=raw_runner)
    monkeypatch.setattr(driver, "_detached_run_command", AsyncMock(side_effect=error))
    driver.install_resilient_cua_commands(interface)

    with pytest.raises(type(error), match=str(error)):
        await interface.run_command("already-started-command")

    raw_runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_windows_detached_command_rejects_invalid_exit_code() -> None:
    interface = _Interface(b"ECHO is off.\r\n")
    raw_runner = _RawRunner()

    with pytest.raises(RuntimeError, match="invalid exit code"):
        await _detached_run_command(
            interface, raw_runner, "powershell -File run.ps1", os_type="windows"
        )

    assert any("rmdir /s /q" in command for command in raw_runner.commands)


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_real_retained_vm_exit_codes(monkeypatch: pytest.MonkeyPatch, fallback: bool) -> None:
    endpoint = os.environ.get("ALE_COMMAND_TEST_CUA_URL")
    if not endpoint:
        pytest.skip("Requires an explicitly selected retained diagnostic VM")
    session = RemoteDesktopSession(api_url=endpoint, os_type="linux")
    try:
        await session._ensure_computer()
        if fallback:
            monkeypatch.setattr(
                driver,
                "_detached_run_command",
                AsyncMock(side_effect=driver._DetachedSetupError("test direct fallback")),
            )
        driver.install_resilient_cua_commands(session.interface)
        for returncode in (0, 1, 7, 127):
            command = f"bash -c 'printf real-output; printf real-diagnostic >&2; exit {returncode}'"
            result = await session.run_command(command, check=False)
            assert result == {
                "success": returncode == 0,
                "stdout": "real-output",
                "stderr": "real-diagnostic",
                "return_code": returncode,
            }
            if returncode:
                with pytest.raises(RuntimeError, match=f"return code {returncode}"):
                    await session.run_command(command, check=True)
    finally:
        await session.close()
