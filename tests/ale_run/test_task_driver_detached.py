from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    wrapper = next(content for path, content in interface.written.items() if path.endswith("wrap.bat"))
    assert "EnableDelayedExpansion" in wrapper
    assert 'set "ALE_RC=!ERRORLEVEL!"' in wrapper
    assert "echo !ALE_RC!" in wrapper
    assert any("rmdir /s /q" in command for command in raw_runner.commands)


@pytest.mark.asyncio
async def test_windows_detached_command_rejects_invalid_exit_code() -> None:
    interface = _Interface(b"ECHO is off.\r\n")
    raw_runner = _RawRunner()

    with pytest.raises(RuntimeError, match="invalid exit code"):
        await _detached_run_command(
            interface, raw_runner, "powershell -File run.ps1", os_type="windows"
        )

    assert any("rmdir /s /q" in command for command in raw_runner.commands)
