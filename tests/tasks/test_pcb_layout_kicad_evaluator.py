from __future__ import annotations

import pytest

from tasks.engineering.pcb_layout_kicad_1 import main


class _Interface:
    def __init__(self) -> None:
        self.created: list[str] = []

    async def create_dir(self, path: str) -> None:
        self.created.append(path)


class _Session:
    def __init__(self) -> None:
        self.interface = _Interface()
        self.written: dict[str, str] = {}
        self.commands: list[tuple[str, bool]] = []

    async def write_file(self, path: str, content: str) -> None:
        self.written[path] = content

    async def run_command(self, command: str, *, check: bool = True) -> dict[str, object]:
        self.commands.append((command, check))
        return {"return_code": 0, "stdout": "", "stderr": ""}

    async def read_bytes(self, path: str) -> bytes:
        assert path.endswith("drc_report.json")
        return b'{"violations": []}'


@pytest.mark.asyncio
async def test_kicad_drc_uses_staged_powershell_script() -> None:
    session = _Session()
    pcb_path = r"E:\agenthle\engineering\pcb_layout_kicad_1\base\output\board.kicad_pcb"

    drc_json, error = await main._run_kicad_drc(
        {"kicad_cli_path": r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe"},
        pcb_path,
        session,
    )

    assert error is None
    assert drc_json == '{"violations": []}'
    assert len(session.written) == 1
    script_path, script = next(iter(session.written.items()))
    assert script_path.endswith("run_drc.ps1")
    assert "& $cli pcb drc" in script
    assert pcb_path in script
    assert session.commands == [
        (
            f'powershell -NoProfile -ExecutionPolicy Bypass -File "{script_path}"',
            False,
        )
    ]
    assert "$cli" not in session.commands[0][0]
