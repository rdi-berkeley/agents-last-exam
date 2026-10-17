import json
from pathlib import Path
import subprocess

import pytest

from tasks.visual_media.project_migration.scripts import build_batch_helper


def test_missing_installed_api_stops_before_compilation(tmp_path, monkeypatch):
    calls = []

    def native_symbols(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", native_symbols)
    output = tmp_path / "batch.so"
    with pytest.raises(RuntimeError, match="required batch API"):
        build_batch_helper.build(tmp_path, None, tmp_path, output)
    receipt = json.loads(output.with_suffix(".build.json").read_text())
    assert receipt["missing_symbols"] == receipt["required_symbols"]
    assert len(receipt["required_symbols"]) == 6
    assert [command[0] for command in calls] == ["nm"]
    assert not output.exists()


def test_previous_build_evidence_is_never_overwritten(tmp_path):
    output = tmp_path / "batch.so"
    output.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        build_batch_helper.build(tmp_path, None, tmp_path, output)
    assert output.read_bytes() == b"preserved"


def test_all_resolved_symbols_are_in_abi_preflight():
    source = Path(build_batch_helper.__file__).with_name("native_batch_export.cc")
    symbols = build_batch_helper.required_symbols(source)
    assert any("add_export_config" in symbol for symbol in symbols)
    assert any("all_channels_have_ports" in symbol for symbol in symbols)
    assert any("set_state" in symbol for symbol in symbols)
