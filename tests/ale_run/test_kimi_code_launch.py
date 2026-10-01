from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ale_run.agents.kimi_code.config import KimiCodeConfig
from ale_run.agents.kimi_code.deployer import KimiCodeDeployer, _launch_command
from ale_run.agents.kimi_code import deployer as deployer_module


def test_native_launcher_is_unchanged(tmp_path: Path) -> None:
    assert _launch_command("/bin/kimi", "/bin/node", tmp_path) == ["/bin/kimi"]


@pytest.mark.parametrize("bin_value", [{"kimi": "dist/main.mjs"}, "dist/main.mjs"])
def test_windows_launches_node_entry_point(tmp_path: Path, bin_value: object) -> None:
    package = tmp_path / "@moonshot-ai" / "kimi-code"
    entry = package / "dist" / "main.mjs"
    entry.parent.mkdir(parents=True)
    entry.write_text("", encoding="utf-8")
    (package / "package.json").write_text(json.dumps({"bin": bin_value}), encoding="utf-8")

    assert _launch_command("kimi.cmd", "node.exe", tmp_path) == ["node.exe", str(entry)]


@pytest.mark.parametrize(
    "entry", [None, {}, {"other": "main.mjs"}, "../outside.mjs", "missing.mjs"]
)
def test_invalid_windows_entry_fails_closed(tmp_path: Path, entry: object) -> None:
    package = tmp_path / "@moonshot-ai" / "kimi-code"
    package.mkdir(parents=True)
    (package.parent / "outside.mjs").write_text("", encoding="utf-8")
    (package / "package.json").write_text(json.dumps({"bin": entry}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="entry point"):
        _launch_command("kimi.cmd", "node.exe", tmp_path)


def test_launch_preserves_complete_multiline_prompt(tmp_path: Path, monkeypatch) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for actual argv round-trip")
    package = tmp_path / "node_modules" / "@moonshot-ai" / "kimi-code"
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({"bin": {"kimi": "main.mjs"}}))
    (package / "main.mjs").write_text(
        "process.stdout.write(JSON.stringify(process.argv.slice(2)));", encoding="utf-8"
    )
    monkeypatch.setattr("ale_run.agents.kimi_code.deployer._POLL_INTERVAL_S", 0.01)
    monkeypatch.delenv("NODE_OPTIONS", raising=False)
    executor = SimpleNamespace(
        config=KimiCodeConfig(otel_enabled=False),
        work_dir=str(tmp_path / "run"),
        env={"MOONSHOT_API_KEY": "test-key"},
    )
    deployer = KimiCodeDeployer(executor)
    deployer._otel_bootstrap_path = None
    deployer._kimi_command = _launch_command("kimi.cmd", node, tmp_path / "node_modules")
    prompt = (
        "First paragraph.\r\n\r\n"
        'Write "CODE-123" to C:\\task\\output\\result.txt.\n'
        "Do not expand %PATH%, !value!, & or ^.\n"
        "\u4e2d\u6587 instruction in the final paragraph."
    )

    result = asyncio.run(deployer.launch(prompt))

    assert result.status == "completed"
    assert json.loads(Path(result.transcript_path).read_text()) == [
        "--prompt",
        prompt,
        "--output-format",
        "stream-json",
    ]
    assert (Path(executor.work_dir) / "prompt.txt").read_bytes().decode() == prompt


@pytest.mark.parametrize("platform,expected_flags", [("nt", 0x08000000), ("posix", 0)])
def test_headless_launch_does_not_open_windows_console(
    tmp_path, monkeypatch, platform, expected_flags
):
    process = Mock(pid=123, returncode=0)
    process.poll.return_value = 0
    popen = Mock(return_value=process)
    monkeypatch.setattr(deployer_module.subprocess, "Popen", popen)
    monkeypatch.setattr(deployer_module.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(deployer_module, "os", SimpleNamespace(name=platform))
    executor = SimpleNamespace(config=KimiCodeConfig(otel_enabled=False), work_dir=str(tmp_path))
    deployer = KimiCodeDeployer(executor)
    deployer._kimi_command = ["kimi"]
    deployer._otel_bootstrap_path = None
    monkeypatch.setattr(deployer, "_build_env", lambda *args, **kwargs: {})

    result = asyncio.run(deployer.launch("headless smoke"))

    assert result.status == "completed"
    assert popen.call_args.kwargs["creationflags"] == expected_flags
    assert popen.call_args.kwargs["stdin"] == deployer_module.subprocess.DEVNULL
    assert popen.call_args.kwargs["stdout"].name == str(tmp_path / "transcript.jsonl")
