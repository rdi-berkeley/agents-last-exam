from __future__ import annotations

import asyncio
from pathlib import Path
import tomllib
from types import SimpleNamespace

import pytest

from ale_run.agents.codex.config import CodexConfig
from ale_run.agents.codex.deployer import CodexDeployer


@pytest.mark.parametrize("is_linux", [True, False])
@pytest.mark.parametrize(
    ("model", "override", "expected"),
    [
        ("openai/gpt-5.6", None, "pixel"),
        ("google/gemini-3.1-pro", None, "normalized"),
        ("google/gemini-3.1-pro", "pixel", "pixel"),
        ("openai/gpt-5.6", "normalized", "normalized"),
    ],
)
def test_codex_coordinate_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    is_linux: bool,
    model: str,
    override: str | None,
    expected: str,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = CodexConfig(provider="direct", model=model, coordinate_space=override)
    executor = SimpleNamespace(
        config=config,
        work_dir=str(tmp_path),
        sandbox=SimpleNamespace(
            is_linux=is_linux,
            node="/usr/bin/node" if is_linux else r"C:\Program Files\nodejs\node.exe",
            mcp_server_dir="/home/user/cua" if is_linux else r"C:\Users\User\cua",
        ),
        cua_bridge_url=lambda: "http://127.0.0.1:8000",
    )
    deployer = CodexDeployer(executor)
    asyncio.run(deployer._write_codex_config(config))
    rendered = tomllib.loads((tmp_path / ".codex/config.toml").read_text())
    bridge = rendered["mcp_servers"]["cua"]
    assert bridge["env"] == {
        "CUA_SERVER_URL": "http://127.0.0.1:8000",
        "CUA_COORDINATE_SPACE": expected,
    }
    assert "\\" not in bridge["command"]
    assert "\\" not in bridge["args"][0]


def test_codex_rejects_unknown_coordinate_space() -> None:
    with pytest.raises(ValueError, match="coordinate_space"):
        CodexConfig(coordinate_space="auto")
