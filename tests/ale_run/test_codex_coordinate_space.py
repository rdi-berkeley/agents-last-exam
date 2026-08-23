from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.agents.codex.config import CodexConfig
from ale_run.agents.codex.deployer import CodexDeployer


def _executor(tmp_path: Path, config: CodexConfig) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        work_dir=str(tmp_path),
        sandbox=SimpleNamespace(
            is_linux=True,
            node="/usr/bin/node",
            mcp_server_dir="/home/user/cua_mcp_server",
        ),
        cua_bridge_url=lambda: "http://127.0.0.1:5000",
    )


def _render_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: CodexConfig,
) -> str:
    monkeypatch.setenv("HOME", str(tmp_path))
    deployer = CodexDeployer(_executor(tmp_path, config))
    asyncio.run(deployer._write_codex_config(config))
    return (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize("coordinate_space", ["pixel", "normalized"])
def test_codex_writes_explicit_cua_coordinate_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    coordinate_space: str,
) -> None:
    rendered = _render_config(
        tmp_path,
        monkeypatch,
        CodexConfig(provider="direct", coordinate_space=coordinate_space),
    )
    assert 'CUA_SERVER_URL = "http://127.0.0.1:5000"' in rendered
    assert f'CUA_COORDINATE_SPACE = "{coordinate_space}"' in rendered


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-5.6", "pixel"),
        ("google/gemini-3.1-pro", "normalized"),
    ],
)
def test_codex_infers_cua_coordinate_space_from_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    expected: str,
) -> None:
    rendered = _render_config(
        tmp_path,
        monkeypatch,
        CodexConfig(provider="direct", model=model),
    )
    assert f'CUA_COORDINATE_SPACE = "{expected}"' in rendered


def test_codex_rejects_unknown_coordinate_space() -> None:
    with pytest.raises(ValueError, match="coordinate_space"):
        CodexConfig(coordinate_space="auto")
