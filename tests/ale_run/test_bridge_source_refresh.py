from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ale_run.agents import _bootstrap as bootstrap


@pytest.mark.parametrize("what", ["cua", "vm"])
def test_installed_bridge_refreshes_all_source_without_npm(tmp_path, monkeypatch, what):
    source = tmp_path / "source"
    target = tmp_path / "target"
    for root in (source, target):
        (root / "src").mkdir(parents=True)
        (root / "package.json").write_text(json.dumps({"dependencies": {"sdk": "1"}}))
        (root / "package-lock.json").write_text('{"lockfileVersion":3}')
        (root / "src/index.js").write_text("old")
    (target / "node_modules/@modelcontextprotocol/sdk").mkdir(parents=True)
    (source / "src/index.js").write_text("new")
    (source / "package.json").write_text(
        json.dumps(
            {
                "dependencies": {"sdk": "1"},
                "scripts": {"test:unit": "node --test"},
            }
        )
    )
    (source / "src/x11-paste.py").write_text("print('helper')")
    (source / "node_modules").mkdir()
    (source / "node_modules/not-shipped").write_text("not portable")
    (source / "src/__pycache__").mkdir()
    (source / "src/__pycache__/helper.pyc").write_bytes(b"cache")
    npm = AsyncMock(side_effect=AssertionError("npm should not run"))
    monkeypatch.setattr(bootstrap, "ensure_npm", npm)
    result = asyncio.run(bootstrap._ensure_bridge_at(source, str(target), what=what))
    assert result == str(target)
    assert (target / "src/index.js").read_text() == "new"
    assert (target / "src/x11-paste.py").read_text() == "print('helper')"
    assert not (target / "src/__pycache__").exists()
    assert not (target / "node_modules/not-shipped").exists()
    npm.assert_not_called()


@pytest.mark.parametrize("locked", [True, False])
def test_changed_dependencies_require_install(tmp_path, monkeypatch, locked):
    source = tmp_path / "source"
    target = tmp_path / "target"
    for root, version in ((source, "2"), (target, "1")):
        (root / "src").mkdir(parents=True)
        (root / "package.json").write_text(json.dumps({"dependencies": {"sdk": version}}))
        (root / "src/index.js").write_text("source")
        if locked:
            (root / "package-lock.json").write_text(json.dumps({"version": version}))
    (target / "node_modules/@modelcontextprotocol/sdk").mkdir(parents=True)
    npm = AsyncMock(return_value="npm")
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(bootstrap, "ensure_npm", npm)
    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    asyncio.run(bootstrap._ensure_bridge_at(source, str(target), what="cua"))
    npm.assert_awaited_once()
    run.assert_called_once()
    assert run.call_args.args[0] == ["npm", "ci" if locked else "install", "--omit=dev"]


def test_missing_shipped_source_cannot_silently_use_old_bridge(tmp_path):
    with pytest.raises(RuntimeError, match="vendored bridge source missing"):
        asyncio.run(bootstrap._ensure_bridge_at(tmp_path / "missing", str(tmp_path), what="cua"))


@pytest.mark.parametrize(("is_linux", "platform"), [(True, "linux"), (False, "win32")])
def test_bridge_uses_guest_platform_not_host_platform(is_linux, platform):
    executor = SimpleNamespace(
        sandbox=SimpleNamespace(is_linux=is_linux),
        cua_bridge_url=lambda: "http://remote:5000",
    )
    assert bootstrap.cua_bridge_env(executor) == {
        "CUA_SERVER_URL": "http://remote:5000",
        "CUA_SERVER_PLATFORM": platform,
    }
