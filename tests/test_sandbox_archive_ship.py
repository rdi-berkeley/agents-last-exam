from __future__ import annotations

import asyncio
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

from ale_run.executors.sandbox import (
    SandboxExecutor,
    _ALE_ARCHIVE_EXTRACT,
    _build_ale_archive,
)


class _FakeSandbox:
    def __init__(self, *, cached: bool = False, upload_failures: int = 0) -> None:
        self.id = "fake-sandbox"
        self.is_linux = True
        self.python = "/usr/bin/python3"
        self.cached = cached
        self.upload_failures = upload_failures
        self.upload_attempts = 0
        self.commands: list[str] = []
        self.writes: list[tuple[str, bytes]] = []

    async def run_command(
        self,
        command: str,
        *,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess:
        self.commands.append(command)
        if len(self.commands) == 1:
            rc = 0 if self.cached else 1
            return subprocess.CompletedProcess(command, rc, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    async def write_file(self, path: str, content: bytes) -> None:
        self.upload_attempts += 1
        if self.upload_attempts <= self.upload_failures:
            raise RuntimeError("temporary upload failure")
        self.writes.append((path, content))


def _source_tree(root: Path) -> None:
    (root / "executors").mkdir(parents=True)
    (root / "executors" / "sandbox.py").write_text("VALUE = 1\n")
    (root / "agents" / "dummy").mkdir(parents=True)
    (root / "agents" / "dummy" / "deployer.py").write_text("VALUE = 2\n")
    (root / "agents" / "dummy" / "pyproject.toml").write_text("[project]\n")
    (root / "agents" / "dummy" / "upstream").mkdir()
    (root / "agents" / "dummy" / "upstream" / "ignored.py").write_text("bad\n")
    (root / "agents" / "_assets" / "cua_mcp_server" / "src").mkdir(parents=True)
    (root / "agents" / "_assets" / "cua_mcp_server" / "src" / "index.js").write_text("ok\n")
    (root / "agents" / "_assets" / "cua_mcp_server" / "package.json").write_text("{}\n")


def test_archive_is_single_deterministic_payload_and_excludes_upstream(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ale_run"
    _source_tree(root)

    first = _build_ale_archive(root)
    _build_ale_archive.cache_clear()
    second = _build_ale_archive(root)

    assert first.payload == second.payload
    assert first.digest == second.digest
    assert first.files == 5
    archive_path = tmp_path / "ale.tar.gz"
    archive_path.write_bytes(first.payload)
    with tarfile.open(archive_path, "r:gz") as tf:
        names = set(tf.getnames())
    assert "ale_run/executors/sandbox.py" in names
    assert "ale_run/agents/dummy/pyproject.toml" in names
    assert not any("upstream" in name for name in names)


def test_ship_uploads_one_archive_or_skips_on_digest_hit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "ale_run"
    _source_tree(root)
    monkeypatch.setattr("ale_run.executors.sandbox._host_ale_root", lambda: root)

    miss = _FakeSandbox()
    executor = SandboxExecutor(
        config=SimpleNamespace(),
        work_dir="/tmp/work",
        sandbox=miss,
        env={},
    )
    asyncio.run(executor._ship_ale_subtree("/home/user/.ale-src"))
    assert len(miss.writes) == 1
    assert miss.writes[0][0].endswith(".tar.gz")
    assert len(miss.commands) == 2

    hit = _FakeSandbox(cached=True)
    executor = SandboxExecutor(
        config=SimpleNamespace(),
        work_dir="/tmp/work",
        sandbox=hit,
        env={},
    )
    asyncio.run(executor._ship_ale_subtree("/home/user/.ale-src"))
    assert hit.writes == []
    assert len(hit.commands) == 1


def test_archive_upload_retries_transient_failure(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "ale_run"
    _source_tree(root)
    monkeypatch.setattr("ale_run.executors.sandbox._host_ale_root", lambda: root)
    monkeypatch.setattr(
        "ale_run.executors.sandbox._ARCHIVE_IO_BACKOFFS_S",
        (0.0, 0.0),
    )

    sandbox = _FakeSandbox(upload_failures=1)
    executor = SandboxExecutor(
        config=SimpleNamespace(),
        work_dir="/tmp/work",
        sandbox=sandbox,
        env={},
    )
    asyncio.run(executor._ship_ale_subtree("/home/user/.ale-src"))

    assert sandbox.upload_attempts == 2
    assert len(sandbox.writes) == 1


def test_remote_extractor_installs_archive_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "ale_run"
    _source_tree(root)
    archive = _build_ale_archive(root)
    archive_path = tmp_path / "payload.tar.gz"
    archive_path.write_bytes(archive.payload)
    dest = tmp_path / ".ale-src"

    first = subprocess.run(
        [sys.executable, "-c", _ALE_ARCHIVE_EXTRACT, str(archive_path), str(dest), archive.digest],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    assert (dest / "ale_run" / "executors" / "sandbox.py").is_file()
    assert (dest / ".archive.sha256").read_text().strip() == archive.digest

    archive_path.write_bytes(archive.payload)
    second = subprocess.run(
        [sys.executable, "-c", _ALE_ARCHIVE_EXTRACT, str(archive_path), str(dest), archive.digest],
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr
    assert not archive_path.exists()
