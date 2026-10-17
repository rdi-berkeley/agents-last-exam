"""``ale list`` — task discovery under ``tasks/``."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ale_run.cli import main
from ale_run.tasks import list_tasks


def _mk_task(root: Path, rel: str, *, title: str | None = None) -> None:
    d = root / rel
    d.mkdir(parents=True)
    (d / "main.py").write_text("def load():\n    return []\n")
    if title is not None:
        (d / "task_card.json").write_text(
            json.dumps({"taskId": rel, "title": title, "vm": {"snapshot": "cpu-free-ubuntu"}})
        )


def test_list_tasks_finds_main_py_dirs_relative_to_root(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    _mk_task(root, "demo/hello")
    _mk_task(root, "legal/contract_review")
    (root / "common_setup.py").write_text("")  # helper module, not a task
    (root / "legal" / "__init__.py").write_text("")  # namespace marker, not a task
    _mk_task(root, "__pycache__/junk")  # skipped
    _mk_task(root, ".hidden/junk")  # skipped

    assert list_tasks(root) == ["demo/hello", "legal/contract_review"]


def test_list_tasks_missing_root_is_empty(tmp_path: Path) -> None:
    assert list_tasks(tmp_path / "nope") == []


def test_cli_list_prints_tasks(tmp_path: Path, monkeypatch, capsys) -> None:
    _mk_task(tmp_path / "tasks", "demo/hello", title="Hello")
    _mk_task(tmp_path / "tasks", "demo/hello_win")
    monkeypatch.chdir(tmp_path)

    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "discoverable tasks (2):" in out
    assert "  demo/hello\n" in out
    assert "  demo/hello_win\n" in out


def test_cli_list_verbose_includes_card_metadata(tmp_path: Path, monkeypatch, capsys) -> None:
    _mk_task(tmp_path / "tasks", "demo/hello", title="Hello World")
    _mk_task(tmp_path / "tasks", "demo/nocard")  # no task_card.json — must not crash
    monkeypatch.chdir(tmp_path)

    assert main(["list", "--verbose"]) == 0
    out = capsys.readouterr().out
    assert "demo/hello" in out and "cpu-free-ubuntu" in out and "Hello World" in out
    assert "demo/nocard" in out


def test_repo_tasks_are_discoverable() -> None:
    """The shipped task library must include the demo smoke tasks."""
    repo_root = Path(os.environ.get("ALE_PROVIDER_REPO", Path(__file__).resolve().parents[2]))
    tasks = list_tasks(repo_root / "tasks")
    assert "demo/hello" in tasks
    assert "demo/hello_win" in tasks
