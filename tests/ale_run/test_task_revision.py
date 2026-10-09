from __future__ import annotations

import json
import os
import shutil
from types import SimpleNamespace

import pytest

from ale_run import cli
from ale_run.orchestration.experiment_spec import AgentSpec, ArtifactsSpec, RunUnit
from ale_run.orchestration.lifecycle import _build_run_meta
from ale_run.tasks.revision import task_revision


@pytest.fixture
def task_folder(tmp_path):
    folder = tmp_path / "tasks" / "demo" / "example"
    folder.mkdir(parents=True)
    (folder / "main.py").write_text("value = 1\n")
    (folder / "scripts").mkdir()
    (folder / "scripts" / "score.py").write_text("score = 1\n")
    return folder


def test_hash_is_portable_and_ignores_timestamps(task_folder, tmp_path):
    expected = task_revision(task_folder)
    copied = tmp_path / "other-checkout"
    shutil.copytree(task_folder, copied)
    os.utime(copied / "main.py", (1, 1))
    assert task_revision(copied) == expected
    assert expected.startswith("sha256:")
    assert len(expected) == 71


@pytest.mark.parametrize("change", ["edit", "add", "delete", "rename", "mode"])
def test_source_tree_changes_invalidate_revision(task_folder, change):
    expected = task_revision(task_folder)
    source = task_folder / "scripts" / "score.py"
    if change == "edit":
        source.write_text("score = 0\n")
    elif change == "add":
        (task_folder / "untracked.py").write_text("new = True\n")
    elif change == "delete":
        source.unlink()
    elif change == "rename":
        source.rename(source.with_name("new_score.py"))
    elif change == "mode":
        source.chmod(source.stat().st_mode ^ 0o111)
    assert task_revision(task_folder) != expected


def test_generated_caches_do_not_change_revision(task_folder):
    expected = task_revision(task_folder)
    cache = task_folder / "scripts" / "__pycache__"
    cache.mkdir()
    (cache / "score.cpython-312.pyc").write_bytes(b"cache")
    (task_folder / ".DS_Store").write_bytes(b"metadata")
    (task_folder / "empty_directory").mkdir()
    assert task_revision(task_folder) == expected


def test_symlinks_hash_the_link_not_an_external_directory(task_folder, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    link = task_folder / "linked"
    link.symlink_to(external, target_is_directory=True)
    expected = task_revision(task_folder)
    (external / "unrelated").write_text("outside task")
    assert task_revision(task_folder) == expected
    link.unlink()
    link.symlink_to("different-target")
    assert task_revision(task_folder) != expected


def test_hash_does_not_require_git(task_folder, monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert task_revision(task_folder).startswith("sha256:")


def test_special_files_are_rejected(task_folder):
    os.mkfifo(task_folder / "pipe")
    with pytest.raises(ValueError, match="Unsupported task file type"):
        task_revision(task_folder)


@pytest.mark.parametrize(
    "status,revision_kind,should_skip",
    [
        ("completed", "current", True),
        ("timeout", "current", True),
        ("failed", "current", False),
        ("cancelled", "current", False),
        ("completed", "old", False),
        ("timeout", "old", False),
        ("completed", "missing", False),
    ],
)
def test_resume_requires_matching_folder_revision(
    task_folder, tmp_path, monkeypatch, status, revision_kind, should_skip,
):
    monkeypatch.chdir(tmp_path)
    unit = RunUnit("codex", AgentSpec("codex", "codex", {"model": "test"}), "demo/example", 0)
    run_dir = tmp_path / "logs" / "codex" / "test" / "demo__example" / "v0" / "run"
    run_dir.mkdir(parents=True)
    task = {}
    if revision_kind != "missing":
        task["revision"] = task_revision(task_folder) if revision_kind == "current" else "old"
    (run_dir / "run.json").write_text(json.dumps({"status": status, "task": task}))
    assert cli._unit_already_done(unit, tmp_path / "logs") is should_skip
    (task_folder / "scripts" / "score.py").write_text("score = 0\n")
    assert not cli._unit_already_done(unit, tmp_path / "logs")


@pytest.mark.parametrize("record", [[], None, {"task": None}, {"status": "completed"}])
def test_resume_rejects_malformed_identity(task_folder, tmp_path, monkeypatch, record):
    monkeypatch.chdir(tmp_path)
    unit = RunUnit("dummy", AgentSpec("dummy", "dummy", {"model": "test"}), "demo/example", 0)
    run_dir = tmp_path / "logs" / "dummy" / "test" / "demo__example" / "v0" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(record))
    assert not cli._unit_already_done(unit, tmp_path / "logs")


def test_resume_hashes_once_per_task_per_scan(task_folder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    units = [
        RunUnit("dummy", AgentSpec("dummy", "dummy", {"model": "test"}), "demo/example", variant)
        for variant in (0, 1)
    ]
    revision = task_revision(task_folder)
    for unit in units:
        run_dir = tmp_path / "logs" / "dummy" / "test" / "demo__example" / f"v{unit.variant_index}" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(json.dumps({
            "status": "completed", "task": {"revision": revision},
        }))
    calls = []

    def counted_hash(directory):
        calls.append(directory)
        return task_revision(directory)

    monkeypatch.setattr(cli, "task_revision", counted_hash)
    assert cli._filter_resume(units, tmp_path / "logs") == []
    assert len(calls) == 1


def test_run_metadata_records_task_and_image_without_data_revision(task_folder):
    unit = RunUnit("codex", AgentSpec("codex", "codex", {"model": "test"}), "demo/example", 0)
    revision = task_revision(task_folder)
    environment = {"provider": "qemu", "image_revision": "immutable-image"}
    record = _build_run_meta(
        run_id="run", unit=unit, config=SimpleNamespace(model="test"),
        executor_type="sandbox", status="completed", score=1.0, phase=None,
        error_obj=None, error_str=None, total_s=2.0, trajectory=None, category=None,
        task_revision_value=revision, environment=environment,
    )
    assert record["task"]["revision"] == revision
    assert record["environment"] == environment
    assert "data_revision" not in record
    assert "commit" not in record["task"]
    assert ArtifactsSpec().output_path is None
