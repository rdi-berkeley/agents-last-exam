from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.base_interface import TaskDataSpec
from ale_run.environments import output_pull
from ale_run.environments.task_data import (
    gsbucket,
    local_host,
    ossbucket,
    prepare_task_subdir,
    s3bucket,
    task_subdir,
)
from ale_run.orchestration.lifecycle import _stage_task_data, stage_reference
from ale_run.tasks.driver import TaskDriver

CANONICAL = "fixture_domain/canonical_answer_task/base"
VISIBLE = "fixture_domain/neutral_workspace/base"
ANSWER = {"value": "synthetic answer"}
REFERENCE = {"answer": ANSWER}


class FilesystemSandbox:
    is_linux = True
    python = sys.executable
    id = "simulated-sandbox"

    def __init__(self, root):
        self.task_data_root = str(root)
        self.metadata = {"provider": "docker"}
        self.interface = self
        self.commands = []
        self.transfers = []
        self.archive_source = None

    async def run_command(self, command, *, timeout=60):
        self.commands.append(command)
        argv = shlex.split(command)
        if argv[:2] == ["gsutil", "cp"]:
            source, destination = argv[2:]
            assert source == self.archive_source[0]
            self.transfers.append((source, destination))
            shutil.copyfile(self.archive_source[1], destination)
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)

    async def exists(self, path):
        return Path(path).exists()

    async def file_exists(self, path):
        return Path(path).is_file()

    async def mkdir(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    async def create_dir(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    async def rm(self, paths):
        for path in map(Path, paths):
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)

    async def read_bytes(self, path):
        return Path(path).read_bytes()

    async def list_dir(self, path):
        return [
            {"relpath": str(entry.relative_to(path)), "is_dir": entry.is_dir()}
            for entry in Path(path).rglob("*")
        ]

    async def download_to_local(self, source, destination, *, timeout=120):
        shutil.copyfile(source, destination)
        return True


def make_source(root):
    base = root / CANONICAL
    (base / "input/runtime_env").mkdir(parents=True)
    (base / "input/input.txt").write_text("synthetic input\n")
    (base / "input/settings.json").write_text("{}")
    (base / "input/runtime_env/pyproject.toml").write_text('[project]\nname = "fixture-env"\n')
    (base / "reference/metadata").mkdir(parents=True)
    (base / "reference/metadata/verification_targets.json").write_text(json.dumps(REFERENCE))
    return base


@pytest.mark.parametrize(
    "output_name",
    ["output", "output_test_pos", "output_test_neg", "output_admin_pos", "output_admin_neg"],
)
@pytest.mark.parametrize(
    "source_kind,prebaked",
    [
        ("baked", True),
        ("local", False),
        ("local", True),
        ("gs", False),
        ("gs", True),
        ("gs_release", True),
        ("gs_encrypted", False),
        ("s3", False),
        ("s3", True),
        ("oss", False),
        ("oss", True),
    ],
)
async def test_aliased_staging_driver_evaluation_and_collection(
    tmp_path, monkeypatch, aliased_task_root, source_kind, prebaked, output_name
):
    source_root = tmp_path / "source"
    source_base = make_source(source_root)
    sandbox = FilesystemSandbox(tmp_path / "guest data")
    canonical = Path(sandbox.task_data_root) / CANONICAL
    visible = Path(sandbox.task_data_root) / VISIBLE
    sibling = canonical.parent.parent / "unrelated_task/base/input.txt"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("unrelated")
    if prebaked:
        shutil.copytree(source_base / "input", canonical / "input")
        original_inode = (canonical / "input/input.txt").stat().st_ino
    if source_kind in {"baked", "gs_encrypted"}:
        archive = source_base / "reference.7z"
        subprocess.run(
            ["7z", "a", "-ptest-only-password", "-mhe=on", str(archive), "."],
            cwd=source_base / "reference", check=True, capture_output=True,
        )
        monkeypatch.setenv("ALE_REFERENCE_ARCHIVE_PASSWORD", "test-only-password")
        if source_kind == "baked":
            shutil.copyfile(archive, canonical / "reference.7z")

    if source_kind == "local":
        source = f"local:{source_root}"

        async def copy_from_host(source_path, target_sandbox, destination):
            assert target_sandbox is sandbox
            assert Path(source_path).is_relative_to(source_base)
            sandbox.transfers.append((source_path, destination))
            shutil.copytree(source_path, destination, dirs_exist_ok=True)

        monkeypatch.setattr(local_host, "_docker_cp", copy_from_host)
    elif source_kind == "baked":
        source = "baked_in_sandbox"
    else:
        scheme = "gs" if source_kind.startswith("gs") else source_kind
        source = f"{scheme}://fixture"
        if source_kind == "gs_release":
            source += "/v1.1"
        prefix = f"{source}/{CANONICAL}/"
        backend = {"gs": gsbucket, "s3": s3bucket, "oss": ossbucket}[scheme]

        def source_path(url):
            assert url.startswith(prefix)
            return source_base / url.removeprefix(prefix)

        async def source_exists(target_sandbox, url):
            assert target_sandbox is sandbox
            return source_path(url).exists()

        def sync_command(target_sandbox, url, destination):
            assert target_sandbox is sandbox
            sandbox.transfers.append((url, destination))
            return (
                f"mkdir -p {shlex.quote(destination)} && "
                + shlex.join(["cp", "-a", str(source_path(url)) + "/.", destination])
            )

        monkeypatch.setattr(backend, f"_{'gcs' if scheme == 'gs' else scheme}_exists", source_exists)
        monkeypatch.setattr(backend, "_rsync_cmd" if scheme == "gs" else "_sync_cmd", sync_command)
        if scheme == "gs":
            monkeypatch.setattr(gsbucket, "_restore_symlinks", AsyncMock())
            sandbox.archive_source = (prefix + "reference.7z", source_base / "reference.7z")

    monkeypatch.setenv("REMOTE_OUTPUT_DIR", output_name)
    monkeypatch.setenv("REMOTE_ROOT_DIR", sandbox.task_data_root)
    driver = TaskDriver(str(aliased_task_root), session=sandbox)
    task_info = driver.task_info
    metadata = driver.task_metadata
    spec = task_info["task_data"]
    assert metadata["task_id"] == "fixture_domain/canonical_answer_task"
    assert spec.source_relpath == CANONICAL
    assert spec.task_name == "canonical_answer_task"
    assert spec.sandbox_task_name == "neutral_workspace"
    assert task_subdir(sandbox, spec) == str(visible)
    for key, value in metadata.items():
        if key.endswith(("_dir", "_file", "_fasta", "_json", "_tsv", "_pyproject")):
            assert "canonical_answer" not in value.lower()
            assert "canonical_answer_task" not in value.lower()
    assert "canonical_answer" not in driver.task_description.lower()
    assert "canonical_answer_task" not in driver.task_description.lower()
    assert str(visible / output_name / "answer.json") in driver.task_description

    artifacts = SimpleNamespace(task_data_source=source)
    env = SimpleNamespace(sandbox=sandbox)
    for _attempt in range(2):
        await _stage_task_data(env=env, provider=None, artifacts=artifacts, task_meta=task_info)
        await driver.setup()
        assert not canonical.parent.exists()
        assert not canonical.parent.is_symlink()
        assert not visible.is_symlink()
        assert not visible.parent.is_symlink()
        assert Path(metadata["input_file"]).read_text() == "synthetic input\n"
        assert Path(metadata["settings_json"]).is_file()
        assert Path(metadata["runtime_pyproject"]).is_file()
        assert Path(metadata["remote_output_dir"]).is_dir()
        assert not Path(metadata["verification_targets_file"]).exists()
        assert not any("canonical_answer" in str(path) for path in Path(sandbox.task_data_root).rglob("*"))
    if prebaked and source_kind not in {"local", "gs_release"}:
        assert Path(metadata["input_file"]).stat().st_ino == original_inode
    assert sibling.read_text() == "unrelated"
    assert (source_base / "input/input.txt").exists()

    answer_path = Path(metadata["answer_file"])
    assert answer_path.parent == visible / output_name
    answer_path.write_text(json.dumps(ANSWER))
    assert output_pull._output_dir(sandbox, spec) == str(answer_path.parent)
    report = await output_pull.pull_to_host(sandbox, spec, dest_dir=tmp_path / "collected")
    assert report["files"] == 1
    assert json.loads((tmp_path / "collected/answer.json").read_text()) == ANSWER

    await stage_reference(
        env=env, provider=None, artifacts=artifacts, task_meta=task_info,
        run_id="fixture", task_id=metadata["task_id"],
        writer=SimpleNamespace(emit_event=lambda *args, **kwargs: None),
    )
    assert Path(metadata["verification_targets_file"]).is_file()
    assert await driver.evaluate() == {"score": 1.0, "raw_scores": [1.0]}
    answer_path.unlink()
    assert await driver.evaluate() == {"score": 0.0, "raw_scores": [0.0]}
    assert not canonical.parent.exists()
    for _source, destination in sandbox.transfers:
        assert Path(destination).is_relative_to(visible)


@pytest.mark.parametrize("alias", ["", ".", "..", "../answer", "/answer", "x/y", "x\\y"])
def test_alias_rejects_non_component_names(alias):
    with pytest.raises(ValueError, match="plain task directory"):
        TaskDataSpec(sandbox_task_name=alias)


async def test_alias_atomically_replaces_empty_destination_and_is_idempotent(tmp_path):
    sandbox = FilesystemSandbox(tmp_path / "guest data")
    source_base = make_source(Path(sandbox.task_data_root))
    destination_base = Path(sandbox.task_data_root) / VISIBLE
    destination_base.parent.mkdir()
    source_inode = source_base.parent.stat().st_ino
    evidence_inode = (source_base / "input/input.txt").stat().st_ino
    spec = TaskDataSpec(
        domain_name="fixture_domain", task_name="canonical_answer_task",
        variant_name="base", sandbox_task_name="neutral_workspace",
    )
    for _attempt in range(2):
        assert await prepare_task_subdir(sandbox, spec) == str(destination_base)
        assert not source_base.parent.exists()
        assert destination_base.parent.stat().st_ino == source_inode
        evidence = destination_base / "input/input.txt"
        assert evidence.stat().st_ino == evidence_inode
        assert evidence.read_text() == "synthetic input\n"


@pytest.mark.parametrize(
    "conflict",
    ["both", "hidden_file", "subdirectory", "destination_file", "source_link",
     "destination_link", "dangling_link", "parent_link"],
)
async def test_alias_conflicts_fail_without_deleting_data(tmp_path, conflict):
    sandbox = FilesystemSandbox(tmp_path / "guest")
    spec = TaskDataSpec(
        domain_name="domain", task_name="canonical", variant_name="base", sandbox_task_name="neutral"
    )
    source = Path(sandbox.task_data_root) / "domain/canonical"
    destination = source.with_name("neutral")
    source.mkdir(parents=True)
    (source / "evidence").write_text("preserve")
    if conflict in {"both", "hidden_file", "subdirectory"}:
        destination.mkdir()
        if conflict == "subdirectory":
            (destination / "base").mkdir()
        else:
            (destination / (".hidden" if conflict == "hidden_file" else "evidence")).write_text(
                "also preserve"
            )
    elif conflict == "destination_file":
        destination.write_text("also preserve")
    elif conflict == "source_link":
        source.rename(source.with_name("original"))
        source.symlink_to(source.with_name("original"), target_is_directory=True)
    elif conflict == "destination_link":
        destination.symlink_to(source, target_is_directory=True)
    elif conflict == "dangling_link":
        destination.symlink_to(source.with_name("missing"), target_is_directory=True)
    else:
        source.parent.rename(source.parent.with_name("original_domain"))
        source.parent.symlink_to(source.parent.with_name("original_domain"), target_is_directory=True)
    with pytest.raises(RuntimeError, match="Could not prepare neutral task workspace"):
        await prepare_task_subdir(sandbox, spec)
    assert (source / "evidence").read_text() == "preserve"
    if conflict in {"both", "hidden_file"}:
        assert (destination / (".hidden" if conflict == "hidden_file" else "evidence")).read_text() == (
            "also preserve"
        )
    elif conflict == "subdirectory":
        assert (destination / "base").is_dir()
    elif conflict == "destination_file":
        assert destination.read_text() == "also preserve"


async def test_unaliased_tasks_keep_existing_paths_without_remote_commands(tmp_path):
    sandbox = FilesystemSandbox(tmp_path)
    spec = TaskDataSpec(domain_name="domain", task_name="task", variant_name="base")
    expected = str(tmp_path / "domain/task/base")
    assert await prepare_task_subdir(sandbox, spec) == expected
    assert await prepare_task_subdir(sandbox, replace(spec, sandbox_task_name="task")) == expected
    assert output_pull._output_dir(sandbox, spec) == expected + "/output"
    assert sandbox.commands == []


async def test_windows_alias_fails_before_staging(tmp_path):
    sandbox = FilesystemSandbox(tmp_path)
    sandbox.is_linux = False
    spec = TaskDataSpec(
        domain_name="domain", task_name="canonical", variant_name="base", sandbox_task_name="neutral"
    )
    with pytest.raises(NotImplementedError, match="require Linux"):
        await prepare_task_subdir(sandbox, spec)
    assert sandbox.commands == []


@pytest.mark.parametrize("scheme", ["gs", "s3", "oss"])
@pytest.mark.parametrize(
    "output_name",
    ["output", "output_test_pos", "output_test_neg", "output_admin_pos", "output_admin_neg"],
)
async def test_alias_cloud_output_uses_neutral_source_and_standard_destination(
    tmp_path, monkeypatch, scheme, output_name
):
    sandbox = FilesystemSandbox(tmp_path / "guest data")
    source = str(Path(sandbox.task_data_root) / VISIBLE / output_name)
    spec = TaskDataSpec(
        domain_name="fixture_domain",
        task_name="canonical_answer_task",
        variant_name="base",
        sandbox_task_name="neutral_workspace",
        remote_output_dir=source,
    )
    assert output_pull._output_dir(sandbox, replace(spec, remote_output_dir=None)) == str(
        Path(sandbox.task_data_root) / VISIBLE / "output"
    )
    run_command = AsyncMock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(sandbox, "run_command", run_command)
    push = {"gs": output_pull.push_to_gcs, "s3": output_pull.push_to_s3,
            "oss": output_pull.push_to_oss}[scheme]
    report = await push(sandbox, spec, run_id="fixture", bucket=f"{scheme}://results")
    command = shlex.split(run_command.await_args.args[0])
    destination = f"{scheme}://results/fixture/output/"
    assert command[-2].rstrip("/") == source
    assert command[-1] == destination
    assert report[f"{'gcs' if scheme == 'gs' else scheme}_path"] == destination
    assert "canonical_answer" not in run_command.await_args.args[0]
    if scheme == "gs":
        assert command[:4] == ["gsutil", "-m", "rsync", "-r"]
