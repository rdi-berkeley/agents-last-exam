import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ale_run.orchestration import host_preflight
from ale_run.orchestration.experiment_spec import EnvironmentSpec, ExperimentSpec, OutputSpec
from ale_run.orchestration.runner import Runner


@pytest.fixture
def dependency_fixture(tmp_path, monkeypatch):
    (tmp_path / "env").mkdir()
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks/pyproject.toml").write_text(
        '[project]\ndependencies = ["networkx==3.6.1"]\n'
    )
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "networkx"\nversion = "3.6.1"\n')
    (tmp_path / "env/host-deps.json").write_text(
        json.dumps(
            {
                "python_imports": {"networkx": ["networkx"]},
                "commands": {},
            }
        )
    )
    monkeypatch.setattr(host_preflight, "_ROOT", tmp_path)
    monkeypatch.setattr(host_preflight.importlib.metadata, "version", lambda _: "3.6.1")
    return tmp_path


def test_wrong_networkx_version_is_rejected(dependency_fixture, monkeypatch):
    monkeypatch.setattr(host_preflight.importlib.metadata, "version", lambda _: "3.7")
    report = host_preflight.inspect_host([])
    assert not report["ok"]
    assert any("requires ==3.6.1" in error for error in report["errors"])


def test_lock_drift_is_rejected(dependency_fixture):
    (dependency_fixture / "tasks/pyproject.toml").write_text(
        '[project]\ndependencies = ["networkx>=3.6"]\n'
    )
    (dependency_fixture / "uv.lock").write_text('[[package]]\nname = "networkx"\nversion = "3.7"\n')
    report = host_preflight.inspect_host([])
    assert not report["ok"]
    assert any("differs from uv.lock" in error for error in report["errors"])


def test_missing_native_tool_is_reported(dependency_fixture, monkeypatch):
    (dependency_fixture / "env/host-deps.json").write_text(
        json.dumps(
            {
                "python_imports": {},
                "commands": {"ffprobe": ["-version"]},
            }
        )
    )
    monkeypatch.setattr(
        host_preflight.subprocess, "run", Mock(side_effect=FileNotFoundError("ffprobe"))
    )
    report = host_preflight.inspect_host([])
    assert not report["ok"]
    assert any("Cannot execute ffprobe" in error for error in report["errors"])


def test_deferred_import_failure_is_reported(dependency_fixture):
    (dependency_fixture / "env/host-deps.json").write_text(
        json.dumps(
            {
                "python_imports": {"missing": ["ale_nonexistent_evaluator_dependency"]},
                "commands": {},
            }
        )
    )
    report = host_preflight.inspect_host([])
    assert not report["ok"]
    assert any("ale_nonexistent_evaluator_dependency" in error for error in report["errors"])


def test_task_import_failure_identifies_the_task(dependency_fixture):
    directory = dependency_fixture / "tasks/example/broken"
    directory.mkdir(parents=True)
    (directory / "main.py").write_text("import ale_nonexistent_task_dependency\n")
    report = host_preflight.inspect_host(["example/broken"])
    assert report["tasks"] == []
    assert any("Cannot load example/broken" in error for error in report["errors"])


def test_successful_task_imports_are_deduplicated(dependency_fixture):
    directory = dependency_fixture / "tasks/example/working"
    directory.mkdir(parents=True)
    (directory / "main.py").write_text("value = 1\n")
    report = host_preflight.inspect_host(["example/working", "example/working"])
    assert report["ok"], report["errors"]
    assert report["tasks"] == ["example/working"]


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CompletedProcess([], 1, "Missing ffprobe", ""),
        subprocess.TimeoutExpired("preflight", 240),
    ],
)
def test_subprocess_failure_is_actionable(monkeypatch, failure):
    run = (
        Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
    )
    monkeypatch.setattr(host_preflight.subprocess, "run", run)
    with pytest.raises(host_preflight.HostDependencyError, match="no sandboxes"):
        host_preflight.check_host_dependencies(["demo/hello"])


@pytest.mark.asyncio
async def test_dependency_failure_prevents_judge_and_vm_creation(monkeypatch, tmp_path):
    check = Mock(side_effect=host_preflight.HostDependencyError("Missing ffprobe"))
    judge = AsyncMock()
    launch = AsyncMock()
    monkeypatch.setattr("ale_run.orchestration.host_preflight.check_host_dependencies", check)
    monkeypatch.setattr("ale_run.orchestration.runner.check_judges", judge)
    monkeypatch.setattr("ale_run.orchestration.lifecycle.run_one_unit", launch)
    spec = ExperimentSpec(
        name="host-preflight-test",
        output=OutputSpec(root=str(tmp_path)),
        environment=EnvironmentSpec(),
        agents=[],
        tasks=[],
    )
    with pytest.raises(host_preflight.HostDependencyError, match="Missing ffprobe"):
        await Runner(spec).run([SimpleNamespace(task_path="demo/hello")])
    judge.assert_not_called()
    launch.assert_not_called()
    assert not (tmp_path / spec.name).exists()


def test_native_installer_has_no_side_effects_for_help_or_bad_arguments(tmp_path):
    script = Path(__file__).resolve().parents[2] / "env/install_host_deps.sh"
    for arguments, status in [(["--help"], 0), (["--unknown"], 2)]:
        result = subprocess.run(
            ["bash", str(script), *arguments],
            check=False,
            cwd=tmp_path,
            capture_output=True,
        )
        assert result.returncode == status


def test_each_declared_evaluator_dependency_has_an_import_probe():
    import tomllib

    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "tasks/pyproject.toml").read_text())
    manifest = json.loads((root / "env/host-deps.json").read_text())
    requirements = {
        canonicalize_name(Requirement(value).name) for value in project["project"]["dependencies"]
    }
    probes = {canonicalize_name(value) for value in manifest["python_imports"]}
    assert requirements == probes
