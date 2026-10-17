from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from ale_run.executors import _sandbox_entry as entry
from ale_run.executors._secrets import SECRET_GATHER_EXCLUDES


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux supervisor launcher")


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    runtime = tmp_path / "systemd"
    runtime.mkdir()
    monkeypatch.setattr(entry, "_SYSTEMD_RUNTIME", runtime)
    monkeypatch.setattr(entry.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(entry.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(entry.os, "getegid", lambda: 1001)
    monkeypatch.setattr(entry.os, "getgroups", lambda: [1001, 27, 44])
    spec = {"work_dir": str(tmp_path), "run_token": "test-token", "timeout_s": 1}
    spec_path = tmp_path / "_spec.json"
    spec_path.write_text(json.dumps(spec))
    run = Mock(return_value=subprocess.CompletedProcess(
        [], 0, "LoadState=not-found\nActiveState=inactive\nMainPID=0\n", "",
    ))
    popen = Mock()
    monkeypatch.setattr(entry.subprocess, "run", run)
    monkeypatch.setattr(entry.subprocess, "Popen", popen)
    return spec_path, spec, run, popen


def test_systemd_service_preserves_identity_environment_without_logging_secrets(launcher, monkeypatch):
    spec_path, spec, run, popen = launcher
    monkeypatch.setenv("DISPLAY", ":7")
    monkeypatch.setenv("XAUTHORITY", "/home/user/GUI auth")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/1000/bus")
    monkeypatch.setenv("PYTHONPATH", "/path with spaces:existing")
    monkeypatch.setenv("LAUNCH_TEST_SECRET", "private\nvalue='quoted'")

    assert entry.launch(spec_path, spec) == 0

    command = run.call_args.args[0]
    assert command[:4] == ["/usr/bin/sudo", "-n", "--", "/usr/bin/systemd-run"]
    assert "--unit=ale-solver-test-token.service" in command
    for option in (
        "--service-type=exec", "--property=OOMPolicy=continue",
        "--property=OOMScoreAdjust=-1000", "--property=Restart=no",
        "--property=KillMode=control-group", "--property=User=1000",
        "--property=Group=1001", "--property=SupplementaryGroups=1001 27 44",
        f"--working-directory={os.getcwd()}",
    ):
        assert option in command
    assert "--scope" not in command
    assert not any("cua" in argument.lower() for argument in command)
    assert not any("private" in argument for argument in command)
    environment_path = spec_path.parent / "_env"
    assert stat.S_IMODE(environment_path.stat().st_mode) == 0o600
    assert environment_path.name in SECRET_GATHER_EXCLUDES
    environment = json.loads(environment_path.read_text())
    for name in ("DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS", "PYTHONPATH", "LAUNCH_TEST_SECRET"):
        assert environment[name] == os.environ[name]
    popen.assert_not_called()
    assert not (spec_path.parent / "_pid").exists()

    bootstrap = command[command.index("-c") + 1]
    monkeypatch.setattr(entry.sys, "argv", ["-c", str(environment_path), str(spec_path)])
    execute = Mock()
    monkeypatch.setattr(entry.os, "execve", execute)
    exec(bootstrap)
    execute.assert_called_once_with(
        sys.executable,
        [sys.executable, "-m", "ale_run.executors._sandbox_entry", str(spec_path), "--systemd-supervisor"],
        environment,
    )
    assert not environment_path.exists()


@pytest.mark.parametrize("state", ["active", "activating"])
def test_systemd_retry_reattaches_without_second_service_or_overwriting_environment(launcher, state):
    spec_path, spec, run, popen = launcher
    run.return_value = subprocess.CompletedProcess(
        [], 0, f"LoadState=loaded\nActiveState={state}\nMainPID=5439\n", "",
    )
    environment_path = spec_path.parent / "_env"
    environment_path.write_text("original")
    assert entry.launch(spec_path, spec) == 0
    assert run.call_count == 1
    assert environment_path.read_text() == "original"
    popen.assert_not_called()
    assert not (spec_path.parent / "_pid").exists()


def test_systemd_unit_names_are_run_token_unique(launcher):
    spec_path, spec, run, _popen = launcher
    entry.launch(spec_path, spec)
    first = run.call_args.args[0]
    spec["run_token"] = "second-token"
    entry.launch(spec_path, spec)
    assert "--unit=ale-solver-test-token.service" in first
    assert "--unit=ale-solver-second-token.service" in run.call_args.args[0]


@pytest.mark.parametrize("failure", ["query", "start", "timeout", "inactive", "tool", "sudo"])
def test_capable_systemd_launch_errors_never_fall_back(launcher, monkeypatch, failure):
    spec_path, spec, run, popen = launcher
    if failure == "query":
        run.return_value = subprocess.CompletedProcess([], 1, "", "permission denied")
    elif failure == "start":
        run.side_effect = [run.return_value, subprocess.CompletedProcess([], 1, "", "start failed")]
    elif failure == "timeout":
        run.side_effect = [run.return_value, subprocess.TimeoutExpired("systemd-run", 30)]
    elif failure == "inactive":
        run.return_value = subprocess.CompletedProcess(
            [], 0, "LoadState=loaded\nActiveState=failed\nMainPID=0\n", "",
        )
    else:
        missing = "systemd-run" if failure == "tool" else "sudo"
        monkeypatch.setattr(entry.shutil, "which", lambda name: None if name == missing else name)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        entry.launch(spec_path, spec)
    popen.assert_not_called()
    assert not (spec_path.parent / "_pid").exists()
    assert not (spec_path.parent / "_env").exists()


def test_environment_cleanup_error_does_not_hide_launch_failure(launcher, monkeypatch):
    spec_path, spec, run, _popen = launcher
    run.side_effect = [run.return_value, subprocess.CompletedProcess([], 1, "", "start failed")]
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=PermissionError("cannot remove environment")))
    with pytest.raises(RuntimeError, match="start failed") as caught:
        entry.launch(spec_path, spec)
    assert "cannot remove environment" in "".join(caught.value.__notes__)


def test_root_launcher_does_not_require_sudo(launcher, monkeypatch):
    spec_path, spec, run, _popen = launcher
    monkeypatch.setattr(entry.os, "geteuid", lambda: 0)
    assert entry.launch(spec_path, spec) == 0
    assert run.call_args.args[0][0] == "/usr/bin/systemd-run"


@pytest.mark.parametrize("systemd", [False, True])
def test_completed_claim_cannot_restart_solver(launcher, monkeypatch, systemd):
    spec_path, spec, run, popen = launcher
    (spec_path.parent / "_supervisor_test-token.claim").touch()
    if not systemd:
        monkeypatch.setattr(entry, "_SYSTEMD_RUNTIME", spec_path.parent / "missing")
    assert entry.launch(spec_path, spec) == 0
    assert run.call_count == int(systemd)
    popen.assert_not_called()


def test_no_systemd_retains_detached_fallback(launcher, monkeypatch):
    spec_path, spec, run, popen = launcher
    monkeypatch.setattr(entry, "_SYSTEMD_RUNTIME", spec_path.parent / "missing")
    assert entry.launch(spec_path, spec) == 0
    run.assert_not_called()
    assert popen.call_args.args[0] == [
        sys.executable, "-m", "ale_run.executors._sandbox_entry", str(spec_path),
    ]
    assert popen.call_args.kwargs["start_new_session"] is True
    assert popen.call_args.kwargs["stdin"] == subprocess.DEVNULL
    assert not (spec_path.parent / "_pid").exists()


@pytest.mark.parametrize("failure", [None, "write", "verify"])
def test_worker_resets_oom_score_before_gate_and_deployer(launcher, monkeypatch, failure):
    spec_path, _spec, _run, _popen = launcher
    events = []
    read_text = Path.read_text
    write_text = Path.write_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/self/oom_score_adj":
            events.append("verify")
            return "-1000" if failure == "verify" else "0"
        return read_text(path, *args, **kwargs)

    def write(path, value, *args, **kwargs):
        if str(path) == "/proc/self/oom_score_adj":
            events.append(("reset", value))
            if failure == "write":
                raise PermissionError("reset denied")
            return 1
        return write_text(path, value, *args, **kwargs)

    gate = Mock()
    gate.read.side_effect = lambda size: events.append("gate") or b"1"
    monkeypatch.setattr(entry.sys, "platform", "linux")
    monkeypatch.setattr(entry.sys, "argv", ["entry", str(spec_path), "--worker"])
    monkeypatch.setattr(entry.sys, "stdin", Mock(buffer=gate))
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "write_text", write)
    solver = Mock(side_effect=lambda *args: events.append("solver") or {"ok": True})
    monkeypatch.setattr(entry, "run", solver)
    assert entry.main() == int(failure is not None)
    assert events == (
        [("reset", "0"), "verify", "gate", "solver"] if failure is None
        else [("reset", "0")] if failure == "write" else [("reset", "0"), "verify"]
    )
    result = json.loads((spec_path.parent / "_worker_result.json").read_text())
    if failure:
        assert result["status"] == "failed"
        assert "worker OOM score reset" in result["error"]
        solver.assert_not_called()


def test_supervisor_publishes_own_pid_before_worker_start(launcher, monkeypatch):
    spec_path, spec, _run, _popen = launcher
    monkeypatch.setattr(entry.sys, "argv", ["entry", str(spec_path), "--systemd-supervisor"])
    monkeypatch.setattr(entry.sys, "platform", "linux")

    def supervise(path, supplied_spec, *, protected):
        assert protected
        assert supplied_spec == spec
        assert int((spec_path.parent / "_pid").read_text()) == os.getpid()
        return {"ok": True}

    monkeypatch.setattr(entry, "supervise", supervise)
    assert entry.main() == 0


def test_supervisor_without_requested_protection_never_launches_worker(launcher, monkeypatch):
    spec_path, spec, _run, popen = launcher
    original = Path.read_text

    def read(path, *args, **kwargs):
        return "0" if str(path) == "/proc/self/oom_score_adj" else original(path, *args, **kwargs)

    monkeypatch.setattr(entry.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", read)
    result = entry.supervise(spec_path, spec, protected=True)
    assert "supervisor OOM protection unavailable" in result["error"]
    popen.assert_not_called()
    assert json.loads((spec_path.parent / "_stopped.json").read_text())["stopped"] is True
