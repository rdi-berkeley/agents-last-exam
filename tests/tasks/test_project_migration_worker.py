import asyncio
import ctypes.util
import json
import os
import shlex
import sys

import pytest

from tasks.visual_media.project_migration import runtime
from tasks.visual_media.project_migration.runtime import checked_command, run_evaluator
from tasks.utils.evaluation import JudgeInfrastructureError


@pytest.fixture
def local_session():
    class Session:
        async def run_command(self, command, check=False):
            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=2)
            return {
                "return_code": 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
            }

    return Session()


@pytest.fixture
def runtime_wrapper(tmp_path, monkeypatch):
    directory = tmp_path / "runtime with spaces"
    directory.mkdir()
    python = directory / "python3"
    python.symlink_to(sys.executable)
    preload = ctypes.util.find_library("c")
    assert preload
    environment = {
        "LV2_PATH": str(directory / "plugins"),
        "LD_PRELOAD": preload,
        "ARDOUR_BATCH_HELPER": str(directory / "batch_export.so"),
    }
    wrapper = directory / "ale-migration-env"
    assignments = [f"PATH={directory}:{os.environ['PATH']}"]
    assignments.extend(f"{name}={value}" for name, value in environment.items())
    wrapper.write_text("#!/bin/sh\nexec env " + shlex.join(assignments) + ' "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setattr(runtime, "RUNTIME_WRAPPER", str(wrapper))
    return wrapper, python, environment


@pytest.mark.asyncio
async def test_checked_command_inherits_runtime_and_preserves_shell_quoting(
    tmp_path, local_session, runtime_wrapper
):
    _, python, environment = runtime_wrapper
    marker = tmp_path / "must-not-exist"
    literal = f"quotes '\"; $(touch {marker})"
    script = (
        "import json, os, sys; "
        f"names = {list(environment)!r}; "
        "print(json.dumps({'python': sys.executable, 'literal': sys.argv[1], "
        "'environment': {name: os.environ[name] for name in names}}))"
    )
    result = await checked_command(local_session, shlex.join(["python3", "-c", script, literal]))
    assert result["exit_code"] == 0
    assert json.loads(result["stdout"]) == {
        "python": str(python),
        "literal": literal,
        "environment": environment,
    }
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", ["missing", "not_executable"])
async def test_unavailable_wrapper_fails_before_intended_command(
    tmp_path, local_session, runtime_wrapper, unavailable
):
    wrapper, _, _ = runtime_wrapper
    if unavailable == "missing":
        wrapper.unlink()
    else:
        wrapper.chmod(0o644)
    marker = tmp_path / "must-not-run"
    with pytest.raises(
        JudgeInfrastructureError, match="exit 127: Required migration runtime wrapper unavailable"
    ):
        await checked_command(local_session, shlex.join(["touch", str(marker)]))
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["wrapper", "command"])
async def test_wrapper_and_command_exit_propagate_despite_lossy_transport(
    local_session, runtime_wrapper, failure
):
    wrapper, _, _ = runtime_wrapper
    command = "printf '%s\\n' 'runtime failure' >&2; exit 23"
    if failure == "wrapper":
        wrapper.write_text("#!/bin/sh\n" + command + "\n")
        command = "exit 0"
    with pytest.raises(JudgeInfrastructureError, match="exit 23: runtime failure"):
        await checked_command(local_session, command)


@pytest.mark.asyncio
async def test_native_worker_is_detached_and_launch_is_idempotent(
    tmp_path, local_session, runtime_wrapper
):
    _, python, environment = runtime_wrapper
    counter = tmp_path / "launch-count"
    evidence = tmp_path / "evidence with spaces"
    script = (
        "import json, os, sys; "
        f"names = {list(environment)!r}; "
        "print(json.dumps({'python': sys.executable, "
        "'environment': {name: os.environ[name] for name in names}}))"
    )
    command = "timeout 5 sh -c " + shlex.quote(
        f"echo started >> {shlex.quote(str(counter))}; "
        + shlex.join(["python3", "-c", script])
        + "; sleep 0.1; exit 7"
    )
    for _ in range(2):
        result = await run_evaluator(
            local_session, command, str(evidence), timeout=5, poll_interval=0.01
        )
        assert result == {"return_code": 7}
    assert counter.read_text() == "started\n"
    assert (evidence / "worker.exit").read_text() == "7\n"
    assert json.loads((evidence / "worker.log").read_text()) == {
        "python": str(python),
        "environment": environment,
    }
    assert await run_evaluator(
        local_session, "exit 9", str(tmp_path / "explicit-exit"), timeout=5, poll_interval=0.01
    ) == {"return_code": 9}


@pytest.mark.asyncio
async def test_sdk_lost_exit_status_does_not_hide_failed_worker_launch(
    tmp_path, local_session, runtime_wrapper
):
    evidence = tmp_path / "ordinary-file"
    evidence.write_text("not a directory")
    with pytest.raises(JudgeInfrastructureError, match="exit 1"):
        await asyncio.wait_for(
            run_evaluator(local_session, "exit 0", str(evidence), timeout=1), timeout=2
        )
