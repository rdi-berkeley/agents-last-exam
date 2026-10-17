from __future__ import annotations

import base64
import asyncio
from functools import partial
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tasks.visual_media.skeletal_animation_reproduction import main as task
from tasks.visual_media.skeletal_animation_reproduction.scripts import remote_render_eval


@pytest.mark.asyncio
async def test_report_wait_bounds_stalled_transport():
    async def stalled_file_exists(path):
        await asyncio.sleep(10)

    session = SimpleNamespace(file_exists=stalled_file_exists)
    start = time.monotonic()
    assert await task._wait_for_file(session, "report.json", timeout_sec=0.02) is False
    assert time.monotonic() - start < 1


@pytest.mark.asyncio
async def test_evaluation_isolates_runs_and_counts_launch_in_budget(monkeypatch, tmp_path):
    session = SimpleNamespace(
        file_exists=AsyncMock(return_value=True),
        read_file=AsyncMock(
            side_effect=lambda path: json.dumps(
                {"sample_count": 10, "image_width": 512, "image_height": 500}
                if path.endswith("evaluation_config.json")
                else {}
            )
        ),
        interface=SimpleNamespace(create_dir=AsyncMock()),
        write_file=AsyncMock(),
    )
    monkeypatch.setattr(
        task, "TemporaryDirectory", partial(tempfile.TemporaryDirectory, dir=tmp_path)
    )
    monkeypatch.setattr(task, "_upload_scripts", AsyncMock())
    monkeypatch.setattr(
        task,
        "_download_reference_package",
        AsyncMock(return_value=(tmp_path, tmp_path / "reference.mp4")),
    )
    monkeypatch.setattr(task, "extract_reference_frames", Mock(return_value=[]))
    monkeypatch.setattr(task, "reference_framing_targets", Mock(return_value={}))

    async def delayed_launch(*args, **kwargs):
        await asyncio.sleep(0.02)

    launch = AsyncMock(side_effect=delayed_launch)
    wait = AsyncMock(return_value=False)
    monkeypatch.setattr(task, "_launch_remote_job", launch)
    monkeypatch.setattr(task, "_wait_for_file", wait)
    cfg = SimpleNamespace(metadata=task._build_config(task.VARIANTS[0]).to_metadata())
    assert await task.evaluate(cfg, session) == [0.0]
    assert await task.evaluate(cfg, session) == [0.0]
    directories = [call.kwargs["remote_scripts_dir"] for call in launch.call_args_list]
    assert directories[0] != directories[1]
    assert all(0 < call.kwargs["timeout_sec"] < 1799.99 for call in wait.call_args_list)
    runtime_configs = [json.loads(call.args[1]) for call in session.write_file.call_args_list]
    assert all(
        config["sample_count"] == 10
        and config["image_width"] == 512
        and config["image_height"] == 500
        for config in runtime_configs
    )


@pytest.fixture
def launch_args(tmp_path):
    return SimpleNamespace(
        blend="scene with spaces.blend",
        output_dir=str(tmp_path / "render output"),
        renderer_script="renderer.py",
        evaluation_config="config.json",
    )


def test_launch_detaches_all_standard_handles_and_is_idempotent(monkeypatch, launch_args):
    popen = Mock(return_value=SimpleNamespace(pid=12345))
    monkeypatch.setattr(remote_render_eval.subprocess, "Popen", popen)
    root = Path(launch_args.output_dir)
    root.mkdir()
    (root / "job_stdout.txt").write_text("existing evidence\n")

    assert remote_render_eval.launch_job(launch_args) == 0
    assert remote_render_eval.launch_job(launch_args) == 0
    popen.assert_called_once()
    command = popen.call_args.args[0]
    assert command[:2] == [sys.executable, str(Path(remote_render_eval.__file__).resolve())]
    assert "--launch" not in command
    assert command[command.index("--blend") + 1] == launch_args.blend
    options = popen.call_args.kwargs
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stdout"].name == str(root / "job_stdout.txt")
    assert options["stderr"].name == str(root / "job_stderr.txt")
    assert options["close_fds"] is True
    if os.name == "nt":
        assert (
            options["creationflags"]
            == subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        assert options["start_new_session"] is True
    assert json.loads((root / "launch.json").read_text()) == {"status": "started", "pid": 12345}
    assert (root / "job_stdout.txt").read_text() == "existing evidence\n"


@pytest.mark.parametrize("ack", ['{"status":"launching"}', '{"status":"started","pid":123}', ""])
def test_existing_reservation_never_relaunches(monkeypatch, launch_args, ack):
    root = Path(launch_args.output_dir)
    root.mkdir()
    (root / "launch.json").write_text(ack)
    popen = Mock()
    monkeypatch.setattr(remote_render_eval.subprocess, "Popen", popen)
    assert remote_render_eval.launch_job(launch_args) == 0
    popen.assert_not_called()
    assert (root / "launch.json").read_text() == ack


def test_existing_report_is_never_removed_or_relaunched(monkeypatch, launch_args):
    root = Path(launch_args.output_dir)
    root.mkdir()
    report = '{"validity_gate_passed":false}'
    (root / "render_report.json").write_text(report)
    popen = Mock()
    monkeypatch.setattr(remote_render_eval.subprocess, "Popen", popen)
    assert remote_render_eval.launch_job(launch_args) == 0
    popen.assert_not_called()
    assert (root / "render_report.json").read_text() == report
    assert json.loads((root / "launch.json").read_text())["status"] == "report_exists"


def test_failed_spawn_keeps_reservation_without_retry(monkeypatch, launch_args):
    popen = Mock(side_effect=OSError("process creation failed"))
    monkeypatch.setattr(remote_render_eval.subprocess, "Popen", popen)
    assert remote_render_eval.launch_job(launch_args) == 1
    assert remote_render_eval.launch_job(launch_args) == 0
    popen.assert_called_once()
    ack = json.loads((Path(launch_args.output_dir) / "launch.json").read_text())
    assert ack["status"] == "launch_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("response_timeout", [False, True])
async def test_remote_launch_checks_ack_without_reissuing(response_timeout):
    session = SimpleNamespace(
        run_command=AsyncMock(side_effect=TimeoutError if response_timeout else None),
        file_exists=AsyncMock(return_value=True),
        read_file=AsyncMock(return_value='{"status":"started","pid":123}'),
    )
    await task._launch_remote_job(
        session,
        remote_scripts_dir=r"C:\job with spaces\scripts",
        args=["--output-dir", r"C:\job with spaces\results"],
    )
    session.run_command.assert_awaited_once()
    command = session.run_command.call_args.args[0]
    script = base64.b64decode(command.split()[-1]).decode("utf-16-le")
    assert "Start-Process" not in script
    assert "--launch" in script
    assert "Remove-Item" not in script
    assert session.run_command.call_args.kwargs == {"check": True}
    session.read_file.assert_awaited_once_with(r"C:\job with spaces\results\launch.json")


@pytest.mark.asyncio
async def test_remote_launch_waits_for_atomic_ack(monkeypatch):
    session = SimpleNamespace(
        run_command=AsyncMock(),
        file_exists=AsyncMock(return_value=True),
        read_file=AsyncMock(
            side_effect=["", '{"status":"launching"}', '{"status":"started","pid":123}']
        ),
    )
    monkeypatch.setattr(task.asyncio, "sleep", AsyncMock())
    await task._launch_remote_job(
        session, remote_scripts_dir=r"C:\scripts", args=["--output-dir", r"C:\results"]
    )
    session.run_command.assert_awaited_once()
    assert session.read_file.await_count == 3


@pytest.mark.asyncio
async def test_remote_launch_surfaces_spawn_failure():
    session = SimpleNamespace(
        run_command=AsyncMock(),
        file_exists=AsyncMock(return_value=True),
        read_file=AsyncMock(return_value='{"status":"launch_failed"}'),
    )
    with pytest.raises(RuntimeError, match="failed to launch"):
        await task._launch_remote_job(
            session, remote_scripts_dir=r"C:\scripts", args=["--output-dir", r"C:\results"]
        )
    session.run_command.assert_awaited_once()


@pytest.mark.skipif(os.name == "nt", reason="Windows process test runs on the retained audit VM")
def test_real_detached_child_does_not_hold_command_pipes(tmp_path):
    fake_blender = tmp_path / "fake blender"
    fake_blender.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys, time\n"
        "arguments=sys.argv[1:]\n"
        "root=pathlib.Path(arguments[arguments.index('--output-dir')+1])\n"
        "with (root/'worker-starts.txt').open('a') as evidence: evidence.write('started\\n')\n"
        "time.sleep(3)\n"
        "(root/'render_report.json').write_text(json.dumps({'validity_gate_passed': True}))\n"
    )
    fake_blender.chmod(0o755)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"sample_count": 10, "image_width": 512, "image_height": 500}))
    root = tmp_path / "render output"
    command = [
        sys.executable,
        remote_render_eval.__file__,
        "--launch",
        "--blend",
        "scene.blend",
        "--output-dir",
        str(root),
        "--renderer-script",
        "renderer.py",
        "--evaluation-config",
        str(config),
    ]
    environment = {**os.environ, "BLENDER_BINARY": str(fake_blender)}
    worker_pid = None
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=2.5, env=environment
        )
        assert result.returncode == 0, result.stderr
        ack = json.loads((root / "launch.json").read_text())
        worker_pid = ack["pid"]
        assert not (root / "render_report.json").exists()
        duplicate = subprocess.run(
            command, capture_output=True, text=True, timeout=2.5, env=environment
        )
        assert duplicate.returncode == 0
        assert json.loads((root / "launch.json").read_text()) == ack
        deadline = time.monotonic() + 10
        while not (root / "render_report.json").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert json.loads((root / "render_report.json").read_text())["validity_gate_passed"] is True
        assert (root / "worker-starts.txt").read_text().splitlines() == ["started"]
    finally:
        if worker_pid is not None:
            try:
                os.killpg(worker_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
