from __future__ import annotations

import json
import subprocess
import traceback
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

import pytest

from ale_run.agents.dummy.config import DummyConfig
from ale_run.agents.dummy.deployer import DummyDeployer
from ale_run.base_interface import AgentRunResult, SandboxHandle
from ale_run.base_interface import sandbox as sandbox_interface
from ale_run.executors import sandbox as sandbox_executor
from ale_run.executors.sandbox import SandboxExecutor, _build_launcher


@dataclass
class _FakeSandbox:
    os: str = "linux"
    id: str = "sandbox-test"
    endpoint: str = "http://127.0.0.1:5000"
    work_dir_base: str = "/home/user/.ale"
    task_data_root: str = "/data"
    node: str = "/usr/bin/node"
    python: str = "/usr/bin/python3"
    mcp_server_dir: str = "/home/user/cua_mcp_server"
    cua_server_port: int = 5000
    metadata: dict | None = None

    def __post_init__(self) -> None:
        self.metadata = self.metadata or {}
        self.alive_results: list[int] = []
        self.alive_calls = 0
        self.result = {
            "ok": True,
            "status": "completed",
            "exit_code": 0,
            "duration_s": 1.0,
        }
        self.files: dict[str, str] = {}

    @property
    def is_linux(self) -> bool:
        return self.os == "linux"

    async def mkdir(self, path: str) -> None:
        _ = path

    async def rm(self, paths: list[str]) -> None:
        _ = paths

    async def write_file(self, path: str, content: str | bytes) -> None:
        self.files[path] = (
            content.decode("utf-8") if isinstance(content, bytes) else content
        )

    async def exists(self, path: str) -> bool:
        return path.endswith("_done.marker") and self.alive_calls > len(
            self.alive_results
        )

    async def read_text(self, path: str) -> str:
        if path.endswith("_stopped.json"):
            token = next(value for name, value in self.files.items() if name.endswith("_stop.request"))
            return json.dumps({"run_token": token, "stopped": True})
        if path.endswith("_result.json"):
            return json.dumps(self.result)
        if path in self.files:
            return self.files[path]
        raise FileNotFoundError(path)

    async def run_command(
        self,
        command: str,
        *,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess:
        _ = timeout
        if "kill -0" not in command and "Get-Process -Id" not in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        self.alive_calls += 1
        if self.alive_calls <= len(self.alive_results):
            returncode = self.alive_results[self.alive_calls - 1]
        else:
            returncode = 0
        return subprocess.CompletedProcess(command, returncode, "", "transport error")


def _executor(sandbox: _FakeSandbox) -> SandboxExecutor:
    return SandboxExecutor(
        config=DummyConfig(),
        work_dir="/home/user/.ale/test-run",
        sandbox=sandbox,  # type: ignore[arg-type]
    )


async def _prepare_executor(
    monkeypatch: pytest.MonkeyPatch,
    sandbox: _FakeSandbox,
) -> SandboxExecutor:
    executor = _executor(sandbox)

    async def no_ship(ale_src_root: str) -> None:
        _ = ale_src_root

    async def read_pid(pid_file: str) -> int:
        _ = pid_file
        return 1234

    async def no_sleep(delay: float) -> None:
        _ = delay

    monkeypatch.setattr(executor, "_ship_ale_subtree", no_ship)
    monkeypatch.setattr(executor, "_read_pid", read_pid)
    monkeypatch.setattr(sandbox_executor.asyncio, "sleep", no_sleep)
    return executor


@pytest.mark.asyncio
async def test_transport_failures_do_not_count_as_process_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeSandbox()
    sandbox.alive_results = [-1, -1, -1]
    executor = await _prepare_executor(monkeypatch, sandbox)
    monkeypatch.setattr(
        sandbox_executor,
        "_LIVENESS_TRANSPORT_GRACE_S",
        60.0,
    )

    result = await executor.run_deployer(
        deployer_cls=DummyDeployer,
        prompt="test",
        timeout_s=60,
    )

    assert result.status == "completed"
    assert sandbox.alive_calls == 4


@pytest.mark.asyncio
async def test_persistent_transport_failure_reports_unknown_process_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeSandbox()
    sandbox.alive_results = [-1]
    executor = await _prepare_executor(monkeypatch, sandbox)
    monkeypatch.setattr(
        sandbox_executor,
        "_LIVENESS_TRANSPORT_GRACE_S",
        0.0,
    )

    result = await executor.run_deployer(
        deployer_cls=DummyDeployer,
        prompt="test",
        timeout_s=60,
    )

    assert result.status == "failed"
    assert result.error is not None
    assert "transport unavailable" in result.error
    assert "process state unknown" in result.error
    assert "process disappeared" not in result.error


@pytest.mark.asyncio
async def test_process_missing_probes_still_fail_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = _FakeSandbox()
    sandbox.alive_results = [1, 1, 1]
    executor = await _prepare_executor(monkeypatch, sandbox)

    result = await executor.run_deployer(
        deployer_cls=DummyDeployer,
        prompt="test",
        timeout_s=60,
    )

    assert result.status == "failed"
    assert result.error is not None
    assert "process disappeared" in result.error
    assert "process-missing probes" in result.error


@pytest.mark.parametrize("os_type", ["linux", "windows"])
@pytest.mark.asyncio
async def test_missing_process_result_is_recorded_before_stop_failure(monkeypatch, os_type):
    sandbox = _FakeSandbox(os=os_type)
    sandbox.alive_results = [1, 1, 1]
    executor = await _prepare_executor(monkeypatch, sandbox)
    recorded = []
    executor.on_run_result = recorded.append

    async def stop():
        assert recorded == [executor.last_run_result]
        raise RuntimeError("termination unconfirmed")

    monkeypatch.setattr(executor, "stop_deployer", stop)
    with pytest.raises(RuntimeError, match="termination unconfirmed") as caught:
        await executor.run_deployer(deployer_cls=DummyDeployer, prompt="test", timeout_s=60)

    assert executor.last_run_result.pid == 1234
    assert "process disappeared" in executor.last_run_result.error
    assert "process disappeared" in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("os_type", ["linux", "windows"])
@pytest.mark.parametrize("worker_pid", [None, 5678])
@pytest.mark.parametrize("stop_fails", [False, True])
@pytest.mark.asyncio
async def test_failed_worker_identity_survives_host_result(
    monkeypatch, os_type, worker_pid, stop_fails,
):
    sandbox = _FakeSandbox(os=os_type)
    sandbox.result = {
        "ok": False,
        "status": "failed",
        "exit_code": -9,
        "error": "solver worker exited with code -9 without _worker_result.json",
    }
    if worker_pid is not None:
        sandbox.result["pid"] = worker_pid
    executor = await _prepare_executor(monkeypatch, sandbox)
    recorded = []
    executor.on_run_result = recorded.append
    stop = AsyncMock(side_effect=RuntimeError("stop failed") if stop_fails else None)
    monkeypatch.setattr(executor, "stop_deployer", stop)

    if stop_fails:
        with pytest.raises(RuntimeError, match="stop failed") as caught:
            await executor.run_deployer(deployer_cls=DummyDeployer, prompt="test", timeout_s=60)
        rendered = "".join(traceback.format_exception(caught.value))
        assert '"exit_code": -9' in rendered
        assert f'"pid": {worker_pid or 1234}' in rendered
    else:
        result = await executor.run_deployer(
            deployer_cls=DummyDeployer, prompt="test", timeout_s=60,
        )
        assert result is executor.last_run_result

    stop.assert_awaited_once()
    assert recorded == [executor.last_run_result]
    assert recorded[0].status == "failed"
    assert recorded[0].error == sandbox.result["error"]
    assert recorded[0].pid == (worker_pid or 1234)
    assert recorded[0].exit_code == -9


@pytest.mark.parametrize("run_raises", [False, True])
@pytest.mark.parametrize("observer_raises", [False, True])
@pytest.mark.asyncio
async def test_primary_and_observer_errors_survive_cleanup_failure(
    monkeypatch, run_raises, observer_raises,
):
    executor = _executor(_FakeSandbox())
    original = AgentRunResult(status="failed", error="primary run failure", pid=42, exit_code=-9)
    run = AsyncMock(
        side_effect=ValueError("primary run failure") if run_raises else None,
        return_value=original,
    )
    observer = Mock(side_effect=LookupError("observer failure") if observer_raises else None)
    executor.on_run_result = observer
    stop = AsyncMock(side_effect=RuntimeError("cleanup failure"))
    monkeypatch.setattr(executor, "_run_deployer", run)
    monkeypatch.setattr(executor, "stop_deployer", stop)

    with pytest.raises(RuntimeError, match="cleanup failure") as caught:
        await executor.run_deployer(deployer_cls=DummyDeployer, prompt="test", timeout_s=60)

    stop.assert_awaited_once()
    observer.assert_called_once_with(executor.last_run_result)
    rendered = "".join(traceback.format_exception(caught.value))
    assert "primary run failure" in rendered
    if observer_raises:
        assert "observer failure" in rendered
    if not run_raises:
        assert executor.last_run_result is original
    else:
        assert "ValueError" in executor.last_run_result.error


@pytest.mark.asyncio
async def test_timeout_result_is_recorded_before_stop_failure(monkeypatch):
    sandbox = _FakeSandbox()
    executor = await _prepare_executor(monkeypatch, sandbox)
    recorded = []
    executor.on_run_result = recorded.append
    monkeypatch.setattr(executor, "stop_deployer", AsyncMock(side_effect=RuntimeError("stop failed")))

    with pytest.raises(RuntimeError, match="stop failed"):
        await executor.run_deployer(deployer_cls=DummyDeployer, prompt="test", timeout_s=0)

    assert len(recorded) == 1
    assert recorded[0].status == "timeout"
    assert "wall budget" in recorded[0].error


@pytest.mark.asyncio
async def test_observer_failure_propagates_after_successful_stop(monkeypatch):
    executor = _executor(_FakeSandbox())
    original = AgentRunResult(status="completed")
    monkeypatch.setattr(executor, "_run_deployer", AsyncMock(return_value=original))
    executor.on_run_result = Mock(side_effect=LookupError("observer failure"))
    stop = AsyncMock()
    monkeypatch.setattr(executor, "stop_deployer", stop)

    with pytest.raises(LookupError, match="observer failure"):
        await executor.run_deployer(deployer_cls=DummyDeployer, prompt="test", timeout_s=60)

    stop.assert_awaited_once()
    assert executor.last_run_result is original


@pytest.mark.asyncio
async def test_windows_entry_tail_includes_redirected_stderr() -> None:
    sandbox = _FakeSandbox(os="windows")
    executor = _executor(sandbox)
    sandbox.files[
        r"C:\Users\User\.ale\test-run\_entry.log"
    ] = ""
    sandbox.files[
        r"C:\Users\User\.ale\test-run\_entry.log.err"
    ] = "entry traceback"

    tail = await executor._tail_log(
        r"C:\Users\User\.ale\test-run\_entry.log"
    )

    assert "[stderr]" in tail
    assert "entry traceback" in tail


def test_exists_distinguishes_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = SandboxHandle(
        id="test",
        endpoint="http://127.0.0.1:5000",
        os="linux",
        work_dir_base="/home/user/.ale",
        task_data_root="/data",
        node="/usr/bin/node",
        python="/usr/bin/python3",
        mcp_server_dir="/home/user/cua_mcp_server",
    )

    monkeypatch.setattr(
        sandbox_interface,
        "_run_remote_sync",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args="test",
            returncode=-1,
            stdout="",
            stderr="transport error",
        ),
    )

    with pytest.raises(RuntimeError, match="transport failure"):
        sandbox_interface._exists_sync(sandbox, "/tmp/result")


def test_windows_launcher_prefers_pythonw_for_infrastructure_process() -> None:
    sandbox = _FakeSandbox(os="windows")

    launcher = _build_launcher(
        sandbox=sandbox,  # type: ignore[arg-type]
        python=r"C:\ale-run\.venv\Scripts\python.exe",
        ale_src_root=r"C:\Users\User\.ale-src",
        spec_path=r"C:\Users\User\.ale\run\_spec.json",
        pid_file=r"C:\Users\User\.ale\run\_pid",
        entry_log=r"C:\Users\User\.ale\run\_entry.log",
    )

    assert "'pythonw.exe'" in launcher
    assert "if (-not (Test-Path -LiteralPath $pythonw))" in launcher
    assert "Start-Process -FilePath $pythonw" in launcher
