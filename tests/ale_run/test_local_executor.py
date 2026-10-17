from dataclasses import dataclass
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ale_run.base_interface import SandboxHandle
from ale_run.executors.local import LocalExecutor


@dataclass
class Config:
    model: str = "test"


@pytest.fixture
def completed_executor(tmp_path, monkeypatch):
    sandbox = SandboxHandle(
        id="test", endpoint="unused", os="linux", work_dir_base="/work",
        task_data_root="/data", python="python", node="node", mcp_server_dir="/mcp",
    )
    executor = LocalExecutor(config=Config(), work_dir=str(tmp_path), sandbox=sandbox)

    def spawn(*args, **kwargs):
        (tmp_path / "_stopped.json").write_text(json.dumps({
            "run_token": executor._run_token, "stopped": True,
        }))
        (tmp_path / "_result.json").write_text(json.dumps({
            "ok": False, "status": "failed", "error": "worker failed",
            "traceback": "Traceback: task.py line 42 raised RuntimeError",
        }))
        return SimpleNamespace(poll=lambda: 0)

    popen = Mock(side_effect=spawn)
    monkeypatch.setattr("ale_run.executors.local.subprocess.Popen", popen)
    return executor, popen


async def test_local_worker_does_not_inherit_reference_password(completed_executor, monkeypatch):
    executor, popen = completed_executor
    monkeypatch.setenv("ALE_REFERENCE_ARCHIVE_PASSWORD", "host-only-reference-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "solver-api-key")
    await executor.run_deployer(deployer_cls=Config, prompt="test", timeout_s=5)
    environment = popen.call_args.kwargs["env"]
    assert environment.get("ALE_REFERENCE_ARCHIVE_PASSWORD") is None
    assert environment["OPENAI_API_KEY"] == "solver-api-key"
    assert os.environ["ALE_REFERENCE_ARCHIVE_PASSWORD"] == "host-only-reference-secret"


async def test_local_failure_surfaces_worker_traceback(completed_executor):
    executor, _popen = completed_executor
    result = await executor.run_deployer(deployer_cls=Config, prompt="test", timeout_s=5)
    assert result.status == "failed"
    assert "worker failed" in result.error
    assert "task.py line 42" in result.error


async def test_local_stop_reads_confirmation_after_observing_supervisor_exit(tmp_path):
    executor = LocalExecutor(config=None, work_dir=str(tmp_path), sandbox=None)

    def exit_and_publish_report():
        (tmp_path / "_stopped.json").write_text(json.dumps({
            "run_token": executor._run_token, "stopped": True,
        }))
        return 0

    executor._supervisor = SimpleNamespace(poll=exit_and_publish_report)
    await executor.stop_deployer()


async def test_local_stop_rejects_exit_without_confirmation(tmp_path):
    executor = LocalExecutor(config=None, work_dir=str(tmp_path), sandbox=None)
    executor._supervisor = SimpleNamespace(poll=lambda: 0)
    with pytest.raises(RuntimeError, match="termination unconfirmed"):
        await executor.stop_deployer()
