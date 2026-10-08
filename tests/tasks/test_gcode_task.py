import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tasks.engineering.gcode import main, native_task


SCRIPTS = main.ASSETS.parent / "scripts"
SPEC = importlib.util.spec_from_file_location("gcode_score_native", SCRIPTS / "score_native.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def test_registration_retains_all_workpieces_and_public_contract():
    tasks = main.load()
    assert len(tasks) == 18
    assert {task.metadata["variant_name"] for task in tasks} == {tag for tag, _ in main.VARIANTS}
    for task in tasks:
        assert task.computer["setup_config"]["os_type"] == "linux"
        assert task.metadata["output_project"].endswith("/output/machining.FCStd")
        assert "A submitted STL never substitutes" in task.description
        assert "0.3 mm and 2.0 mm bands, weighted 70% and 30%" in task.description
        assert task.metadata["variant_name"] in task.description
    card = json.loads((main.ASSETS.parent / "task_card.json").read_text())
    expected = (
        (main.ASSETS / "instruction.md")
        .read_text()
        .format(
            variant_name="125162_319",
            input_dir="125162_319/input",
            output_dir="125162_319/output",
            software_dir="125162_319/software",
        )
    )
    assert card["taskPrompt"] == expected
    assert card["releaseStatus"] == tasks[0].metadata["release_status"]


class SetupSession:
    def __init__(self, runtime, *, exit_code=0):
        self.runtime = runtime
        self.exit_code = exit_code
        self.files = {}
        self.commands = []

    async def read_file(self, path):
        return json.dumps(self.runtime)

    async def write_file(self, path, content):
        self.files[path] = content

    async def run_command(self, command, *, check=True):
        self.commands.append(command)
        return {
            "return_code": 0 if check else self.exit_code,
            "stderr": "runtime missing",
            "stdout": json.dumps({"runtime_ready": True, "original_tools": 99})
            if not self.exit_code
            else "",
        }


@pytest.mark.asyncio
async def test_setup_uses_supported_session_api_and_excludes_hidden_answers():
    runtime = {
        key: "/runtime/" + key
        for key in (
            "freecad",
            "interpreter_runtime",
            "collision_python",
            "stock_binary",
        )
    }
    runtime["reference"] = "/private/answer.stl"
    session = SetupSession(runtime)
    metadata = main.GCodeTaskConfig().to_metadata()
    await native_task.stage_native_task(metadata, session)
    public = json.loads(session.files[metadata["software_dir"] + "/preview.json"])
    assert "reference" not in public
    assert public["source_document"] == metadata["input_dir"] + "/blank.FCStd"
    assert not any(
        Path(path).name in ("native_geometry.py", "score_native.py") for path in session.files
    )
    assert all("/private/answer" not in content for content in session.files.values())


@pytest.mark.asyncio
async def test_missing_runtime_is_setup_error_not_zero_score():
    runtime = {
        key: "/missing/" + key
        for key in (
            "freecad",
            "interpreter_runtime",
            "collision_python",
            "stock_binary",
        )
    }
    with pytest.raises(native_task.EvaluationUnavailableError, match="incomplete"):
        await native_task.stage_native_task(
            main.GCodeTaskConfig().to_metadata(), SetupSession(runtime, exit_code=1)
        )


@pytest.mark.asyncio
async def test_lost_sdk_exit_code_does_not_accept_failed_runtime_probe():
    class ExitCodeDroppingSession(SetupSession):
        async def run_command(self, command, *, check=True):
            return {"return_code": 0, "stdout": "", "stderr": "missing native executable"}

    runtime = {
        key: "/missing/" + key
        for key in (
            "freecad",
            "interpreter_runtime",
            "collision_python",
            "stock_binary",
        )
    }
    with pytest.raises(native_task.EvaluationUnavailableError, match="incomplete"):
        await native_task.stage_native_task(
            main.GCodeTaskConfig().to_metadata(), ExitCodeDroppingSession(runtime)
        )


@pytest.fixture
def native_request(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    reference = tmp_path / "reference.stl"
    reference.write_bytes(b"reference fixture")
    region = tmp_path / "region.json"
    region.write_text("{}")
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "project": str(tmp_path / "machining.FCStd"),
                "reference": str(reference),
                "region": str(region),
                "runtime": {},
            }
        )
    )
    return path


def preview_result(monkeypatch, *, status, exit_code, calls):
    def invoke(command, **kwargs):
        calls.append(command)
        replay = Path(command[3]) / "replay"
        replay.mkdir(parents=True)
        (replay / "result.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "score": 0,
                    "error": "native diagnostic",
                }
            )
        )
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(worker.subprocess, "run", invoke)


def test_submitted_stl_cannot_skip_replay(native_request, monkeypatch):
    (native_request.parent / "agent_sim.stl").write_bytes(b"candidate-provided output")
    calls = []
    preview_result(monkeypatch, status="invalid_delivery", exit_code=2, calls=calls)
    result = worker.score_request(native_request)
    assert len(calls) == 1
    assert calls[0][2].endswith("machining.FCStd")
    assert result["status"] == "completed"
    assert result["score"] == 0


@pytest.mark.parametrize(
    "status,exit_code", [("unavailable", 1), ("unavailable", 0), ("invalid_delivery", 1)]
)
def test_native_failure_is_unavailable_not_solver_zero(
    native_request, monkeypatch, status, exit_code
):
    calls = []
    preview_result(monkeypatch, status=status, exit_code=exit_code, calls=calls)
    result = worker.score_request(native_request)
    assert result["status"] == "unavailable"
    assert "score" not in result
    assert json.loads((native_request.parent / "terminal.json").read_text()) == result


def test_missing_reference_is_preparation_failure(native_request, monkeypatch):
    (native_request.parent / "reference.stl").unlink()
    calls = []
    preview_result(monkeypatch, status="invalid_delivery", exit_code=2, calls=calls)
    result = worker.score_request(native_request)
    assert result["status"] == "unavailable"
    assert "Evaluator reference is missing" in result["error"]
    assert not calls


def test_timeout_preserves_terminal_error(native_request, monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 3700)

    monkeypatch.setattr(worker.subprocess, "run", timeout)
    result = worker.score_request(native_request)
    assert result["status"] == "unavailable"
    assert "TimeoutExpired" in result["error"]
    assert not (native_request.parent / "terminal.json.partial").exists()
