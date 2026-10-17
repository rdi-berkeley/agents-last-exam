import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from test_gcode_mount import environment, evaluate, native_pipeline
from test_gcode_task import native_request as native_request, worker


def test_nested_safety_timeout_reaches_terminal_diagnostics(native_request, monkeypatch):
    failure = {"type": "SafetyUnavailable", "reason": "Native safety time budget exhausted"}
    receipt = str(native_request.parent / "native/replay/safety-fixture/receipt.json")

    def invoke(command, **kwargs):
        replay = Path(command[3]) / "replay"
        replay.mkdir(parents=True)
        (replay / "result.json").write_text(
            json.dumps(
                {
                    "status": "unavailable",
                    "error": "Native safety unavailable: " + failure["reason"],
                    "safety": {"failure": failure, "collision_free": None},
                    "safety_receipt": receipt,
                }
            )
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(worker.subprocess, "run", invoke)
    result = worker.score_request(native_request)
    assert result["status"] == "unavailable"
    assert "score" not in result
    assert failure["reason"] in result["error"]
    assert result["safety_failure"] == failure
    assert result["safety_receipt"] == receipt
    assert result["replay_result"].endswith("native/replay/result.json")


def test_pipeline_preserves_budget_reason_and_receipt(monkeypatch, tmp_path):
    fixture = environment(monkeypatch, tmp_path, 84)
    failure = {"type": "SafetyUnavailable", "reason": "Native safety time budget exhausted"}
    receipt = str(tmp_path / "safety-fixture/receipt.json")
    monkeypatch.setattr(
        sys.modules["native_safety"].SafetyReplay,
        "consume",
        lambda self, motions: {
            "status": "unavailable",
            "failure": failure,
            "receipt_file": receipt,
        },
    )
    result = evaluate(native_pipeline, fixture)
    assert result["status"] == "unavailable"
    assert "score" not in result
    assert result["safety_receipt"] == receipt
    assert failure["reason"] in result["error"]
    assert not fixture.observations.stock


def test_nested_deadlines_allow_safety_stock_and_terminal_diagnostics(native_request, monkeypatch):
    scripts = Path(__file__).parents[2] / "tasks/engineering/gcode/scripts"
    spec = importlib.util.spec_from_file_location(
        "preview_budget_fixture", scripts / "preview_stock.py"
    )
    preview = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preview)
    request = json.loads(native_request.read_text())
    request["runtime"] = dict.fromkeys(
        (
            "freecad",
            "source_document",
            "library_root",
            "interpreter_runtime",
            "collision_python",
            "stock_binary",
        ),
        "/fixture",
    )
    native_request.write_text(json.dumps(request))
    deadlines = {}
    failure = "Native safety unavailable: Native safety time budget exhausted"

    def invoke(command, **kwargs):
        if "--config" in command:
            deadlines["score"] = kwargs["timeout"]
            monkeypatch.setattr(sys, "argv", command[1:])
            return SimpleNamespace(returncode=preview.main())
        deadlines["native"] = kwargs["timeout"]
        parameters = json.loads(Path(kwargs["env"]["ALE_CAM_PREVIEW_REQUEST"]).read_text())
        replay = Path(parameters["output"]) / "replay"
        replay.mkdir()
        (replay / "result.json").write_text(json.dumps({"status": "unavailable", "error": failure}))
        return SimpleNamespace(returncode=2)

    monkeypatch.setenv("DISPLAY", ":fixture")
    monkeypatch.setattr(worker.subprocess, "run", invoke)
    result = worker.score_request(native_request)
    safety_seconds = (
        inspect.signature(native_pipeline.evaluate_native).parameters["safety_seconds"].default
    )
    assert safety_seconds + 900 <= deadlines["native"]
    assert deadlines["native"] + 100 <= deadlines["score"]
    assert deadlines["score"] + 500 <= 6900
    assert result["status"] == "unavailable"
    assert failure in result["error"]
    assert "score" not in result
