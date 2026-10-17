from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import cv2
import numpy as np
import pytest

from tasks.visual_media.chroma_key_from_reference import main as task


def configuration():
    return SimpleNamespace(metadata=task._build_config(task.VARIANTS[0]).to_metadata())


def session_with_result(result):
    return SimpleNamespace(
        interface=SimpleNamespace(create_dir=AsyncMock()),
        write_file=AsyncMock(),
        read_bytes=AsyncMock(side_effect=OSError("no report")),
        file_exists=AsyncMock(return_value=True),
        directory_exists=AsyncMock(return_value=False),
        run_command=AsyncMock(return_value=result),
    )


@pytest.mark.asyncio
async def test_setup_and_fresh_evaluation_config_use_same_interpreter(monkeypatch):
    monkeypatch.setattr(task, "_setup", AsyncMock())
    setup_cfg = configuration()
    interpreter = setup_cfg.metadata["evaluation_python"]
    session = session_with_result({
        "return_code": 0, "stdout": json.dumps({"python": interpreter}),
    })
    await task.start(setup_cfg, session)
    assert "--input-video" in session.run_command.call_args.args[0]
    assert "exit $LASTEXITCODE" in session.run_command.call_args.args[0]
    session.run_command.return_value = {
        "return_code": 0,
        "stdout": json.dumps({"score": 0.75, "details": [], "frame_pairs": []}),
    }
    score, *_ = await task._run_remote_hard_eval(configuration(), session)
    assert score == 0.75
    command = session.run_command.call_args.args[0]
    assert task.ps_literal(interpreter) in command
    assert "& python " not in command
    assert "exit $LASTEXITCODE" in command


@pytest.mark.asyncio
async def test_dependency_install_failure_stops_setup(monkeypatch):
    monkeypatch.setattr(task, "_setup", AsyncMock())
    session = session_with_result({"return_code": 1, "stderr": "wheel unavailable"})
    with pytest.raises(RuntimeError, match="wheel unavailable"):
        await task.start(configuration(), session)


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    {"return_code": 1, "stderr": "ModuleNotFoundError: No module named 'cv2'"},
    {"return_code": 0, "stdout": "no valid result"},
])
async def test_broken_evaluator_is_not_a_solver_zero(result):
    with pytest.raises(RuntimeError, match="chroma hard evaluator"):
        await task._run_remote_hard_eval(configuration(), session_with_result(result))


@pytest.mark.asyncio
async def test_missing_submission_is_zero():
    cfg = configuration()
    session = session_with_result({})
    session.file_exists.side_effect = lambda path: path != cfg.metadata["output_video"]
    score, _, _, payload = await task._run_remote_hard_eval(cfg, session)
    assert score == 0
    assert payload["reason"] == "missing_output_video"
    session.run_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_reference_is_an_environment_error():
    session = session_with_result({})
    reference = task._build_config(task.VARIANTS[0]).remote_reference_video
    session.file_exists.side_effect = lambda path: path != reference
    with pytest.raises(RuntimeError, match="reference video is missing"):
        await task._run_remote_hard_eval(configuration(), session)


@pytest.mark.parametrize("output_kind, expected", [("correct", 1), ("green", 0), ("corrupt", 0)])
def test_real_hard_evaluator_video_controls(tmp_path, output_kind, expected):
    reference = np.full((96, 128, 3), (0, 255, 0), dtype=np.uint8)
    cv2.rectangle(reference, (32, 24), (95, 71), (255, 255, 255), -1)
    cv2.line(reference, (32, 24), (95, 71), (0, 0, 0), 3)
    frames = {"input": reference, "reference": reference}
    if output_kind != "corrupt":
        frames["output"] = reference if output_kind == "correct" else np.full_like(reference, (0, 255, 0))
    for name, frame in frames.items():
        writer = cv2.VideoWriter(str(tmp_path / f"{name}.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 5, (128, 96))
        assert writer.isOpened()
        for _ in range(5):
            writer.write(frame)
        writer.release()
    if output_kind == "corrupt":
        (tmp_path / "output.avi").write_bytes(b"not a video")
    (tmp_path / "points.json").write_text(json.dumps({"points": [{"time_sec": 0.2}]}))
    result = subprocess.run([
        sys.executable, str(task.REMOTE_HARD_EVAL_SCRIPT),
        "--input-video", str(tmp_path / "input.avi"),
        "--reference-video", str(tmp_path / "reference.avi"),
        "--output-video", str(tmp_path / "output.avi"),
        "--breakpoint-json", str(tmp_path / "points.json"),
        "--report-json", str(tmp_path / "report.json"),
        "--frames-dir", str(tmp_path / "frames"),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score"] == expected
    if output_kind == "corrupt":
        assert payload["reason"] == "invalid_output_video"
