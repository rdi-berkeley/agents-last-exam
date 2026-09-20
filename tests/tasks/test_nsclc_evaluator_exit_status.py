import asyncio
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from cua_bench.computers.remote import RemoteDesktopSession

from tasks.health_medicine.nsclc_radiomics_cox_signature_v1 import main as task


@pytest.fixture
def bundle(tmp_path):
    output = tmp_path / "output"
    reference = tmp_path / "reference"
    output.mkdir()
    reference.mkdir()
    with (reference / "ground_truth.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PatientID", "survival_time_days", "deadstatus.event"])
        writer.writerows((f"patient-{index}", index + 1, 1) for index in range(6))
    with (output / "risk_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PatientID", "risk_score"])
        writer.writerows(
            (f"patient-{index}", risk) for index, risk in enumerate([5, 3, 0, 1, 2, 4])
        )
    return SimpleNamespace(output=output, reference=reference, report=tmp_path / "result.json")


def run_verifier(bundle):
    return subprocess.run(
        [
            sys.executable,
            "-B",
            str(task.SCRIPTS_DIR / "verify_outputs.py"),
            "--output-dir",
            str(bundle.output),
            "--reference-dir",
            str(bundle.reference),
            "--out",
            str(bundle.report),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.fixture
def evaluate(bundle, tmp_path, monkeypatch):
    def run_command(command):
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        return SimpleNamespace(
            return_code=result.returncode, stdout=result.stdout, stderr=result.stderr
        )

    def write_text(path, content):
        Path(path).write_text(content, encoding="utf-8")

    session = RemoteDesktopSession()
    session._computer = SimpleNamespace(
        interface=SimpleNamespace(
            run_command=AsyncMock(side_effect=run_command),
            write_text=AsyncMock(side_effect=write_text),
        )
    )
    monkeypatch.setattr(session, "_ensure_computer", AsyncMock())
    monkeypatch.setattr(task, "EVAL_TMP_DIR", str(tmp_path / "evaluator"))
    task_cfg = SimpleNamespace(
        metadata={
            "risk_scores_file": str(bundle.output / "risk_scores.csv"),
            "remote_output_dir": str(bundle.output),
            "reference_dir": str(bundle.reference),
        }
    )

    def evaluate_task():
        return asyncio.run(task.evaluate(task_cfg, session))

    return evaluate_task


@pytest.mark.parametrize(
    "risks,score,passed",
    [
        ([5, 3, 0, 1, 2, 4], 0.0, False),
        ([1, 1, 1, 1, 1, 1], 0.0, False),
        ([0, 1, 2, 3, 4, 5], 0.0, False),
        ([5, 4, 0, 1, 2, 3], 0.6, True),
        ([5, 4, 3, 2, 1, 0], 1.0, True),
    ],
)
def test_completed_scoring_exits_successfully_independent_of_score(
    bundle, evaluate, risks, score, passed
):
    with (bundle.output / "risk_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PatientID", "risk_score"])
        writer.writerows((f"patient-{index}", risk) for index, risk in enumerate(risks))
    result = run_verifier(bundle)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == json.loads(bundle.report.read_text())
    assert payload["score"] == score
    assert payload["passed"] is passed
    if risks == [5, 3, 0, 1, 2, 4]:
        assert payload["c_index"] == 0.533333
        assert payload["pass_threshold"] == 0.55
        assert payload["n_patients"] == 6
    assert evaluate() == [score]


@pytest.mark.parametrize(
    "defect", ["missing", "rows", "duplicate", "unknown", "nonfinite", "column"]
)
def test_invalid_candidate_is_a_completed_zero(bundle, evaluate, defect):
    path = bundle.output / "risk_scores.csv"
    if defect == "missing":
        path.unlink()
    else:
        rows = path.read_text().splitlines()
        if defect == "rows":
            rows.pop()
        elif defect == "duplicate":
            rows[-1] = rows[1]
        elif defect == "unknown":
            rows[-1] = "unknown,1"
        elif defect == "nonfinite":
            rows[-1] = "patient-5,nan"
        else:
            rows[0] = "PatientID,wrong_column"
        path.write_text("\n".join(rows) + "\n")
    result = run_verifier(bundle)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["score"] == 0.0
    assert payload["passed"] is False
    assert payload["reason"]
    assert evaluate() == [0.0]


@pytest.mark.parametrize("defect", ["missing", "directory", "bad_row", "bad_encoding"])
def test_reference_errors_remain_execution_failures(bundle, evaluate, defect):
    path = bundle.reference / "ground_truth.csv"
    if defect in {"missing", "directory"}:
        path.unlink()
        if defect == "directory":
            path.mkdir()
    elif defect == "bad_row":
        path.write_text("PatientID,survival_time_days,deadstatus.event\npatient-0,invalid,1\n")
    else:
        path.write_bytes(b"\xff\xfe")
    result = run_verifier(bundle)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "Error" in result.stderr
    assert not bundle.report.exists()
    with pytest.raises(RuntimeError, match="Command failed with return code"):
        evaluate()


def test_unexpected_scorer_exception_is_not_a_zero(bundle, evaluate, monkeypatch):
    monkeypatch.setattr(task, "_read_script", lambda name: "raise RuntimeError('scorer crashed')\n")
    with pytest.raises(RuntimeError, match="scorer crashed"):
        evaluate()
