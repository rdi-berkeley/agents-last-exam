from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import pandas as pd

from tasks.engineering.mpc_control_building_v1.main import _parse_verifier_result
from tasks.engineering.mpc_control_building_v1.scripts.official_driver import _metrics, run
from tasks.engineering.mpc_control_building_v1.scripts.verify_outputs import (
    _metrics_from_timeseries,
    verify,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPO_ROOT / "tasks" / "engineering" / "mpc_control_building_v1"
DATA_ROOT = (
    REPO_ROOT / "task-data-hf" / "extracted" / "engineering" / "mpc_control_building_v1" / "base"
)

VALID_CONTROLLER = """\
import numpy as np

def _features(row):
    flow = float(row.get("flow_kg_s", 0.0))
    cooling_w = flow * 1006.0 * max(float(row["t_zone"]) - 13.0, 0.0)
    return [1.0, float(row["t_zone"]), float(row["t_outdoor"]),
            float(row["solar"]), float(row["occupancy"]), cooling_w]

def fit_model(training_rows):
    design = np.asarray([_features(row) for row in training_rows], dtype=float)
    targets = np.asarray([row["next_t_zone"] for row in training_rows], dtype=float)
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return {"coefficients": coefficients.tolist()}

def predict_next_temperature(model, observation):
    return float(np.dot(np.asarray(model["coefficients"]), np.asarray(_features(observation))))

def select_action(mode, observation, forecast, model):
    del forecast, model
    temperature = float(observation["t_zone"])
    setpoint = float(observation["setpoint"])
    hour = float(observation["hour_of_day"])
    if mode == "demand_response" and 17.0 <= hour < 20.0:
        return 0.0 if temperature < setpoint + 1.0 else 0.1
    if mode == "demand_response" and 14.0 <= hour < 17.0:
        return 0.3 if temperature > setpoint + 0.2 else 0.1
    if temperature > setpoint + 0.8:
        return 0.3
    if temperature > setpoint + 0.45:
        return 0.1
    return 0.0
"""


def test_failed_verifier_json_is_scored_as_candidate_failure() -> None:
    result = {
        "stdout": json.dumps(
            {
                "score": 0.0,
                "passed": False,
                "reason": "missing required output file",
            }
        ),
        "stderr": "",
        "return_code": 1,
    }

    assert _parse_verifier_result(result) == 0.0


@pytest.mark.parametrize(
    "result",
    [
        {"stdout": "", "stderr": "uv failed", "return_code": 1},
        {
            "stdout": json.dumps({"score": 0.0, "passed": False}),
            "stderr": "verifier crashed after printing",
            "return_code": 2,
        },
        {
            "stdout": json.dumps({"score": float("nan"), "passed": True}),
            "stderr": "",
            "return_code": 0,
        },
    ],
)
def test_evaluator_failures_are_not_scored_as_candidate_failures(result: dict) -> None:
    with pytest.raises(RuntimeError, match="MPC verifier failed"):
        _parse_verifier_result(result)


@pytest.mark.skipif(not DATA_ROOT.is_dir(), reason="Requires the gated MPC reference fixture")
def test_official_driver_fixture_is_deterministic_and_verifiable(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("SFH.idf", "Denver_current_TMY.epw", "task_spec.json"):
        shutil.copy2(DATA_ROOT / "input" / name, input_dir / name)
    shutil.copy2(TASK_ROOT / "scripts" / "official_driver.py", input_dir / "benchmark_driver.py")
    (input_dir / "controller.py").write_text(VALID_CONTROLLER)
    shutil.copy2(
        DATA_ROOT / "reference" / "baseline_data.csv", input_dir / "canonical_baseline.csv"
    )

    output_dir = tmp_path / "output"
    run(input_dir / "controller.py", input_dir / "canonical_baseline.csv", output_dir)
    first = verify(output_dir, input_dir, DATA_ROOT / "reference")
    assert first["passed"], first
    assert first["score"] > 0.0

    metrics_path = output_dir / "metrics_comparison.csv"
    original_metrics = metrics_path.read_bytes()
    metrics = pd.read_csv(metrics_path)
    metrics["peak_hour_avg_kw"] += 10
    metrics.to_csv(metrics_path, index=False)
    wrong_units = verify(output_dir, input_dir, DATA_ROOT / "reference")
    assert not wrong_units["passed"]
    assert "peak_hour_avg_kw disagrees" in wrong_units["reason"]
    metrics_path.write_bytes(original_metrics)

    baseline = (output_dir / "baseline_data.csv").read_text()
    (output_dir / "baseline_data.csv").write_text(baseline.replace("1552.2411672458215", "0", 1))
    tampered = verify(output_dir, input_dir, DATA_ROOT / "reference")
    assert not tampered["passed"]
    assert "canonical EnergyPlus baseline" in tampered["reason"]


@pytest.mark.parametrize("absolute_hours", [False, True])
def test_driver_and_scorer_share_electrical_peak_units(tmp_path, absolute_hours):
    frame = pd.DataFrame(
        {
            "hour": [
                index * 0.25 if absolute_hours else (index % 96) * 0.25 for index in range(2688)
            ],
            "cooling_w": [6000.0 if 68 <= index % 96 < 80 else 3000.0 for index in range(2688)],
            "t_zone": [24.0] * 2688,
            "setpoint": [24.0] * 2688,
        }
    )
    path = tmp_path / "trace.csv"
    frame.to_csv(path, index=False)
    measured = _metrics_from_timeseries(path)
    assert measured == pytest.approx(_metrics(frame))
    assert measured["peak_load_kw"] == 6.0
    assert measured["peak_hour_avg_kw"] == 2.0
