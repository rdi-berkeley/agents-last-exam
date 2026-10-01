from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.engineering.sumo_urban_am_peak_calibration import main as task
from tasks.engineering.sumo_urban_am_peak_calibration.scripts import verify_submission as wrapper


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def evaluator():
    path = Path(
        os.environ.get(
            "SUMO_EVALUATOR_PATH",
            ROOT
            / "task-data-hf/extracted/engineering/sumo_urban_am_peak_calibration"
            / "base/reference/evaluator_only/evaluate.py",
        )
    )
    if not path.is_file():
        pytest.skip("SUMO evaluator data is not installed")
    spec = importlib.util.spec_from_file_location("sumo_public_contract_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_and_card_disclose_identical_existing_criteria():
    config = task.SumoUrbanCalibrationConfig()
    prompt = config.task_description
    for actual, portable in (
        (config.input_dir, "base/input"),
        (config.software_dir, "base/software"),
        (config.remote_output_dir, "base/output"),
    ):
        prompt = prompt.replace(actual, portable)
    card = json.loads((Path(task.__file__).parent / "task_card.json").read_text())
    assert card["taskPrompt"] == prompt
    for clause in (
        "all 12 gates must pass",
        "deduplicated union",
        "at least 3 integers",
        "3 evaluator-hidden seeds",
        "completed tripinfo records, not scheduled vehicle entries",
        "RMSE / mean observed flow <= 12%",
        "matched detector/15-minute bin",
        "GEH < 5 on at least 85% of holdout detectors",
        "combined simulated and observed interval duration",
        "every declared and hidden seed",
        "full 0-3600-second horizon",
        "RMSE across rerun seeds <= 15%",
        "max(1% of the rerun mean, 0.5 vehicles/hour)",
        "max(2% of the rerun mean, 0.5 seconds)",
    ):
        assert clause in prompt


def test_dedicated_public_override_has_all_rows_and_matching_contract():
    input_dir = os.environ.get("SUMO_PUBLIC_INPUT_DIR")
    if not input_dir:
        pytest.skip("Set SUMO_PUBLIC_INPUT_DIR to validate a staged public-data override")
    prompt = (Path(input_dir) / "task_prompt.md").read_text()
    contract = json.loads((Path(input_dir) / "output_contract.json").read_text())
    rows = [
        int(line.split("|")[1].strip())
        for line in prompt.splitlines()
        if line.startswith("| ") and line.split("|")[1].strip().isdigit()
    ]
    assert rows == list(range(1, 13))
    assert task.PUBLIC_RERUN_CONTRACT in prompt
    assert contract["evaluator_rerun_contract"] == task.PUBLIC_RERUN_CONTRACT
    assert contract["scoring_model"] == {
        "type": "hard_gates",
        "gate_count": 12,
        "requires_all_pass": True,
    }
    assert "`calibration_report.json` plus 3 evaluator-hidden seeds (deduplicated)" in prompt
    assert "includes all simulations and gate aggregation" in prompt


@pytest.mark.parametrize("trips,passed", [(94, False), (95, True), (105, True), (106, False)])
def test_existing_trip_total_boundary(evaluator, trips, passed):
    rerun = {"seeds_completed": [101], "per_seed": {101: {"tripinfo": [{}] * trips}}}
    assert evaluator.gate_5_trip_total(rerun, {"am_peak_trip_total": 100}).passed is passed


@pytest.mark.parametrize("count,passed", [(111, True), (112, True), (113, False)])
def test_existing_holdout_rmse_boundary(evaluator, count, passed):
    rerun = {
        "seeds_completed": [101, 1001],
        "per_seed": {
            seed: {"holdout_detectors": {("holdout", 0, 900): {"n": value}}}
            for seed, value in ((101, count - 1), (1001, count + 1))
        },
    }
    result = evaluator.gate_6_flow_rmse_holdout(rerun, {"holdout|0|900": 100}, {"holdout"})
    assert result.passed is passed
    assert result.metric["mean_obs_vph"] == 400
    assert result.metric["ratio"] == pytest.approx(abs(count - 100) / 100)


@pytest.mark.parametrize(
    "values,passed", [([100, 100], True), ([85, 115], True), ([84, 116], False)]
)
def test_existing_corridor_rmse_is_across_seeds_not_just_mean(evaluator, values, passed):
    rerun = {
        "seeds_completed": [101, 1001],
        "per_seed": {
            seed: {"corridor_edges": {("edge", 0, 3600): {"traveltime": value}}}
            for seed, value in zip([101, 1001], values)
        },
    }
    reference = {"corridor": {"edges": ["edge"], "bluetooth_tt_mean_s": 100}}
    assert evaluator.gate_7_corridor_tt_rmse(rerun, reference, 0, 3600).passed is passed


@pytest.mark.parametrize("matching,passed", [(16, False), (17, True), (20, True)])
def test_existing_geh_fraction_and_strict_threshold(evaluator, matching, passed):
    identifiers = [f"holdout_{index}" for index in range(20)]
    holdout = {
        (identifier, 0, 3600): {"n": 25 if index < matching else 0}
        for index, identifier in enumerate(identifiers)
    }
    rerun = {"seeds_completed": [101], "per_seed": {101: {"holdout_detectors": holdout}}}
    observed = {f"{identifier}|0|3600": 25 for identifier in identifiers}
    result = evaluator.gate_8_geh_lt5_holdout(rerun, observed, set(identifiers))
    assert result.passed is passed
    assert result.metric["fraction"] == matching / 20
    assert all(row["obs_vph"] == 12.5 for row in result.metric["rows"])
    assert all(row["geh"] == 5 for row in result.metric["rows"][matching:])


@pytest.mark.parametrize(
    "completed,error,passed",
    [
        ([101, 102, 103, 1001], None, True),
        ([101, 102, 103], None, False),
        ([101, 102, 103, 1001], "simulation error", False),
    ],
)
def test_existing_rerun_success_gate(evaluator, completed, error, passed):
    rerun = {"seeds_completed": completed}
    if error:
        rerun["error"] = error
    assert evaluator.gate_9_rerun_success(rerun, [101, 102, 103, 1001]).passed is passed


@pytest.mark.parametrize("actual,tolerance", [(25, 0.5), (100, 1.0)])
@pytest.mark.parametrize("outside", [False, True])
def test_existing_flow_report_tolerance_and_declared_seed_subset(
    evaluator, actual, tolerance, outside
):
    report = {
        "agent_seeds": [101],
        "detectors_public": [
            {
                "detector_id": "public",
                "simulated_flow_vph_mean": actual + tolerance + outside * 0.001,
            }
        ],
    }
    rerun = {
        "seeds_completed": [101, 1001],
        "per_seed": {
            seed: {"public_detectors": {("public", 0, 3600): {"n": value}}}
            for seed, value in ((101, actual), (1001, 999))
        },
    }
    assert evaluator.gate_10_report_flow_consistency(report, rerun, {}).passed is not outside


@pytest.mark.parametrize("actual,tolerance", [(25, 0.5), (100, 2.0)])
@pytest.mark.parametrize("outside", [False, True])
def test_existing_corridor_report_tolerance_and_horizon(evaluator, actual, tolerance, outside):
    report = {
        "agent_seeds": [101],
        "corridors": [
            {
                "corridor_id": "corridor",
                "edges": ["edge"],
                "simulated_tt_mean_s": actual + tolerance + outside * 0.001,
            }
        ],
    }
    rerun = {
        "seeds_completed": [101, 1001],
        "per_seed": {
            seed: {
                "corridor_edges": {
                    ("edge", 0, 3600): {"traveltime": value},
                    ("edge", 0, 900): {"traveltime": 777},
                }
            }
            for seed, value in ((101, actual), (1001, 999))
        },
    }
    assert (
        evaluator.gate_11_report_corridor_consistency(report, rerun, 0, 3600).passed is not outside
    )


@pytest.mark.parametrize("failed_gate", [None, *range(1, 13)])
def test_wrapper_keeps_all_twelve_gates_mandatory(tmp_path, monkeypatch, capsys, failed_gate):
    report = {
        "all_passed": failed_gate is None,
        "passed_count": 12 if failed_gate is None else 11,
        "total_gates": 12,
        "gates": [{"name": str(index), "passed": index != failed_gate} for index in range(1, 13)],
    }
    monkeypatch.setattr(
        wrapper,
        "_parse_args",
        lambda: SimpleNamespace(
            evaluator_python=Path(sys.executable),
            evaluator_script=tmp_path / "evaluate.py",
            submission_dir=tmp_path / "submission",
            ground_truth_dir=tmp_path / "ground_truth",
            tmp_dir=tmp_path / "scratch",
            hidden_seeds=[1001, 1002, 1003],
        ),
    )
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(report),
            stderr="",
            returncode=0 if failed_gate is None else 1,
        ),
    )
    assert wrapper.main() == (0 if failed_gate is None else 1)
    payload = json.loads(capsys.readouterr().out)
    assert payload["score"] == (1.0 if failed_gate is None else 0.0)
    assert payload["failed_gates"] == ([] if failed_gate is None else [str(failed_gate)])
