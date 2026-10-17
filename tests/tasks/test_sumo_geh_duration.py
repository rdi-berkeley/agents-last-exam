import importlib.util
import math
import os
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def evaluator():
    default = Path(__file__).parents[2] / (
        "task-data-hf/extracted/engineering/sumo_urban_am_peak_calibration/"
        "base/reference/evaluator_only/evaluate.py"
    )
    path = Path(os.environ.get("ALE_SUMO_EVALUATOR", default))
    if not path.is_file():
        pytest.skip("Requires the private SUMO evaluator; set ALE_SUMO_EVALUATOR")
    spec = importlib.util.spec_from_file_location("sumo_duration_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rerun_counts(seed_counts, duration=3600):
    return {
        "seeds_completed": list(seed_counts),
        "per_seed": {
            seed: {
                "holdout_detectors": {
                    (
                        "H_fixture",
                        index * duration / len(counts),
                        (index + 1) * duration / len(counts),
                    ): {"n": count}
                    for index, count in enumerate(counts)
                }
            }
            for seed, counts in seed_counts.items()
        },
    }


@pytest.mark.parametrize("counts,passed", [([100, 100, 100, 100], True), ([70, 70, 70, 70], False)])
def test_hourly_geh_does_not_combine_simulated_and_observed_durations(evaluator, counts, passed):
    observed = {f"H_fixture|{index * 900}|{(index + 1) * 900}": 100 for index in range(4)}
    rerun = rerun_counts({17: counts, 31: counts, 57: counts})
    result = evaluator.gate_8_geh_lt5_holdout(rerun, observed, {"H_fixture"})
    (row,) = result.metric["rows"]
    assert row["obs_vph"] == 400
    assert row["sim_vph"] == sum(counts)
    assert row["geh"] == pytest.approx(
        math.sqrt(2 * (sum(counts) - 400) ** 2 / (sum(counts) + 400))
    )
    assert result.passed is passed
    gate6 = evaluator.gate_6_flow_rmse_holdout(rerun, observed, {"H_fixture"})
    assert gate6.metric["ratio"] == pytest.approx(abs(sum(counts) - 400) / 400)


@pytest.mark.parametrize("duration,bins", [(900, 1), (1800, 2), (3600, 4)])
def test_hourly_units_are_independent_of_bin_count_and_horizon(evaluator, duration, bins):
    rerun = rerun_counts({17: [100] * bins, 31: [80] * bins}, duration)
    observed = {f"H_fixture|0|{duration}": 90 * bins}
    result = evaluator.gate_8_geh_lt5_holdout(rerun, observed, {"H_fixture"})
    assert result.metric["rows"] == [{"det": "H_fixture", "obs_vph": 360, "sim_vph": 360, "geh": 0}]
    assert result.passed


def test_each_seed_uses_its_own_duration_and_order_does_not_change_flow(evaluator):
    rerun = rerun_counts({17: [100, 100], 31: [200, 200]})
    rerun["per_seed"][17]["holdout_detectors"] = {("H_fixture", 0, 1800): {"n": 200}}
    observed = {"H_fixture|0|3600": 400}
    first = evaluator.gate_8_geh_lt5_holdout(rerun, observed, {"H_fixture"})
    rerun["seeds_completed"].reverse()
    second = evaluator.gate_8_geh_lt5_holdout(rerun, observed, {"H_fixture"})
    assert first.metric == second.metric
    assert first.metric["rows"][0]["sim_vph"] == 400
    assert first.metric["rows"][0]["geh"] == 0
