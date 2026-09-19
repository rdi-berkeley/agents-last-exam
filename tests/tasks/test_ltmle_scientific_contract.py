import copy
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import pytest

from tasks.health_medicine.ltmle_targeted_bootstrap_simulation_study.scripts import (
    score_outputs as scorer,
    verify_hidden_smoke as verifier,
)
from tests.tasks.test_ltmle_evaluator_integrity import (
    METHODS,
    PLAN,
    RAW_COLUMNS,
    bundle as independent_bundle,
    setup_main,
    write_csv,
)


@pytest.fixture(name="bundle")
def scientific_bundle(tmp_path):
    return independent_bundle.__wrapped__(tmp_path)


def validate(rows, fixture):
    return verifier._validate_raw_results(
        raw_fieldnames=RAW_COLUMNS,
        raw_rows=rows,
        plan_rows=[PLAN],
        canonical_tau_true_by_scenario={"independent": 0.2},
        hidden_contract=fixture.contract["hidden_smoke"],
    )


@pytest.mark.parametrize("method", [METHODS[2], METHODS[3]])
def test_shared_point_violation_rejected_even_inside_reference_tolerance(bundle, method):
    rows = copy.deepcopy(bundle.raw)
    row = next(row for row in rows if row["method"] == method)
    for column in ("estimate", "conf_low", "conf_high"):
        row[column] = str(float(row[column]) + 0.001)
    assert any(
        error["type"] == "shared_full_data_point_mismatch" for error in validate(rows, bundle)
    )


def test_bootstrap_shared_se(bundle):
    rows = copy.deepcopy(bundle.raw)
    rows[2]["std_error"] = ".04"
    assert any(error["type"] == "shared_bootstrap_se_mismatch" for error in validate(rows, bundle))


@pytest.mark.parametrize(
    "column,value,reason",
    [
        ("std_error", "-.1", "invalid_scientific_range"),
        ("conf_low", "2", "invalid_scientific_range"),
        ("estimate", "1.01", "invalid_scientific_range"),
        ("conf_high", ".9", "wald_interval_mismatch"),
    ],
)
def test_invalid_scientific_values(bundle, column, value, reason):
    rows = copy.deepcopy(bundle.raw)
    rows[0][column] = value
    assert any(error["type"] == reason for error in validate(rows, bundle))


def test_quantile_need_not_be_centered_or_contain_point(bundle):
    rows = copy.deepcopy(bundle.raw)
    rows[2]["conf_low"] = ".4"
    rows[2]["conf_high"] = ".5"
    assert not validate(rows, bundle)
    rows[2]["conf_high"] = "1.01"
    assert any(
        error["type"] == "quantile_outside_estimand_range" for error in validate(rows, bundle)
    )


def test_stale_contract_is_infrastructure_error(bundle):
    contract = copy.deepcopy(bundle.contract)
    contract.pop("scientific_contract_version")
    (bundle.reference / "evaluation_contract.json").write_text(json.dumps(contract))
    with pytest.raises(verifier.EvaluatorConfigurationError, match="version_mismatch"):
        verifier._load_evaluator_contract(bundle.input, bundle.reference, bundle.reference)


def test_inconsistent_reference_rejected_even_when_hashes_refrozen(bundle):
    rows = copy.deepcopy(bundle.raw)
    rows[2]["estimate"] = ".8"
    write_csv(bundle.reference / "raw_results.csv", RAW_COLUMNS, rows)
    contract = copy.deepcopy(bundle.contract)
    contract["hidden_smoke"]["frozen_expected_hashes"]["output_test_pos"]["raw_results.csv"] = (
        hashlib.sha256((bundle.reference / "raw_results.csv").read_bytes()).hexdigest()
    )
    (bundle.reference / "evaluation_contract.json").write_text(json.dumps(contract))
    with pytest.raises(
        verifier.EvaluatorConfigurationError, match="reference_scientific_invariants"
    ):
        verifier._load_evaluator_contract(bundle.input, bundle.reference, bundle.reference)


def test_summary_derivation_tolerance_not_reference_tolerance(bundle, monkeypatch, capsys):
    setup_main(monkeypatch, bundle)
    bundle.contract["public_benchmark"]["metric_tolerances"]["bias"] = 0.03
    bundle.contract["hidden_smoke"]["summary_metric_tolerances"]["bias"] = 0.03
    (bundle.reference / "evaluation_contract.json").write_text(json.dumps(bundle.contract))
    columns, rows = verifier._read_csv(bundle.candidate / "summary.csv")
    rows[0]["bias"] = ".01"
    write_csv(bundle.candidate / "summary.csv", columns, rows)
    assert verifier.main() == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "hidden_summary_derivation_failed"


def test_public_raw_positive_and_negative(bundle):
    raw = (bundle.candidate / "raw_results.csv").read_text()
    summary = (bundle.candidate / "summary.csv").read_text()
    assert scorer.validate_public_raw(raw, summary, bundle.contract).passed
    assert not scorer.validate_public_raw(
        raw.replace("never_treat", "always_treat"), summary, bundle.contract
    ).passed
    assert not scorer.validate_public_raw(raw + "short,row\n", summary, bundle.contract).passed


def test_unknown_scenario_labels_and_row_order(bundle):
    assert not validate(list(reversed(bundle.raw)), bundle)


def test_duplicate_or_missing_replication(bundle):
    assert validate(bundle.raw[:-1], bundle)
    assert validate(bundle.raw + [bundle.raw[0]], bundle)


def test_collapsed_policy_numerical_negative(bundle):
    rows = copy.deepcopy(bundle.raw)
    for row in rows:
        for name in ("estimate", "std_error", "conf_low", "conf_high"):
            row[name] = "0"
    assert not validate(rows, bundle)
    derived = verifier._recompute_summary_rows(
        raw_rows=rows, scenario_levels=["independent"], method_levels=METHODS
    )
    _, expected = verifier._read_csv(bundle.reference / "summary.csv")
    assert verifier._compare_summary_maps(
        candidate_rows=derived,
        reference_rows=expected,
        contract_section=bundle.contract["public_benchmark"],
    )


def test_public_contract_and_six_script_interface():
    root = Path(verifier.__file__).parents[1]
    text = (root / "SCIENTIFIC_CONTRACT.md").read_text()
    for token in (
        "sample.int",
        "cf_sub",
        "cf_r",
        "s+100",
        "s+200",
        "s+5000",
        "analyze_part2_results_longitudinal",
        "analyze_results_by_sample_size",
        "fixture_smoke_plan.csv",
        "1.1e-6",
        "Do not",
        "100 replications",
    ):
        assert token in text
    for filename in verifier.REQUIRED_SCRIPT_NAMES:
        if (root / "scripts/reference").is_dir():
            assert (root / "scripts/reference" / filename).is_file()


def test_small_wrong_se_cannot_hide_inside_summary_tolerances(bundle, monkeypatch, capsys):
    setup_main(monkeypatch, bundle)
    changed = copy.deepcopy(bundle.raw)
    selected = changed[1]
    selected["std_error"] = ".031"
    normal = statistics.NormalDist().inv_cdf(0.975)
    selected["conf_low"] = str(float(selected["estimate"]) - normal * 0.031)
    selected["conf_high"] = str(float(selected["estimate"]) + normal * 0.031)
    write_csv(bundle.candidate / "raw_results.csv", RAW_COLUMNS, changed)
    assert verifier.main() == 1
    assert json.loads(capsys.readouterr().out)["reason"] == "hidden_raw_numerical_mismatch"


def test_runtime_timeout_is_not_solver_failure(bundle, monkeypatch, capsys):
    setup_main(monkeypatch, bundle)

    def failed_preparation(**kwargs):
        raise subprocess.TimeoutExpired("runtime probe", 120)

    monkeypatch.setattr(verifier, "_materialize_candidate_bundle", failed_preparation)
    assert verifier.main() == 2
    assert json.loads(capsys.readouterr().out)["error_type"] == "evaluator"


def test_mutated_dgp_is_evaluator_error(bundle):
    (
        bundle.input / "LTMLE_Targeted_Bootstrap_Task_INPUT/01_data_generation_longitudinal.R"
    ).write_text("modified")
    with pytest.raises(verifier.EvaluatorConfigurationError, match="input_hash_mismatch"):
        verifier._load_evaluator_contract(bundle.input, bundle.reference, bundle.reference)
