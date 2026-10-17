import csv
import json
import math
import os
import statistics
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def evidence():
    path = os.environ.get("LTMLE_REPAIR_EVIDENCE")
    if not path:
        pytest.skip("set LTMLE_REPAIR_EVIDENCE to the independently computed v2 bundle")
    root = Path(path)
    assert root.is_dir()
    return root


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("label,count,bootstrap_count", [("public", 200, 499), ("hidden", 6, 99)])
def test_computed_references_independent_mathematics(evidence, label, count, bootstrap_count):
    output = evidence / label / "output"
    completion = json.loads((output / "completion.json").read_text())
    assert completion["return_code"] == 0
    raw = rows(output / "raw_results.csv")
    assert len(raw) == 4 * count
    normal = statistics.NormalDist().inv_cdf(0.975)
    grouped = {}
    paired = {}
    for row in raw:
        grouped.setdefault((row["method"], row["scenario"]), []).append(row)
        paired.setdefault((row["scenario"], row["replicate_id"]), {})[row["method"]] = row
        assert int(row["bootstrap_B"]) == bootstrap_count
        assert int(row["bootstrap_seed"]) == int(row["seed"]) + 5000
        estimate, error, lower, upper = [
            float(row[key]) for key in ("estimate", "std_error", "conf_low", "conf_high")
        ]
        assert all(math.isfinite(value) for value in (estimate, error, lower, upper))
        assert -1 <= estimate <= 1 and error >= 0 and lower <= upper
        if row["method"] != "Targeted Bootstrap (Quantile)":
            assert lower == pytest.approx(estimate - normal * error, abs=1e-10)
            assert upper == pytest.approx(estimate + normal * error, abs=1e-10)
    for methods in paired.values():
        full = methods["Full-Data Targeting Contrast (EIF)"]
        wald = methods["Targeted Bootstrap (Wald)"]
        quantile = methods["Targeted Bootstrap (Quantile)"]
        assert full["estimate"] == wald["estimate"] == quantile["estimate"]
        assert wald["std_error"] == quantile["std_error"]
    for summary in rows(output / "summary.csv"):
        selected = grouped[(summary["method"], summary["scenario"])]
        estimates = [float(row["estimate"]) for row in selected]
        mean = math.fsum(estimates) / len(estimates)
        empirical = math.sqrt(
            math.fsum((value - mean) ** 2 for value in estimates) / (len(estimates) - 1)
        )
        estimated = math.fsum(float(row["std_error"]) for row in selected) / len(selected)
        derived = dict(
            bias=mean - float(selected[0]["tau_true"]),
            empirical_se=empirical,
            estimated_se=estimated,
            se_ratio=estimated / empirical,
            coverage=sum(
                float(row["conf_low"]) <= float(row["tau_true"]) <= float(row["conf_high"])
                for row in selected
            )
            / len(selected),
            ci_width=math.fsum(float(row["conf_high"]) - float(row["conf_low"]) for row in selected)
            / len(selected),
        )
        for key, value in derived.items():
            assert float(summary[key]) == pytest.approx(value, abs=5.1e-7)


def test_actual_independent_numerical_audit(evidence):
    result = json.loads((evidence / "independent-numerics.json").read_text())
    assert result["return_code"] == 0
    assert "PASS receipts=6" in result["stdout"]
    assert "wrong_weight_delta=" in result["stdout"]
    assert "covariance_delta=" in result["stdout"]


@pytest.mark.parametrize(
    "mode,passed,reason",
    [
        ("positive", True, "hidden_smoke_passed"),
        ("independent", True, "hidden_smoke_passed"),
        ("negative", False, "hidden_raw_contract_failed"),
        ("relabeled-negative", False, "hidden_raw_numerical_mismatch"),
        ("unpaired-negative", False, "hidden_raw_numerical_mismatch"),
        ("raw-weight-negative", False, "hidden_raw_numerical_mismatch"),
    ],
)
def test_actual_verifier_replays(evidence, mode, passed, reason):
    receipt = json.loads((evidence / f"verifier-{mode}.json").read_text())
    result = receipt["result"]
    payload = json.loads(result["stdout"])
    assert result["return_code"] == (0 if passed else 1)
    assert payload["passed"] is passed
    assert payload["reason"] == reason
    assert "error_type" not in payload


def test_collapsed_policy_was_computed_not_relabeling_reference(evidence):
    output = evidence / "negative/output"
    assert json.loads((output / "completion.json").read_text())["return_code"] == 0
    raw = rows(output / "raw_results.csv")
    assert len(raw) == 24
    for row in raw:
        assert row["reference_policy"] == row["target_policy"] == "always_treat"
        assert all(
            float(row[key]) == 0 for key in ("estimate", "std_error", "conf_low", "conf_high")
        )


def test_old_provenance_preserved_and_new_reference_versioned(evidence):
    legacy = evidence / "legacy"
    assert (legacy / "reference/source_manifest.json").is_file()
    assert (legacy / "public_benchmark/reference_summary.csv").is_file()
    contract = json.loads((evidence / "staged/base/reference/evaluation_contract.json").read_text())
    assert contract["scientific_contract_version"] == "ltmle-cumulative-v2-20260910"
    assert len(contract["input_sha256"]) >= 20
    assert len(contract["hidden_smoke"]["lmtp_function_fingerprints"]) >= 12
    assert contract["public_benchmark"]["metric_tolerances"]["bias"] == 0.03


def test_parallel_public_points_match_sequential_pilot(evidence):
    public = {
        (row["n"], row["seed"], row["method"]): row
        for row in rows(evidence / "public/output/raw_results.csv")
    }
    for pilot in rows(evidence / "pilot/output/raw_results.csv"):
        computed = public[(pilot["n"], pilot["seed"], pilot["method"])]
        assert float(pilot["estimate"]) == pytest.approx(float(computed["estimate"]), abs=1e-12)
        if pilot["method"].endswith("(EIF)"):
            assert float(pilot["std_error"]) == pytest.approx(
                float(computed["std_error"]), abs=1e-12
            )


@pytest.mark.parametrize("label,score", [("positive", [1.0]), ("negative", [0.0])])
def test_actual_task_entrypoint(evidence, label, score):
    result = json.loads((evidence / f"entrypoint-{label}.json").read_text())
    assert result["score"] == score


@pytest.mark.parametrize("name", ["independent-boundary-numerics.json", "api-rng-check.json"])
def test_actual_boundary_and_rng_audits(evidence, name):
    result = json.loads((evidence / name).read_text())
    assert result["return_code"] == 0
    assert "PASS" in result["stdout"]
