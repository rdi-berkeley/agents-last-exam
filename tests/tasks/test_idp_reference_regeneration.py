import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def evidence():
    value = os.environ.get("IDP_FRESH_REFERENCE_EVIDENCE")
    if not value:
        pytest.skip("Set IDP_FRESH_REFERENCE_EVIDENCE after full regeneration and validation")
    return Path(value)


def test_fresh_reference_is_full_pool_with_independent_validation(evidence):
    guest = evidence / "guest"
    completion = json.loads((guest / "full-pool-v1/complete.json").read_text())
    assert completion["ensembles"] == 95
    assert completion["conformers"] == 19000
    assert completion["raw_scores"] == 150
    for name in ["full-pool-v1", "validate-full-pool-v1"]:
        receipt = json.loads((guest / (name + "-receipt.json")).read_text())
        assert receipt["status"] == "complete"
        assert receipt["returncode"] == 0
    validation = json.loads((guest / "fresh-validation.json").read_text())
    assert validation["status"] == "passed"
    assert len(validation["direct_scorer_checks"]) == 150
    assert len(validation["arrays_checked"]) == 150
    assert len(validation["conformer_hashes"]) == 19000
    assert validation["aggregation_max_abs_difference"] < 1e-12
    stock = validation["stock_prediction_checks"]
    assert len(stock) == 19 and len({sample["protein"] for sample in stock}) == 19
    assert {sample["method"] for sample in stock} == {"Model" + str(index) for index in range(1, 6)}
    assert max(sample["max_abs_difference"] for sample in stock) < 1e-10
    for entry in validation["arrays_checked"]:
        path = guest / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        values = np.load(path, allow_pickle=False)
        assert values.shape[0] == 200 and values.shape[1] > 0
        assert list(values.shape) == entry["shape"]
        assert np.isfinite(values).all()


def test_new_gold_is_staged_from_completed_regeneration_not_legacy_or_solver(evidence):
    override = (
        evidence.parent / "data-overrides/life_sciences/idp_ensemble_scoring/default/reference"
    )
    generated = evidence / "guest/full-pool-v1/Final_Output.csv"
    staged = override / "Expected_Final_Output.csv"
    assert staged.read_bytes() == generated.read_bytes()
    provenance = json.loads((override / "reference-provenance.json").read_text())
    assert provenance["reference_sha256"] == hashlib.sha256(generated.read_bytes()).hexdigest()
    assert (
        provenance["legacy_reference_sha256"]
        == "32d8f5dfc6e70d2f9c141d7d465d0fe5c63900f5a173d8a42842fe65189acb0a"
    )
    assert provenance["reference_sha256"] != provenance["legacy_reference_sha256"]
    assert "Not authenticated legacy author gold" in provenance["provenance"]
    assert provenance["full_equal_pool"] == 200
    assert provenance["cs_stock_validation_proteins"] == 19
    with staged.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert all(
        math.isfinite(float(row[column])) and 0 <= float(row[column]) <= 1
        for row in rows
        for column in ["Total", "CS", "JC", "NOE/PRE"]
    )
    assert json.loads((evidence / "original-solver/eval_result.json").read_text())["score"] == 0
    assert not (evidence / "original-solver/output/Final_Output.csv").exists()
    original = json.loads((evidence / "original-solver-replay.json").read_text())
    for path, entry in original["artifacts"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == entry["sha256"]
        assert hashlib.sha256(Path(entry["preserved"]).read_bytes()).hexdigest() == entry["sha256"]


def test_fresh_structural_arrays_match_prior_independent_calculations(evidence):
    checked = json.loads((evidence / "structural-independent-comparison.json").read_text())
    assert len(checked) == 55
    assert max(row["max_abs_difference"] for row in checked) <= 2e-5
    local = json.loads((evidence / "local-validation.json").read_text())
    assert local["canonical_conformers"] == 19000
    assert local["canonical_experimental_files"] == 30
    assert local["independent_structural_arrays"] == 55
    replays = json.loads((evidence / "new-reference-replays.json").read_text())
    assert {row["label"]: row["result"]["score"] for row in replays} == {
        "new_reference": 1,
        "equivalent_csv": 1,
        "actual_original_missing_output": 0,
    }
