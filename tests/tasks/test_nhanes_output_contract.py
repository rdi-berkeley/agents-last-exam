from __future__ import annotations

import csv
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import pytest

from tasks.health_medicine.nhanes_confounder_sensitivity_analysis.scripts import score_outputs


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "tasks/health_medicine/nhanes_confounder_sensitivity_analysis"
DATA_DIR = ROOT / "task-data-hf/extracted/health_medicine/nhanes_confounder_sensitivity_analysis/base"


@pytest.fixture
def task_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "score_outputs", score_outputs)
    spec = importlib.util.spec_from_file_location("nhanes_contract_task", TASK_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_prompt_and_task_card_agree_on_summary_definition(task_module):
    runtime = task_module.NHANESConfounderSensitivityConfig().task_description
    card = json.loads((TASK_DIR / "task_card.json").read_text())["taskPrompt"]
    runtime_definition = runtime.split("3. In every summary row,", 1)[1].split("4. Write only", 1)[0]
    card_definition = card.split("3. In every summary row,", 1)[1].split("4. Write only", 1)[0]

    assert runtime_definition == card_definition
    for required in (
        "one-unit increase in `MM_count`",
        "unpenalized maximum likelihood",
        "model-based (non-robust)",
        "odds ratio `exp(beta)`",
        "95% Wald",
        "exp(beta - z * SE)",
        "exp(beta + z * SE)",
        "2 * Phi(-abs(beta / SE))",
        "blank for both Model A rows",
        "100 * (OR_B / OR_A_subset - 1)",
    ):
        assert required in runtime


def test_task_card_documents_actual_partial_scoring():
    evaluation = json.loads((TASK_DIR / "task_card.json").read_text())["evaluation"]

    for weight in ("0.15", "0.10", "0.30", "0.25"):
        assert weight in evaluation
    assert "binary evaluator" not in evaluation
    assert "odds-ratio scale" in evaluation


@pytest.fixture
def reference_bundle():
    if not (DATA_DIR / "reference/sensitivity_summary.csv").exists():
        pytest.skip("NHANES release data is not installed")
    return {
        "reference_subset_csv": (DATA_DIR / "reference/hpyl_nsaid_subset.csv").read_text(),
        "reference_summary_csv": (DATA_DIR / "reference/sensitivity_summary.csv").read_text(),
    }


def test_reference_keeps_full_credit(reference_bundle):
    result = score_outputs.score_output_bundle(
        candidate_subset_csv=reference_bundle["reference_subset_csv"],
        candidate_summary_csv=reference_bundle["reference_summary_csv"],
        **reference_bundle,
    )

    assert result.score == 1.0


@pytest.mark.parametrize("change", ["log_odds", "wrong_interval", "wrong_change_direction"])
def test_clarification_preserves_rejection_of_wrong_statistics(reference_bundle, change):
    reader = csv.DictReader(io.StringIO(reference_bundle["reference_summary_csv"]))
    rows = list(reader)
    if change == "log_odds":
        for row in rows:
            for field in ("estimate", "ci_low", "ci_high"):
                row[field] = str(math.log(float(row[field])))
    elif change == "wrong_interval":
        rows[0]["ci_low"] = rows[0]["estimate"]
    else:
        rows[2]["or_change_pct_vs_modelA_subset"] = str(-float(rows[2]["or_change_pct_vs_modelA_subset"]))
    candidate = io.StringIO()
    writer = csv.DictWriter(candidate, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    result = score_outputs.score_output_bundle(
        candidate_subset_csv=reference_bundle["reference_subset_csv"],
        candidate_summary_csv=candidate.getvalue(),
        **reference_bundle,
    )

    assert 0 < result.score < 1
    assert result.details["summary_stats"]["gate_score"] < 1
    assert result.details["subset_numeric"]["gate_score"] == 1
