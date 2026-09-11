"""Regression cases for prose that resembles, but is not, a failed grading condition."""

import json

import pytest

from tasks.business_finance.legal_ma_consistency_audit_01.scripts import score_audit_report as legal
from tasks.health_medicine.healthcare_bias_audit_27a_public_replication_v1.scripts import (
    score_outputs as healthcare,
)


@pytest.mark.parametrize("separator", [". ", "\n\n", "; "])
def test_memo_negation_does_not_cross_statements(separator):
    reference = {
        "figure1b_percentile_97_before": 0.1,
        "figure1b_percentile_97_after": 0.2,
        "figure1b_percentile_97_ratio": 2.0,
        "table2_race_black_total_costs": 0.3,
        "table2_race_black_active_chronic_conditions": 0.4,
        "table2_race_black_best_worst_difference": 0.1,
        "table3_observed_program_frac_black": 0.2,
        "table3_predicted_health_in_cost_bin_frac_black": 0.3,
        "table3_highest_predicted_cost_frac_black": 0.4,
        "table3_worst_predicted_health_frac_black": 0.5,
    }
    conclusion = (
        "This public synthetic replication differs from the original paper's private data. "
        "The cost proxy understates Black patients' need. Diagnosis of the mechanism, not just retraining, "
        "supports a better need-based allocation. Figure 1b reports 0.1; Table 2 reports 0.3."
    )
    benign = "The alternative does not sacrifice cost targeting" + separator + conclusion
    assert healthcare._check_memo(benign, json.dumps(reference)) is None
    denial = "The cost proxy does not systematically understate Black patients' need. " + conclusion
    result = healthcare._check_memo(denial, json.dumps(reference))
    assert result is not None and result.reason == "audit_memo.md: incorrect_conclusion"


def test_report_prose_and_formatted_evidence_do_not_become_extra_claims(monkeypatch):
    monkeypatch.setattr(
        legal,
        "TARGET_RULES",
        {
            "required": {
                "match": lambda text, _: (
                    "required issue" in text.lower() and "quoted fact" in text.lower()
                ),
                "evidence": lambda _: True,
                "location": lambda _: True,
            }
        },
    )
    report = """Every finding below was checked against the source.

## Finding 1: required issue
- **Evidence:** the quoted fact demonstrates the numerical error.
- **Why it is inconsistent:** explanation of the discrepancy.

## Checks that passed
To bound the findings above, the following items were cross-checked
and found to be consistent (no finding is reported for them).
- No numerical mismatch found in the payment schedule.
"""
    reference = {"required_findings": [{"id": "required"}]}
    result = legal.score_report_text(report_text=report, reference_payload=reference)
    assert result["score"] == 1.0 and result["false_positive_count"] == 0

    extra = "\n## Finding 2: unsupported numerical mismatch\nEvidence: invented discrepancy."
    result = legal.score_report_text(report_text=report + extra, reference_payload=reference)
    assert result["score"] == 0.8 and result["false_positive_count"] == 1
