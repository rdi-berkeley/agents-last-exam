from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

VERIFIER_PATH = (
    Path(__file__).resolve().parents[2]
    / "tasks/business_finance/digital_marketing_ab_test_analysis_1/scripts/verify_ab_test_outputs.py"
)
SPEC = importlib.util.spec_from_file_location("digital_marketing_ab_verifier", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
extract_recommendation = VERIFIER.extract_recommendation
component_scores = VERIFIER.component_scores
validate_result_rows = VERIFIER.validate_result_rows
validate_results_tsv = VERIFIER.validate_results_tsv
validate_report_md = VERIFIER.validate_report_md

EXPECTED_STATS = {
    "opened_rate": {
        "control_rate": 0.20,
        "treatment_rate": 0.234,
        "absolute_lift": 0.034,
        "relative_lift_pct": 17.0,
        "ci_lower_95": 0.01,
        "ci_upper_95": 0.058,
        "z_statistic": 3.7,
        "p_value_raw": 0.0002,
        "significant_at_05": True,
        "is_primary": True,
    },
    "clicked_rate": {
        "control_rate": 0.10,
        "treatment_rate": 0.12,
        "absolute_lift": 0.02,
        "relative_lift_pct": 20.0,
        "ci_lower_95": 0.001,
        "ci_upper_95": 0.039,
        "z_statistic": 2.0,
        "p_value_raw": 0.04,
        "significant_at_05": True,
        "is_primary": False,
    },
    "converted_rate": {
        "control_rate": 0.0,
        "treatment_rate": 0.0,
        "absolute_lift": 0.0,
        "relative_lift_pct": 0.0,
        "ci_lower_95": 0.0,
        "ci_upper_95": 0.0,
        "z_statistic": 0.0,
        "p_value_raw": 1.0,
        "significant_at_05": False,
        "is_primary": False,
    },
    "unsubscribed_rate": {
        "control_rate": 0.001,
        "treatment_rate": 0.003,
        "absolute_lift": 0.002,
        "relative_lift_pct": 200.0,
        "ci_lower_95": 0.0001,
        "ci_upper_95": 0.0039,
        "z_statistic": 2.0,
        "p_value_raw": 0.04,
        "significant_at_05": True,
        "is_primary": False,
    },
}
RESULT_FIELDS = [
    "metric",
    "is_primary",
    "control_rate",
    "treatment_rate",
    "absolute_lift",
    "relative_lift_pct",
    "ci_lower_95",
    "ci_upper_95",
    "z_statistic",
    "p_value_raw",
    "significant_at_05",
    "bh_rank",
    "bh_threshold",
    "bh_significant",
]


def write_results(path: Path, overrides: dict[str, dict[str, object]] | None = None) -> None:
    bh = VERIFIER.bh_correct(
        {metric: EXPECTED_STATS[metric]["p_value_raw"] for metric in VERIFIER.SECONDARY_METRICS}
    )
    rows = []
    for metric, expected in EXPECTED_STATS.items():
        row = {"metric": metric, **expected}
        if metric == VERIFIER.PRIMARY_METRIC:
            row.update({"bh_rank": "", "bh_threshold": "", "bh_significant": ""})
        else:
            row.update(bh[metric])
        row.update((overrides or {}).get(metric, {}))
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("Recommendation: SHIP", "ship"),
        ("## Recommendation\n**HOLD** the rollout.", "hold"),
        ("## Recommendation\nHold rather than ship it now.", "hold"),
        ("## Recommendation\nDo not ship this rollout.", "hold"),
        ("## Recommendation\nShip this rollout; do not hold it.", "ship"),
        ("## Recommendation\nShip or hold after another review.", None),
    ],
)
def test_extract_recommendation_resolves_section_semantics(
    section: str, expected: str | None
) -> None:
    assert extract_recommendation(section) == expected


def test_extract_recommendation_ignores_ship_outside_recommendation() -> None:
    report = "A ship carries cargo.\n\n## Recommendation\nHOLD the rollout."

    assert extract_recommendation(report) == "hold"


@pytest.mark.parametrize(
    ("recommendation", "expected"),
    [
        ("SHIP: launch the treatment.", True),
        ("HOLD: collect reliable click data first.", False),
        ("Hold the rollout rather than ship it now.", False),
        ("Do not ship this rollout.", False),
    ],
)
def test_validate_report_rejects_semantically_opposite_recommendations(
    tmp_path, recommendation: str, expected: bool
) -> None:
    report = tmp_path / "experiment_report.md"
    report.write_text(
        "Required per-arm sample size: 3,122\n\n"
        "Observed primary lift: 3.40 percentage points.\n\n"
        f"## Recommendation\n{recommendation}\n",
        encoding="utf-8",
    )
    expected_stats = {
        "opened_rate": {"significant_at_05": True},
        "unsubscribed_rate": {"absolute_lift": 0.002722},
    }

    assert validate_report_md(report, expected_stats, required_n=3122) is expected


def test_results_accept_blank_inference_for_degenerate_metric(tmp_path: Path) -> None:
    results = tmp_path / "experiment_results.tsv"
    write_results(
        results,
        {"converted_rate": {"z_statistic": "", "p_value_raw": ""}},
    )

    assert validate_results_tsv(results, EXPECTED_STATS)


def test_results_reject_blank_inference_for_regular_metric(tmp_path: Path) -> None:
    results = tmp_path / "experiment_results.tsv"
    write_results(results, {"opened_rate": {"z_statistic": "", "p_value_raw": ""}})

    assert not validate_results_tsv(results, EXPECTED_STATS)


def test_component_scoring_preserves_other_correct_results(tmp_path: Path) -> None:
    results = tmp_path / "experiment_results.tsv"
    write_results(results, {"opened_rate": {"control_rate": "not-a-number"}})

    result_checks = validate_result_rows(results, EXPECTED_STATS)
    scores = component_scores(assignment_ok=True, result_checks=result_checks, report_ok=True)

    assert result_checks == {
        "opened_rate": False,
        "clicked_rate": True,
        "converted_rate": True,
        "unsubscribed_rate": True,
    }
    assert sum(scores.values()) == pytest.approx(0.85)
