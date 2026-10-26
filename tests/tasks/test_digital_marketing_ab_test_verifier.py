from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
        ("## Recommendation\nWe cannot ship.", "hold"),
        ("## Recommendation\nWe cannot safely ship.", "hold"),
        ("## Recommendation\nDo not **ship** the rollout.", "hold"),
        ("## Recommendation\nDo not wait to ship.", "ship"),
        ("## Recommendation\nShip this rollout; do not hold it.", "ship"),
        ("## Recommendation\nWe recommend **ship**, not **hold**.", "ship"),
        ("## Recommendation\nShip or hold after another review.", None),
        ("## Recommendation\nShip.\n\nActually hold.\n\n## Results\nDone.", None),
        ("## Recommendation\nShip.\n\n## Recommendation\nHold.", None),
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
        "opened_rate": {"significant_at_05": True, "absolute_lift": 0.034},
        "unsubscribed_rate": {"absolute_lift": 0.002722},
    }

    assert validate_report_md(report, expected_stats, required_n=3122) is expected


def test_guardrail_boundary_is_inclusive(tmp_path: Path) -> None:
    report = tmp_path / "experiment_report.md"
    report.write_text(
        "Required per-arm sample size: 3,122\n\n"
        "Observed primary lift: 3.40 percentage points.\n\n"
        "## Recommendation\nSHIP the rollout.\n",
        encoding="utf-8",
    )
    expected_stats = {
        "opened_rate": {"significant_at_05": True, "absolute_lift": 0.034},
        "unsubscribed_rate": {"absolute_lift": 0.005},
    }

    assert validate_report_md(report, expected_stats, required_n=3122)


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


@pytest.fixture
def assignment_raw(tmp_path: Path) -> Path:
    path = tmp_path / "raw.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "delivered"])
        writer.writerows([["control", int(index < 3)] for index in range(4)])
        writer.writerows([["treatment", int(index < 4)] for index in range(6)])
    return path


def write_assignment(path: Path, **updates: str) -> None:
    values = {
        "n_control": "4",
        "n_treatment": "6",
        "ratio": "1.5",
        "srm_chi2": "0.4",
        "srm_pvalue": "0.5270892568655381",
        "srm_pass": "True",
    }
    values.update(updates)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(reversed(list(values.items())))


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"n_control": " 4.0 ", "n_treatment": "6e0", "ratio": "1.500000"},
        {"srm_chi2": "4e-1", "srm_pvalue": "0.527089", "srm_pass": " YES "},
        {"srm_pass": "1"},
    ],
)
def test_assignment_matches_full_raw_population_not_delivered_only(
    tmp_path: Path, assignment_raw: Path, updates: dict[str, str]
) -> None:
    output = tmp_path / "assignment.csv"
    write_assignment(output, **updates)
    assert VERIFIER.validate_assignment_csv(output, assignment_raw)


@pytest.mark.parametrize("column", ["n_control", "n_treatment", "ratio", "srm_chi2", "srm_pvalue"])
@pytest.mark.parametrize("value", ["", "NaN", "inf", "-Infinity", "not-a-number", "1e999"])
def test_assignment_rejects_nonfinite_or_malformed_numeric_cells(
    tmp_path: Path, assignment_raw: Path, column: str, value: str
) -> None:
    output = tmp_path / "assignment.csv"
    write_assignment(output, **{column: value})
    assert not VERIFIER.validate_assignment_csv(output, assignment_raw)


@pytest.mark.parametrize(
    "column,value",
    [
        ("n_control", "3"),
        ("n_control", "4.1"),
        ("n_control", "-4"),
        ("n_treatment", "4"),
        ("n_treatment", "999999"),
        ("ratio", "1"),
        ("ratio", "0.6666667"),
        ("srm_chi2", "0"),
        ("srm_chi2", "-0.0001"),
        ("srm_pvalue", "1"),
        ("srm_pvalue", "1.00001"),
        ("srm_pvalue", "-0.00001"),
        ("srm_pass", "False"),
        ("srm_pass", "not true"),
        ("srm_pass", "NaN"),
    ],
)
def test_assignment_rejects_false_quantities_and_flags(
    tmp_path: Path, assignment_raw: Path, column: str, value: str
) -> None:
    output = tmp_path / "assignment.csv"
    write_assignment(output, **{column: value})
    assert not VERIFIER.validate_assignment_csv(output, assignment_raw)


@pytest.mark.parametrize("flag", ["False", " no ", "0"])
def test_correctly_reported_srm_failure_is_valid_analysis(tmp_path: Path, flag: str) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("variant,delivered\n" + "control,1\n" * 20 + "treatment,1\n" * 80)
    output = tmp_path / "assignment.csv"
    write_assignment(
        output,
        n_control="20",
        n_treatment="80",
        ratio="4",
        srm_chi2="36",
        srm_pvalue="1.9731752900754036e-09",
        srm_pass=flag,
    )
    assert VERIFIER.validate_assignment_csv(output, raw)


@pytest.mark.parametrize("damage", ["duplicate", "missing", "short", "extra", "header", "quote"])
def test_assignment_rejects_malformed_structure(
    tmp_path: Path, assignment_raw: Path, damage: str
) -> None:
    output = tmp_path / "assignment.csv"
    write_assignment(output)
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if damage == "duplicate":
        rows.append(rows[-1])
    elif damage == "missing":
        rows.pop()
    elif damage == "short":
        rows[-1].pop()
    elif damage == "extra":
        rows[-1].append("unheaded")
    elif damage == "header":
        rows[0] = ["metric", "value", "value"]
    with output.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
        if damage == "quote":
            handle.write('"unterminated')
    assert not VERIFIER.validate_assignment_csv(output, assignment_raw)


@pytest.mark.parametrize("value", ["17", "17.0000", "1.7e1", " 17.0 "])
def test_relative_lift_accepts_numeric_equivalents(tmp_path: Path, value: str) -> None:
    output = tmp_path / "results.tsv"
    write_results(output, {"opened_rate": {"relative_lift_pct": value}})
    assert validate_results_tsv(output, EXPECTED_STATS)


@pytest.mark.parametrize("metric", list(EXPECTED_STATS))
@pytest.mark.parametrize("value", ["999999999", "-999999999", "inf", "-inf", "bad"])
def test_wrong_relative_lift_loses_only_its_metric(tmp_path: Path, metric: str, value: str) -> None:
    output = tmp_path / "results.tsv"
    write_results(output, {metric: {"relative_lift_pct": value}})
    checks = validate_result_rows(output, EXPECTED_STATS)
    assert checks == {name: name != metric for name in EXPECTED_STATS}
    assert sum(component_scores(True, checks, True).values()) == pytest.approx(0.85)


@pytest.mark.parametrize("value", ["", "NaN", "N/A", "undefined", "null"])
def test_undefined_relative_lift_only_allowed_with_zero_baseline(
    tmp_path: Path, value: str
) -> None:
    output = tmp_path / "results.tsv"
    write_results(output, {"converted_rate": {"relative_lift_pct": value}})
    assert validate_results_tsv(output, EXPECTED_STATS)
    write_results(output, {"opened_rate": {"relative_lift_pct": value}})
    assert not validate_results_tsv(output, EXPECTED_STATS)


@pytest.mark.parametrize(
    "value,valid",
    [
        ("", True),
        ("NaN", True),
        ("inf", True),
        ("+Infinity", True),
        ("0", False),
        ("999999999", False),
        ("-inf", False),
    ],
)
def test_zero_to_positive_rate_does_not_have_finite_relative_lift(
    tmp_path: Path, value: str, valid: bool
) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "variant,delivered,opened,clicked,converted,unsubscribed\n"
        "control,1,0,0,0,0\ncontrol,1,0,0,0,0\n"
        "treatment,1,1,0,0,0\ntreatment,1,0,0,0,0\n"
    )
    stats = VERIFIER.metric_stats(raw)
    assert stats["opened_rate"]["relative_lift_pct"] == math.inf
    row = {key: str(item) for key, item in stats["opened_rate"].items()}
    row.update(relative_lift_pct=value, bh_rank="", bh_threshold="", bh_significant="")
    assert VERIFIER.validate_metric_row("opened_rate", row, stats["opened_rate"], {}) is valid


def test_invalid_zero_baseline_relative_lift_is_candidate_failure(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "variant,delivered,opened,clicked,converted,unsubscribed\n"
        "control,1,0,0,0,0\ncontrol,1,0,0,0,0\n"
        "treatment,1,1,0,0,0\ntreatment,1,0,0,0,0\n"
    )
    stats = VERIFIER.metric_stats(raw)
    row = {key: str(item) for key, item in stats["opened_rate"].items()}
    row.update(relative_lift_pct="not-a-number", bh_rank="", bh_threshold="", bh_significant="")

    assert not VERIFIER.validate_metric_row("opened_rate", row, stats["opened_rate"], {})


def test_relative_lift_uses_delivered_denominators(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "variant,delivered,opened,clicked,converted,unsubscribed\n"
        "control,1,1,0,0,0\ncontrol,1,0,0,0,0\ncontrol,0,0,0,0,0\n"
        "treatment,1,1,0,0,0\ntreatment,1,1,0,0,0\n"
    )
    assert VERIFIER.metric_stats(raw)["opened_rate"]["relative_lift_pct"] == 100.0


def test_missing_relative_lift_column_is_not_an_undefined_value(tmp_path: Path) -> None:
    output = tmp_path / "results.tsv"
    write_results(output)
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    columns = [column for column in RESULT_FIELDS if column != "relative_lift_pct"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    assert not any(validate_result_rows(output, EXPECTED_STATS).values())


@pytest.mark.parametrize("value,valid", [("0", True), ("0.0", True), ("", False), ("NaN", False)])
def test_equal_nonzero_rates_have_defined_zero_relative_lift(
    tmp_path: Path, value: str, valid: bool
) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(
        "variant,delivered,opened,clicked,converted,unsubscribed\n"
        "control,1,1,0,0,0\ntreatment,1,1,0,0,0\n"
    )
    stats = VERIFIER.metric_stats(raw)
    row = {key: str(item) for key, item in stats["opened_rate"].items()}
    row.update(relative_lift_pct=value, bh_rank="", bh_threshold="", bh_significant="")
    assert VERIFIER.validate_metric_row("opened_rate", row, stats["opened_rate"], {}) is valid


@pytest.mark.parametrize("sample_size", ["3120", "3,120", "3121", "3122", "3,122"])
def test_existing_power_analysis_conventions_remain_accepted(
    tmp_path: Path, sample_size: str
) -> None:
    report = tmp_path / "report.md"
    report.write_text(
        f"Required sample size per arm: {sample_size}\n"
        "Observed lift: 3.40 percentage points.\n\n## Recommendation\nShip.\n"
    )
    assert validate_report_md(report, EXPECTED_STATS, required_n=3122)


@pytest.mark.parametrize(
    ("sample_size", "valid"),
    [
        (3118, False),
        (3119, True),
        (3120, True),
        (3121, True),
        (3122, True),
        (3123, True),
        (3124, False),
    ],
)
def test_sample_size_accepts_normal_approximation_variants(
    tmp_path: Path, sample_size: int, valid: bool
) -> None:
    historical = tmp_path / "historical_metrics.csv"
    historical.write_text("open_rate\n0.218575\n")
    required_n = VERIFIER.sample_size_required(historical)
    bounds = VERIFIER.sample_size_bounds(historical, required_n)
    assert required_n == 3122
    assert bounds == (3119, 3123)
    assert (
        VERIFIER.report_has_required_sample_size(
            f"Required sample size per arm: {sample_size}", required_n, bounds
        )
        is valid
    )


@pytest.mark.parametrize(
    "lift",
    [
        "0.034",
        ".034",
        "+0.0340",
        "3.4e-2",
        "3.40 percentage points",
        "3.4 pp",
        "3.4pp",
        "3.40 p.p.",
        "340 basis points",
        "340 bps",
        "**0.034**",
        "`3.4e-2`",
        '"0.034"',
        "3.402 percentage points",
    ],
)
def test_report_accepts_equivalent_lift_units(tmp_path: Path, lift: str) -> None:
    report = tmp_path / "report.md"
    report.write_text(
        f"Required sample size per arm: 3,122\nObserved primary absolute lift: {lift}.\n"
        "\n## Recommendation\nShip.\n"
    )
    assert validate_report_md(report, EXPECTED_STATS, required_n=3122)


@pytest.mark.parametrize(
    "lift",
    [
        "13.4 pp",
        "-3.40pp",
        "0.034%",
        "3.40",
        "34 bps",
        "1e999",
        "NaN",
        "0.0034",
        "0.0349",
        "3.49 pp",
        "reference_0.034",
        "-0.034",
    ],
)
def test_report_rejects_wrong_lift_values_units_and_substrings(lift: str) -> None:
    assert not VERIFIER.report_has_primary_lift(f"Observed primary lift: {lift}", 0.034)


@pytest.mark.parametrize(
    "text",
    [
        "Version 3.4.0",
        "Required sample size: 3122; deadline 0.034.",
        "Secondary clicked_rate lift: 0.034",
        "Unsubscribe lift: 3.4pp",
        "Observed primary relative lift: 3.4%",
    ],
)
def test_report_other_numbers_do_not_establish_primary_absolute_lift(text: str) -> None:
    assert not VERIFIER.report_has_primary_lift(text, 0.034)


@pytest.mark.parametrize("expected", [0.017, -0.043, 0.0, 0.125])
def test_report_lift_is_derived_from_current_visible_data(expected: float) -> None:
    assert VERIFIER.report_has_primary_lift(f"Observed primary absolute lift: {expected}", expected)
    assert VERIFIER.report_has_primary_lift(
        f"Observed primary absolute lift: {expected * 100} percentage points", expected
    )
    assert not VERIFIER.report_has_primary_lift("Observed primary lift: 3.4 pp", expected)


@pytest.mark.parametrize("text", ["31220", "13122", "3,122.5", "-3122", "id3122", "3122e1"])
def test_sample_size_is_not_a_matching_substring(text: str) -> None:
    assert not VERIFIER.report_has_acceptable_sample_size(text, 3122)


@pytest.mark.parametrize("text", ["3,122", '"3122"', "`3122`", "3.122e3", "3122.0"])
def test_sample_size_accepts_integral_numeric_equivalence(text: str) -> None:
    assert VERIFIER.report_has_acceptable_sample_size(text, 3122)


@pytest.mark.parametrize(
    "text",
    [
        (
            "The primary opened_rate changed from 22.2499% to 25.6521%. "
            "Its observed absolute lift is\n**+0.0340213200 (+3.4021 percentage points)**."
        ),
        'Observed primary absolute lift:\n"3.4" pp.',
        "Observed primary absolute lift (percentage points): +3.4021",
        "Primary absolute lift: 0.034; relative lift: 15.29%",
        "Primary absolute lift: 0.034, relative lift: 15.29%",
        (
            "| Metric | Absolute lift (pp) | Relative lift (%) |\n"
            "| --- | ---: | ---: |\n| opened_rate | +3.4021 | 15.29 |\n"
            "| clicked_rate | -22.2499 | -100 |"
        ),
        "Metric | Absolute lift (basis points)\n--- | ---:\nopened_rate | 340.21",
        (
            "| Metric | Absolute lift | 95% CI for lift |\n"
            "| --- | --- | --- |\n| opened_rate | 0.03402 | [0.01622, 0.05182] |"
        ),
        "| Quantity | Value |\n| --- | --- |\n| Primary absolute lift | 0.034 |",
        "Primary absolute lift: `3.4021e-2` (3.4021 percentage points)",
        "Primary absolute lift: +3.4021pp",
    ],
)
def test_report_lift_accepts_layout_and_unit_equivalents(text: str) -> None:
    assert VERIFIER.report_has_primary_lift(text, 150 / 4409)


@pytest.mark.parametrize(
    "text",
    [
        "Primary absolute lift: 0.015, relative lift: 3.4%",
        "Primary absolute lift: 0.015; 95% CI: [0.0, 0.034]",
        "Primary absolute lift: 0.015 (95% CI: [0.0, 0.034])",
        (
            "| Metric | Absolute lift (pp) | Relative lift (%) |\n"
            "| --- | --- | --- |\n| opened_rate | 1.5 | 3.4 |"
        ),
        "| Metric | Absolute lift |\n| --- | --- |\n| clicked_rate | 0.034 |",
        "| Metric | 95% CI for lift |\n| --- | --- |\n| opened_rate | [0.034, 0.05] |",
        "Primary absolute lift: 0.034e1",
        "Primary absolute lift: 0.034wrong",
        "Primary absolute lift: 0.034_ignored",
    ],
)
def test_report_does_not_use_relative_lift_or_ci_as_absolute_lift(text: str) -> None:
    assert not VERIFIER.report_has_primary_lift(text, 150 / 4409)


@pytest.mark.parametrize(
    "text",
    [
        (
            "The treatment increased the primary open rate by 3.402 percentage points, "
            "from 22.250% to 25.652%."
        ),
        "The open rate increased by 3.402 percentage points.",
        "The primary metric rose by 340.2 basis points.",
        "The treatment open rate was 3.402 percentage points higher than control.",
    ],
)
def test_report_accepts_primary_rate_increase_without_lift_label(text: str) -> None:
    assert VERIFIER.report_has_primary_lift(text, 150 / 4409)


@pytest.mark.parametrize(
    "text",
    [
        "Revenue lift: 3.402 percentage points.",
        "The primary open rate rose by 1.5 percentage points and revenue lift was 3.402 pp.",
        "The open rate decreased by 3.402 percentage points.",
        "The unsubscribe rate increased by 3.402 percentage points.",
        "Primary absolute lift was not 3.402 percentage points.",
        "Primary absolute lift wasn't 3.402 percentage points.",
        "The open rate increased by 3.402% relative to control.",
    ],
)
def test_report_rejects_other_metrics_or_wrong_lift_direction(text: str) -> None:
    assert not VERIFIER.report_has_primary_lift(text, 150 / 4409)


def test_report_rejects_conflicting_current_primary_lifts() -> None:
    report = (
        "Observed primary absolute lift: 3.402 percentage points. "
        "Actually, the primary absolute lift is -8 percentage points."
    )
    assert not VERIFIER.report_has_primary_lift(report, 150 / 4409)


def test_report_does_not_confuse_historical_lift_with_current_result() -> None:
    report = (
        "The historical primary absolute lift was -8 percentage points. "
        "The observed primary absolute lift is 3.402 percentage points."
    )
    assert VERIFIER.report_has_primary_lift(report, 150 / 4409)


def test_report_does_not_credit_historical_lift_as_current_result() -> None:
    assert not VERIFIER.report_has_primary_lift(
        "The historical primary absolute lift was 3.402 percentage points.", 150 / 4409
    )


def test_historical_context_does_not_hide_observed_lift() -> None:
    assert VERIFIER.report_has_primary_lift(
        "Using a historical benchmark, the observed primary absolute lift is 3.402 pp.",
        150 / 4409,
    )


@pytest.mark.parametrize(
    "text",
    [
        "Required per-arm sample size: 3,122",
        "| Required sample size per arm | 3,122 |",
        "Power analysis requires 3,122 delivered emails per arm.",
        "Power analysis gives n = 3,122 per arm.",
        "## Power analysis\n\nThe normal-approximation requirement is 3,122 delivered emails per arm (6,244 total).",
        "Required per arm (power analysis): 3,122",
        "Required sample size per arm:\n3,122",
        (
            "## Power analysis\n\nThe normal approximation gives 3121.2398, which rounds up to "
            "**3,122 participants per\narm** (6,244 total)."
        ),
    ],
)
def test_report_requires_sample_size_in_context(text: str) -> None:
    assert VERIFIER.report_has_required_sample_size(text, 3122)


@pytest.mark.parametrize(
    "text",
    [
        "There are 3,122 subscribers.",
        "Observed sample size: 3,122.",
        "Total sample size: 3,122.",
        "Required sample size: 4,000; 3,122 subscribers responded.",
        "Required sample size: 4,000 (3,122 observed).",
        "Required sample size: 3,122 total (1,561 per arm).",
        "Required sample size is not 3,122 per arm.",
        "## Results\n\nThe requirement is 3,122 per arm.",
    ],
)
def test_report_rejects_unrelated_or_observed_sample_size(text: str) -> None:
    assert not VERIFIER.report_has_required_sample_size(text, 3122)


def test_report_rejects_conflicting_required_sample_sizes() -> None:
    report = (
        "Required sample size per arm: 3,122. Actually, the required sample size per arm is 4,000."
    )
    assert not VERIFIER.report_has_required_sample_size(report, 3122)


def test_report_accepts_historical_sample_size_with_correct_current_requirement() -> None:
    report = (
        "The previous required sample size per arm was 4,000. "
        "The current required sample size per arm is 3,122."
    )
    assert VERIFIER.report_has_required_sample_size(report, 3122)


def test_historical_data_and_formula_do_not_hide_required_sample_size() -> None:
    report = (
        "Using historical data, n = [z(1-alpha/2)]^2 / 0.03^2.\n"
        "- Required per-arm sample size: 3,122\n"
        "- Required total (both arms, 1:1): 6,244"
    )
    assert VERIFIER.report_has_required_sample_size(report, 3122)


def test_report_does_not_credit_unrelated_counts_or_lifts(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text(
        "There are 3,122 subscribers. Revenue lift: 3.402 percentage points.\n\n"
        "## Recommendation\nShip.\n",
        encoding="utf-8",
    )
    assert not validate_report_md(report, EXPECTED_STATS, 3122)


def test_results_accept_utf8_bom(tmp_path: Path) -> None:
    results = tmp_path / "experiment_results.tsv"
    write_results(results)
    results.write_bytes(b"\xef\xbb\xbf" + results.read_bytes())
    assert validate_results_tsv(results, EXPECTED_STATS)


def test_invalid_result_encoding_is_an_incorrect_submission(tmp_path: Path) -> None:
    results = tmp_path / "experiment_results.tsv"
    results.write_bytes(b"\xff\xfeinvalid")
    assert not any(validate_result_rows(results, EXPECTED_STATS).values())


def test_invalid_assignment_encoding_is_an_incorrect_submission(
    tmp_path: Path, assignment_raw: Path
) -> None:
    output = tmp_path / "assignment.csv"
    output.write_bytes(b"\xff\xfeinvalid")
    assert not VERIFIER.validate_assignment_csv(output, assignment_raw)


@pytest.mark.parametrize(
    "metric,field,value",
    [
        ("clicked_rate", "is_primary", "nonsense"),
        ("converted_rate", "significant_at_05", ""),
        ("converted_rate", "bh_significant", "nonsense"),
        ("converted_rate", "control_rate", "-0.0001"),
        ("opened_rate", "p_value_raw", "-0.0001"),
        ("converted_rate", "bh_rank", "3.0001"),
    ],
)
def test_invalid_metric_fields_lose_only_that_metric(
    tmp_path: Path, metric: str, field: str, value: str
) -> None:
    results = tmp_path / "experiment_results.tsv"
    write_results(results, {metric: {field: value}})
    checks = validate_result_rows(results, EXPECTED_STATS)
    assert checks == {name: name != metric for name in EXPECTED_STATS}


def test_duplicate_results_header_is_rejected(tmp_path: Path) -> None:
    results = tmp_path / "experiment_results.tsv"
    write_results(results)
    contents = results.read_text()
    results.write_text(contents.replace("metric\t", "metric\tmetric\t", 1))
    assert not any(validate_result_rows(results, EXPECTED_STATS).values())


def test_extra_unheaded_result_cell_loses_only_that_metric(tmp_path: Path) -> None:
    results = tmp_path / "experiment_results.tsv"
    write_results(results)
    lines = results.read_text().splitlines()
    lines[1] += "\textra"
    results.write_text("\n".join(lines) + "\n")
    checks = validate_result_rows(results, EXPECTED_STATS)
    assert checks == {name: name != "opened_rate" for name in EXPECTED_STATS}


class LocalVerifierSession:
    def __init__(self) -> None:
        self.interface = self

    async def create_dir(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    async def write_file(self, path: str, data: str) -> None:
        Path(path).write_text(data, encoding="utf-8")

    async def read_file(self, path: str) -> str:
        return Path(path).read_text(encoding="utf-8")

    async def run_command(self, command: str, *, check: bool = False) -> dict[str, object]:
        result = await asyncio.to_thread(
            subprocess.run, command, shell=True, capture_output=True, text=True, check=check
        )
        return {"return_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


@pytest.mark.parametrize(
    ("rule_report", "judge_report", "expected"),
    [
        (0.0, True, 1.0),
        (0.2, False, 0.8),
        (0.2, True, 1.0),
        (0.0, False, 0.8),
    ],
)
def test_report_judge_controls_only_report_component(
    monkeypatch: pytest.MonkeyPatch, rule_report: float, judge_report: bool, expected: float
) -> None:
    from tasks.business_finance.digital_marketing_ab_test_analysis_1 import main as task

    payload = {
        "score": 0.8 + rule_report,
        "component_scores": {"report": rule_report},
        "missing_files": [],
        "sample_bounds": [3119, 3123],
        "primary_absolute_lift": 0.034,
        "expected_recommendation": "ship",
        "report_size": 6,
    }
    session = SimpleNamespace(
        interface=SimpleNamespace(create_dir=AsyncMock()),
        write_file=AsyncMock(),
        read_file=AsyncMock(return_value="report"),
        run_command=AsyncMock(return_value={"return_code": 0, "stdout": json.dumps(payload)}),
    )
    judge = AsyncMock(return_value={"passed": judge_report})
    monkeypatch.setattr(task, "judge_report", judge)
    config = SimpleNamespace(
        metadata={"input_dir": "/input", "reference_dir": "/reference", "remote_output_dir": "/out"}
    )
    assert asyncio.run(task.evaluate(config, session)) == [expected]
    judge.assert_awaited_once_with(
        "report", sample_bounds=(3119, 3123), expected_lift=0.034, expected_recommendation="ship"
    )


def test_report_judge_failure_is_not_a_score(monkeypatch: pytest.MonkeyPatch) -> None:
    from tasks.business_finance.digital_marketing_ab_test_analysis_1 import main as task
    from tasks.utils.evaluation import JudgeInfrastructureError

    payload = {
        "score": 1.0,
        "component_scores": {"report": 0.2},
        "missing_files": [],
        "sample_bounds": [3119, 3123],
        "primary_absolute_lift": 0.034,
        "expected_recommendation": "ship",
        "report_size": 6,
    }
    session = SimpleNamespace(
        interface=SimpleNamespace(create_dir=AsyncMock()),
        write_file=AsyncMock(),
        read_file=AsyncMock(return_value="report"),
        run_command=AsyncMock(return_value={"return_code": 0, "stdout": json.dumps(payload)}),
    )
    monkeypatch.setattr(
        task, "judge_report", AsyncMock(side_effect=JudgeInfrastructureError("judge failed"))
    )
    config = SimpleNamespace(
        metadata={"input_dir": "/input", "reference_dir": "/reference", "remote_output_dir": "/out"}
    )
    with pytest.raises(JudgeInfrastructureError):
        asyncio.run(task.evaluate(config, session))


def test_oversized_report_is_not_read_or_graded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tasks.business_finance.digital_marketing_ab_test_analysis_1 import main as task
    from tasks.utils.evaluation import JudgeInfrastructureError

    input_dir, output_dir, reference_dir = (
        tmp_path / name for name in ("input", "output", "reference")
    )
    for directory in (input_dir, output_dir, reference_dir):
        directory.mkdir()
    (input_dir / "experiment_results_raw.csv").write_text(
        "variant,delivered,opened,clicked,converted,unsubscribed\n"
        "control,1,0,0,0,0\ntreatment,1,1,0,0,0\n"
    )
    (input_dir / "historical_metrics.csv").write_text("open_rate\n0.2\n")
    for name in ("randomization_assignment.csv", "experiment_results.tsv", "experiment_report.md"):
        (reference_dir / name).write_text("reference")
        (output_dir / name).write_text("reference")
    (output_dir / "experiment_report.md").write_text("x" * (task.MAX_REPORT_CHARS * 4 + 1))
    monkeypatch.setattr(task, "EVAL_TMP_DIR", str(tmp_path / "eval"))
    judge = AsyncMock()
    monkeypatch.setattr(task, "judge_report", judge)
    session = LocalVerifierSession()
    session.read_file = AsyncMock(side_effect=AssertionError("oversized report was read"))
    config = SimpleNamespace(
        metadata={
            "input_dir": str(input_dir),
            "reference_dir": str(reference_dir),
            "remote_output_dir": str(output_dir),
        }
    )
    with pytest.raises(JudgeInfrastructureError, match="input capacity"):
        asyncio.run(task.evaluate(config, session))
    session.read_file.assert_not_awaited()
    judge.assert_not_called()


def test_external_report_mode_skips_regex_grading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_dir, output_dir, reference_dir = (
        tmp_path / name for name in ("input", "output", "reference")
    )
    for directory in (input_dir, output_dir, reference_dir):
        directory.mkdir()
    (input_dir / "experiment_results_raw.csv").write_text(
        "variant,delivered,opened,clicked,converted,unsubscribed\n"
        "control,1,0,0,0,0\ntreatment,1,1,0,0,0\n"
    )
    (input_dir / "historical_metrics.csv").write_text("open_rate\n0.2\n")
    for name in ("randomization_assignment.csv", "experiment_results.tsv", "experiment_report.md"):
        (output_dir / name).write_text("invalid")
        (reference_dir / name).write_text("reference")
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--reference-dir",
            str(reference_dir),
            "--external-report",
        ],
    )
    monkeypatch.setattr(VERIFIER, "validate_report_md", lambda *args: pytest.fail("used regex"))
    assert VERIFIER.main() == 0
    assert json.loads(capsys.readouterr().out)["component_scores"]["report"] == 0.0


@pytest.mark.parametrize(
    "result",
    [
        {"return_code": 137, "stdout": '{"score": 0.0}', "stderr": "process killed"},
        {"return_code": 0, "stdout": "", "stderr": '{"score": 0.0}'},
        {"return_code": 0, "stdout": "not JSON", "stderr": ""},
        {"return_code": 0, "stdout": "{}", "stderr": ""},
        {"return_code": 0, "stdout": '{"score": "NaN"}', "stderr": ""},
        {"return_code": 0, "stdout": '{"score": 1.1}', "stderr": ""},
    ],
)
def test_verifier_execution_errors_do_not_produce_zero_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: dict[str, object]
) -> None:
    from tasks.business_finance.digital_marketing_ab_test_analysis_1 import main as task

    class FailedSession(LocalVerifierSession):
        async def run_command(self, command: str, *, check: bool = False) -> dict[str, object]:
            return result

    monkeypatch.setattr(task, "EVAL_TMP_DIR", str(tmp_path / "eval"))
    config = SimpleNamespace(
        metadata={"input_dir": "/input", "reference_dir": "/reference", "remote_output_dir": "/out"}
    )
    with pytest.raises(RuntimeError):
        asyncio.run(task.evaluate(config, FailedSession()))


@pytest.mark.parametrize("missing_reference", [False, True])
def test_verifier_reference_failure_propagates_as_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_reference: bool
) -> None:
    from tasks.business_finance.digital_marketing_ab_test_analysis_1 import main as task

    async def reject_report(*args, **kwargs):
        return {"passed": False}

    monkeypatch.setattr(task, "judge_report", reject_report)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    reference_dir = tmp_path / "reference"
    for directory in (input_dir, output_dir, reference_dir):
        directory.mkdir()
    (input_dir / "experiment_results_raw.csv").write_text(
        "variant,delivered,opened,clicked,converted,unsubscribed\n"
        "control,1,0,0,0,0\ntreatment,1,1,0,0,0\n"
    )
    (input_dir / "historical_metrics.csv").write_text("open_rate\n0.2\n")
    for name in ("randomization_assignment.csv", "experiment_results.tsv", "experiment_report.md"):
        (output_dir / name).write_text("invalid")
        if not missing_reference:
            (reference_dir / name).write_text("reference")
    monkeypatch.setattr(task, "EVAL_TMP_DIR", str(tmp_path / "eval"))
    config = SimpleNamespace(
        metadata={
            "input_dir": str(input_dir),
            "reference_dir": str(reference_dir),
            "remote_output_dir": str(output_dir),
        }
    )
    if missing_reference:
        with pytest.raises(RuntimeError, match="verifier failed"):
            asyncio.run(task.evaluate(config, LocalVerifierSession()))
    else:
        assert asyncio.run(task.evaluate(config, LocalVerifierSession())) == [0.0]


@pytest.mark.parametrize(
    "directory_name",
    ["randomization_assignment.csv", "experiment_results.tsv", "experiment_report.md"],
)
def test_output_directory_is_a_missing_submission_not_verifier_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory_name: str
) -> None:
    from tasks.business_finance.digital_marketing_ab_test_analysis_1 import main as task

    async def reject_report(*args, **kwargs):
        return {"passed": False}

    monkeypatch.setattr(task, "judge_report", reject_report)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    reference_dir = tmp_path / "reference"
    for directory in (input_dir, output_dir, reference_dir):
        directory.mkdir()
    (input_dir / "experiment_results_raw.csv").write_text(
        "variant,delivered,opened,clicked,converted,unsubscribed\n"
        "control,1,0,0,0,0\ntreatment,1,1,0,0,0\n"
    )
    (input_dir / "historical_metrics.csv").write_text("open_rate\n0.2\n")
    for name in ("randomization_assignment.csv", "experiment_results.tsv", "experiment_report.md"):
        if name == directory_name:
            (output_dir / name).mkdir()
        else:
            (output_dir / name).write_text("invalid")
        (reference_dir / name).write_text("reference")
    monkeypatch.setattr(task, "EVAL_TMP_DIR", str(tmp_path / "eval"))
    config = SimpleNamespace(
        metadata={
            "input_dir": str(input_dir),
            "reference_dir": str(reference_dir),
            "remote_output_dir": str(output_dir),
        }
    )
    assert asyncio.run(task.evaluate(config, LocalVerifierSession())) == [0.0]
