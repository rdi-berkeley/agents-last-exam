from __future__ import annotations

import csv
import importlib.util
import math
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
        "The primary opened_rate changed from 22.2499% to 25.6521%. "
        "Its observed absolute lift is\n**+0.0340213200 (+3.4021 percentage points)**.",
        'Observed primary absolute lift:\n"3.4" pp.',
        "Observed primary absolute lift (percentage points): +3.4021",
        "Primary absolute lift: 0.034; relative lift: 15.29%",
        "Primary absolute lift: 0.034, relative lift: 15.29%",
        "| Metric | Absolute lift (pp) | Relative lift (%) |\n"
        "| --- | ---: | ---: |\n| opened_rate | +3.4021 | 15.29 |\n"
        "| clicked_rate | -22.2499 | -100 |",
        "Metric | Absolute lift (basis points)\n--- | ---:\nopened_rate | 340.21",
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
        "| Metric | Absolute lift (pp) | Relative lift (%) |\n"
        "| --- | --- | --- |\n| opened_rate | 1.5 | 3.4 |",
        "| Metric | Absolute lift |\n| --- | --- |\n| clicked_rate | 0.034 |",
        "Primary absolute lift: 0.034e1",
        "Primary absolute lift: 0.034wrong",
        "Primary absolute lift: 0.034_ignored",
    ],
)
def test_report_does_not_use_relative_lift_or_ci_as_absolute_lift(text: str) -> None:
    assert not VERIFIER.report_has_primary_lift(text, 150 / 4409)
