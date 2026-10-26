from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from tasks.business_finance.digital_marketing_ab_test_analysis_1 import main as task
from tasks.business_finance.digital_marketing_ab_test_analysis_1.scripts import (
    verify_ab_test_outputs as verifier,
)


TASK_ROOT = Path(task.__file__).resolve().parent
EVENT_COUNTS = {
    "control": {"opened": 981, "clicked": 981, "converted": 0, "unsubscribed": 5},
    "treatment": {"opened": 1131, "clicked": 0, "converted": 0, "unsubscribed": 17},
}


@pytest.fixture
def frozen_counts_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "experiment_results_raw.csv"
    with raw.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "delivered", "opened", "clicked", "converted", "unsubscribed"])
        for arm, counts in EVENT_COUNTS.items():
            for index in range(4409):
                writer.writerow([arm, 1, *[int(index < count) for count in counts.values()]])
    return raw


def test_runtime_and_card_declare_the_same_statistical_contract() -> None:
    config = task.DigitalMarketingABTestConfig()
    runtime = config.task_description.replace(config.remote_output_dir, "base/output").replace(
        config.input_dir, "base/input"
    )
    card = json.loads((TASK_ROOT / "task_card.json").read_text())
    assert card["taskPrompt"] == runtime
    normalized = " ".join(runtime.split())
    for clause in (
        "**full** `experiment_results_raw.csv` population",
        "equal 1:1 allocation using Pearson chi-square with 1 degree of freedom",
        "`srm_pvalue > 0.01`",
        "pooled, two-sided two-proportion z test without continuity correction",
        "`significant_at_05` means p < 0.05",
        "unpooled Wald 95% confidence intervals",
        "Benjamini-Hochberg step-up correction to the three secondary metrics",
        "at FDR 0.05",
        "equal 1:1 allocation, two-sided alpha=0.05, 80% power",
        "arithmetic mean of historical `open_rate`",
        "minimum detectable effect is an absolute increase of 0.03",
        "Cohen's h normal-power analysis with the same inputs",
        "Recommend **ship** only when the primary metric is significant at 0.05",
        "treatment unsubscribe lift is at most 0.5 percentage points; otherwise recommend **hold**",
    ):
        assert clause in normalized
    assert "Phi_inverse" not in normalized


@pytest.mark.parametrize(
    "metric,z_statistic,ci_lower,ci_upper",
    [
        ("opened_rate", 3.742808221853023, 0.016219826788059534, 0.05182281326637458),
        ("clicked_rate", -33.22344226963035, -0.23477644986167276, -0.21022241609432632),
        ("converted_rate", 0.0, 0.0, 0.0),
        ("unsubscribed_rate", 2.5616060632378983, 0.0006400194767130743, 0.004803391727641654),
    ],
)
def test_existing_pooled_z_and_unpooled_ci_numbers(
    frozen_counts_raw: Path, metric: str, z_statistic: float, ci_lower: float, ci_upper: float
) -> None:
    stats = verifier.metric_stats(frozen_counts_raw)[metric]
    assert stats["z_statistic"] == pytest.approx(z_statistic, abs=1e-12)
    assert stats["ci_lower_95"] == pytest.approx(ci_lower, abs=1e-12)
    assert stats["ci_upper_95"] == pytest.approx(ci_upper, abs=1e-12)
    assert stats["p_value_raw"] == pytest.approx(
        math.erfc(abs(z_statistic) / math.sqrt(2)), abs=1e-12
    )


def test_fresh_clicked_wald_z_is_the_only_losing_cell(frozen_counts_raw: Path) -> None:
    stats = verifier.metric_stats(frozen_counts_raw)
    bh = verifier.bh_correct(
        {metric: stats[metric]["p_value_raw"] for metric in verifier.SECONDARY_METRICS}
    )
    row = {
        "is_primary": "False",
        "control_rate": "0.22249943297799954",
        "treatment_rate": "0.0",
        "absolute_lift": "-0.22249943297799954",
        "relative_lift_pct": "-100.0",
        "ci_lower_95": "-0.2347766754588844",
        "ci_upper_95": "-0.21022219049711469",
        "z_statistic": "-35.52091516607794",
        "p_value_raw": "2.337482553323282e-276",
        "significant_at_05": "True",
        "bh_rank": "1",
        "bh_threshold": "0.016666666666666666",
        "bh_significant": "True",
    }
    assert not verifier.validate_metric_row("clicked_rate", row, stats["clicked_rate"], bh)
    corrected = dict(row, z_statistic=str(stats["clicked_rate"]["z_statistic"]))
    assert verifier.validate_metric_row("clicked_rate", corrected, stats["clicked_rate"], bh)
    checks = {metric: metric != "clicked_rate" for metric in stats}
    assert verifier.component_scores(True, checks, True) == {
        "assignment": 0.2,
        "results_opened_rate": 0.15,
        "results_clicked_rate": 0.0,
        "results_converted_rate": 0.15,
        "results_unsubscribed_rate": 0.15,
        "report": 0.2,
    }
    assert sum(verifier.component_scores(True, checks, True).values()) == pytest.approx(0.85)
    for delta, accepted in [(0.009, True), (0.011, False)]:
        candidate = dict(corrected, z_statistic=str(stats["clicked_rate"]["z_statistic"] + delta))
        assert (
            verifier.validate_metric_row("clicked_rate", candidate, stats["clicked_rate"], bh)
            is accepted
        )


def test_two_sided_z_accepts_either_group_order(frozen_counts_raw: Path) -> None:
    stats = verifier.metric_stats(frozen_counts_raw)
    bh = verifier.bh_correct(
        {metric: stats[metric]["p_value_raw"] for metric in verifier.SECONDARY_METRICS}
    )
    for metric, expected in stats.items():
        row = {key: str(value) for key, value in expected.items()}
        if metric == verifier.PRIMARY_METRIC:
            row.update(bh_rank="", bh_threshold="", bh_significant="")
        else:
            row.update({key: str(value) for key, value in bh[metric].items()})
        row["z_statistic"] = str(-expected["z_statistic"])
        assert verifier.validate_metric_row(metric, row, expected, bh)


def test_correct_rounded_relative_percentage_is_accepted(frozen_counts_raw: Path) -> None:
    stats = verifier.metric_stats(frozen_counts_raw)
    expected = stats["opened_rate"]
    row = {key: str(value) for key, value in expected.items()}
    row.update(relative_lift_pct="15.3", bh_rank="", bh_threshold="", bh_significant="")
    assert verifier.validate_metric_row("opened_rate", row, expected, {})
    row["relative_lift_pct"] = "15.4"
    assert not verifier.validate_metric_row("opened_rate", row, expected, {})


@pytest.mark.parametrize(
    "metric,p_value,accepted",
    [
        ("opened_rate", "0.0002", True),
        ("opened_rate", "0.000200", True),
        ("opened_rate", "2.00e-4", True),
        ("opened_rate", "0", True),
        ("opened_rate", "0.000000", True),
        ("opened_rate", "0.001", False),
        ("opened_rate", "0.001000", False),
        ("opened_rate", "1e9999999", False),
        ("opened_rate", "NaN", False),
        ("unsubscribed_rate", "0.0104", True),
        ("unsubscribed_rate", "0.010400", True),
        ("unsubscribed_rate", "1.0400e-2", True),
        ("unsubscribed_rate", "0.0106", False),
        ("unsubscribed_rate", "0.010600", False),
    ],
)
def test_p_value_check_is_independent_of_trailing_zeros(
    frozen_counts_raw: Path, metric: str, p_value: str, accepted: bool
) -> None:
    stats = verifier.metric_stats(frozen_counts_raw)
    bh = verifier.bh_correct({key: stats[key]["p_value_raw"] for key in verifier.SECONDARY_METRICS})
    row = {key: str(value) for key, value in stats[metric].items()}
    if metric == verifier.PRIMARY_METRIC:
        row.update(bh_rank="", bh_threshold="", bh_significant="")
    else:
        row.update({key: str(value) for key, value in bh[metric].items()})
    row["p_value_raw"] = p_value
    assert verifier.validate_metric_row(metric, row, stats[metric], bh) is accepted


@pytest.mark.parametrize(
    "metric,changes",
    [
        (
            "unsubscribed_rate",
            {"control_rate": "0.0002", "treatment_rate": "0.0048", "absolute_lift": "0.0018"},
        ),
        ("converted_rate", {"ci_lower_95": "0.0009", "ci_upper_95": "-0.0009"}),
        ("clicked_rate", {"p_value_raw": "0.001"}),
    ],
)
def test_inconsistent_or_wrong_values_fail_their_metric(
    frozen_counts_raw: Path, metric: str, changes: dict[str, str]
) -> None:
    stats = verifier.metric_stats(frozen_counts_raw)
    bh = verifier.bh_correct({key: stats[key]["p_value_raw"] for key in verifier.SECONDARY_METRICS})
    row = {key: str(value) for key, value in stats[metric].items()}
    row.update({key: str(value) for key, value in bh[metric].items()})
    assert verifier.validate_metric_row(metric, row, stats[metric], bh)
    row.update(changes)
    assert not verifier.validate_metric_row(metric, row, stats[metric], bh)


def test_bh_equal_p_values_accept_either_valid_tie_order(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "delivered", "opened", "clicked", "converted", "unsubscribed"])
        for variant in ("control", "treatment"):
            for index in range(10):
                writer.writerow([variant, 1, int(index < 5), int(index < 5), 0, 0])
    stats = verifier.metric_stats(raw)
    ranks = {"clicked_rate": 1, "converted_rate": 3, "unsubscribed_rate": 2}
    output = tmp_path / "results.tsv"
    fields = [
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
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for metric, expected in stats.items():
            row = {"metric": metric, **expected}
            if metric == verifier.PRIMARY_METRIC:
                row.update(bh_rank="", bh_threshold="", bh_significant="")
            else:
                row.update(
                    bh_rank=ranks[metric],
                    bh_threshold=ranks[metric] / 3 * 0.05,
                    bh_significant=False,
                )
            writer.writerow(row)
    assert all(verifier.validate_result_rows(output, stats).values())
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    next(row for row in rows if row["metric"] == "converted_rate")["bh_rank"] = "2"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    assert not all(verifier.validate_result_rows(output, stats).values())


def test_bh_step_up_preserves_frozen_secondary_outcomes(frozen_counts_raw: Path) -> None:
    stats = verifier.metric_stats(frozen_counts_raw)
    ordered = sorted(verifier.SECONDARY_METRICS, key=lambda metric: stats[metric]["p_value_raw"])
    assert ordered == ["clicked_rate", "unsubscribed_rate", "converted_rate"]
    largest_rank = max(
        rank
        for rank, metric in enumerate(ordered, 1)
        if stats[metric]["p_value_raw"] <= rank / 3 * 0.05
    )
    assert largest_rank == 2
    bh = verifier.bh_correct({metric: stats[metric]["p_value_raw"] for metric in ordered})
    for rank, metric in enumerate(ordered, 1):
        assert bh[metric] == {
            "bh_rank": rank,
            "bh_threshold": rank / 3 * 0.05,
            "bh_significant": rank <= largest_rank,
        }


@pytest.mark.parametrize(
    "p_values,decisions",
    [
        ([0.02, 0.03, 0.9], [True, True, False]),
        ([0.02, 0.04, 0.049], [True, True, True]),
        ([0.02, 0.04, 0.9], [False, False, False]),
        ([0.01, 0.04, 0.9], [True, False, False]),
        ([0.9, 0.03, 0.02], [False, True, True]),
        ([0.03, 0.03, 0.9], [True, True, False]),
        ([0.05, 0.05, 0.05], [True, True, True]),
        ([0.051, 0.051, 0.051], [False, False, False]),
    ],
)
def test_bh_step_up_rejects_all_hypotheses_through_last_passing_rank(
    frozen_counts_raw: Path, p_values: list[float], decisions: list[bool]
) -> None:
    corrected = verifier.bh_correct(dict(zip(verifier.SECONDARY_METRICS, p_values)))
    assert [
        corrected[metric]["bh_significant"] for metric in verifier.SECONDARY_METRICS
    ] == decisions
    expected = verifier.metric_stats(frozen_counts_raw)["clicked_rate"]
    row = {key: str(value) for key, value in expected.items()}
    row.update({key: str(value) for key, value in corrected["clicked_rate"].items()})
    assert verifier.validate_metric_row("clicked_rate", row, expected, corrected)
    row["bh_significant"] = str(not decisions[0])
    assert not verifier.validate_metric_row("clicked_rate", row, expected, corrected)


@pytest.mark.parametrize(
    "sample_size,accepted", [(3119, False), (3120, True), (3121, True), (3122, True), (3133, False)]
)
def test_arithmetic_mean_power_and_existing_cohen_h_allowance(
    tmp_path: Path, sample_size: int, accepted: bool
) -> None:
    historical = tmp_path / "historical_metrics.csv"
    historical.write_text("open_rate,emails_delivered\n0.2,100\n0.23715,900\n")
    required = verifier.sample_size_required(historical)
    assert required == 3122
    assert verifier.POWER_SAMPLE_SIZE_LOWER_TOLERANCE == 2
    assert verifier.report_has_acceptable_sample_size(str(sample_size), required) is accepted
