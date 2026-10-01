import csv
import hashlib
import io
import json
import os
import random
from pathlib import Path

import pytest

from tasks.health_medicine.obermeyer_bias_reproduction.scripts import score_outputs


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "task-data-hf/extracted/health_medicine/obermeyer_bias_reproduction/base"


def csv_text(rows, *, quoting=csv.QUOTE_MINIMAL, newline="\n", bom=False):
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(rows[0]), quoting=quoting, lineterminator=newline
    )
    writer.writeheader()
    writer.writerows(rows)
    return ("\ufeff" if bom else "") + buffer.getvalue()


@pytest.fixture
def bundle():
    patients = [
        {
            "patient_id": str(index),
            "risk_score_t": 100 - index,
            "gagne_sum_t": 10 if index in (0, 5, 6, 7) else 1,
            "race": "black" if index in (0, 5) else "white",
        }
        for index in range(100)
    ]
    predictions = [
        {
            "patient_id": row["patient_id"],
            "baseline_score": row["risk_score_t"],
            "revised_score": 100 if index in (0, 5, 6, 7) else 1 - index / 1000,
        }
        for index, row in enumerate(patients)
    ]
    return {
        "predictions_csv": csv_text(predictions),
        "analysis_data_csv": csv_text(patients),
        "reference_predictions_csv": csv_text(predictions),
        "reference_metrics_json": json.dumps(
            {
                "n_rows": 100,
                "top_n": 3,
                "baseline_black_fraction": 1 / 3,
                "revised_black_fraction": 2 / 3,
                "baseline_help_rate": 1 / 3,
                "revised_help_rate": 1,
                "baseline_max_abs_diff": 0,
            }
        ),
        "baseline_report_md": "black white risk gagne cost similar decile 1 2 3 4 5 6",
        "revised_report_md": "counterfactual revised high-risk threshold black white health 1 2 3 4 5",
    }


@pytest.mark.parametrize("order", ["original", "reverse", "shuffled"])
@pytest.mark.parametrize("quoting", [csv.QUOTE_MINIMAL, csv.QUOTE_ALL])
@pytest.mark.parametrize("bom,newline", [(False, "\n"), (True, "\r\n")])
def test_patient_keyed_output_is_order_invariant(bundle, order, quoting, bom, newline):
    expected = score_outputs.score_output_bundle(**bundle)
    assert expected.score == 1
    rows = list(csv.DictReader(io.StringIO(bundle["predictions_csv"])))
    if order == "reverse":
        rows.reverse()
    elif order == "shuffled":
        random.Random(42).shuffle(rows)
    for row in rows:
        row["baseline_score"] = f"{float(row['baseline_score']):.10e}"
        row["revised_score"] = f"{float(row['revised_score']):.10e}"
    bundle["predictions_csv"] = csv_text(rows, quoting=quoting, bom=bom, newline=newline)
    actual = score_outputs.score_output_bundle(**bundle)
    assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize("value", ["NaN", "inf", "-inf", "", "bad"])
@pytest.mark.parametrize("field", ["baseline_score", "revised_score"])
def test_invalid_scores_still_rejected(bundle, value, field):
    rows = list(csv.DictReader(io.StringIO(bundle["predictions_csv"])))
    rows[0][field] = value
    bundle["predictions_csv"] = csv_text(rows)
    result = score_outputs.score_output_bundle(**bundle)
    assert result.score == 0
    assert result.reason.startswith("numeric_validation_error")


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("duplicate", "duplicate_patient_id_in_output"),
        ("missing", "row_count_mismatch"),
        ("unknown", "unexpected_patient_id_in_output"),
        ("baseline", "baseline_score_must_equal_observed_risk_score_t"),
        ("help", "revised_help_rate_below_baseline"),
    ],
)
def test_scientific_and_identity_failures_are_preserved(bundle, mutation, reason):
    rows = list(csv.DictReader(io.StringIO(bundle["predictions_csv"])))
    if mutation == "duplicate":
        rows[1]["patient_id"] = rows[0]["patient_id"]
    elif mutation == "missing":
        rows.pop()
    elif mutation == "unknown":
        rows[0]["patient_id"] = "not-in-input"
    elif mutation == "baseline":
        rows[0]["baseline_score"] = "1"
    elif mutation == "help":
        for index, row in enumerate(rows):
            row["revised_score"] = 100 if index in (1, 2, 3) else index / 1000
    bundle["predictions_csv"] = csv_text(rows)
    result = score_outputs.score_output_bundle(**bundle)
    assert result.score == 0
    assert result.reason == reason


@pytest.mark.skipif(not BASE.is_dir(), reason="Local audited reference data unavailable")
@pytest.mark.parametrize("order", ["original", "reverse", "shuffled"])
def test_real_reference_retains_full_score_after_row_permutation(order):
    reference = BASE / "reference"
    text = (reference / "full_predictions.csv").read_text()
    rows = list(csv.DictReader(io.StringIO(text)))
    if order == "reverse":
        rows.reverse()
    elif order == "shuffled":
        random.Random(42).shuffle(rows)
    result = score_outputs.score_output_bundle(
        predictions_csv=csv_text(rows),
        analysis_data_csv=(BASE / "input/analysis_data.csv").read_text(),
        reference_metrics_json=(reference / "reference_metrics.json").read_text(),
        reference_predictions_csv=text,
        baseline_report_md=(reference / "baseline_analysis_report.md").read_text(),
        revised_report_md=(reference / "revised_analysis_report.md").read_text(),
    )
    assert result.score == 1, result
    assert result.details["baseline_help_rate"] == 0.36295283663704714


def test_publicly_valid_ranking_does_not_require_hidden_reference_overlap(bundle):
    rows = list(csv.DictReader(io.StringIO(bundle["reference_predictions_csv"])))
    for index, row in enumerate(rows):
        row["revised_score"] = str(-index)
    bundle["reference_predictions_csv"] = csv_text(rows)

    result = score_outputs.score_output_bundle(**bundle)

    assert result.score == 1
    assert "reference_top_overlap" not in result.details


@pytest.mark.parametrize(
    "wording",
    [
        "high risk",
        "highest-score",
        "high-need",
        "Larger scores mean more predicted need.",
        "Select the highest 1,464 scores, breaking ties by numeric ID ascending.",
    ],
)
def test_cohort_wording_does_not_change_score(bundle, wording):
    expected = score_outputs.score_output_bundle(**bundle)
    bundle["revised_report_md"] = (
        bundle["revised_report_md"].replace("high-risk", wording).replace("threshold", "")
    )

    actual = score_outputs.score_output_bundle(**bundle)

    assert actual.score == 1
    assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize("grouping", ["quantile", "percentile", "top 3%", "top group"])
def test_equivalent_grouping_language_is_not_penalized(bundle, grouping):
    expected = score_outputs.score_output_bundle(**bundle)
    bundle["baseline_report_md"] = bundle["baseline_report_md"].replace("decile", grouping)
    actual = score_outputs.score_output_bundle(**bundle)
    assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize("grouping", ["ventile", "stratification", "stratified", "bin"])
def test_additional_grouping_language_is_not_penalized(bundle, grouping):
    expected = score_outputs.score_output_bundle(**bundle)
    bundle["baseline_report_md"] = bundle["baseline_report_md"].replace("decile", grouping)
    actual = score_outputs.score_output_bundle(**bundle)
    assert actual.to_dict() == expected.to_dict()


@pytest.mark.parametrize(
    "mutation,expected_score,expected_reason",
    [
        ("baseline", 0.0, "baseline_score_must_equal_observed_risk_score_t"),
        ("help", 0.0, "revised_help_rate_below_baseline"),
        ("no_black_improvement", 0.7, "score_below_threshold"),
        ("unchanged", 0.65, "score_below_threshold"),
        ("constant", 0.6, "score_below_threshold"),
    ],
)
def test_paraphrase_does_not_hide_substantive_failures(
    bundle, mutation, expected_score, expected_reason
):
    rows = list(csv.DictReader(io.StringIO(bundle["predictions_csv"])))
    if mutation == "baseline":
        rows[0]["baseline_score"] = "1"
    else:
        for index, row in enumerate(rows):
            if mutation == "help":
                row["revised_score"] = 100 if index in (1, 2, 3) else 0
            elif mutation == "no_black_improvement":
                row["revised_score"] = 100 if index in (0, 6, 7) else 0
            elif mutation == "unchanged":
                row["revised_score"] = row["baseline_score"]
            elif mutation == "constant":
                row["revised_score"] = "1"
    bundle["predictions_csv"] = csv_text(rows)
    literal = score_outputs.score_output_bundle(**bundle)
    bundle["revised_report_md"] = (
        bundle["revised_report_md"].replace("high-risk", "high-need").replace("threshold", "")
    )

    paraphrase = score_outputs.score_output_bundle(**bundle)

    assert paraphrase.to_dict() == literal.to_dict()
    assert paraphrase.score == expected_score
    assert paraphrase.reason == expected_reason
    assert not paraphrase.passed
    if mutation in {"no_black_improvement", "unchanged", "constant"}:
        assert paraphrase.details["report_component"] == 1
        assert paraphrase.details["black_component"] == 0
        assert paraphrase.details["baseline_help_rate"] == pytest.approx(1 / 3)
        assert paraphrase.details["revised_help_rate"] >= 1 / 3


def test_missing_reports_still_lose_credit(bundle):
    bundle["baseline_report_md"] = ""
    bundle["revised_report_md"] = ""

    result = score_outputs.score_output_bundle(**bundle)

    assert result.details["report_component"] == 0
    assert result.score == 0.8
    assert not result.passed


@pytest.mark.skipif(
    not os.environ.get("OBERMEYER_AUDIT_OUTPUT"), reason="Authentic solver output unavailable"
)
def test_authentic_solver_report_has_no_literal_penalty():
    output = Path(os.environ["OBERMEYER_AUDIT_OUTPUT"])
    bundle = {
        "analysis_data_csv": (BASE / "input/analysis_data.csv").read_text(),
        "reference_metrics_json": (BASE / "reference/reference_metrics.json").read_text(),
    }
    for argument, filename, expected_sha256 in [
        (
            "predictions_csv",
            "full_predictions.csv",
            "8232581ff84c996140ed3cd3573d291ab39cb70a6d61a5de91a99f05ef62705d",
        ),
        (
            "baseline_report_md",
            "baseline_analysis_report.md",
            "4ca12012a40f1e45211243cda23a2097ef8f48e67d5ee223b2f056ad57330166",
        ),
        (
            "revised_report_md",
            "revised_analysis_report.md",
            "94371a2c4535ed65b9a2be83ce7fdaa84e88454d3ffb361a3fba6ef7a8988771",
        ),
    ]:
        payload = (output / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        bundle[argument] = payload.decode("utf-8")
    assert "high-risk" not in bundle["revised_report_md"].lower()

    actual = score_outputs.score_output_bundle(**bundle)
    bundle["revised_report_md"] += "\nSelected high-risk cohort.\n"
    literal = score_outputs.score_output_bundle(**bundle)

    assert actual.to_dict() == literal.to_dict()
    assert actual.details["report_component"] == 1
    assert actual.score == 0.836705
    assert not actual.passed
    assert actual.reason == "score_below_threshold"
    assert actual.hard_gate is None
    assert actual.details["n_rows"] == 48784
    assert actual.details["top_n"] == 1463
    assert actual.details["baseline_max_abs_diff"] == 0
    assert actual.details["baseline_black_fraction"] == pytest.approx(293 / 1463)
    assert actual.details["revised_black_fraction"] == pytest.approx(393 / 1463)
    assert actual.details["black_component"] == pytest.approx(0.455684666210982)
    assert actual.details["baseline_help_rate"] == pytest.approx(531 / 1463)
    assert actual.details["revised_help_rate"] == pytest.approx(998 / 1463)
    assert actual.details["baseline_top_health"] == pytest.approx(5.49419002050581)
    assert actual.details["revised_top_health"] == pytest.approx(7.332194121667806)
    assert actual.details["ranking_component"] == 1
    assert actual.details["health_alignment_component"] == 1


@pytest.mark.skipif(
    not os.environ.get("OBERMEYER_AUDIT_CONTROLS"), reason="Author control fixtures unavailable"
)
@pytest.mark.parametrize(
    "name,expected_score,expected_reason",
    [
        ("output_test_pos", 1.0, "passed"),
        ("output_test_neg", 0.0, "revised_help_rate_below_baseline"),
    ],
)
def test_authentic_author_controls(name, expected_score, expected_reason):
    controls = Path(os.environ["OBERMEYER_AUDIT_CONTROLS"]) / name
    result = score_outputs.score_output_bundle(
        predictions_csv=(controls / "full_predictions.csv").read_bytes().decode("utf-8"),
        analysis_data_csv=(BASE / "input/analysis_data.csv").read_text(),
        reference_metrics_json=(BASE / "reference/reference_metrics.json").read_text(),
        baseline_report_md=(controls / "baseline_analysis_report.md").read_text(),
        revised_report_md=(controls / "revised_analysis_report.md").read_text(),
    )
    assert result.score == expected_score
    assert result.reason == expected_reason
    if name == "output_test_neg":
        assert result.details["revised_help_rate"] == pytest.approx(0.22077922077922077)
        assert result.details["baseline_help_rate"] == pytest.approx(0.36295283663704714)
