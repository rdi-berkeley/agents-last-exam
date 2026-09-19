import base64
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import urllib.request

import pytest

from tasks.health_medicine.healthcare_bias_audit_27a_public_replication_v1.scripts import (
    score_outputs as scorer,
)


REPO = Path(__file__).resolve().parents[2]
TASK = "health_medicine/healthcare_bias_audit_27a_public_replication_v1"
DATA = REPO / "task-data-hf/extracted" / TASK / "base"


def answers_from_results(bundle):
    tables = {}
    for filename in ("figure1b", "table2_concentration_metric", "table3", "model_r2"):
        rows = list(csv.DictReader(io.StringIO(bundle[f"results/{filename}.csv"])))
        key = {
            "figure1b": "percentile",
            "table2_concentration_metric": "predictor",
            "table3": "population",
            "model_r2": "formula (y ~ x)",
        }[filename]
        tables[filename] = {row[key]: row for row in rows}
    figure = tables["figure1b"]["97"]
    table2 = tables["table2_concentration_metric"]
    table3 = tables["table3"]
    model = tables["model_r2"]
    answers = dict(scorer.EXACT_JSON_STRING_FIELDS)
    for suffix in ("before", "after", "ratio"):
        answers[f"figure1b_percentile_97_{suffix}"] = float(figure[suffix])
    for suffix, row in {
        "total_costs": "Total costs",
        "avoidable_costs": "Avoidable costs",
        "active_chronic_conditions": "Active chronic conditions",
        "best_worst_difference": "Best-worst difference",
    }.items():
        answers[f"table2_race_black_{suffix}"] = float(table2[row]["Race black"])
    for suffix, formula in {
        "risk_score": "gagne_sum_t ~ risk_score_t",
        "gagne_hat": "gagne_sum_t ~ gagne_sum_t_hat",
    }.items():
        answers[f"model_r2_gagne_on_{suffix}"] = float(model[formula]["holdout_r2"])
    for suffix, row in {
        "observed_program": "Observed program enrollment",
        "predicted_health_in_cost_bin": "Predicted health, in predicted-cost bin",
        "highest_predicted_cost": "Highest predicted cost",
        "worst_predicted_health": "Worst predicted health",
    }.items():
        answers[f"table3_{suffix}_frac_black"] = float(table3[row]["frac_black"])
    return answers


def complete_bundle(results):
    answers = answers_from_results(results)
    memo = (
        "For the clinical analytics lead: This public synthetic replication indicates that "
        "cost-based ranking understates clinical need for Black patients. Cost is an imperfect "
        "proxy for need because spending depends on access and utilization as well as illness. "
        "Diagnosis of this mechanism matters: retraining on the same cost outcome does not "
        "address the target mismatch. Prefer a need-based label for allocation, followed by "
        "clinical review. These synthetic results are distinct from the original paper's "
        "private data, not a replication of its numerical estimates. "
        f"Figure 1b at percentile 97 gives a Black share of {answers['figure1b_percentile_97_before']:.6f} "
        f"before and {answers['figure1b_percentile_97_after']:.6f} after the swap. "
        f"Table 2 gives {answers['table2_race_black_total_costs']:.6f} for total-cost ranking "
        f"and {answers['table2_race_black_active_chronic_conditions']:.6f} for active conditions."
    )
    return results | {"audit_answers.json": json.dumps(answers), "audit_memo.md": memo}


@pytest.fixture(scope="module")
def reference():
    if not DATA.is_dir():
        pytest.skip("local task data not staged")
    return {name: (DATA / "reference" / name).read_text() for name in scorer.REQUIRED_OUTPUT_FILES}


def test_independently_derived_answers_and_memo(reference):
    candidate = complete_bundle(
        {name: value for name, value in reference.items() if name.endswith(".csv")}
    )
    result = scorer.score_output_bundle(candidate_files=candidate, reference_files=reference)
    assert result.passed, result.to_dict()
    assert candidate["audit_memo.md"] != reference["audit_memo.md"]


@pytest.mark.parametrize("style", ["scientific", "quoted_crlf", "bom"])
def test_numeric_and_csv_serialization_equivalence(reference, style):
    candidate = dict(reference)
    for filename, content in reference.items():
        if not filename.endswith(".csv"):
            continue
        rows = list(csv.reader(io.StringIO(content)))
        if style == "scientific":
            for row in rows[1:]:
                for column, value in enumerate(row):
                    try:
                        row[column] = format(float(value), "+.16e")
                    except ValueError:
                        pass
        output = io.StringIO()
        writer = csv.writer(
            output, quoting=csv.QUOTE_ALL if style == "quoted_crlf" else csv.QUOTE_MINIMAL
        )
        writer.writerows(rows)
        candidate[filename] = ("\ufeff" if style == "bom" else "") + output.getvalue()
    result = scorer.score_output_bundle(candidate_files=candidate, reference_files=reference)
    assert result.passed, result.to_dict()


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "", "bad", "0.218", "0.7327"])
def test_rejects_wrong_or_malformed_scientific_answer(reference, value):
    candidate = dict(reference)
    answers = json.loads(candidate["audit_answers.json"])
    answers["model_r2_gagne_on_gagne_hat"] = value
    candidate["audit_answers.json"] = json.dumps(answers)
    assert not scorer.score_output_bundle(
        candidate_files=candidate, reference_files=reference
    ).passed


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e999", "bad", ""])
def test_rejects_malformed_predictions(reference, value):
    candidate = dict(reference)
    filename = "results/model_lasso_predictors.csv"
    rows = list(csv.reader(io.StringIO(candidate[filename])))
    rows[1][7] = value
    output = io.StringIO()
    csv.writer(output).writerows(rows)
    candidate[filename] = output.getvalue()
    assert not scorer.score_output_bundle(
        candidate_files=candidate, reference_files=reference
    ).passed


@pytest.mark.parametrize(
    "change",
    ["header", "order", "short_row", "missing_row", "extra_file", "json_array", "json_syntax"],
)
def test_rejects_malformed_bundle_structure(reference, change):
    candidate = dict(reference)
    filename = "results/model_lasso_predictors.csv"
    rows = list(csv.reader(io.StringIO(candidate[filename])))
    if change == "header":
        rows[0][0] = "wrong"
    elif change == "order":
        rows[1], rows[2] = rows[2], rows[1]
    elif change == "short_row":
        rows[1].pop()
    elif change == "missing_row":
        rows.pop()
    elif change == "extra_file":
        candidate["extra.txt"] = "extra"
    elif change == "json_array":
        candidate["audit_answers.json"] = "[]"
    else:
        candidate["audit_answers.json"] = "{"
    output = io.StringIO()
    csv.writer(output).writerows(rows)
    candidate[filename] = output.getvalue()
    assert not scorer.score_output_bundle(
        candidate_files=candidate, reference_files=reference
    ).passed


@pytest.mark.parametrize("delta,passed", [(5e-7, True), (2e-6, False), (5.7e-5, False)])
def test_scientific_tolerance_is_not_widened(reference, delta, passed):
    assert scorer.NUMERIC_TOLERANCE == 1e-6
    candidate = dict(reference)
    answers = json.loads(candidate["audit_answers.json"])
    answers["model_r2_gagne_on_gagne_hat"] += delta
    candidate["audit_answers.json"] = json.dumps(answers)
    assert (
        scorer.score_output_bundle(candidate_files=candidate, reference_files=reference).passed
        == passed
    )


def test_reference_r2_independently_recomputed(reference):
    rows = list(csv.DictReader(io.StringIO(reference["results/model_lasso_predictors.csv"])))
    for summary in csv.DictReader(io.StringIO(reference["results/model_r2.csv"])):
        outcome, predictor = summary["formula (y ~ x)"].split(" ~ ")
        observed = [
            math.log10(float(row[outcome[4:]]) + 1)
            if outcome.startswith("log_")
            else float(row[outcome])
            for row in rows
        ]
        predicted = [float(row[predictor]) for row in rows]
        mean_observed = math.fsum(observed) / len(rows)
        mean_predicted = math.fsum(predicted) / len(rows)
        covariance = math.fsum(
            (actual - mean_observed) * (prediction - mean_predicted)
            for actual, prediction in zip(observed, predicted)
        )
        variance_observed = math.fsum((actual - mean_observed) ** 2 for actual in observed)
        variance_predicted = math.fsum(
            (prediction - mean_predicted) ** 2 for prediction in predicted
        )
        r_squared = covariance**2 / (variance_observed * variance_predicted)
        assert r_squared == pytest.approx(float(summary["holdout_r2"]), abs=1e-12)
        assert len(rows) == int(float(summary["holdout_obs"]))


@pytest.fixture(scope="module")
def guest_results():
    port = os.environ.get("HEALTHCARE_BIAS_GUEST_PORT")
    if not port:
        pytest.skip(
            "set HEALTHCARE_BIAS_GUEST_PORT for independently computed guest artifact checks"
        )
    root = os.environ.get("HEALTHCARE_BIAS_GUEST_ROOT", "/tmp/fairness-healthcare-20260907")
    script = f"""from pathlib import Path
import base64, gzip, hashlib, json
root = Path({root!r})
payload = {{}}
for name in ('legacy-run', 'recipe-run', 'modern-r-run', 'unnormalized-run'):
    folder = root / name
    payload[name] = {{'results/' + path.name: path.read_text() for path in (folder / 'results').glob('*.csv')}}
payload['source_hashes'] = {{str(path.relative_to(root / 'legacy-run')): hashlib.sha256(path.read_bytes()).hexdigest() for path in (root / 'legacy-run' / 'code').rglob('*') if path.is_file() and path.suffix in ('.py', '.R')}}
payload['data_hash'] = hashlib.sha256((root / 'legacy-run/data/data_new.csv').read_bytes()).hexdigest()
payload['packages'] = {{entry['name']: (entry['version'], entry['build']) for path in (root / 'legacy/conda-meta').glob('*.json') for entry in [json.loads(path.read_text())]}}
print(base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode())
"""
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}/cmd",
        data=json.dumps(
            {"command": "run_command", "params": {"command": "python3 -c " + shlex.quote(script)}}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        events = [
            json.loads(line[6:])
            for line in response.read().decode().splitlines()
            if line.startswith("data: ")
        ]
    result = events[-1]
    assert result["return_code"] == 0, result
    return json.loads(gzip.decompress(base64.b64decode(result["stdout"])))


@pytest.mark.parametrize("run", ["legacy-run", "recipe-run"])
def test_actual_legacy_pipeline_matches_all_reference_cells(reference, guest_results, run):
    assert (
        guest_results["data_hash"]
        == hashlib.sha256((DATA / "input/data/data_new.csv").read_bytes()).hexdigest()
    )
    expected_sources = {
        str(path.relative_to(DATA / "input")): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (DATA / "input/code").rglob("*")
        if path.is_file() and path.suffix in (".py", ".R")
    }
    assert guest_results["source_hashes"] == expected_sources
    candidate = complete_bundle(guest_results[run])
    result = scorer.score_output_bundle(candidate_files=candidate, reference_files=reference)
    assert result.passed, result.to_dict()


@pytest.mark.parametrize(
    "run,filename",
    [("modern-r-run", "table3.csv"), ("unnormalized-run", "model_lasso_predictors.csv")],
)
def test_actual_changed_runtime_or_model_remains_rejected(reference, guest_results, run, filename):
    candidate = complete_bundle(guest_results["legacy-run"])
    key = "results/" + filename
    candidate[key] = guest_results[run][key]
    result = scorer.score_output_bundle(candidate_files=candidate, reference_files=reference)
    assert not result.passed
    assert result.reason == key + ": value_mismatch", result.to_dict()


def test_generated_prompt_and_card_publish_identical_runtime_contract():
    from tasks.health_medicine.healthcare_bias_audit_27a_public_replication_v1 import main

    card = json.loads(Path(main.__file__).with_name("task_card.json").read_text())
    for prompt in (main.config.task_description, card["taskPrompt"]):
        assert main.RUNTIME_CONTRACT in prompt
        assert "absolute tolerance `1e-6`" in prompt
        assert "takes precedence" in prompt
        assert "wrappers are not provided" in prompt
        assert "LassoCV(normalize=True)" in prompt
        assert "R 3.5.1 sampling behavior" in prompt
    assert "preprovisioned" not in card["summary"]
    assert "exact isolated runtime recipe" in card["agentMustDo"][1]


def test_published_runtime_package_builds_were_actually_used(guest_results):
    from tasks.health_medicine.healthcare_bias_audit_27a_public_replication_v1 import main

    command = next(
        line
        for line in main.RUNTIME_CONTRACT.replace("\\\n", "").splitlines()
        if line.startswith("micromamba create ")
    )
    specs = [word for word in shlex.split(command) if word.count("=") == 2]
    assert len(specs) == 14
    for spec in specs:
        package, version, build = spec.split("=")
        assert guest_results["packages"][package] == [version, build]


def test_clean_recipe_run_reproduces_all_csv_bytes(guest_results):
    assert guest_results["recipe-run"] == guest_results["legacy-run"]
