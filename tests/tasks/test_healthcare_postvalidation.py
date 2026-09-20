import ast
import asyncio
import csv
import hashlib
import io
import json
import logging
import re
import shlex
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tasks.health_medicine.healthcare_bias_audit_27a_public_replication_v1.scripts import (
    score_outputs as healthcare,
)
from tasks.health_medicine.obermeyer_bias_reproduction.scripts import score_outputs as obermeyer


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT.parent / "release-staging/ale-task-fixes-20260913"
TASKS = {
    "healthcare_bias_audit_27a_public_replication_v1": healthcare,
    "obermeyer_bias_reproduction": obermeyer,
}
FROZEN_HASHES = {
    "healthcare_bias_audit_27a_public_replication_v1": "404032ce7cb6cc874d7fe2c28d6637f0ed770085ba670c8b62e7d0438f317d15",
    "obermeyer_bias_reproduction": "e5baa9d2cc174a31f4f852e09da6b6cb28f7cc4f6931f86ee1d933faafc5ac1f",
}
OBERMEYER_FILES = {
    "predictions_csv": "full_predictions.csv",
    "baseline_report_md": "baseline_analysis_report.md",
    "revised_report_md": "revised_analysis_report.md",
}
BASELINE_REPORT = (
    "Black and White patients have different Gagne burden at similar risk. "
    "Cost also differs at similar need. The top 3% group has 6 patients; "
    "mean burden is 5 versus 4 and mean cost is 200 versus 100."
)
REVISED_REPORT = (
    "The counterfactual revised ranking predicts medical need, `gagne_sum_t`. "
    "Black patients account for 2 of the 6 selected patients; non-Black patients "
    "account for 4. Mean selected burden rises from 5 to 7."
)


@pytest.fixture(scope="module")
def artifacts():
    comparison = RELEASE / "validation-queue-20260917/comparison.json"
    if not comparison.is_file():
        pytest.skip("retained R02 healthcare validation artifacts unavailable")
    protected = {}

    def read(path):
        payload = path.read_bytes()
        protected[path] = hashlib.sha256(payload).hexdigest()
        return payload.decode("utf-8")

    rows = json.loads(read(comparison))["rows"]
    bundles = {}
    for task, scorer in TASKS.items():
        selection = next(row for row in rows if row["task"] == f"health_medicine/{task}")
        base = RELEASE / "data-v1.1/files/health_medicine" / task / "base"
        if scorer is healthcare:
            reference = {
                name: read(base / "reference" / name) for name in scorer.REQUIRED_OUTPUT_FILES
            }
        else:
            reference = {
                "analysis_data_csv": read(base / "input/analysis_data.csv"),
                "reference_metrics_json": read(base / "reference/reference_metrics.json"),
            }
        for label in ("kimi_after", "codex_after_repair"):
            record = selection[label]
            run = Path(record["run_json"])
            read(run)
            assert protected[run] == record["sha256"]
            output = run.parent / "output"
            if scorer is healthcare:
                arguments = {
                    "candidate_files": {
                        name: read(output / name) for name in scorer.REQUIRED_OUTPUT_FILES
                    },
                    "reference_files": reference,
                }
            else:
                arguments = reference | {
                    argument: read(output / filename)
                    for argument, filename in OBERMEYER_FILES.items()
                }
            bundles[task, label] = arguments
    yield bundles
    for path, expected in protected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path


@pytest.fixture(scope="module")
def frozen_scorers(artifacts):
    archive_path = (
        RELEASE / "validation-source-context-qemu-r02-20260917/private-validation-context.tar.gz"
    )
    modules = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for task in TASKS:
            filename = f"tasks/health_medicine/{task}/scripts/score_outputs.py"
            payload = archive.extractfile(filename).read()
            assert hashlib.sha256(payload).hexdigest() == FROZEN_HASHES[task]
            module = ModuleType(f"healthcare_postvalidation_frozen_{task}")
            sys.modules[module.__name__] = module
            exec(compile(payload, filename, "exec"), module.__dict__)
            modules[task] = module
    yield modules
    for module in modules.values():
        sys.modules.pop(module.__name__, None)


def replay_evaluate(task, arguments):
    scorer = TASKS[task]
    path = ROOT / "tasks/health_medicine" / task / "main.py"
    tree = ast.parse(path.read_text())
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
            "_as_text",
            "_run_command",
            "_list_relative_files",
            "_log_score",
            "evaluate",
        }:
            node.decorator_list = []
            functions.append(node)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        + functions,
        type_ignores=[],
    )
    namespace = {
        "json": json,
        "shlex": shlex,
        "logger": logging.getLogger(__name__),
        "TASK_NAME": task,
        "score_output_bundle": scorer.score_output_bundle,
        "REQUIRED_OUTPUT_FILES": healthcare.REQUIRED_OUTPUT_FILES,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    if scorer is healthcare:
        files = {
            f"/{prefix}/{name}": value
            for prefix, key in (("candidate", "candidate_files"), ("reference", "reference_files"))
            for name, value in arguments[key].items()
        }
        metadata = {
            "remote_output_dir": "/candidate",
            "reference_dir": "/reference",
            "reference_answers_file": "/reference/audit_answers.json",
            "reference_memo_file": "/reference/audit_memo.md",
        }
    else:
        files = {f"/{argument}": value for argument, value in arguments.items()}
        metadata = {
            "variant_name": "base",
            "predictions_output": "/predictions_csv",
            "baseline_report_output": "/baseline_report_md",
            "revised_report_output": "/revised_report_md",
            "analysis_data_file": "/analysis_data_csv",
            "reference_metrics_file": "/reference_metrics_json",
        }

    class ReadOnlySession:
        async def file_exists(self, filename):
            return filename in files

        async def directory_exists(self, filename):
            return False

        async def read_file(self, filename):
            return files[filename]

        async def run_command(self, command, **kwargs):
            assert "find /candidate -type f" in command
            return {"return_code": 0, "stdout": "\n".join(arguments["candidate_files"])}

    return asyncio.run(namespace["evaluate"](SimpleNamespace(metadata=metadata), ReadOnlySession()))


@pytest.mark.parametrize(
    "task,label,before,after",
    [
        ("healthcare_bias_audit_27a_public_replication_v1", "kimi_after", 0.0, 1.0),
        ("healthcare_bias_audit_27a_public_replication_v1", "codex_after_repair", 1.0, 1.0),
        ("obermeyer_bias_reproduction", "kimi_after", 0.821642, 0.847642),
        ("obermeyer_bias_reproduction", "codex_after_repair", 0.838072, 0.838072),
    ],
)
def test_selected_native_artifact_replay(artifacts, frozen_scorers, task, label, before, after):
    arguments = artifacts[task, label]
    original = frozen_scorers[task].score_output_bundle(**arguments)
    actual = TASKS[task].score_output_bundle(**arguments)
    print(
        json.dumps(
            {"task": task, "run": label, "before": original.to_dict(), "after": actual.to_dict()}
        )
    )
    assert original.score == before
    assert actual.score == after
    assert replay_evaluate(task, arguments) == [after]
    if TASKS[task] is obermeyer:
        assert actual.details["report_component"] == 1.0
        assert not actual.passed
        assert actual.hard_gate is None
        assert {
            key: value for key, value in actual.details.items() if key != "report_component"
        } == {key: value for key, value in original.details.items() if key != "report_component"}


@pytest.mark.parametrize(
    "claim",
    [
        "Our Figure 1b result increases from 17.7% to 46.5% Black.",
        "Our Figure 1b result is 59% Black after correction.",
        "The public synthetic Figure 1b result is 46.5% Black after correction.",
        "For the graded answer, Figure 1b reaches 59 percent Black.",
        "Random selection within predicted-cost bins enrolls 17.7% Black; Figure 1b reaches 46.5%.",
        "Highest predicted cost enrolls 17.7% Black.",
    ],
)
def test_genuine_private_value_substitutions_remain_rejected(artifacts, claim):
    original = artifacts["healthcare_bias_audit_27a_public_replication_v1", "codex_after_repair"]
    candidate = dict(original["candidate_files"])
    candidate["audit_memo.md"] += "\n\n" + claim
    result = healthcare.score_output_bundle(
        candidate_files=candidate, reference_files=original["reference_files"]
    )
    assert result.score == 0
    assert result.reason == "audit_memo.md: private_paper_value_present_without_context"


def test_truthful_random_row_is_not_a_global_number_exemption(artifacts):
    arguments = artifacts["healthcare_bias_audit_27a_public_replication_v1", "kimi_after"]
    candidate = dict(arguments["candidate_files"])
    candidate["audit_memo.md"] += "\n\nFigure 1b starts with 17.7% Black."
    result = healthcare.score_output_bundle(
        candidate_files=candidate, reference_files=arguments["reference_files"]
    )
    assert result.reason == "audit_memo.md: private_paper_value_present_without_context"


@pytest.mark.parametrize("reference_fraction", [None, 0.115, 0.224])
def test_random_percentage_requires_the_matching_table3_cell(reference_fraction):
    claim = "Random selection within predicted-cost bins enrolled 17.7% Black."
    assert healthcare._check_forbidden_paper_values(claim.lower(), reference_fraction)
    assert healthcare._check_forbidden_paper_values(claim.lower(), 0.177) is None


@pytest.mark.parametrize(
    "claim",
    [
        "The original paper reported 17.7% to 46.5%.",
        "The private-data result was corrected to 59 percent.",
        "Do not use 17.7% to 46.5% as the graded answer.",
    ],
)
def test_explicit_private_paper_context_remains_allowed(claim):
    assert healthcare._check_forbidden_paper_values(claim.lower(), 0.177) is None


@pytest.mark.parametrize("task", list(TASKS))
def test_off_topic_reports_do_not_get_report_credit(artifacts, task):
    original = artifacts[task, "codex_after_repair"]
    off_topic = "Office relocation memo: arrange packing, desk delivery, and building access."
    if TASKS[task] is healthcare:
        arguments = original | {
            "candidate_files": original["candidate_files"] | {"audit_memo.md": off_topic}
        }
        result = healthcare.score_output_bundle(**arguments)
        assert result.score == 0
        assert result.reason == "audit_memo.md: missing_required_content"
    else:
        arguments = original | {"baseline_report_md": off_topic, "revised_report_md": off_topic}
        result = obermeyer.score_output_bundle(**arguments)
        assert result.details["report_component"] == 0
        assert result.score == 0.638072


def test_obermeyer_equivalences_are_grounded_in_cohort_and_outcome():
    literal = REVISED_REPORT.replace("non-Black patients", "White patients").replace(
        "`gagne_sum_t`", "health"
    )
    expected = obermeyer._report_component(
        BASELINE_REPORT, literal, cohort_races={"black", "white"}
    )
    assert expected == 1
    assert (
        obermeyer._report_component(
            BASELINE_REPORT, REVISED_REPORT, cohort_races={"black", "white"}
        )
        == expected
    )
    assert (
        obermeyer._report_component(
            BASELINE_REPORT, REVISED_REPORT, cohort_races={"black", "white", "other"}
        )
        < expected
    )
    assert (
        obermeyer._report_component(
            BASELINE_REPORT,
            REVISED_REPORT.replace("gagne_sum_t", "gagne_sum_tm1"),
            cohort_races={"black", "white"},
        )
        < expected
    )


def test_actual_obermeyer_numeric_gates_are_unchanged(artifacts):
    original = artifacts["obermeyer_bias_reproduction", "kimi_after"]
    rows = list(csv.DictReader(io.StringIO(original["predictions_csv"])))
    rows[0]["baseline_score"] = str(float(rows[0]["baseline_score"]) + 0.01)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=obermeyer.REQUIRED_OUTPUT_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    result = obermeyer.score_output_bundle(**(original | {"predictions_csv": buffer.getvalue()}))
    assert result.score == 0
    assert result.reason == "baseline_score_must_equal_observed_risk_score_t"


def test_actual_healthcare_numeric_cells_are_still_required(artifacts):
    original = artifacts["healthcare_bias_audit_27a_public_replication_v1", "kimi_after"]
    candidate = dict(original["candidate_files"])
    historical_table = (
        RELEASE / "validation-checks/healthcare-bias-r02-old-table3-from-terminal.csv"
    )
    candidate["results/table3.csv"] = historical_table.read_text()
    assert candidate["results/table3.csv"] != original["candidate_files"]["results/table3.csv"]
    result = healthcare.score_output_bundle(
        candidate_files=candidate, reference_files=original["reference_files"]
    )
    assert result.score == 0
    assert result.reason == "results/table3.csv: value_mismatch"


def test_actual_obermeyer_report_uses_the_documented_equivalent_concepts(artifacts):
    report = artifacts["obermeyer_bias_reproduction", "kimi_after"]["revised_report_md"].lower()
    assert not re.search(r"\bwhite\b|\bhealth\b", report)
    assert "gagne_sum_t" in report
    assert "non-black patients" in report
