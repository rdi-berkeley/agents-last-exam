from __future__ import annotations

import importlib.util
import contextlib
import hashlib
import io
import json
import re
import shutil
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from tasks.education_info.moodle_gradebook_closeout_reconciliation.scripts import bundle_lib


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "task-data-hf/extracted/education_info/moodle_gradebook_closeout_reconciliation/base"
SCRIPTS = ROOT / "tasks/education_info/moodle_gradebook_closeout_reconciliation/scripts"
TASK = "education_info/moodle_gradebook_closeout_reconciliation"
RELEASE = ROOT.parent / "release-staging/ale-task-fixes-20260913"


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.setitem(sys.modules, "bundle_lib", bundle_lib)
    spec = importlib.util.spec_from_file_location(
        "moodle_post_validation", SCRIPTS / "score_outputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_csv_blank_score_does_not_erase_unchanged_rows(scorer, tmp_path):
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    candidate.write_text("sourcedId,score,status\nu001,75.60,completed\nu002,0.00,completed\n")
    reference.write_text("sourcedId,score,status\nu001,75.60,completed\nu002,,exempt\n")
    assert scorer.compare_csv_rows(candidate, reference, ["sourcedId"]) == 0.5
    candidate.write_text("sourcedId,score,status\nu001,75.61,completed\nu002,0.00,completed\n")
    assert scorer.compare_csv_rows(candidate, reference, ["sourcedId"]) == 0


def test_csv_identifier_spelling_is_not_numeric(scorer, tmp_path):
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    candidate.write_text("sourcedId,score\n001,0.00\n")
    reference.write_text("sourcedId,score\n1,0.00\n")
    assert scorer.compare_csv_rows(candidate, reference, ["sourcedId"]) == 0


def test_actual_missing_late_homework_remains_missing():
    if not DATA.exists():
        pytest.skip("Moodle release data is not installed")
    state = bundle_lib.read_backup_state(DATA / "reference/corrected_course.mbz")
    roster = pd.read_csv(DATA / "input/starter_project/roster.csv")
    arguments = {
        key: state[key]
        for key in (
            "items",
            "submissions",
            "policy",
            "section_cutoffs",
            "id_map",
            "final_grade_flags",
        )
    }
    detail, audit, final = bundle_lib.compute_grade_outputs(roster=roster, **arguments)
    results = bundle_lib.build_oneroster_tables(roster, state["items"], detail, final)[
        "results.csv"
    ]
    lookup = results.set_index("sourcedId")
    for student in ("u083", "u156"):
        raw = (
            state["submissions"].query("moodle_user_id == @student and item_id == 'hw_02'").iloc[0]
        )
        assert raw["missing"] == "true" and raw["excused"] == "false"
        assert raw["raw_score"] == raw["override_score"] == ""
        assert int(raw["late_days"]) > 0
        row = lookup.loc[f"{student}-hw_02"]
        assert row["score"] == "0.00"
        assert row["status"] == "missing"
    assert lookup.loc["u088-quiz_01", "status"] == "exempt"
    assert lookup.loc["u088-quiz_01", "score"] == ""
    assert lookup.loc["u067-hw_03", "status"] == "completed"
    assert float(lookup.loc["u067-hw_03", "score"]) > 0
    assert lookup.loc["u101-project", "status"] == "completed"
    assert set(audit.loc[audit.moodle_user_id.isin(["u083", "u156"]), "final_numeric_grade"]) == {
        69.77,
        73.86,
    }
    for raw_score, override_score, missing in [("0", "", "false"), ("", "0", "true")]:
        selected = (state["submissions"].moodle_user_id == "u083") & (
            state["submissions"].item_id == "hw_02"
        )
        state["submissions"].loc[selected, ["raw_score", "override_score", "missing"]] = [
            raw_score,
            override_score,
            missing,
        ]
        detail, _, final = bundle_lib.compute_grade_outputs(roster=roster, **arguments)
        results = bundle_lib.build_oneroster_tables(roster, state["items"], detail, final)[
            "results.csv"
        ]
        row = results.set_index("sourcedId").loc["u083-hw_02"]
        assert row["score"] == "0.00" and row["status"] == "completed"


def load_source(name, source, monkeypatch):
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def score_outputs(scorer, submission, reference):
    scorer.parse_args = lambda: types.SimpleNamespace(
        submission=submission, ground_truth=reference, output=None
    )
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        assert scorer.main() == 0
    return json.loads(stream.getvalue())


@pytest.fixture
def overlay_release(tmp_path):
    selector = RELEASE / "validation-queue-20260917/comparison.json"
    if not selector.exists():
        pytest.skip("Selected Moodle validation artifacts are not installed")
    selected = next(row for row in json.loads(selector.read_text())["rows"] if row["task"] == TASK)
    for label in ("kimi_before", "kimi_after", "codex_after_repair"):
        path = Path(selected[label]["run_json"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == selected[label]["sha256"]
    overlay = RELEASE / "post-validation-20260918/data-overrides" / TASK / "base"
    reference = tmp_path / "reference"
    shutil.copytree(DATA / "reference", reference)
    shutil.copytree(overlay / "reference", reference, dirs_exist_ok=True)
    return selected, overlay, reference


def test_staged_helper_and_reference_have_only_observed_changes(overlay_release):
    _, overlay, _ = overlay_release
    original = (DATA / "input/bundle_lib.py").read_text()
    patched = (overlay / "input/bundle_lib.py").read_text()
    assert patched == original.replace(
        'detail["status"] == "missing_zero"',
        'detail["status"].removesuffix("_late") == "missing_zero"',
    )
    old_rows = pd.read_csv(
        DATA / "reference/oneroster_package/results.csv", dtype=str, keep_default_na=False
    ).set_index("sourcedId")
    new_rows = pd.read_csv(
        overlay / "reference/oneroster_package/results.csv", dtype=str, keep_default_na=False
    ).set_index("sourcedId")
    expected = old_rows.copy()
    expected.loc[["u083-hw_02", "u156-hw_02"], "status"] = "missing"
    pd.testing.assert_frame_equal(new_rows, expected)
    manifest = pd.read_csv(
        overlay / "reference/oneroster_package/manifest.csv", dtype=str
    ).set_index("fileName")
    content = (overlay / "reference/oneroster_package/results.csv").read_bytes()
    assert manifest.loc["results.csv", "sha256"] == hashlib.sha256(content).hexdigest()
    assert manifest.loc["results.csv", "rowCount"] == "2820"


@pytest.mark.parametrize("label,expected", [("kimi_after", 100.0), ("codex_after_repair", 85.69)])
def test_actual_repaired_backups_regenerate_only_missing_statuses(
    overlay_release, scorer, monkeypatch, tmp_path, label, expected
):
    selected, overlay, reference = overlay_release
    original = Path(selected[label]["run_json"]).parent / "output"
    frozen = load_source(
        "moodle_frozen_helper", (DATA / "input/bundle_lib.py").read_text(), monkeypatch
    )
    fixed = load_source(
        "moodle_fixed_helper", (overlay / "input/bundle_lib.py").read_text(), monkeypatch
    )
    before = tmp_path / "before"
    after = tmp_path / "after"
    for helper, destination in ((frozen, before), (fixed, after)):
        helper.build_submission_outputs(
            DATA / "input",
            original / "corrected_course.mbz",
            DATA / "input/starter_project/roster.csv",
            destination,
        )
    paths = [path.relative_to(before) for path in before.rglob("*") if path.is_file()]
    assert len(paths) == 13
    assert all((before / path).read_bytes() == (original / path).read_bytes() for path in paths)
    changed = {
        str(path) for path in paths if (before / path).read_bytes() != (after / path).read_bytes()
    }
    assert changed == {"oneroster_package/results.csv", "oneroster_package/manifest.csv"}
    before_score = score_outputs(scorer, before, DATA / "reference")
    after_score = score_outputs(scorer, after, reference)
    stale_score = score_outputs(scorer, before, reference)
    assert before_score["score"] == after_score["score"] == expected
    assert stale_score["score"] < after_score["score"]
    assert (after / "oneroster_package/results.csv").read_bytes() == (
        reference / "oneroster_package/results.csv"
    ).read_bytes()
    print(
        json.dumps(
            {
                "moodle": label,
                "before": before_score["score"],
                "rebuild": after_score["score"],
                "stale_output": stale_score["score"],
                "changed": sorted(changed),
            }
        )
    )


@pytest.mark.parametrize(
    "backup,expected",
    [("reference/corrected_course.mbz", 100), ("input/starter_project/course_backup.mbz", 12.33)],
)
def test_patched_visible_helper_positive_and_unrepaired_controls(
    overlay_release, scorer, monkeypatch, tmp_path, backup, expected
):
    _, overlay, reference = overlay_release
    helper = load_source(
        "moodle_control_helper", (overlay / "input/bundle_lib.py").read_text(), monkeypatch
    )
    output = tmp_path / "output"
    helper.build_submission_outputs(
        DATA / "input", DATA / backup, DATA / "input/starter_project/roster.csv", output
    )
    report = score_outputs(scorer, output, reference)
    assert report["score"] == expected, report
    assert not report["hard_failed"]


def test_hash_verified_old_workflow_recovers_only_correct_csv_rows(
    overlay_release, scorer, monkeypatch, tmp_path
):
    selected, _, _ = overlay_release
    trajectory = json.loads(
        Path(selected["kimi_before"]["run_json"]).with_name("trajectory.json").read_text()
    )
    calls = {
        call["id"]: call for step in trajectory["steps"] for call in step.get("tool_calls", [])
    }
    lines = {}
    for step in trajectory["steps"]:
        for result in (step.get("observation") or {}).get("results", []):
            call = calls.get(result["tool_call_id"], {})
            if call.get("name") != "Read" or not call["arguments"]["path"].endswith(
                "/input/bundle_lib.py"
            ):
                continue
            for content in result.get("content", []):
                for line in content.get("text", "").splitlines():
                    match = re.match(r"^(\d+)\t(.*)$", line)
                    if match:
                        number, value = int(match[1]), match[2]
                        assert number not in lines or lines[number] == value
                        lines[number] = value
    assert sorted(lines) == list(range(1, 1625))
    source = "\n".join(lines[number] for number in sorted(lines)) + "\n"
    assert (
        hashlib.sha256(source.encode()).hexdigest()
        == "e9ca736194c3fc7a8a951ba486a6497c437126ceb24b409921b347cad0b7b70c"
    )
    helper = load_source("moodle_old_observed_helper", source, monkeypatch)
    old_output = tmp_path / "old_reconstruction"
    backup = Path(selected["kimi_after"]["run_json"]).parent / "output/corrected_course.mbz"
    helper.build_submission_outputs(
        DATA / "input", backup, DATA / "input/starter_project/roster.csv", old_output
    )
    result_path = old_output / "oneroster_package/results.csv"
    assert (
        hashlib.sha256(result_path.read_bytes()).hexdigest()
        == "80b8443d21fe0991f59f9f37defde5fd49240c529836c0993eba08062d7c72d8"
    )
    before_path = RELEASE / "post-validation-20260918/education-evidence/moodle-scorer-before.py"
    before = load_source("moodle_before_scorer", before_path.read_text(), monkeypatch)
    assert (
        hashlib.sha256(before_path.read_bytes()).hexdigest()
        == "f1c8b3ded6e0820900d85eda28b1c592042d9f1d994ede06eab17b4cceaad8c1"
    )
    reference_path = DATA / "reference/oneroster_package/results.csv"
    assert before.compare_csv_rows(result_path, reference_path, ["sourcedId"]) == 0
    assert scorer.compare_csv_rows(result_path, reference_path, ["sourcedId"]) == pytest.approx(
        2812 / 2820
    )
    assert score_outputs(before, old_output, DATA / "reference")["score"] == 86.7
    assert score_outputs(scorer, old_output, DATA / "reference")["score"] == 88.03
    print(
        json.dumps(
            {
                "moodle": "old_workflow_reconstruction",
                "results_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
                "old_matching_rows": 0,
                "new_matching_rows": 2812,
                "total_rows": 2820,
                "before": 86.7,
                "after_dtype_only": 88.03,
            }
        )
    )
