from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tasks.education_info.moodle_gradebook_closeout_reconciliation.scripts import bundle_lib


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "task-data-hf/extracted/education_info/moodle_gradebook_closeout_reconciliation/base"


@pytest.fixture
def release():
    if not DATA.exists():
        pytest.skip("Moodle release data is not installed")
    return DATA


def test_backup_keeps_boolean_strings_and_numeric_item_fields(release):
    state = bundle_lib.read_backup_state(release / "input/starter_project/course_backup.mbz")
    assert set(state["submissions"]["excused"]) == {"true", "false"}
    assert set(state["submissions"]["missing"]) == {"true", "false"}
    assert set(state["final_grade_flags"]["locked"]) == {"true", "false"}
    assert pd.api.types.is_numeric_dtype(state["items"]["max_points"])
    assert pd.api.types.is_bool_dtype(state["items"]["lateness_allowed"])


def test_excused_and_locked_grades_reach_computation(release):
    state = bundle_lib.read_backup_state(release / "reference/corrected_course.mbz")
    roster = pd.read_csv(release / "input/starter_project/roster.csv")
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
    detail, audit, _ = bundle_lib.compute_grade_outputs(roster=roster, **arguments)
    expected_excused = state["submissions"].query("excused == 'true'")
    assert len(expected_excused) > 0
    actual_excused = detail[detail["status"] == "excused"]
    assert set(zip(actual_excused.moodle_user_id, actual_excused.item_id)) == set(
        zip(expected_excused.moodle_user_id, expected_excused.item_id)
    )
    assert (actual_excused.effective_score == "").all()
    assert audit.excused_count.sum() == len(expected_excused)
    student = roster.loc[roster.status == "active", "moodle_user_id"].iloc[0]
    flags = state["final_grade_flags"].copy()
    flags.loc[
        flags.moodle_user_id == student, ["locked", "cached_numeric_grade", "cached_letter_grade"]
    ] = ["true", "42", "F"]
    arguments["final_grade_flags"] = flags
    _, _, exported = bundle_lib.compute_grade_outputs(roster=roster, **arguments)
    row = exported.loc[exported.moodle_user_id == student].iloc[0]
    assert row.final_numeric_grade == "42.00"
    assert row.final_letter_grade == "F"


@pytest.mark.parametrize(
    "backup,expected",
    [("reference/corrected_course.mbz", 100.0), ("input/starter_project/course_backup.mbz", 12.33)],
)
def test_visible_rebuild_against_regenerated_reference(release, tmp_path, backup, expected):
    spec = importlib.util.spec_from_file_location(
        "moodle_visible_release", release / "input/bundle_lib.py"
    )
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    try:
        spec.loader.exec_module(helper)
        helper.build_submission_outputs(
            release / "input",
            release / backup,
            release / "input/starter_project/roster.csv",
            tmp_path,
        )
    finally:
        sys.modules.pop(spec.name, None)
    result = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "tasks/education_info/moodle_gradebook_closeout_reconciliation/scripts/score_outputs.py"
            ),
            "--submission",
            str(tmp_path),
            "--ground-truth",
            str(release / "reference"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["score"] == expected, report
    assert not report["hard_failed"]
