from __future__ import annotations

import csv
import hashlib
import json
import runpy
import shutil
from pathlib import Path

import pytest

from tasks.education_info.homework_grading_numerical_pdes_instance_02.scripts.score_outputs import (
    _lower_contains,
    _matches_requirement,
    score_submission,
)


ROOT = Path(__file__).resolve().parents[2]
TASK = "education_info/homework_grading_numerical_pdes_instance_02"
DATA = ROOT / "task-data-hf/extracted" / TASK / "base"
RUBRIC = json.loads((ROOT / "tasks" / TASK / "grading_rubric.json").read_text())


def test_published_component_credits_reproduce_existing_gold():
    if not DATA.exists():
        pytest.skip("PDE release data is not installed")
    parts = RUBRIC["problems"]
    full = {
        part: sum(max(component.values()) for component in rule["components"].values())
        for part, rule in parts.items()
    }
    assert full == {"1a": 2, "1b": 3, "2a": 2, "2b": 3}
    scores = {student: dict(full) for student in ("S01", "S02", "S03", "S04", "S05")}
    scores["S02"]["1b"] = parts["1b"]["components"]["numerical_substitution"][
        "correct_including_error_carried_forward"
    ]
    scores["S02"]["2b"] -= parts["2b"]["components"]["test_space"]["correct"]
    scores["S03"]["1a"] = parts["1a"]["components"]["euler_structure_and_scaling"]["correct"]
    scores["S03"]["1b"] = scores["S02"]["1b"]
    scores["S03"]["2a"] = 0
    components = parts["2b"]["components"]
    scores["S03"]["2b"] = (
        components["bilinear_form"]["derivative_product_with_wrong_overall_sign"]
        + components["linear_functional"]["correct"]
        + components["test_space"]["correct"]
    )
    scores["S04"]["2b"] = components["bilinear_form"]["correct"]
    scores["S05"]["1b"] = parts["1b"]["components"]["sharp_symbolic_restriction"]["correct"]
    with (DATA / "reference/gold_scores.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            expected = scores[row["student_id"]]
            assert all(float(row[f"problem_{part}"]) == credit for part, credit in expected.items())
            assert float(row["total_score"]) == sum(expected.values())
    assert "either direction" in RUBRIC["error_tags"]["stability_factor_of_two_error"]
    assert "incorrect symbolic bound" in RUBRIC["error_tags"]["wrong_dt_max"]


@pytest.mark.parametrize("phrase", ["-2 u_i^n", "-2*u_i^n", "- 2 * u_i^{n}"])
def test_equivalent_stencil_notation(phrase):
    assert _matches_requirement(phrase, RUBRIC["feedback_checks"]["centered_stencil"])


@pytest.mark.parametrize(
    "text,check",
    [
        ("Use -u_i^n", "centered_stencil"),
        ("Use +2*u_i^n", "centered_stencil"),
        ("dt_max = 0.001", "numerical_maximum"),
        ("dt_max = 0.0101", "numerical_maximum"),
        ("dt <= dx^2/(4*kappa)", "sharp_stability_bound"),
        ("a(u,v) = - integral_0^1 u'(x) v'(x) dx", "positive_weak_form"),
        ("a(u,v) is correct with a negative sign", "positive_weak_form"),
        ("All admissible test functions", "test_space"),
        ("l(v) = 0", "linear_functional"),
    ],
)
def test_incorrect_math_does_not_satisfy_published_checks(text, check):
    assert not _matches_requirement(text, RUBRIC["feedback_checks"][check])


def test_feedback_check_requires_all_snippets_in_one_alternative():
    check = RUBRIC["feedback_checks"]["positive_weak_form"]
    assert not _matches_requirement("positive sign", check)
    assert not _matches_requirement("a(u,v)", check)
    assert _matches_requirement("a(u,v) must have a positive sign", check)
    assert not _lower_contains("unstability condition", "stability condition")


@pytest.fixture
def release():
    release_root = ROOT.parent / "release-staging/ale-task-fixes-20260913"
    selector = release_root / "validation-queue-20260917/comparison.json"
    if not selector.exists():
        pytest.skip("Selected validation artifacts are not installed")
    row = next(row for row in json.loads(selector.read_text())["rows"] if row["task"] == TASK)
    override = release_root / "post-validation-20260918/data-overrides" / TASK / "base"
    return row, override


def test_staged_feedback_requirements_are_exactly_public(release):
    _, override = release
    assert json.loads((override / "input/released/rubric.json").read_text()) == RUBRIC
    feedback = json.loads((override / "reference/feedback_requirements.json").read_text())
    required = {student: set() for student in feedback}
    with (DATA / "reference/gold_error_tags.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            required[row["student_id"]].update(RUBRIC["feedback_checks_by_tag"][row["error_tag"]])
    for student, checks in feedback.items():
        assert {check["concept"] for check in checks} == required[student]
        for check in checks:
            assert check["any_of"] == RUBRIC["feedback_checks"][check["concept"]]["any_of"]
    summary = json.loads((override / "reference/summary_requirements.json").read_text())
    assert [check["concept"] for check in summary["required_phrases"]] == RUBRIC["summary_checks"]


@pytest.mark.parametrize(
    "label,grades,tags", [("kimi_after", 15 / 25, 9 / 11), ("codex_after_repair", 20 / 25, 10 / 11)]
)
def test_actual_correct_feedback_gets_credit_without_changing_grades(
    release, tmp_path, label, grades, tags
):
    row, override = release
    reference = tmp_path / "reference"
    shutil.copytree(DATA / "reference", reference)
    shutil.copytree(override / "reference", reference, dirs_exist_ok=True)
    run = Path(row[label]["run_json"])
    assert hashlib.sha256(run.read_bytes()).hexdigest() == row[label]["sha256"]
    original = run.parent / "output"
    before_path = (
        ROOT.parent
        / "release-staging/ale-task-fixes-20260913/post-validation-20260918/education-evidence/pde-scorer-before.py"
    )
    assert (
        hashlib.sha256(before_path.read_bytes()).hexdigest()
        == "8b0bc1cff52452b3524afb4ed2b998eae8fd7af5ca29cfb85f8dbd63d6e5b1b9"
    )
    before = runpy.run_path(str(before_path))["score_submission"](
        submission_dir=original, reference_dir=DATA / "reference"
    )
    assert before["score"] == row[label]["score"]
    report = score_submission(submission_dir=original, reference_dir=reference)
    assert report["feedback_score"] == report["summary_score"] == 1
    assert report["grades_score"] == grades
    assert report["tags_score"] == tags
    print(
        json.dumps(
            {
                "pde": label,
                "before": before["score"],
                "after": report["score"],
                "feedback": report["feedback_score"],
                "summary": report["summary_score"],
                "grades": grades,
                "tags": tags,
            }
        )
    )
    corrected = tmp_path / "corrected"
    shutil.copytree(original, corrected)
    shutil.copyfile(reference / "gold_scores.csv", corrected / "grades.csv")
    shutil.copyfile(reference / "gold_error_tags.csv", corrected / "error_tags.csv")
    assert score_submission(submission_dir=corrected, reference_dir=reference)["score"] == 1
    feedback_path = corrected / "per_student_feedback.json"
    feedback = json.loads(feedback_path.read_text())
    feedback["S03"] = "The stencil, bound, and negative weak form are all correct."
    feedback_path.write_text(json.dumps(feedback))
    failed = score_submission(submission_dir=corrected, reference_dir=reference)
    assert failed["feedback_score"] == 0.8
    assert failed["score"] < 1
