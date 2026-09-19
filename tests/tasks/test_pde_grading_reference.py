from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tasks.education_info.homework_grading_numerical_pdes_instance_02.scripts.score_outputs import (
    score_submission,
)


DATA = (
    Path(__file__).resolve().parents[2]
    / "task-data-hf/extracted/education_info/homework_grading_numerical_pdes_instance_02/base"
)
EXPECTED_GRADES = [
    ["S01", 2, 3, 2, 3, 10],
    ["S02", 2, 1, 2, 2.25, 7.25],
    ["S03", 1, 1, 0, 2, 4],
    ["S04", 2, 3, 2, 1.5, 8.5],
    ["S05", 2, 2, 2, 3, 9],
]
EXPECTED_TAGS = {
    "S02": ["missing_function_space", "stability_factor_of_two_error", "wrong_dt_max"],
    "S03": [
        "stability_factor_of_two_error",
        "wrong_dt_max",
        "wrong_discrete_laplacian",
        "wrong_weak_form_sign",
        "wrong_bilinear_form",
    ],
    "S04": ["missing_function_space", "missing_linear_functional"],
    "S05": ["wrong_dt_max"],
}


@pytest.fixture
def submission(tmp_path):
    if not DATA.exists():
        pytest.skip("PDE release data is not installed")
    with (tmp_path / "grades.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["student_id", "problem_1a", "problem_1b", "problem_2a", "problem_2b", "total_score"]
        )
        writer.writerows(EXPECTED_GRADES)
    with (tmp_path / "error_tags.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["student_id", "error_tag"])
        writer.writerows((student, tag) for student, tags in EXPECTED_TAGS.items() for tag in tags)
    feedback = {
        "S01": "All four parts are correct.",
        "S02": "Restore the factor of two: dt_max = 0.01. Include H_0^1 as the test space.",
        "S03": "Use -2*u_i^n in the stencil and the factor of two in the stability bound: dt_max = 0.01. Integration by parts gives positive integral u'v'; fix the sign in a(u,v). Your linear functional and function space are correct.",
        "S04": "Give the linear functional and specify H_0^1 as the test space.",
        "S05": "The symbolic bound is correct; arithmetic gives dt_max = 0.01.",
    }
    (tmp_path / "per_student_feedback.json").write_text(json.dumps(feedback))
    (tmp_path / "grader_manifest.json").write_text(
        json.dumps({"python_version": "3.12", "platform": "Linux", "rubric_version": "1"})
    )
    (tmp_path / "common_mistakes_summary.md").write_text(
        "Check the stability condition and numerical substitution. Preserve the positive sign structure in the weak form, and specify H_0^1 and the linear functional."
    )
    return tmp_path


def test_corrected_mathematical_grading_gets_full_credit(submission):
    result = score_submission(submission_dir=submission, reference_dir=DATA / "reference")
    assert result["score"] == 1, result


def test_old_wrong_sign_full_credit_is_rejected(submission):
    path = submission / "grades.csv"
    path.write_text(path.read_text().replace("S03,1,1,0,2,4", "S03,1,1,2,2,6"))
    result = score_submission(submission_dir=submission, reference_dir=DATA / "reference")
    assert result["grades_score"] == pytest.approx(23 / 25)
    assert result["score"] < 1


def test_correct_weak_form_and_wrong_submission_remain_distinct(submission):
    released = DATA / "input/released"
    student = (released / "submissions/S03.md").read_text()
    solution = (released / "solution_key.md").read_text()
    assert "- integral_0^1 u'(x) v'(x)" in student
    assert "- integral_0^1 u'(x) v'(x)" not in solution
    assert "integral_0^1 u'(x) v'(x) dx = integral_0^1 f(x) v(x) dx" in solution
    assert 0.1**2 / (2 * 0.5) == pytest.approx(0.01)


def test_weak_form_sign_by_exact_polynomial_integration():
    from fractions import Fraction

    derivative_product_integral = 1 - Fraction(4, 2) + Fraction(4, 3)
    load_integral = Fraction(2, 2) - Fraction(2, 3)
    assert derivative_product_integral == load_integral == Fraction(1, 3)
    assert -derivative_product_integral != load_integral
