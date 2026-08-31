"""Grader tests for education_info/conference_reviewer_assignment (v3).

Runs against the generated task data in task-data/ (pos fixture must score
1.0, neg fixture 0.0) plus synthetic violation, audit, and partial-credit
cases covering the entity-resolution traps.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tasks.education_info.conference_reviewer_assignment.scripts.score_outputs import (
    affinity,
    build_conflict_codes,
    parse_institutions,
    parse_reviewers,
    parse_submissions,
    score_submission,
)

DATA = (
    Path(__file__).resolve().parents[2]
    / "task-data" / "education_info" / "conference_reviewer_assignment" / "base"
)

pytestmark = pytest.mark.skipif(
    not (DATA / "input" / "submissions.csv").exists(),
    reason="task data not generated (run the generator first)",
)


@pytest.fixture(scope="module")
def env():
    inputs = {
        name: (DATA / "input" / name).read_text()
        for name in ("submissions.csv", "reviewers.csv", "coauthorships.csv",
                     "declared_conflicts.csv", "institutions.csv")
    }
    params = json.loads((DATA / "reference" / "scoring.json").read_text())
    return inputs, params


@pytest.fixture(scope="module")
def world(env):
    inputs, params = env
    alias = parse_institutions(inputs["institutions.csv"])
    reviewers = parse_reviewers(inputs["reviewers.csv"], alias)
    submissions = parse_submissions(inputs["submissions.csv"], alias)
    truth = build_conflict_codes(
        reviewers, submissions, inputs["coauthorships.csv"],
        inputs["declared_conflicts.csv"], params["coi_year_cutoff"],
    )
    return reviewers, submissions, truth


def run(assignment: str, env, audit: str | None = None) -> dict:
    inputs, params = env
    return score_submission(
        assignment_csv=assignment,
        submissions_csv=inputs["submissions.csv"],
        reviewers_csv=inputs["reviewers.csv"],
        coauthorships_csv=inputs["coauthorships.csv"],
        declared_conflicts_csv=inputs["declared_conflicts.csv"],
        institutions_csv=inputs["institutions.csv"],
        scoring_params=params,
        audit_csv=audit,
    )


def pos_assignment() -> str:
    return (DATA / "output_test_pos" / "assignment.csv").read_text()


def pos_audit() -> str:
    return (DATA / "output_test_pos" / "conflict_audit.csv").read_text()


def test_pos_fixture_scores_full(env):
    report = run(pos_assignment(), env, audit=pos_audit())
    assert report["violations"] == []
    assert report["audit_f1"] == 1.0
    assert report["score"] == 1.0
    assert report["total_affinity"] == env[1]["optimal_affinity"]


def test_optimal_without_audit_capped(env):
    report = run(pos_assignment(), env, audit=None)
    assert report["violations"] == []
    assert report["score"] == pytest.approx(0.25 + 0.45)


def test_neg_fixture_scores_zero(env):
    report = run((DATA / "output_test_neg" / "assignment.csv").read_text(), env)
    assert report["score"] == 0.0
    assert any("conflict of interest" in v for v in report["violations"])


def test_empty_and_garbage_score_zero(env):
    assert run("", env)["score"] == 0.0
    assert run("hello\nworld", env)["violations"] == ["bad_header"]


def test_reference_assignment_is_feasible_and_optimal(env):
    report = run((DATA / "reference" / "optimal_assignment.csv").read_text(), env,
                 audit=(DATA / "reference" / "conflict_audit.csv").read_text())
    assert report["violations"] == []
    assert report["quality"] == 1.0
    assert report["audit_f1"] == 1.0


def test_withdrawn_paper_assignment_zero(env, world):
    _, submissions, _ = world
    withdrawn = next(pid for pid, p in submissions.items() if not p["active"])
    text = pos_assignment() + f"{withdrawn},R01\n"
    report = run(text, env, audit=pos_audit())
    assert report["score"] == 0.0
    assert any("withdrawn" in v for v in report["violations"])


def test_unavailable_reviewer_zero(env, world):
    reviewers, submissions, truth = world
    bad = next(rid for rid, r in reviewers.items() if not r["available"])
    lines = pos_assignment().splitlines()
    pid, old = lines[1].split(",")
    swapped = "\n".join([lines[0], f"{pid},{bad}"] + lines[2:]) + "\n"
    report = run(swapped, env)
    assert report["score"] == 0.0
    assert any("unavailable" in v for v in report["violations"])


def test_missing_paper_scores_zero(env, world):
    _, submissions, _ = world
    active = next(pid for pid, p in submissions.items() if p["active"])
    lines = pos_assignment().splitlines()
    truncated = "\n".join(line for line in lines
                          if not line.startswith(active + ",")) + "\n"
    report = run(truncated, env)
    assert report["score"] == 0.0
    assert any(active in v for v in report["violations"])


def test_duplicate_pair_scores_zero(env):
    text = pos_assignment()
    report = run(text + text.splitlines()[1] + "\n", env)
    assert report["score"] == 0.0
    assert any(v.startswith("duplicate pair") for v in report["violations"])


def test_naive_string_matching_audit_is_penalized(env, world):
    """An audit built with exact-string entity logic (no alias resolution, no
    name-format handling, declared table copied verbatim) must land clearly
    below a perfect audit."""
    inputs, params = env
    reviewers, submissions, truth = world
    naive_rows = ["reviewer_id,paper_id,reason_code"]
    raw_reviewers = {}
    for line in inputs["reviewers.csv"].splitlines()[1:]:
        cells = next(iter([line.split(",")]))
        raw_reviewers[cells[0]] = cells
    for line in inputs["declared_conflicts.csv"].splitlines()[1:]:
        rid, pid = line.split(",")
        naive_rows.append(f"{rid},{pid},DECLARED")
    report_perfect = run(pos_assignment(), env, audit=pos_audit())
    report_naive = run(pos_assignment(), env, audit="\n".join(naive_rows) + "\n")
    assert report_naive["violations"] == []
    assert report_naive["audit_f1"] < 0.5
    assert report_naive["score"] < report_perfect["score"] - 0.15


def test_audit_wrong_codes_penalized(env, world):
    _, _, truth = world
    wrong = ["reviewer_id,paper_id,reason_code"]
    for (rid, pid), codes in truth.items():
        wrong.append(f"{rid},{pid},DECLARED")
    report = run(pos_assignment(), env, audit="\n".join(wrong) + "\n")
    assert 0.0 < report["audit_f1"] < 0.6


def test_feasible_suboptimal_gets_partial_credit(env, world):
    """Swap first-slot reviewers of two papers where the swap stays feasible
    but strictly lowers affinity: score must land in (base, 1.0)."""
    reviewers, submissions, truth = world
    lines = pos_assignment().splitlines()
    pairs = [tuple(line.split(",")) for line in lines[1:]]
    by_paper: dict[str, list[str]] = {}
    for pid, rid in pairs:
        by_paper.setdefault(pid, []).append(rid)

    papers = sorted(by_paper)
    for i, pa in enumerate(papers):
        for pb in papers[i + 1:]:
            ra = by_paper[pa][0]
            rb = by_paper[pb][0]
            if ra == rb or rb in by_paper[pa] or ra in by_paper[pb]:
                continue
            if (ra, pb) in truth or (rb, pa) in truth:
                continue
            new_a = [rb] + by_paper[pa][1:]
            new_b = [ra] + by_paper[pb][1:]
            if not any(reviewers[r]["senior"] for r in new_a):
                continue
            if not any(reviewers[r]["senior"] for r in new_b):
                continue
            if len({reviewers[r]["canon_inst"] for r in new_a}) != len(new_a):
                continue
            if len({reviewers[r]["canon_inst"] for r in new_b}) != len(new_b):
                continue
            old = (affinity(reviewers[ra], submissions[pa])
                   + affinity(reviewers[rb], submissions[pb]))
            new = (affinity(reviewers[rb], submissions[pa])
                   + affinity(reviewers[ra], submissions[pb]))
            if new >= old:
                continue
            swapped = dict(by_paper)
            swapped[pa] = new_a
            swapped[pb] = new_b
            text = "paper_id,reviewer_id\n" + "\n".join(
                f"{pid},{rid}" for pid in papers for rid in swapped[pid]) + "\n"
            report = run(text, env, audit=pos_audit())
            assert report["violations"] == []
            assert 0.25 <= report["score"] < 1.0
            return
    pytest.skip("no feasible degrading swap found")
