from __future__ import annotations

import pytest

from ale_run.tasks.conformance import (
    ConformanceError,
    GraderConformanceContract,
    InfrastructureFailureCase,
    MonotonicRelation,
    ScoreCase,
    ScoreExpectation,
    run_conformance,
)


def test_contract_accepts_scores_faults_and_monotonicity() -> None:
    def evaluator_fault() -> float:
        raise RuntimeError("reference missing")

    report = run_conformance(
        GraderConformanceContract(
            score_cases=(
                ScoreCase("positive", lambda: 1.0, ScoreExpectation.exact(1.0)),
                ScoreCase("negative", lambda: 0.25, ScoreExpectation.at_most(0.5)),
            ),
            infrastructure_failure_cases=(
                InfrastructureFailureCase(
                    "missing_reference",
                    evaluator_fault,
                    message_substring="reference missing",
                ),
            ),
            monotonic_relations=(MonotonicRelation("positive", "negative"),),
        )
    )

    assert report.scores == {"positive": 1.0, "negative": 0.25}
    assert report.score_cases_checked == 2
    assert report.infrastructure_failure_cases_checked == 1
    assert report.monotonic_relations_checked == 1


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_contract_rejects_invalid_scores(score: object) -> None:
    with pytest.raises(ConformanceError):
        run_conformance(
            GraderConformanceContract(score_cases=(ScoreCase("invalid", lambda: score),))
        )


def test_contract_rejects_candidate_exception() -> None:
    def broken_candidate_case() -> float:
        raise ValueError("bad candidate parser")

    with pytest.raises(ConformanceError, match="candidate score case raised"):
        run_conformance(
            GraderConformanceContract(
                score_cases=(ScoreCase("candidate", broken_candidate_case),)
            )
        )


def test_contract_rejects_swallowed_infrastructure_failure() -> None:
    with pytest.raises(ConformanceError, match="instead of raising"):
        run_conformance(
            GraderConformanceContract(
                score_cases=(),
                infrastructure_failure_cases=(
                    InfrastructureFailureCase("missing_reference", lambda: 0.0),
                ),
            )
        )


def test_contract_rejects_wrong_infrastructure_exception() -> None:
    def wrong_failure() -> float:
        raise ValueError("bad reference")

    with pytest.raises(ConformanceError, match="expected RuntimeError"):
        run_conformance(
            GraderConformanceContract(
                score_cases=(),
                infrastructure_failure_cases=(
                    InfrastructureFailureCase("missing_reference", wrong_failure),
                ),
            )
        )


def test_contract_rejects_monotonicity_regression() -> None:
    with pytest.raises(ConformanceError, match="monotonicity violated"):
        run_conformance(
            GraderConformanceContract(
                score_cases=(
                    ScoreCase("minor_error", lambda: 0.3),
                    ScoreCase("major_error", lambda: 0.7),
                ),
                monotonic_relations=(MonotonicRelation("minor_error", "major_error"),),
            )
        )
