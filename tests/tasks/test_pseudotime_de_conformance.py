from __future__ import annotations

from ale_run.tasks.conformance import (
    GraderConformanceContract,
    InfrastructureFailureCase,
    MonotonicRelation,
    ScoreCase,
    ScoreExpectation,
    run_conformance,
)
from tasks.life_sciences.pseudotime_de.scripts.score_outputs import score_submission


def _csv(header: str, *values: str) -> bytes:
    return (header + "\n" + "\n".join(values) + "\n").encode()


def test_pseudotime_grader_conformance() -> None:
    reference = _csv("x", "A", "B", "C", "D", "E")
    positive = {"de_genes.csv": _csv("gene", "A", "B", "C", "D", "E")}
    negative = {"de_genes.csv": _csv("gene", "A")}
    malformed_candidate = {"de_genes.csv": b"gene\n"}

    report = run_conformance(
        GraderConformanceContract(
            score_cases=(
                ScoreCase(
                    "positive",
                    lambda: score_submission(positive, reference).score,
                    ScoreExpectation.exact(1.0),
                ),
                ScoreCase(
                    "negative",
                    lambda: score_submission(negative, reference).score,
                    ScoreExpectation.exact(0.0),
                ),
                ScoreCase(
                    "malformed_candidate",
                    lambda: score_submission(malformed_candidate, reference).score,
                    ScoreExpectation.exact(0.0),
                ),
            ),
            infrastructure_failure_cases=(
                InfrastructureFailureCase(
                    "malformed_reference",
                    lambda: score_submission(positive, b"wrong,columns\nA,B\n").score,
                    message_substring="evaluator reference",
                ),
                InfrastructureFailureCase(
                    "empty_reference",
                    lambda: score_submission(positive, b"x\n").score,
                    message_substring="zero genes",
                ),
            ),
            monotonic_relations=(
                MonotonicRelation("positive", "negative"),
                MonotonicRelation("negative", "malformed_candidate"),
            ),
        )
    )

    assert report.scores["positive"] == 1.0
    assert report.infrastructure_failure_cases_checked == 2
