"""Reusable conformance checks for deterministic ALE graders.

The helpers in this module are intentionally evaluator-agnostic.  They let task
regression tests describe three properties without changing task scoring logic:

* expected score bands for positive, negative, and malformed candidate fixtures;
* evaluator/infrastructure faults that must raise instead of becoming scores;
* monotonic score relations between named fixtures.

A contract is executed explicitly from tests with :func:`run_conformance`; ALE
runtime execution does not invoke these checks automatically.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real


class ConformanceError(AssertionError):
    """Raised when a grader violates its declared conformance contract."""


@dataclass(frozen=True)
class ScoreExpectation:
    """Allowed score interval, optionally centered on an exact target."""

    minimum: float = 0.0
    maximum: float = 1.0
    target: float | None = None
    atol: float = 1e-9

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("minimum score cannot exceed maximum score")
        if self.atol < 0:
            raise ValueError("score tolerance must be non-negative")
        if self.target is not None and not self.minimum <= self.target <= self.maximum:
            raise ValueError("target score must lie inside the allowed score interval")

    @classmethod
    def exact(cls, target: float, *, atol: float = 1e-9) -> ScoreExpectation:
        return cls(target=target, atol=atol)

    @classmethod
    def at_most(cls, maximum: float) -> ScoreExpectation:
        return cls(maximum=maximum)

    @classmethod
    def at_least(cls, minimum: float) -> ScoreExpectation:
        return cls(minimum=minimum)

    def validate(self, score: float, *, case_name: str) -> None:
        if not self.minimum <= score <= self.maximum:
            raise ConformanceError(
                f"{case_name}: score {score!r} is outside declared interval "
                f"[{self.minimum}, {self.maximum}]"
            )
        if self.target is not None and not math.isclose(
            score, self.target, rel_tol=0.0, abs_tol=self.atol
        ):
            raise ConformanceError(
                f"{case_name}: score {score!r} does not match expected "
                f"{self.target!r} within atol={self.atol}"
            )


@dataclass(frozen=True)
class ScoreCase:
    """A candidate fixture that must return a valid benchmark score."""

    name: str
    evaluate: Callable[[], float]
    expectation: ScoreExpectation = ScoreExpectation()


@dataclass(frozen=True)
class InfrastructureFailureCase:
    """An evaluator fault that must surface as an exception, never as a score."""

    name: str
    evaluate: Callable[[], object]
    exception_types: tuple[type[BaseException], ...] = (RuntimeError,)
    message_substring: str | None = None


@dataclass(frozen=True)
class MonotonicRelation:
    """Require the score of ``higher`` to be no lower than ``lower``."""

    higher: str
    lower: str
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise ValueError("monotonicity tolerance must be non-negative")


@dataclass(frozen=True)
class GraderConformanceContract:
    """Declarative score and failure-behavior contract for one grader."""

    score_cases: tuple[ScoreCase, ...]
    infrastructure_failure_cases: tuple[InfrastructureFailureCase, ...] = ()
    monotonic_relations: tuple[MonotonicRelation, ...] = ()


@dataclass(frozen=True)
class ConformanceReport:
    """Summary returned after all declared checks pass."""

    scores: Mapping[str, float]
    score_cases_checked: int
    infrastructure_failure_cases_checked: int
    monotonic_relations_checked: int


def _normalize_score(raw: object, *, case_name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise ConformanceError(
            f"{case_name}: grader returned {type(raw).__name__}, expected a numeric score"
        )
    score = float(raw)
    if not math.isfinite(score):
        raise ConformanceError(f"{case_name}: grader returned non-finite score {score!r}")
    if not 0.0 <= score <= 1.0:
        raise ConformanceError(f"{case_name}: grader returned score outside [0, 1]: {score!r}")
    return score


def _ensure_unique_names(contract: GraderConformanceContract) -> None:
    names = [case.name for case in contract.score_cases]
    names.extend(case.name for case in contract.infrastructure_failure_cases)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"conformance case names must be unique: {duplicates}")


def run_conformance(contract: GraderConformanceContract) -> ConformanceReport:
    """Execute a grader contract and raise :class:`ConformanceError` on drift."""

    _ensure_unique_names(contract)
    scores: dict[str, float] = {}

    for case in contract.score_cases:
        try:
            raw_score = case.evaluate()
        except Exception as exc:
            raise ConformanceError(
                f"{case.name}: candidate score case raised {type(exc).__name__}: {exc}"
            ) from exc
        score = _normalize_score(raw_score, case_name=case.name)
        case.expectation.validate(score, case_name=case.name)
        scores[case.name] = score

    for case in contract.infrastructure_failure_cases:
        try:
            result = case.evaluate()
        except case.exception_types as exc:
            if case.message_substring is not None and case.message_substring not in str(exc):
                raise ConformanceError(
                    f"{case.name}: infrastructure exception message did not contain "
                    f"{case.message_substring!r}: {exc}"
                ) from exc
        except BaseException as exc:
            expected = ", ".join(exc_type.__name__ for exc_type in case.exception_types)
            raise ConformanceError(
                f"{case.name}: raised {type(exc).__name__}, expected {expected}"
            ) from exc
        else:
            raise ConformanceError(
                f"{case.name}: evaluator fault returned {result!r} instead of raising"
            )

    for relation in contract.monotonic_relations:
        try:
            higher = scores[relation.higher]
            lower = scores[relation.lower]
        except KeyError as exc:
            raise ValueError(
                "monotonic relations may reference only declared score cases"
            ) from exc
        if higher + relation.tolerance < lower:
            raise ConformanceError(
                f"monotonicity violated: {relation.higher}={higher} < "
                f"{relation.lower}={lower}"
            )

    return ConformanceReport(
        scores=scores,
        score_cases_checked=len(contract.score_cases),
        infrastructure_failure_cases_checked=len(contract.infrastructure_failure_cases),
        monotonic_relations_checked=len(contract.monotonic_relations),
    )
