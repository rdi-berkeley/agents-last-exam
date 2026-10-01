# Grader conformance checks

ALE deterministic graders can use `ale_run.tasks.conformance` from regression tests to declare a
small set of evaluator invariants without changing runtime scoring behavior.

The framework covers three properties:

1. **Candidate score cases.** Positive, negative, malformed, or boundary fixtures must return finite
   scores in `[0, 1]` and may declare exact values or score ranges.
2. **Evaluator failures.** Broken evaluator-controlled references, crashed verifiers, or other
   infrastructure failures must raise instead of silently becoming candidate scores.
3. **Monotonic relations.** A task may declare relations such as
   `score(correct) >= score(minor_error) >= score(major_error)` when that property is part of the
   task's intended scoring contract.

The framework is intentionally opt-in. It does not impose one scoring policy on all tasks, expose
hidden references, or run during benchmark execution.

## Example

```python
from ale_run.tasks.conformance import (
    GraderConformanceContract,
    InfrastructureFailureCase,
    MonotonicRelation,
    ScoreCase,
    ScoreExpectation,
    run_conformance,
)

report = run_conformance(
    GraderConformanceContract(
        score_cases=(
            ScoreCase("positive", score_positive, ScoreExpectation.exact(1.0)),
            ScoreCase("negative", score_negative, ScoreExpectation.at_most(0.5)),
        ),
        infrastructure_failure_cases=(
            InfrastructureFailureCase(
                "missing_reference",
                score_with_missing_reference,
                message_substring="reference",
            ),
        ),
        monotonic_relations=(MonotonicRelation("positive", "negative"),),
    )
)
```

`run_conformance` raises `ConformanceError` on a contract violation and returns a compact report
when every declared invariant passes.

## Initial migrations

The first two migrations exercise different historical defect classes:

- `mpc_control_building_v1` declares that a valid verifier result with `passed=false` is a candidate
  failure, while malformed output, impossible status combinations, and non-finite verifier scores
  are evaluator failures.
- `pseudotime_de` declares positive, negative, and malformed-candidate fixtures and now treats a
  malformed or empty evaluator reference as infrastructure failure instead of silently returning a
  candidate score of `0.0`.

Additional deterministic tasks can adopt the framework incrementally by adding focused fixtures to
their existing regression tests.
