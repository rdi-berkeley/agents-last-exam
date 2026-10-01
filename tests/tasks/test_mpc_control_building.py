from __future__ import annotations

import json

from ale_run.tasks.conformance import (
    GraderConformanceContract,
    InfrastructureFailureCase,
    ScoreCase,
    ScoreExpectation,
    run_conformance,
)
from tasks.engineering.mpc_control_building_v1.main import _parse_verifier_result


def test_mpc_verifier_conformance() -> None:
    failed_candidate = {
        "stdout": json.dumps(
            {
                "score": 0.0,
                "passed": False,
                "reason": "missing required output file",
            }
        ),
        "stderr": "",
        "return_code": 1,
    }
    evaluator_faults = (
        {"stdout": "", "stderr": "uv failed", "return_code": 1},
        {
            "stdout": json.dumps({"score": 0.0, "passed": False}),
            "stderr": "verifier crashed after printing",
            "return_code": 2,
        },
        {
            "stdout": json.dumps({"score": float("nan"), "passed": True}),
            "stderr": "",
            "return_code": 0,
        },
    )

    report = run_conformance(
        GraderConformanceContract(
            score_cases=(
                ScoreCase(
                    "candidate_failure",
                    lambda: _parse_verifier_result(failed_candidate),
                    ScoreExpectation.exact(0.0),
                ),
            ),
            infrastructure_failure_cases=tuple(
                InfrastructureFailureCase(
                    f"evaluator_fault_{index}",
                    lambda result=result: _parse_verifier_result(result),
                    message_substring="MPC verifier failed",
                )
                for index, result in enumerate(evaluator_faults)
            ),
        )
    )

    assert report.scores == {"candidate_failure": 0.0}
    assert report.infrastructure_failure_cases_checked == 3
