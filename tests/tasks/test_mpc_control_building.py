from __future__ import annotations

import json

import pytest

from tasks.engineering.mpc_control_building_v1.main import _parse_verifier_result


def test_failed_verifier_json_is_scored_as_candidate_failure() -> None:
    result = {
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

    assert _parse_verifier_result(result) == 0.0


@pytest.mark.parametrize(
    "result",
    [
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
    ],
)
def test_evaluator_failures_are_not_scored_as_candidate_failures(result: dict) -> None:
    with pytest.raises(RuntimeError, match="MPC verifier failed"):
        _parse_verifier_result(result)
