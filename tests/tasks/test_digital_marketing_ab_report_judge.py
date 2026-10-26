from __future__ import annotations

from copy import deepcopy
from functools import partial
import importlib
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


MODULE_NAME = "tasks.business_finance.digital_marketing_ab_test_analysis_1.report_judge"
SAMPLE_BOUNDS = (3100, 3140)
EXPECTED_LIFT = 0.034


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def fail_network(*args, **kwargs):
        pytest.fail("check_claims must not make network or judge API calls")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)


@pytest.fixture
def check_claims():
    module = importlib.import_module(MODULE_NAME)
    return partial(
        module.check_claims,
        sample_bounds=SAMPLE_BOUNDS,
        expected_lift=EXPECTED_LIFT,
        expected_recommendation="ship",
    )


@pytest.fixture
def claims():
    return {
        "sample_size": [
            {
                "number": "3,122",
                "unit": "per_arm",
                "evidence": "Required sample size: 3,122 per arm.",
            }
        ],
        "primary_lift": [
            {
                "number": "3.40",
                "unit": "percentage_points",
                "direction": "increase",
                "evidence": "Primary absolute lift: 3.40 percentage points.",
            }
        ],
        "recommendation": [{"decision": "ship", "evidence": "Ship the treatment; do not hold it."}],
        "unresolved_conflicts": [],
    }


def report_for(claims):
    return "\n\n".join(
        item["evidence"]
        for category in ("sample_size", "primary_lift", "recommendation")
        for item in claims[category]
    )


def test_accepts_complete_grounded_claims_without_mutating_them(check_claims, claims):
    original = deepcopy(claims)

    assert check_claims(report_for(claims), claims) is True
    assert claims == original


def test_sample_bounds_come_from_caller(check_claims, claims):
    claims["sample_size"] = [
        {"number": "4000", "unit": "per_arm", "evidence": "Require 4000 per arm."}
    ]
    report = report_for(claims)

    assert check_claims(report, claims, sample_bounds=(4000, 4000)) is True
    assert check_claims(report, claims) is False


@pytest.mark.parametrize("number", ["3100", "3122", "3140", "3,140"])
def test_per_arm_sample_bounds_are_inclusive(check_claims, claims, number):
    claims["sample_size"] = [
        {"number": number, "unit": "per_arm", "evidence": f"Require {number} per arm."}
    ]

    assert check_claims(report_for(claims), claims) is True


@pytest.mark.parametrize("number", ["3099", "3141", "0", "-3122", "3122.5"])
def test_rejects_wrong_or_nonintegral_sample_sizes(check_claims, claims, number):
    claims["sample_size"] = [
        {"number": number, "unit": "per_arm", "evidence": f"Require {number} per arm."}
    ]

    assert check_claims(report_for(claims), claims) is False


@pytest.mark.parametrize(
    ("number", "unit"),
    [("0.034", "proportion"), ("3.40", "percentage_points"), ("340", "basis_points")],
)
def test_normalizes_explicit_lift_units(check_claims, claims, number, unit):
    claims["primary_lift"] = [
        {
            "number": number,
            "unit": unit,
            "direction": "increase",
            "evidence": f"Primary lift: {number} {unit}.",
        }
    ]

    assert check_claims(report_for(claims), claims) is True


@pytest.mark.parametrize(
    ("number", "unit"),
    [
        ("3.40", "proportion"),
        ("0.034", "percentage_points"),
        ("3.40", "basis_points"),
        ("340", "percentage_points"),
        ("-3.40", "percentage_points"),
        ("0", "proportion"),
        ("17", "percentage_points"),
    ],
)
def test_wrong_lifts_and_unit_scales_fail(check_claims, claims, number, unit):
    claims["primary_lift"] = [
        {
            "number": number,
            "unit": unit,
            "direction": "decrease"
            if number.startswith("-")
            else ("zero" if float(number) == 0 else "increase"),
            "evidence": f"Primary lift: {number} {unit}.",
        }
    ]

    assert check_claims(report_for(claims), claims) is False


@pytest.mark.parametrize(
    ("number", "unit", "expected_lift"),
    [
        ("-0.034", "proportion", -0.034),
        ("-3.40", "percentage_points", -0.034),
        ("-340", "basis_points", -0.034),
        ("0", "percentage_points", 0.0),
        ("2.0", "percentage_points", 0.02),
    ],
)
def test_lift_comparison_uses_supplied_expectation(
    check_claims, claims, number, unit, expected_lift
):
    claims["primary_lift"] = [
        {
            "number": number,
            "unit": unit,
            "direction": "decrease"
            if number.startswith("-")
            else ("zero" if float(number) == 0 else "increase"),
            "evidence": f"Primary lift: {number} {unit}.",
        }
    ]

    assert check_claims(report_for(claims), claims, expected_lift=expected_lift) is True


@pytest.mark.parametrize(
    ("evidence", "direction", "expected_lift", "expected"),
    [
        ("The open rate decreased by 3.40 percentage points.", "decrease", 0.034, False),
        ("The open rate decreased by 3.40 percentage points.", "decrease", -0.034, True),
        ("The open rate increased by 3.40 percentage points.", "increase", 0.034, True),
        ("The open rate changed by 3.40 percentage points.", "unknown", 0.034, False),
    ],
)
def test_lift_direction_is_not_inferred_from_unsigned_number(
    check_claims, claims, evidence, direction, expected_lift, expected
):
    claims["primary_lift"] = [
        {
            "number": "3.40",
            "unit": "percentage_points",
            "direction": direction,
            "evidence": evidence,
        }
    ]
    assert check_claims(report_for(claims), claims, expected_lift=expected_lift) is expected


def test_inconsistent_explicit_sign_and_direction_is_judge_error(check_claims, claims):
    claims["primary_lift"] = [
        {
            "number": "-3.40",
            "unit": "percentage_points",
            "direction": "increase",
            "evidence": "Primary lift: -3.40 percentage points.",
        }
    ]
    with pytest.raises(ValueError, match="direction"):
        check_claims(report_for(claims), claims)


@pytest.mark.parametrize("total", ["6244", "6,244"])
@pytest.mark.parametrize("total_first", [False, True])
def test_accepts_total_exactly_twice_per_arm(check_claims, claims, total, total_first):
    claims["sample_size"].insert(
        0 if total_first else len(claims["sample_size"]),
        {"number": total, "unit": "total", "evidence": f"Required total: {total}."},
    )

    assert check_claims(report_for(claims), claims) is True


@pytest.mark.parametrize("total", ["3122", "6200", "6243", "6245", "6280", "6244.5"])
def test_total_must_match_reported_per_arm_not_just_doubled_bounds(check_claims, claims, total):
    claims["sample_size"].append(
        {"number": total, "unit": "total", "evidence": f"Required total: {total}."}
    )

    assert check_claims(report_for(claims), claims) is False


def test_total_can_match_any_existing_acceptable_per_arm(check_claims, claims):
    claims["sample_size"].extend(
        [
            {"number": "3140", "unit": "per_arm", "evidence": "Round to 3140 per arm."},
            {"number": "6280", "unit": "total", "evidence": "Rounded total: 6280."},
        ]
    )

    assert check_claims(report_for(claims), claims) is True


@pytest.mark.parametrize("number", ["3122", "6244"])
def test_total_alone_does_not_replace_per_arm_claim(check_claims, claims, number):
    claims["sample_size"] = [
        {"number": number, "unit": "total", "evidence": f"Required total: {number}."}
    ]

    assert check_claims(report_for(claims), claims) is False


def test_each_total_must_be_consistent(check_claims, claims):
    claims["sample_size"].extend(
        [
            {"number": "6244", "unit": "total", "evidence": "Required total: 6244."},
            {"number": "6245", "unit": "total", "evidence": "Required total: 6245."},
        ]
    )

    assert check_claims(report_for(claims), claims) is False


@pytest.mark.parametrize("category", ["sample_size", "primary_lift"])
@pytest.mark.parametrize("alongside_valid", [False, True])
def test_unknown_unit_is_valid_schema_but_never_a_verified_claim(
    check_claims, claims, category, alongside_valid
):
    unknown = {**claims[category][0], "unit": "unknown"}
    claims[category] = claims[category] + [unknown] if alongside_valid else [unknown]

    assert check_claims(report_for(claims), claims) is False


@pytest.mark.parametrize("category", ["sample_size", "primary_lift", "recommendation"])
def test_each_category_requires_at_least_one_claim(check_claims, claims, category):
    report = report_for(claims)
    claims[category] = []

    assert check_claims(report, claims) is False


def test_all_empty_arrays_are_a_failed_report_not_a_schema_error(check_claims):
    empty = {
        "sample_size": [],
        "primary_lift": [],
        "recommendation": [],
        "unresolved_conflicts": [],
    }

    assert check_claims("", empty) is False


@pytest.mark.parametrize("category", ["sample_size", "primary_lift", "recommendation"])
@pytest.mark.parametrize("wrong_first", [False, True])
def test_any_conflicting_claim_fails_regardless_of_order(
    check_claims, claims, category, wrong_first
):
    wrong = {
        "sample_size": {
            "number": "9999",
            "unit": "per_arm",
            "evidence": "Required sample size: 9999 per arm.",
        },
        "primary_lift": {
            "number": "4.40",
            "unit": "percentage_points",
            "direction": "increase",
            "evidence": "Primary absolute lift: 4.40 percentage points.",
        },
        "recommendation": {"decision": "hold", "evidence": "Hold the treatment."},
    }[category]
    claims[category].insert(0 if wrong_first else len(claims[category]), wrong)

    assert check_claims(report_for(claims), claims) is False


def test_equivalent_repeated_claims_are_not_conflicts(check_claims, claims):
    claims["sample_size"].append(deepcopy(claims["sample_size"][0]))
    claims["primary_lift"].extend(
        [
            {
                "number": "0.034",
                "unit": "proportion",
                "direction": "increase",
                "evidence": "Primary lift: 0.034.",
            },
            {
                "number": "340",
                "unit": "basis_points",
                "direction": "increase",
                "evidence": "Primary lift: 340 bp.",
            },
        ]
    )
    claims["recommendation"].append(deepcopy(claims["recommendation"][0]))

    assert check_claims(report_for(claims), claims) is True


@pytest.mark.parametrize(
    ("evidence", "decision"),
    [
        ("Do not ship this rollout.", "hold"),
        ("We cannot safely ship.", "hold"),
        ("Hold rather than ship it now.", "hold"),
        ("Do not **ship** the rollout.", "hold"),
        ("Do not wait to ship.", "ship"),
        ("Ship this rollout; do not hold it.", "ship"),
        ("We recommend **ship**, not **hold**.", "ship"),
    ],
)
def test_accepts_extracted_decision_for_negated_recommendation(
    check_claims, claims, evidence, decision
):
    claims["recommendation"] = [{"decision": decision, "evidence": evidence}]
    report = report_for(claims)
    opposite = "hold" if decision == "ship" else "ship"

    assert check_claims(report, claims, expected_recommendation=decision) is True
    assert check_claims(report, claims, expected_recommendation=opposite) is False


@pytest.mark.parametrize("category", ["sample_size", "primary_lift", "recommendation"])
@pytest.mark.parametrize("bad_evidence", ["Invented quotation.", "", "   ", None, 42, True])
def test_invalid_evidence_raises_value_error(check_claims, claims, category, bad_evidence):
    report = report_for(claims)
    claims[category][0]["evidence"] = bad_evidence

    with pytest.raises(ValueError):
        check_claims(report, claims)


@pytest.mark.parametrize(
    "evidence",
    [
        "Required sample size: 3122 per arm.",
        "required sample size: 3,122 per arm.",
        "Required sample size:  3,122 per arm.",
        "Required sample size: 3,122\nper arm.",
        "Required sample size: 3,122 ... per arm.",
    ],
)
def test_evidence_must_be_verbatim_not_normalized_or_paraphrased(check_claims, claims, evidence):
    report = report_for(claims)
    claims["sample_size"][0]["evidence"] = evidence

    with pytest.raises(ValueError):
        check_claims(report, claims)


@pytest.mark.parametrize("category", ["sample_size", "primary_lift"])
def test_number_must_occur_in_its_own_evidence_not_merely_in_report(check_claims, claims, category):
    report = report_for(claims) + "\n\nNo numerical estimate is stated here."
    claims[category][0]["evidence"] = "No numerical estimate is stated here."

    with pytest.raises(ValueError):
        check_claims(report, claims)


@pytest.mark.parametrize("category", ["sample_size", "primary_lift"])
def test_number_lexeme_must_match_evidence_even_when_numerically_equivalent(
    check_claims, claims, category
):
    report = report_for(claims)
    claims[category][0]["number"] = "3122" if category == "sample_size" else "3.400"

    with pytest.raises(ValueError):
        check_claims(report, claims)


@pytest.mark.parametrize("invalid", [None, [], "{}", True, 1])
def test_claims_must_be_a_dict(check_claims, invalid):
    with pytest.raises(ValueError):
        check_claims("Report", invalid)


@pytest.mark.parametrize("category", ["sample_size", "primary_lift", "recommendation"])
def test_missing_top_level_keys_are_schema_errors(check_claims, claims, category):
    report = report_for(claims)
    del claims[category]

    with pytest.raises(ValueError):
        check_claims(report, claims)


def test_extra_top_level_keys_are_schema_errors(check_claims, claims):
    report = report_for(claims)
    claims["explanation"] = []

    with pytest.raises(ValueError):
        check_claims(report, claims)


@pytest.mark.parametrize("category", ["sample_size", "primary_lift", "recommendation"])
@pytest.mark.parametrize("invalid", [None, {}, "[]", (), True, 1])
def test_categories_must_be_arrays(check_claims, claims, category, invalid):
    report = report_for(claims)
    claims[category] = invalid

    with pytest.raises(ValueError):
        check_claims(report, claims)


@pytest.mark.parametrize("category", ["sample_size", "primary_lift", "recommendation"])
@pytest.mark.parametrize("invalid", [None, [], "claim", True, 42])
def test_array_items_must_be_dicts(check_claims, claims, category, invalid):
    report = report_for(claims)
    claims[category] = [invalid]

    with pytest.raises(ValueError):
        check_claims(report, claims)


@pytest.mark.parametrize(
    ("category", "key"),
    [
        ("sample_size", "number"),
        ("sample_size", "unit"),
        ("sample_size", "evidence"),
        ("primary_lift", "number"),
        ("primary_lift", "unit"),
        ("primary_lift", "direction"),
        ("primary_lift", "evidence"),
        ("recommendation", "decision"),
        ("recommendation", "evidence"),
    ],
)
def test_item_keys_are_required(check_claims, claims, category, key):
    report = report_for(claims)
    del claims[category][0][key]

    with pytest.raises(ValueError):
        check_claims(report, claims)


@pytest.mark.parametrize("category", ["sample_size", "primary_lift", "recommendation"])
def test_extra_item_keys_are_schema_errors(check_claims, claims, category):
    report = report_for(claims)
    claims[category][0]["confidence"] = 1.0

    with pytest.raises(ValueError):
        check_claims(report, claims)


@pytest.mark.parametrize(
    ("category", "key", "invalid"),
    [
        ("sample_size", "unit", "proportion"),
        ("sample_size", "unit", "PER_ARM"),
        ("sample_size", "unit", None),
        ("sample_size", "unit", True),
        ("sample_size", "unit", []),
        ("primary_lift", "unit", "per_arm"),
        ("primary_lift", "unit", "percentile"),
        ("primary_lift", "unit", "percent"),
        ("primary_lift", "unit", None),
        ("primary_lift", "unit", {}),
        ("primary_lift", "direction", None),
        ("primary_lift", "direction", "higher"),
        ("primary_lift", "direction", []),
        ("recommendation", "decision", "SHIP"),
        ("recommendation", "decision", "unknown"),
        ("recommendation", "decision", "do not ship"),
        ("recommendation", "decision", None),
        ("recommendation", "decision", False),
        ("recommendation", "decision", []),
    ],
)
def test_enums_are_strict_and_category_specific(check_claims, claims, category, key, invalid):
    report = report_for(claims)
    claims[category][0][key] = invalid

    with pytest.raises(ValueError):
        check_claims(report, claims)


@pytest.mark.parametrize("category", ["sample_size", "primary_lift"])
@pytest.mark.parametrize("invalid", [True, False, 3122, 3.4, None, [], {}])
def test_number_must_be_a_string_including_when_value_is_boolean(
    check_claims, claims, category, invalid
):
    report = report_for(claims)
    claims[category][0]["number"] = invalid

    with pytest.raises(ValueError):
        check_claims(report, claims)


@pytest.mark.parametrize("category", ["sample_size", "primary_lift"])
@pytest.mark.parametrize(
    "number", ["", " ", "True", "False", "not-a-number", "NaN", "nan", "inf", "-Infinity", "1e9999"]
)
def test_unparseable_and_nonfinite_numeric_strings_raise(check_claims, claims, category, number):
    claims[category][0]["number"] = number
    claims[category][0]["evidence"] = f"Reported value: [{number}]."

    with pytest.raises(ValueError):
        check_claims(report_for(claims), claims)


@pytest.mark.parametrize("bad_first", [False, True])
def test_schema_validation_does_not_short_circuit_on_wrong_claim(check_claims, claims, bad_first):
    claims["sample_size"][0].update(number="9999", evidence="Require 9999 per arm.")
    invalid = {
        "number": "NaN",
        "unit": "proportion",
        "direction": "increase",
        "evidence": "Primary lift: NaN.",
    }
    claims["primary_lift"].insert(0 if bad_first else len(claims["primary_lift"]), invalid)

    with pytest.raises(ValueError):
        check_claims(report_for(claims), claims)


def test_invalid_schema_is_not_hidden_by_missing_claims(check_claims, claims):
    claims["sample_size"] = []
    claims["recommendation"][0]["evidence"] = "Fabricated evidence."

    with pytest.raises(ValueError):
        check_claims("Unrelated report.", claims)


def test_report_conflict_is_rejected_with_exact_evidence(check_claims, claims):
    evidence = "The rate is higher than control, but did not increase."
    report = report_for(claims) + "\n" + evidence
    claims["unresolved_conflicts"] = [{"evidence": evidence}]
    assert check_claims(report, claims) is False
    claims["unresolved_conflicts"] = [{"evidence": "Invented contradiction."}]
    with pytest.raises(ValueError, match="evidence"):
        check_claims(report, claims)


@pytest.mark.asyncio
async def test_judge_extracts_claims_and_retains_receipt(monkeypatch, claims):
    import json

    module = importlib.import_module(MODULE_NAME)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=json.dumps(claims), refusal=None),
            )
        ],
        model="test-model",
        id="response-id",
        _request_id="request-id",
        usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 12}),
    )
    call = AsyncMock(return_value=response)
    monkeypatch.setattr(module, "llm_judge_completion", call)
    monkeypatch.setattr(module, "resolve_llm_judge_model", lambda: "test-model")
    receipt = await module.judge_report(
        report_for(claims),
        sample_bounds=SAMPLE_BOUNDS,
        expected_lift=EXPECTED_LIFT,
        expected_recommendation="ship",
    )
    assert receipt["passed"] is True
    assert receipt["model"] == "test-model"
    assert receipt["request_id"] == "request-id"
    assert receipt["usage"] == {"total_tokens": 12}
    assert receipt["claims"] == claims
    call.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("secret-key"), TimeoutError("timed out")])
async def test_judge_failure_is_sanitized_and_not_a_score(monkeypatch, failure):
    from tasks.utils.evaluation import JudgeInfrastructureError
    from ale_run.tasks.driver import _is_transient_cua_error

    module = importlib.import_module(MODULE_NAME)
    monkeypatch.setattr(module, "llm_judge_completion", AsyncMock(side_effect=failure))
    monkeypatch.setattr(module, "resolve_llm_judge_model", lambda: "test-model")
    with pytest.raises(JudgeInfrastructureError) as caught:
        await module.judge_report(
            "report",
            sample_bounds=SAMPLE_BOUNDS,
            expected_lift=EXPECTED_LIFT,
            expected_recommendation="ship",
        )
    assert "secret-key" not in str(caught.value)
    assert not _is_transient_cua_error(caught.value)


@pytest.mark.asyncio
async def test_bad_judge_json_is_infrastructure_failure(monkeypatch):
    from tasks.utils.evaluation import JudgeInfrastructureError

    module = importlib.import_module(MODULE_NAME)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="invalid", refusal=None)
            )
        ]
    )
    monkeypatch.setattr(module, "llm_judge_completion", AsyncMock(return_value=response))
    monkeypatch.setattr(module, "resolve_llm_judge_model", lambda: "test-model")
    with pytest.raises(JudgeInfrastructureError):
        await module.judge_report(
            "report",
            sample_bounds=SAMPLE_BOUNDS,
            expected_lift=EXPECTED_LIFT,
            expected_recommendation="ship",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length", message=SimpleNamespace(content="{}", refusal=None)
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=None, refusal="declined")
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=" ", refusal=None)
                )
            ]
        ),
    ],
)
async def test_incomplete_or_refused_judge_response_is_not_graded(monkeypatch, response):
    from tasks.utils.evaluation import JudgeInfrastructureError

    module = importlib.import_module(MODULE_NAME)
    monkeypatch.setattr(module, "llm_judge_completion", AsyncMock(return_value=response))
    monkeypatch.setattr(module, "resolve_llm_judge_model", lambda: "test-model")
    with pytest.raises(JudgeInfrastructureError):
        await module.judge_report(
            "report",
            sample_bounds=SAMPLE_BOUNDS,
            expected_lift=EXPECTED_LIFT,
            expected_recommendation="ship",
        )


@pytest.mark.asyncio
async def test_oversized_report_fails_without_judge_call(monkeypatch):
    from tasks.utils.evaluation import JudgeInfrastructureError

    module = importlib.import_module(MODULE_NAME)
    call = AsyncMock()
    monkeypatch.setattr(module, "llm_judge_completion", call)
    with pytest.raises(JudgeInfrastructureError):
        await module.judge_report(
            "x" * (module.MAX_REPORT_CHARS + 1),
            sample_bounds=SAMPLE_BOUNDS,
            expected_lift=EXPECTED_LIFT,
            expected_recommendation="ship",
        )
    call.assert_not_awaited()
