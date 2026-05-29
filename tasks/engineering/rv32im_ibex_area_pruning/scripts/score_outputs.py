"""Scoring helpers for engineering/rv32im_ibex_area_pruning.

The task asks the agent to parse an RV32IM workload, derive an ISA subset and
data-level constraints, and report gate-level synthesis areas. The evaluator
compares the agent's ``output.json`` against a hidden ``reference/output.json``
using five hard gates:

1. all required keys present and well-typed
2. ``instruction_subset`` exact match (lowercase, sorted, set-equal)
3. ``isa_pruned_area_um2`` within +/-1% of the reference
4. ``data_pruned_area_um2`` <= reference value (more aggressive pruning allowed)
5. both ``area_reduction_percent`` fields arithmetically consistent with the
   reported areas using the fixed ``gp_area_um2`` constant

Any failure scores 0.0; otherwise 1.0. Partial credit is intentionally avoided
because the submission defines the benchmark as outcome-focused.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


GP_AREA_UM2: float = 126176.0128
ISA_AREA_TOLERANCE: float = 0.01
ARITH_CONSISTENCY_TOLERANCE_PERCENT: float = 0.01
DATA_AREA_SLACK: float = 1e-6

REQUIRED_KEYS: tuple[str, ...] = (
    "instruction_subset",
    "isa_pruned_area_um2",
    "data_pruned_area_um2",
    "isa_area_reduction_percent",
    "data_area_reduction_percent",
)


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hard_fail(reason: str, details: dict[str, Any] | None = None) -> ScoreResult:
    return ScoreResult(0.0, False, reason, reason, details or {})


def _parse_json(text: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object, got {type(parsed).__name__}")
    return parsed


def _coerce_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number, got {type(value).__name__}")
    return float(value)


def _coerce_instruction_subset(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON list, got {type(value).__name__}")
    items: list[str] = []
    for index, element in enumerate(value):
        if not isinstance(element, str):
            raise ValueError(
                f"{field_name}[{index}] must be a string, got {type(element).__name__}"
            )
        items.append(element)
    return items


def score_output_json(*, output_json: str, reference_json: str) -> ScoreResult:
    try:
        reference = _parse_json(reference_json, "reference output.json")
    except ValueError as exc:
        return _hard_fail("reference_invalid_json", {"error": str(exc)})

    try:
        submission = _parse_json(output_json, "submission output.json")
    except ValueError as exc:
        return _hard_fail("submission_invalid_json", {"error": str(exc)})

    missing = [key for key in REQUIRED_KEYS if key not in submission]
    if missing:
        return _hard_fail("missing_required_keys", {"missing": missing})

    try:
        submission_subset = _coerce_instruction_subset(
            submission["instruction_subset"], field_name="instruction_subset"
        )
        submission_isa_area = _coerce_number(
            submission["isa_pruned_area_um2"], field_name="isa_pruned_area_um2"
        )
        submission_data_area = _coerce_number(
            submission["data_pruned_area_um2"], field_name="data_pruned_area_um2"
        )
        submission_isa_percent = _coerce_number(
            submission["isa_area_reduction_percent"],
            field_name="isa_area_reduction_percent",
        )
        submission_data_percent = _coerce_number(
            submission["data_area_reduction_percent"],
            field_name="data_area_reduction_percent",
        )
    except ValueError as exc:
        return _hard_fail("submission_schema_error", {"error": str(exc)})

    try:
        reference_subset = _coerce_instruction_subset(
            reference["instruction_subset"], field_name="reference instruction_subset"
        )
        reference_isa_area = _coerce_number(
            reference["isa_pruned_area_um2"], field_name="reference isa_pruned_area_um2"
        )
        reference_data_area = _coerce_number(
            reference["data_pruned_area_um2"], field_name="reference data_pruned_area_um2"
        )
    except (KeyError, ValueError) as exc:
        return _hard_fail("reference_schema_error", {"error": str(exc)})

    lowercase_subset = [item.lower() for item in submission_subset]
    if lowercase_subset != submission_subset:
        return _hard_fail(
            "instruction_subset_not_lowercase",
            {"observed": submission_subset, "expected": lowercase_subset},
        )
    sorted_subset = sorted(lowercase_subset)
    if sorted_subset != lowercase_subset:
        return _hard_fail(
            "instruction_subset_not_sorted",
            {"observed": submission_subset, "expected": sorted_subset},
        )
    reference_subset_normalized = sorted(item.lower() for item in reference_subset)
    if lowercase_subset != reference_subset_normalized:
        missing_vs_reference = sorted(set(reference_subset_normalized) - set(lowercase_subset))
        extra_vs_reference = sorted(set(lowercase_subset) - set(reference_subset_normalized))
        return _hard_fail(
            "instruction_subset_mismatch",
            {
                "missing_vs_reference": missing_vs_reference,
                "extra_vs_reference": extra_vs_reference,
            },
        )

    isa_band = abs(reference_isa_area) * ISA_AREA_TOLERANCE
    isa_deviation = abs(submission_isa_area - reference_isa_area)
    if isa_deviation > isa_band:
        return _hard_fail(
            "isa_pruned_area_out_of_band",
            {
                "submission": submission_isa_area,
                "reference": reference_isa_area,
                "abs_deviation": isa_deviation,
                "band": isa_band,
            },
        )

    if submission_data_area > reference_data_area + DATA_AREA_SLACK:
        return _hard_fail(
            "data_pruned_area_exceeds_reference",
            {
                "submission": submission_data_area,
                "reference": reference_data_area,
            },
        )

    expected_isa_percent = (GP_AREA_UM2 - submission_isa_area) / GP_AREA_UM2 * 100.0
    expected_data_percent = (GP_AREA_UM2 - submission_data_area) / GP_AREA_UM2 * 100.0

    isa_percent_delta = abs(submission_isa_percent - expected_isa_percent)
    if isa_percent_delta > ARITH_CONSISTENCY_TOLERANCE_PERCENT:
        return _hard_fail(
            "isa_reduction_percent_inconsistent",
            {
                "submission": submission_isa_percent,
                "expected_from_areas": expected_isa_percent,
                "abs_deviation": isa_percent_delta,
                "tolerance": ARITH_CONSISTENCY_TOLERANCE_PERCENT,
            },
        )

    data_percent_delta = abs(submission_data_percent - expected_data_percent)
    if data_percent_delta > ARITH_CONSISTENCY_TOLERANCE_PERCENT:
        return _hard_fail(
            "data_reduction_percent_inconsistent",
            {
                "submission": submission_data_percent,
                "expected_from_areas": expected_data_percent,
                "abs_deviation": data_percent_delta,
                "tolerance": ARITH_CONSISTENCY_TOLERANCE_PERCENT,
            },
        )

    return ScoreResult(
        score=1.0,
        passed=True,
        reason="ok",
        hard_gate=None,
        details={
            "instruction_subset_len": len(lowercase_subset),
            "isa_pruned_area_um2": submission_isa_area,
            "reference_isa_pruned_area_um2": reference_isa_area,
            "isa_area_abs_deviation": isa_deviation,
            "isa_area_band": isa_band,
            "data_pruned_area_um2": submission_data_area,
            "reference_data_pruned_area_um2": reference_data_area,
            "isa_reduction_percent_delta": isa_percent_delta,
            "data_reduction_percent_delta": data_percent_delta,
        },
    )
