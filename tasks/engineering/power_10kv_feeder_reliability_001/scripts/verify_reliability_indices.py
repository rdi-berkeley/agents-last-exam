#!/usr/bin/env python
"""Validate the reliability-indices JSON against the hidden reference JSON.

Scoring: leaf-level partial credit.  Every leaf comparison (scalar field or
cell inside a table row) counts equally.  score = correct / total.

Table rows are matched order-insensitively by their contents.  The literal
value of the agent's ``section`` identifier is ignored, but the field must be
present.  A one-to-one maximum-weight assignment prevents one row from being
reused to satisfy multiple reference rows.

Scope, missing-length completeness and affine coefficients form a mandatory
scientific gate. They add no scored leaves. Failed gates return zero while
preserving the underlying leaf score in the diagnostic detail.
"""

from __future__ import annotations

import argparse
import json
import math
import hashlib
import sys
from pathlib import Path
from typing import Any


REL_TOL = 0.05
ASAI_ABS_TOL = 1e-4
EXPECTED_INPUT_MD5S = {
    "input/gis.null.xml": "97f26866681529150b0e1c8f8f2b09ad",
    "input/gis.null.svg": "9eb545802972e0d6b931627bc789c2dd",
    "input/params.json": "b2481a8ef0e822403e4b21f8deedb381",
    "input/pyproject.toml": "5046f37c279e5569da6573efa660024e",
    "input/uv.lock": "fabc12e00ffe5c2f2dc9dc7a120b34cd",
}

TABLE_NAMES = frozenset({"fault_rows", "device_fault_rows", "scheduled_rows"})
SCIENTIFIC_FIELDS = frozenset({"data_quality", "missing_exposure_terms"})
QUALITY_FIELDS = frozenset(
    {
        "assessment_scope",
        "line_exposure_complete",
        "missing_length_line_ids",
        "missing_length_count",
        "recorded_length_km_by_type",
        "full_feeder_point_estimate_available",
    }
)
COEFFICIENT_FIELDS = (
    "SAIFI_F_per_km",
    "SAIDI_F_h_per_km",
    "SAIFI_S_per_km",
    "SAIDI_S_h_per_km",
)
TERM_FIELDS = frozenset({"line_id", "line_type", *COEFFICIENT_FIELDS})
SCALAR_FIELDS = frozenset(
    {
        "feeder",
        "N_T",
        "SAIFI_F",
        "SAIDI_F_h",
        "SAIDI_F_min",
        "SAIFI_D",
        "SAIDI_D_h",
        "SAIDI_D_min",
        "SAIFI_S",
        "SAIDI_S_h",
        "SAIDI_S_min",
        "SAIFI",
        "SAIDI_h",
        "SAIDI_min",
        "CAIDI_h",
        "CAIDI_min",
        "ASAI",
    }
)
TABLE_FIELDS = {
    "fault_rows": frozenset(
        {
            "section",
            "name",
            "length_km",
            "lambda_i",
            "perm_users",
            "t_iso_h",
            "r_i_h",
            "lambda_N",
            "lambda_N_r",
        }
    ),
    "device_fault_rows": frozenset(
        {
            "type",
            "section",
            "name",
            "device_count",
            "lambda_per_device",
            "affected_users",
            "t_repair_h",
            "lambda_N",
            "lambda_N_r",
        }
    ),
    "scheduled_rows": frozenset(
        {"section", "name", "length_km", "users", "lambda_N", "lambda_N_r"}
    ),
}


def _load_json(path: str) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)
    return json.loads(file_path.read_text(encoding="utf-8"))


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_text(agent_value: Any, ref_value: Any) -> bool:
    return str(agent_value).strip() == str(ref_value).strip()


def _same_number(agent_value: Any, ref_value: Any, *, field: str) -> bool:
    try:
        agent_num = float(agent_value)
        ref_num = float(ref_value)
    except Exception:
        return False

    if field == "ASAI":
        return math.isclose(agent_num, ref_num, rel_tol=0.0, abs_tol=ASAI_ABS_TOL)

    if ref_num == 0.0:
        return math.isclose(agent_num, ref_num, rel_tol=0.0, abs_tol=1e-9)
    return math.isclose(agent_num, ref_num, rel_tol=REL_TOL, abs_tol=1e-9)


def _check_leaf(agent_value: Any, ref_value: Any, field: str) -> bool:
    if isinstance(ref_value, (int, float)) and not isinstance(ref_value, bool):
        return _same_number(agent_value, ref_value, field=field)
    return _same_text(agent_value, ref_value)


def _section_is_present(row: dict) -> bool:
    return "section" in row and bool(str(row["section"]).strip())


def _row_match_score(agent_row: Any, ref_row: dict) -> int:
    """Count correct leaves for one possible agent/reference row pairing."""
    if not isinstance(agent_row, dict):
        return 0

    correct = 0
    for field, ref_val in ref_row.items():
        if field == "section":
            correct += int(_section_is_present(agent_row))
        elif field in agent_row and _check_leaf(agent_row[field], ref_val, field):
            correct += 1
    return correct


def _max_weight_assignment(weights: list[list[int]]) -> list[tuple[int, int]]:
    """Return maximum-weight one-to-one (row, column) pairs.

    This is the rectangular Hungarian algorithm.  It assigns every element
    on the smaller side and leaves surplus rows on the larger side unmatched.
    """
    if not weights or not weights[0]:
        return []

    original_rows = len(weights)
    original_cols = len(weights[0])
    transposed = original_rows > original_cols
    matrix = [list(row) for row in weights]
    if transposed:
        matrix = [list(row) for row in zip(*matrix)]

    row_count = len(matrix)
    col_count = len(matrix[0])
    max_weight = max(max(row) for row in matrix)
    costs = [[max_weight - value for value in row] for row in matrix]

    u = [0] * (row_count + 1)
    v = [0] * (col_count + 1)
    p = [0] * (col_count + 1)
    way = [0] * (col_count + 1)

    for i in range(1, row_count + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (col_count + 1)
        used = [False] * (col_count + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, col_count + 1):
                if used[j]:
                    continue
                cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(col_count + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = [(p[j] - 1, j - 1) for j in range(1, col_count + 1) if p[j]]
    if transposed:
        return [(col, row) for row, col in pairs]
    return pairs


def _score_table(
    agent_rows: Any,
    ref_rows: list[dict],
    table_name: str,
    issues: list[str],
) -> tuple[int, int]:
    """Return (correct, total) leaf counts for one table."""
    if not isinstance(agent_rows, list):
        issues.append(f"{table_name}: expected list")
        total = sum(len(r) for r in ref_rows)
        return 0, total

    fields_per_row = len(ref_rows[0]) if ref_rows else 0
    weights = [
        [_row_match_score(agent_row, ref_row) for agent_row in agent_rows] for ref_row in ref_rows
    ]
    pairs = _max_weight_assignment(weights)
    matched_ref = {ref_index for ref_index, _ in pairs}
    matched_agent = {agent_index for _, agent_index in pairs}

    correct = 0
    total = sum(len(ref_row) for ref_row in ref_rows)
    total += max(0, len(agent_rows) - len(ref_rows)) * fields_per_row

    for ref_index, agent_index in pairs:
        ref_row = ref_rows[ref_index]
        agent_row = agent_rows[agent_index]
        ref_label = str(ref_row.get("section", ref_index)).strip()
        if not isinstance(agent_row, dict):
            issues.append(f"{table_name}[ref={ref_label}]: matched agent row is not an object")
            continue

        for field, ref_val in ref_row.items():
            if field == "section":
                if _section_is_present(agent_row):
                    correct += 1
                else:
                    issues.append(f"{table_name}[ref={ref_label}].section: missing field")
                continue
            if field not in agent_row:
                issues.append(f"{table_name}[ref={ref_label}].{field}: missing field")
                continue
            if _check_leaf(agent_row[field], ref_val, field):
                correct += 1
            else:
                issues.append(f"{table_name}[ref={ref_label}].{field}: mismatch")

    for ref_index, ref_row in enumerate(ref_rows):
        if ref_index not in matched_ref:
            issues.append(
                f"{table_name}[ref={str(ref_row.get('section', ref_index)).strip()}]: missing row"
            )

    for agent_index, agent_row in enumerate(agent_rows):
        if agent_index not in matched_agent:
            label = agent_index
            if isinstance(agent_row, dict) and _section_is_present(agent_row):
                label = str(agent_row["section"]).strip()
            issues.append(f"{table_name}[agent={label}]: extra row not in reference")

    return correct, total


def _nonnegative_number(value: Any) -> bool:
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )
    except OverflowError:
        return False


def _scientific_contract_issues(agent: dict, ref: dict) -> list[str]:
    """Validate mandatory scope and one coefficient row per unknown line ID."""
    issues = []
    ref_quality = ref.get("data_quality")
    ref_terms = ref.get("missing_exposure_terms")
    if not isinstance(ref_quality, dict) or not isinstance(ref_terms, list):
        raise ValueError("Reference lacks the affine scientific contract")
    expected_ids = {row["line_id"].strip() for row in ref_terms}
    if len(expected_ids) != len(ref_terms):
        raise ValueError("Reference has duplicate affine line IDs")
    quality = agent.get("data_quality")
    if not isinstance(quality, dict):
        issues.append("data_quality: required object")
    else:
        if set(quality) != QUALITY_FIELDS:
            issues.append("data_quality: incorrect field set")
        if not _same_text(quality.get("assessment_scope"), "recorded_line_exposure_subtotal"):
            issues.append("data_quality.assessment_scope: subtotal scope required")
        complete = quality.get("line_exposure_complete")
        if type(complete) is not bool or complete != (not expected_ids):
            issues.append("data_quality.line_exposure_complete: mismatch")
        available = quality.get("full_feeder_point_estimate_available")
        expected_available = not any(
            row[field] > 0 for row in ref_terms for field in COEFFICIENT_FIELDS
        )
        if type(available) is not bool or available != expected_available:
            issues.append("data_quality.full_feeder_point_estimate_available: mismatch")
        identifiers = quality.get("missing_length_line_ids")
        if not isinstance(identifiers, list) or not all(
            isinstance(key, str) and key.strip() for key in identifiers
        ):
            issues.append("data_quality.missing_length_line_ids: required ID list")
        else:
            normalized = [key.strip() for key in identifiers]
            if len(normalized) != len(set(normalized)) or set(normalized) != expected_ids:
                issues.append(
                    "data_quality.missing_length_line_ids: incomplete, extra or duplicate IDs"
                )
        count = quality.get("missing_length_count")
        if not _nonnegative_number(count) or count != len(expected_ids):
            issues.append("data_quality.missing_length_count: mismatch")
        lengths = quality.get("recorded_length_km_by_type")
        if not isinstance(lengths, dict) or set(lengths) != {"overhead", "cable"}:
            issues.append("data_quality.recorded_length_km_by_type: overhead and cable required")
        else:
            for kind, value in lengths.items():
                if not _nonnegative_number(value) or not _same_number(
                    value, ref_quality["recorded_length_km_by_type"][kind], field="length_km"
                ):
                    issues.append(f"data_quality.recorded_length_km_by_type.{kind}: mismatch")
    terms = agent.get("missing_exposure_terms")
    if not isinstance(terms, list):
        issues.append("missing_exposure_terms: required list")
        return issues
    indexed = {}
    for row in terms:
        if not isinstance(row, dict) or set(row) != TERM_FIELDS:
            issues.append("missing_exposure_terms: incorrect row fields")
            continue
        identifier = row.get("line_id")
        if not isinstance(identifier, str) or not identifier.strip():
            issues.append("missing_exposure_terms.line_id: required ID")
            continue
        identifier = identifier.strip()
        if identifier in indexed:
            issues.append("missing_exposure_terms.line_id: duplicate")
        indexed[identifier] = row
    if len(terms) != len(expected_ids) or set(indexed) != expected_ids:
        issues.append("missing_exposure_terms: incomplete or extra line IDs")
    for ref_row in ref_terms:
        row = indexed.get(ref_row["line_id"].strip())
        if row is None:
            continue
        if (
            not isinstance(row["line_type"], str)
            or row["line_type"].strip() not in {"overhead", "cable"}
            or not _same_text(row["line_type"], ref_row["line_type"])
        ):
            issues.append("missing_exposure_terms.line_type: mismatch")
        for field in COEFFICIENT_FIELDS:
            if not _nonnegative_number(row[field]) or not _same_number(
                row[field], ref_row[field], field=field
            ):
                issues.append(f"missing_exposure_terms.{field}: mismatch")
    return issues


def validate_reference(ref: Any) -> None:
    """Reject incomplete or malformed benchmark-controlled reference data."""
    if not isinstance(ref, dict) or set(ref) != SCALAR_FIELDS | TABLE_NAMES | SCIENTIFIC_FIELDS:
        raise ValueError("Reference lacks the complete affine scientific contract")
    if not isinstance(ref["feeder"], str) or not ref["feeder"].strip():
        raise ValueError("Invalid reference feeder")
    for field in SCALAR_FIELDS - {"feeder"}:
        if not _nonnegative_number(ref[field]):
            raise ValueError(f"Invalid reference scalar: {field}")
    if ref["N_T"] <= 0 or int(ref["N_T"]) != ref["N_T"] or not 0 <= ref["ASAI"] <= 1:
        raise ValueError("Invalid reference population or availability")
    for name, fields in TABLE_FIELDS.items():
        rows = ref[name]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Invalid reference table: {name}")
        for row in rows:
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError(f"Invalid reference row fields: {name}")
            for field, value in row.items():
                if field in {"name", "section", "type"}:
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(f"Invalid reference text: {name}.{field}")
                elif not _nonnegative_number(value):
                    raise ValueError(f"Invalid reference number: {name}.{field}")
    if _scientific_contract_issues(ref, ref):
        raise ValueError("Invalid affine reference contract")


def score_submission(agent: dict, ref: dict) -> dict:
    """Return the unchanged leaf score subject to the mandatory scientific gate."""
    validate_reference(ref)
    if not isinstance(agent, dict):
        raise ValueError("top-level JSON must be an object")
    gate_issues = _scientific_contract_issues(agent, ref)
    issues = list(gate_issues)
    correct = 0
    total = 0
    for key, ref_val in ref.items():
        if key in SCIENTIFIC_FIELDS:
            continue
        if key in TABLE_NAMES:
            if not isinstance(ref_val, list):
                continue
            matched, leaves = _score_table(agent.get(key), ref_val, key, issues)
            correct += matched
            total += leaves
        else:
            total += 1
            if key not in agent:
                issues.append(f"{key}: missing")
            elif _check_leaf(agent[key], ref_val, key):
                correct += 1
            else:
                issues.append(f"{key}: mismatch")
    leaf_score = correct / total if total > 0 else 0.0
    gate_passed = not gate_issues
    return {
        "status": "ok",
        "score": round(leaf_score, 6) if gate_passed else 0.0,
        "passed": gate_passed and leaf_score == 1.0,
        "reason": "ok" if not issues else issues[0],
        "issues": issues[:20],
        "scientific_contract": {"passed": gate_passed, "issues": gate_issues[:20]},
        "detail": {"correct": correct, "total": total, "leaf_score": round(leaf_score, 6)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "status": "evaluation_error",
        "score": 0.0,
        "passed": False,
        "reason": "",
        "issues": [],
    }

    try:
        ref = _load_json(args.ref)
        validate_reference(ref)
    except Exception as exc:
        payload["reason"] = f"reference_error:{type(exc).__name__}:{exc}"
        print(json.dumps(payload, ensure_ascii=False))
        return 2

    try:
        agent = _load_json(args.agent)
    except (
        FileNotFoundError,
        IsADirectoryError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
    ) as exc:
        payload.update(
            status="candidate_error", reason=f"invalid_candidate:{type(exc).__name__}:{exc}"
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        payload["reason"] = f"candidate_read_error:{type(exc).__name__}:{exc}"
        print(json.dumps(payload, ensure_ascii=False))
        return 2
    if not isinstance(agent, dict):
        payload.update(
            status="candidate_error", reason="invalid_candidate:top-level JSON must be an object"
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    try:
        issues: list[str] = []

        agent_path = Path(args.agent)
        input_root = agent_path.parent.parent
        for rel_path, expected_md5 in EXPECTED_INPUT_MD5S.items():
            candidate = input_root / rel_path
            if not candidate.exists():
                issues.append(f"missing_input:{rel_path}")
                continue
            if _md5(candidate) != expected_md5:
                issues.append(f"input_md5:{rel_path}")

        payload.update(score_submission(agent, ref))
        if issues:
            payload["reason"] = issues[0]
            payload["issues"] = (issues + payload["issues"])[:20]

    except Exception as exc:
        payload.update(status="evaluation_error", score=0.0, passed=False)
        payload["reason"] = f"unexpected_error:{type(exc).__name__}:{exc}"
        print(json.dumps(payload, ensure_ascii=False))
        return 2

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
