#!/usr/bin/env python
"""Validate the reliability-indices JSON against the hidden reference JSON.

Scoring: leaf-level partial credit.  Every leaf comparison (scalar field or
cell inside a table row) counts equally.  score = correct / total.

Table rows are matched order-insensitively by their contents.  The literal
value of the agent's ``section`` identifier is ignored, but the field must be
present.  A one-to-one maximum-weight assignment prevents one row from being
reused to satisfy multiple reference rows.
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
        [_row_match_score(agent_row, ref_row) for agent_row in agent_rows]
        for ref_row in ref_rows
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "score": 0.0,
        "passed": False,
        "reason": "",
        "issues": [],
    }

    try:
        agent = _load_json(args.agent)
        ref = _load_json(args.ref)

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

        if not isinstance(agent, dict) or not isinstance(ref, dict):
            raise ValueError("top-level JSON must be an object")

        correct = 0
        total = 0

        for key, ref_val in ref.items():
            if key in TABLE_NAMES:
                if not isinstance(ref_val, list):
                    continue
                c, t = _score_table(
                    agent.get(key),
                    ref_val,
                    key,
                    issues,
                )
                correct += c
                total += t
            else:
                total += 1
                if key not in agent:
                    issues.append(f"{key}: missing")
                    continue
                if _check_leaf(agent[key], ref_val, key):
                    correct += 1
                else:
                    issues.append(f"{key}: mismatch")

        score = correct / total if total > 0 else 0.0
        payload["score"] = round(score, 6)
        payload["passed"] = score == 1.0
        payload["reason"] = "ok" if not issues else issues[0]
        payload["issues"] = issues[:20]
        payload["detail"] = {"correct": correct, "total": total}

    except FileNotFoundError as exc:
        payload["reason"] = f"missing_file:{exc}"
    except json.JSONDecodeError as exc:
        payload["reason"] = f"invalid_json:{exc}"
    except Exception as exc:
        payload["reason"] = f"unexpected_error:{type(exc).__name__}:{exc}"

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
