#!/usr/bin/env python
"""Validate the reliability-indices JSON against the hidden reference JSON.

Scoring: leaf-level partial credit.  Every leaf comparison (scalar field or
cell inside a table row) counts equally.  score = correct / total.

Table rows are matched order-insensitively using natural key fields so that
a correct answer in a different sort order is not penalised.
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

TABLE_KEYS: dict[str, list[str]] = {
    "fault_rows": ["section"],
    "device_fault_rows": ["section", "type"],
    "scheduled_rows": ["section"],
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


def _row_key(row: dict, key_fields: list[str]) -> tuple:
    return tuple(str(row.get(k, "")).strip() for k in key_fields)


def _score_table(
    agent_rows: Any,
    ref_rows: list[dict],
    key_fields: list[str],
    table_name: str,
    issues: list[str],
) -> tuple[int, int]:
    """Return (correct, total) leaf counts for one table."""
    if not isinstance(agent_rows, list):
        issues.append(f"{table_name}: expected list")
        total = sum(len(r) for r in ref_rows)
        return 0, total

    fields_per_row = len(ref_rows[0]) if ref_rows else 0
    ref_by_key: dict[tuple, dict] = {}
    for r in ref_rows:
        ref_by_key[_row_key(r, key_fields)] = r

    agent_by_key: dict[tuple, dict] = {}
    for r in agent_rows:
        if isinstance(r, dict):
            agent_by_key[_row_key(r, key_fields)] = r

    all_keys = set(ref_by_key.keys()) | set(agent_by_key.keys())
    correct = 0
    total = 0

    for key in all_keys:
        ref_row = ref_by_key.get(key)
        agent_row = agent_by_key.get(key)

        if ref_row is None:
            total += fields_per_row
            issues.append(f"{table_name}[{key}]: extra row not in reference")
            continue

        if agent_row is None:
            total += len(ref_row)
            issues.append(f"{table_name}[{key}]: missing row")
            continue

        for field, ref_val in ref_row.items():
            total += 1
            if field not in agent_row:
                issues.append(f"{table_name}[{key}].{field}: missing field")
                continue
            if _check_leaf(agent_row[field], ref_val, field):
                correct += 1
            else:
                issues.append(f"{table_name}[{key}].{field}: mismatch")

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
            if key in TABLE_KEYS:
                if not isinstance(ref_val, list):
                    continue
                c, t = _score_table(
                    agent.get(key),
                    ref_val,
                    TABLE_KEYS[key],
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
