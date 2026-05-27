#!/usr/bin/env python
"""Score SuiteCRM merge state snapshots against hidden expected results."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTACT_FIELD_MAP = {
    "first_name": "first_name",
    "last_name": "last_name",
    "email": "email1",
    "phone_work": "phone_work",
    "phone_mobile": "phone_mobile",
    "address_street": "primary_address_street",
    "address_city": "primary_address_city",
    "address_postcode": "primary_address_postalcode",
    "address_country": "primary_address_country",
    "lead_source": "lead_source",
    "email_opt_out": "email_opt_out",
}


@dataclass
class ScoreResult:
    score: float
    passed: bool
    threshold: float
    pass_rate: float
    total_checks: int
    passed_checks: int
    failed_checks: list[dict[str, Any]]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "pass_rate": self.pass_rate,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "reason": self.reason,
        }


def _normalise_text(value: Any) -> str:
    return str("" if value is None else value).strip().lower()


def _normalise_name(first_name: Any, last_name: Any) -> str:
    return f"{_normalise_text(first_name)} {_normalise_text(last_name)}".strip()


def _field_match(got: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return (
            str(got).lower() in {"1", "true"} if expected else str(got).lower() not in {"1", "true"}
        )
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(got) == expected
        except (TypeError, ValueError):
            return False
    return _normalise_text(got) == _normalise_text(expected)


def _build_contact_lookup(
    snapshot_payload: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    contacts = snapshot_payload.get("contacts", [])
    if not isinstance(contacts, list):
        raise ValueError("snapshot payload must contain a list field named 'contacts'")

    contact_lookup: dict[str, dict[str, Any]] = {}
    derived_counts: dict[str, int] = {}
    for entry in contacts:
        if not isinstance(entry, dict):
            continue
        name_key = _normalise_name(entry.get("first_name"), entry.get("last_name"))
        if not name_key:
            continue
        contact_lookup.setdefault(name_key, entry)
        derived_counts[name_key] = derived_counts.get(name_key, 0) + 1

    raw_counts = snapshot_payload.get("count_by_name", {})
    if isinstance(raw_counts, dict):
        count_lookup = {str(key).strip().lower(): int(value) for key, value in raw_counts.items()}
    else:
        count_lookup = {}

    for key, value in derived_counts.items():
        count_lookup.setdefault(key, value)

    return contact_lookup, count_lookup


def score_snapshot_payload(
    *,
    snapshot_payload: dict[str, Any],
    expected_payload: dict[str, Any],
    threshold: float = 0.85,
) -> ScoreResult:
    contact_lookup, count_lookup = _build_contact_lookup(snapshot_payload)

    total_checks = 0
    passed_checks = 0
    failed_checks: list[dict[str, Any]] = []
    hard_gate_failed = False

    for pair_id, expected_entry in expected_payload.items():
        if pair_id.startswith("_"):
            continue
        if not isinstance(expected_entry, dict):
            continue
        expected_fields = expected_entry.get("fields", {})
        if not isinstance(expected_fields, dict):
            continue

        name_key = _normalise_name(
            expected_fields.get("first_name", ""),
            expected_fields.get("last_name", ""),
        )
        contact = contact_lookup.get(name_key)
        count_for_name = count_lookup.get(name_key, 0)

        def record_check(label: str, ok: bool, *, got: Any = None, expected: Any = None) -> None:
            nonlocal total_checks, passed_checks
            total_checks += 1
            if ok:
                passed_checks += 1
                return
            failed_checks.append(
                {
                    "label": label,
                    "got": got,
                    "expected": expected,
                }
            )

        record_check(
            f"{pair_id}.master_exists",
            contact is not None,
            got="found" if contact is not None else "not found",
            expected="found",
        )
        if contact is None:
            hard_gate_failed = True

        if contact is not None:
            for field_name, expected_value in expected_fields.items():
                api_field = CONTACT_FIELD_MAP.get(field_name, field_name)
                record_check(
                    f"{pair_id}.fields.{field_name}",
                    _field_match(contact.get(api_field), expected_value),
                    got=contact.get(api_field),
                    expected=expected_value,
                )

        record_check(
            f"{pair_id}.non_master_deleted",
            count_for_name == 1,
            got=count_for_name,
            expected=1,
        )
        if count_for_name != 1:
            hard_gate_failed = True

    pass_rate = (passed_checks / total_checks) if total_checks else 0.0
    binary_score = 1.0 if pass_rate >= threshold and not hard_gate_failed else 0.0
    reason = "ok"
    if binary_score != 1.0:
        if hard_gate_failed and failed_checks:
            reason = failed_checks[0]["label"]
        elif failed_checks:
            reason = failed_checks[0]["label"]
        else:
            reason = f"pass_rate<{threshold:.2f}"
    return ScoreResult(
        score=binary_score,
        passed=binary_score == 1.0,
        threshold=threshold,
        pass_rate=pass_rate,
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks[:20],
        reason=reason,
    )


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    try:
        result = score_snapshot_payload(
            snapshot_payload=_read_json(args.snapshot),
            expected_payload=_read_json(args.expected),
            threshold=float(args.threshold),
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "score": 0.0,
                    "passed": False,
                    "reason": f"{type(exc).__name__}:{exc}",
                },
                ensure_ascii=False,
            )
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
