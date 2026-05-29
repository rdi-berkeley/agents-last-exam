"""Scoring utilities for peer-review response drafting benchmarks."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any


OUTPUT_FILES = (
    "response_matrix.csv",
    "follow_up_items.csv",
    "response_letter.md",
    "response_manifest.json",
)


@dataclass
class _CheckResult:
    ok: bool
    details: dict[str, Any]


def _read_text_payload(payload: bytes | bytearray | str | Path | Mapping[str, Any]) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return payload.decode("utf-8")
    if isinstance(payload, str):
        if payload.strip().startswith("{") or payload.strip().startswith("["):
            return payload
        if "\n" in payload:
            return payload
        if Path(payload).exists():
            return Path(payload).read_text(encoding="utf-8")
        return payload
    if isinstance(payload, Mapping):
        return json.dumps(payload, sort_keys=True)
    return str(payload)


def _read_json_payload(payload: bytes | bytearray | str | Path) -> dict[str, Any]:
    text = _read_text_payload(payload)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON payload: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON payload must be an object")
    return parsed


def _read_csv_rows(payload: bytes | bytearray | str | Path) -> list[dict[str, str]]:
    text = _read_text_payload(payload)
    f = StringIO(text)
    rows = list(csv.DictReader(f))
    if not rows and not text.strip():
        return []
    if rows and rows[0]:
        # Keep header-driven parsing; key/value parity is evaluated later.
        return [{k: v for k, v in row.items()} for row in rows]
    return []


def _extract_headings(markdown_text: str) -> list[str]:
    headings = []
    for line in markdown_text.splitlines():
        match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if match:
            headings.append(match.group(1).strip())
    return headings


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _validate_required_headings(text: str, expected_headings: list[str]) -> _CheckResult:
    actual = {name.strip() for name in _extract_headings(text)}
    missing = [name for name in expected_headings if name not in actual]
    return _CheckResult(
        ok=not missing,
        details={"present": sorted(actual), "missing": missing},
    )


def _normalize_keyed_rows(
    rows: list[dict[str, str]], key_field: str
) -> dict[str, dict[str, str]]:
    if not rows and not rows:
        return {}
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        if key_field not in row:
            raise ValueError(f"Row missing required key '{key_field}': {row}")
        key = row[key_field]
        if key is None:
            raise ValueError(f"Row contains empty key for '{key_field}': {row}")
        if key in result:
            raise ValueError(f"Duplicate key '{key}' in rows for '{key_field}'")
        result[key] = row
    return result


def _compare_csv_exact(
    output_payload: bytes | bytearray | str | Path,
    reference_payload: bytes | bytearray | str | Path,
    key_field: str,
) -> _CheckResult:
    output_rows = _read_csv_rows(output_payload)
    reference_rows = _read_csv_rows(reference_payload)

    if len(output_rows) != len(reference_rows):
        return _CheckResult(
            ok=False,
            details={
                "reason": "row_count_mismatch",
                "output_rows": len(output_rows),
                "reference_rows": len(reference_rows),
            },
        )

    output_map = _normalize_keyed_rows(output_rows, key_field)
    reference_map = _normalize_keyed_rows(reference_rows, key_field)
    output_keys = set(output_map)
    reference_keys = set(reference_map)
    if output_keys != reference_keys:
        return _CheckResult(
            ok=False,
            details={
                "reason": "key_mismatch",
                "missing_in_output": sorted(reference_keys - output_keys),
                "extra_in_output": sorted(output_keys - reference_keys),
            },
        )

    mismatches = []
    for key in sorted(reference_keys):
        if output_map[key] != reference_map[key]:
            mismatches.append(
                {
                    "key": key,
                    "output": output_map[key],
                    "reference": reference_map[key],
                }
            )
    if mismatches:
        return _CheckResult(ok=False, details={"reason": "row_value_mismatch", "items": mismatches})

    return _CheckResult(ok=True, details={"n_rows": len(output_rows)})


def _validate_manifest(
    output_payload: bytes | bytearray | str | Path,
    reference_payload: bytes | bytearray | str | Path,
) -> _CheckResult:
    output_manifest = _read_json_payload(output_payload)
    reference_manifest = _read_json_payload(reference_payload)
    if output_manifest != reference_manifest:
        return _CheckResult(
            ok=False,
            details={"reason": "manifest_mismatch"},
        )
    return _CheckResult(ok=True, details={"manifest_keys": sorted(output_manifest.keys())})


def _validate_letter(
    output_payload: bytes | bytearray | str | Path,
    required_headings: list[str],
    required_phrases: list[str],
) -> _CheckResult:
    text = _read_text_payload(output_payload)
    heading_check = _validate_required_headings(text, required_headings)
    if not heading_check.ok:
        return _CheckResult(ok=False, details={"reason": "missing_headings", **heading_check.details})

    norm_text = _normalize_text(text)
    missing_phrases = [
        phrase for phrase in required_phrases if _normalize_text(phrase) not in norm_text
    ]
    if missing_phrases:
        return _CheckResult(
            ok=False,
            details={"reason": "missing_phrases", "missing_phrases": missing_phrases},
        )
    return _CheckResult(ok=True, details={"headings_found": heading_check.details["present"]})


def score_submission(
    output_payloads: Mapping[str, bytes | bytearray],
    reference_payloads: Mapping[str, bytes | bytearray],
    *,
    evaluation_spec_file: bytes | bytearray | str | Path = b"{}",
    required_phrase_file: bytes | bytearray | str | Path = b"{}",
):
    """Evaluate four output artifacts and return a binary score payload."""

    spec = _read_json_payload(evaluation_spec_file)
    required_phrases = []
    required_headings = []
    spec_matrix = spec.get("response_matrix.csv", {})
    spec_follow = spec.get("follow_up_items.csv", {})
    spec_letter = spec.get("response_letter.md", {})
    if not isinstance(spec_matrix, dict):
        spec_matrix = {}
    if not isinstance(spec_follow, dict):
        spec_follow = {}
    if not isinstance(spec_letter, dict):
        spec_letter = {}

    required_headings = spec_letter.get("headings", [])
    if not isinstance(required_headings, list):
        required_headings = []

    if required_phrase_file:
        phrase_payload = _read_json_payload(required_phrase_file)
        required_phrases = phrase_payload.get("phrases", [])
        if not isinstance(required_phrases, list):
            required_phrases = []

    matrix_check = _compare_csv_exact(
        output_payloads.get("response_matrix.csv", b""),
        reference_payloads.get("response_matrix.csv", b""),
        key_field="comment_id",
    )
    follow_check = _compare_csv_exact(
        output_payloads.get("follow_up_items.csv", b""),
        reference_payloads.get("follow_up_items.csv", b""),
        key_field="item_id",
    )
    manifest_check = _validate_manifest(
        output_payloads.get("response_manifest.json", b""),
        reference_payloads.get("response_manifest.json", b""),
    )
    letter_check = _validate_letter(
        output_payloads.get("response_letter.md", b""),
        required_headings=required_headings,
        required_phrases=required_phrases,
    )

    all_checks = [matrix_check, follow_check, manifest_check, letter_check]
    score = 1.0 if all(c.ok for c in all_checks) else 0.0
    return {
        "score": score,
        "details": {
            "response_matrix": matrix_check.details,
            "follow_up_items": follow_check.details,
            "response_manifest": manifest_check.details,
            "response_letter": letter_check.details,
            "pass": score == 1.0,
        },
    }
