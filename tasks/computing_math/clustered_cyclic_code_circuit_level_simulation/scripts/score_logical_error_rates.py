"""Score clustered-cyclic code logical error-rate CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Any

EXPECTED_COLUMNS = [
    "code",
    "n",
    "k",
    "d",
    "p_physical",
    "num_rounds",
    "num_shots",
    "num_failures",
    "p_logical",
    "lfr_per_round",
    "lfr_per_round_per_qubit",
]

KEY_FIELDS = ["code", "p_physical"]
EXACT_INT_FIELDS = ["n", "k", "d", "num_rounds", "num_shots"]
LOG_TOLERANCE = 0.1


@dataclass
class LogicalErrorRateScoreResult:
    score: float
    passed: bool
    reasons: list[str]
    rows_checked: int = 0


def _parse_csv_bytes(data: bytes, label: str) -> tuple[list[str], list[dict[str, str]], list[str]]:
    reasons: list[str] = []
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return [], [], [f"{label}:decode_error:{exc}"]

    try:
        reader = csv.DictReader(StringIO(text))
        fieldnames = reader.fieldnames or []
        rows = [dict(row) for row in reader]
    except csv.Error as exc:
        return [], [], [f"{label}:csv_error:{exc}"]

    if fieldnames != EXPECTED_COLUMNS:
        reasons.append(f"{label}:column_mismatch")
    if not rows:
        reasons.append(f"{label}:no_rows")
    return fieldnames, rows, reasons


def _parse_decimal(value: Any, field: str, reasons: list[str], row_key: str) -> Decimal | None:
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, AttributeError):
        reasons.append(f"invalid_decimal:{row_key}:{field}")
        return None
    if not parsed.is_finite():
        reasons.append(f"nonfinite_decimal:{row_key}:{field}")
        return None
    return parsed


def _parse_float(value: Any, field: str, reasons: list[str], row_key: str) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        reasons.append(f"invalid_float:{row_key}:{field}")
        return None
    if not math.isfinite(parsed):
        reasons.append(f"nonfinite_float:{row_key}:{field}")
        return None
    return parsed


def _parse_int(value: Any, field: str, reasons: list[str], row_key: str) -> int | None:
    raw = str(value).strip()
    try:
        parsed = Decimal(raw)
    except InvalidOperation:
        reasons.append(f"invalid_integer:{row_key}:{field}")
        return None
    if parsed != parsed.to_integral_value():
        reasons.append(f"non_integer:{row_key}:{field}")
        return None
    return int(parsed)


def _row_key(row: dict[str, str], reasons: list[str], label: str) -> tuple[str, Decimal] | None:
    code = (row.get("code") or "").strip()
    if not code:
        reasons.append(f"{label}:missing_code")
        return None
    p_physical = _parse_decimal(row.get("p_physical"), "p_physical", reasons, code)
    if p_physical is None:
        return None
    return code, p_physical


def _index_rows(rows: list[dict[str, str]], label: str, reasons: list[str]) -> dict[tuple[str, Decimal], dict[str, str]]:
    indexed: dict[tuple[str, Decimal], dict[str, str]] = {}
    for idx, row in enumerate(rows, start=1):
        key = _row_key(row, reasons, f"{label}:row{idx}")
        if key is None:
            continue
        if key in indexed:
            reasons.append(f"{label}:duplicate_row:{key[0]}:{key[1]}")
            continue
        indexed[key] = row
    return indexed


def _consistent_rate(p_logical: float, rounds: int, logical_qubits: int = 1) -> float:
    if p_logical >= 1.0:
        return 1.0
    if p_logical <= 0.0:
        return 0.0
    exponent = 1.0 / float(rounds * logical_qubits)
    return 1.0 - math.pow(1.0 - p_logical, exponent)


def _close_numeric(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-12, abs(expected) * 1e-9)


def _log_close(actual: float, expected: float) -> bool:
    if actual <= 0.0 or expected <= 0.0:
        return False
    return abs(math.log10(actual) - math.log10(expected)) < LOG_TOLERANCE


def score_logical_error_rates_bytes(
    *, agent_bytes: bytes, reference_bytes: bytes
) -> LogicalErrorRateScoreResult:
    _, reference_rows, reasons = _parse_csv_bytes(reference_bytes, "reference")
    _, agent_rows, agent_parse_reasons = _parse_csv_bytes(agent_bytes, "agent")
    reasons.extend(agent_parse_reasons)
    if reasons:
        return LogicalErrorRateScoreResult(score=0.0, passed=False, reasons=reasons)

    reference = _index_rows(reference_rows, "reference", reasons)
    agent = _index_rows(agent_rows, "agent", reasons)
    if reasons:
        return LogicalErrorRateScoreResult(score=0.0, passed=False, reasons=reasons)

    reference_keys = set(reference)
    agent_keys = set(agent)
    missing = sorted(reference_keys - agent_keys, key=lambda key: (key[0], key[1]))
    extra = sorted(agent_keys - reference_keys, key=lambda key: (key[0], key[1]))
    if missing:
        reasons.append("missing_rows:" + ",".join(f"{code}:{p}" for code, p in missing[:10]))
    if extra:
        reasons.append("extra_rows:" + ",".join(f"{code}:{p}" for code, p in extra[:10]))
    if reasons:
        return LogicalErrorRateScoreResult(score=0.0, passed=False, reasons=reasons)

    for key in sorted(reference_keys, key=lambda item: (item[0], item[1])):
        row_key = f"{key[0]}:{key[1]}"
        ref = reference[key]
        got = agent[key]

        if (got.get("code") or "").strip() != (ref.get("code") or "").strip():
            reasons.append(f"code_mismatch:{row_key}")

        got_p = _parse_decimal(got.get("p_physical"), "p_physical", reasons, row_key)
        ref_p = _parse_decimal(ref.get("p_physical"), "p_physical", reasons, row_key)
        if got_p is not None and ref_p is not None and got_p != ref_p:
            reasons.append(f"p_physical_mismatch:{row_key}")

        parsed_ints: dict[str, int] = {}
        for field in EXACT_INT_FIELDS:
            got_int = _parse_int(got.get(field), field, reasons, row_key)
            ref_int = _parse_int(ref.get(field), field, reasons, row_key)
            if got_int is None or ref_int is None:
                continue
            parsed_ints[field] = got_int
            if got_int != ref_int:
                reasons.append(f"{field}_mismatch:{row_key}")

        failures = _parse_int(got.get("num_failures"), "num_failures", reasons, row_key)
        shots = parsed_ints.get("num_shots")
        rounds = parsed_ints.get("num_rounds")
        k = parsed_ints.get("k")
        p_logical = _parse_float(got.get("p_logical"), "p_logical", reasons, row_key)
        lfr = _parse_float(got.get("lfr_per_round"), "lfr_per_round", reasons, row_key)
        lfr_per_qubit = _parse_float(
            got.get("lfr_per_round_per_qubit"), "lfr_per_round_per_qubit", reasons, row_key
        )
        ref_lfr = _parse_float(ref.get("lfr_per_round"), "ref_lfr_per_round", reasons, row_key)
        ref_lfr_per_qubit = _parse_float(
            ref.get("lfr_per_round_per_qubit"),
            "ref_lfr_per_round_per_qubit",
            reasons,
            row_key,
        )

        if failures is not None and shots is not None:
            if failures < 0 or failures > shots:
                reasons.append(f"num_failures_out_of_range:{row_key}")
            if p_logical is not None and not _close_numeric(p_logical, failures / shots):
                reasons.append(f"p_logical_inconsistent:{row_key}")

        if (
            p_logical is not None
            and lfr is not None
            and rounds is not None
            and not _close_numeric(lfr, _consistent_rate(p_logical, rounds))
        ):
            reasons.append(f"lfr_formula_inconsistent:{row_key}")

        if (
            p_logical is not None
            and lfr_per_qubit is not None
            and rounds is not None
            and k is not None
            and not _close_numeric(lfr_per_qubit, _consistent_rate(p_logical, rounds, k))
        ):
            reasons.append(f"lfr_per_qubit_formula_inconsistent:{row_key}")

        if lfr is not None and ref_lfr is not None and not _log_close(lfr, ref_lfr):
            reasons.append(f"lfr_per_round_log_mismatch:{row_key}")
        if (
            lfr_per_qubit is not None
            and ref_lfr_per_qubit is not None
            and not _log_close(lfr_per_qubit, ref_lfr_per_qubit)
        ):
            reasons.append(f"lfr_per_round_per_qubit_log_mismatch:{row_key}")

    passed = not reasons
    return LogicalErrorRateScoreResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        reasons=reasons[:50],
        rows_checked=len(reference_keys),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    result = score_logical_error_rates_bytes(
        agent_bytes=Path(args.agent).read_bytes(),
        reference_bytes=Path(args.reference).read_bytes(),
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
