"""Score clustered-cyclic code logical error-rate CSV outputs.

Scoring model (revised 2026-06-13, per task author Andy Liu):

The original grader compared every ``lfr_per_round`` / ``lfr_per_round_per_qubit``
value against a frozen benchmark reference on a log scale (``LOG_TOLERANCE``).
That exact-value match is not appropriate for this task: the reference was
generated on a compute cluster with a large shot budget, whereas agents are
evaluated on a single CPU VM where the achievable shot count is much lower, so
the absolute logical-error-rate values carry significant Monte-Carlo noise even
for a correct implementation. Matching absolute values therefore false-fails
correct agents.

The author's guiding success criterion is a *physical invariant* that is robust
to the absolute-scale sampling difference because it compares two codes the
agent simulated under identical conditions:

    Across physical error rates, increasing code distance suppresses the logical
    error rate along the ladder of the three k=8 codes:

        [24,8,3] (d=3)  >=  [40,8,5] (d=5)  >=  [56,8,7] (d=7).

This is the standard below-threshold distance-suppression signature: a higher
code distance suppresses logical errors more strongly. A broken decoder (the
failure mode that motivated this task's audit) inverts or flattens it. We check
each adjacent rung of the ladder in the sub-threshold regime, where it is both
physically meaningful and statistically resolvable, with a noise margin so a
correct but under-sampled agent is not false-failed. (Confirmed 2026-06-23 by a
controlled experiment: the absolute logical-error rate is driven by the CNOT
schedule, which the staged spec under-specifies — two valid direction-aware
schedules of identical depth differ ~19x at p=1e-4 — but the distance-suppression
*ordering* is invariant to that choice. The author endorsed the [56,8,7] ladder
on 2026-06-22.)

The schema, simulation-grid, declared-parameter, and internal-consistency
checks (``p_logical == num_failures / num_shots``, the LFR formulas) are
retained unchanged — the reference is still used to pin the required
``(code, p_physical)`` grid and the declared integer parameters, just not the
outcome values.
"""

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

# ---------------------------------------------------------------------------
# Distance-suppression criterion parameters (author-specified invariant).
#
# The ladder: the three k=8 codes in order of increasing distance. They share
# k=8 and the same noise/decoder pipeline, differing only in distance, so they
# are directly comparable block-for-block. Each adjacent rung must show
# suppression (the higher-distance code's p_logical no greater than the
# lower-distance code's, within a noise margin) across the sub-threshold rates.
# ---------------------------------------------------------------------------
SUPPRESSION_LADDER = ["[24,8,3]", "[40,8,5]", "[56,8,7]"]  # d=3 -> d=5 -> d=7

# Only assess the invariant where the d=3 code is still well below the
# random-guess saturation ceiling. Above threshold both codes saturate near 1.0
# and their ordering is pure Monte-Carlo noise.
SUBTHRESHOLD_CEILING = 0.4

# Noise margin: the d=5 rate may exceed the d=3 rate by at most this much before
# it counts as a suppression violation. Absorbs binomial sampling error on a
# CPU-VM shot budget; physically, sub-threshold suppression is strong (the d=5
# rate is typically many times below the d=3 rate), so a modest margin does not
# admit a broken/inverted decoder.
SUPPRESSION_ABS_MARGIN = 0.02
SUPPRESSION_REL_MARGIN = 0.25

# Guards so a degenerate output (e.g. all-zeros, or a flat curve) cannot pass
# the invariant trivially.
MIN_SUBTHRESHOLD_POINTS = 3   # need enough resolvable sub-threshold comparisons
MIN_SIGNAL_PLOGICAL = 0.01    # the d=3 curve must reach a real error rate somewhere


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


def _check_distance_suppression(
    agent: dict[tuple[str, Decimal], dict[str, str]],
    reasons: list[str],
) -> None:
    """Assert the author's distance-suppression invariant on the agent's output.

    Walk the ladder of k=8 codes in increasing-distance order. For each adjacent
    rung (lower-distance ``lo`` -> higher-distance ``hi``), the higher-distance
    code must achieve a logical error rate no higher (within a noise margin) than
    the lower-distance one across the sub-threshold physical error rates. Compares
    the agent's own codes, sampled under identical conditions, so the check is
    robust to the absolute-scale difference between the reference cluster run and
    the evaluation VM (and to the under-specified CNOT schedule).
    """
    def curve(code: str) -> dict[Decimal, float]:
        out: dict[Decimal, float] = {}
        for (row_code, p_physical), row in agent.items():
            if row_code.strip() != code:
                continue
            value = _parse_float(row.get("p_logical"), "p_logical", [], f"{code}:{p_physical}")
            if value is not None:
                out[p_physical] = value
        return out

    curves = {code: curve(code) for code in SUPPRESSION_LADDER}
    missing = [code for code in SUPPRESSION_LADDER if not curves[code]]
    if missing:
        reasons.append("suppression:missing_code:" + ",".join(missing))
        return

    # Non-trivial-signal guard: the lowest-distance code must reach a real error
    # rate somewhere, otherwise an all-zero / flat output would pass trivially.
    base_code = SUPPRESSION_LADDER[0]
    if max(curves[base_code].values()) < MIN_SIGNAL_PLOGICAL:
        reasons.append("suppression:no_signal:base_curve_below_floor")
        return

    # Each adjacent rung must independently show suppression across enough
    # resolvable sub-threshold points. "Sub-threshold" is defined by the
    # lower-distance code of the rung still being below the saturation ceiling.
    for lo_code, hi_code in zip(SUPPRESSION_LADDER, SUPPRESSION_LADDER[1:]):
        lo, hi = curves[lo_code], curves[hi_code]
        shared = sorted(set(lo) & set(hi))
        subthreshold = [p for p in shared if lo[p] <= SUBTHRESHOLD_CEILING]

        if len(subthreshold) < MIN_SUBTHRESHOLD_POINTS:
            reasons.append(
                f"suppression:insufficient_subthreshold_points:{lo_code}->{hi_code}:"
                f"{len(subthreshold)}<{MIN_SUBTHRESHOLD_POINTS}"
            )
            continue

        for p in subthreshold:
            lo_v = lo[p]
            hi_v = hi[p]
            margin = max(SUPPRESSION_ABS_MARGIN, SUPPRESSION_REL_MARGIN * lo_v)
            if hi_v > lo_v + margin:
                reasons.append(
                    f"distance_suppression_violated:{lo_code}->{hi_code}:p={p}:"
                    f"hi={hi_v:.6g}>lo={lo_v:.6g}+margin={margin:.6g}"
                )


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

    # Author's physical-invariant criterion (replaces the absolute log-scale
    # value match against the reference).
    _check_distance_suppression(agent, reasons)

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
