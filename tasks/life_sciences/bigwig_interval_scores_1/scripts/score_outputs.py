"""Score bigwig_interval_scores_1 output tables."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path


FLOAT_COLUMNS = (3, 4, 5)
INT_COLUMNS = (1, 2)


@dataclass
class ScoreResult:
    score: float
    passed: bool
    passed_rows: int
    total_rows: int
    reasons: list[str]


def _parse_table(text: str, *, expected_rows: int | None = None) -> tuple[dict[str, list[str]], list[str]]:
    rows: dict[str, list[str]] = {}
    reasons: list[str] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        parts = raw_line.rstrip("\n").split("\t")
        if len(parts) != 6:
            reasons.append(f"malformed_row:{line_no}:columns={len(parts)}")
            continue
        name = parts[0]
        if not name:
            reasons.append(f"missing_name:{line_no}")
            continue
        if name in rows:
            reasons.append(f"duplicate_name:{name}")
            continue
        try:
            int(parts[1])
            int(parts[2])
            float_values = [float(parts[idx]) for idx in FLOAT_COLUMNS]
        except ValueError:
            reasons.append(f"non_numeric:{name}")
            continue
        if not all(math.isfinite(value) for value in float_values):
            reasons.append(f"non_finite:{name}")
            continue
        rows[name] = parts
    if expected_rows is not None and len(rows) != expected_rows:
        reasons.append(f"row_count:{len(rows)}!=expected:{expected_rows}")
    return rows, reasons


def score_text(
    *,
    agent_text: str,
    reference_text: str,
    expected_rows: int = 78691,
    float_abs_tolerance: float = 1e-6,
) -> ScoreResult:
    reference_rows, reference_errors = _parse_table(reference_text, expected_rows=expected_rows)
    if reference_errors:
        raise RuntimeError("invalid hidden reference: " + "; ".join(reference_errors[:10]))

    agent_rows, agent_errors = _parse_table(agent_text, expected_rows=expected_rows)
    if agent_errors:
        return ScoreResult(
            score=0.0,
            passed=False,
            passed_rows=0,
            total_rows=len(reference_rows),
            reasons=agent_errors[:20],
        )

    expected_names = set(reference_rows)
    agent_names = set(agent_rows)
    if agent_names != expected_names:
        missing = sorted(expected_names - agent_names)[:10]
        extra = sorted(agent_names - expected_names)[:10]
        return ScoreResult(
            score=0.0,
            passed=False,
            passed_rows=0,
            total_rows=len(reference_rows),
            reasons=[f"name_set_mismatch:missing={missing}:extra={extra}"],
        )

    passed_rows = 0
    reasons: list[str] = []
    for name, expected in reference_rows.items():
        actual = agent_rows[name]
        row_ok = True
        for idx in INT_COLUMNS:
            if int(actual[idx]) != int(expected[idx]):
                row_ok = False
                break
        if row_ok:
            for idx in FLOAT_COLUMNS:
                actual_value = float(actual[idx])
                expected_value = float(expected[idx])
                if (
                    not math.isfinite(actual_value)
                    or not math.isfinite(expected_value)
                    or abs(actual_value - expected_value) > float_abs_tolerance
                ):
                    row_ok = False
                    break
        if row_ok:
            passed_rows += 1
        elif len(reasons) < 20:
            reasons.append(f"row_mismatch:{name}")

    total = len(reference_rows)
    score = passed_rows / total if total else 0.0
    return ScoreResult(
        score=score,
        passed=passed_rows == total,
        passed_rows=passed_rows,
        total_rows=total,
        reasons=reasons,
    )


def score_files(
    *,
    agent_path: Path,
    reference_path: Path,
    expected_rows: int = 78691,
    float_abs_tolerance: float = 1e-6,
) -> ScoreResult:
    return score_text(
        agent_text=agent_path.read_text(encoding="utf-8"),
        reference_text=reference_path.read_text(encoding="utf-8"),
        expected_rows=expected_rows,
        float_abs_tolerance=float_abs_tolerance,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=78691)
    parser.add_argument("--float-abs-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    result = score_files(
        agent_path=args.agent,
        reference_path=args.reference,
        expected_rows=args.expected_rows,
        float_abs_tolerance=args.float_abs_tolerance,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
