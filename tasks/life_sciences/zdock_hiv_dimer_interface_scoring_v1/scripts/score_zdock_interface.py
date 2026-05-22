"""Score ZDOCK HIV dimer interface CSV outputs against the hidden reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = (
    "Pose Rank (ZDOCK)",
    "Overlap Score",
    "Fnat",
    "IRMSD",
    "Final Score",
    "Final Rank",
)

TOLERANCES = {
    "Overlap Score": 0.01,
    "Fnat": 0.05,
    "IRMSD": 0.5,
    "Final Score": 0.05,
}

EXPECTED_ROW_COUNT = 10


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    return rows, fieldnames


def _parse_int(value: str, *, field: str, row_label: str) -> int:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{row_label}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed) or int(parsed) != parsed:
        raise ValueError(f"{row_label}: {field} is not an integer: {value!r}")
    return int(parsed)


def _parse_float(value: str, *, field: str, row_label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{row_label}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{row_label}: {field} is not finite: {value!r}")
    return parsed


def evaluate_files(output_file: Path, reference_file: Path) -> dict[str, Any]:
    """Return a JSON-serializable score payload.

    The benchmark contract is all-or-nothing: a correct CSV must satisfy every
    structural check, every numeric tolerance, and every exact final rank.
    """

    failures: list[str] = []
    if not output_file.exists():
        return {"score": 0.0, "passed": False, "failures": [f"missing output: {output_file}"]}
    if not reference_file.exists():
        return {
            "score": 0.0,
            "passed": False,
            "failures": [f"missing reference: {reference_file}"],
        }

    try:
        output_rows, output_columns = _read_csv(output_file)
        reference_rows, reference_columns = _read_csv(reference_file)
    except Exception as exc:
        return {"score": 0.0, "passed": False, "failures": [f"CSV parse failed: {exc}"]}

    for source_name, columns in [
        ("output", output_columns),
        ("reference", reference_columns),
    ]:
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            failures.append(f"{source_name} missing required columns: {missing}")

    if len(output_rows) != EXPECTED_ROW_COUNT:
        failures.append(f"output row count {len(output_rows)} != {EXPECTED_ROW_COUNT}")
    if len(reference_rows) != EXPECTED_ROW_COUNT:
        failures.append(f"reference row count {len(reference_rows)} != {EXPECTED_ROW_COUNT}")
    if failures:
        return {"score": 0.0, "passed": False, "failures": failures}

    try:
        reference_by_pose = {
            _parse_int(row["Pose Rank (ZDOCK)"], field="Pose Rank (ZDOCK)", row_label="reference"): row
            for row in reference_rows
        }
        output_by_pose = {
            _parse_int(row["Pose Rank (ZDOCK)"], field="Pose Rank (ZDOCK)", row_label="output"): row
            for row in output_rows
        }
    except ValueError as exc:
        return {"score": 0.0, "passed": False, "failures": [str(exc)]}

    if len(reference_by_pose) != EXPECTED_ROW_COUNT:
        failures.append("reference contains duplicate pose ranks")
    if len(output_by_pose) != EXPECTED_ROW_COUNT:
        failures.append("output contains duplicate pose ranks")
    if set(output_by_pose) != set(reference_by_pose):
        failures.append(
            "output pose ranks do not match reference: "
            f"expected {sorted(reference_by_pose)}, got {sorted(output_by_pose)}"
        )
    if failures:
        return {"score": 0.0, "passed": False, "failures": failures}

    checked_values = 0
    for pose_rank in sorted(reference_by_pose):
        ref_row = reference_by_pose[pose_rank]
        out_row = output_by_pose[pose_rank]
        row_label = f"pose {pose_rank}"

        try:
            expected_final_rank = _parse_int(
                ref_row["Final Rank"], field="Final Rank", row_label=f"reference {row_label}"
            )
            observed_final_rank = _parse_int(
                out_row["Final Rank"], field="Final Rank", row_label=f"output {row_label}"
            )
        except ValueError as exc:
            failures.append(str(exc))
            continue

        if observed_final_rank != expected_final_rank:
            failures.append(
                f"{row_label}: Final Rank {observed_final_rank} != {expected_final_rank}"
            )
        checked_values += 1

        for field, tolerance in TOLERANCES.items():
            try:
                expected = _parse_float(
                    ref_row[field], field=field, row_label=f"reference {row_label}"
                )
                observed = _parse_float(out_row[field], field=field, row_label=f"output {row_label}")
            except ValueError as exc:
                failures.append(str(exc))
                continue
            difference = abs(observed - expected)
            checked_values += 1
            if difference > tolerance + 1e-12:
                failures.append(
                    f"{row_label}: {field} diff {difference:.6g} exceeds tolerance {tolerance}"
                )

    return {
        "score": 0.0 if failures else 1.0,
        "passed": not failures,
        "failures": failures,
        "checked_poses": EXPECTED_ROW_COUNT,
        "checked_values": checked_values,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--reference-file", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_files(args.output_file, args.reference_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["score"] >= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
