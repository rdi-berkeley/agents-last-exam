#!/usr/bin/env python3
"""VM-side verifier for the OpenEMR screening forms task family."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_FILES = [
    "form_export.tsv",
    "response_export.tsv",
    "score_summary.tsv",
]


def _read_tsv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        return [[cell.strip() for cell in row] for row in reader]


def _compare_table(output_path: Path, reference_path: Path) -> dict[str, object]:
    if not output_path.exists():
        return {"file": output_path.name, "passed": False, "reason": "missing"}
    if not reference_path.exists():
        return {"file": output_path.name, "passed": False, "reason": "missing_reference"}

    output_rows = _read_tsv(output_path)
    reference_rows = _read_tsv(reference_path)
    passed = output_rows == reference_rows
    return {
        "file": output_path.name,
        "passed": passed,
        "output_rows": len(output_rows),
        "reference_rows": len(reference_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_dir)

    checks = []
    passed_count = 0
    for name in REQUIRED_FILES:
        result = _compare_table(output_dir / name, reference_dir / name)
        checks.append(result)
        passed_count += int(bool(result["passed"]))

    score = passed_count / len(REQUIRED_FILES)
    print(
        json.dumps(
            {
                "score": score,
                "checks": checks,
            }
        )
    )
    return 0 if score == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
