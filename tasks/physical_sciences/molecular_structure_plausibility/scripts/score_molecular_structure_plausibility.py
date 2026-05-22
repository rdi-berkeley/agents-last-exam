"""Local evaluator for the molecular structure plausibility task."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FILENAME_RE = re.compile(r"[A-Za-z0-9_.-]+\.xyz")


def parse_filename_set(text: str) -> set[str]:
    """Extract .xyz filenames from common line, bullet, comma, or prose outputs."""
    return set(FILENAME_RE.findall(text))


def parse_reference(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size <= 0:
        raise ValueError("missing or empty reference answer key")
    names = parse_filename_set(path.read_text(encoding="utf-8"))
    if not names:
        raise ValueError("reference answer key contains no .xyz filenames")
    return names


def parse_input_filenames(input_xyz_dir: Path) -> set[str]:
    if not input_xyz_dir.exists() or not input_xyz_dir.is_dir():
        raise ValueError("missing input xyz directory")
    names = {path.name for path in input_xyz_dir.glob("*.xyz")}
    if not names:
        raise ValueError("input xyz directory contains no .xyz files")
    return names


def evaluate_files(
    *,
    output_file: Path,
    reference_file: Path,
    input_xyz_dir: Path,
) -> dict[str, Any]:
    if not output_file.exists():
        return {"score": 0.0, "reason": "missing_output", "details": {}}
    if output_file.stat().st_size <= 0:
        return {"score": 0.0, "reason": "empty_output", "details": {}}
    if output_file.stat().st_size > 20_000:
        return {"score": 0.0, "reason": "output_too_large", "details": {}}

    try:
        input_names = parse_input_filenames(input_xyz_dir)
        reference_names = parse_reference(reference_file)
        output_names = parse_filename_set(output_file.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"score": 0.0, "reason": f"parse_failed: {exc}", "details": {}}

    unknown = sorted(output_names - input_names)
    if unknown:
        return {
            "score": 0.0,
            "reason": "unknown_filenames",
            "details": {"unknown": unknown},
        }
    if not output_names:
        return {"score": 0.0, "reason": "no_filenames", "details": {}}

    missing = sorted(reference_names - output_names)
    extra = sorted(output_names - reference_names)
    intersection = len(reference_names & output_names)
    union = len(reference_names | output_names)
    score = intersection / union if union else 0.0
    return {
        "score": score,
        "reason": "exact_match" if score == 1.0 else "partial_match",
        "details": {
            "output_count": len(output_names),
            "reference_count": len(reference_names),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing": missing,
            "extra": extra,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--reference-file", required=True, type=Path)
    parser.add_argument("--input-xyz-dir", required=True, type=Path)
    args = parser.parse_args()

    result = evaluate_files(
        output_file=args.output_file,
        reference_file=args.reference_file,
        input_xyz_dir=args.input_xyz_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
