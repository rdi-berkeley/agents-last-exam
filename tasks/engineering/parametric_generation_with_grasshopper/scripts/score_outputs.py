"""Local scorer for engineering/parametric_generation_with_grasshopper."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

CASES = ("A", "B")
REQUIRED_OUTPUT_FILE_STEMS = {
    "submission_gh": "submission.gh",
    "output_A": "outputA.3dm",
    "output_B": "outputB.3dm",
}
REQUIRED_CSV_FIELDS = ("case_id", "bbox_x", "bbox_y", "bbox_z", "height_ratio")
BBOX_REL_TOL = 0.15
HEIGHT_RATIO_ABS_TOL = 0.02


def _result(
    score: float,
    reasons: list[str],
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "score": float(score),
        "passed": score >= 1.0,
        "reasons": reasons,
        "details": details or {},
    }


def _load_reference_summary(csv_path: Path) -> dict[str, dict[str, float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")
        missing = [f for f in REQUIRED_CSV_FIELDS if f not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {missing}"
            )
        rows: dict[str, dict[str, float]] = {}
        for raw in reader:
            case = (raw.get("case_id") or "").strip()
            if case not in CASES:
                continue
            rows[case] = {
                "bbox_x": float(raw["bbox_x"]),
                "bbox_y": float(raw["bbox_y"]),
                "bbox_z": float(raw["bbox_z"]),
                "height_ratio": float(raw["height_ratio"]),
            }
    missing_cases = [c for c in CASES if c not in rows]
    if missing_cases:
        raise ValueError(f"{csv_path} is missing case rows: {missing_cases}")
    return rows


def _compute_aggregate_bbox(model_path: Path) -> dict[str, float]:
    import rhino3dm

    model = rhino3dm.File3dm.Read(str(model_path))
    if model is None:
        raise ValueError(f"rhino3dm could not read {model_path}")
    bbs = []
    for obj in model.Objects:
        geom = obj.Geometry
        try:
            bb = geom.GetBoundingBox()
        except Exception:
            continue
        bbs.append(bb)
    if not bbs:
        raise ValueError(f"{model_path} contained no geometry with bounding boxes")
    min_x = min(b.Min.X for b in bbs)
    max_x = max(b.Max.X for b in bbs)
    min_y = min(b.Min.Y for b in bbs)
    max_y = max(b.Max.Y for b in bbs)
    min_z = min(b.Min.Z for b in bbs)
    max_z = max(b.Max.Z for b in bbs)
    bbox_x = max_x - min_x
    bbox_y = max_y - min_y
    bbox_z = max_z - min_z
    denom = math.sqrt(bbox_x * bbox_x + bbox_y * bbox_y)
    if denom <= 0:
        raise ValueError(f"{model_path} has a degenerate footprint (bbox_x=bbox_y=0)")
    return {
        "bbox_x": float(bbox_x),
        "bbox_y": float(bbox_y),
        "bbox_z": float(bbox_z),
        "height_ratio": float(bbox_z / denom),
    }


def _compare_case(
    case: str,
    computed: dict[str, float],
    expected: dict[str, float],
) -> list[str]:
    reasons: list[str] = []
    for dim in ("bbox_x", "bbox_y", "bbox_z"):
        ref = expected[dim]
        got = computed[dim]
        if ref <= 0:
            reasons.append(f"case {case} reference {dim}={ref} is non-positive")
            continue
        rel = abs(got - ref) / abs(ref)
        if rel > BBOX_REL_TOL:
            reasons.append(
                f"case {case} {dim} rel error {rel:.3f} exceeds tolerance "
                f"{BBOX_REL_TOL:.2f} (got {got:.4f}, ref {ref:.4f})"
            )
    hr_ref = expected["height_ratio"]
    hr_got = computed["height_ratio"]
    if abs(hr_got - hr_ref) > HEIGHT_RATIO_ABS_TOL:
        reasons.append(
            f"case {case} height_ratio abs error {abs(hr_got - hr_ref):.4f} exceeds tolerance "
            f"{HEIGHT_RATIO_ABS_TOL:.2f} (got {hr_got:.4f}, ref {hr_ref:.4f})"
        )
    return reasons


def evaluate_submission(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    submission_gh = output_dir / REQUIRED_OUTPUT_FILE_STEMS["submission_gh"]
    output_files = {
        "A": output_dir / REQUIRED_OUTPUT_FILE_STEMS["output_A"],
        "B": output_dir / REQUIRED_OUTPUT_FILE_STEMS["output_B"],
    }
    reference_csv = reference_dir / "reference_summary.csv"

    missing = [
        str(path)
        for path in [submission_gh, output_files["A"], output_files["B"], reference_csv]
        if not path.exists()
    ]
    if missing:
        return _result(0.0, [f"missing required files: {', '.join(missing)}"])

    if submission_gh.stat().st_size == 0:
        return _result(0.0, [f"submission.gh is empty at {submission_gh}"])

    for case, path in output_files.items():
        if path.stat().st_size == 0:
            return _result(0.0, [f"output{case}.3dm is empty at {path}"])

    try:
        reference_rows = _load_reference_summary(reference_csv)
    except Exception as exc:
        return _result(0.0, [f"failed to read reference summary: {exc}"])

    details: dict[str, Any] = {"tolerances": {
        "bbox_rel": BBOX_REL_TOL,
        "height_ratio_abs": HEIGHT_RATIO_ABS_TOL,
    }, "per_case": {}}
    reasons: list[str] = []

    for case in CASES:
        try:
            computed = _compute_aggregate_bbox(output_files[case])
        except Exception as exc:
            return _result(0.0, [f"failed to parse output{case}.3dm: {exc}"], details=details)
        expected = reference_rows[case]
        details["per_case"][case] = {
            "computed": computed,
            "expected": expected,
        }
        reasons.extend(_compare_case(case, computed, expected))

    score = 1.0 if not reasons else 0.0
    return _result(score, reasons, details=details)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_submission(
        output_dir=Path(args.output_dir),
        reference_dir=Path(args.reference_dir),
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
