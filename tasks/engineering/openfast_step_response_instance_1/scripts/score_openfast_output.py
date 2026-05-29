"""Score a staged OpenFAST step-response output tree."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_response_summary import SUMMARY_COLUMNS, compute_summary


ALLOWED_EDIT_FILES = {
    "NREL-5MW.fst",
    "NRELOffshrBsline5MW_InflowWind.dat",
}
REQUIRED_FILES = {
    "NREL-5MW.fst",
    "NRELOffshrBsline5MW_InflowWind.dat",
    "NREL-5MW.outb",
    "response_summary.csv",
}
TOLERANCES = {
    "max_gen_speed_rpm_60_120s": 0.5,
    "settling_time_s_after_step": 1.0,
    "mean_gen_pwr_kw_150_180s": 5.0,
    "max_abs_tower_top_fa_m_0_180s": 0.02,
    "mean_collective_pitch_deg_150_180s": 0.05,
}
WEIGHTS = {
    "wind_path": 0.10,
    "tmax": 0.10,
    "allowed_files": 0.10,
    "valid_outb": 0.15,
    "summary_schema": 0.15,
    "max_gen_speed_rpm_60_120s": 0.10,
    "settling_time_s_after_step": 0.10,
    "mean_gen_pwr_kw_150_180s": 0.10,
    "max_abs_tower_top_fa_m_0_180s": 0.05,
    "mean_collective_pitch_deg_150_180s": 0.05,
}


def _relative_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _parse_tmax(text: str) -> float | None:
    for line in text.splitlines():
        match = re.match(r"^\s*([0-9.]+)\s+TMax\b", line)
        if match:
            return float(match.group(1))
    return None


def _parse_wind_reference(text: str) -> str | None:
    for line in text.splitlines():
        if "FileName_Uni" in line or "Filename_Uni" in line:
            match = re.search(r'"([^"]+)"', line)
            if match:
                return match.group(1).replace("/", "\\").lower()
    return None


def _load_submitted_summary(csv_path: Path) -> tuple[list[str], dict[str, float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) != 2:
        raise ValueError("response_summary.csv must contain exactly one header row and one data row")
    header = rows[0]
    if header != SUMMARY_COLUMNS:
        raise ValueError(f"unexpected response_summary.csv header: {header}")
    values = rows[1]
    if len(values) != len(SUMMARY_COLUMNS):
        raise ValueError("response_summary.csv data row has the wrong number of columns")
    summary: dict[str, float] = {}
    for key, raw in zip(SUMMARY_COLUMNS, values, strict=True):
        summary[key] = float(raw)
    return header, summary


def _compare_starter_tree(starter_output_dir: Path, output_dir: Path) -> list[str]:
    violations: list[str] = []
    for rel in sorted(_relative_files(starter_output_dir)):
        if rel in ALLOWED_EDIT_FILES:
            continue
        starter_file = starter_output_dir / rel
        output_file = output_dir / rel
        if not output_file.exists():
            violations.append(rel)
            continue
        if starter_file.read_bytes() != output_file.read_bytes():
            violations.append(rel)
    return violations


def _metric_within_tolerance(actual: float, expected: float, tolerance: float) -> bool:
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def score_output_tree(
    *,
    starter_output_dir: Path,
    output_dir: Path,
    reference_summary_path: Path,
) -> dict[str, object]:
    missing = sorted(name for name in REQUIRED_FILES if not (output_dir / name).exists())
    if missing:
        return {"score": 0.0, "hard_fail_reason": f"missing required files: {missing}"}

    fst_text = (output_dir / "NREL-5MW.fst").read_text(encoding="utf-8")
    inflow_text = (output_dir / "NRELOffshrBsline5MW_InflowWind.dat").read_text(encoding="utf-8")

    tmax_value = _parse_tmax(fst_text)
    if tmax_value != 180.0:
        return {"score": 0.0, "hard_fail_reason": f"TMax is not 180.0: {tmax_value}"}

    wind_ref = _parse_wind_reference(inflow_text)
    if wind_ref != r"..\input\wind\step_8to12mps.wnd":
        return {"score": 0.0, "hard_fail_reason": f"unexpected FileName_Uni: {wind_ref}"}

    starter_violations = _compare_starter_tree(starter_output_dir, output_dir)
    if starter_violations:
        return {"score": 0.0, "hard_fail_reason": f"unexpected starter-file edits: {starter_violations}"}

    try:
        submitted_summary = _load_submitted_summary(output_dir / "response_summary.csv")[1]
    except Exception as exc:
        return {"score": 0.0, "hard_fail_reason": f"invalid response_summary.csv: {exc}"}

    try:
        outb_summary = compute_summary(output_dir / "NREL-5MW.outb")
    except Exception as exc:
        return {"score": 0.0, "hard_fail_reason": f"invalid outb: {exc}"}

    for key, tolerance in TOLERANCES.items():
        if not _metric_within_tolerance(submitted_summary[key], outb_summary[key], tolerance):
            return {"score": 0.0, "hard_fail_reason": f"response_summary.csv mismatch for {key}"}

    expert_summary = json.loads(reference_summary_path.read_text(encoding="utf-8"))

    score = 0.0
    score += WEIGHTS["wind_path"]
    score += WEIGHTS["tmax"]
    score += WEIGHTS["allowed_files"]
    score += WEIGHTS["valid_outb"]
    score += WEIGHTS["summary_schema"]

    metric_hits: dict[str, bool] = {}
    for key in SUMMARY_COLUMNS:
        hit = _metric_within_tolerance(outb_summary[key], float(expert_summary[key]), TOLERANCES[key])
        metric_hits[key] = hit
        if hit:
            score += WEIGHTS[key]

    return {
        "score": round(score, 10),
        "hard_fail_reason": None,
        "metric_hits": metric_hits,
        "submitted_summary": submitted_summary,
        "outb_summary": outb_summary,
        "expert_summary": expert_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        json.dumps(
            score_output_tree(
                starter_output_dir=args.starter_output_dir,
                output_dir=args.output_dir,
                reference_summary_path=args.reference_summary,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
