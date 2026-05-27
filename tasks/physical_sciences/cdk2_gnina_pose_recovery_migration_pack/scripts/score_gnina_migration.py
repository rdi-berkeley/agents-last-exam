#!/usr/bin/env python
"""Score a CDK2 GNINA migration submission on the VM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROW_FLOAT_TOLS = {
    "top1_rmsd_A": 0.02,
    "top1_affinity_kcal_mol": 0.02,
    "cnn_pose_score": 0.005,
    "cnn_affinity": 0.02,
}
SUMMARY_FLOAT_TOLS = {
    "success_rate": 0.02,
    "median_top1_rmsd_A_success_only": 0.02,
    "median_cnn_pose_score": 0.02,
}
VISIBLE_ROW_POINTS = 5
HIDDEN_ROW_POINTS = 5
MEMO_TOKENS = [
    "GNINA 1.3.1",
    "AutoDock Vina 1.2.6",
    "legacy baseline",
    "RMSD",
    "success rate",
]
MEMO_REQUIRED_CASES = ["1DI8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--result-path", required=True, type=Path)
    return parser.parse_args()


def resolve_project_root(submission_root: Path) -> Path:
    starter = submission_root / "starter_project"
    return starter if starter.exists() else submission_root


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def compare_row(sub_row: dict[str, str], ref_row: dict[str, str]) -> bool:
    for key, tol in ROW_FLOAT_TOLS.items():
        sval = to_float(sub_row.get(key, ""))
        rval = to_float(ref_row.get(key, ""))
        if math.isnan(sval) or math.isnan(rval) or abs(sval - rval) > tol:
            return False
    for key in ("case_id", "legacy_vina_rmsd_A", "beats_legacy_vina", "success_flag"):
        if sub_row.get(key) != ref_row.get(key):
            return False
    return True


def compare_summary(sub_summary: dict, ref_summary: dict) -> tuple[bool, list[str]]:
    notes: list[str] = []
    for key in (
        "n_cases",
        "success_count",
        "legacy_success_count",
        "cases_beating_legacy_vina",
        "recommended_engine",
    ):
        if sub_summary.get(key) != ref_summary.get(key):
            notes.append(f"summary mismatch: {key}")
    for key, tol in SUMMARY_FLOAT_TOLS.items():
        try:
            sub_value = float(sub_summary.get(key))
            ref_value = float(ref_summary.get(key))
        except (TypeError, ValueError):
            notes.append(f"summary mismatch: {key}")
            continue
        if not math.isfinite(sub_value) or not math.isfinite(ref_value) or abs(sub_value - ref_value) > tol:
            notes.append(f"summary mismatch: {key}")
    return (not notes, notes)


def run_benchmark(project_root: Path, manifest: Path, baseline: Path, output_dir: Path) -> tuple[bool, str | None]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    command = [
        sys.executable,
        "-m",
        "benchmark.run",
        "--manifest",
        str(manifest),
        "--baseline",
        str(baseline),
        "--output-dir",
        str(output_dir),
    ]
    try:
        subprocess.run(command, cwd=project_root, env=env, check=True, capture_output=True, text=True)
        return True, None
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr or exc.stdout or str(exc)


def artifact_score(project_root: Path, visible_dir: Path, hidden_dir: Path) -> tuple[int, list[str]]:
    score = 0
    notes: list[str] = []
    required = [
        project_root / "benchmark" / "run.py",
        project_root / "benchmark" / "report.py",
        project_root / "requirements.txt",
        visible_dir / "gnina_pose_metrics.csv",
        visible_dir / "benchmark_summary.json",
        hidden_dir / "gnina_pose_metrics.csv",
    ]
    present = sum(1 for path in required if path.exists())
    if present == len(required):
        score += 6
    notes.append(f"artifact presence: {present}/{len(required)}")

    visible_poses = len(list((visible_dir / "poses").glob("*_docked.sdf.gz")))
    hidden_poses = len(list((hidden_dir / "poses").glob("*_docked.sdf.gz")))
    visible_logs = len(list((visible_dir / "logs").glob("*.log")))
    hidden_logs = len(list((hidden_dir / "logs").glob("*.log")))
    if visible_poses == 8 and hidden_poses == 7 and visible_logs == 8 and hidden_logs == 7:
        score += 4
    notes.append(f"pose/log outputs: visible={visible_poses}/{visible_logs} hidden={hidden_poses}/{hidden_logs}")
    return score, notes


def evaluate_rows(sub_rows: list[dict[str, str]], ref_rows: list[dict[str, str]], row_points: int) -> tuple[int, int]:
    sub_ids = [row.get("case_id", "") for row in sub_rows]
    ref_ids = [row["case_id"] for row in ref_rows]
    if len(sub_ids) != len(set(sub_ids)) or set(sub_ids) != set(ref_ids):
        return 0, 0
    ref_by_id = {row["case_id"]: row for row in ref_rows}
    sub_by_id = {row["case_id"]: row for row in sub_rows}
    matched = 0
    score = 0
    for case_id in sorted(ref_by_id):
        if compare_row(sub_by_id[case_id], ref_by_id[case_id]):
            matched += 1
            score += row_points
    return score, matched


def memo_score(path: Path) -> tuple[int, list[str]]:
    if not path.exists():
        return 0, ["memo missing"]
    text = path.read_text(encoding="utf-8")
    found = [token for token in MEMO_TOKENS if token in text]
    improving_cases = [
        case_id
        for case_id in ("1B38", "1B39", "1CKP", "1DM2", "1E1V", "1E1X", "1FIN")
        if case_id in text
    ]
    missing_cases = [case_id for case_id in MEMO_REQUIRED_CASES if case_id not in text]
    score = 5 if len(found) == len(MEMO_TOKENS) and improving_cases and not missing_cases else 0
    return score, [
        f"memo tokens: {len(found)}/{len(MEMO_TOKENS)}",
        f"memo improving cases mentioned: {len(improving_cases)}",
        f"memo required review cases missing: {','.join(missing_cases) if missing_cases else 'none'}",
    ]


def rewrite_hidden_manifest(reference_root: Path, destination: Path) -> None:
    source = reference_root / "evaluator_only" / "hidden_manifest.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    hidden_cases = reference_root / "evaluator_only" / "hidden_cases"
    for case in data["cases"]:
        case_id = case["case_id"]
        case["protein_path"] = str(hidden_cases / f"{case_id}_protein.pdb")
        case["ligand_path"] = str(hidden_cases / f"{case_id}_ligand.sdf")
    destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def score_submission(submission_root: Path, reference_root: Path) -> dict:
    project_root = resolve_project_root(submission_root.resolve())
    reference_root = reference_root.resolve()

    with tempfile.TemporaryDirectory(prefix="gnina-migration-eval-") as tmp:
        tmp_root = Path(tmp)
        visible_dir = tmp_root / "visible_run" / "visible"
        hidden_dir = tmp_root / "hidden_run" / "hidden"
        hidden_manifest = tmp_root / "hidden_manifest.json"
        rewrite_hidden_manifest(reference_root, hidden_manifest)

        visible_ok, visible_error = run_benchmark(
            project_root,
            project_root / "inputs" / "visible_manifest.json",
            project_root / "inputs" / "baseline" / "vina_visible_reference.csv",
            visible_dir,
        )
        hidden_ok, hidden_error = run_benchmark(
            project_root,
            hidden_manifest,
            reference_root / "evaluator_only" / "vina_hidden_reference.csv",
            hidden_dir,
        )

        ref_visible_rows = read_csv(reference_root / "reference_outputs" / "visible" / "gnina_pose_metrics.csv")
        ref_hidden_rows = read_csv(reference_root / "reference_outputs" / "hidden" / "gnina_pose_metrics.csv")
        ref_visible_summary = json.loads(
            (reference_root / "reference_outputs" / "visible" / "benchmark_summary.json").read_text(encoding="utf-8")
        )

        visible_rows = read_csv(visible_dir / "gnina_pose_metrics.csv") if visible_ok and (visible_dir / "gnina_pose_metrics.csv").exists() else []
        hidden_rows = read_csv(hidden_dir / "gnina_pose_metrics.csv") if hidden_ok and (hidden_dir / "gnina_pose_metrics.csv").exists() else []
        visible_summary = (
            json.loads((visible_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
            if visible_ok and (visible_dir / "benchmark_summary.json").exists()
            else {}
        )

        artifact_points, artifact_notes = artifact_score(project_root, visible_dir, hidden_dir)
        visible_points, visible_matched = evaluate_rows(visible_rows, ref_visible_rows, VISIBLE_ROW_POINTS)
        hidden_points, hidden_matched = evaluate_rows(hidden_rows, ref_hidden_rows, HIDDEN_ROW_POINTS)
        summary_ok, summary_notes = compare_summary(visible_summary, ref_visible_summary) if visible_summary else (False, ["visible summary missing"])
        summary_points = 10 if summary_ok else 0
        memo_points, memo_notes = memo_score(visible_dir.parent / "reports" / "benchmark_migration_memo.md")

        total = artifact_points + visible_points + hidden_points + summary_points + memo_points
        passed = total >= 80 and visible_matched == 8 and hidden_matched >= 6
        return {
            "score": total / 100.0,
            "points": total,
            "passed": passed,
            "breakdown": {
                "artifact_completeness": artifact_points,
                "visible_rows": visible_points,
                "hidden_rows": hidden_points,
                "visible_summary": summary_points,
                "memo": memo_points,
            },
            "matched_rows": {"visible": visible_matched, "hidden": hidden_matched},
            "notes": artifact_notes
            + ([] if visible_ok else [f"visible run failed: {visible_error}"])
            + ([] if hidden_ok else [f"hidden run failed: {hidden_error}"])
            + summary_notes
            + memo_notes,
        }


def main() -> None:
    args = parse_args()
    result = score_submission(args.submission_root, args.reference_root)
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
