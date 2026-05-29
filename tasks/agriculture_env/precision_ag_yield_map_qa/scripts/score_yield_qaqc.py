from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_fieldnames(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return csv.DictReader(handle).fieldnames or []


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def grid_rmse(actual_rows: list[dict[str, str]], expected_rows: list[dict[str, str]]) -> float:
    actual = {
        (row["field_id"], row["cell_x_m"], row["cell_y_m"]): float(row["mean_yield_bu_ac"])
        for row in actual_rows
    }
    expected = {
        (row["field_id"], row["cell_x_m"], row["cell_y_m"]): float(row["mean_yield_bu_ac"])
        for row in expected_rows
    }
    if set(actual) != set(expected):
        return 9999.0
    return math.sqrt(mean([(actual[key] - expected[key]) ** 2 for key in expected]))


def compare_field(actual_dir: Path, expected_dir: Path) -> tuple[float, list[dict[str, object]]]:
    score = 0.0
    checks: list[dict[str, object]] = []
    required = [
        "cleaned_points.csv",
        "cleaned_points.geojson",
        "rejected_points.csv",
        "yield_grid_10m.csv",
        "strip_summary.json",
        "filter_audit.json",
    ]
    missing = [name for name in required if not (actual_dir / name).exists()]
    if missing:
        return 0.0, [{"check": "required_files", "passed": False, "detail": ", ".join(missing)}]

    actual_clean = read_csv(actual_dir / "cleaned_points.csv")
    expected_clean = read_csv(expected_dir / "cleaned_points.csv")
    actual_reject = read_csv(actual_dir / "rejected_points.csv")
    expected_reject = read_csv(expected_dir / "rejected_points.csv")
    actual_grid = read_csv(actual_dir / "yield_grid_10m.csv")
    expected_grid = read_csv(expected_dir / "yield_grid_10m.csv")
    actual_strip = load_json(actual_dir / "strip_summary.json")
    expected_strip = load_json(expected_dir / "strip_summary.json")
    actual_audit = load_json(actual_dir / "filter_audit.json")
    expected_audit = load_json(expected_dir / "filter_audit.json")
    actual_geo = load_json(actual_dir / "cleaned_points.geojson")

    def add(name: str, passed: bool, weight: float, detail: str = "") -> None:
        nonlocal score
        checks.append({"check": name, "passed": passed, "weight": weight, "detail": detail})
        if passed:
            score += weight

    add("retained_point_count", abs(len(actual_clean) - len(expected_clean)) <= 2, 2.5)
    add("rejected_point_count", abs(len(actual_reject) - len(expected_reject)) <= 2, 2.0)
    for reason, expected_count in expected_audit["removal_counts"].items():
        actual_count = actual_audit.get("removal_counts", {}).get(reason, 0)
        add(f"reject_reason_{reason}", abs(actual_count - expected_count) <= 2, 1.0)

    expected_reject_counts: dict[str, int] = {}
    actual_reject_counts: dict[str, int] = {}
    for row in expected_reject:
        expected_reject_counts[row["reject_reason"]] = expected_reject_counts.get(row["reject_reason"], 0) + 1
    for row in actual_reject:
        reason = row.get("reject_reason", "")
        actual_reject_counts[reason] = actual_reject_counts.get(reason, 0) + 1
    export_counts_ok = all(
        abs(actual_reject_counts.get(reason, 0) - expected_count) <= 2
        for reason, expected_count in expected_reject_counts.items()
    )
    add("rejected_export_reason_counts", export_counts_ok, 2.0)

    add(
        "cleaned_schema",
        csv_fieldnames(actual_dir / "cleaned_points.csv") == csv_fieldnames(expected_dir / "cleaned_points.csv"),
        1.0,
    )
    add(
        "rejected_schema",
        csv_fieldnames(actual_dir / "rejected_points.csv") == csv_fieldnames(expected_dir / "rejected_points.csv"),
        1.0,
    )
    add(
        "grid_schema",
        csv_fieldnames(actual_dir / "yield_grid_10m.csv") == csv_fieldnames(expected_dir / "yield_grid_10m.csv"),
        1.0,
    )

    expected_dry = sum(float(row["calibrated_dry_mass_kg"]) for row in expected_clean)
    actual_dry = sum(float(row["calibrated_dry_mass_kg"]) for row in actual_clean)
    add("calibrated_dry_mass", abs(actual_dry - expected_dry) / expected_dry <= 0.0035, 3.0)
    add(
        "mean_moisture",
        abs(
            mean([float(row["moisture_pct"]) for row in actual_clean])
            - mean([float(row["moisture_pct"]) for row in expected_clean])
        )
        <= 0.15,
        2.0,
    )
    add(
        "mean_yield",
        abs(
            mean([float(row["calibrated_yield_bu_ac"]) for row in actual_clean])
            - mean([float(row["calibrated_yield_bu_ac"]) for row in expected_clean])
        )
        <= 0.75,
        3.0,
    )
    add("grid_rmse", grid_rmse(actual_grid, expected_grid) <= 1.25, 4.0)

    expected_strips = {row["strip_id"]: row for row in expected_strip["strips"]}
    actual_strips = {row["strip_id"]: row for row in actual_strip.get("strips", [])}
    strip_ok = set(actual_strips) == set(expected_strips)
    for row in actual_strip.get("strips", []):
        exp = expected_strips.get(row.get("strip_id"))
        if not exp or abs(float(row["mean_yield_bu_ac"]) - float(exp["mean_yield_bu_ac"])) > 1.0:
            strip_ok = False
    add("strip_means", strip_ok, 2.5)

    audit_keys = {
        "grain_flow_delay_s",
        "start_pass_delay_s",
        "end_pass_delay_s",
        "speed_min_mps",
        "speed_max_mps",
        "swath_min_m",
        "yield_min_bu_ac",
        "yield_max_bu_ac",
        "calibration_factor",
    }
    add("audit_metadata", audit_keys.issubset(actual_audit) and actual_audit.get("crs") == "EPSG:32615", 2.0)

    audit_values_ok = True
    for key in [
        "grain_flow_delay_s",
        "start_pass_delay_s",
        "end_pass_delay_s",
        "speed_min_mps",
        "speed_max_mps",
        "swath_min_m",
        "yield_min_bu_ac",
        "yield_max_bu_ac",
        "calibration_factor",
        "accepted_points",
        "rejected_points",
    ]:
        if actual_audit.get(key) != expected_audit.get(key):
            audit_values_ok = False
            break
    add("audit_values", audit_values_ok, 2.0)
    add("geojson_metadata", actual_geo.get("crs", {}).get("properties", {}).get("name") == "EPSG:32615", 1.0)
    return score, checks


def score_outputs(actual_root: Path, expected_root: Path, total_weight: float) -> tuple[float, list[dict[str, object]]]:
    field_dirs = sorted(path for path in expected_root.iterdir() if path.is_dir())
    if not field_dirs:
        return 0.0, []
    per_field_target = total_weight / len(field_dirs)
    total = 0.0
    reports = []
    for expected_dir in field_dirs:
        actual_dir = actual_root / expected_dir.name
        raw_score, checks = compare_field(actual_dir, expected_dir)
        max_score = sum(float(check.get("weight", 0.0)) for check in checks) or 1.0
        scaled = round(raw_score * per_field_target / max_score, 3)
        total += scaled
        reports.append({"field_id": expected_dir.name, "score": scaled, "checks": checks})
    return round(total, 3), reports


def root_artifacts(output_dir: Path) -> tuple[float, list[dict[str, object]]]:
    score = 0.0
    checks: list[dict[str, object]] = []
    for name, weight in [
        ("calibration_summary.json", 2),
        ("consultant_memo.md", 1),
        ("qgis_processing_manifest.json", 2),
        ("run_manifest.json", 2),
    ]:
        ok = (output_dir / name).exists() and (output_dir / name).stat().st_size > 0
        checks.append({"check": name, "passed": ok, "weight": weight})
        if ok:
            score += weight
    if (output_dir / "run_manifest.json").exists():
        manifest = load_json(output_dir / "run_manifest.json")
        ok = manifest.get("crs") == "EPSG:32615" and int(manifest.get("grid_size_m", 0)) == 10
        checks.append({"check": "manifest_contract", "passed": ok, "weight": 1})
        if ok:
            score += 1
    if (output_dir / "qgis_processing_manifest.json").exists() and (output_dir / "calibration_summary.json").exists():
        qgis_model = load_json(output_dir / "qgis_processing_manifest.json")
        summary = load_json(output_dir / "calibration_summary.json")
        field_ids = [field["field_id"] for field in summary.get("fields", [])]
        ok = (
            qgis_model.get("type") == "qgis_processing_manifest"
            and qgis_model.get("qgis_ltr") == "3.44.9"
            and qgis_model.get("crs") == "EPSG:32615"
            and int(qgis_model.get("grid_size_m", 0)) == 10
            and set(qgis_model.get("fields", [])) == set(field_ids)
            and {
                "clip_to_field_boundary",
                "aggregate_points_to_10m_grid",
                "export_csv_geojson_json",
            }.issubset(set(qgis_model.get("steps", [])))
        )
        checks.append({"check": "qgis_model_contract", "passed": ok, "weight": 1})
        if ok:
            score += 1
    if (output_dir / "consultant_memo.md").exists() and (output_dir / "calibration_summary.json").exists():
        memo = (output_dir / "consultant_memo.md").read_text(encoding="utf-8").lower()
        summary = load_json(output_dir / "calibration_summary.json")
        field_ids = [field["field_id"].lower() for field in summary.get("fields", [])]
        ok = "calibration factor" in memo and all(field_id in memo for field_id in field_ids)
        for field in summary.get("fields", []):
            ok = ok and str(field["accepted_points"]) in memo and str(field["rejected_points"]) in memo
        checks.append({"check": "memo_objective_content", "passed": ok, "weight": 1})
        if ok:
            score += 1
    return score, checks


def find_submission_dir(output_dir: Path) -> Path:
    explicit = output_dir / "submission"
    if (explicit / "run_yield_qaqc.py").exists():
        return explicit
    if (output_dir / "run_yield_qaqc.py").exists():
        return output_dir
    raise FileNotFoundError("expected run_yield_qaqc.py in output/submission/ or output/")


def run_cli(submission_dir: Path, input_dir: Path) -> tuple[Path, subprocess.CompletedProcess[str]]:
    out = Path(tempfile.mkdtemp(prefix="yield-qaqc-"))
    result = subprocess.run(
        [
            sys.executable,
            str(submission_dir / "run_yield_qaqc.py"),
            "--input",
            str(input_dir),
            "--output",
            str(out),
        ],
        cwd=submission_dir,
        text=True,
        capture_output=True,
        timeout=60,
    )
    return out, result


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    visible_input = input_dir / "starter_project" / "input"
    hidden_input = reference_dir / "evaluator_only" / "hidden_fields"
    visible_reference = reference_dir / "reference_outputs" / "visible"
    hidden_reference = reference_dir / "reference_outputs" / "hidden"

    report: dict[str, object] = {
        "score": 0.0,
        "raw_score_100": 0.0,
        "passed": False,
        "pass_threshold_100": 80.0,
        "root_artifact_gate": False,
        "visible_score": 0.0,
        "hidden_score": 0.0,
        "artifact_score": 0.0,
        "notes": [],
    }

    try:
        submission_dir = find_submission_dir(output_dir)
    except FileNotFoundError as exc:
        report["notes"] = [str(exc)]
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    visible_output, visible_run = run_cli(submission_dir, visible_input)
    try:
        if visible_run.returncode != 0:
            report["notes"].append("visible run failed")
            report["notes"].append((visible_run.stderr or visible_run.stdout)[-4000:])
        else:
            visible_artifact_score, visible_artifact_checks = root_artifacts(visible_output)
            visible_score, visible_reports = score_outputs(visible_output, visible_reference, 45.0)
            report["visible_artifact_score"] = visible_artifact_score
            report["visible_artifact_checks"] = visible_artifact_checks
            report["visible_score"] = visible_score
            report["visible_reports"] = visible_reports
            report["artifact_score"] = float(report["artifact_score"]) + visible_artifact_score / 2.0
    finally:
        shutil.rmtree(visible_output, ignore_errors=True)

    hidden_output, hidden_run = run_cli(submission_dir, hidden_input)
    try:
        if hidden_run.returncode != 0:
            report["notes"].append("hidden run failed")
            report["notes"].append((hidden_run.stderr or hidden_run.stdout)[-4000:])
        else:
            hidden_artifact_score, hidden_artifact_checks = root_artifacts(hidden_output)
            hidden_score, hidden_reports = score_outputs(hidden_output, hidden_reference, 45.0)
            report["hidden_artifact_score"] = hidden_artifact_score
            report["hidden_artifact_checks"] = hidden_artifact_checks
            report["hidden_score"] = hidden_score
            report["hidden_reports"] = hidden_reports
            report["artifact_score"] = float(report["artifact_score"]) + hidden_artifact_score / 2.0
    finally:
        shutil.rmtree(hidden_output, ignore_errors=True)

    raw_score = round(
        float(report["artifact_score"]) + float(report["visible_score"]) + float(report["hidden_score"]),
        2,
    )
    report["raw_score_100"] = raw_score
    report["score"] = round(raw_score / 100.0, 4)
    report["root_artifact_gate"] = (
        report.get("visible_artifact_score") == 10.0 and report.get("hidden_artifact_score") == 10.0
    )
    report["passed"] = raw_score >= 80.0 and bool(report["root_artifact_gate"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
