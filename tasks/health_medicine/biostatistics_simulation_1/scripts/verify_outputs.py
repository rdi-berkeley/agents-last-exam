#!/usr/bin/env python
from __future__ import annotations

"""Deterministic verifier for the CHW 1999 replication task."""

import argparse
import csv
import json
import math
from pathlib import Path


TYPE1_TOL = 0.003
TYPE1_CEIL = 0.01
POWER_TOL = 0.04
POWER_CEIL = 0.10
TRIAL_SUMMARY_COLUMNS = [
    "table",
    "scenario",
    "schedule",
    "t_L",
    "variant",
    "n_rep",
    "rejections",
    "estimate",
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def _parse_float(value: str) -> float:
    return float(str(value).strip())


def _graduated_score(diff: float, full_tol: float, zero_tol: float) -> float:
    d = abs(diff)
    if d <= full_tol:
        return 1.0
    if d >= zero_tol:
        return 0.0
    return 1.0 - (d - full_tol) / (zero_tol - full_tol)


def _is_finite(value: str) -> bool:
    try:
        return math.isfinite(_parse_float(value))
    except Exception:
        return False


def _subset_matches(expected, actual, path: str, reasons: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            reasons.append(f"{path} expected object")
            return
        for key, expected_value in expected.items():
            if key not in actual:
                reasons.append(f"{path}.{key} missing")
                continue
            _subset_matches(expected_value, actual[key], f"{path}.{key}", reasons)
        return
    if isinstance(expected, list):
        if actual != expected:
            reasons.append(f"{path} expected {expected!r}, found {actual!r}")
        return
    if actual != expected:
        reasons.append(f"{path} expected {expected!r}, found {actual!r}")


def _key_map(rows: list[dict[str, str]], key_fields: list[str], table_name: str, reasons: list[str]) -> dict[tuple[str, ...], dict[str, str]]:
    mapping: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(field, "").strip() for field in key_fields)
        if key in mapping:
            reasons.append(f"{table_name} duplicate key {key}")
        mapping[key] = row
    return mapping


def _require_columns(
    actual_fields: list[str],
    expected_fields: list[str],
    table_name: str,
    reasons: list[str],
) -> None:
    if actual_fields != expected_fields:
        reasons.append(f"{table_name} columns expected {expected_fields}, found {actual_fields}")


def _check_nonempty_file(path: Path, label: str, reasons: list[str]) -> None:
    if not path.exists():
        reasons.append(f"missing {label}")
        return
    if path.stat().st_size <= 0:
        reasons.append(f"empty {label}")


def _validate_boundary_csv(output_path: Path, helper_path: Path, reasons: list[str], details: dict) -> None:
    fields, rows = _load_csv(output_path)
    expected_fields = ["schedule", "look", "information_time", "z_boundary"]
    _require_columns(fields, expected_fields, "critical_values_used.csv", reasons)
    helper = _load_json(helper_path)
    expected_rows: dict[tuple[str, str], tuple[float, float]] = {}
    for schedule, payload in helper["schedules"].items():
        for idx, (info_time, z_boundary) in enumerate(
            zip(payload["information_times"], payload["z_boundaries"]),
            start=1,
        ):
            expected_rows[(schedule, str(idx))] = (float(info_time), float(z_boundary))

    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.get("schedule", "").strip(), row.get("look", "").strip())
        seen.add(key)
        if key not in expected_rows:
            reasons.append(f"unexpected critical-value row {key}")
            continue
        if not _is_finite(row["information_time"]) or not _is_finite(row["z_boundary"]):
            reasons.append(f"non-finite critical-value row {key}")
            continue
        info_time, z_boundary = expected_rows[key]
        if abs(_parse_float(row["information_time"]) - info_time) > 1e-6:
            reasons.append(f"information_time mismatch for {key}")
        if abs(_parse_float(row["z_boundary"]) - z_boundary) > 1e-6:
            reasons.append(f"z_boundary mismatch for {key}")

    if seen != set(expected_rows):
        reasons.append("critical_values_used.csv row coverage mismatch")
    details["critical_values_rows"] = len(rows)


def _validate_trial_summary(path: Path, reasons: list[str], details: dict) -> None:
    fields, rows = _load_csv(path)
    missing = [field for field in TRIAL_SUMMARY_COLUMNS if field not in fields]
    if missing:
        reasons.append(f"trial_level_summary.csv missing columns: {missing}")
        return
    if not rows:
        reasons.append("trial_level_summary.csv has no data rows")
        return
    for field in ("n_rep", "rejections", "estimate"):
        for row in rows:
            if not _is_finite(row[field]):
                reasons.append(f"trial_level_summary.csv has non-finite {field}")
                break
    details["trial_summary_rows"] = len(rows)


def _validate_tables(
    agent_path: Path,
    reference_path: Path,
    spec: dict,
    table_name: str,
    key_fields: list[str],
    reasons: list[str],
    details: dict,
) -> list[float]:
    agent_fields, agent_rows = _load_csv(agent_path)
    ref_fields, ref_rows = _load_csv(reference_path)
    _require_columns(agent_fields, ref_fields, table_name, reasons)
    if not agent_rows:
        reasons.append(f"{table_name} has no rows")
        return []

    agent_map = _key_map(agent_rows, key_fields, table_name, reasons)
    ref_map = _key_map(ref_rows, key_fields, f"{table_name} reference", reasons)
    if set(agent_map) != set(ref_map):
        reasons.append(f"{table_name} row coverage mismatch")

    for row in agent_rows:
        for field in ("type1", "power", "mcse_type1", "mcse_power", "n_rep_type1", "n_rep_power"):
            if field not in row or not _is_finite(row[field]):
                reasons.append(f"{table_name} has non-finite or missing {field}")
                break

    expected_type1_n = str(spec["monte_carlo"]["n_rep_type1"])
    expected_power_n = str(spec["monte_carlo"]["n_rep_power"])
    for row in agent_rows:
        if row.get("n_rep_type1", "").strip() != expected_type1_n:
            reasons.append(f"{table_name} wrong n_rep_type1 for key {[row.get(k, '') for k in key_fields]}")
            break
        if row.get("n_rep_power", "").strip() != expected_power_n:
            reasons.append(f"{table_name} wrong n_rep_power for key {[row.get(k, '') for k in key_fields]}")
            break

    row_scores: list[float] = []
    for key, ref_row in ref_map.items():
        agent_row = agent_map.get(key)
        if agent_row is None:
            row_scores.append(0.0)
            continue
        type1_diff = _parse_float(agent_row["type1"]) - _parse_float(ref_row["type1"])
        power_diff = _parse_float(agent_row["power"]) - _parse_float(ref_row["power"])
        t1 = _graduated_score(type1_diff, TYPE1_TOL, TYPE1_CEIL)
        pw = _graduated_score(power_diff, POWER_TOL, POWER_CEIL)
        row_scores.append((t1 + pw) / 2)
        if abs(type1_diff) > TYPE1_TOL:
            reasons.append(f"{table_name} type1 outside full-credit tolerance for {key} (diff={type1_diff:+.6f}, score={t1:.3f})")
        if abs(power_diff) > POWER_TOL:
            reasons.append(f"{table_name} power outside full-credit tolerance for {key} (diff={power_diff:+.6f}, score={pw:.3f})")

    details[f"{table_name}_rows"] = len(agent_rows)
    details[f"{table_name}_row_scores"] = row_scores
    return row_scores


def evaluate(output_dir: Path, input_dir: Path, reference_dir: Path) -> dict:
    reasons: list[str] = []
    details: dict[str, object] = {}

    required_files = {
        "table_A1_replication.csv": output_dir / "table_A1_replication.csv",
        "table_A2_replication.csv": output_dir / "table_A2_replication.csv",
        "simulation_config_used.json": output_dir / "simulation_config_used.json",
        "critical_values_used.csv": output_dir / "critical_values_used.csv",
        "run_log.txt": output_dir / "run_log.txt",
        "trial_level_summary.csv": output_dir / "trial_level_summary.csv",
    }
    for label, path in required_files.items():
        _check_nonempty_file(path, label, reasons)
    if reasons:
        return {"score": 0.0, "passed": False, "reasons": reasons, "details": details}

    spec = _load_json(input_dir / "task_spec.json")
    helper = input_dir / "boundary_helper.json"
    config = _load_json(required_files["simulation_config_used.json"])
    required_subset = {
        "task_id": spec["task_id"],
        "alpha_one_sided": spec["alpha_one_sided"],
        "sigma2": spec["sigma2"],
        "planned_N_per_group": spec["planned_N_per_group"],
        "planned_effect_delta": spec["planned_effect_delta"],
        "power_true_effect_Delta": spec["power_true_effect_Delta"],
        "look_schedules": spec["look_schedules"],
        "A1": spec["A1"],
        "A2": spec["A2"],
        "monte_carlo": spec["monte_carlo"],
    }
    _subset_matches(required_subset, config, "simulation_config_used", reasons)
    max_cores_used = config.get("max_cores_used")
    if max_cores_used is None:
        reasons.append("simulation_config_used.max_cores_used missing")
    else:
        try:
            if int(max_cores_used) > int(spec["monte_carlo"]["parallel_workers_max"]):
                reasons.append("max_cores_used exceeds parallel_workers_max")
        except Exception:
            reasons.append("max_cores_used is not an integer")
    details["max_cores_used"] = max_cores_used

    _validate_boundary_csv(required_files["critical_values_used.csv"], helper, reasons, details)

    log_text = required_files["run_log.txt"].read_text(encoding="utf-8", errors="replace")
    if "sessionInfo" not in log_text and "R version" not in log_text:
        reasons.append("run_log.txt missing R runtime info")

    _validate_trial_summary(required_files["trial_level_summary.csv"], reasons, details)

    a1_scores = _validate_tables(
        required_files["table_A1_replication.csv"],
        reference_dir / "table_A1_replication.csv",
        spec,
        "table_A1_replication.csv",
        ["schedule", "t_L", "policy"],
        reasons,
        details,
    )
    a2_scores = _validate_tables(
        required_files["table_A2_replication.csv"],
        reference_dir / "table_A2_replication.csv",
        spec,
        "table_A2_replication.csv",
        ["schedule", "t_L", "test"],
        reasons,
        details,
    )

    all_scores = a1_scores + a2_scores
    if not all_scores:
        score = 0.0
    else:
        score = sum(all_scores) / len(all_scores)

    score = round(score, 4)
    passed = score >= 1.0
    return {
        "score": score,
        "passed": passed,
        "reasons": reasons,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    result = evaluate(Path(args.output_dir), Path(args.input_dir), Path(args.reference_dir))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
