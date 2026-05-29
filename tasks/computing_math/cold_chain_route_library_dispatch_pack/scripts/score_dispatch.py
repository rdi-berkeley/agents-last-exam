#!/usr/bin/env python
"""Score cold-chain dispatch candidate projects.

This verifier intentionally depends only on the Python standard library. It
executes the candidate project through the staged task runtime wrapper so the
Pyomo/PySCIPOpt dependency boundary remains explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path


REQUIRED_ARTIFACTS = [
    "dispatch/vrptw_model.py",
    "dispatch/cli.py",
    "dispatch/scenario_io.py",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_candidate_project(output_dir: Path) -> Path | None:
    direct = output_dir / "dispatch" / "cli.py"
    nested = output_dir / "dispatch_project" / "dispatch" / "cli.py"
    if direct.exists():
        return output_dir
    if nested.exists():
        return output_dir / "dispatch_project"
    return None


def copy_candidate_project(candidate_dir: Path, work_dir: Path) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    symlinks = [str(path.relative_to(candidate_dir)) for path in candidate_dir.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError("candidate project contains symlinks: " + ", ".join(symlinks[:20]))
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".runtime_state", "outputs")
    shutil.copytree(candidate_dir, work_dir, ignore=ignore)
    return work_dir


def route_map_from_csv(path: Path) -> dict[str, list[str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"scenario_id", "vehicle_id", "stop_sequence", "customer_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path.name}: missing required route columns")
        rows.extend(reader)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        vehicle_id = (row.get("vehicle_id") or "").strip()
        customer_id = (row.get("customer_id") or "").strip()
        if not vehicle_id or not customer_id:
            continue
        grouped.setdefault(vehicle_id, []).append(row)

    route_map: dict[str, list[str]] = {}
    for vehicle_id, items in grouped.items():
        try:
            ordered = sorted(items, key=lambda item: int(item["stop_sequence"]))
        except Exception as exc:
            raise ValueError(f"{path.name}: invalid stop_sequence") from exc
        route_map[vehicle_id] = [item["customer_id"].strip() for item in ordered]
    return route_map


def evaluate_route_rows(scenario: dict, route_map: dict[str, list[str]]) -> tuple[float, float, list[str]]:
    customers = {item["id"]: item for item in scenario["customers"]}
    vehicles = {item["id"]: item for item in scenario["vehicles"]}
    depot_id = scenario["depot"]["id"]
    served: dict[str, int] = {}
    objective = 0.0
    travel_minutes = 0.0
    violations: list[str] = []

    for vehicle_id, route in route_map.items():
        vehicle = vehicles.get(vehicle_id)
        if vehicle is None:
            violations.append(f"unknown_vehicle:{vehicle_id}")
            continue

        known_customers = [customer_id for customer_id in route if customer_id in customers]
        for customer_id in route:
            if customer_id not in customers:
                violations.append(f"{vehicle_id}:unknown_customer:{customer_id}")

        total_demand = sum(customers[customer_id]["demand_bins"] for customer_id in known_customers)
        if total_demand > vehicle["capacity_bins"]:
            violations.append(f"{vehicle_id}:capacity")

        current = depot_id
        clock = vehicle["start_minute"]
        route_travel = 0.0

        for customer_id in route:
            customer = customers.get(customer_id)
            if customer is None:
                continue
            served[customer_id] = served.get(customer_id, 0) + 1
            if customer["temperature_class"] == "frozen" and not vehicle["can_carry_frozen"]:
                violations.append(f"{vehicle_id}:temperature:{customer_id}")

            try:
                leg = scenario["travel_minutes"][current][customer_id]
            except KeyError:
                violations.append(f"{vehicle_id}:missing_leg:{current}:{customer_id}")
                current = customer_id
                continue
            route_travel += leg
            travel_minutes += leg
            clock += leg
            if clock < customer["open_minute"]:
                clock = customer["open_minute"]
            if clock > customer["close_minute"]:
                violations.append(f"{vehicle_id}:time_window:{customer_id}")
            clock += customer["service_minutes"]
            current = customer_id

        try:
            back_leg = scenario["travel_minutes"][current][depot_id]
        except KeyError:
            violations.append(f"{vehicle_id}:missing_leg:{current}:{depot_id}")
            back_leg = 0
        route_travel += back_leg
        travel_minutes += back_leg
        clock += back_leg
        if clock > vehicle["end_minute"]:
            violations.append(f"{vehicle_id}:end_minute")
        if clock - vehicle["start_minute"] > vehicle["max_route_minutes"]:
            violations.append(f"{vehicle_id}:max_route_minutes")

        objective += vehicle["fixed_cost"] + route_travel + 0.05 * clock

    for customer_id in customers:
        count = served.get(customer_id, 0)
        if count != 1:
            violations.append(f"coverage:{customer_id}:{count}")

    return round(objective, 2), round(travel_minutes, 2), sorted(set(violations))


def run_cli(candidate_dir: Path, scenario_dir: Path, output_dir: Path, python_wrapper: Path) -> dict:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            str(python_wrapper),
            "-m",
            "dispatch.cli",
            "--scenario-dir",
            str(scenario_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(candidate_dir),
        text=True,
        capture_output=True,
        env=env,
        timeout=180,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def score_scenarios(output_dir: Path, scenario_dir: Path, targets: dict, points_per_pass: float) -> tuple[float, list[dict]]:
    reports: list[dict] = []
    points = 0.0

    for scenario_path in sorted(scenario_dir.glob("*.json")):
        scenario = load_json(scenario_path)
        scenario_id = scenario["scenario_id"]
        route_file = output_dir / f"{scenario_path.stem}_routes.csv"
        if not route_file.exists():
            reports.append({"scenario_id": scenario_id, "passed": False, "reason": "missing route file"})
            continue

        try:
            objective, travel_minutes, violations = evaluate_route_rows(scenario, route_map_from_csv(route_file))
        except Exception as exc:
            reports.append({"scenario_id": scenario_id, "passed": False, "reason": f"invalid route csv: {exc}"})
            continue

        target = targets.get(scenario_id)
        if not target:
            reports.append({"scenario_id": scenario_id, "passed": False, "reason": "missing target"})
            continue

        gap = round(objective - float(target["objective"]), 2)
        passed = not violations and gap <= 1.0
        if passed:
            points += points_per_pass
        reports.append(
            {
                "scenario_id": scenario_id,
                "objective": objective,
                "target_objective": target["objective"],
                "gap": gap,
                "travel_minutes": travel_minutes,
                "violations": violations,
                "passed": passed,
            }
        )

    return points, reports


def score_planner_note(candidate_dir: Path) -> tuple[float, list[str]]:
    note_path = candidate_dir / "docs" / "planner_note.md"
    if not note_path.exists():
        return 0.0, ["docs/planner_note.md missing"]
    text = note_path.read_text(encoding="utf-8", errors="replace").lower().replace("-", " ").replace("_", " ")
    required_terms = ["vehicle", "cost", "time window", "base day", "bridge delay"]
    missing = [term for term in required_terms if term not in text]
    return (10.0 if not missing else 0.0), missing


def score_candidate(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    candidate_src = detect_candidate_project(output_dir)
    report = {
        "candidate_output_dir": str(output_dir),
        "candidate_project": str(candidate_src) if candidate_src else None,
        "artifact_score": 0.0,
        "planner_note_score": 0.0,
        "visible_score": 0.0,
        "hidden_score": 0.0,
        "score_points": 0.0,
        "score": 0.0,
        "passed": False,
        "notes": [],
        "visible_reports": [],
        "hidden_reports": [],
    }
    if candidate_src is None:
        report["notes"].append("candidate project missing: expected dispatch/cli.py or dispatch_project/dispatch/cli.py")
        return report

    eval_root = Path(args.work_dir)
    if eval_root.exists():
        shutil.rmtree(eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    candidate_dir = copy_candidate_project(candidate_src, eval_root / "candidate_project")

    existing = sum(1 for rel in REQUIRED_ARTIFACTS if (candidate_dir / rel).exists())
    report["artifact_score"] = round(10.0 * existing / len(REQUIRED_ARTIFACTS), 1)

    targets = load_json(Path(args.targets))
    visible_dir = Path(args.visible_dir)
    hidden_dir = Path(args.hidden_dir)
    python_wrapper = Path(args.python_wrapper)

    visible_output = eval_root / "visible_outputs"
    visible_run = run_cli(candidate_dir, visible_dir, visible_output, python_wrapper)
    if visible_run["returncode"] != 0:
        report["notes"].append("visible run failed")
        report["notes"].append((visible_run["stderr"] or visible_run["stdout"])[-4000:])
        planner_score, planner_missing = score_planner_note(candidate_dir)
        report["planner_note_score"] = planner_score
        if planner_missing:
            report["notes"].append(f"planner note missing terms: {planner_missing}")
        report["score_points"] = round(report["artifact_score"] + report["planner_note_score"], 1)
        report["score"] = round(report["score_points"] / 100.0, 4)
        return report

    visible_targets = {key: value for key, value in targets.items() if key.startswith("visible::")}
    hidden_targets = {key: value for key, value in targets.items() if key.startswith("hidden::")}
    report["visible_score"], report["visible_reports"] = score_scenarios(
        visible_output, visible_dir, visible_targets, 10.0
    )

    hidden_output = eval_root / "hidden_outputs"
    hidden_run = run_cli(candidate_dir, hidden_dir, hidden_output, python_wrapper)
    if hidden_run["returncode"] != 0:
        report["notes"].append("hidden run failed")
        report["notes"].append((hidden_run["stderr"] or hidden_run["stdout"])[-4000:])
    else:
        report["hidden_score"], report["hidden_reports"] = score_scenarios(
            hidden_output, hidden_dir, hidden_targets, 10.0
        )

    planner_score, planner_missing = score_planner_note(candidate_dir)
    report["planner_note_score"] = planner_score
    if planner_missing:
        report["notes"].append(f"planner note missing terms: {planner_missing}")

    report["score_points"] = round(
        report["artifact_score"] + report["planner_note_score"] + report["visible_score"] + report["hidden_score"],
        1,
    )
    visible_gate = bool(report["visible_reports"]) and all(item.get("passed") for item in report["visible_reports"])
    hidden_gate = sum(1 for item in report["hidden_reports"] if item.get("passed")) >= 3
    threshold_gate = report["score_points"] >= 80.0
    report["passed"] = bool(visible_gate and hidden_gate and threshold_gate)
    normalized = report["score_points"] / 100.0
    if not report["passed"]:
        normalized = min(normalized, 0.79)
    report["score"] = round(max(0.0, min(normalized, 1.0)), 4)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--visible-dir", required=True)
    parser.add_argument("--hidden-dir", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--python-wrapper", required=True)
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args()


def main() -> None:
    report = score_candidate(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
