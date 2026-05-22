"""Score candidate CVRP solution directories for the selected-instance benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


SELECTED_INSTANCES = ("M-n101-k10", "X-n101-k25", "X-n106-k14")


@dataclass
class InstanceScore:
    name: str
    exists: bool
    feasible: bool
    gap: float | None
    passed: bool
    details: dict


@dataclass
class ScoreResult:
    score: float
    passed_count: int
    per_instance: list[InstanceScore]


def _parse_instance(path: Path) -> dict:
    header: dict[str, str] = {}
    coords: dict[int, tuple[int, int]] = {}
    demands: dict[int, int] = {}
    section = None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF"}:
            section = None if line == "EOF" else line
            continue
        if ":" in line and section is None:
            key, value = [part.strip().strip('"') for part in line.split(":", 1)]
            header[key] = value
            continue
        if section == "NODE_COORD_SECTION":
            node_id, x, y = line.split()
            coords[int(node_id)] = (int(x), int(y))
            continue
        if section == "DEMAND_SECTION":
            node_id, demand = line.split()
            demands[int(node_id)] = int(demand)
            continue

    if header.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise RuntimeError(f"unsupported EDGE_WEIGHT_TYPE: {header.get('EDGE_WEIGHT_TYPE')}")

    dimension = int(header["DIMENSION"])
    return {
        "capacity": int(header["CAPACITY"]),
        "dimension": dimension,
        "depot_coord": coords[1],
        "customer_coords": {node_id - 1: coords[node_id] for node_id in range(2, dimension + 1)},
        "customer_demands": {node_id - 1: demands[node_id] for node_id in range(2, dimension + 1)},
    }


def _parse_solution(path: Path) -> list[list[int]]:
    routes: list[list[int]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("route"):
            if ":" not in line:
                raise ValueError(f"route line missing ':' in {path.name}")
            _, payload = line.split(":", 1)
            routes.append([int(token) for token in payload.strip().split()])
    return routes


def _route_distance(route: list[int], instance: dict) -> int:
    points = [instance["depot_coord"]]
    points.extend(instance["customer_coords"][customer] for customer in route)
    points.append(instance["depot_coord"])

    total = 0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        total += int(math.hypot(x1 - x2, y1 - y2) + 0.5)
    return total


def _score_instance(solution_path: Path, instance_path: Path, bks_path: Path, relative_gap_limit: float) -> InstanceScore:
    if not solution_path.exists():
        return InstanceScore(
            name=instance_path.stem,
            exists=False,
            feasible=False,
            gap=None,
            passed=False,
            details={"error": "missing solution file"},
        )

    try:
        instance = _parse_instance(instance_path)
        routes = _parse_solution(solution_path)
        bks_routes = _parse_solution(bks_path)
    except Exception as exc:
        return InstanceScore(
            name=instance_path.stem,
            exists=True,
            feasible=False,
            gap=None,
            passed=False,
            details={"error": f"parse failure: {exc}"},
        )

    expected_customers = set(instance["customer_coords"].keys())
    seen: list[int] = []
    capacity_ok = True
    valid_customer_ids = True
    total_cost = 0
    route_errors: list[str] = []

    for route_index, route in enumerate(routes, start=1):
        load = 0
        for customer in route:
            if customer not in expected_customers:
                valid_customer_ids = False
                route_errors.append(f"route {route_index} contains invalid customer {customer}")
            else:
                load += instance["customer_demands"][customer]
                seen.append(customer)
        if load > instance["capacity"]:
            capacity_ok = False
            route_errors.append(f"route {route_index} load {load} exceeds capacity {instance['capacity']}")
        if valid_customer_ids:
            total_cost += _route_distance(route, instance)

    missing = sorted(expected_customers - set(seen))
    duplicates = sorted({customer for customer in seen if seen.count(customer) > 1})
    feasible = valid_customer_ids and capacity_ok and not missing and not duplicates

    bks_cost = sum(_route_distance(route, instance) for route in bks_routes)
    gap = None if not feasible else (total_cost - bks_cost) / bks_cost
    passed = feasible and gap is not None and gap <= relative_gap_limit

    return InstanceScore(
        name=instance_path.stem,
        exists=True,
        feasible=feasible,
        gap=gap,
        passed=passed,
        details={
            "total_cost": total_cost if feasible else None,
            "bks_cost": bks_cost,
            "missing": missing,
            "duplicates": duplicates,
            "capacity_ok": capacity_ok,
            "valid_customer_ids": valid_customer_ids,
            "route_errors": route_errors,
            "relative_gap_limit": relative_gap_limit,
        },
    )


def score_solution_dir(solution_dir: Path, instances_dir: Path, reference_dir: Path) -> ScoreResult:
    manifest = json.loads((reference_dir / "reference_manifest.json").read_text(encoding="utf-8"))
    relative_gap_limit = float(manifest["relative_gap_limit"])

    per_instance = []
    for stem in SELECTED_INSTANCES:
        per_instance.append(
            _score_instance(
                solution_dir / f"{stem}.sol",
                instances_dir / f"{stem}.vrp",
                reference_dir / "bks" / f"{stem}.sol",
                relative_gap_limit,
            )
        )

    passed_count = sum(1 for row in per_instance if row.passed)
    score_map = {0: 0.0, 1: 0.33, 2: 0.67, 3: 1.0}
    return ScoreResult(score=score_map[passed_count], passed_count=passed_count, per_instance=per_instance)
