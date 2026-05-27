from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


VARIANT_NAME = "rand50_adapt_s01"
NUM_AGENTS = 50
REQUIRED_FILES = (
    f"{VARIANT_NAME}.solution.csv",
    f"{VARIANT_NAME}.iter_stats.csv",
    f"{VARIANT_NAME}.paths.txt",
)
SOLUTION_HEADER = [
    "runtime",
    "solution cost",
    "initial solution cost",
    "min f value",
    "root g value",
    "iterations",
    "group size",
    "runtime of initial solution",
    "area under curve",
    "preprocessing runtime",
    "solver name",
    "instance name",
]
ITER_STATS_HEADER = [
    "num of agents",
    "sum of costs",
    "runtime",
    "cost lowerbound",
    "sum of distances",
    "MAPF algorithm",
]


@dataclass
class ScoreReport:
    score: float
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "errors": self.errors, "details": self.details}


def _decode_text(payload: bytes | str) -> str:
    if isinstance(payload, bytes):
        return payload.decode("utf-8-sig")
    return payload


def _parse_numeric_cell(value: str, field_name: str, errors: list[str]) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f"{field_name} is not numeric: {value!r}")
        return None


def _parse_integral_cell(value: str, field_name: str, errors: list[str]) -> int | None:
    number = _parse_numeric_cell(value, field_name, errors)
    if number is None:
        return None
    rounded = round(number)
    if abs(number - rounded) > 1e-9:
        errors.append(f"{field_name} must be an integer value, got {value!r}")
        return None
    return int(rounded)


def _csv_rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text.strip())))


def parse_solution_cost(solution_csv: bytes | str, errors: list[str]) -> int | None:
    rows = _csv_rows(_decode_text(solution_csv))
    if len(rows) != 2:
        errors.append(f"solution.csv must contain exactly 2 rows, found {len(rows)}")
        return None
    if rows[0] != SOLUTION_HEADER:
        errors.append("solution.csv header does not match the required 12-column header")
        return None
    if len(rows[1]) != len(SOLUTION_HEADER):
        errors.append(f"solution.csv data row has {len(rows[1])} columns")
        return None
    return _parse_integral_cell(rows[1][1], "solution cost", errors)


def parse_iter_costs(iter_stats_csv: bytes | str, errors: list[str]) -> list[int]:
    rows = _csv_rows(_decode_text(iter_stats_csv))
    if len(rows) < 2:
        errors.append("iter_stats.csv must contain a header and at least one data row")
        return []
    if rows[0] != ITER_STATS_HEADER:
        errors.append("iter_stats.csv header does not match the required 6-column header")
        return []

    costs: list[int] = []
    for row_index, row in enumerate(rows[1:], start=2):
        if len(row) != len(ITER_STATS_HEADER):
            errors.append(f"iter_stats.csv row {row_index} has {len(row)} columns")
            continue
        cost = _parse_integral_cell(row[1], f"iter_stats.csv row {row_index} sum of costs", errors)
        if cost is not None:
            costs.append(cost)

    for index, (previous, current) in enumerate(zip(costs, costs[1:]), start=3):
        if current > previous:
            errors.append(
                "iter_stats.csv sum of costs is not monotonically non-increasing "
                f"at row {index}: {previous} -> {current}"
            )
            break
    return costs


def parse_map(map_text: bytes | str, errors: list[str]) -> tuple[int, int, set[int]]:
    lines = _decode_text(map_text).splitlines()
    try:
        height_line = next(line for line in lines if line.startswith("height "))
        width_line = next(line for line in lines if line.startswith("width "))
        map_index = lines.index("map")
        height = int(height_line.split()[1])
        width = int(width_line.split()[1])
    except (StopIteration, ValueError, IndexError) as exc:
        errors.append(f"map file is malformed: {exc}")
        return 0, 0, set()

    grid = lines[map_index + 1 : map_index + 1 + height]
    if len(grid) != height or any(len(row) != width for row in grid):
        errors.append("map dimensions do not match the declared width/height")
        return width, height, set()

    free: set[int] = set()
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char in {".", "G", "S"}:
                free.add(y * width + x)
    return width, height, free


def parse_scenario(
    scen_text: bytes | str,
    width: int,
    height: int,
    expected_agents: int,
    errors: list[str],
) -> tuple[list[int], list[int]]:
    lines = [line for line in _decode_text(scen_text).splitlines() if line.strip()]
    if not lines or not lines[0].lower().startswith("version"):
        errors.append("scenario file must start with a version line")
        return [], []
    rows = lines[1 : expected_agents + 1]
    if len(rows) != expected_agents:
        errors.append(f"scenario file has only {len(rows)} rows for {expected_agents} agents")
        return [], []

    starts: list[int] = []
    goals: list[int] = []
    for index, line in enumerate(rows):
        fields = line.split()
        if len(fields) < 9:
            errors.append(f"scenario row {index + 2} has too few fields")
            continue
        try:
            scen_width = int(fields[2])
            scen_height = int(fields[3])
            start_x = int(fields[4])
            start_y = int(fields[5])
            goal_x = int(fields[6])
            goal_y = int(fields[7])
        except ValueError:
            errors.append(f"scenario row {index + 2} has non-integer coordinates")
            continue
        if scen_width != width or scen_height != height:
            errors.append(f"scenario row {index + 2} dimensions do not match the map")
        starts.append(start_y * width + start_x)
        goals.append(goal_y * width + goal_x)
    return starts, goals


def parse_paths(paths_txt: bytes | str, expected_agents: int, errors: list[str]) -> list[list[int]]:
    lines = [line.strip() for line in _decode_text(paths_txt).splitlines() if line.strip()]
    if not lines:
        errors.append("paths.txt is empty")
        return []
    try:
        declared_agents = int(lines[0])
    except ValueError:
        errors.append(f"paths.txt first line must be an integer, got {lines[0]!r}")
        return []
    if declared_agents != expected_agents:
        errors.append(f"paths.txt declares {declared_agents} agents, expected {expected_agents}")
    if len(lines) != expected_agents + 1:
        errors.append(f"paths.txt must contain {expected_agents} path lines, found {len(lines) - 1}")
        return []

    paths: list[list[int]] = []
    for agent_index, line in enumerate(lines[1:]):
        tokens = [token for token in line.split(",") if token != ""]
        if not tokens:
            errors.append(f"agent {agent_index} path is empty")
            paths.append([])
            continue
        try:
            paths.append([int(token) for token in tokens])
        except ValueError:
            errors.append(f"agent {agent_index} path contains a non-integer cell id")
            paths.append([])
    return paths


def _position_at(path: list[int], timestep: int) -> int:
    return path[min(timestep, len(path) - 1)]


def validate_paths(
    paths: list[list[int]],
    starts: list[int],
    goals: list[int],
    width: int,
    height: int,
    free_cells: set[int],
    errors: list[str],
) -> int | None:
    if len(paths) != NUM_AGENTS or len(starts) != NUM_AGENTS or len(goals) != NUM_AGENTS:
        errors.append("path, start, and goal counts must all be 50")
        return None

    cell_count = width * height
    for agent_index, path in enumerate(paths):
        if not path:
            continue
        if path[0] != starts[agent_index]:
            errors.append(f"agent {agent_index} path starts at {path[0]}, expected {starts[agent_index]}")
        if path[-1] != goals[agent_index]:
            errors.append(f"agent {agent_index} path ends at {path[-1]}, expected {goals[agent_index]}")
        for step_index, cell in enumerate(path):
            if cell < 0 or cell >= cell_count:
                errors.append(f"agent {agent_index} step {step_index} cell id {cell} is out of bounds")
                continue
            if cell not in free_cells:
                errors.append(f"agent {agent_index} step {step_index} enters blocked cell {cell}")
        for step_index, (previous, current) in enumerate(zip(path, path[1:]), start=1):
            py, px = divmod(previous, width)
            cy, cx = divmod(current, width)
            if abs(px - cx) + abs(py - cy) > 1:
                errors.append(
                    f"agent {agent_index} step {step_index} is not 4-connected: "
                    f"{previous} -> {current}"
                )

    if errors:
        return None

    max_steps = max(len(path) for path in paths)
    for timestep in range(max_steps):
        occupied: dict[int, int] = {}
        for agent_index, path in enumerate(paths):
            cell = _position_at(path, timestep)
            other = occupied.get(cell)
            if other is not None:
                errors.append(f"vertex conflict at t={timestep}: agents {other} and {agent_index} at {cell}")
                return None
            occupied[cell] = agent_index

        for parked_agent, parked_path in enumerate(paths):
            parked_from = len(parked_path) - 1
            if timestep < parked_from:
                continue
            goal_cell = parked_path[-1]
            for agent_index, path in enumerate(paths):
                if agent_index == parked_agent:
                    continue
                if _position_at(path, timestep) == goal_cell:
                    errors.append(
                        f"target conflict at t={timestep}: agent {agent_index} uses "
                        f"parked goal {goal_cell} of agent {parked_agent}"
                    )
                    return None

    for timestep in range(max_steps - 1):
        transitions: dict[tuple[int, int], int] = {}
        for agent_index, path in enumerate(paths):
            start_cell = _position_at(path, timestep)
            end_cell = _position_at(path, timestep + 1)
            reverse_agent = transitions.get((end_cell, start_cell))
            if reverse_agent is not None and start_cell != end_cell:
                errors.append(
                    f"edge conflict at t={timestep}: agents {reverse_agent} and {agent_index} "
                    f"swap {end_cell}<->{start_cell}"
                )
                return None
            transitions[(start_cell, end_cell)] = agent_index

    return sum(len(path) - 1 for path in paths)


def score_submission(
    output_files: dict[str, bytes | str],
    map_text: bytes | str,
    scenario_text: bytes | str,
    metadata: dict[str, Any],
) -> ScoreReport:
    errors: list[str] = []
    missing = [name for name in REQUIRED_FILES if name not in output_files]
    if missing:
        return ScoreReport(score=0.0, errors=[f"missing required output files: {missing}"])

    target = int(metadata.get("target_sum_of_costs", 0))
    tolerance = float(metadata.get("allowed_relative_cost_tolerance", 0.10))
    expected_agents = int(metadata.get("num_agents", NUM_AGENTS))
    if expected_agents != NUM_AGENTS:
        errors.append(f"metadata num_agents must be {NUM_AGENTS}, got {expected_agents}")

    solution_cost = parse_solution_cost(output_files[f"{VARIANT_NAME}.solution.csv"], errors)
    iter_costs = parse_iter_costs(output_files[f"{VARIANT_NAME}.iter_stats.csv"], errors)
    width, height, free_cells = parse_map(map_text, errors)
    starts, goals = parse_scenario(scenario_text, width, height, NUM_AGENTS, errors)
    paths = parse_paths(output_files[f"{VARIANT_NAME}.paths.txt"], NUM_AGENTS, errors)

    derived_cost = None
    if width and height and free_cells and starts and goals and paths:
        derived_cost = validate_paths(paths, starts, goals, width, height, free_cells, errors)

    if solution_cost is not None and derived_cost is not None and solution_cost != derived_cost:
        errors.append(f"solution cost {solution_cost} does not equal derived path cost {derived_cost}")
    if solution_cost is not None and iter_costs and solution_cost != min(iter_costs):
        errors.append(f"solution cost {solution_cost} does not equal best iter_stats cost {min(iter_costs)}")
    if solution_cost is not None and target > 0:
        max_delta = target * tolerance
        if abs(solution_cost - target) > max_delta:
            errors.append(
                f"solution cost {solution_cost} is outside {tolerance:.0%} of the hidden target"
            )

    details = {
        "solution_cost": solution_cost,
        "derived_path_cost": derived_cost,
        "best_iter_cost": min(iter_costs) if iter_costs else None,
        "target_sum_of_costs": target,
        "allowed_relative_cost_tolerance": tolerance,
    }
    return ScoreReport(score=0.0 if errors else 1.0, errors=errors, details=details)


def _read_dir_payloads(output_dir: Path) -> dict[str, bytes]:
    return {name: (output_dir / name).read_bytes() for name in REQUIRED_FILES if (output_dir / name).exists()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score MAPF-LNS output files for the rand50_adapt_s01 task.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    metadata = json.loads((args.reference_dir / "evaluation_metadata.json").read_text(encoding="utf-8"))
    report = score_submission(
        output_files=_read_dir_payloads(args.output_dir),
        map_text=(args.input_dir / "random-32-32-20.map").read_bytes(),
        scenario_text=(args.input_dir / "random-32-32-20-random-1.scen").read_bytes(),
        metadata=metadata,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.score == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
