#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TILE_RE = re.compile(
    r"^TILE_(?P<zone>[^_]+(?:_[^_]+)*)__"
    r"(?P<left>[A-Za-z0-9]+)__"
    r"(?P<right>[A-Za-z0-9]+)__"
    r"(?P<idx>\d+)$"
)
COVER_RE = re.compile(r"^COVER_(?P<zone>[^_]+(?:_[^_]+)*)__\d+$")
BLOCKER_RE = re.compile(r"^BLOCKER_(?P<name>.+)$")
ANCHOR_RE = re.compile(r"^ANCHOR_(?P<name>.+)$")

VISIBLE_ROUTES = ("west_infiltration", "roof_reroute", "escort_egress")
HIDDEN_ROUTES = ("hidden_generator_breach", "hidden_detour_extract")
ZONE_TARGETS = ("relay_bay", "catwalk", "service_tunnel", "medbay")

ROUTE_SPECS = {
    "west_infiltration": ["EntryWest", "RelayConsole", "VaultTerminal", "ExtractionLift"],
    "roof_reroute": ["RoofInsert", "GeneratorSwitch", "SecurityCheckpoint", "ExtractionLift"],
    "escort_egress": ["BrigGate", "MedBayDoor", "AtriumCheckpoint", "ExtractionLift"],
    "hidden_generator_breach": ["RoofInsert", "GeneratorSwitch", "AtriumCheckpoint", "ExtractionLift"],
    "hidden_detour_extract": ["EntryWest", "MedBayDoor", "ExtractionLift"],
}

REFERENCE_METRICS = {
    "route_lengths": {
        "west_infiltration": 48.0,
        "roof_reroute": 54.0,
        "escort_egress": 44.0,
        "hidden_generator_breach": 52.0,
        "hidden_detour_extract": 58.0,
    },
    "zone_tile_counts": {
        "west_lane": 6,
        "relay_bay": 2,
        "vault": 4,
        "east_detour": 12,
        "atrium_exit": 11,
        "catwalk": 7,
        "service_tunnel": 8,
        "security_descent": 12,
        "catwalk_drop": 8,
        "brig": 5,
        "medbay": 6,
    },
    "cover_counts": {
        "relay_bay": 4,
        "atrium": 6,
        "extraction": 3,
    },
    "los_blocked": {
        "sniper_west_to_vault": True,
        "roof_to_extract": False,
    },
}

SCENE_MARKERS = (
    "EntryWestMarker",
    "RelayConsoleMarker",
    "VaultTerminalMarker",
    "ExtractionLiftMarker",
    "RoofInsertMarker",
    "GeneratorSwitchMarker",
    "SecurityCheckpointMarker",
    "BrigGateMarker",
    "MedBayDoorMarker",
    "AtriumCheckpointMarker",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def segment_zone(left: str, right: str) -> str:
    names = {left, right}
    if names == {"EntryWest", "AtriumCheckpoint"}:
        return "west_lane"
    if names == {"AtriumCheckpoint", "RelayConsole"}:
        return "relay_bay"
    if names == {"RelayConsole", "VaultTerminal"}:
        return "vault"
    if names == {"VaultTerminal", "ExtractionLift"}:
        return "east_detour"
    if names == {"AtriumCheckpoint", "ExtractionLift"}:
        return "atrium_exit"
    if names == {"RoofInsert", "GeneratorSwitch"}:
        return "catwalk"
    if names == {"GeneratorSwitch", "SecurityCheckpoint"}:
        return "service_tunnel"
    if names == {"SecurityCheckpoint", "ExtractionLift"}:
        return "security_descent"
    if names == {"GeneratorSwitch", "AtriumCheckpoint"}:
        return "catwalk_drop"
    if names == {"BrigGate", "MedBayDoor"}:
        return "brig"
    if names == {"MedBayDoor", "AtriumCheckpoint"}:
        return "medbay"
    raise KeyError(f"unexpected segment: {left} <-> {right}")


def load_node_names(gltf_path: Path) -> list[str]:
    data = json.loads(gltf_path.read_text(encoding="utf-8"))
    return [node.get("name", "") for node in data.get("nodes", []) if node.get("name")]


def missing_gltf_sidecars(submission_dir: Path, gltf_path: Path) -> list[str]:
    data = json.loads(gltf_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    for buffer in data.get("buffers", []):
        uri = buffer.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        sidecar = (gltf_path.parent / uri).resolve()
        try:
            sidecar.relative_to(submission_dir.resolve())
        except ValueError:
            missing.append(uri)
            continue
        if not sidecar.exists():
            missing.append(uri)
    return sorted(set(missing))


def build_layout_from_nodes(node_names: list[str]) -> dict:
    segment_counts: dict[tuple[str, str], int] = {}
    tiles = []
    cover_counts: dict[str, int] = {}
    blockers: dict[str, bool] = {}
    anchors: dict[str, str] = {}

    for name in node_names:
        match = TILE_RE.match(name)
        if match:
            left = match.group("left")
            right = match.group("right")
            segment_counts[(left, right)] = max(
                segment_counts.get((left, right), 0),
                int(match.group("idx")),
            )
            tiles.append({"id": name, "zone": match.group("zone")})
            continue

        match = COVER_RE.match(name)
        if match:
            zone = match.group("zone")
            cover_counts[zone] = cover_counts.get(zone, 0) + 1
            continue

        match = BLOCKER_RE.match(name)
        if match:
            blockers[match.group("name")] = True
            continue

        match = ANCHOR_RE.match(name)
        if match:
            anchors[match.group("name")] = name

    graph_edges = [(left, right, count) for (left, right), count in segment_counts.items()]
    return {
        "graph_edges": graph_edges,
        "tiles": tiles,
        "stairs": [],
        "anchors": anchors,
        "cover": cover_counts,
        "blockers": blockers,
    }


def _shortest_path_length(graph: dict[str, list[tuple[str, int]]], start: str, end: str) -> float:
    queue: list[tuple[int, str]] = [(0, start)]
    seen = {start: 0}
    while queue:
        cost, node = queue.pop(0)
        if node == end:
            return float(cost * 2)
        if cost > seen[node]:
            continue
        for neighbor, weight in graph[node]:
            next_cost = cost + weight
            if next_cost < seen.get(neighbor, 10**9):
                seen[neighbor] = next_cost
                queue.append((next_cost, neighbor))
                queue.sort()
    raise KeyError(f"unreachable route segment: {start} -> {end}")


def compute_layout_metrics(layout: dict) -> dict:
    graph: dict[str, list[tuple[str, int]]] = defaultdict(list)
    zone_counter: Counter[str] = Counter()
    for left, right, weight in layout["graph_edges"]:
        graph[left].append((right, weight))
        graph[right].append((left, weight))
        zone_counter[segment_zone(left, right)] += weight

    route_lengths: dict[str, float] = {}
    for route_name, anchors in ROUTE_SPECS.items():
        total = 0.0
        for left, right in zip(anchors, anchors[1:]):
            try:
                total += _shortest_path_length(graph, left, right)
            except KeyError:
                total = float("inf")
                break
        route_lengths[route_name] = total

    return {
        "route_lengths": route_lengths,
        "zone_tile_counts": dict(zone_counter),
        "cover_counts": dict(layout["cover"]),
        "los_blocked": dict(layout["blockers"]),
    }


def score_submission(submission_dir: Path) -> dict:
    gltf_path = submission_dir / "exports" / "atlas_outpost_blockout.gltf"
    blend_path = submission_dir / "blender" / "atlas_outpost_blockout.blend"
    scene_path = submission_dir / "godot" / "scenes" / "atlas_outpost_validation.tscn"
    handoff_path = submission_dir / "docs" / "layout_handoff.md"

    results: dict[str, object] = {
        "required_files": {},
        "route_checks": {},
        "zone_checks": {},
        "cover_checks": {},
        "los_checks": {},
        "scene_checks": {},
        "handoff_checks": {},
    }
    raw_score = 0.0

    required_files = {
        "blend": blend_path.exists(),
        "gltf": gltf_path.exists(),
        "scene": scene_path.exists(),
        "handoff": handoff_path.exists(),
    }
    results["required_files"] = required_files
    if all(required_files.values()):
        raw_score += 10.0
    else:
        results["raw_total_score"] = raw_score
        results["score"] = 0.0
        results["passes"] = False
        return results

    sidecar_missing = missing_gltf_sidecars(submission_dir, gltf_path)
    results["gltf_sidecar_checks"] = {
        "all_referenced_sidecars_present": not sidecar_missing,
        "missing_sidecars": sidecar_missing,
    }

    layout = build_layout_from_nodes(load_node_names(gltf_path))
    metrics = compute_layout_metrics(layout)
    results["computed_metrics"] = metrics

    route_checks: dict[str, bool] = {}
    for route_name in (*VISIBLE_ROUTES, *HIDDEN_ROUTES):
        actual = metrics["route_lengths"].get(route_name, math.inf)
        expected = REFERENCE_METRICS["route_lengths"][route_name]
        route_checks[route_name] = actual == expected
        if actual == expected:
            raw_score += 8.0
    results["route_checks"] = route_checks

    zone_checks: dict[str, bool] = {}
    for zone_name in ZONE_TARGETS:
        actual = metrics["zone_tile_counts"].get(zone_name, 0)
        expected = REFERENCE_METRICS["zone_tile_counts"][zone_name]
        zone_checks[zone_name] = actual == expected
        if actual == expected:
            raw_score += 5.0
    results["zone_checks"] = zone_checks

    cover_checks: dict[str, bool] = {}
    for zone_name, expected in REFERENCE_METRICS["cover_counts"].items():
        actual = metrics["cover_counts"].get(zone_name, 0)
        cover_checks[zone_name] = actual == expected
        if actual == expected:
            raw_score += 10.0 / len(REFERENCE_METRICS["cover_counts"])
    results["cover_checks"] = cover_checks

    los_checks: dict[str, bool] = {}
    for blocker_name, expected in REFERENCE_METRICS["los_blocked"].items():
        actual = metrics["los_blocked"].get(blocker_name, False)
        los_checks[blocker_name] = actual == expected
        if actual == expected:
            raw_score += 5.0
    results["los_checks"] = los_checks

    scene_text = scene_path.read_text(encoding="utf-8")
    scene_checks = {
        "imports_level_gltf": "atlas_outpost_blockout.gltf" in scene_text,
        "has_navigation_root": 'node name="NavigationRoot"' in scene_text,
    }
    for marker in SCENE_MARKERS:
        scene_checks[marker] = marker in scene_text
    raw_score += 10.0 * (sum(scene_checks.values()) / len(scene_checks))
    results["scene_checks"] = scene_checks

    handoff_text = handoff_path.read_text(encoding="utf-8")
    handoff_checks = {
        route_name: route_name in handoff_text
        and str(int(REFERENCE_METRICS["route_lengths"][route_name])) in handoff_text
        for route_name in VISIBLE_ROUTES
    }
    raw_score += 10.0 * (sum(handoff_checks.values()) / len(handoff_checks))
    results["handoff_checks"] = handoff_checks

    raw_score = round(raw_score, 2)
    passes = (
        raw_score >= 85.0
        and not sidecar_missing
        and all(route_checks.values())
        and all(zone_checks.values())
        and all(cover_checks.values())
        and all(los_checks.values())
        and all(scene_checks.values())
        and all(handoff_checks.values())
    )
    results["raw_total_score"] = raw_score
    results["pass_threshold"] = 85.0
    results["passes"] = passes
    results["score"] = 1.0 if passes else 0.0
    return results


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    results = json_safe(score_submission(Path(args.submission_dir)))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=2, allow_nan=False) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(json.dumps(results, sort_keys=True, allow_nan=False))
    return 0 if results.get("passes") else 1


if __name__ == "__main__":
    raise SystemExit(main())
