"""Deterministic preflight for the public A-zone model contract."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

if __package__:
    from .native_input_adapter import contains, read_family, read_json
    from .native_member_sections import member_sections, on_perimeter, section_properties
else:
    from native_input_adapter import contains, read_family, read_json
    from native_member_sections import member_sections, on_perimeter, section_properties


CONTRACT = "support-deterministic-20261006"


def cross(first, second):
    return first[0] * second[1] - first[1] * second[0]


def distance(point, start, end):
    direction = end - start
    fraction = np.clip(np.dot(point - start, direction) / np.dot(direction, direction), 0, 1)
    return float(np.linalg.norm(point - start - fraction * direction))


def polygon_geometry(points):
    polygon = np.asarray(points, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("Polygon needs at least three xy vertices")
    if not np.isfinite(polygon).all() or len(np.unique(polygon, axis=0)) != len(polygon):
        raise ValueError("Nonfinite or repeated polygon vertex")
    edges = list(zip(polygon, np.roll(polygon, -1, axis=0)))
    for index, (start, end) in enumerate(edges):
        for other_index, (other_start, other_end) in enumerate(edges[index + 1 :], index + 1):
            if other_index in (index + 1, len(edges) - 1 if index == 0 else -1):
                continue
            directions = (end - start, other_end - other_start)
            denominator = cross(*directions)
            if abs(denominator) > 1e-10:
                offset = other_start - start
                first = cross(offset, directions[1]) / denominator
                second = cross(offset, directions[0]) / denominator
                if -1e-9 <= first <= 1 + 1e-9 and -1e-9 <= second <= 1 + 1e-9:
                    raise ValueError("Self-intersecting polygon")
            elif min(distance(point, start, end) for point in (other_start, other_end)) < 1e-8:
                raise ValueError("Overlapping polygon edges")
    area = abs(sum(cross(start, end) for start, end in edges)) / 2
    boxes = []
    for start, end in edges:
        direction = (end - start) / np.linalg.norm(end - start)
        frame = np.array([direction, [-direction[1], direction[0]]])
        spans = np.ptp(polygon @ frame.T, axis=0)
        boxes.append(sorted(spans))
    width, length = min(boxes, key=lambda spans: spans[0] * spans[1])
    if area <= 1e-8 or width <= 1e-8:
        raise ValueError("Degenerate polygon")
    return polygon, edges, float(width), float(length), float(area / (width * length))


def read_mesh(path):
    nodes, elements, stack = {}, {}, []
    for raw in Path(path).read_text().splitlines():
        cells = raw.split("//", 1)[0].split()
        if not cells:
            continue
        if cells[0] == "Begin":
            stack.append(cells[1:])
        elif cells[0] == "End":
            if not stack or stack.pop()[0] != cells[1]:
                raise ValueError("Unbalanced native mesh block")
        elif not stack:
            raise ValueError("Mesh data outside a block")
        elif stack[-1][0] == "Nodes":
            identifier = int(cells[0])
            point = np.asarray(cells[1:], dtype=float)
            if identifier in nodes or point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("Invalid or duplicate native mesh node")
            nodes[identifier] = point
        elif stack[-1][0] == "Elements":
            identifier, property_id, *connectivity = map(int, cells)
            arity = {"UPwSmallStrainElement3D4N": 4, "UPwSmallStrainElement3D8N": 8}.get(
                stack[-1][1]
            )
            if (
                identifier in elements
                or len(connectivity) != arity
                or len(set(connectivity)) != arity
            ):
                raise ValueError("Invalid or duplicate native soil element")
            elements[identifier] = (property_id, connectivity)
    if stack or not nodes or not elements:
        raise ValueError("Incomplete native mesh")
    if any(not set(connectivity).issubset(nodes) for _, connectivity in elements.values()):
        raise ValueError("Native element refers to unknown nodes")
    return nodes, elements


def check_geometry(family, nodes, elements):
    domain = np.asarray(family["domain"], dtype=float)
    if domain.shape != (6,) or not np.isfinite(domain).all():
        raise ValueError("Invalid domain bounds")
    lower, upper = domain[[0, 2, 4]], domain[[1, 3, 5]]
    points = np.asarray(list(nodes.values()))
    if (
        np.any(upper <= lower)
        or not np.allclose(points.min(axis=0), lower, atol=1e-7, rtol=0)
        or not np.allclose(points.max(axis=0), upper, atol=1e-7, rtol=0)
    ):
        raise ValueError("Declared domain differs from native mesh bounds")
    if upper[2] != 6.5 or lower[2] > family["layers"][-1]["bottom"] + 1e-8:
        raise ValueError("Domain must contain all fourteen strata up to ground +6.5m")
    polygons = {
        system: polygon_geometry(family[f"{system}_polygon"]) for system in ("outer", "inner")
    }
    for system, (_, edges, width, length, fill) in polygons.items():
        limits = (60, 120, 100, 140) if system == "outer" else (25, 45, 75, 105)
        if not limits[0] <= width <= limits[1] or not limits[2] <= length <= limits[3]:
            raise ValueError(
                f"{system} minimum-area edge-aligned rectangle outside public size ranges"
            )
        if system == "outer" and fill > 0.95:
            raise ValueError("Outer plan must be irregular: polygon/rectangle area ratio <=0.95")
        polygon = polygons[system][0]
        if np.any(polygon.min(axis=0) <= lower[:2]) or np.any(polygon.max(axis=0) >= upper[:2]):
            raise ValueError("Pit must lie strictly inside lateral domain boundaries")
        crown = family["engineering_choices"].get("crowns", {}).get(system)
        wall_top = 7.0 if system == "outer" else -10.1
        if crown:
            section_properties(crown["section_m"])
            wall_top = crown["shell_top_m"]
            axis = crown["axis_z_m"]
            if abs(axis - (6.5 if system == "outer" else -10.1)) > 1e-8:
                raise ValueError("Crown axis must coincide with the supplied installation level")
            half_height = crown["section_m"]["height"] / 2
            if not axis - half_height <= wall_top < axis + half_height:
                raise ValueError("Crown attachment must lie inside its section below the top")
            if system == "outer" and abs(axis + half_height - 7.0) > 1e-8:
                raise ValueError("Outer crown must complete the profile to +7m")
        toe = -26.5 if system == "outer" else -21.6
        rectangles = [[] for _ in edges]
        for face in family[f"{system}_wall_faces"]:
            if len(set(face)) != 4 or not set(face).issubset(nodes):
                raise ValueError("Invalid native Q4 wall connectivity")
            coordinates = np.asarray([nodes[node] for node in face])
            plan = np.unique(coordinates[:, :2], axis=0)
            levels = np.unique(coordinates[:, 2])
            if len(plan) != 2 or len(levels) != 2 or len(np.unique(coordinates, axis=0)) != 4:
                raise ValueError("Wall panels must be vertically extruded Q4 rectangles")
            if levels[0] < toe - 1e-8 or levels[1] > wall_top + 1e-8:
                raise ValueError("Wall face outside declared toe/attachment levels")
            matched = False
            for index, (start, end) in enumerate(edges):
                if max(distance(point, start, end) for point in plan) < 1e-7:
                    direction = end - start
                    fractions = sorted((plan - start) @ direction / np.dot(direction, direction))
                    rectangles[index].append((*fractions, *levels))
                    matched = True
                    break
            if not matched:
                raise ValueError("Wall face is off its excavation perimeter")
        for tiles in rectangles:
            levels = sorted({toe, wall_top, *(value for tile in tiles for value in tile[2:])})
            for bottom, top in zip(levels, levels[1:]):
                middle = (bottom + top) / 2
                reached = 0.0
                for start, end in sorted(
                    (tile[0], tile[1]) for tile in tiles if tile[2] < middle < tile[3]
                ):
                    if abs(start - reached) > 1e-7:
                        raise ValueError("Gap or overlap in native wall coverage")
                    reached = end
                if abs(reached - 1) > 1e-7:
                    raise ValueError("Incomplete native wall perimeter/top-to-toe coverage")
    outer = family["outer_polygon"]
    outer_edges = polygons["outer"][1]
    for start, end in polygons["inner"][1]:
        if any(
            not contains(point, outer)
            or min(distance(point, first, second) for first, second in outer_edges) < 1e-7
            for point in (start, (start + end) / 2, end)
        ):
            raise ValueError("Inner pit must lie strictly inside the outer excavation")
        for first, second in outer_edges:
            denominator = cross(end - start, second - first)
            if abs(denominator) > 1e-10:
                along = cross(first - start, second - first) / denominator
                across = cross(first - start, end - start) / denominator
                if 0 <= along <= 1 and 0 <= across <= 1:
                    raise ValueError("Inner boundary crosses outer boundary")
    monitors = family["monitors"]
    if set(monitors) != {
        f"{prefix}{number}" for prefix in ("DBC", "CX") for number in range(1, 11)
    }:
        raise ValueError("Exactly DBC1..10 and CX1..10 are required")
    for prefix in ("DBC", "CX"):
        registered = np.asarray(
            [monitors[f"{prefix}{number}"] for number in range(1, 11)], dtype=float
        )
        if registered.shape != (10, 3) or not np.isfinite(registered).all():
            raise ValueError("Invalid monitor coordinates")
        for index, point in enumerate(registered):
            if any(np.linalg.norm(point[:2] - other[:2]) < 1 for other in registered[:index]):
                raise ValueError("Monitor locations within each group must be at least 1m apart")
            offset = min(distance(point[:2], start, end) for start, end in outer_edges)
            if prefix == "CX":
                if offset > 1e-7 or not -26.5 <= point[2] <= 7:
                    raise ValueError(
                        "CX must register on the outer perimeter within the wall range"
                    )
            elif (
                contains(point[:2], outer)
                or not 1e-7 < offset <= 15
                or abs(point[2] - 6.5) > 1e-8
                or np.any(point[:2] <= lower[:2])
                or np.any(point[:2] >= upper[:2])
            ):
                raise ValueError("DBC must lie on ground outside the pit within 15m of its wall")
    horizontal_sizes, centers = [], []
    for arity in (4, 8):
        connectivity_group = [
            connectivity for _, connectivity in elements.values() if len(connectivity) == arity
        ]
        if not connectivity_group:
            continue
        coordinates = np.asarray(
            [[nodes[node] for node in connectivity] for connectivity in connectivity_group]
        )
        horizontal_sizes.extend(
            np.max(
                [
                    np.linalg.norm(coordinates[:, first, :2] - coordinates[:, second, :2], axis=1)
                    for first, second in combinations(range(arity), 2)
                ],
                axis=0,
            )
        )
        centers.extend(coordinates[:, :, :2].mean(axis=1))
    sizes, centers = np.asarray(horizontal_sizes), np.asarray(centers)
    distances = np.full(len(sizes), np.inf)
    for geometry in polygons.values():
        for start, end in geometry[1]:
            direction = end - start
            fraction = np.clip((centers - start) @ direction / np.dot(direction, direction), 0, 1)
            distances = np.minimum(
                distances, np.linalg.norm(centers - start - fraction[:, None] * direction, axis=1)
            )
    near, far = sizes[distances <= 10], sizes[distances >= 20]
    if not len(near) or not len(far) or np.median(near) > 0.95 * np.median(far):
        raise ValueError(
            "Local refinement requires near-wall median horizontal diameter <=95% of far-field median"
        )
    return {
        "nodes": len(nodes),
        "soil_elements": len(elements),
        "near_median_m": float(np.median(near)),
        "far_median_m": float(np.median(far)),
    }


def check_braces(family):
    if len(family["braces"]) != 2 or {row["system"] for row in family["braces"]} != {
        "outer",
        "inner",
    }:
        raise ValueError("Exactly one outer and one inner brace graph are required")
    section_properties(family["engineering_choices"]["beam_section_m"])
    for system in ("outer", "inner"):
        polygon = family[f"{system}_polygon"]
        perimeter = list(zip(polygon, polygon[1:] + polygon[:1]))
        graph = next(row for row in family["braces"] if row["system"] == system)
        adjacency = {}
        cross_pit = []
        for start, end in graph["segments_xy_m"]:
            first, second = tuple(start), tuple(end)
            if (
                len(first) != 2
                or len(second) != 2
                or not np.isfinite([first, second]).all()
                or first == second
            ):
                raise ValueError("Invalid brace segment")
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
            if not on_perimeter(first, second, perimeter):
                cross_pit.append((np.asarray(first), np.asarray(second)))
        if not adjacency or not cross_pit:
            raise ValueError("Brace graph needs cross-pit members")
        first = next(iter(adjacency))
        visited, pending = set(), [first]
        while pending:
            point = pending.pop()
            if point not in visited:
                visited.add(point)
                pending.extend(adjacency[point] - visited)
        if visited != set(adjacency):
            raise ValueError(
                "Disconnected brace graph; join crossing members with shared endpoints"
            )
        for point in adjacency:
            if (
                not contains(point, polygon)
                and min(
                    distance(np.asarray(point), np.asarray(start), np.asarray(end))
                    for start, end in perimeter
                )
                > 1e-7
            ):
                raise ValueError("Brace endpoint outside its pit")
        if (
            sum(
                min(
                    distance(np.asarray(point), np.asarray(start), np.asarray(end))
                    for start, end in perimeter
                )
                < 1e-7
                for point in adjacency
            )
            < 2
        ):
            raise ValueError("Brace graph needs at least two wall attachments")
        if not any(
            abs(cross(end - start, np.asarray(other_end) - other_start)) > 1e-7
            and abs(np.dot(end - start, np.asarray(other_end) - other_start)) > 1e-7
            for start, end in cross_pit
            for other_start, other_end in perimeter
        ):
            raise ValueError("Brace graph needs a diagonal cross-pit member")
        if not any(on_perimeter(start, end, perimeter) for start, end in graph["segments_xy_m"]):
            raise ValueError("Brace graph must include perimeter waler members")
        elevations = (
            (6.5, 0.7, -3.3, -6.6) if system == "outer" else (-10.1, -11.1, -12.1, -13.1, -14.1)
        )
        for elevation in elevations:
            member_sections(family, system, elevation)
        crown = family["engineering_choices"].get("crowns", {}).get(system)
        if crown and system == "inner":
            member_sections(family, system, crown["axis_z_m"], crown_only=True)
            if not all(
                on_perimeter(start, end, graph["crown_segments_xy_m"], tolerance=2e-6)
                for start, end in perimeter
            ):
                raise ValueError("Inner crown must cover its wall perimeter")


def validate_model(directory):
    family, benchmark = read_family(directory)
    tolerance_limits = {
        "residual_relative_tolerance": 1e-6,
        "displacement_relative_tolerance": 1e-6,
        "residual_absolute_tolerance": 1e-5,
        "displacement_absolute_tolerance": 1e-7,
    }
    for name, value in family["engineering_choices"].get("solver_controls", {}).items():
        if value > tolerance_limits[name]:
            raise ValueError("Solver tolerance exceeds public maximum: " + name)
    materials = read_json(Path(directory) / "Materials.json")["properties"]
    for material in materials:
        index = int(material["model_part_name"][-2:]) - 1
        source = benchmark["soils"][index]
        values = material["Material"]["Variables"]
        for key, expected in zip(
            ("GEO_COHESION", "GEO_FRICTION_ANGLE", "K0_NC", "POISSON_RATIO"), source[3:7]
        ):
            if not np.isclose(values[key], expected, rtol=1e-9, atol=1e-9):
                raise ValueError(f"Source material mismatch: {source[0]} {key}")
        modulus = values["YOUNG_MODULUS"]
        if (
            not np.isfinite(modulus)
            or (source[7] is not None and not np.isclose(modulus, source[7] * 1000, rtol=1e-9))
            or (source[7] is None and modulus <= 46000)
        ):
            raise ValueError("Wrong soil modulus or rock not stiffer than supplied soils")
        weight = 9.81 * (
            (1 - values["POROSITY"]) * values["DENSITY_SOLID"]
            + values["POROSITY"] * values["DENSITY_WATER"]
        )
        if (
            not 0 <= values["POROSITY"] < 1
            or not np.isclose(weight, source[2], rtol=1e-9)
            or values["K0_MAIN_DIRECTION"] != 2
        ):
            raise ValueError("Wrong source unit weight, porosity or vertical K0 axis")
    nodes, elements = read_mesh(Path(directory) / "model.mdpa")
    geometry = check_geometry(family, nodes, elements)
    check_braces(family)
    return {
        "contract": CONTRACT,
        "input_hashes": family["input_hashes"],
        "monitors": family["monitors"],
        "wall_range_z_m": [-26.5, 7.0],
        "geometry": geometry,
        "source_registration_reviewed": True,
        "native_compliance_completed": False,
        "full_task_acceptance": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("family")
    arguments = parser.parse_args()
    print(json.dumps(validate_model(arguments.family), indent=2, allow_nan=False))
