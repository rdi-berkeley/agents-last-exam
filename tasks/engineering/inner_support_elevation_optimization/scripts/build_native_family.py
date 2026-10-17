"""Generate an explicitly provisional source-digitized native preparation case."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay


def contains(point, polygon):
    inside = False
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        if (first[1] > point[1]) != (second[1] > point[1]):
            cross = first[0] + (point[1] - first[1]) * (second[0] - first[0]) / (
                second[1] - first[1]
            )
            if point[0] < cross:
                inside = not inside
    return inside


def build(destination, *, column=False, geometry=None):
    assets = Path(__file__).parents[1] / "assets"
    benchmark = json.loads((assets / "benchmark.json").read_text())
    sources = json.loads((assets / "source_manifest.json").read_text())["sha256"]
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    outer_pixels = [
        [145, 248],
        [391, 248],
        [319, 133],
        [403, 78],
        [453, 151],
        [526, 95],
        [639, 248],
        [946, 248],
        [946, 522],
        [989, 565],
        [807, 722],
        [642, 550],
        [575, 595],
        [506, 498],
        [145, 498],
    ]
    inner_pixels = [[397, 251], [649, 251], [944, 523], [640, 546]]

    def transform(point):
        return [(point[0] - 145) * 115 / 844, (722 - point[1]) * 89 / 644]

    outer = [transform(point) for point in outer_pixels]
    inner = [transform(point) for point in inner_pixels]
    if geometry is not None:
        outer = geometry["outer_polygon"]
        inner = geometry["inner_polygon"]
    if column:
        outer = [[0, 0], [2, 0], [2, 2], [0, 2]]
        inner = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]]
        limits = [0, 2, 0, 2]
        points = [[0, 0], [2, 0], [2, 2], [0, 2]]
        boundaries = []
    else:
        limits = (
            geometry.get("domain_xy", [-57.5, 172.5, -48, 137])
            if geometry
            else [-57.5, 172.5, -48, 137]
        )
        points = [
            [east, north]
            for east in np.linspace(limits[0], limits[1], 10)
            for north in np.linspace(limits[2], limits[3], 9)
        ]
        if geometry:
            points.extend(geometry.get("extra_points_xy", []))
        boundaries = []
        for polygon in (outer, inner):
            for first, second in zip(polygon, polygon[1:] + polygon[:1]):
                count = max(1, int(np.ceil(np.linalg.norm(np.subtract(first, second)) / 14)))
                segment = [
                    np.add(first, fraction * np.subtract(second, first)).tolist()
                    for fraction in np.linspace(0, 1, count + 1)
                ]
                points.extend(segment)
                boundaries.extend(zip(segment[:-1], segment[1:]))
    for attempt in range(9):
        points = sorted(set(tuple(np.round(point, 8)) for point in points))
        lookup = {point: index for index, point in enumerate(points)}
        triangulation = Delaunay(points)
        edges = {
            tuple(sorted((int(triangle[index]), int(triangle[(index + 1) % 3]))))
            for triangle in triangulation.simplices
            for index in range(3)
        }
        missing = []
        updated = []
        for first, second in boundaries:
            first = tuple(np.round(first, 8))
            second = tuple(np.round(second, 8))
            if tuple(sorted((lookup[first], lookup[second]))) in edges:
                updated.append((first, second))
            else:
                midpoint = tuple(np.round(np.mean([first, second], axis=0), 8))
                missing.append(midpoint)
                updated.extend([(first, midpoint), (midpoint, second)])
        boundaries = updated
        if not missing:
            break
        points.extend(missing)
    else:
        raise ValueError("Source polygon segment recovery did not converge")
    depths = [6.5]
    layers = []
    elevation = 6.5
    for source in benchmark["soils"]:
        bottom = round(elevation - source[1], 8)
        layers.append({"id": source[0], "top": elevation, "bottom": bottom})
        depths.append(bottom)
        elevation = bottom
    depths = sorted(set(depths + [0.7, -3.3, -6.6, -10.1, -15.1, -21.6, -26.5]))
    if geometry:
        depths = sorted(set(depths + geometry.get("extra_levels_m", [])))
    native_layers = [dict(layer) for layer in layers]
    if geometry and geometry["engineering_choices"].get("deep_rock_continuation"):
        native_layers[-1]["bottom"] = geometry["engineering_choices"]["deep_rock_continuation"][
            "bottom_m"
        ]
    count_points = len(points)
    nodes = []
    for height in depths:
        nodes.extend([[east, north, height] for east, north in points])
    elements = []
    for level, (bottom, top) in enumerate(zip(depths[:-1], depths[1:])):
        layer_index = next(
            index
            for index, layer in enumerate(native_layers)
            if layer["bottom"] - 1e-8 <= (bottom + top) / 2 <= layer["top"] + 1e-8
        )
        for simplex in triangulation.simplices:
            vertices = sorted(int(value) for value in simplex)
            lower = [level * count_points + index + 1 for index in vertices]
            upper = [(level + 1) * count_points + index + 1 for index in vertices]
            for connectivity in (
                [lower[0], lower[1], lower[2], upper[2]],
                [lower[0], lower[1], upper[1], upper[2]],
                [lower[0], upper[0], upper[1], upper[2]],
            ):
                coordinates = np.array([nodes[identifier - 1] for identifier in connectivity])
                if np.linalg.det((coordinates[1:] - coordinates[0]).T) < 0:
                    connectivity[0], connectivity[1] = connectivity[1], connectivity[0]
                elements.append([layer_index + 1, *connectivity])
    lines = ["Begin ModelPartData", "End ModelPartData"]
    for identifier in range(1, 15):
        lines.extend([f"Begin Properties {identifier}", "End Properties"])
    lines.append("Begin Nodes")
    lines.extend(
        f"{index} {east:.10g} {north:.10g} {height:.10g}"
        for index, (east, north, height) in enumerate(nodes, 1)
    )
    lines.extend(["End Nodes", "Begin Elements UPwSmallStrainElement3D4N"])
    lines.extend(
        f"{index} " + " ".join(map(str, record)) for index, record in enumerate(elements, 1)
    )
    lines.append("End Elements")
    for identifier in range(1, 15):
        selected = [
            (index, record) for index, record in enumerate(elements, 1) if record[0] == identifier
        ]
        member_nodes = sorted({node for _, record in selected for node in record[1:]})
        lines.extend([f"Begin SubModelPart Layer{identifier:02}", "Begin SubModelPartNodes"])
        lines.extend(map(str, member_nodes))
        lines.extend(["End SubModelPartNodes", "Begin SubModelPartElements"])
        lines.extend(str(index) for index, _ in selected)
        lines.extend(["End SubModelPartElements", "End SubModelPart"])
    lines.append("")
    (root / "model.mdpa").write_text("\n".join(lines))
    materials = []
    for index, source in enumerate(benchmark["soils"], 1):
        modulus = source[7] if source[7] is not None else (150 if index == 13 else 1000)
        variables = {
            "GEO_DRAINAGE_TYPE": "CONSTANT_PW_FIELD",
            "YOUNG_MODULUS": modulus * 1000,
            "POISSON_RATIO": source[6],
            "DENSITY_SOLID": (source[2] / 9.81 - 0.3) / 0.7,
            "DENSITY_WATER": 1.0,
            "POROSITY": 0.3,
            "BULK_MODULUS_SOLID": 1e9,
            "BULK_MODULUS_FLUID": 2e6,
            "DYNAMIC_VISCOSITY": 1e-6,
            "BIOT_COEFFICIENT": 1.0,
            "RETENTION_LAW": "SaturatedLaw",
            "SATURATED_SATURATION": 1.0,
            "GEO_COHESION": source[3],
            "GEO_FRICTION_ANGLE": source[4],
            "GEO_DILATANCY_ANGLE": 0.0,
            "GEO_ENABLE_TENSION_CUT_OFF": False,
            "K0_MAIN_DIRECTION": 2,
            "K0_NC": source[5],
            "PERMEABILITY_XX": 1e-9,
            "PERMEABILITY_YY": 1e-9,
            "PERMEABILITY_ZZ": 1e-9,
            "PERMEABILITY_XY": 0.0,
            "PERMEABILITY_YZ": 0.0,
            "PERMEABILITY_ZX": 0.0,
        }
        materials.append(
            {
                "model_part_name": f"Support.Layer{index:02}",
                "properties_id": index,
                "Material": {
                    "constitutive_law": {"name": "GeoMohrCoulombLaw3D"},
                    "Variables": variables,
                    "Tables": {},
                },
            }
        )
    (root / "Materials.json").write_text(json.dumps({"properties": materials}, indent=2))

    def wall_faces(polygon, toe, wall_top):
        wall_edges = []
        for first, second in boundaries:
            midpoint = np.mean([first, second], axis=0)
            for start, end in zip(polygon, polygon[1:] + polygon[:1]):
                direction = np.subtract(end, start)
                fraction = np.dot(midpoint - start, direction) / np.dot(direction, direction)
                distance = np.linalg.norm(midpoint - (np.array(start) + fraction * direction))
                if -1e-7 <= fraction <= 1 + 1e-7 and distance < 1e-6:
                    wall_edges.append((lookup[tuple(first)], lookup[tuple(second)]))
                    break
        faces = []
        for level, (bottom, top) in enumerate(zip(depths[:-1], depths[1:])):
            if bottom < toe - 1e-8 or top > wall_top + 1e-8:
                continue
            for first, second in wall_edges:
                faces.append(
                    [
                        level * count_points + first + 1,
                        level * count_points + second + 1,
                        (level + 1) * count_points + second + 1,
                        (level + 1) * count_points + first + 1,
                    ]
                )
        return faces

    family = {
        "format": "support-native-family-1",
        "units": "m-kN-s",
        "ground_elevation": 6.5,
        "domain": [*limits, depths[0], 6.5],
        "layers": layers,
        "outer_polygon": outer,
        "inner_polygon": inner,
        "outer_wall_faces": wall_faces(
            outer, -26.5, geometry.get("outer_shell_top_m", 6.5) if geometry else 6.5
        ),
        "inner_wall_faces": wall_faces(
            inner, -21.6, geometry.get("inner_shell_top_m", -10.1) if geometry else -10.1
        ),
        "braces": [],
        "monitors": {},
        "source_hashes": sources,
        "engineering_choices": {
            "scope": "14-layer-column"
            if column
            else "source-digitized-preparation-not-accepted-full-case",
            "rock_moduli_MPa": [150, 1000],
            "porosity": 0.3,
            "dilation_deg": 0,
            "hydraulics": "prescribed hydrostatic pressure; no transient seepage",
            "plan_source": "figure_2_6_monitoring_layout.png",
            "outer_trace_pixels": outer_pixels,
            "inner_trace_pixels": inner_pixels,
            "plan_transform": "x=(pixel_x-145)*115/844; y=(722-pixel_y)*89/644",
            "unresolved": [
                "inner 92 by 32 m dimension registration",
                "complete diagonal brace graph",
                "DBC/CX registration",
                "outer wall extension from +6.50 to +7.00",
                "mesh and domain independence",
            ],
        },
    }
    if geometry:
        family["monitors"] = geometry["monitors"]
        family["engineering_choices"].update(geometry["engineering_choices"])
    (root / "family.json").write_text(json.dumps(family, indent=2))
    return {
        "nodes": len(nodes),
        "soil_elements": len(elements),
        "layers": 14,
        "outer_wall_faces": len(family["outer_wall_faces"]),
        "inner_wall_faces": len(family["inner_wall_faces"]),
        "full_case_ready": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--column", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.output, column=arguments.column)))
