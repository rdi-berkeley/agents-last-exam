"""Descriptive native mesh measurements for source review, without acceptance cutoffs."""

from itertools import combinations

import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        **dict(
            zip(
                ("min", "p10", "median", "p90", "max"),
                np.percentile(values, [0, 10, 50, 90, 100]).tolist(),
            )
        ),
    }


def describe_mesh(nodes, elements, panels, geometry):
    tetrahedra = [row for row in elements.values() if len(row[2]) == 4]
    if len(tetrahedra) != len(elements):
        return {
            "measured": False,
            "reason": "Local size measurement currently supports linear tetrahedra",
        }
    coordinates = np.asarray([[nodes[identifier] for identifier in row[2]] for row in tetrahedra])
    centroids = coordinates.mean(axis=1)
    volumes = np.abs(np.linalg.det(coordinates[:, 1:] - coordinates[:, :1])) / 6
    edges = np.stack(
        [
            coordinates[:, first] - coordinates[:, second]
            for first, second in combinations(range(4), 2)
        ],
        axis=1,
    )
    diameter = np.linalg.norm(edges, axis=2).max(axis=1)
    horizontal = np.linalg.norm(edges[:, :, :2], axis=2).max(axis=1)
    vertical = np.ptp(coordinates[:, :, 2], axis=1)
    wall_distance = np.full(len(centroids), np.inf)
    for system in panels:
        polygon = np.asarray(geometry[f"{system}_polygon"])
        for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
            delta = second - first
            fraction = np.clip((centroids[:, :2] - first) @ delta / (delta @ delta), 0, 1)
            distance = np.linalg.norm(centroids[:, :2] - first - fraction[:, None] * delta, axis=1)
            wall_distance = np.minimum(wall_distance, distance)
    bins = {}
    for lower, upper in ((0, 2), (2, 5), (5, 10), (10, 20), (20, float("inf"))):
        selected = (wall_distance >= lower) & (wall_distance < upper)
        bins[f"{lower}_to_{upper}_m"] = {
            "count": int(selected.sum()),
            "maximum_horizontal_edge_m": distribution(horizontal[selected]),
            "vertical_span_m": distribution(vertical[selected]),
            "diameter_m": distribution(diameter[selected]),
        }
    bounds = np.asarray(list(nodes.values()))
    minimum, maximum = bounds.min(axis=0), bounds.max(axis=0)
    walls = {}
    for system, faces in panels.items():
        lengths = np.linalg.norm(np.roll(faces[:, :, :2], -1, axis=1) - faces[:, :, :2], axis=2)
        toe = float(faces[:, :, 2].min())
        walls[system] = {
            "plan_panel_length_m": distribution(lengths.max(axis=1)),
            "vertical_panel_span_m": distribution(np.ptp(faces[:, :, 2], axis=1)),
            "toe_z_m": toe,
            "toe_to_fixed_bottom_m": toe - float(minimum[2]),
            "plan_side_clearances_m": np.minimum(
                faces[:, :, :2].min(axis=(0, 1)) - minimum[:2],
                maximum[:2] - faces[:, :, :2].max(axis=(0, 1)),
            ).tolist(),
        }
    return {
        "measured": True,
        "method": "Native linear-tetrahedron vertex coordinates; horizontal centroid distance to nearest declared wall plan segment. Bins describe geometry only and are not acceptance thresholds.",
        "tetrahedron_volume_m3": distribution(volumes),
        "zero_volume_count": int((volumes == 0).sum()),
        "wall_distance_bins": bins,
        "wall_panels_and_boundary_clearance": walls,
        "layers": [
            {
                "id": layer["id"],
                "top_m": layer["top"],
                "bottom_m": layer["bottom"],
                "vertical_span_m": distribution(
                    vertical[(centroids[:, 2] < layer["top"]) & (centroids[:, 2] > layer["bottom"])]
                ),
            }
            for layer in geometry["layers"]
        ],
        "limits": "Sizes and clearance establish actual discretization, not error bounds. Fixed-bottom zero displacement is imposed and cannot prove boundary independence. No result or preferred mesh is used.",
    }
