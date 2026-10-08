"""Attach declared trial bracing choices to the retained fourteen-layer mesh."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def split_segments(segments, merge_distance=0.0):
    cuts = [[0.0, 1.0] for _ in segments]
    for first_index, (first_start, first_end) in enumerate(segments):
        first_direction = np.subtract(first_end, first_start)
        first_length_squared = float(np.dot(first_direction, first_direction))
        for second_index in range(first_index):
            second_start, second_end = segments[second_index]
            second_direction = np.subtract(second_end, second_start)
            second_length_squared = float(np.dot(second_direction, second_direction))
            if min(first_length_squared, second_length_squared) < 1e-20:
                continue
            matrix = np.column_stack((first_direction, -second_direction))
            if abs(np.linalg.det(matrix)) < 1e-10:
                separation = np.subtract(second_start, first_start)
                projection = np.dot(separation, first_direction) / first_length_squared
                if np.linalg.norm(separation - projection * first_direction) > 1e-8:
                    continue
                for point in (second_start, second_end):
                    fraction = np.dot(np.subtract(point, first_start), first_direction)
                    fraction /= first_length_squared
                    if -1e-8 <= fraction <= 1 + 1e-8:
                        cuts[first_index].append(float(np.clip(fraction, 0, 1)))
                for point in (first_start, first_end):
                    fraction = np.dot(np.subtract(point, second_start), second_direction)
                    fraction /= second_length_squared
                    if -1e-8 <= fraction <= 1 + 1e-8:
                        cuts[second_index].append(float(np.clip(fraction, 0, 1)))
                continue
            first_fraction, second_fraction = np.linalg.solve(
                matrix, np.subtract(second_start, first_start)
            )
            if -1e-8 <= first_fraction <= 1 + 1e-8 and -1e-8 <= second_fraction <= 1 + 1e-8:
                cuts[first_index].append(float(np.clip(first_fraction, 0, 1)))
                cuts[second_index].append(float(np.clip(second_fraction, 0, 1)))
    edges = set()
    for (start, end), fractions in zip(segments, cuts):
        points = [
            tuple(np.round(np.add(start, fraction * np.subtract(end, start)), 7))
            for fraction in sorted(set(fractions))
        ]
        edges.update(
            tuple(sorted((first, second)))
            for first, second in zip(points, points[1:])
            if first != second
        )
    representatives = []
    aliases = {}
    for point in sorted({point for edge in edges for point in edge}):
        nearby = next(
            (
                other
                for other in representatives
                if np.linalg.norm(np.subtract(point, other)) <= merge_distance
            ),
            None,
        )
        if nearby is None:
            representatives.append(point)
            nearby = point
        aliases[point] = nearby
    edges = {
        tuple(sorted((aliases[first], aliases[second])))
        for first, second in edges
        if aliases[first] != aliases[second]
    }
    return [[list(first), list(second)] for first, second in sorted(edges)]


def inside_or_on_boundary(point, polygon):
    inside = False
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        direction = np.subtract(end, start)
        fraction = np.clip(
            np.dot(np.subtract(point, start), direction) / np.dot(direction, direction), 0, 1
        )
        if np.linalg.norm(np.subtract(point, start) - fraction * direction) < 1e-5:
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing = start[0] + (point[1] - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            if point[0] < crossing:
                inside = not inside
    return inside


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    family = json.loads(source.joinpath("family.json").read_text())
    for filename in ("model.mdpa", "Materials.json"):
        shutil.copyfile(source / filename, destination / filename)
    traces = [
        [[145, 365], [259, 248]],
        [[145, 385], [258, 498]],
        [[145, 306], [205, 306], [205, 248]],
        [[145, 306], [205, 248]],
        [[145, 440], [194, 434], [205, 498]],
        [[145, 440], [205, 498]],
        [[278, 248], [303, 290], [303, 345], [303, 401], [303, 446], [280, 498]],
        [[388, 248], [358, 290], [358, 345], [358, 401], [358, 446], [393, 498]],
        [[303, 290], [358, 290]],
        [[303, 345], [358, 345]],
        [[303, 401], [358, 401]],
        [[303, 446], [358, 446]],
        [[303, 345], [358, 290]],
        [[303, 345], [358, 401]],
        [[303, 446], [358, 401]],
        [[303, 446], [330, 498], [358, 446]],
        [[303, 290], [330, 248], [358, 290]],
        [[303, 248], [303, 498]],
        [[358, 248], [358, 498]],
        [[319, 133], [350, 176], [430, 131], [403, 78]],
        [[350, 176], [393, 237], [507, 165], [584, 175]],
        [[350, 176], [404, 178], [430, 131]],
        [[350, 176], [363, 200], [404, 178], [393, 237]],
        [[404, 178], [433, 212]],
        [[404, 178], [476, 185], [430, 131]],
        [[453, 151], [476, 185], [507, 165]],
        [[485, 126], [507, 165], [548, 135], [526, 95]],
        [[485, 126], [548, 135]],
        [[783, 248], [832, 304], [887, 304], [946, 365]],
        [[832, 248], [832, 304]],
        [[887, 248], [887, 365], [946, 422]],
        [[832, 248], [887, 304]],
        [[887, 248], [946, 304]],
        [[887, 304], [946, 304]],
        [[946, 422], [917, 488], [944, 523]],
        [[946, 422], [863, 483], [809, 528], [743, 583], [684, 589]],
        [[684, 589], [715, 632], [767, 594], [809, 646], [863, 604], [919, 558], [989, 565]],
        [[743, 583], [767, 594]],
        [[809, 528], [767, 594]],
        [[809, 528], [863, 604]],
        [[863, 483], [919, 558]],
        [[917, 488], [919, 558]],
        [[944, 523], [919, 558]],
        [[715, 632], [769, 674], [809, 646], [807, 722]],
        [[767, 594], [769, 674]],
        [[506, 498], [525, 525], [575, 536], [642, 550]],
        [[525, 525], [548, 576], [575, 536], [616, 576]],
        [[548, 576], [575, 595]],
    ]
    for left_start, left_end, right_start, right_end, divisions in (
        ([595, 192], [430, 498], [637, 248], [508, 498], 8),
        ([703, 248], [637, 545], [745, 248], [684, 589], 6),
    ):
        left = np.linspace(left_start, left_end, divisions + 1)
        right = np.linspace(right_start, right_end, divisions + 1)
        traces.extend([left.tolist(), right.tolist()])
        for index in range(divisions + 1):
            traces.append([left[index].tolist(), right[index].tolist()])
        for index in range(divisions):
            traces.append(
                [left[index].tolist(), right[index + 1].tolist()]
                if index % 2
                else [right[index].tolist(), left[index + 1].tolist()]
            )

    def transform(point):
        affine = family["engineering_choices"].get("plan_affine")
        transformed = (
            np.array(affine) @ np.array([*point, 1.0])
            if affine
            else np.array([(point[0] - 145) * 115 / 844, (722 - point[1]) * 89 / 644])
        )
        nearest = None
        for start, end in zip(
            family["outer_polygon"], family["outer_polygon"][1:] + family["outer_polygon"][:1]
        ):
            direction = np.subtract(end, start)
            fraction = np.clip(
                np.dot(transformed - start, direction) / np.dot(direction, direction), 0, 1
            )
            projected = np.add(start, fraction * direction)
            distance = np.linalg.norm(projected - transformed)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, projected)
        return (
            nearest[1]
            if nearest[0] < 0.9 or not inside_or_on_boundary(transformed, family["outer_polygon"])
            else transformed
        ).tolist()

    outer_segments = [
        (transform(first), transform(second))
        for trace in traces
        for first, second in zip(trace, trace[1:])
    ]
    outer = family["outer_polygon"]
    outer_segments.extend(zip(outer, outer[1:] + outer[:1]))
    inner = family["inner_polygon"]
    inner_segments = list(zip(inner, inner[1:] + inner[:1]))
    declared = family["engineering_choices"].get("inner_strut_segments_xy_m")
    inner_segments.extend(
        declared if declared is not None else [(inner[0], inner[2]), (inner[1], inner[3])]
    )
    outer_edges = split_segments(outer_segments, merge_distance=0.07)
    outer_edges = [
        edge for edge in outer_edges if inside_or_on_boundary(np.mean(edge, axis=0), outer)
    ]
    family["braces"] = [
        {"system": "outer", "elevations_m": [6.5, 0.7, -3.3, -6.6], "segments_xy_m": outer_edges},
        {
            "system": "inner",
            "elevations_m": [-10.1],
            "segments_xy_m": split_segments(inner_segments),
            "crown_segments_xy_m": split_segments(list(zip(inner, inner[1:] + inner[:1]))),
        },
    ]
    defaults = {
        "trial_case": "pos_0m",
        "beam_section_m": {"width": 1.0, "height": 1.2},
        "beam_connections": "rigid joints; native interpolation to wall translations and rotations; installation offsets retained",
        "outer_beam_trace": "approximate centerline digitization of visible monitoring-figure trusses, not a recovered CAD graph",
        "outer_beam_polylines_pixels": traces,
        "trace_cleanup": "project exterior trace vertices to retained wall polygon; split intersections and omit exterior portions; merge centers within 0.07m (half a raster pixel), never modify soil or wall geometry",
        "inner_brace_choice": "two crossing diagonals joined at crossing plus perimeter; undimensioned inner-level layout is an explicit trial engineering choice, not recovered historical geometry",
        "hydraulics": "uniform prescribed hydrostatic water table over entire domain; max 2m drawdown substeps; no transient seepage",
        "support_centroids": "use section elevation annotations as trial beam centroid elevations",
        "solver_controls": {
            "residual_relative_tolerance": 1e-6,
            "residual_absolute_tolerance": 1e-5,
            "displacement_relative_tolerance": 1e-6,
            "displacement_absolute_tolerance": 1e-8,
        },
        "unresolved": [
            "source trace/inner 92 by 32 m dimension and bracing review",
            "DBC/CX registration",
            "outer wall extension from +6.50 to +7.00",
            "mesh, hydraulic and domain independence",
        ],
    }
    defaults.update(family["engineering_choices"])
    family["engineering_choices"] = defaults
    destination.joinpath("family.json").write_text(json.dumps(family, indent=2))
    return {
        "case": "pos_0m",
        "soil_mesh_unchanged": True,
        "outer_segments": len(family["braces"][0]["segments_xy_m"]),
        "inner_segments": len(family["braces"][1]["segments_xy_m"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.source, arguments.destination)))
