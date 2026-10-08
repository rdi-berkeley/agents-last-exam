import csv
import hashlib
import math
from pathlib import Path
import sys

import FreeCAD
import numpy as np


TERRAIN_HASH = "f5b136d233c8bb62bfbb6f1cfba44a48470a6cccf4077a7e883c9286b9f356df"


def load_road(runtime):
    sys.path.insert(0, str(Path(runtime) / "road-python"))
    sys.path.insert(0, str(Path(runtime) / "Road"))
    import freecad

    freecad.__path__.append(str(Path(runtime) / "Road/freecad"))
    from road_compat import install
    install(read_only=True)


def source_mesh(path):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != TERRAIN_HASH:
        raise ValueError("Fixed terrain hash mismatch")
    vertices = []
    faces = []
    for line in content.decode().splitlines():
        parts = line.split()
        if parts and parts[0] == "v":
            vertices.append([float(value) for value in parts[1:]])
        elif parts and parts[0] == "f":
            faces.append([int(value) - 1 for value in parts[1:]])
    return np.array(vertices), np.array(faces)


def world_point(alignment, station):
    point = alignment.Model.get_point_at_station(station)
    return alignment.Model.coordinate_system.transform_from_system(point)[::-1]


def independent_ground(vertices, faces, queries):
    triangles = vertices[faces]
    first = triangles[:, 1, :2] - triangles[:, 0, :2]
    second = triangles[:, 2, :2] - triangles[:, 0, :2]
    determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    elevations = []
    for query in queries:
        relative = np.array(query) - triangles[:, 0, :2]
        weight_first = (relative[:, 0] * second[:, 1] - relative[:, 1] * second[:, 0]) / determinant
        weight_second = (first[:, 0] * relative[:, 1] - first[:, 1] * relative[:, 0]) / determinant
        selected = np.flatnonzero((weight_first >= -1e-9) & (weight_second >= -1e-9) & (weight_first + weight_second <= 1 + 1e-9))
        if not len(selected):
            raise ValueError("Route leaves fixed terrain domain")
        index = selected[0]
        heights = triangles[index, :, 2]
        elevations.append(float(heights[0] + weight_first[index] * (heights[1] - heights[0]) + weight_second[index] * (heights[2] - heights[0])))
    return elevations


def terrain_integrity(terrain, vertices, faces):
    saved = np.array([terrain.Points[str(index)] for index in range(len(vertices))]) / 1000
    saved_faces = np.array([[int(value) for value in face] for face in terrain.Faces["Visible"]])
    if not np.array_equal(saved_faces, faces) or np.max(np.abs(saved - vertices)) > 1e-10:
        raise ValueError("Saved native terrain differs from fixed source")
    if terrain.Mesh.CountFacets != len(faces) or terrain.Operations or terrain.Clusters:
        raise ValueError("Unexpected native terrain topology/operations")
    origin = terrain.Geolocation.Base.sub(terrain.Placement.Base)
    maximum = 0.0
    for facet, indices in zip(terrain.Mesh.Facets, faces):
        for actual, expected in zip(facet.Points, vertices[indices]):
            recovered = [(actual[0] + origin.x) / 1000, (actual[1] + origin.y) / 1000, actual[2] / 1000]
            maximum = max(maximum, math.dist(recovered, expected))
    if maximum > 0.001:
        raise ValueError(f"Native mesh placement or precision differs: {maximum}")
    return maximum


def verify(request):
    vertices, faces = source_mesh(request["terrain"])
    source_document = FreeCAD.openDocument(str(Path(request["terrain"]).with_suffix(".FCStd")))
    source_terrains = [obj for obj in source_document.Objects if getattr(getattr(obj, "Proxy", None), "Type", None) == "Road::Terrain"]
    if len(source_terrains) != 1 or any(getattr(getattr(obj, "Proxy", None), "Type", None) in {"Road::Alignment", "Road::ProfileFrame"} for obj in source_document.Objects):
        raise ValueError("Native source input must contain only the fixed terrain, without route answers")
    terrain_integrity(source_terrains[0], vertices, faces)
    FreeCAD.closeDocument(source_document.Name)
    document = FreeCAD.openDocument(request["alignment"])
    alignments = [obj for obj in document.Objects if getattr(getattr(obj, "Proxy", None), "Type", None) == "Road::Alignment"]
    terrains = [obj for obj in document.Objects if getattr(getattr(obj, "Proxy", None), "Type", None) == "Road::Terrain"]
    if len(alignments) != 1 or len(terrains) != 1:
        raise ValueError("Need one native Road alignment and its fixed source terrain")
    alignment, terrain = alignments[0], terrains[0]
    mesh_error = terrain_integrity(terrain, vertices, faces)
    elements = alignment.Model.get_elements()
    if len(alignment.Shape.Edges) != len(elements):
        raise ValueError("Native shape/model element mismatch")
    for edge, element in zip(alignment.Shape.Edges, elements):
        if abs(edge.Length / 1000 - element.get_length()) > 1e-6:
            raise ValueError("Native edge/model length mismatch")
        if element.get_type() == "Curve" and (not hasattr(edge.Curve, "Radius") or abs(edge.Curve.Radius / 1000 - element.radius) > 1e-7):
            raise ValueError("Native edge is not the declared circular arc")
        for fraction in (0, 0.5, 1):
            point = edge.valueAt(edge.FirstParameter + fraction * (edge.LastParameter - edge.FirstParameter))
            expected = alignment.Model.coordinate_system.transform_to_system(element.get_point_at_distance(element.get_length() * fraction))
            expected = FreeCAD.Vector(*expected).multiply(1000).add(alignment.Placement.Base)
            if point.distanceToPoint(expected) > 0.001:
                raise ValueError("Native edge placement/model mismatch")
    profiles = alignment.Model.get_profiles().surface_profiles
    frames = [obj for obj in document.Objects if getattr(getattr(obj, "Proxy", None), "Type", None) == "Road::ProfileFrame" and terrain in obj.Terrains and obj.getParentGroup().getParentGroup() == alignment]
    if len(profiles) != 1 or not frames or not frames[0].Shape.Edges:
        raise ValueError("Missing associated native terrain profile")
    length = alignment.Model.get_length()
    start = elements[0].get_start_point()
    end = elements[-1].get_end_point()
    with Path(request["tsv"]).open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        headers = reader.fieldnames
        rows = list(reader)
    stations = [float(row["Station"]) for row in rows]
    expected_stations = list(np.arange(0, length, 20)) + [length]
    if len(stations) != len(expected_stations) or any(abs(actual - expected) > 0.5 for actual, expected in zip(stations, expected_stations)):
        raise ValueError("TSV does not cover the saved route at 20 m stations including its end")
    queries = [world_point(alignment, min(length, max(0, station))) for station in stations]
    xy_errors = [math.dist(point, (float(row["Y"]), float(row["X"]))) for point, row in zip(queries, rows)]
    if not all(math.isfinite(float(value)) for row in rows for value in row.values()) or max(xy_errors) > 0.001:
        raise ValueError("TSV XY does not belong to saved native alignment")
    reference_heights = independent_ground(vertices, faces, queries)
    profile_heights = [profiles[0].get_elevation_at_station(min(length, max(0, station))) for station in stations]
    profile_error = max(abs(actual - expected) for actual, expected in zip(profile_heights, reference_heights))
    if profile_error > 0.2:
        raise ValueError(f"Native profile differs from fixed source terrain: {profile_error}")
    dense_stations = np.linspace(0, length, math.ceil(length / 0.25) + 1)
    dense_heights = independent_ground(vertices, faces, [world_point(alignment, float(station)) for station in dense_stations])
    dense_profile_error = max(abs(profiles[0].get_elevation_at_station(float(station)) - expected) for station, expected in zip(dense_stations, dense_heights))
    if dense_profile_error > 0.2:
        raise ValueError(f"Native profile interpolation differs from source: {dense_profile_error}")
    curves = [element.to_dict() for element in elements if element.get_type() == "Curve"]
    spirals = [element.to_dict() for element in elements if element.get_type() == "Spiral"]
    return {"native_road_verified": True, "terrain_sha256": TERRAIN_HASH, "alignment_info": {"start_x": start[0], "start_y": start[1], "end_x": end[0], "end_y": end[1], "length": length, "n_curves": len(curves), "curves": curves, "spirals": spirals}, "profile_info": {"count": len(profiles)}, "tsv_exists": True, "tsv_headers": headers, "tsv_row_count": len(rows), "surface_elevations": reference_heights, "tsv_rows": rows, "native_checks": {"fresh_process_reopen": True, "shape_edges": len(alignment.Shape.Edges), "mesh_vertices": terrain.Mesh.CountPoints, "mesh_faces": terrain.Mesh.CountFacets, "mesh_max_vertex_error_m": mesh_error, "tsv_max_xy_error_m": max(xy_errors), "profile_station_max_z_error_m": profile_error, "dense_profile_max_z_error_m": dense_profile_error, "dense_samples": len(dense_stations), "native_shape_length_m": alignment.Shape.Length / 1000}}


def run(request):
    load_road(request["runtime"])
    return verify(request)
