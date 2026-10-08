"""Compare native stock surfaces in regions frozen from the public workpiece."""

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import shapely
import trimesh


TOLERANCES_MM = (0.3, 2.0)
WEIGHTS = (0.7, 0.3)
RECORD = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binary_surface(path):
    path = Path(path)
    with path.open("rb") as stream:
        stream.seek(80)
        header = stream.read(4)
    if len(header) != 4:
        raise ValueError("Missing binary STL header")
    count = struct.unpack("<I", header)[0]
    if count == 0 or path.stat().st_size != 84 + count * RECORD.itemsize:
        raise ValueError("Expected a complete nonempty binary STL")
    return np.memmap(path, dtype=RECORD, mode="r", offset=84, shape=(count,))


def prepare_region(public_meshes, stock_bounds, output):
    bounds = np.asarray(stock_bounds, dtype=float).reshape(2, 3)
    if not np.isfinite(bounds).all() or np.any(bounds[1] - bounds[0] <= 0.6):
        raise ValueError("Invalid original stock bounds")
    footprints = []
    sources = []
    for path in public_meshes:
        mesh = trimesh.load_mesh(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not np.isfinite(mesh.vertices).all():
            raise ValueError("Public machining input is not a finite surface mesh")
        for axis in range(3):
            normal = np.eye(3)[axis]
            lower = bounds[0].copy()
            upper = bounds[1].copy()
            lower[axis] += TOLERANCES_MM[0]
            upper[axis] -= TOLERANCES_MM[0]
            mesh = mesh.slice_plane(lower, normal, cap=False)
            mesh = mesh.slice_plane(upper, -normal, cap=False)
        for offset in range(0, len(mesh.faces), 10000):
            triangles = mesh.vertices[mesh.faces[offset : offset + 10000]][:, :, :2]
            polygons = shapely.polygons(triangles)
            polygons = polygons[shapely.area(polygons) > 0]
            if len(polygons):
                footprints.append(shapely.union_all(polygons))
        sources.append({"name": Path(path).name, "sha256": file_hash(path)})
    footprint = shapely.union_all(footprints)
    if footprint.is_empty or not footprint.is_valid or footprint.area <= 0:
        raise ValueError("Original geometry has no resolved machining region")
    record = {
        "sources": sources,
        "stock_bounds_mm": bounds.reshape(-1).tolist(),
        "boundary_clearance_mm": TOLERANCES_MM[0],
        "footprint": json.loads(shapely.to_geojson(footprint)),
        "footprint_area_mm2": footprint.area,
        "definition": "XY projection of original surfaces inside the stock's 0.3 mm boundary",
        "derived_from_reference_or_candidate": False,
    }
    Path(output).write_text(json.dumps(record, indent=2) + "\n")
    return record


def region_mask(points, region):
    bounds = np.asarray(region["stock_bounds_mm"], dtype=float).reshape(2, 3)
    clearance = region["boundary_clearance_mm"]
    footprint = shapely.from_geojson(json.dumps(region["footprint"]))
    inside = np.all((points > bounds[0] + clearance) & (points < bounds[1] - clearance), axis=1)
    return inside & shapely.intersects_xy(footprint, points[:, 0], points[:, 1])


def sample_regions(path, region, *, count=10000, seed=1901, max_batches=100):
    records = binary_surface(path)
    areas = np.empty(len(records))
    for offset in range(0, len(records), 100000):
        triangles = records["vertices"][offset : offset + 100000].astype(float)
        if not np.isfinite(triangles).all():
            raise ValueError("Nonfinite native stock mesh")
        areas[offset : offset + len(triangles)] = (
            np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=1,
            )
            / 2
        )
    cumulative = np.cumsum(areas)
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0:
        raise ValueError("Native stock has no finite surface area")
    generator = np.random.default_rng(seed)
    selected = {"machining": [], "remainder": []}
    totals = {name: 0 for name in selected}
    sampled = 0
    for _ in range(max_batches):
        triangles = records["vertices"][
            np.searchsorted(cumulative, generator.random(count) * cumulative[-1], side="right")
        ].astype(float)
        weights = generator.random((count, 2))
        flipped = weights.sum(axis=1) > 1
        weights[flipped] = 1 - weights[flipped]
        points = (
            triangles[:, 0]
            + weights[:, :1] * (triangles[:, 1] - triangles[:, 0])
            + weights[:, 1:] * (triangles[:, 2] - triangles[:, 0])
        )
        machining = region_mask(points, region)
        sampled += len(points)
        for name, mask in (("machining", machining), ("remainder", ~machining)):
            part = points[mask][: max(0, count - totals[name])]
            selected[name].append(part)
            totals[name] += len(part)
        if all(total == count for total in totals.values()):
            break
    return {name: np.concatenate(parts) for name, parts in selected.items()}, sampled


def surface_distances(path, points, *, chunk_size=200000):
    import fcl

    records = binary_surface(path)
    minimum = np.full(len(points), np.inf)
    if not len(points):
        return minimum
    probes = [fcl.CollisionObject(fcl.Sphere(1e-9), fcl.Transform(point)) for point in points]
    for offset in range(0, len(records), chunk_size):
        vertices = np.asarray(
            records["vertices"][offset : offset + chunk_size], dtype=float
        ).reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
        model = fcl.BVHModel()
        model.beginModel(len(vertices), len(faces))
        model.addSubModel(vertices, faces)
        model.endModel()
        target = fcl.CollisionObject(model)
        distances = np.asarray(
            [
                fcl.distance(probe, target, fcl.DistanceRequest(), fcl.DistanceResult())
                for probe in probes
            ]
        )
        np.minimum(minimum, distances, out=minimum)
    return minimum


def compare_surfaces(candidate, reference, region, *, count=10000):
    reference_points, reference_sampled = sample_regions(reference, region, count=count)
    candidate_points, candidate_sampled = sample_regions(candidate, region, count=count)
    if not len(reference_points["machining"]):
        raise ValueError("Reference has no surface in the frozen public machining region")
    rows = {}
    for name in reference_points:
        expected = reference_points[name]
        actual = candidate_points[name]
        if not len(expected) and not len(actual):
            continue
        forward = surface_distances(candidate, expected)
        reverse = surface_distances(reference, actual)
        coverage = [
            float(np.mean(forward <= tolerance)) if len(forward) else 0.0
            for tolerance in TOLERANCES_MM
        ]
        conformity = [
            float(np.mean(reverse <= tolerance)) if len(reverse) else 0.0
            for tolerance in TOLERANCES_MM
        ]
        ratios = [min(before, after) for before, after in zip(coverage, conformity)]
        rows[name] = {
            "reference_samples": len(expected),
            "candidate_samples": len(actual),
            "coverage": coverage,
            "conformity": conformity,
            "score": sum(weight * ratio for weight, ratio in zip(WEIGHTS, ratios)),
        }
    return {
        "geometry_score": min(row["score"] for row in rows.values()),
        "regions": rows,
        "tolerances_mm": TOLERANCES_MM,
        "weights": WEIGHTS,
        "reference_sha256": file_hash(reference),
        "candidate_sha256": file_hash(candidate),
        "region_sha256": hashlib.sha256(json.dumps(region, sort_keys=True).encode()).hexdigest(),
        "reference_draws": reference_sampled,
        "candidate_draws": candidate_sampled,
        "safety_checked": False,
    }


def grade_replay(replay_report, reference, region, *, count=10000):
    status = replay_report.get("status")
    if status == "invalid_delivery":
        return {"score": 0.0, "reason": replay_report.get("error", "Native safety gate failed")}
    if status != "native_replay_complete":
        raise RuntimeError("Native replay is incomplete or unavailable; no score can be assigned")
    safety = replay_report["safety"]
    stock = replay_report["stock"]
    if (
        safety.get("status") != "clear"
        or safety.get("collision_free") is not True
        or safety.get("whole_path_checked") is not True
        or safety.get("tool_events_checked") is not True
        or safety.get("motion_count") != stock.get("motion_count")
        or not safety.get("motions_sha256")
        or safety["motions_sha256"] != stock.get("motions_sha256")
    ):
        raise RuntimeError("Complete native safety coverage is required before geometry scoring")
    candidate = Path(stock["output"])
    if file_hash(candidate) != stock["output_sha256"]:
        raise RuntimeError("Replayed stock changed after native simulation")
    result = compare_surfaces(candidate, reference, region, count=count)
    result.update(
        score=result["geometry_score"],
        safety_checked=True,
        saved_project_sha256=replay_report["project"]["project_sha256"],
        executed_program_sha256=replay_report["canonical"]["input_sha256"],
    )
    return result
