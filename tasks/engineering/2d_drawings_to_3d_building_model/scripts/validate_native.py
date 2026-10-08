"""Compare evaluated Blender surfaces with the submitted OBJ in millimeters."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
import numpy as np


def evaluated_surface():
    vertices = []
    triangles = []
    graph = bpy.context.evaluated_depsgraph_get()
    for instance in graph.object_instances:
        instance_object = instance.object
        if instance_object.hide_render or instance_object.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
            continue
        mesh = instance_object.to_mesh()
        if mesh is None:
            continue
        try:
            mesh.calc_loop_triangles()
            offset = len(vertices)
            vertices.extend(tuple(instance.matrix_world @ vertex.co) for vertex in mesh.vertices)
            triangles.extend(tuple(offset + index for index in triangle.vertices) for triangle in mesh.loop_triangles)
        finally:
            instance_object.to_mesh_clear()
    points = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    if not len(faces) or not np.isfinite(points).all():
        raise ValueError("Geometry must contain finite, nonempty surfaces")
    return points, faces


def corresponding_triangles(source, source_faces, target, target_faces):
    if len(source_faces) != len(target_faces):
        return None
    tree = KDTree(len(target))
    for index, point in enumerate(target):
        tree.insert(Vector(point), index)
    tree.balance()
    queries = [tree.find(Vector(point)) for point in source]
    maximum = max(query[2] for query in queries)
    if maximum > 1.0:
        return None
    source_labels = [query[1] for query in queries]
    target_labels = [tree.find(Vector(point))[1] for point in target]
    source_triangles = Counter(tuple(sorted(source_labels[index] for index in face)) for face in source_faces)
    target_triangles = Counter(tuple(sorted(target_labels[index] for index in face)) for face in target_faces)
    return maximum if source_triangles == target_triangles else None


def triangle_distances(points, triangles):
    offsets = np.asarray(points)[..., None, :] - triangles
    edges = np.roll(triangles, -1, axis=1) - triangles
    lengths = np.sum(edges * edges, axis=2)
    factors = np.divide(np.sum(offsets * edges, axis=2), lengths, out=np.zeros_like(lengths), where=lengths > 0)
    residuals = offsets - np.clip(factors, 0, 1)[..., None] * edges
    edge_distances = np.sum(residuals * residuals, axis=2).min(axis=1)
    normals = np.cross(edges[:, 0], -edges[:, 2])
    normal_squared = np.sum(normals * normals, axis=1)
    numerators = np.sum(np.cross(offsets, np.roll(offsets, -1, axis=1)) * normals[:, None, :], axis=2)
    inside = (numerators >= -1e-12 * normal_squared[:, None]).all(axis=1) & (normal_squared > 1e-24)
    signed = np.sum(offsets[:, 0] * normals, axis=1)
    plane_distances = np.divide(signed * signed, normal_squared, out=np.full_like(signed, np.inf), where=normal_squared > 1e-24)
    return np.sqrt(np.where(inside, np.minimum(edge_distances, plane_distances), edge_distances))


def sampled_distance(source, source_faces, target, target_faces, origin):
    source = source - origin
    target = target - origin
    tree = BVHTree.FromPolygons([Vector(point) for point in target], target_faces.tolist(), all_triangles=True)
    source_triangles = source[source_faces]
    areas = np.linalg.norm(np.cross(source_triangles[:, 1] - source_triangles[:, 0], source_triangles[:, 2] - source_triangles[:, 0]), axis=1)
    useful = areas > 1e-6
    if not useful.any():
        raise ValueError("Geometry contains no nondegenerate surfaces")
    generator = np.random.default_rng(1729)
    selected = source_triangles[generator.choice(len(areas), 12000, p=areas / areas.sum())]
    first = np.sqrt(generator.random((len(selected), 1)))
    second = generator.random((len(selected), 1))
    samples = (1 - first) * selected[:, 0] + first * (1 - second) * selected[:, 1] + first * second * selected[:, 2]
    queries = np.vstack((source[np.unique(source_faces[useful])], source_triangles[useful].mean(axis=1), samples))
    nearest_faces = []
    for point in queries:
        nearest = tree.find_nearest(Vector(point))
        if nearest[0] is None:
            raise ValueError("Surface comparison failed")
        nearest_faces.append(nearest[2])
    target_triangles = target[target_faces]
    distances = triangle_distances(queries, target_triangles[nearest_faces])
    largest = float(distances[distances <= 1].max(initial=0))
    lower = target_triangles.min(axis=1)
    upper = target_triangles.max(axis=1)
    for index in np.flatnonzero(distances > 1):
        point = queries[index]
        outside = np.maximum(np.maximum(lower - point, point - upper), 0)
        candidates = np.sum(outside * outside, axis=1) <= distances[index] ** 2 + 1e-8
        exact = float(triangle_distances(point, target_triangles[candidates]).min())
        largest = max(largest, exact)
        if largest > 1:
            return largest
    return largest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--blend", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    try:
        bpy.ops.wm.open_mainfile(filepath=args.blend, load_ui=False, use_scripts=False)
        for obj in bpy.context.scene.objects:
            for modifier in obj.modifiers:
                modifier.show_viewport = modifier.show_render
        native_points, native_faces = evaluated_surface()
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.obj_import(filepath=args.obj, forward_axis="Y", up_axis="Z")
        obj_points, obj_faces = evaluated_surface()
        origin = native_points.mean(axis=0)
        equivalent = corresponding_triangles(native_points - origin, native_faces, obj_points - origin, obj_faces)
        distances = {"native_to_obj_mm": equivalent, "obj_to_native_mm": equivalent}
        if equivalent is None:
            distances = {
                "native_to_obj_mm": sampled_distance(native_points, native_faces, obj_points, obj_faces, origin),
                "obj_to_native_mm": sampled_distance(obj_points, obj_faces, native_points, native_faces, origin),
            }
        report = {"valid": max(distances.values()) <= 1.0, "tolerance_mm": 1.0, **distances,
                  "native_triangles": len(native_faces), "obj_triangles": len(obj_faces)}
    except (ValueError, RuntimeError, OSError) as error:
        report = {"valid": False, "error": str(error)}
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
