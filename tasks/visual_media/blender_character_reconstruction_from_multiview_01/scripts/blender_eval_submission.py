from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-obj", required=True)
    parser.add_argument("--views-config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--render-dir", required=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablock in [bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights]:
        for item in list(datablock):
            if item.users == 0:
                datablock.remove(item)


def _import_obj(filepath: str) -> list[bpy.types.Object]:
    before = {obj.name for obj in bpy.data.objects}
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=filepath)
    else:
        bpy.ops.import_scene.obj(filepath=filepath)
    return [
        obj for obj in bpy.data.objects if obj.name not in before and obj.type == "MESH"
    ]


def _bbox_world(meshes: list[bpy.types.Object]) -> tuple[list[float], list[float], list[float], list[float]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in meshes:
        obj_eval = obj.evaluated_get(depsgraph)
        for corner in obj_eval.bound_box:
            world = obj_eval.matrix_world @ Vector(corner)
            for axis in range(3):
                mins[axis] = min(mins[axis], world[axis])
                maxs[axis] = max(maxs[axis], world[axis])
    center = (mins + maxs) * 0.5
    extent = maxs - mins
    return (
        [float(value) for value in mins],
        [float(value) for value in maxs],
        [float(value) for value in center],
        [float(value) for value in extent],
    )


def _topology_counts(meshes: list[bpy.types.Object]) -> tuple[int, int]:
    vertex_count = 0
    face_count = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in meshes:
        mesh = obj.evaluated_get(depsgraph).to_mesh()
        vertex_count += len(mesh.vertices)
        face_count += len(mesh.polygons)
        obj.evaluated_get(depsgraph).to_mesh_clear()
    return vertex_count, face_count


def _setup_world(views: dict) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = int(views["render_resolution"][0])
    scene.render.resolution_y = int(views["render_resolution"][1])
    scene.render.image_settings.file_format = "PNG"
    try:
        scene.render.engine = str(views["render_engine"])
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"

    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = str(views["color_management"]["view_transform"])
    scene.view_settings.look = str(views["color_management"]["look"])
    scene.view_settings.exposure = float(views["color_management"]["exposure"])
    scene.view_settings.gamma = float(views["color_management"]["gamma"])

    if scene.world is None:
        scene.world = bpy.data.worlds.new("EvalWorld")
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    background = nodes.get("Background")
    background.inputs[0].default_value = tuple(views["background_color"])
    background.inputs[1].default_value = float(views["background_strength"])


def _create_clay_material(views: dict) -> bpy.types.Material:
    spec = views["material_override_policy"]
    material = bpy.data.materials.new(spec["material_name"])
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    principled.inputs["Base Color"].default_value = tuple(spec["base_color_rgba"])
    principled.inputs["Roughness"].default_value = float(spec["roughness"])
    principled.inputs["Specular IOR Level"].default_value = float(spec["specular_ior_level"])
    return material


def _create_lights(views: dict) -> None:
    for spec in views["lighting_settings"]["lights"]:
        light_data = bpy.data.lights.new(spec["name"], type=spec["type"])
        light_data.energy = float(spec["energy"])
        if hasattr(light_data, "shape"):
            light_data.shape = "SQUARE"
        if hasattr(light_data, "size"):
            light_data.size = float(spec["size"])
        light_obj = bpy.data.objects.new(spec["name"], light_data)
        light_obj.location = tuple(spec["position"])
        light_obj.rotation_euler = tuple(spec["rotation_euler"])
        bpy.context.scene.collection.objects.link(light_obj)


def _create_cameras_and_render(views: dict, render_dir: Path) -> None:
    scene = bpy.context.scene
    render_dir.mkdir(parents=True, exist_ok=True)
    for view in views["per_view_cameras"]:
        camera_data = bpy.data.cameras.new(view["camera_name"])
        camera_data.type = view["camera_type"]
        if camera_data.type == "ORTHO":
            camera_data.ortho_scale = float(view["orthographic_scale"])
        camera_data.clip_start = float(view["clip_range"][0])
        camera_data.clip_end = float(view["clip_range"][1])
        camera_obj = bpy.data.objects.new(view["camera_name"], camera_data)
        camera_obj.location = tuple(view["camera_position"])
        camera_obj.rotation_euler = tuple(view["camera_rotation_euler"])
        bpy.context.scene.collection.objects.link(camera_obj)
        scene.camera = camera_obj
        scene.render.filepath = str(render_dir / Path(str(view["output_image_path"])).name)
        bpy.ops.render.render(write_still=True)


def main() -> None:
    args = parse_args()
    _clear_scene()
    meshes = _import_obj(args.submission_obj)
    if not meshes:
        raise RuntimeError("OBJ import produced no mesh objects")

    views = json.loads(Path(args.views_config).read_text(encoding="utf-8"))
    _setup_world(views)
    _create_lights(views)
    clay = _create_clay_material(views)
    bpy.context.view_layer.material_override = clay
    _create_cameras_and_render(views, Path(args.render_dir))

    bbox_min, bbox_max, bbox_center, bbox_extent = _bbox_world(meshes)
    vertex_count, face_count = _topology_counts(meshes)
    diagonal = math.sqrt(sum(float(value) * float(value) for value in bbox_extent))
    payload = {
        "mesh_object_count": len(meshes),
        "vertex_count": int(vertex_count),
        "face_count": int(face_count),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_center": bbox_center,
        "bbox_extent": bbox_extent,
        "bbox_diagonal": diagonal,
        "render_dir": str(Path(args.render_dir)),
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
