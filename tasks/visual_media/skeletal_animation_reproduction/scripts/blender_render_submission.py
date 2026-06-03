from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=500)
    parser.add_argument("--evaluation-config")
    return parser.parse_args(argv)


def _find_primary_armature() -> bpy.types.Object | None:
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if not armatures:
        return None

    def score(obj: bpy.types.Object) -> tuple[int, int]:
        linked_meshes = 0
        for mesh in bpy.data.objects:
            if mesh.type != "MESH":
                continue
            for modifier in mesh.modifiers:
                if modifier.type == "ARMATURE" and modifier.object == obj:
                    linked_meshes += 1
        return (linked_meshes, len(obj.data.bones))

    return max(armatures, key=score)


def _find_target_meshes(armature: bpy.types.Object | None) -> list[bpy.types.Object]:
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if armature is None:
        return meshes
    linked = []
    for mesh in meshes:
        for modifier in mesh.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == armature:
                linked.append(mesh)
                break
    return linked or meshes


def _has_animation(armature: bpy.types.Object | None) -> bool:
    if bpy.data.actions:
        return True
    if armature and armature.animation_data:
        if armature.animation_data.action is not None:
            return True
        if armature.animation_data.nla_tracks:
            return True
    return False


def _animation_frame_range() -> tuple[int, int]:
    starts = [int(bpy.context.scene.frame_start)]
    ends = [int(bpy.context.scene.frame_end)]
    for action in bpy.data.actions:
        starts.append(int(math.floor(action.frame_range[0])))
        ends.append(int(math.ceil(action.frame_range[1])))
    return (min(starts), max(ends))


def _frame_at_position(start: int, end: int, position: float) -> int:
    span = max(end - start, 1)
    return start + int(round(position * span))


def _sample_frames(start: int, end: int, sample_count: int) -> tuple[list[int], list[float]]:
    if sample_count <= 1:
        return [start], [0.0]
    frames = []
    positions = []
    span = max(end - start, 1)
    for idx in range(sample_count):
        pos = idx / (sample_count - 1)
        frame = start + int(round(pos * span))
        frames.append(frame)
        positions.append(pos)
    return frames, positions


def _evaluated_bbox_world(meshes: list[bpy.types.Object], depsgraph) -> tuple[Vector, Vector]:
    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    for mesh in meshes:
        obj_eval = mesh.evaluated_get(depsgraph)
        for corner in obj_eval.bound_box:
            world = obj_eval.matrix_world @ Vector(corner)
            for axis in range(3):
                mins[axis] = min(mins[axis], world[axis])
                maxs[axis] = max(maxs[axis], world[axis])
    return mins, maxs


def _look_at(camera_obj: bpy.types.Object, target: Vector) -> None:
    direction = target - camera_obj.location
    camera_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _ensure_world(color: tuple[float, float, float, float]) -> None:
    scene = bpy.context.scene
    if scene.world is None:
        scene.world = bpy.data.worlds.new("EvalWorld")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = color
        bg.inputs[1].default_value = 1.0


def _ensure_camera() -> bpy.types.Object:
    scene = bpy.context.scene
    if scene.camera:
        return scene.camera
    cam_data = bpy.data.cameras.new("EvalCamera")
    cam_obj = bpy.data.objects.new("EvalCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    return cam_obj


def _ensure_light() -> None:
    if any(obj.type == "LIGHT" for obj in bpy.data.objects):
        return
    light_data = bpy.data.lights.new("EvalSun", type="SUN")
    light_data.energy = 2.0
    light_obj = bpy.data.objects.new("EvalSun", light_data)
    light_obj.location = (4.0, -4.0, 8.0)
    light_obj.rotation_euler = (0.9, 0.0, 0.8)
    bpy.context.scene.collection.objects.link(light_obj)


def _build_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    shader = nodes.new(type="ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = 1.0
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def _build_emission_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    shader = nodes.new(type="ShaderNodeEmission")
    shader.inputs["Color"].default_value = color
    shader.inputs["Strength"].default_value = 1.0
    links.new(shader.outputs["Emission"], output.inputs["Surface"])
    return material


def _assign_override(meshes: list[bpy.types.Object], material: bpy.types.Material) -> dict[str, list[bpy.types.Material | None]]:
    original: dict[str, list[bpy.types.Material | None]] = {}
    for mesh in meshes:
        original[mesh.name] = [slot.material for slot in mesh.material_slots]
        if mesh.material_slots:
            for slot in mesh.material_slots:
                slot.material = material
        else:
            mesh.data.materials.append(material)
    return original


def _restore_materials(meshes: list[bpy.types.Object], original: dict[str, list[bpy.types.Material | None]]) -> None:
    for mesh in meshes:
        mats = original.get(mesh.name, [])
        for idx, material in enumerate(mats):
            if idx < len(mesh.material_slots):
                mesh.material_slots[idx].material = material


def _resolve_bone_mapping(armature: bpy.types.Object, package: dict[str, object]) -> tuple[dict[str, str], list[str]]:
    resolved: dict[str, str] = {}
    missing: list[str] = []
    bone_names = {bone.name for bone in armature.data.bones}
    semantics = package.get("bone_semantics", {}) if isinstance(package, dict) else {}
    for semantic in package.get("required_bones", []) if isinstance(package, dict) else []:
        info = semantics.get(semantic, {}) if isinstance(semantics, dict) else {}
        candidates = [info.get("primary")] + list(info.get("aliases", []))
        candidates.append(str(semantic))
        matched = next((name for name in candidates if name and name in bone_names), None)
        if matched:
            resolved[str(semantic)] = str(matched)
        else:
            missing.append(str(semantic))
    return resolved, missing


def _pose_bone_state(armature: bpy.types.Object, bone_name: str, center: Vector, diag: float) -> dict[str, list[float]]:
    pose_bone = armature.pose.bones[bone_name]
    head = armature.matrix_world @ pose_bone.head
    tail = armature.matrix_world @ pose_bone.tail
    scale = max(diag, 1e-6)
    head_n = [(head[i] - center[i]) / scale for i in range(3)]
    tail_n = [(tail[i] - center[i]) / scale for i in range(3)]
    direction = Vector((tail_n[0] - head_n[0], tail_n[1] - head_n[1], tail_n[2] - head_n[2]))
    if direction.length > 1e-6:
        direction.normalize()
    return {
        "head": [float(v) for v in head_n],
        "tail": [float(v) for v in tail_n],
        "direction": [float(v) for v in direction],
    }


def _compute_joint_ranges(
    *,
    armature: bpy.types.Object,
    bone_mapping: dict[str, str],
    frame_positions: list[float],
    frame_start: int,
    frame_end: int,
    scene: bpy.types.Scene,
    depsgraph,
) -> dict[str, dict[str, float]]:
    samples: dict[str, list[list[float]]] = {semantic: [] for semantic in bone_mapping}
    for position in frame_positions:
        frame = _frame_at_position(frame_start, frame_end, float(position))
        scene.frame_set(frame)
        depsgraph.update()
        for semantic, bone_name in bone_mapping.items():
            pose_bone = armature.pose.bones[bone_name]
            euler = pose_bone.matrix_basis.to_euler("XYZ")
            samples[semantic].append([float(euler.x), float(euler.y), float(euler.z)])

    joint_ranges: dict[str, dict[str, float]] = {}
    for semantic, rows in samples.items():
        if not rows:
            joint_ranges[semantic] = {"x": 0.0, "y": 0.0, "z": 0.0, "magnitude": 0.0}
            continue
        arr = np.unwrap(np.array(rows, dtype=np.float32), axis=0)
        ranges = arr.max(axis=0) - arr.min(axis=0)
        x, y, z = [float(v) for v in ranges]
        joint_ranges[semantic] = {
            "x": x,
            "y": y,
            "z": z,
            "magnitude": float(math.sqrt(x * x + y * y + z * z)),
        }
    return joint_ranges


def _compute_bbox_shape_motion_extent(
    frame_bboxes: list[tuple[Vector, Vector]],
    bbox_diag: float,
) -> float:
    if not frame_bboxes:
        return 0.0
    reference_size = frame_bboxes[0][1] - frame_bboxes[0][0]
    return max(((maxs - mins) - reference_size).length for mins, maxs in frame_bboxes) / max(bbox_diag, 1e-6)


def _compute_pose_motion_extent(
    *,
    armature: bpy.types.Object | None,
    bone_mapping: dict[str, str],
    sample_frames: list[int],
    scene: bpy.types.Scene,
    depsgraph,
    bbox_diag: float,
) -> float:
    if armature is None or not sample_frames:
        return 0.0

    tracked_bones: list[str] = []
    seen = set()
    for bone_name in bone_mapping.values():
        if bone_name in armature.pose.bones and bone_name not in seen:
            tracked_bones.append(bone_name)
            seen.add(bone_name)

    if not tracked_bones:
        for pose_bone in armature.pose.bones:
            if not pose_bone.bone.use_deform:
                continue
            if pose_bone.name in seen:
                continue
            tracked_bones.append(pose_bone.name)
            seen.add(pose_bone.name)
            if len(tracked_bones) >= 32:
                break

    if not tracked_bones:
        return 0.0

    def bone_midpoint(bone_name: str) -> Vector:
        pose_bone = armature.pose.bones[bone_name]
        head = armature.matrix_world @ pose_bone.head
        tail = armature.matrix_world @ pose_bone.tail
        return (head + tail) * 0.5

    scene.frame_set(sample_frames[0])
    depsgraph.update()
    reference_positions = {bone_name: bone_midpoint(bone_name).copy() for bone_name in tracked_bones}

    extent = 0.0
    for frame in sample_frames[1:]:
        scene.frame_set(frame)
        depsgraph.update()
        for bone_name in tracked_bones:
            extent = max(
                extent,
                (bone_midpoint(bone_name) - reference_positions[bone_name]).length / max(bbox_diag, 1e-6),
            )
    return extent


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "render_report.json"
    eval_config = {}
    if args.evaluation_config:
        eval_config = json.loads(Path(args.evaluation_config).read_text(encoding="utf-8"))
    reference_framing = eval_config.get("reference_framing", [])
    skeleton_package = eval_config.get("skeleton_package", {})

    scene = bpy.context.scene
    scene.render.resolution_x = args.image_width
    scene.render.resolution_y = args.image_height
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    if "BLENDER_EEVEE_NEXT" in engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in engines:
        scene.render.engine = "BLENDER_EEVEE"

    _ensure_world((0.40, 0.42, 0.46, 1.0))
    _ensure_light()
    camera = _ensure_camera()
    camera.data.type = "ORTHO"

    armature = _find_primary_armature()
    meshes = _find_target_meshes(armature)
    gate_fail_reasons: list[str] = []
    validity_gate_passed = True
    bone_mapping: dict[str, str] = {}
    missing_required_bones: list[str] = []

    if armature is None:
        gate_fail_reasons.append("missing_armature")
        validity_gate_passed = False
    if not meshes:
        gate_fail_reasons.append("missing_target_mesh")
        validity_gate_passed = False
    if armature is not None:
        has_armature_modifier = any(
            modifier.type == "ARMATURE" and modifier.object == armature
            for mesh in meshes
            for modifier in mesh.modifiers
        )
        if not has_armature_modifier:
            gate_fail_reasons.append("mesh_not_driven_by_armature")
            validity_gate_passed = False
        bone_mapping, missing_required_bones = _resolve_bone_mapping(armature, skeleton_package)
        if missing_required_bones:
            gate_fail_reasons.append("missing_required_bones")
            validity_gate_passed = False
    if not _has_animation(armature):
        gate_fail_reasons.append("missing_animation_data")
        validity_gate_passed = False

    frame_start, frame_end = _animation_frame_range()
    if frame_end <= frame_start:
        gate_fail_reasons.append("invalid_frame_range")
        validity_gate_passed = False

    sample_frames, sample_positions = _sample_frames(frame_start, frame_end, args.sample_count)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    global_min = Vector((float("inf"), float("inf"), float("inf")))
    global_max = Vector((float("-inf"), float("-inf"), float("-inf")))
    frame_bboxes: list[tuple[Vector, Vector]] = []
    centers = []
    for frame in sample_frames:
        scene.frame_set(frame)
        depsgraph.update()
        mins, maxs = _evaluated_bbox_world(meshes, depsgraph)
        frame_bboxes.append((mins.copy(), maxs.copy()))
        if any(not math.isfinite(v) for v in (*mins, *maxs)):
            gate_fail_reasons.append("bbox_contains_nan")
            validity_gate_passed = False
            break
        global_min = Vector((min(global_min[i], mins[i]) for i in range(3)))
        global_max = Vector((max(global_max[i], maxs[i]) for i in range(3)))
        centers.append((mins + maxs) * 0.5)

    bbox_size = global_max - global_min
    bbox_diag = bbox_size.length
    if not math.isfinite(bbox_diag) or bbox_diag <= 1e-6:
        gate_fail_reasons.append("degenerate_bbox")
        validity_gate_passed = False
        bbox_diag = 1.0

    bbox_center_motion_extent = 0.0
    if centers:
        anchor = centers[0]
        bbox_center_motion_extent = max((center - anchor).length for center in centers) / max(bbox_diag, 1e-6)
    bbox_shape_motion_extent = _compute_bbox_shape_motion_extent(frame_bboxes, bbox_diag)
    pose_motion_extent = _compute_pose_motion_extent(
        armature=armature,
        bone_mapping=bone_mapping,
        sample_frames=sample_frames,
        scene=scene,
        depsgraph=depsgraph,
        bbox_diag=bbox_diag,
    )
    motion_extent = max(bbox_center_motion_extent, bbox_shape_motion_extent, pose_motion_extent)
    if motion_extent < 0.001:
        gate_fail_reasons.append("animation_nearly_static")
        validity_gate_passed = False

    max_frame_width = 1e-3
    max_frame_depth = 1e-3
    max_frame_height = 1e-3
    for mins, maxs in frame_bboxes:
        size = maxs - mins
        max_frame_width = max(max_frame_width, float(size.x))
        max_frame_depth = max(max_frame_depth, float(size.y))
        max_frame_height = max(max_frame_height, float(size.z))

    height = max(max_frame_height, 1e-3)
    distance = max(max_frame_width, max_frame_depth, height) * 6.0
    aspect = args.image_width / max(args.image_height, 1)
    views = {
        "front": Vector((0.0, -distance, height * 0.1)),
        "three_quarter": Vector((distance * 0.7, -distance * 0.7, height * 0.12)),
        "side": Vector((distance, 0.0, height * 0.08)),
    }
    silhouette = _build_emission_material("EvalSilhouette", (0.0, 0.0, 0.0, 1.0))
    clay = _build_material("EvalClay", (0.74, 0.74, 0.76, 1.0))
    original_materials = _assign_override(meshes, clay)

    view_paths: dict[str, list[str]] = {view: [] for view in views}
    silhouette_paths: list[str] = []

    _ensure_world((1.0, 1.0, 1.0, 1.0))
    scene.render.film_transparent = True
    for view_name, offset in views.items():
        base_width = max(max_frame_width, 1e-3) if view_name == "front" else max(max_frame_depth, 1e-3)
        if view_name == "three_quarter":
            base_width = max(max_frame_width, max_frame_depth, 1e-3)
        target_width = base_width / 0.30
        target_height = height / 0.64
        camera.data.ortho_scale = max(target_width, target_height * aspect)
        for idx, frame in enumerate(sample_frames):
            scene.frame_set(frame)
            frame_center = (frame_bboxes[idx][0] + frame_bboxes[idx][1]) * 0.5
            location = frame_center + offset + Vector((0.0, 0.0, height * 0.16))
            target = frame_center + Vector((0.0, 0.0, height * 0.08))
            camera.data.shift_x = 0.0
            camera.data.shift_y = 0.0
            if view_name == "front" and idx < len(reference_framing):
                target_frame = reference_framing[idx]
                size = frame_bboxes[idx][1] - frame_bboxes[idx][0]
                target_width = max(float(size.x) / max(float(target_frame.get("width", 0.3)), 1e-3), 1e-3)
                target_height = max((float(size.z) * aspect) / max(float(target_frame.get("height", 0.6)), 1e-3), 1e-3)
                camera.data.ortho_scale = max(target_width, target_height)
                camera.data.shift_x = float(0.5 - float(target_frame.get("center_x", 0.5)))
                camera.data.shift_y = float(float(target_frame.get("center_y", 0.5)) - 0.5)
            camera.location = location
            _look_at(camera, target)
            render_path = output_dir / f"{view_name}_{idx:02d}.png"
            scene.render.filepath = str(render_path)
            bpy.ops.render.render(write_still=True)
            view_paths[view_name].append(str(render_path))

    _assign_override(meshes, silhouette)
    _ensure_world((1.0, 1.0, 1.0, 1.0))
    scene.render.film_transparent = True
    camera.data.ortho_scale = max(max_frame_width / 0.30, (height / 0.64) * aspect)
    for idx, frame in enumerate(sample_frames):
        scene.frame_set(frame)
        frame_center = (frame_bboxes[idx][0] + frame_bboxes[idx][1]) * 0.5
        location = frame_center + views["front"] + Vector((0.0, 0.0, height * 0.16))
        target = frame_center + Vector((0.0, 0.0, height * 0.08))
        camera.data.shift_x = 0.0
        camera.data.shift_y = 0.0
        if idx < len(reference_framing):
            target_frame = reference_framing[idx]
            size = frame_bboxes[idx][1] - frame_bboxes[idx][0]
            target_width = max(float(size.x) / max(float(target_frame.get("width", 0.3)), 1e-3), 1e-3)
            target_height = max((float(size.z) * aspect) / max(float(target_frame.get("height", 0.6)), 1e-3), 1e-3)
            camera.data.ortho_scale = max(target_width, target_height)
            camera.data.shift_x = float(0.5 - float(target_frame.get("center_x", 0.5)))
            camera.data.shift_y = float(float(target_frame.get("center_y", 0.5)) - 0.5)
        camera.location = location
        _look_at(camera, target)
        render_path = output_dir / f"silhouette_{idx:02d}.png"
        scene.render.filepath = str(render_path)
        bpy.ops.render.render(write_still=True)
        silhouette_paths.append(str(render_path))

    _restore_materials(meshes, original_materials)
    _assign_override(meshes, silhouette)
    _ensure_world((1.0, 1.0, 1.0, 1.0))
    scene.render.film_transparent = True

    joint_ranges = {}
    pose_states_report: list[dict[str, object]] = []
    if armature is not None and bone_mapping:
        joint_ranges = _compute_joint_ranges(
            armature=armature,
            bone_mapping=bone_mapping,
            frame_positions=list(skeleton_package.get("frame_samples", sample_positions)),
            frame_start=frame_start,
            frame_end=frame_end,
            scene=scene,
            depsgraph=depsgraph,
        )
        for pose_state in skeleton_package.get("pose_states", []):
            sample_position = float(pose_state.get("sample_position", 0.0))
            frame = _frame_at_position(frame_start, frame_end, sample_position)
            scene.frame_set(frame)
            depsgraph.update()
            mins, maxs = _evaluated_bbox_world(meshes, depsgraph)
            center = (mins + maxs) * 0.5
            diag = max((maxs - mins).length, 1e-6)
            camera.data.ortho_scale = max(float((maxs - mins).x) / 0.30, (float((maxs - mins).z) / 0.64) * aspect, 1e-3)
            camera.data.shift_x = 0.0
            camera.data.shift_y = 0.0
            camera.location = center + views["front"] + Vector((0.0, 0.0, float((maxs - mins).z) * 0.16))
            _look_at(camera, center + Vector((0.0, 0.0, float((maxs - mins).z) * 0.08)))
            pose_name = str(pose_state.get("name", f"pose_{len(pose_states_report):02d}"))
            render_path = output_dir / f"{pose_name}_candidate.png"
            scene.render.filepath = str(render_path)
            bpy.ops.render.render(write_still=True)
            bone_states = {
                semantic: _pose_bone_state(armature, bone_name, center, diag)
                for semantic, bone_name in bone_mapping.items()
            }
            pose_states_report.append(
                {
                    "name": pose_name,
                    "sample_position": sample_position,
                    "frame": int(frame),
                    "image_path": str(render_path),
                    "bone_states": bone_states,
                }
            )

    report = {
        "validity_gate_passed": validity_gate_passed,
        "gate_fail_reasons": gate_fail_reasons,
        "armature_name": armature.name if armature else None,
        "mesh_names": [mesh.name for mesh in meshes],
        "animation_frame_start": frame_start,
        "animation_frame_end": frame_end,
        "sample_frames": sample_frames,
        "sample_positions": sample_positions,
        "motion_extent": motion_extent,
        "bbox_center_motion_extent": bbox_center_motion_extent,
        "bbox_shape_motion_extent": bbox_shape_motion_extent,
        "pose_motion_extent": pose_motion_extent,
        "view_paths": view_paths,
        "silhouette_paths": silhouette_paths,
        "recognized_bones": bone_mapping,
        "missing_required_bones": missing_required_bones,
        "joint_ranges": joint_ranges,
        "pose_states": pose_states_report,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
