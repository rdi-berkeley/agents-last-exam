from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import mathutils

PART_VIEW_SPECS = [
    ('front', 0.0, 18.0),
]
SCENE_VIEW_SPECS = [
    ('front', 0.0, 18.0),
    ('back', 180.0, 18.0),
    ('left', 90.0, 18.0),
    ('right', 270.0, 18.0),
]


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description='Render multipart material views from current Blender scene')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--parts-json', default='[]')
    parser.add_argument('--required-parts-json', default='')
    parser.add_argument('--detail-resolution', type=int, default=768)
    parser.add_argument('--scene-resolution', type=int, default=1024)
    return parser.parse_args(argv)


def mesh_objects() -> list[bpy.types.Object]:
    return sorted([obj for obj in bpy.context.scene.objects if obj.type == 'MESH'], key=lambda o: o.name)


def world_bbox(obj: bpy.types.Object) -> list[mathutils.Vector]:
    return [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]


def combined_bbox(objects: list[bpy.types.Object]) -> tuple[mathutils.Vector, mathutils.Vector]:
    points = []
    for obj in objects:
        points.extend(world_bbox(obj))
    min_v = mathutils.Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_v = mathutils.Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return min_v, max_v


def get_or_create_camera_rig() -> tuple[bpy.types.Object, bpy.types.Object]:
    scene = bpy.context.scene
    rig = bpy.data.objects.get('MultipartRenderRig')
    if rig is None:
        rig = bpy.data.objects.new('MultipartRenderRig', None)
        scene.collection.objects.link(rig)
    cam = bpy.data.objects.get('MultipartRenderCamera')
    if cam is None or cam.type != 'CAMERA':
        cam_data = bpy.data.cameras.new(name='MultipartRenderCameraData')
        cam = bpy.data.objects.new('MultipartRenderCamera', cam_data)
        scene.collection.objects.link(cam)
    cam.parent = rig
    track = None
    for constraint in cam.constraints:
        if constraint.type == 'TRACK_TO':
            track = constraint
            break
    if track is None:
        track = cam.constraints.new(type='TRACK_TO')
    track.target = rig
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    scene.camera = cam
    return rig, cam


def set_workbench_texture_render(resolution: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.display.render_aa = 'FXAA'
    shading = scene.display.shading
    shading.light = 'STUDIO'
    shading.background_type = 'VIEWPORT'
    shading.background_color = (0.18, 0.19, 0.21)
    shading.color_type = 'TEXTURE'
    shading.show_object_outline = False
    shading.show_cavity = False
    shading.show_shadows = True
    shading.shadow_intensity = 0.2


def set_visibility(all_objs: list[bpy.types.Object], visible: list[bpy.types.Object]) -> None:
    names = {obj.name for obj in visible}
    for obj in all_objs:
        state = obj.name in names
        obj.hide_set(not state)
        obj.hide_render = not state


def project_distance(size: mathutils.Vector) -> float:
    return max(size.x, size.y, size.z, 0.001) * 2.8


def set_pose(rig: bpy.types.Object, cam: bpy.types.Object, center: mathutils.Vector, az_deg: float, el_deg: float, distance: float) -> None:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    rig.location = center
    cam.location = mathutils.Vector((
        math.sin(az) * distance * math.cos(el),
        -math.cos(az) * distance * math.cos(el),
        distance * math.sin(el),
    ))
    bpy.context.view_layer.update()


def _find_basecolor(obj: bpy.types.Object) -> str | None:
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                return node.image.name
    return None


def _part_gate_info(obj: bpy.types.Object) -> dict:
    has_uv = bool(getattr(obj.data, 'uv_layers', None)) and len(obj.data.uv_layers) > 0
    has_mtl = any(slot.material is not None for slot in obj.material_slots)
    basecolor = _find_basecolor(obj)
    return {
        'has_uv': bool(has_uv),
        'has_mtl': bool(has_mtl),
        'has_basecolor_texture': bool(basecolor),
        'basecolor_texture_name': basecolor,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_objs = mesh_objects()
    if not all_objs:
        raise RuntimeError('No mesh objects in scene')

    requested = json.loads(args.parts_json)
    required_parts = json.loads(args.required_parts_json) if args.required_parts_json else requested
    name_map = {obj.name: obj for obj in all_objs}
    parts = [name_map[name] for name in requested if name in name_map]
    missing_parts = [name for name in required_parts if name not in name_map]

    rig, cam = get_or_create_camera_rig()
    cam.data.type = 'PERSP'
    cam.data.lens = 55.0
    cam.data.clip_start = 0.01
    cam.data.clip_end = 10000.0

    scene_min, scene_max = combined_bbox(all_objs)
    scene_center = (scene_min + scene_max) * 0.5
    scene_size = scene_max - scene_min
    scene_dist = project_distance(scene_size)

    scene_dir = out_dir / 'scene_views'
    scene_dir.mkdir(parents=True, exist_ok=True)
    set_workbench_texture_render(args.scene_resolution)
    set_visibility(all_objs, all_objs)
    scene_views = []
    for view_name, az, el in SCENE_VIEW_SPECS:
        set_pose(rig, cam, scene_center, az, el, scene_dist)
        out_path = scene_dir / f'{view_name}.png'
        bpy.context.scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        scene_views.append({'view': view_name, 'image': str(out_path), 'azimuth_deg': az, 'elevation_deg': el})
        print(f'[scene-render] {view_name}')

    parts_dir = out_dir / 'parts'
    manifest_parts = []
    for obj in parts:
        obj_dir = parts_dir / obj.name
        detail_dir = obj_dir / 'detail'
        detail_dir.mkdir(parents=True, exist_ok=True)
        bbox_min, bbox_max = combined_bbox([obj])
        center = (bbox_min + bbox_max) * 0.5
        size = bbox_max - bbox_min
        dist = project_distance(size)
        gate = _part_gate_info(obj)
        view_entries = []
        set_workbench_texture_render(args.detail_resolution)
        set_visibility(all_objs, [obj])
        for view_name, az, el in PART_VIEW_SPECS:
            set_pose(rig, cam, center, az, el, dist)
            out_path = detail_dir / f'{view_name}.png'
            bpy.context.scene.render.filepath = str(out_path)
            bpy.ops.render.render(write_still=True)
            view_entries.append({
                'view': view_name,
                'detail_image': str(out_path),
                'context_image': str(scene_dir / 'front.png'),
                'azimuth_deg': az,
                'elevation_deg': el,
            })
            print(f'[part-render] {obj.name} {view_name}')
        manifest_parts.append({
            'name': obj.name,
            'views': view_entries,
            **gate,
        })

    payload = {
        'parts': manifest_parts,
        'scene_views': scene_views,
        'missing_parts': missing_parts,
    }
    (out_dir / 'candidate_manifest.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'[done] output: {out_dir}')


if __name__ == '__main__':
    main()
