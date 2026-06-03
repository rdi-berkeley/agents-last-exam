from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import mathutils

VIEW_SPECS = [
    ('front', 0.0, 18.0),
    ('back', 180.0, 18.0),
    ('left', 90.0, 18.0),
    ('right', 270.0, 18.0),
    ('top_front', 0.0, 45.0),
    ('bottom_front', 0.0, -18.0),
]


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(description='Render textured material views for a single OBJ')
    parser.add_argument('--obj', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_obj(filepath: Path) -> None:
    if hasattr(bpy.ops.wm, 'obj_import'):
        bpy.ops.wm.obj_import(filepath=str(filepath))
    else:
        bpy.ops.import_scene.obj(filepath=str(filepath))


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']


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
    rig = bpy.data.objects.new('RenderRig', None)
    scene.collection.objects.link(rig)
    cam_data = bpy.data.cameras.new(name='RenderCameraData')
    cam = bpy.data.objects.new('RenderCamera', cam_data)
    scene.collection.objects.link(cam)
    cam.parent = rig
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


def main() -> None:
    args = parse_args()
    obj_path = Path(args.obj).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    import_obj(obj_path)
    objs = mesh_objects()
    if not objs:
        raise RuntimeError(f'No mesh objects imported from {obj_path}')

    min_v, max_v = combined_bbox(objs)
    center = (min_v + max_v) * 0.5
    size = max_v - min_v
    dist = project_distance(size)

    rig, cam = get_or_create_camera_rig()
    cam.data.type = 'PERSP'
    cam.data.lens = 55.0
    cam.data.clip_start = 0.01
    cam.data.clip_end = 10000.0
    set_workbench_texture_render(args.resolution)

    manifest = []
    for name, az, el in VIEW_SPECS:
        set_pose(rig, cam, center, az, el, dist)
        out_path = out_dir / f'{name}.png'
        bpy.context.scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        manifest.append({'name': name, 'azimuth_deg': az, 'elevation_deg': el, 'image': str(out_path)})
        print(f'[render] {name}')

    (out_dir / 'views_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'[done] output: {out_dir}')


if __name__ == '__main__':
    main()
