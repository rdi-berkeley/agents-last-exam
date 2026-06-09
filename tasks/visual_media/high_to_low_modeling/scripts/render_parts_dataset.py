"""Render paired detail/context images for Blender part datasets.

Usage:
  <blender_binary> -b --factory-startup \
    --python tasks/visual_media/high_to_low_modeling/scripts/render_parts_dataset.py -- \
    --input-obj-dir "<input_obj_dir>" \
    --output-dir "<output_dir>" \
    --parts "Helmet_deco_L__12__,Helmet_deco_R__12__" \
    --angles "0,90,180,270" \
    --clear-scene

The script generates, for each selected mesh object:
  - paired detail renders
  - paired context renders
  - per-angle metadata JSON
  - a top-level manifest.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import bpy
import mathutils
from bpy_extras.object_utils import world_to_camera_view


def parse_args() -> argparse.Namespace:
    argv: list[str] = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(description="Render paired detail/context part dataset")
    parser.add_argument("--input-obj-dir", type=str, default="", help="Directory containing OBJ files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--parts", type=str, default="", help="Comma-separated mesh object names to render")
    parser.add_argument("--angles", type=str, default="0,90,180,270", help="Comma-separated azimuth angles")
    parser.add_argument("--limit", type=int, default=0, help="Render only the first N mesh objects")
    parser.add_argument("--elevation-deg", type=float, default=18.0, help="Camera elevation in degrees")
    parser.add_argument("--lens-mm", type=float, default=55.0, help="Camera lens in mm")
    parser.add_argument("--detail-resolution", type=int, default=1024, help="Square resolution for detail renders")
    parser.add_argument("--context-resolution", type=int, default=1280, help="Square resolution for context renders")
    parser.add_argument(
        "--context-mode",
        type=str,
        default="highlight",
        choices=["highlight", "bbox"],
        help="Context output style: full-scene highlight render or cached bbox annotation",
    )
    parser.add_argument(
        "--detail-target-span",
        type=float,
        default=0.72,
        help="Target normalized frame occupancy for detail renders",
    )
    parser.add_argument(
        "--context-target-span",
        type=float,
        default=0.82,
        help="Target normalized frame occupancy for context renders",
    )
    parser.add_argument("--clear-scene", action="store_true", help="Delete all existing scene objects before import")
    parser.add_argument(
        "--render-existing-meshes",
        action="store_true",
        help="Render meshes already present in the scene instead of importing OBJs",
    )
    return parser.parse_args(argv)


def parse_angles(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("At least one azimuth angle is required")
    return values


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_obj(filepath: Path) -> None:
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(filepath))
    else:
        bpy.ops.import_scene.obj(filepath=str(filepath))


def import_all_objs(input_dir: Path) -> list[bpy.types.Object]:
    before = {obj.name for obj in bpy.data.objects}
    for fp in sorted(input_dir.glob("*.obj")):
        import_obj(fp)
    return [obj for obj in bpy.data.objects if obj.name not in before and obj.type == "MESH"]


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def sanitize_name(name: str) -> str:
    """Preserve the Blender object name as much as possible for output paths."""
    safe = name.replace("/", "_").replace("\\", "_").replace("\0", "_")
    return safe or "part"


def world_bbox(obj: bpy.types.Object) -> list[mathutils.Vector]:
    return [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]


def points_bounds(points: Iterable[mathutils.Vector]) -> tuple[mathutils.Vector, mathutils.Vector]:
    pts = list(points)
    if not pts:
        zero = mathutils.Vector((0.0, 0.0, 0.0))
        return zero, zero
    min_v = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_v = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return min_v, max_v


def combined_bbox(objects: Iterable[bpy.types.Object]) -> tuple[mathutils.Vector, mathutils.Vector]:
    points: list[mathutils.Vector] = []
    for obj in objects:
        points.extend(world_bbox(obj))
    return points_bounds(points)


def get_or_create_camera_rig() -> tuple[bpy.types.Object, bpy.types.Object]:
    scene = bpy.context.scene

    rig = bpy.data.objects.get("Dataset_Rig")
    if rig is None:
        rig = bpy.data.objects.new("Dataset_Rig", None)
        scene.collection.objects.link(rig)

    cam = bpy.data.objects.get("Dataset_Camera")
    if cam is None or cam.type != "CAMERA":
        cam_data = bpy.data.cameras.new(name="Dataset_CameraData")
        cam = bpy.data.objects.new("Dataset_Camera", cam_data)
        scene.collection.objects.link(cam)

    cam.parent = rig

    track = None
    for constraint in cam.constraints:
        if constraint.type == "TRACK_TO":
            track = constraint
            break
    if track is None:
        track = cam.constraints.new(type="TRACK_TO")
    track.target = rig
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    scene.camera = cam
    return rig, cam


def setup_workbench(
    detail_bg: tuple[float, float, float],
    color_type: str,
    *,
    transparent: bool = False,
    show_cavity: bool = True,
    show_shadows: bool = True,
) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = transparent
    scene.display.render_aa = "FXAA"

    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.background_type = "VIEWPORT"
    shading.background_color = (*detail_bg,)
    shading.color_type = color_type
    shading.show_object_outline = False
    shading.show_cavity = show_cavity
    shading.cavity_type = "BOTH"
    shading.curvature_ridge_factor = 1.4
    shading.curvature_valley_factor = 1.1
    shading.show_shadows = show_shadows
    shading.shadow_intensity = 0.35 if show_shadows else 0.0


def set_viewports_to_camera() -> None:
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "SOLID"
                    space.region_3d.view_perspective = "CAMERA"


def set_pose(
    rig: bpy.types.Object,
    cam: bpy.types.Object,
    center: mathutils.Vector,
    azimuth_deg: int,
    elevation_rad: float,
    distance: float,
) -> None:
    rig.location = center
    rig.rotation_euler = (0.0, 0.0, math.radians(azimuth_deg))
    cam.location = mathutils.Vector((0.0, -distance * math.cos(elevation_rad), distance * math.sin(elevation_rad)))
    bpy.context.view_layer.update()


def project_bounds(scene: bpy.types.Scene, cam: bpy.types.Object, points: Sequence[mathutils.Vector]) -> tuple[float, float, float, float]:
    coords = [world_to_camera_view(scene, cam, p) for p in points]
    xs = [c.x for c in coords]
    ys = [c.y for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def fit_distance(
    scene: bpy.types.Scene,
    rig: bpy.types.Object,
    cam: bpy.types.Object,
    center: mathutils.Vector,
    points: Sequence[mathutils.Vector],
    azimuth_deg: int,
    elevation_rad: float,
    target_span: float,
) -> float:
    min_v, max_v = points_bounds(points)
    size = max_v - min_v
    max_dim = max(size.x, size.y, size.z, 0.001)
    near = max_dim * 0.2
    far = max_dim * 80.0

    for _ in range(40):
        mid = (near + far) * 0.5
        set_pose(rig, cam, center, azimuth_deg, elevation_rad, mid)
        min_x, min_y, max_x, max_y = project_bounds(scene, cam, points)
        span = max(max_x - min_x, max_y - min_y)
        if span > target_span:
            near = mid
        else:
            far = mid

    set_pose(rig, cam, center, azimuth_deg, elevation_rad, far)
    return far


def set_visibility(all_objs: Sequence[bpy.types.Object], visible: Sequence[bpy.types.Object]) -> None:
    visible_names = {obj.name for obj in visible}
    for obj in all_objs:
        is_visible = obj.name in visible_names
        obj.hide_set(not is_visible)
        obj.hide_render = not is_visible


def set_object_colors(all_objs: Sequence[bpy.types.Object], target: bpy.types.Object, detail_mode: bool) -> None:
    if detail_mode:
        for obj in all_objs:
            obj.color = (0.86, 0.86, 0.87, 1.0)
        target.color = (0.86, 0.86, 0.87, 1.0)
        return

    for obj in all_objs:
        obj.color = (0.72, 0.74, 0.78, 1.0)
    target.color = (0.94, 0.46, 0.12, 1.0)


def set_uniform_object_color(all_objs: Sequence[bpy.types.Object], color: tuple[float, float, float, float]) -> None:
    for obj in all_objs:
        obj.color = color


def draw_bbox_context(base_path: Path, out_path: Path, bbox_norm: Sequence[float]) -> None:
    code = """
import json
import sys
from PIL import Image, ImageDraw

base_path, out_path, bbox_json = sys.argv[1:4]
min_x, min_y, max_x, max_y = json.loads(bbox_json)

img = Image.open(base_path).convert("RGBA")
w, h = img.size

def clamp(value):
    return max(0.0, min(1.0, float(value)))

min_x, min_y, max_x, max_y = map(clamp, (min_x, min_y, max_x, max_y))
left = min_x * w
right = max_x * w
top = (1.0 - max_y) * h
bottom = (1.0 - min_y) * h

thickness = max(4, int(min(w, h) * 0.004))
if right - left < thickness * 2:
    cx = (left + right) * 0.5
    left = cx - thickness
    right = cx + thickness
if bottom - top < thickness * 2:
    cy = (top + bottom) * 0.5
    top = cy - thickness
    bottom = cy + thickness

draw = ImageDraw.Draw(img, "RGBA")
draw.rectangle(
    [left, top, right, bottom],
    outline=(240, 117, 24, 255),
    width=thickness,
    fill=(240, 117, 24, 36),
)

cx = (left + right) * 0.5
cy = (top + bottom) * 0.5
r = max(6, thickness * 2)
draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(240, 117, 24, 220))

img.save(out_path)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(base_path), str(out_path), json.dumps([float(v) for v in bbox_norm])],
        check=True,
    )


def image_metrics(image_path: Path) -> tuple[float, float]:
    img = bpy.data.images.load(str(image_path), check_existing=False)
    pixels = list(img.pixels)
    bpy.data.images.remove(img)

    sample_stride = 400
    luminances = []
    for i in range(0, len(pixels) - 3, 4 * sample_stride):
        r, g, b = pixels[i], pixels[i + 1], pixels[i + 2]
        luminances.append(0.2126 * r + 0.7152 * g + 0.0722 * b)

    mean_lum = sum(luminances) / len(luminances)
    bright_ratio = sum(1 for value in luminances if value > 0.08) / len(luminances)
    return mean_lum, bright_ratio


def main() -> None:
    args = parse_args()
    angles = parse_angles(args.angles)
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.clear_scene:
        clear_scene()

    if args.input_obj_dir:
        input_dir = Path(args.input_obj_dir).expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        imported = import_all_objs(input_dir)
        print(f"[import] imported {len(imported)} OBJ meshes from {input_dir}")

    objs = sorted(mesh_objects(), key=lambda obj: obj.name)
    if not objs:
        raise RuntimeError("No mesh objects found in scene")
    if not args.input_obj_dir and not args.render_existing_meshes:
        raise RuntimeError("Pass --render-existing-meshes when rendering the current scene without importing OBJs")

    if args.parts:
        requested = [name.strip() for name in args.parts.split(",") if name.strip()]
        name_map = {obj.name: obj for obj in objs}
        missing = [name for name in requested if name not in name_map]
        if missing:
            raise RuntimeError(f"Missing mesh objects: {', '.join(missing)}")
        target_objs = [name_map[name] for name in requested]
    elif args.limit > 0:
        target_objs = objs[: args.limit]
    else:
        target_objs = objs

    scene = bpy.context.scene
    rig, cam = get_or_create_camera_rig()
    cam.data.type = "PERSP"
    cam.data.lens = args.lens_mm
    cam.data.clip_start = 0.01
    cam.data.clip_end = 10000.0

    set_viewports_to_camera()

    model_min, model_max = combined_bbox(objs)
    model_center = (model_min + model_max) * 0.5
    all_points = []
    for obj in objs:
        all_points.extend(world_bbox(obj))

    elev_rad = math.radians(args.elevation_deg)
    setup_workbench((0.18, 0.19, 0.21), "OBJECT")
    scene.render.resolution_x = args.context_resolution
    scene.render.resolution_y = args.context_resolution
    context_distance_by_angle = {
        azimuth: fit_distance(scene, rig, cam, model_center, all_points, azimuth, elev_rad, args.context_target_span)
        for azimuth in angles
    }

    context_base_paths: dict[int, Path] = {}
    context_base_tmpdir: str | None = None
    if args.context_mode == "bbox":
        context_base_tmpdir = tempfile.mkdtemp(prefix="dataset_context_base_")
        base_root = Path(context_base_tmpdir)
        setup_workbench(
            (0.18, 0.19, 0.21),
            "OBJECT",
            transparent=False,
            show_cavity=False,
            show_shadows=False,
        )
        set_visibility(objs, objs)
        set_uniform_object_color(objs, (0.72, 0.74, 0.78, 1.0))
        scene.render.resolution_x = args.context_resolution
        scene.render.resolution_y = args.context_resolution
        for azimuth in angles:
            set_pose(rig, cam, model_center, azimuth, elev_rad, context_distance_by_angle[azimuth])
            base_path = base_root / f"context_base__az{azimuth:03d}_el{int(round(args.elevation_deg)):02d}.png"
            scene.render.filepath = str(base_path)
            bpy.ops.render.render(write_still=True)
            context_base_paths[azimuth] = base_path

    manifest: list[dict] = []
    try:
        for obj in target_objs:
            safe = sanitize_name(obj.name)
            obj_root = out_root / safe
            detail_dir = obj_root / "detail"
            context_dir = obj_root / "context"
            meta_dir = obj_root / "meta"
            detail_dir.mkdir(parents=True, exist_ok=True)
            context_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)

            target_points = world_bbox(obj)
            target_min, target_max = points_bounds(target_points)
            target_center = (target_min + target_max) * 0.5
            target_size = target_max - target_min

            angle_entries: list[dict] = []
            for azimuth in angles:
                scene.render.resolution_x = args.context_resolution
                scene.render.resolution_y = args.context_resolution
                set_pose(rig, cam, model_center, azimuth, elev_rad, context_distance_by_angle[azimuth])
                context_bbox = project_bounds(scene, cam, target_points)
                context_path = context_dir / f"{safe}__az{azimuth:03d}_el{int(round(args.elevation_deg)):02d}__context.png"

                if args.context_mode == "highlight":
                    setup_workbench((0.18, 0.19, 0.21), "OBJECT")
                    set_visibility(objs, objs)
                    set_object_colors(objs, obj, detail_mode=False)
                    scene.render.filepath = str(context_path)
                    bpy.ops.render.render(write_still=True)
                else:
                    draw_bbox_context(context_base_paths[azimuth], context_path, context_bbox)

                context_lum, context_bright = image_metrics(context_path)

                setup_workbench((0.18, 0.19, 0.21), "OBJECT")
                set_visibility(objs, [obj])
                set_object_colors(objs, obj, detail_mode=True)
                scene.render.resolution_x = args.detail_resolution
                scene.render.resolution_y = args.detail_resolution
                detail_distance = fit_distance(
                    scene,
                    rig,
                    cam,
                    target_center,
                    target_points,
                    azimuth,
                    elev_rad,
                    args.detail_target_span,
                )

                detail_path = detail_dir / f"{safe}__az{azimuth:03d}_el{int(round(args.elevation_deg)):02d}__detail.png"
                scene.render.filepath = str(detail_path)
                bpy.ops.render.render(write_still=True)
                detail_bbox = project_bounds(scene, cam, target_points)
                detail_lum, detail_bright = image_metrics(detail_path)

                angle_entries.append(
                    {
                        "azimuth_deg": azimuth,
                        "elevation_deg": float(args.elevation_deg),
                        "context_mode": args.context_mode,
                        "context_image": str(context_path),
                        "context_screen_bbox_norm": [float(v) for v in context_bbox],
                        "context_distance": float(context_distance_by_angle[azimuth]),
                        "context_mean_luminance": float(context_lum),
                        "context_bright_ratio": float(context_bright),
                        "context_usable": bool(context_lum > 0.12 and context_bright > 0.2),
                        "detail_image": str(detail_path),
                        "detail_screen_bbox_norm": [float(v) for v in detail_bbox],
                        "detail_distance": float(detail_distance),
                        "detail_mean_luminance": float(detail_lum),
                        "detail_bright_ratio": float(detail_bright),
                        "detail_usable": bool(detail_lum > 0.12 and detail_bright > 0.2),
                    }
                )
                print(f"[render] {obj.name} az={azimuth}")

            metadata = {
                "name": obj.name,
                "path_name": safe,
                "safe_name": safe,
                "required": True,
                "part_role": "editable",
                "world_center": [float(target_center.x), float(target_center.y), float(target_center.z)],
                "world_bbox_min": [float(target_min.x), float(target_min.y), float(target_min.z)],
                "world_bbox_max": [float(target_max.x), float(target_max.y), float(target_max.z)],
                "world_size": [float(target_size.x), float(target_size.y), float(target_size.z)],
                "model_center_world": [float(model_center.x), float(model_center.y), float(model_center.z)],
                "camera": {
                    "lens_mm": float(args.lens_mm),
                    "angles": angles,
                    "elevation_deg": float(args.elevation_deg),
                },
                "angles": angle_entries,
            }
            meta_path = meta_dir / f"{safe}__meta.json"
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            manifest.append(
                {
                    "name": obj.name,
                    "path_name": safe,
                    "required": True,
                    "part_role": "editable",
                    "folder": str(obj_root),
                    "meta": str(meta_path),
                    "angles": angle_entries,
                }
            )

        manifest_path = out_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        evaluation_config = {
            "schema_version": "evaluation_config/v1",
            "project_name": input_dir.name,
            "reference_manifest": str(manifest_path),
            "angles": [
                {
                    "azimuth_deg": int(azimuth),
                    "elevation_deg": float(args.elevation_deg),
                    "context_mode": args.context_mode,
                }
                for azimuth in angles
            ],
            "camera": {
                "lens_mm": float(args.lens_mm),
            },
            "context_mode": args.context_mode,
            "score_weights": {
                "completeness_score": 0.20,
                "geometry_score": 0.25,
                "render_score": 0.20,
                "mesh_health_score": 0.10,
                "judge_score": 0.25,
            },
            "judge": {
                "enabled": True,
                "backend": "heuristic",
                "model": "gpt-4.1-mini",
                "temperature": 0,
            },
            "expected_parts": [
                {
                    "name": entry["name"],
                    "path_name": entry["path_name"],
                    "required": entry["required"],
                    "part_role": entry["part_role"],
                }
                for entry in manifest
            ],
        }
        evaluation_config_path = out_root / "evaluation_config.json"
        evaluation_config_path.write_text(json.dumps(evaluation_config, indent=2), encoding="utf-8")
        print(f"[done] manifest: {manifest_path}")
        print(f"[done] evaluation config: {evaluation_config_path}")
        print(f"[done] output root: {out_root}")
    finally:
        if context_base_tmpdir:
            shutil.rmtree(context_base_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
