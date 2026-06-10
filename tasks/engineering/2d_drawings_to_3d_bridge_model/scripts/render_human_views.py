"""Render 11 architectural views matching PDF-drawing conventions.

Output filenames (in output_dir):
  plan_hall_ground.png       (top-down, cut at z=1500mm)
  plan_hall_first.png        (top-down, cut at z=5500mm)
  plan_hall_second.png       (top-down, cut at z=9300mm)
  plan_tower_typical.png     (top-down, cut at z=21500mm)
  elevation_north.png        (looking -Y, ortho)
  elevation_south.png        (looking +Y, ortho)
  elevation_east.png         (looking -X, ortho)
  elevation_west.png         (looking +X, ortho)
  section_NS.png             (cut at X=centroid_x, looking +X, half visible)
  section_EW.png             (cut at Y=centroid_y, looking +Y, half visible)
  axonometric.png            (iso 45°/30°)

Usage (headless Blender):
  blender --background --python render_human_views.py -- \
      --obj <path> --out <dir>
"""

import argparse
import os
import sys
from math import radians

# Allow `import detect_floors` regardless of which directory Blender was invoked from
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bpy

from detect_floors import detect_floors_with_weights, select_plan_cuts, split_hall_tower

# ---------- args ----------
argv = sys.argv
argv = argv[argv.index("--") + 1 :] if "--" in argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--obj", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=2048)
ap.add_argument("--samples", type=int, default=64,
                help="EEVEE TAA samples; higher = cleaner anti-aliasing")
ap.add_argument("--line-thickness", type=float, default=1.5,
                help="Freestyle line thickness in pixels at target resolution")
ap.add_argument("--plans-only", action="store_true",
                help="Render only the 4 plan views (useful for re-rendering after "
                     "tuning plan-specific logic)")
ap.add_argument("--colored", action="store_true",
                help="Color faces by OBJ material name (glas=blue, metall=orange, "
                     "weiß=light gray) instead of uniform white. Helps human "
                     "review identify facade element types; not used for scoring.")
ap.add_argument("--color-map-json", default=None,
                help="Path to a JSON file overriding the default COLOR_MAP. The "
                     "file must contain {<substring>: [r,g,b,a], ...} with a 'default' "
                     "key. Substring match is case-insensitive against material names. "
                     "Has no effect unless --colored is also passed.")
ap.add_argument("--slab-mm", type=float, default=800.0,
                help="Thickness of plan-view slab in mm (will be scaled by --units).")
ap.add_argument("--floor-offset-mm", type=float, default=1500.0,
                help="Cut height above each detected floor in mm (will be scaled by --units).")
ap.add_argument("--units", choices=["mm", "m"], default="mm",
                help="Model coordinate units. Scales every mm-based parameter "
                     "(slab, floor offset, floor-detection bin size, merge radius). "
                     "Default mm; use 'm' for meter-scaled models (e.g. bridge_the_gap).")
ap.add_argument("--source-up-axis", choices=["Z", "Y"], default="Z",
                help="Which axis is up in the source OBJ. The Blender importer is "
                     "told accordingly so the loaded mesh always ends up Z-up internally. "
                     "Default Z (Betonwerk uses Z-up); set to 'Y' for Y-up OBJs like bridge_the_gap.")
args = ap.parse_args(argv)

# Unit factor: scales mm-based defaults to model coordinate units.
#   mm model: factor 1.0  (1 mm = 1 model unit)
#   m  model: factor 0.001 (1 mm = 0.001 m)
UNIT_FACTOR = 1.0 if args.units == "mm" else 0.001
SLAB = args.slab_mm * UNIT_FACTOR
FLOOR_OFFSET = args.floor_offset_mm * UNIT_FACTOR
DETECT_BIN = 100.0 * UNIT_FACTOR
DETECT_MERGE = 500.0 * UNIT_FACTOR
print(f"UNITS: {args.units}  unit_factor={UNIT_FACTOR}")
print(f"  SLAB={SLAB}  FLOOR_OFFSET={FLOOR_OFFSET}  DETECT_BIN={DETECT_BIN}  DETECT_MERGE={DETECT_MERGE}")

os.makedirs(args.out, exist_ok=True)

# ---------- clean scene ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# White background world — bright so plan/elevation/section views look like
# clean CAD line drawings (no surface shading, just Freestyle lines).
world = bpy.data.worlds.new("W")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs[0].default_value = (1, 1, 1, 1)
bg.inputs[1].default_value = 1.0

# Sun light, created OFF by default. Will be toggled on only for axonometric
# so plans/elevations stay pure-line; axon gets soft shading for 3D readability.
sun_data = bpy.data.lights.new("Sun", type="SUN")
sun_data.energy = 2.0
sun_obj = bpy.data.objects.new("Sun", sun_data)
sun_obj.rotation_euler = (radians(55), radians(0), radians(-45))
sun_obj.hide_render = True
scene.collection.objects.link(sun_obj)


def set_sun(enabled: bool):
    """Toggle the sun for the next render."""
    sun_obj.hide_render = not enabled
    if enabled:
        bg.inputs[1].default_value = 0.6  # dim ambient so sun reads
    else:
        bg.inputs[1].default_value = 1.0  # bright flat for line drawings


# Emission shader self-illuminates → no extra lighting needed for colored mode.
# Keep sun off and BG bright; axon block will still toggle sun for the 3D feel.

# Import OBJ with correct axis convention (no Y↔Z swap)
# Pass the source's up-axis to the importer so the in-Blender mesh ends up Z-up.
# Source Z-up: forward=Y, up=Z (preserves original orientation).
# Source Y-up: forward=-Z, up=Y → Blender swaps so output is Z-up.
_obj_axes = (
    {"forward_axis": "Y", "up_axis": "Z"} if args.source_up_axis == "Z"
    else {"forward_axis": "NEGATIVE_Z", "up_axis": "Y"}
)
bpy.ops.wm.obj_import(filepath=args.obj, **_obj_axes)

# Find the imported mesh (single object expected)
mesh_obj = None
for o in bpy.data.objects:
    if o.type == "MESH":
        mesh_obj = o
        break
if mesh_obj is None:
    raise RuntimeError("No mesh imported")

# Compute world-space bbox
import mathutils

bb_world = [mesh_obj.matrix_world @ mathutils.Vector(c) for c in mesh_obj.bound_box]
xs = [v.x for v in bb_world]
ys = [v.y for v in bb_world]
zs = [v.z for v in bb_world]
xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)
zmin, zmax = min(zs), max(zs)
cx = (xmin + xmax) / 2
cy = (ymin + ymax) / 2
cz = (zmin + zmax) / 2
print(f"BBOX: X[{xmin:.0f},{xmax:.0f}] Y[{ymin:.0f},{ymax:.0f}] Z[{zmin:.0f},{zmax:.0f}]")
print(f"CENTROID: ({cx:.0f},{cy:.0f},{cz:.0f})")

# ---------- Render settings ----------
_engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
if "BLENDER_EEVEE_NEXT" in _engines:
    scene.render.engine = "BLENDER_EEVEE_NEXT"
elif "BLENDER_EEVEE" in _engines:
    scene.render.engine = "BLENDER_EEVEE"
else:
    raise RuntimeError(f"No supported EEVEE engine found in {sorted(_engines)}")
scene.render.resolution_x = args.res
scene.render.resolution_y = args.res
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.eevee.taa_render_samples = args.samples

# In --colored mode, disable AgX/Filmic view transform so Emission outputs
# render at their literal RGB values — otherwise blue glass / orange metal
# get tonemapped to muted grays. For the default line-drawing mode, AgX gives
# nicer highlight rolloff when the sun is on, so we keep it.
if args.colored:
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

# ---------- Freestyle (line drawing) — aggressive mode ----------
# Goal: draw every meaningful edge so thin geometry (glass mullions, open
# structural frames) becomes visible even when it doesn't form a clean
# silhouette. Trade-off: very dense meshes can look noisy.
scene.render.use_freestyle = True
view_layer = scene.view_layers[0]
view_layer.use_freestyle = True
fs = view_layer.freestyle_settings
fs.crease_angle = radians(30)   # was 140 — catches almost every face-to-face fold

while len(fs.linesets) > 0:
    fs.linesets.remove(fs.linesets[0])

ls = fs.linesets.new("BuildingLines")
ls.select_silhouette = True
ls.select_border = True
ls.select_crease = True
ls.select_contour = True
ls.select_external_contour = True
ls.select_material_boundary = True
ls.select_edge_mark = False
# Show partially-occluded lines too (QI 0–2): glass mullions behind front panes
# get drawn as well.
ls.select_by_visibility = True
ls.visibility = "RANGE"
ls.qi_start = 0
ls.qi_end = 2

# Create a linestyle: black, thickness controlled by CLI flag
linestyle = bpy.data.linestyles.new("BlackThick")
linestyle.color = (0, 0, 0)
linestyle.thickness = args.line_thickness
ls.linestyle = linestyle

if args.colored:
    # Default map covers the Betonwerk-style ArchiCAD export (white paint /
    # glass / steel). Override with --color-map-json for variants whose
    # material vocabulary is different (e.g. bridge_the_gap has Asphalt,
    # Beton, Marble, Farbe grün/grau/weiß, etc.).
    COLOR_MAP = {
        "glas": (0.55, 0.78, 0.92, 1.0),
        "metall": (1.0, 0.55, 0.20, 1.0),
        "stahl": (1.0, 0.55, 0.20, 1.0),
        "default": (0.92, 0.92, 0.92, 1.0),
    }
    if args.color_map_json:
        import json as _json
        with open(args.color_map_json) as _fh:
            raw = _json.load(_fh)
        if "default" not in raw:
            raise ValueError(f"--color-map-json must include a 'default' key: {args.color_map_json}")
        # Normalize 3-tuples to RGBA (append alpha=1.0 when missing)
        COLOR_MAP = {
            k.lower(): tuple(v) if len(v) == 4 else (*v, 1.0)
            for k, v in raw.items()
        }
        print(f"Loaded {len(COLOR_MAP) - 1} keyword colors + default from {args.color_map_json}")
    print("Applying flat emissive colors by OBJ material name:")
    for mat in bpy.data.materials:
        name_lower = mat.name.lower()
        chosen = COLOR_MAP["default"]
        for key, c in COLOR_MAP.items():
            if key == "default":
                continue
            if key in name_lower:
                chosen = c
                break
        if not mat.use_nodes:
            mat.use_nodes = True
        nt = mat.node_tree
        # Wipe existing shader; rebuild as Emission → Material Output
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = chosen
        emit.inputs["Strength"].default_value = 1.0
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        print(f"  {mat.name:>30s} → rgba {chosen}")
else:
    # Default: replace whatever materials the OBJ has with a single bright white
    # so Freestyle line work stands out against a clean fill.
    white_mat = bpy.data.materials.new("White")
    white_mat.use_nodes = True
    bsdf = white_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
        bsdf.inputs["Roughness"].default_value = 1.0
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(white_mat)


# ---------- Helper: place ortho camera, render to file ----------
# Shared target empty at building centroid — all non-top-down cameras track this.
_target = bpy.data.objects.new("CamTarget", None)
_target.location = (cx, cy, cz)
scene.collection.objects.link(_target)


def setup_camera_topdown(name, z_cut, ortho_scale, clip_start=0.1, clip_end=1e6):
    """Top-down ortho camera placed just above z_cut. Anything with z>z_cut is
    behind the camera and culled; everything z<z_cut is rendered."""
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name])
    cam_data = bpy.data.cameras.new(name)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = clip_start
    cam_data.clip_end = clip_end
    cam_obj = bpy.data.objects.new(name, cam_data)
    cam_obj.location = (cx, cy, z_cut + 100)
    cam_obj.rotation_euler = (0, 0, 0)  # default looks -Z
    scene.collection.objects.link(cam_obj)
    return cam_obj


def setup_camera_lookat(name, location, ortho_scale, target=_target, clip_start=10, clip_end=1e6):
    """Ortho camera at location with track_to constraint pointing at target."""
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name])
    cam_data = bpy.data.cameras.new(name)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = clip_start
    cam_data.clip_end = clip_end
    cam_obj = bpy.data.objects.new(name, cam_data)
    cam_obj.location = location
    scene.collection.objects.link(cam_obj)
    tc = cam_obj.constraints.new("TRACK_TO")
    tc.target = target
    tc.track_axis = "TRACK_NEGATIVE_Z"
    tc.up_axis = "UP_Y"
    bpy.context.view_layer.update()
    return cam_obj


def render_to(filepath, cam_obj):
    scene.camera = cam_obj
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    print(f"  -> {filepath}")


# ---------- Common framing ----------
size_x = xmax - xmin
size_y = ymax - ymin
size_z = zmax - zmin
# Use a single ortho scale for elevations so they're comparable
ortho_xy = max(size_x, size_y) * 1.05
ortho_xz = max(size_x, size_z) * 1.05
ortho_yz = max(size_y, size_z) * 1.05
PAD = 1.05

# ============================================================
# 1-4. PLANS (top-down, cut at given z)
# ============================================================
# Camera looks down -Z. To "cut" everything above z_cut:
#   place camera at z=cam_z above building, set clip_end = cam_z - z_cut.
#   Anything with z > z_cut becomes too-close (within clip_start) — actually
#   wrong direction. Top-down camera: world-z = cam_z - depth_from_camera.
#   So depth=0 at camera, increases going down. clip_start=10 means z <= cam_z-10
#   visible. clip_end=D means z >= cam_z - D visible.
# To hide stuff ABOVE z_cut: need camera below cut? No, top-down sees stuff in
# front of it (below). To show ONLY z < z_cut: place camera at z=z_cut,
# clip_start=0.01, clip_end=large. Then anything at z>z_cut is behind camera.

# --- Auto-detect floor elevations and derive plan cut heights ---
# detect_floors reads the OBJ directly (file-level, not Blender memory),
# so we must tell it which OBJ-vertex column is "up" — Blender already
# swapped axes for the in-Blender mesh, but the raw OBJ columns are unchanged.
floors_w = detect_floors_with_weights(
    args.obj,
    bin_size_mm=DETECT_BIN,
    merge_radius_mm=DETECT_MERGE,
    up_axis=args.source_up_axis,
)
plan_cuts = select_plan_cuts(zmin, floors_w, floor_offset_mm=FLOOR_OFFSET)
print("DETECTED_FLOORS_MM:", [(round(z, 1), round(w, 1)) for z, w in floors_w])
print("PLAN_CUTS_MM:", {k: round(v, 1) for k, v in plan_cuts.items()})

# Plan view modes:
#   Default line mode  : thin slab — shows only the geometry at THIS floor
#   --colored          : X-ray below cut, but bottom-clamped at Halle ceiling
#                        for Tower-zone plans so the Hall geometry (especially
#                        the curved Halle_Decke roof) doesn't leak into Tower
#                        plans.
slab = SLAB
_, _, hall_ceiling_z = split_hall_tower(floors_w)
for name, z_cut in plan_cuts.items():
    if args.colored:
        # Plans above Halle_Decke (= Tower zone) use a thin slab so the Hall's
        # parapets / extended envelope don't leak in. Plans inside the Hall use
        # full X-ray below the cut so vertical facade panels become visible as
        # colored regions.
        if z_cut > hall_ceiling_z:
            cam_z = z_cut + slab / 2
            clip_start = 0.1
            clip_end = slab + 0.1
        else:
            cam_z = z_cut + 100
            clip_start = 0.1
            clip_end = cam_z - zmin + 1000
    else:
        cam_z = z_cut + slab / 2
        clip_start = 0.1
        clip_end = slab + 0.1            # thin slab only
    cam_name = f"Cam_plan_{name}"
    if cam_name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[cam_name])
    cd = bpy.data.cameras.new(cam_name)
    cd.type = "ORTHO"
    cd.ortho_scale = ortho_xy * PAD
    cd.clip_start = clip_start
    cd.clip_end = clip_end
    cam = bpy.data.objects.new(cam_name, cd)
    cam.location = (cx, cy, cam_z)
    cam.rotation_euler = (0, 0, 0)
    scene.collection.objects.link(cam)
    render_to(os.path.join(args.out, f"plan_{name}.png"), cam)

if args.plans_only:
    print("plans-only mode: skipping remaining views")
    print("DONE")
    sys.exit(0)

# ============================================================
# 5-8. ELEVATIONS (track_to centroid, full building)
# ============================================================
elevation_cam_offset = max(size_x, size_y, size_z) * 1.5
elevations = [
    # name, location (camera is outside the bbox on this side)
    ("elevation_north.png", (cx, ymax + elevation_cam_offset, cz), max(size_x, size_z)),
    ("elevation_south.png", (cx, ymin - elevation_cam_offset, cz), max(size_x, size_z)),
    ("elevation_east.png", (xmax + elevation_cam_offset, cy, cz), max(size_y, size_z)),
    ("elevation_west.png", (xmin - elevation_cam_offset, cy, cz), max(size_y, size_z)),
]
for filename, loc, ortho_dim in elevations:
    cam = setup_camera_lookat(f"Cam_{filename}", loc, ortho_dim * PAD)
    render_to(os.path.join(args.out, filename), cam)

# ============================================================
# 9-10. SECTIONS (vertical cut at centroid plane, near half clipped)
# ============================================================
# Section NS = vertical cut at Y=cy, looking -Y from the +Y side. Use
# clip_start to bury everything on the near side of cy.
sections = [
    # name, cam_location, ortho_dim
    ("section_NS.png", (cx, cy + elevation_cam_offset, cz), max(size_x, size_z)),
    ("section_EW.png", (cx + elevation_cam_offset, cy, cz), max(size_y, size_z)),
]
for filename, loc, ortho_dim in sections:
    cam = setup_camera_lookat(
        f"Cam_{filename}",
        location=loc,
        ortho_scale=ortho_dim * PAD,
        clip_start=elevation_cam_offset,  # near plane sits at the building centroid
    )
    render_to(os.path.join(args.out, filename), cam)

# ============================================================
# 11-14. AXONOMETRIC — 4 corner views (NE/NW/SE/SW), each lit
# ============================================================
import math

iso_dist = max(size_x, size_y, size_z) * 2
el = radians(30)
axon_views = [
    ("axon_NE", radians(45)),
    ("axon_NW", radians(135)),
    ("axon_SW", radians(225)),
    ("axon_SE", radians(315)),
]
# In colored mode, keep emission flat (no sun mixing). In default line mode,
# the sun gives axon faces a soft gradient for 3D readability.
axon_with_sun = not args.colored
set_sun(axon_with_sun)
for name, az in axon_views:
    cam_loc = (
        cx + iso_dist * math.cos(el) * math.cos(az),
        cy + iso_dist * math.cos(el) * math.sin(az),
        cz + iso_dist * math.sin(el),
    )
    cam = setup_camera_lookat(
        f"Cam_{name}",
        location=cam_loc,
        ortho_scale=max(size_x, size_y, size_z) * 1.4,
    )
    render_to(os.path.join(args.out, f"{name}.png"), cam)
set_sun(False)

print("DONE")
