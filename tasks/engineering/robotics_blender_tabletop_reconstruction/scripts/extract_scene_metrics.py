from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


TARGET_OBJECTS = ["mustard_bottle", "mug", "potted_meat_can"]


def _rotation_deg(obj) -> list[float]:
    return [math.degrees(v) for v in obj.rotation_euler]


def _find_principled_material(obj) -> dict | None:
    material = obj.active_material
    if material is None and obj.material_slots:
        material = obj.material_slots[0].material
    if material is None or not material.use_nodes or material.node_tree is None:
        return None
    principled = None
    for node in material.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            principled = node
            break
    if principled is None:
        return None
    base_color = principled.inputs["Base Color"].default_value
    return {
        "base_color_rgb": [float(base_color[0]), float(base_color[1]), float(base_color[2])],
        "roughness": float(principled.inputs["Roughness"].default_value),
        "metallic": float(principled.inputs["Metallic"].default_value),
    }


def _object_metrics(name: str) -> dict | None:
    obj = bpy.data.objects.get(name)
    if obj is None:
        return None
    return {
        "location": [float(v) for v in obj.location],
        "rotation_deg": _rotation_deg(obj),
        "material": _find_principled_material(obj),
    }


def _light_metrics() -> list[dict]:
    lights = []
    for obj in bpy.data.objects:
        if obj.type != "LIGHT" or obj.data is None:
            continue
        lights.append(
            {
                "name": obj.name,
                "type": obj.data.type,
                "location": [float(v) for v in obj.location],
                "energy": float(getattr(obj.data, "energy", 0.0)),
            }
        )
    return lights


def _exists(path_str: str) -> bool:
    return Path(path_str).exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract scene metrics from the current Blender scene.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--render-path", required=True)
    parser.add_argument("--mustard-path", required=True)
    parser.add_argument("--mug-path", required=True)
    parser.add_argument("--potted-meat-path", required=True)
    parser.add_argument("--full-scene-path", required=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    payload = {
        "objects": {name: _object_metrics(name) for name in TARGET_OBJECTS},
        "lights": _light_metrics(),
        "exports": {
            "verification_render": _exists(args.render_path),
            "mustard_bottle": _exists(args.mustard_path),
            "mug": _exists(args.mug_path),
            "potted_meat_can": _exists(args.potted_meat_path),
            "full_scene": _exists(args.full_scene_path),
        },
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
