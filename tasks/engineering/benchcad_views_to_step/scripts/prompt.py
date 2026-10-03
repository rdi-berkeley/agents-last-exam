"""Single source of the agent-facing text for engineering/benchcad_views_to_step.

main.py renders task_text() with the VM's absolute paths; task_card.json's taskPrompt and the staged
input/README.md are the same text with the relative paths input/parts, input/README.md and output
(tests/test_task_hooks.py checks that the three stay identical)."""
from __future__ import annotations

TITLE = "Rebuild standard mechanical parts as solids from orthographic renders"


def task_text(parts_dir: str, readme_path: str, output_dir: str) -> str:
    return (
        "You are a mechanical design engineer. A customer sends you catalogue drawings of standard parts and needs "
        "editable 3D models.\n\n"
        "## Input\n"
        f"- Twelve parts in `{parts_dir}/<stem>/`: `view_0.png` .. `view_3.png` are orthographic renders, "
        "`composite.png` puts them side by side, and `meta.json` names the part family (for example helical_gear, "
        "round_flange, ball_knob), the governing standard when there is one (for example ISO 53, DIN 319), and a "
        "difficulty tag.\n"
        f"- `{readme_path}` repeats these instructions.\n\n"
        "## What you must do\n"
        "1. Study the renders and the standard to infer the geometry and proportions of each part.\n"
        "2. Model each part as a single closed solid with any open-source CAD tool (CadQuery, build123d, FreeCAD, "
        "OpenSCAD are installed).\n"
        f"3. Export every part to `{output_dir}/<stem>.step` (STEP AP214, exactly one closed solid per file, any units, "
        "any placement).\n\n"
        "## Scoring\n"
        "Each output solid is centred on its bounding box and scaled to unit bounding-box diagonal, then compared with "
        "the hidden reference solid by volume IoU (exact OpenCascade boolean intersection). All 24 proper axis-aligned "
        "rotations of the output are evaluated and the best IoU counts, so orientation is free. Part score = "
        "(IoU - 0.5) / 0.5 clipped to [0, 1]: a plain bounding box scores about 0, a faithful model above 0.9. Task "
        "score = mean over the twelve parts. A missing, unreadable, invalid or zero-volume STEP scores 0 for that part. "
        "If a file contains several solids they are fused into one shape before grading; do not rely on this, deliver "
        "one solid."
    )


README_TIPS = (
    "\n\n## Tips\n"
    "- `cadquery.exporters.export(wp, \"output/<stem>.step\")` writes a valid STEP AP214 file; FreeCAD's `Part.export` "
    "and build123d's `export_step` work too. OpenSCAD models must be converted to a true B-rep solid (not a mesh) "
    "before export.\n"
    "- Proportions and internal features (holes, teeth, fillets, wall thickness) determine the IoU: a solid with the "
    "right outline but a wrong interior loses credit in proportion to the misplaced volume.\n"
)


def readme_text() -> str:
    return f"# {TITLE}\n\n" + task_text("input/parts", "input/README.md", "output") + README_TIPS
