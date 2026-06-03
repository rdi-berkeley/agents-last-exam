from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--renderer-script", required=True)
    parser.add_argument("--evaluation-config", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "render_report.json"
    config = json.loads(Path(args.evaluation_config).read_text(encoding="utf-8"))
    blender_binary = os.environ.get("BLENDER_BINARY")
    if not blender_binary:
        failure = {
            "validity_gate_passed": False,
            "gate_fail_reasons": ["missing_blender_binary_env"],
            "view_paths": {},
            "silhouette_paths": [],
            "sample_positions": [],
        }
        report_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    sample_count = int(config.get("sample_count", 10))
    image_width = int(config.get("image_width", 512))
    image_height = int(config.get("image_height", 500))

    cmd = [
        blender_binary,
        "--background",
        args.blend,
        "--python",
        args.renderer_script,
        "--",
        "--output-dir",
        str(output_dir),
        "--sample-count",
        str(sample_count),
        "--image-width",
        str(image_width),
        "--image-height",
        str(image_height),
        "--evaluation-config",
        args.evaluation_config,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        failure = {
            "validity_gate_passed": False,
            "gate_fail_reasons": ["blender_render_failed"],
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "view_paths": {},
            "silhouette_paths": [],
            "sample_positions": [],
        }
        report_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
