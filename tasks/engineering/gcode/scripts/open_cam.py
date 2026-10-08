"""Open a working CAM project with the original tool bindings enabled."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("preview.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    environment = {
        **os.environ,
        "ALE_CAM_ADAPTER_DIR": str(Path(__file__).resolve().parent),
        "ALE_CAM_LIBRARY_DIR": config["library_root"],
        "ALE_CAM_DOCUMENT": str(args.project.resolve(strict=True)),
    }
    return subprocess.call(
        [
            config["freecad"],
            str(Path(__file__).with_name("open_cam.FCMacro")),
        ],
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
