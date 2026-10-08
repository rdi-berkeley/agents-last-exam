"""Check the preinstalled CAM runtime and converted public inputs before solving."""

import importlib
import json
import os
from pathlib import Path
import shutil
import sys


def check_runtime(config_path, region_path):
    if sys.version_info < (3, 11):
        raise RuntimeError("The geometry evaluator requires Python 3.11 or newer")
    config = json.loads(Path(config_path).read_text())
    for name in ("numpy", "scipy.spatial", "trimesh", "shapely", "fcl"):
        importlib.import_module(name)
    for name in ("freecad", "collision_python", "stock_binary"):
        path = Path(config[name])
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"Required runtime executable: {path}")
    interpreter = Path(config["interpreter_runtime"]) / "usr/bin/rs274"
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise FileNotFoundError(f"Required LinuxCNC interpreter: {interpreter}")
    for name in ("bwrap", "xvfb-run"):
        if shutil.which(name) is None:
            raise FileNotFoundError(f"Required executable: {name}")
    if not Path(config["source_document"]).is_file():
        raise FileNotFoundError("Converted original blank.FCStd is missing")
    library = Path(config["library_root"])
    manifest = json.loads((library / "original-tools.json").read_text())
    if not manifest.get("tools"):
        raise ValueError("Converted original tool library is empty")
    region = json.loads(Path(region_path).read_text())
    if region.get("derived_from_reference_or_candidate") is not False or not region.get("sources"):
        raise ValueError("Machining region must be prepared from original public geometry")
    return {"runtime_ready": True, "original_tools": len(manifest["tools"])}


if __name__ == "__main__":
    print(json.dumps(check_runtime(sys.argv[1], sys.argv[2])))
