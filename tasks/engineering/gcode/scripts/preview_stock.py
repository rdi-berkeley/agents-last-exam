"""Replay a saved CAM project against public inputs without loading reference answers."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=os.environ.get("ALE_CAM_PREVIEW_CONFIG"))
    args = parser.parse_args()
    if args.config is None:
        parser.error("The environment must provide ALE_CAM_PREVIEW_CONFIG or --config")
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = {
        "project": str(args.project.resolve()),
        "output": str(output),
        "source_document": config["source_document"],
        "library_root": config["library_root"],
        "interpreter_runtime": config["interpreter_runtime"],
        "collision_python": config["collision_python"],
        "stock_binary": config["stock_binary"],
    }
    request_path = output / "request.json"
    request_path.write_text(json.dumps(request, indent=2))
    environment = {**os.environ, "ALE_CAM_PREVIEW_REQUEST": str(request_path)}
    command = [
        config["freecad"],
        "-u",
        str(output / "freecad.cfg"),
        str(Path(__file__).with_name("preview_stock.FCMacro")),
    ]
    if not environment.get("DISPLAY"):
        display_runner = shutil.which("xvfb-run")
        if display_runner is None:
            raise RuntimeError("A display or xvfb-run is required by the native CAM exporter")
        command = [display_runner, "-a", *command]
    with (output / "native.log").open("w") as log:
        process = subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=3600,
        )
    result_path = output / "replay/result.json"
    if not result_path.exists():
        raise RuntimeError(f"Native CAM preview failed; inspect {output / 'native.log'}")
    result = json.loads(result_path.read_text())
    summary = {
        "status": result["status"],
        "result": str(result_path),
        "stock": result.get("stock", {}).get("output"),
        "error": result.get("error"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    if result["status"] == "native_replay_complete" and process.returncode == 0:
        return 0
    return 2 if result["status"] == "invalid_delivery" else 1


if __name__ == "__main__":
    raise SystemExit(main())
