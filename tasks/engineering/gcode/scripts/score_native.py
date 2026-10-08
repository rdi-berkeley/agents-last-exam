"""Recompute native machining before comparing the resulting stock with the reference."""

import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def score_request(request_path):
    request_path = Path(request_path)
    work = request_path.parent
    began = time.monotonic()
    result = {"status": "unavailable"}
    try:
        from native_geometry import grade_replay

        request = json.loads(request_path.read_text())
        reference = Path(request["reference"])
        if not reference.is_file():
            raise FileNotFoundError("Evaluator reference is missing")
        region = json.loads(Path(request["region"]).read_text())
        config_path = work / "preview.json"
        config_path.write_text(json.dumps(request["runtime"]))
        preview = work / "native"
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("preview_stock.py")),
                request["project"],
                str(preview),
                "--config",
                str(config_path),
            ],
            timeout=3700,
            check=False,
        )
        replay = json.loads((preview / "replay/result.json").read_text())
        expected_code = 2 if replay["status"] == "invalid_delivery" else 0
        if process.returncode != expected_code:
            raise RuntimeError(f"Native preview failed with exit {process.returncode}")
        grade = grade_replay(replay, reference, region)
        result.update(status="completed", score=grade["score"], grading=grade)
    except Exception:
        result["error"] = traceback.format_exc()
    result["seconds"] = time.monotonic() - began
    partial = work / "terminal.json.partial"
    partial.write_text(json.dumps(result, indent=2))
    partial.replace(work / "terminal.json")
    return result


if __name__ == "__main__":
    outcome = score_request(sys.argv[1])
    raise SystemExit(0 if outcome["status"] == "completed" else 1)
