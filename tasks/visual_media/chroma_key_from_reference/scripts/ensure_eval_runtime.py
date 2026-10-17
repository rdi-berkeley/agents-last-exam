from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import venv


REQUIREMENTS = ("numpy==2.5.0", "opencv-python-headless==4.13.0.92")
PROBE = """
import importlib.metadata
import json
import sys
import cv2
import numpy

expected = dict(requirement.split('==') for requirement in sys.argv[1:])
versions = {name: importlib.metadata.version(name) for name in expected}
assert versions == expected, versions
assert cv2.Canny(numpy.zeros((8, 8), dtype=numpy.uint8), 80, 160).shape == (8, 8)
print(json.dumps({'python': sys.executable, 'versions': versions}))
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--input-video", required=True)
    args = parser.parse_args()
    runtime = args.runtime_dir.resolve()
    interpreter = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.is_file():
        venv.EnvBuilder(with_pip=True).create(runtime)
    probe = subprocess.run(
        [str(interpreter), "-c", PROBE, *REQUIREMENTS], capture_output=True, text=True
    )
    if probe.returncode:
        subprocess.run(
            [str(interpreter), "-m", "pip", "install", "--disable-pip-version-check",
             "--only-binary=:all:", *REQUIREMENTS],
            check=True,
        )
        probe = subprocess.run(
            [str(interpreter), "-c", PROBE, *REQUIREMENTS],
            capture_output=True, text=True, check=True,
        )
    subprocess.run(
        [str(interpreter), "-c",
         "import cv2, sys; capture = cv2.VideoCapture(sys.argv[1]); "
         "success, frame = capture.read(); capture.release(); "
         "assert success and frame is not None, 'Cannot decode task input video'",
         args.input_video],
        check=True,
    )
    print(json.dumps(json.loads(probe.stdout)))


if __name__ == "__main__":
    main()
