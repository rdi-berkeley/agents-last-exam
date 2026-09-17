"""Stage only solver-visible files and print their hashes for an isolated evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_fixtures import CALCULUS_SPEC, TASK_SPEC, build_fixtures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    input_dir = destination / "input"
    output_dir = destination / "output"
    if input_dir.exists() or output_dir.exists():
        parser.error("destination must not already contain input/ or output/")
    scenarios, _, _, _ = build_fixtures()
    files = {
        "TASK.md": TASK_SPEC,
        "calculus.md": CALCULUS_SPEC,
        "scenarios.json": json.dumps(scenarios, indent=2, sort_keys=True),
        "check.py": Path(__file__).with_name("verify_outputs.py").read_text(encoding="utf-8"),
    }
    input_dir.mkdir(parents=True)
    output_dir.mkdir()
    for filename, content in files.items():
        path = input_dir / filename
        path.write_text(content, encoding="utf-8")
        print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  input/{filename}")
    print(f"Task directory: {destination}")


if __name__ == "__main__":
    main()
