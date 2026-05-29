#!/usr/bin/env python
"""Run the packaged subscription ops evaluator and emit JSON."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


NODE_HOME = Path.home() / ".local" / "agenthle" / "node-v22.22.2-linux-x64"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir).resolve()
    evaluator = reference_dir / "evaluate.py"
    if not evaluator.is_file():
        raise FileNotFoundError(f"missing evaluator: {evaluator}")

    env = dict(os.environ)
    if (NODE_HOME / "bin" / "node").is_file():
        env["PATH"] = f"{NODE_HOME / 'bin'}:{env.get('PATH', '')}"

    result = subprocess.run(
        ["python", str(evaluator), "--submission-dir", str(Path(args.submission_dir).resolve())],
        cwd=reference_dir,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.stderr:
        print(result.stderr, end="", file=__import__("sys").stderr)
    print(result.stdout, end="")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
