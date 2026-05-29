"""VM-side evaluator launcher for the brain MRI registration benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--evaluator-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_dir = Path(args.candidate_dir)
    evaluator_dir = Path(args.evaluator_dir)
    evaluate_py = evaluator_dir / "evaluate.py"
    requirements = evaluator_dir / "evaluator_only" / "requirements.txt"

    for path in [candidate_dir, evaluate_py, requirements]:
        if not path.exists():
            print(
                json.dumps(
                    {
                        "score": 0.0,
                        "normalized_score": 0.0,
                        "passed": False,
                        "error": f"missing required path: {path}",
                    }
                )
            )
            return 0

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    cmd = [
        "uv",
        "run",
        "--isolated",
        "--python",
        "python3.12",
        "--no-project",
        "--with-requirements",
        str(requirements),
        "python",
        str(evaluate_py),
        "--submission-dir",
        str(candidate_dir),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        print(
            json.dumps(
                {
                    "score": 0.0,
                    "normalized_score": 0.0,
                    "passed": False,
                    "error": "evaluator command failed",
                    "return_code": completed.returncode,
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                }
            )
        )
        return 0

    report = json.loads(completed.stdout)
    raw_score = float(report.get("score", 0.0))
    report["normalized_score"] = max(0.0, min(1.0, raw_score / 100.0))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
