"""VM-side evaluator launcher for clinic_energy_model_calibration_ecm."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--evaluator-dir", required=True)
    parser.add_argument("--energyplus-bin", required=True)
    return parser.parse_args()


def _zero_report(error: str, **extra: object) -> dict[str, object]:
    report: dict[str, object] = {
        "score": 0.0,
        "normalized_score": 0.0,
        "passed": False,
        "error": error,
    }
    report.update(extra)
    return report


def main() -> int:
    args = parse_args()
    candidate_dir = Path(args.candidate_dir)
    evaluator_dir = Path(args.evaluator_dir)
    energyplus_bin = Path(args.energyplus_bin)
    evaluate_py = evaluator_dir / "evaluate.py"

    for path in [candidate_dir, evaluate_py, energyplus_bin]:
        if not path.exists():
            print(json.dumps(_zero_report(f"missing required path: {path}"), sort_keys=True))
            return 0

    completed = subprocess.run(
        [
            sys.executable,
            str(evaluate_py),
            "--submission-dir",
            str(candidate_dir),
            "--energyplus-bin",
            str(energyplus_bin),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")

    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        print(
            json.dumps(
                _zero_report(
                    "evaluator returned non-JSON stdout",
                    return_code=completed.returncode,
                    stdout_tail=completed.stdout[-4000:],
                    stderr_tail=completed.stderr[-4000:],
                ),
                sort_keys=True,
            )
        )
        return 0

    raw_score = float(report.get("score", 0.0))
    report["normalized_score"] = max(0.0, min(1.0, raw_score / 100.0))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

