"""Score humanoid loco-manipulation trajectory submissions.

The performance signal is functional: the hidden grader replays the submitted
`trajectory.npz` in MuJoCo under the bundled AMO locomotion policy and reports
whether the target object was lifted. This module adds the structure + schema
gates and turns the grader's result into a scalar score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_ARRAYS = {
    "action": 43,
    "amo_policy_command": 9,
    "amo_policy_target_yaw": None,
    "amo_policy_turning_flag": None,
}


def _structure_gate(output_dir: Path) -> tuple[bool, str]:
    files = {p.name for p in output_dir.iterdir() if p.is_file()}
    if "trajectory.npz" not in files:
        return False, f"missing trajectory.npz (found {sorted(files)})"
    return True, "ok"


def _schema_gate(npz_path: Path) -> tuple[bool, str]:
    try:
        npz = np.load(npz_path)
    except Exception as exc:
        return False, f"cannot read npz: {type(exc).__name__}: {exc}"
    errs = []
    for key, cols in REQUIRED_ARRAYS.items():
        if key not in npz.files:
            errs.append(f"missing array '{key}'")
            continue
        arr = npz[key]
        if arr.ndim != 2:
            errs.append(f"'{key}' must be 2-D [T,C], got {arr.shape}")
        elif cols is not None and arr.shape[1] != cols:
            errs.append(f"'{key}' must have {cols} columns, got {arr.shape[1]}")
        if not np.isfinite(np.asarray(arr, dtype=np.float64)).all():
            errs.append(f"'{key}' contains NaN/Inf")
    if not errs:
        lengths = {k: len(npz[k]) for k in REQUIRED_ARRAYS}
        if len(set(lengths.values())) != 1:
            errs.append(f"arrays must share length T, got {lengths}")
        elif next(iter(lengths.values())) < 1:
            errs.append("trajectory is empty")
    return (not errs), ("ok" if not errs else "; ".join(errs))


def evaluate_submission(
    output_dir: Path,
    *,
    results_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()

    ok, reason = _structure_gate(output_dir)
    if not ok:
        return {"score": 0.0, "reason": "structure_gate_failed", "details": reason}

    ok, reason = _schema_gate(output_dir / "trajectory.npz")
    if not ok:
        return {"score": 0.0, "reason": "schema_gate_failed", "details": reason}

    if results_path is None or not Path(results_path).exists():
        return {
            "score": 0.0,
            "reason": "no_grader_results",
            "details": "submission passed structure/schema gates, but the replay grader produced no results",
        }

    grader = json.loads(Path(results_path).read_text(encoding="utf-8"))
    score = float(grader.get("score", 0.0))
    # A fallen robot is a hard failure regardless of transient lift.
    if not grader.get("robot_upright", True):
        score = 0.0
    return {
        "score": score,
        "reason": "graded" if score > 0 else "grader_failed",
        "grader": grader,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--results-path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_submission(args.output_dir, results_path=args.results_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
