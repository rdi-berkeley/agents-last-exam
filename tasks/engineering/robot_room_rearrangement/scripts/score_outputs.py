"""Score humanoid object stowing submissions.

The performance signal is functional: the hidden grader replays the submitted
whole-body trajectory in the frozen MuJoCo scene and reports where each target
object ended up. This module adds the structure + schema gates and turns the
grader's result into a scalar score.

Scoring is binary: the target object must end inside the bin and the robot must
still be upright.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

TRAJECTORY_NAME = "trajectory.npz"

# (array name, required column count). The robot is driven by the same 4-array
# whole-body schema used by humanoid_loco_manipulation: joint targets plus the
# balance-policy command stream.
REQUIRED_ARRAYS = (
    ("action", 43),
    ("amo_policy_command", 9),
    ("amo_policy_target_yaw", 1),
    ("amo_policy_turning_flag", 1),
)


def _structure_gate(output_dir: Path) -> tuple[bool, str]:
    p = output_dir / TRAJECTORY_NAME
    if not p.exists():
        found = sorted(q.name for q in output_dir.iterdir() if q.is_file())
        return False, f"missing {TRAJECTORY_NAME} (found {found})"
    return True, "ok"


def _schema_gate(output_dir: Path) -> tuple[bool, str]:
    p = output_dir / TRAJECTORY_NAME
    try:
        npz = np.load(p)
    except Exception as exc:
        return False, f"{p.name}: cannot read npz ({type(exc).__name__}: {exc})"

    errs: list[str] = []
    lengths: set[int] = set()
    for name, cols in REQUIRED_ARRAYS:
        if name not in npz.files:
            errs.append(f"{p.name}: missing array '{name}' (found {list(npz.files)})")
            continue
        a = np.asarray(npz[name])
        if a.ndim != 2:
            errs.append(f"{p.name}: '{name}' must be 2-D [T,{cols}], got {a.shape}")
            continue
        if a.shape[1] != cols:
            errs.append(f"{p.name}: '{name}' must have {cols} columns, got {a.shape[1]}")
            continue
        if len(a) < 1:
            errs.append(f"{p.name}: '{name}' is empty")
            continue
        if not np.isfinite(np.asarray(a, dtype=np.float64)).all():
            errs.append(f"{p.name}: '{name}' contains NaN/Inf")
        lengths.add(len(a))

    # All four streams are consumed step-by-step in lockstep; ragged lengths
    # would silently truncate the episode during replay.
    if len(lengths) > 1:
        errs.append(f"{p.name}: arrays disagree on T: {sorted(lengths)}")

    return (not errs), ("ok" if not errs else "; ".join(errs))


def evaluate_submission(
    output_dir: Path,
    *,
    results_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return {"score": 0.0, "success": False, "error": "output_dir_missing",
                "details": str(output_dir)}

    ok, msg = _structure_gate(output_dir)
    if not ok:
        return {"score": 0.0, "success": False, "error": "structure_gate", "details": msg}

    ok, msg = _schema_gate(output_dir)
    if not ok:
        return {"score": 0.0, "success": False, "error": "schema_gate", "details": msg}

    if results_path is None or not Path(results_path).exists():
        return {"score": 0.0, "success": False, "error": "grader_did_not_run",
                "details": "the hidden grader produced no results file"}

    try:
        report = json.loads(Path(results_path).read_text())
    except Exception as exc:
        return {"score": 0.0, "success": False, "error": "grader_result_unreadable",
                "details": f"{type(exc).__name__}: {exc}"}

    success = bool(report.get("success", False))
    result = {
        "score": 1.0 if success else 0.0,
        "success": success,
        "in_bin_xy": report.get("in_bin_xy"),
        "below_rim": report.get("below_rim"),
        "robot_upright": report.get("robot_upright"),
        "object_final_pos": report.get("object_final_pos"),
        "replay_steps": report.get("replay_steps"),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--results", type=Path, default=None)
    args = ap.parse_args()
    print(json.dumps(evaluate_submission(args.output, results_path=args.results), indent=2))


if __name__ == "__main__":
    main()
