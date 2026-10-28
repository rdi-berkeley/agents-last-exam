"""Score robot dexterous grasping submissions.

The performance signal is functional: the hidden grader replays each submitted
upper-body trajectory in its object's MuJoCo scene (lower body held standing by
the bundled balance policy) and reports whether the object was grasped and
lifted. This module adds the structure + schema gates and turns the grader's
result into a scalar score. Scoring is all-or-nothing over the target objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

UPPER_BODY_DIM = 31


def _structure_gate(output_dir: Path, object_ids) -> tuple[bool, str]:
    missing = [oid for oid in object_ids
               if not (output_dir / f"trajectory_{oid}.npz").exists()]
    if missing:
        found = sorted(p.name for p in output_dir.iterdir() if p.is_file())
        return False, f"missing trajectory files for objects {missing} (found {found})"
    return True, "ok"


def _schema_gate(output_dir: Path, object_ids) -> tuple[bool, str]:
    errs = []
    for oid in object_ids:
        p = output_dir / f"trajectory_{oid}.npz"
        try:
            npz = np.load(p)
        except Exception as exc:
            errs.append(f"{p.name}: cannot read npz ({type(exc).__name__}: {exc})")
            continue
        if "upper_body" not in npz.files:
            errs.append(f"{p.name}: missing array 'upper_body' (found {list(npz.files)})")
            continue
        a = npz["upper_body"]
        if a.ndim != 2:
            errs.append(f"{p.name}: 'upper_body' must be 2-D [T,{UPPER_BODY_DIM}], got {a.shape}")
        elif a.shape[1] != UPPER_BODY_DIM:
            errs.append(f"{p.name}: 'upper_body' must have {UPPER_BODY_DIM} columns, got {a.shape[1]}")
        elif len(a) < 1:
            errs.append(f"{p.name}: trajectory is empty")
        if not np.isfinite(np.asarray(a, dtype=np.float64)).all():
            errs.append(f"{p.name}: 'upper_body' contains NaN/Inf")
    return (not errs), ("ok" if not errs else "; ".join(errs))


def evaluate_submission(
    output_dir: Path,
    *,
    object_ids,
    results_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()

    ok, reason = _structure_gate(output_dir, object_ids)
    if not ok:
        return {"score": 0.0, "reason": "structure_gate_failed", "details": reason}

    ok, reason = _schema_gate(output_dir, object_ids)
    if not ok:
        return {"score": 0.0, "reason": "schema_gate_failed", "details": reason}

    if results_path is None or not Path(results_path).exists():
        return {
            "score": 0.0,
            "reason": "no_grader_results",
            "details": "submission passed structure/schema gates, but the replay grader produced no results",
        }

    grader = json.loads(Path(results_path).read_text(encoding="utf-8"))
    n_ok = int(grader.get("objects_succeeded", 0))
    n = int(grader.get("objects_total", len(object_ids)))
    score = 1.0 if (n > 0 and n_ok == n) else 0.0   # all-or-nothing
    return {
        "score": score,
        "reason": "graded" if score > 0 else "grader_failed",
        "objects_succeeded": n_ok,
        "objects_total": n,
        "grader": grader,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--object-ids", required=True,
                        help="comma-separated object ids, e.g. 102,103,105")
    parser.add_argument("--results-path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ids = [s.strip() for s in args.object_ids.split(",") if s.strip()]
    result = evaluate_submission(args.output_dir, object_ids=ids,
                                 results_path=args.results_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
