"""Grader for engineering/benchcad_views_to_step.

For every staged part the agent must deliver output/<stem>.step. The hidden reference holds the single solid built
from the ground-truth CadQuery program. Both shapes are normalised (translated to the bounding-box centre, scaled so
the bounding-box diagonal is 1) and compared by volume IoU = V(A∩B) / (V(A) + V(B) - V(A∩B)), computed with an exact
OpenCascade boolean intersection.

Orientation is free: ALL 24 proper axis-aligned rotations of the prediction (the signed permutation matrices with
determinant +1, enumerated in a fixed order) are evaluated and the best IoU is kept. There is no candidate pruning
and no early exit, so the result does not depend on timing or on the machine.

Output files that contain several solids are fused (boolean union) into one shape before grading, so overlapping
volume is not double-counted; the deliverable is nevertheless one closed solid per file. Reference files must
contain exactly one solid (checked, RuntimeError otherwise).

Each part is graded in its own child process (this file run with --part) so that a pathological input, for example
a mesh-converted STEP with tens of thousands of faces, cannot stall evaluate(): a part whose 24 booleans do not
finish within PART_TIME_BUDGET_S keeps the best IoU found so far and is flagged in its note. Well-formed B-rep
solids finish far inside the budget (the full reference set self-graded takes about 2 s to 5 min per part).

Part score = clip((IoU - 0.5) / 0.5): a plain bounding box scores about 0, a faithful model above 0.9.
Task score = mean over parts (missing or invalid STEP = 0 for that part).
Requires CadQuery / OCP.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import cadquery as cq
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopTools import TopTools_ListOfShape

N_ROTATIONS = 24
PART_TIME_BUDGET_S = 900.0   # safety valve only; see module docstring
MAX_WORKERS = max(1, min(4, os.cpu_count() or 1))
IOU_ZERO = 0.5               # part score = (IoU - IOU_ZERO) / (1 - IOU_ZERO), clipped: a bounding-box guess scores ~0

ROTATIONS: list[list[list[float]]] = []
for _perm in itertools.permutations(range(3)):
    for _signs in itertools.product((1, -1), repeat=3):
        m = [[0.0] * 3 for _ in range(3)]
        for i, p in enumerate(_perm):
            m[i][p] = float(_signs[i])
        det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        if det > 0:
            ROTATIONS.append(m)
assert len(ROTATIONS) == N_ROTATIONS


def _volume(shape) -> float:
    props = GProp_GProps(); BRepGProp.VolumeProperties_s(shape, props); return props.Mass()


def _shape_list(shapes) -> TopTools_ListOfShape:
    lst = TopTools_ListOfShape()
    for s in shapes:
        lst.Append(s.wrapped)
    return lst


def fuse_solids(solids):
    """Boolean union of several cadquery Solids; returns a cadquery Shape (a Solid, or a Compound if disjoint)."""
    op = BRepAlgoAPI_Fuse(); op.SetArguments(_shape_list(solids[:1])); op.SetTools(_shape_list(solids[1:])); op.SetRunParallel(True); op.Build()
    if not op.IsDone():
        raise RuntimeError("boolean union of the solids failed")
    return cq.Shape.cast(op.Shape())


def load_shape(path: str):
    """Return (cadquery Shape, n_solids, error). Several solids are fused into one shape."""
    try:
        wp = cq.importers.importStep(path)
    except Exception as exc:
        return None, 0, f"unreadable STEP: {exc}"
    solids = wp.solids().vals()
    if not solids:
        return None, 0, "no solid in STEP"
    try:
        shape = solids[0] if len(solids) == 1 else fuse_solids(solids)
    except Exception as exc:
        return None, len(solids), f"{len(solids)} solids could not be fused: {exc}"
    if not BRepCheck_Analyzer(shape.wrapped).IsValid():
        return None, len(solids), "invalid B-rep"
    if _volume(shape.wrapped) <= 1e-9:
        return None, len(solids), "zero volume"
    return shape, len(solids), None


def load_solid(path: str):
    """Backwards-compatible wrapper: (shape, error)."""
    shape, _, err = load_shape(path)
    return shape, err


def load_reference(path: str):
    shape, n, err = load_shape(path)
    if shape is None:
        raise RuntimeError(f"reference unusable: {err}")
    if n != 1:
        raise RuntimeError(f"reference must contain exactly one solid, found {n}")
    return shape


def _extents(shape) -> tuple[float, float, float]:
    bb = shape.BoundingBox(); return (bb.xlen, bb.ylen, bb.zlen)


def normalise(shape):
    bb = shape.BoundingBox(); c = cq.Vector((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2)
    diag = math.sqrt(bb.xlen ** 2 + bb.ylen ** 2 + bb.zlen ** 2) or 1.0
    return shape.translate(-c).scale(1.0 / diag)


def rotation_matrix(k: int) -> cq.Matrix:
    m = ROTATIONS[k]
    return cq.Matrix([[m[0][0], m[0][1], m[0][2], 0], [m[1][0], m[1][1], m[1][2], 0], [m[2][0], m[2][1], m[2][2], 0], [0, 0, 0, 1]])


def iou(a, b) -> float:
    va, vb = _volume(a.wrapped), _volume(b.wrapped)
    op = BRepAlgoAPI_Common(); op.SetArguments(_shape_list([a])); op.SetTools(_shape_list([b])); op.SetRunParallel(True); op.Build()
    vi = _volume(op.Shape()) if op.IsDone() else 0.0
    vu = va + vb - vi
    return max(0.0, min(1.0, vi / vu)) if vu > 1e-12 else 0.0


def iou_per_rotation(pred, ref):
    """Yield (k, IoU) for all 24 rotations of the normalised prediction against the normalised reference."""
    ref_n = normalise(ref); pred_n = normalise(pred)
    for k in range(N_ROTATIONS):
        try:
            v = iou(normalise(pred_n.transformGeometry(rotation_matrix(k))), ref_n)
        except Exception:
            v = 0.0
        yield k, v


def best_iou(pred, ref) -> float:
    """In-process full enumeration (used by tests and calibration; evaluate() goes through grade_part)."""
    return max(v for _, v in iou_per_rotation(pred, ref))


def _child_main(pred_path: str, ref_path: str) -> None:
    """Entry point of the per-part worker: one JSON line per rotation, flushed as soon as it is known."""
    pred, n, err = load_shape(pred_path)
    if pred is None:
        print(json.dumps({"error": err, "n_solids": n}), flush=True); return
    ref = load_reference(ref_path)
    print(json.dumps({"n_solids": n}), flush=True)
    for k, v in iou_per_rotation(pred, ref):
        print(json.dumps({"k": k, "iou": v}), flush=True)


def grade_part(pred_path: str, ref_path: str, time_budget_s: float = PART_TIME_BUDGET_S) -> dict:
    """Grade one part in a child process. Returns {"iou", "part_score", "note", "n_rotations", "n_solids"}."""
    cmd = [sys.executable or "python3", os.path.abspath(__file__), "--part", pred_path, ref_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=time_budget_s); budget_hit = False
    except subprocess.TimeoutExpired:
        proc.kill(); out, err = proc.communicate(); budget_hit = True
    best = 0.0; n_rot = 0; n_solids = 0; error = None
    for line in out.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if "error" in rec:
            error = rec["error"]
        if "n_solids" in rec:
            n_solids = rec["n_solids"]
        if "k" in rec:
            n_rot += 1; best = max(best, float(rec["iou"]))
    if error is not None:
        note = error
    elif proc.returncode not in (0, None) and not budget_hit:
        note = f"grader worker failed: {err.strip()[-300:]}"
    elif budget_hit:
        note = f"time budget {time_budget_s:.0f} s hit after {n_rot}/{N_ROTATIONS} rotations; best so far kept"
    elif n_rot < N_ROTATIONS:
        note = f"worker returned only {n_rot}/{N_ROTATIONS} rotations"
    else:
        note = "ok" if n_solids == 1 else f"ok ({n_solids} solids fused before grading)"
    part_score = max(0.0, min(1.0, (best - IOU_ZERO) / (1 - IOU_ZERO)))
    return {"iou": round(best, 4), "part_score": round(part_score, 4), "note": note, "n_rotations": n_rot, "n_solids": n_solids}


def score_parts(pred_paths: dict[str, str | None], ref_paths: dict[str, str], time_budget_s: float = PART_TIME_BUDGET_S, workers: int = MAX_WORKERS) -> dict:
    """pred_paths: stem -> STEP path or None (missing). ref_paths: stem -> reference STEP. Raises RuntimeError if a
    reference is unusable (infrastructure error, not the agent's fault)."""
    for stem, rp in ref_paths.items():
        try:
            load_reference(rp)
        except RuntimeError as exc:
            raise RuntimeError(f"reference {stem}: {exc}")
    per = {stem: {"iou": 0.0, "part_score": 0.0, "note": "missing", "n_rotations": 0, "n_solids": 0} for stem in ref_paths}
    todo = [stem for stem in ref_paths if pred_paths.get(stem)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for stem, res in zip(todo, ex.map(lambda s: grade_part(pred_paths[s], ref_paths[s], time_budget_s), todo)):
            per[stem] = res
    score = sum(v["part_score"] for v in per.values()) / len(per) if per else 0.0
    return {"score": round(max(0.0, min(1.0, score)), 4), "per_part": per}


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--part":
        _child_main(sys.argv[2], sys.argv[3])
    else:
        sys.exit("usage: grade.py --part <pred.step> <ref.step>   (worker mode; use score_parts() from Python)")
