"""Grader for engineering/cad_part_from_drawing_rubric.

For each part the agent delivers output/<stem>.step. Host-side we
  1. load the STEP with CadQuery (hard gate: readable, one or more valid solids, positive volume),
  2. render six deterministic shaded views (front, back, top, bottom, right, isometric) from the tessellated
     solid with matplotlib (back-face culling, flat directional shading; no GPU, no fonts),
  3. ask a vision judge, `ALE_JUDGE_VOTES` times, whether each hidden rubric requirement written by the domain
     expert (reference/<stem>/rubrics.json) is clearly satisfied by the rendered model, with the original drawing
     (reference/<stem>/drawing_images.json, base64 PNG pages) shown alongside so proportions can be compared;
     majority vote per rubric,
  4. when the drawing states the part's overall dimensions (reference/<stem>/dims.json: sorted extents in mm), compare
     them with the solid's sorted bounding-box extents: dim_factor = 1 at <= 5 % maximum relative error, falling
     linearly to 0 at 30 %; parts whose drawing gives no overall dimensions use dim_factor = 1.
part score = (satisfied rubrics / rubrics) * dim_factor; task score = mean over parts. Missing or invalid output = 0.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from tasks.engineering.cad_part_from_drawing_rubric.scripts.judge_client import chat_json, votes

VIEWS = {
    "front (looking along -Y)": (90, 0), "back (looking along +Y)": (90, 180), "top (looking down -Z)": (0, 0),
    "bottom (looking up +Z)": (180, 0), "right (looking along -X)": (90, 90), "isometric": (35.264, 45),
}
SYSTEM = (
    "You are a strict mechanical-engineering design reviewer. You are shown the original dimensioned drawing of a part, "
    "rendered views of a 3D CAD model produced by a candidate, and a list of requirements written by the original designer. "
    "For each requirement decide whether the candidate's model CLEARLY satisfies it AS DRAWN: the feature must exist, with the "
    "count and arrangement stated, and its proportions must visibly agree with the drawing. Be strict: if the geometry is "
    "missing, clearly wrong, visibly out of proportion with the drawing, or you cannot tell from the views, answer false. "
    "The model may be positioned or oriented differently from the drawing (upside down, rotated, mirrored views): judge "
    "shape, counts, arrangement and proportions, never orientation or placement. Judge only geometry; ignore colour and "
    "rendering style. "
    'Answer with a JSON object {"verdicts": {"R1": true/false, ...}, "notes": "<one line>"}.'
)
DIM_TOL_FULL, DIM_TOL_ZERO = 0.05, 0.30


def load_solid(path: str):
    import cadquery as cq
    try:
        shape = cq.importers.importStep(path).val()
    except Exception as exc:
        return None, f"unreadable STEP: {exc}"
    solids = shape.Solids() if hasattr(shape, "Solids") else []
    if not solids:
        return None, "no solid in STEP"
    comp = cq.Compound.makeCompound(solids) if len(solids) > 1 else solids[0]
    if not comp.isValid():
        return None, "solid fails BRepCheck"
    if comp.Volume() <= 1e-9:
        return None, "zero volume"
    return comp, None


def _mesh(shape, tol_rel=0.002):
    bb = shape.BoundingBox(); diag = math.sqrt(bb.xlen ** 2 + bb.ylen ** 2 + bb.zlen ** 2) or 1.0
    verts, tris = shape.tessellate(diag * tol_rel, 0.3)
    v = np.array([[p.x, p.y, p.z] for p in verts]); t = np.array(tris, dtype=int)
    v = (v - (v.max(0) + v.min(0)) / 2) / diag
    return v, t


def _rot(elev_deg: float, azim_deg: float) -> np.ndarray:
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    return rx @ rz


def render_views(shape, out_dir: str, size_px: int = 640) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    v, t = _mesh(shape); os.makedirs(out_dir, exist_ok=True); paths = []
    for i, (name, (elev, azim)) in enumerate(VIEWS.items()):
        r = _rot(elev, azim); pv = v @ r.T  # camera looks along -Z of the rotated frame
        tri = pv[t]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nn = np.linalg.norm(n, axis=1, keepdims=True); n = n / np.where(nn == 0, 1, nn)
        front = n[:, 2] > 1e-6  # back-face culling: the camera looks along -Z, so only triangles facing +Z are visible on a closed solid
        tri, n = tri[front], n[front]
        light = np.array([0.3, 0.4, 0.866]); depth = np.clip(tri[:, :, 2].mean(1) + 0.5, 0.0, 1.0)  # 0 = far, 1 = near (unit-diagonal frame)
        shade = 0.25 + 0.5 * np.clip(n @ light, 0.0, 1.0) + 0.25 * depth  # directional light plus a depth cue so raised features stand out in flat views
        order = np.argsort(tri[:, :, 2].mean(1))  # painter: far first
        fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_facecolor("white")
        ax.add_collection(PolyCollection(tri[order][:, :, :2], facecolors=[(0.2 + 0.6 * s, 0.3 + 0.55 * s, 0.55 + 0.4 * s) for s in shade[order]], edgecolors="none", antialiased=False))
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_aspect("equal")
        ax.text(-0.58, 0.55, name, fontsize=11, family="DejaVu Sans")
        p = os.path.join(out_dir, f"view_{i}.png"); fig.savefig(p, dpi=100, facecolor="white"); plt.close(fig); paths.append(p)
    return paths


def judge_rubrics(view_paths: list[str], rubrics: list[dict], brief: str, drawing_paths: list[str] | None = None) -> dict:
    ids = [r["id"] for r in rubrics]; drawing_paths = drawing_paths or []
    text = ("Design brief the candidate received:\n" + brief.strip()[:2500] + "\n\nRequirements to check (from the original designer):\n" + "\n".join(f"{r['id']}: {r['requirement']}" for r in rubrics)
            + (f"\n\nThe first {len(drawing_paths)} image(s) are the original drawing the candidate was given. " if drawing_paths else "\n\n")
            + f"The {'remaining' if drawing_paths else ''} {len(view_paths)} images are labelled renders of the candidate's model. Return verdicts for every id: " + ", ".join(ids) + ".")
    tally = {i: 0 for i in ids}; n = votes(); notes = []
    for _ in range(n):
        res = chat_json(SYSTEM, text, images=drawing_paths + view_paths); ver = res.get("verdicts", {}); notes.append(str(res.get("notes", ""))[:200])
        for i in ids:
            if bool(ver.get(i, False)):
                tally[i] += 1
    verdict = {i: tally[i] * 2 > n for i in ids}
    return {"verdicts": verdict, "votes": tally, "n_votes": n, "notes": notes}


def dim_factor(shape, dims: dict | None) -> tuple[float, dict]:
    """1.0 when the sorted bounding-box extents match the drawing's stated overall dimensions within DIM_TOL_FULL,
    linear to 0.0 at DIM_TOL_ZERO maximum relative error; 1.0 with note when the drawing states no overall dimensions."""
    if not dims or not dims.get("extents_mm"):
        return 1.0, {"dim_factor": 1.0, "dim_note": "no overall dimensions on the drawing"}
    bb = shape.BoundingBox(); got = sorted([bb.xlen, bb.ylen, bb.zlen], reverse=True); exp = list(dims["extents_mm"])[:3]
    # exp is ordered largest..smallest; None marks an extent the drawing does not state (compared positionally, skipped)
    pairs = [(g, float(e)) for g, e in zip(got, exp) if e is not None]
    if not pairs:
        return 1.0, {"dim_factor": 1.0, "dim_note": "no overall dimensions on the drawing"}
    err = max(abs(g - e) / e for g, e in pairs)
    f = 1.0 if err <= DIM_TOL_FULL else 0.0 if err >= DIM_TOL_ZERO else 1.0 - (err - DIM_TOL_FULL) / (DIM_TOL_ZERO - DIM_TOL_FULL)
    return round(f, 4), {"dim_factor": round(f, 4), "extents_mm": [round(x, 2) for x in got], "expected_mm": exp, "max_rel_err": round(err, 4)}


def drawing_images(ref_dir: str, stem: str, work_dir: str) -> list[str]:
    """Decode reference/<stem>/drawing_images.json ({"images": [{"name", "png_b64"}]}) to PNG files for the judge."""
    import base64
    p = os.path.join(ref_dir, stem, "drawing_images.json")
    if not os.path.exists(p):
        return []
    out = []; os.makedirs(work_dir, exist_ok=True)
    for i, im in enumerate(json.load(open(p)).get("images", [])[:2]):
        fp = os.path.join(work_dir, f"drawing_{i}.png")
        with open(fp, "wb") as f:
            f.write(base64.b64decode(im["png_b64"]))
        out.append(fp)
    return out


def score_parts(pred_paths: dict[str, str | None], ref_dir: str, work_dir: str) -> dict:
    per = {}; stems = sorted(d for d in os.listdir(ref_dir) if os.path.isdir(os.path.join(ref_dir, d)))
    for stem in stems:
        rub = json.load(open(os.path.join(ref_dir, stem, "rubrics.json")))["rubrics"]
        td = json.load(open(os.path.join(ref_dir, stem, "task_desc.json"))); brief = td.get("task", "") + "\n" + td.get("description", "")
        dims_p = os.path.join(ref_dir, stem, "dims.json"); dims = json.load(open(dims_p)) if os.path.exists(dims_p) else None
        pp = pred_paths.get(stem)
        if not pp or not os.path.exists(pp):
            per[stem] = {"score": 0.0, "note": "missing output"}; continue
        shape, err = load_solid(pp)
        if shape is None:
            per[stem] = {"score": 0.0, "note": err}; continue
        views = render_views(shape, os.path.join(work_dir, stem)); drawings = drawing_images(ref_dir, stem, os.path.join(work_dir, stem))
        try:
            j = judge_rubrics(views, rub, brief, drawings)
        except Exception as exc:  # judge unavailable: report, do not raise
            per[stem] = {"score": 0.0, "note": f"judge failed: {exc}"}; continue
        sat = sum(1 for v in j["verdicts"].values() if v); df, dinfo = dim_factor(shape, dims)
        per[stem] = {"score": round(sat / len(rub) * df, 4), "rubric_score": round(sat / len(rub), 4), "satisfied": sat, "n_rubrics": len(rub), "verdicts": j["verdicts"], "votes": j["votes"], "notes": j["notes"], "note": "ok", **dinfo}
    return {"score": round(sum(v["score"] for v in per.values()) / len(per), 4) if per else 0.0, "per_part": per}
