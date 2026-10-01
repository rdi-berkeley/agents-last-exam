"""Grader for engineering/autocad_drafting_from_reference.

The agent delivers output/<name>.dxf (any DXF version ezdxf reads). The hidden reference is the expert's DWG converted
to DXF with LibreDWG. Both drawings are rendered to a binary line raster over the reference's model-space extents
(the prediction is fitted to the same normalised frame: translated to its own extents' centre and scaled so its
larger extent matches the reference), then compared with a tolerant line-overlap score:
  precision = fraction of predicted line pixels within `tol` pixels of a reference line pixel,
  recall    = fraction of reference line pixels within `tol` pixels of a predicted line pixel,
  score     = harmonic mean (F1) of precision and recall.
Dimension and leader entities are left out of the raster: where a drafter places a dimension line is a layout choice,
so they are scored by count instead: dim_ratio = min(1, predicted DIMENSION count / reference DIMENSION count).
  score = 0.8 * geometry F1 + 0.2 * dim_ratio          (dim_ratio = 1 when the reference has no dimensions)
Text entities are ignored entirely (fonts vary).
Hard gates (score 0): missing/unreadable DXF, no drawable entities.
"""
from __future__ import annotations

import numpy as np

RASTER = 1024
TOL_PX = 5
W_GEOM, W_DIM = 0.8, 0.2
DRAWABLE = {"LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "ELLIPSE", "SPLINE", "HATCH", "SOLID", "INSERT", "3DFACE"}
DIMENSIONAL = {"DIMENSION", "LEADER", "MLEADER", "ARC_DIMENSION"}


def _load(path: str):
    import ezdxf
    try:
        doc = ezdxf.readfile(path)
    except Exception as exc:
        return None, f"unreadable DXF: {exc}"
    msp = doc.modelspace()
    ents = [e for e in msp if e.dxftype() in DRAWABLE or e.dxftype() in ("TEXT", "MTEXT", "DIMENSION")]
    if not any(e.dxftype() in DRAWABLE for e in ents):
        return None, "no drawable entities in model space"
    return doc, None


def _polylines(doc, flat_tol=0.02):
    """Flatten every drawable model-space entity (blocks and dimensions decomposed) into point lists."""
    from ezdxf import path as ezpath
    from ezdxf.disassemble import recursive_decompose
    out = []
    for e in recursive_decompose(top for top in doc.modelspace() if top.dxftype() not in DIMENSIONAL):
        if e.dxftype() not in DRAWABLE:
            continue
        try:
            p = ezpath.make_path(e)
        except Exception:
            continue
        for sub in p.sub_paths():
            pts = [(v.x, v.y) for v in sub.flattening(distance=flat_tol, segments=8)]
            if len(pts) >= 2:
                out.append(pts)
    return out


def _extents(polys):
    xs = [x for pl in polys for x, _ in pl]; ys = [y for pl in polys for _, y in pl]
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _raster(polys, frame, size=RASTER):
    """Draw the polylines as 1-px lines into a bool array (PIL, deterministic, no fonts involved)."""
    from PIL import Image, ImageDraw
    xmin, ymin, xmax, ymax = frame; sx = (size - 1) / (xmax - xmin); sy = (size - 1) / (ymax - ymin)
    img = Image.new("L", (size, size), 0); d = ImageDraw.Draw(img)
    for pl in polys:
        pts = [((x - xmin) * sx, (size - 1) - (y - ymin) * sy) for x, y in pl]
        d.line(pts, fill=255, width=1)
    return np.asarray(img) > 0


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, iterations=r)


def _square_frame(ext, pad=0.04):
    xmin, ymin, xmax, ymax = ext; cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2; half = max(xmax - xmin, ymax - ymin) / 2 * (1 + pad) or 1.0
    return cx - half, cy - half, cx + half, cy + half


def score_drawing(pred_path: str, ref_path: str) -> dict:
    ref, err = _load(ref_path)
    if ref is None:
        raise RuntimeError(f"reference unusable: {err}")
    pred, err = _load(pred_path)
    if pred is None:
        return {"score": 0.0, "precision": 0.0, "recall": 0.0, "note": err}
    rpolys, ppolys = _polylines(ref), _polylines(pred)
    rext, pext = _extents(rpolys), _extents(ppolys)
    if rext is None or pext is None:
        return {"score": 0.0, "precision": 0.0, "recall": 0.0, "note": "empty extents"}
    rmask = _raster(rpolys, _square_frame(rext)); pmask = _raster(ppolys, _square_frame(pext))
    rd, pd = _dilate(rmask, TOL_PX), _dilate(pmask, TOL_PX)
    precision = float((pmask & rd).sum() / max(pmask.sum(), 1)); recall = float((rmask & pd).sum() / max(rmask.sum(), 1))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    def ndim(doc): return sum(1 for e in doc.modelspace() if e.dxftype() in DIMENSIONAL)
    dim_ratio = min(1.0, ndim(pred) / ndim(ref)) if ndim(ref) else 1.0
    score = W_GEOM * f1 + W_DIM * dim_ratio
    return {"score": round(max(0.0, min(1.0, score)), 4), "geom_f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4), "dim_ratio": round(dim_ratio, 4), "n_dim": [ndim(pred), ndim(ref)], "note": "ok"}


def score_parts(pred_paths: dict[str, str | None], ref_paths: dict[str, str]) -> dict:
    per = {}
    for name, rp in ref_paths.items():
        pp = pred_paths.get(name)
        per[name] = {"score": 0.0, "note": "missing"} if not pp else score_drawing(pp, rp)
    return {"score": round(sum(v["score"] for v in per.values()) / len(per), 4) if per else 0.0, "per_part": per}
