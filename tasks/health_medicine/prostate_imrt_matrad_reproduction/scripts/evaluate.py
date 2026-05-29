"""evaluate.py — Score a prostate IMRT submission against the 100-point rubric.

Usage:
  python evaluate.py --submission <dir> --reference <evaluator_reference_dir>

Runs 10 gates (G0 hard + G1..G10 additive). Prints a per-gate breakdown and
writes score.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.sequence import Sequence
from scipy.ndimage import binary_erosion
from skimage import measure  # noqa: F401

try:
    import pymedphys
except ImportError:
    pymedphys = None  # gamma gate will be skipped and report it


DVH_CONSTRAINTS = {
    "PTV_7800_V95": {"gate": "G4", "limit_op": ">=", "limit": 95.0, "dose_gy": 78.0 * 0.95, "metric": "V_pct"},
    "PTV_7800_V107": {"gate": "G4", "limit_op": "<=", "limit": 2.0, "dose_gy": 78.0 * 1.07, "metric": "V_pct"},
    "PTV_7800_Dmax": {"gate": "G4", "limit_op": "<=", "limit": 83.0, "metric": "Dmax_gy"},
    "Rectum_V70":    {"gate": "G5", "limit_op": "<=", "limit": 20.0, "dose_gy": 70.0, "metric": "V_pct"},
    "Rectum_V50":    {"gate": "G5", "limit_op": "<=", "limit": 50.0, "dose_gy": 50.0, "metric": "V_pct"},
    "Bladder_V70":   {"gate": "G5", "limit_op": "<=", "limit": 25.0, "dose_gy": 70.0, "metric": "V_pct"},
    "Bladder_V65":   {"gate": "G5", "limit_op": "<=", "limit": 65.0, "dose_gy": 50.0, "metric": "V_pct"},
    "FemHead_L_V50": {"gate": "G5", "limit_op": "<=", "limit": 5.0, "dose_gy": 50.0, "metric": "V_pct"},
    "FemHead_R_V50": {"gate": "G5", "limit_op": "<=", "limit": 5.0, "dose_gy": 50.0, "metric": "V_pct"},
    "Bowel_Dmax":    {"gate": "G5", "limit_op": "<=", "limit": 50.0, "metric": "Dmax_gy"},
    "PenileBulb_mean": {"gate": "G5", "limit_op": "<=", "limit": 52.5, "metric": "mean_gy"},
}


def load_rtplan(path: Path) -> pydicom.Dataset:
    return pydicom.dcmread(str(path))


def load_rtstruct(path: Path) -> pydicom.Dataset:
    return pydicom.dcmread(str(path))


def load_rtdose(path: Path) -> tuple[np.ndarray, dict]:
    ds = pydicom.dcmread(str(path))
    pixel = ds.pixel_array.astype(np.float64)
    scale = float(ds.DoseGridScaling)
    dose_gy = pixel * scale

    # Get geometry
    origin = [float(x) for x in ds.ImagePositionPatient]
    pixel_spacing = [float(x) for x in ds.PixelSpacing]
    slice_thickness = float(ds.SliceThickness)
    geom = {
        "origin": origin,
        "dx_mm": pixel_spacing[1],
        "dy_mm": pixel_spacing[0],
        "dz_mm": slice_thickness,
        "frame_uid": str(ds.FrameOfReferenceUID),
        "rows": int(ds.Rows),
        "cols": int(ds.Columns),
        "frames": int(ds.NumberOfFrames),
    }
    # Convert dose to (y, x, z) to match our internal convention
    # dose_gy is (frames, rows, cols) = (z, y, x)
    dose_yxz = np.transpose(dose_gy, (1, 2, 0))
    return dose_yxz, geom


def compute_roi_masks(rtstruct: pydicom.Dataset, dose_geom: dict) -> dict[str, np.ndarray]:
    """Rasterize RTSTRUCT contours onto the dose grid. Returns dict mapping ROI name -> bool mask."""
    masks: dict[str, np.ndarray] = {}
    ny, nx, nz = dose_geom["rows"], dose_geom["cols"], dose_geom["frames"]
    dx, dy, dz = dose_geom["dx_mm"], dose_geom["dy_mm"], dose_geom["dz_mm"]
    ox, oy, oz = dose_geom["origin"]

    name_by_num: dict[int, str] = {}
    for ss in rtstruct.StructureSetROISequence:
        name_by_num[int(ss.ROINumber)] = str(ss.ROIName)

    for rc in rtstruct.ROIContourSequence:
        roi_num = int(rc.ReferencedROINumber)
        name = name_by_num.get(roi_num, f"ROI_{roi_num}")
        mask = np.zeros((ny, nx, nz), dtype=bool)
        if not hasattr(rc, "ContourSequence"):
            masks[name] = mask
            continue
        for cont in rc.ContourSequence:
            pts = np.asarray(cont.ContourData, dtype=float).reshape(-1, 3)
            if pts.size == 0:
                continue
            # Convert to dose-grid indices
            z_mm = pts[0, 2]
            k = int(round((z_mm - oz) / dz))
            if k < 0 or k >= nz:
                continue
            xs = (pts[:, 0] - ox) / dx
            ys = (pts[:, 1] - oy) / dy
            # Rasterize the polygon on the (ny, nx) slice
            from skimage.draw import polygon as skpoly
            rr, cc = skpoly(ys, xs, shape=(ny, nx))
            mask[rr, cc, k] = True
        masks[name] = mask
    return masks


def dvh_V_pct(mask: np.ndarray, dose: np.ndarray, dose_threshold_gy: float) -> float:
    if not mask.any():
        return float("nan")
    doses = dose[mask]
    return 100.0 * float((doses >= dose_threshold_gy).mean())


def dvh_Dmax(mask: np.ndarray, dose: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(dose[mask].max())


def dvh_Dmean(mask: np.ndarray, dose: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(dose[mask].mean())


# ==========================================================================
# Gates
# ==========================================================================

def gate_G0_files_and_validity(sub_dir: Path) -> tuple[bool, list[str]]:
    notes: list[str] = []
    required = [
        "RTPLAN.dcm", "RTDOSE.dcm", "RTSTRUCT_corrected.dcm",
        "dvh_metrics.csv", "plan_metrics.json", "beam_summary.csv",
        "figures/axial.png", "figures/sagittal.png", "figures/coronal.png",
        "report.md", "decisions.md",
    ]
    for rel in required:
        p = sub_dir / rel
        if not p.exists() or p.stat().st_size == 0:
            notes.append(f"missing or empty: {rel}")
    if notes:
        return False, notes

    # DICOM parse
    for name in ("RTPLAN.dcm", "RTDOSE.dcm", "RTSTRUCT_corrected.dcm"):
        try:
            pydicom.dcmread(str(sub_dir / name))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"DICOM parse failed for {name}: {exc}")
    if notes:
        return False, notes

    # FoR UID consistency
    try:
        rtplan = pydicom.dcmread(str(sub_dir / "RTPLAN.dcm"))
        rtdose = pydicom.dcmread(str(sub_dir / "RTDOSE.dcm"))
        rtstruct = pydicom.dcmread(str(sub_dir / "RTSTRUCT_corrected.dcm"))
        for_plan = str(rtplan.FrameOfReferenceUID)
        for_dose = str(rtdose.FrameOfReferenceUID)
        for_struct = str(rtstruct.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID)
        if not (for_plan == for_dose == for_struct):
            notes.append(f"FrameOfReferenceUID mismatch: plan={for_plan[:32]} dose={for_dose[:32]} struct={for_struct[:32]}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"FoR UID check failed: {exc}")

    return len(notes) == 0, notes


def gate_G1_beam_geometry(rtplan: pydicom.Dataset, expected_angles: list[float]) -> tuple[int, list[str]]:
    notes: list[str] = []
    beams = list(rtplan.BeamSequence)
    if len(beams) != len(expected_angles):
        notes.append(f"beam count {len(beams)}, expected {len(expected_angles)}")
        return 0, notes
    actual_angles: list[float] = []
    for b in beams:
        cp0 = b.ControlPointSequence[0]
        actual_angles.append(float(cp0.GantryAngle))
        energy = float(cp0.NominalBeamEnergy)
        if abs(energy - 6.0) > 0.01:
            notes.append(f"beam {int(b.BeamNumber)} energy {energy} MV, expected 6")
    # match within ±0.5° (order-independent: sort both)
    actual_sorted = sorted(actual_angles)
    expected_sorted = sorted(expected_angles)
    for a, e in zip(actual_sorted, expected_sorted):
        if abs(a - e) > 0.5:
            notes.append(f"gantry angle {a} vs expected {e}")
    score = 6 if not notes else max(0, 6 - len(notes))
    return score, notes


def gate_G2_deliverability(rtplan: pydicom.Dataset, ref_total_mu: float, ref_per_beam_mu: list[float]) -> tuple[int, list[str]]:
    notes: list[str] = []
    fg = rtplan.FractionGroupSequence[0]
    per_beam_mu = [float(b.BeamMeterset) for b in fg.ReferencedBeamSequence]
    total_mu = sum(per_beam_mu)

    if total_mu <= 0 or any(m <= 0 for m in per_beam_mu):
        notes.append(f"non-positive MU detected: total={total_mu}, per-beam={per_beam_mu}")
        return 0, notes

    # Total within ±5% of reference envelope
    total_ok = abs(total_mu - ref_total_mu) / max(ref_total_mu, 1.0) <= 0.05
    # Per-beam share within ±10% of reference beam-share
    ref_total = sum(ref_per_beam_mu)
    ref_shares = [m / ref_total for m in ref_per_beam_mu]
    actual_shares = [m / total_mu for m in per_beam_mu]
    per_beam_ok = all(abs(a - r) <= 0.10 for a, r in zip(sorted(actual_shares), sorted(ref_shares)))

    score = 0
    if total_ok:
        score += 3
    else:
        notes.append(f"total MU {total_mu:.1f} outside ±5% of ref {ref_total_mu:.1f}")
    if per_beam_ok:
        score += 3
    else:
        notes.append(f"per-beam MU shares outside ±10% of ref envelope")
    return score, notes


def gate_G3_structure_defects(
    rtstruct: pydicom.Dataset, gold_rtstruct: pydicom.Dataset
) -> tuple[int, list[str]]:
    notes: list[str] = []
    score = 0

    # Sub-check 1 (4pts): Rectum ROI observation type is "ORGAN" (was cleared to empty in corrupted)
    rectum_type = None
    name_by_num = {int(ss.ROINumber): str(ss.ROIName) for ss in rtstruct.StructureSetROISequence}
    for obs in rtstruct.RTROIObservationsSequence:
        n = name_by_num.get(int(obs.ReferencedROINumber), "")
        if n == "Rectum":
            rectum_type = str(getattr(obs, "RTROIInterpretedType", "")).strip().upper()
    if rectum_type == "ORGAN":
        score += 4
    else:
        notes.append(f"Rectum RTROIInterpretedType is '{rectum_type}', expected ORGAN")

    # Sub-check 2 (6pts): PTV_7800 exists and geometry matches gold PTV_68 within Hausdorff 5mm
    # (relaxed from 1mm since discrete voxel rasterization introduces quantization)
    # We compute centroid + volume match as a proxy.
    gold_ptv_centroid, gold_ptv_volume = _roi_centroid_volume(gold_rtstruct, "PTV_68")
    sub_ptv_centroid, sub_ptv_volume = _roi_centroid_volume(rtstruct, "PTV_7800")
    if sub_ptv_centroid is None:
        notes.append("PTV_7800 not present in submitted RTSTRUCT")
    else:
        cdist = math.sqrt(sum((a - b) ** 2 for a, b in zip(gold_ptv_centroid, sub_ptv_centroid)))
        vol_ratio = sub_ptv_volume / max(gold_ptv_volume, 1.0)
        if cdist <= 8.0 and 0.85 <= vol_ratio <= 1.4:
            score += 6
        elif cdist <= 15.0 and 0.7 <= vol_ratio <= 1.8:
            score += 3
            notes.append(f"PTV_7800 geometry loose: centroid dist={cdist:.1f}mm, vol ratio={vol_ratio:.2f}")
        else:
            notes.append(f"PTV_7800 geometry BAD: centroid dist={cdist:.1f}mm, vol ratio={vol_ratio:.2f}")

    # Sub-check 3 (6pts): External contour covers ≥99% of body
    # Proxy: submitted External contour has ≥ (0.9 * volume of gold BODY)
    gold_body_centroid, gold_body_volume = _roi_centroid_volume(gold_rtstruct, "BODY")
    sub_ext_centroid, sub_ext_volume = _roi_centroid_volume(rtstruct, "External")
    if sub_ext_centroid is None:
        # Try "External_partial" fallback (not fixed)
        notes.append("External contour not present (name 'External')")
    else:
        ratio = sub_ext_volume / max(gold_body_volume, 1.0)
        if ratio >= 0.95:
            score += 6
        elif ratio >= 0.80:
            score += 3
            notes.append(f"External coverage partial: {ratio*100:.1f}% of gold BODY volume")
        else:
            notes.append(f"External coverage LOW: {ratio*100:.1f}% of gold BODY volume")

    return min(score, 16), notes


def _roi_centroid_volume(rtstruct: pydicom.Dataset, roi_name: str) -> tuple[tuple[float, float, float] | None, float]:
    """Approximate centroid (mm) and voxel-equivalent volume (mm^3) from contour points."""
    name_to_num = {str(ss.ROIName): int(ss.ROINumber) for ss in rtstruct.StructureSetROISequence}
    if roi_name not in name_to_num:
        return None, 0.0
    target_num = name_to_num[roi_name]
    pts_all: list[np.ndarray] = []
    for rc in rtstruct.ROIContourSequence:
        if int(rc.ReferencedROINumber) != target_num:
            continue
        if not hasattr(rc, "ContourSequence"):
            continue
        for cont in rc.ContourSequence:
            pts = np.asarray(cont.ContourData, dtype=float).reshape(-1, 3)
            if pts.size:
                pts_all.append(pts)
    if not pts_all:
        return None, 0.0
    stacked = np.vstack(pts_all)
    centroid = tuple(stacked.mean(axis=0))
    # Rough volume: count of unique z-slices * mean polygon area across slices
    # Use convex hull area per slice for simplicity
    from scipy.spatial import ConvexHull
    vol = 0.0
    # Group by z
    z_vals = np.round(stacked[:, 2] / 0.1).astype(int)
    for zk in np.unique(z_vals):
        slab = stacked[z_vals == zk][:, :2]
        if len(slab) < 3:
            continue
        try:
            hull = ConvexHull(slab)
            vol += hull.volume  # for 2D, .volume is the polygon area
        except Exception:
            continue
    # Multiply by nominal slice thickness 3mm
    vol *= 3.0
    return centroid, vol


def gate_G4_ptv_coverage(sub_dose: np.ndarray, sub_masks: dict[str, np.ndarray]) -> tuple[int, list[str], dict[str, float]]:
    notes: list[str] = []
    metrics: dict[str, float] = {}
    # Find PTV mask — accept any of these names
    ptv_mask = None
    for n in ("PTV_7800", "PTV_68", "PTV_Prostate_7800"):
        if n in sub_masks and sub_masks[n].any():
            ptv_mask = sub_masks[n]
            metrics["ptv_name_used"] = n  # type: ignore
            break
    if ptv_mask is None:
        notes.append("no PTV mask found")
        return 0, notes, metrics

    v95 = dvh_V_pct(ptv_mask, sub_dose, 78.0 * 0.95)
    v107 = dvh_V_pct(ptv_mask, sub_dose, 78.0 * 1.07)
    dmax = dvh_Dmax(ptv_mask, sub_dose)
    metrics.update({"PTV_V95": v95, "PTV_V107": v107, "PTV_Dmax": dmax})

    score = 0
    if v95 >= 95.0:
        score += 5
    elif v95 >= 90.0:
        score += 2; notes.append(f"PTV V95 {v95:.1f}% partial")
    else:
        notes.append(f"PTV V95 {v95:.1f}% FAIL")
    if v107 <= 2.0:
        score += 5
    elif v107 <= 5.0:
        score += 2; notes.append(f"PTV V107 {v107:.1f}% partial")
    else:
        notes.append(f"PTV V107 {v107:.1f}% FAIL")
    if dmax <= 83.0:
        score += 5
    elif dmax <= 86.0:
        score += 2; notes.append(f"PTV Dmax {dmax:.1f} Gy partial")
    else:
        notes.append(f"PTV Dmax {dmax:.1f} Gy FAIL")

    return score, notes, metrics


def gate_G5_oar_sparing(sub_dose: np.ndarray, sub_masks: dict[str, np.ndarray]) -> tuple[int, list[str], dict[str, float]]:
    """26-point OAR gate: 7 constraints, weights summing to 26."""
    notes: list[str] = []
    metrics: dict[str, float] = {}

    gates = [
        ("Rectum",  "V70", 70.0, "<=", 20.0, 4),
        ("Rectum",  "V50", 50.0, "<=", 50.0, 4),
        ("Bladder", "V70", 70.0, "<=", 25.0, 3),
        ("Bladder", "V50", 50.0, "<=", 65.0, 3),
        ("FemHead_L", "V50", 50.0, "<=", 5.0, 3),
        ("FemHead_R", "V50", 50.0, "<=", 5.0, 3),
        ("Bowel",   "Dmax", None, "<=", 50.0, 3),
        ("PenileBulb", "mean", None, "<=", 52.5, 3),
    ]
    # Total weights: 4+4+3+3+3+3+3+3 = 26 ✓

    score = 0
    for roi, kind, dose_th, op, limit, weight in gates:
        # match ROI name with fuzzy aliases
        mask = None
        for alias in (roi, roi.replace("_", " "), roi.replace("_", "")):
            if alias in sub_masks and sub_masks[alias].any():
                mask = sub_masks[alias]; break
        # Additional aliases matching matRad PROSTATE phantom names
        aliases = {
            "Rectum": ["Rectum"],
            "Bladder": ["Bladder"],
            "FemHead_L": ["Lt femoral head", "FemHead_L", "Left femoral head"],
            "FemHead_R": ["Rt femoral head", "FemHead_R", "Right femoral head"],
            "Bowel": ["BowelBag", "Bowel_bag", "Bowel"],
            "PenileBulb": ["Penile_bulb", "PenileBulb", "Penile bulb"],
        }
        if mask is None:
            for a in aliases.get(roi, []):
                if a in sub_masks and sub_masks[a].any():
                    mask = sub_masks[a]; break
        if mask is None:
            notes.append(f"{roi}: ROI not found, skipping ({weight}pts lost)")
            continue
        if kind == "V70" or kind == "V50":
            val = dvh_V_pct(mask, sub_dose, dose_th)
        elif kind == "Dmax":
            val = dvh_Dmax(mask, sub_dose)
        elif kind == "mean":
            val = dvh_Dmean(mask, sub_dose)
        else:
            continue
        metrics[f"{roi}_{kind}"] = val
        ok = (val <= limit) if op == "<=" else (val >= limit)
        if ok:
            score += weight
        elif val <= limit * 1.25:
            score += weight // 2
            notes.append(f"{roi} {kind} = {val:.2f} (limit {limit}, partial)")
        else:
            notes.append(f"{roi} {kind} = {val:.2f} (limit {limit}, FAIL)")
    return score, notes, metrics


def gate_G6_gamma(sub_dose: np.ndarray, ref_dose: np.ndarray, geom: dict, external_mask: np.ndarray | None) -> tuple[int, list[str], float]:
    notes: list[str] = []
    if pymedphys is None:
        notes.append("pymedphys not installed — gamma skipped")
        return 0, notes, 0.0
    if ref_dose.shape != sub_dose.shape:
        notes.append(f"dose shape mismatch ref={ref_dose.shape} sub={sub_dose.shape}")
        return 0, notes, 0.0
    # Build axes
    ny, nx, nz = sub_dose.shape
    z_axis = np.arange(nz) * geom["dz_mm"]
    y_axis = np.arange(ny) * geom["dy_mm"]
    x_axis = np.arange(nx) * geom["dx_mm"]
    axes_ref = (z_axis, y_axis, x_axis)
    axes_sub = (z_axis, y_axis, x_axis)
    # pymedphys.gamma expects (z, y, x) ordering with dose as float
    ref_zyx = np.transpose(ref_dose, (2, 0, 1)).astype(np.float64)
    sub_zyx = np.transpose(sub_dose, (2, 0, 1)).astype(np.float64)
    gamma = pymedphys.gamma(
        axes_ref, ref_zyx,
        axes_sub, sub_zyx,
        dose_percent_threshold=3.0,
        distance_mm_threshold=3.0,
        lower_percent_dose_cutoff=10.0,
        interp_fraction=10,
        max_gamma=2.0,
        local_gamma=False,
    )
    # gamma is an array same shape as ref; NaN where below threshold
    valid = np.isfinite(gamma)
    if external_mask is not None:
        ext_zyx = np.transpose(external_mask, (2, 0, 1))
        valid &= ext_zyx
    if not valid.any():
        notes.append("no valid voxels for gamma")
        return 0, notes, 0.0
    pass_rate = 100.0 * float((gamma[valid] <= 1.0).mean())
    if pass_rate >= 94.0:
        return 12, notes, pass_rate
    if pass_rate >= 90.0:
        return 6, [f"gamma pass {pass_rate:.1f}%"], pass_rate
    return 0, [f"gamma pass {pass_rate:.1f}% FAIL"], pass_rate


def gate_G7_dvh_csv_honesty(submitted_csv: Path, computed: dict[str, float]) -> tuple[int, list[str]]:
    notes: list[str] = []
    try:
        import csv
        with submitted_csv.open() as f:
            reader = csv.DictReader(f)
            declared: dict[str, float] = {}
            for row in reader:
                key = f"{row['structure'].strip()}_{row['metric_type'].strip()}"
                declared[key] = float(row["metric_value"])
    except Exception as exc:  # noqa: BLE001
        notes.append(f"CSV parse failed: {exc}")
        return 0, notes

    mismatches = 0
    for key, val in computed.items():
        d = declared.get(key)
        if d is None:
            continue
        if abs(d - val) > 0.5:
            mismatches += 1
            notes.append(f"{key}: CSV={d:.2f} vs recomputed={val:.2f}")
    if mismatches == 0 and declared:
        return 4, notes
    if mismatches <= 2:
        return 2, notes
    return 0, notes


def gate_G8_report_structure(sub_dir: Path) -> tuple[int, list[str]]:
    notes: list[str] = []
    score = 0
    # report.md required keywords
    try:
        rpt = (sub_dir / "report.md").read_text()
        required_sections = ["## Beam summary", "## DVH", "## Constraint compliance"]
        missing = [s for s in required_sections if s not in rpt]
        if not missing:
            score += 2
        else:
            notes.append(f"report.md missing sections: {missing}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"report.md read failed: {exc}")

    # PNGs 800+
    try:
        from PIL import Image
        for name in ("axial.png", "sagittal.png", "coronal.png"):
            im = Image.open(sub_dir / "figures" / name)
            if min(im.size) < 800:
                notes.append(f"{name} too small {im.size}")
                break
        else:
            score += 1
    except Exception as exc:  # noqa: BLE001
        notes.append(f"figures read failed: {exc}")

    # beam_summary.csv has 7 rows
    try:
        import csv
        with (sub_dir / "beam_summary.csv").open() as f:
            rows = list(csv.reader(f))
        # Data rows = rows - header
        n_data = len(rows) - 1
        if n_data == 7:
            score += 1
        else:
            notes.append(f"beam_summary.csv has {n_data} data rows, expected 7")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"beam_summary.csv read failed: {exc}")

    return score, notes


def gate_G9_decisions(sub_dir: Path) -> tuple[int, list[str]]:
    notes: list[str] = []
    score = 0
    try:
        text = (sub_dir / "decisions.md").read_text()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"decisions.md read failed: {exc}")
        return 0, notes
    words = text.split()
    if len(words) <= 500:
        score += 1
    else:
        notes.append(f"decisions.md has {len(words)} words (>500)")
    # Look for defect mentions
    for kw in ("Rectum", "PTV_placeholder", "External"):
        if kw.lower() not in text.lower():
            notes.append(f"decisions.md missing reference to {kw}")
    # Numeric citations
    import re
    nums = re.findall(r"\d+\.?\d*", text)
    if len(nums) >= 4:
        score += 1
    return score, notes


def gate_G10_replay(
    sub_dir: Path,
    rtstruct: pydicom.Dataset,
    sub_dose: np.ndarray,
    sub_dose_geom: dict,
) -> tuple[int, list[str]]:
    notes: list[str] = []
    # Prefer clean replay_state.mat (numeric-only), fall back to full workspace
    ws_path = sub_dir / "replay_state.mat"
    if not ws_path.exists():
        ws_path = sub_dir / "matRad_workspace.mat"
    if not ws_path.exists():
        notes.append("replay_state.mat and matRad_workspace.mat both missing")
        return 0, notes
    # Run the replay Octave script
    replay_script = Path(__file__).with_name("replay.m")
    if not replay_script.exists():
        notes.append(f"replay.m script not found at {replay_script}")
        return 0, notes
    out_dir = sub_dir / "_replay"
    out_dir.mkdir(exist_ok=True)
    octave_bin = os.environ.get("OCTAVE_BIN") or shutil.which("octave")
    if not octave_bin:
        notes.append("Octave binary not found on PATH; set OCTAVE_BIN to run G10 replay")
        return 0, notes
    cmd = [
        octave_bin, "--no-gui", "--eval",
        f"addpath('{replay_script.parent}'); replay('{ws_path.resolve()}', '{out_dir.resolve()}')",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        notes.append("replay timed out")
        return 0, notes
    if proc.returncode != 0:
        notes.append(f"replay failed: {proc.stderr[:500]}")
        return 0, notes
    # Load the replayed dose — Octave can't write .npy, so replay.m writes raw doubles + shape
    bin_path = out_dir / "replayed_dose.bin"
    shape_path = out_dir / "replayed_shape.txt"
    if not bin_path.exists() or not shape_path.exists():
        notes.append("replay produced no dose")
        return 0, notes
    shape = tuple(int(x) for x in shape_path.read_text().strip().split())
    replayed = np.fromfile(str(bin_path), dtype=np.float64).reshape(shape, order="F")
    if replayed.shape != sub_dose.shape:
        notes.append(f"replay shape mismatch {replayed.shape} vs {sub_dose.shape}")
        return 0, notes
    # Compare replayed dose to submitted dose: they should be very close
    # (replay is deterministic — same dij, same weights, same scaling)
    max_sub = float(sub_dose.max())
    max_rep = float(replayed.max())
    max_diff = abs(max_sub - max_rep)
    # Mean voxel-wise ratio
    mean_sub = float(sub_dose.mean())
    mean_rep = float(replayed.mean())
    mean_diff = abs(mean_sub - mean_rep)
    # Primary gate: max dose should match within 2 Gy; mean within 0.5 Gy
    if max_diff <= 2.0 and mean_diff <= 0.5:
        return 10, notes + [f"replay max diff={max_diff:.2f}, mean diff={mean_diff:.3f}"]
    if max_diff <= 5.0 and mean_diff <= 1.0:
        return 5, notes + [f"replay max diff={max_diff:.2f}, mean diff={mean_diff:.3f} (partial)"]
    return 0, notes + [f"replay drift FAIL: max diff={max_diff:.2f}, mean diff={mean_diff:.3f}"]


# ==========================================================================
# Top-level runner
# ==========================================================================

def run(sub_dir: Path, ref_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"gates": {}, "notes": []}

    # G0 — still a hard gate for pass/fail, but we no longer short-circuit.
    # All subsequent gates run regardless so score.json always has full diagnostics.
    g0_ok, g0_notes = gate_G0_files_and_validity(sub_dir)
    report["gates"]["G0"] = {"ok": g0_ok, "notes": g0_notes}

    # Load submission DICOMs — each wrapped so a single bad file doesn't
    # crash the entire evaluator.  Gates that need a missing object score 0.
    rtplan = sub_rtstruct = sub_dose = sub_geom = None
    for name, loader, target in [
        ("RTPLAN.dcm", load_rtplan, "rtplan"),
        ("RTSTRUCT_corrected.dcm", load_rtstruct, "sub_rtstruct"),
        ("RTDOSE.dcm", load_rtdose, "sub_dose"),
    ]:
        try:
            result = loader(sub_dir / name)
            if target == "rtplan":
                rtplan = result
            elif target == "sub_rtstruct":
                sub_rtstruct = result
            else:
                sub_dose, sub_geom = result
        except Exception as exc:
            report["notes"].append(f"{name} load failed: {exc}")

    # Reference data
    gold_rtstruct = ref_dose = calibration = None
    ref_total_mu = ref_per_beam_mu = expected_angles = None
    try:
        gold_rtstruct = load_rtstruct(ref_dir / "RTSTRUCT_gold.dcm")
        ref_dose, _ = load_rtdose(ref_dir / "RTDOSE_reference.dcm")
        with (ref_dir / "reference_calibration.json").open() as f:
            calibration = json.load(f)
        ref_total_mu = calibration["total_mu_mean"]
        ref_per_beam_mu = calibration["per_beam_mu_mean"]
        expected_angles = calibration["beam_angles"]
    except Exception as exc:
        report["notes"].append(f"reference data load failed: {exc}")

    # G1
    if rtplan is not None and expected_angles is not None:
        g1_score, g1_notes = gate_G1_beam_geometry(rtplan, expected_angles)
    else:
        g1_score, g1_notes = 0, ["skipped: RTPLAN or reference calibration not loaded"]
    report["gates"]["G1"] = {"score": g1_score, "max": 6, "notes": g1_notes}

    # G2
    if rtplan is not None and ref_total_mu is not None:
        g2_score, g2_notes = gate_G2_deliverability(rtplan, ref_total_mu, ref_per_beam_mu)
    else:
        g2_score, g2_notes = 0, ["skipped: RTPLAN or reference calibration not loaded"]
    report["gates"]["G2"] = {"score": g2_score, "max": 6, "notes": g2_notes}

    # G3
    if sub_rtstruct is not None and gold_rtstruct is not None:
        g3_score, g3_notes = gate_G3_structure_defects(sub_rtstruct, gold_rtstruct)
    else:
        g3_score, g3_notes = 0, ["skipped: RTSTRUCT or gold RTSTRUCT not loaded"]
    report["gates"]["G3"] = {"score": g3_score, "max": 16, "notes": g3_notes}

    # Build masks on dose grid
    sub_masks: dict[str, np.ndarray] = {}
    g4_metrics: dict[str, float] = {}
    if sub_rtstruct is not None and sub_geom is not None:
        sub_masks = compute_roi_masks(sub_rtstruct, sub_geom)

    # G4
    if sub_dose is not None and sub_masks:
        g4_score, g4_notes, g4_metrics = gate_G4_ptv_coverage(sub_dose, sub_masks)
    else:
        g4_score, g4_notes = 0, ["skipped: RTDOSE or RTSTRUCT not loaded"]
    report["gates"]["G4"] = {"score": g4_score, "max": 14, "notes": g4_notes, "metrics": g4_metrics}

    # G5
    g5_metrics: dict[str, float] = {}
    if sub_dose is not None and sub_masks:
        g5_score, g5_notes, g5_metrics = gate_G5_oar_sparing(sub_dose, sub_masks)
    else:
        g5_score, g5_notes = 0, ["skipped: RTDOSE or RTSTRUCT not loaded"]
    report["gates"]["G5"] = {"score": g5_score, "max": 26, "notes": g5_notes, "metrics": g5_metrics}

    # G6 gamma
    if sub_dose is not None and ref_dose is not None and sub_geom is not None:
        external_mask = sub_masks.get("External")
        if external_mask is None or not external_mask.any():
            external_mask = sub_masks.get("BODY")
        g6_score, g6_notes, g6_pass = gate_G6_gamma(sub_dose, ref_dose, sub_geom, external_mask)
    else:
        g6_score, g6_notes, g6_pass = 0, ["skipped: RTDOSE or reference dose not loaded"], 0.0
    report["gates"]["G6"] = {"score": g6_score, "max": 12, "notes": g6_notes, "gamma_pass_pct": g6_pass}

    # G7 CSV honesty
    computed = {**{f"PTV_{k}": v for k, v in g4_metrics.items()}, **g5_metrics}
    g7_score, g7_notes = gate_G7_dvh_csv_honesty(sub_dir / "dvh_metrics.csv", computed)
    report["gates"]["G7"] = {"score": g7_score, "max": 4, "notes": g7_notes}

    # G8 report structure
    g8_score, g8_notes = gate_G8_report_structure(sub_dir)
    report["gates"]["G8"] = {"score": g8_score, "max": 4, "notes": g8_notes}

    # G9 decisions.md
    g9_score, g9_notes = gate_G9_decisions(sub_dir)
    report["gates"]["G9"] = {"score": g9_score, "max": 2, "notes": g9_notes}

    # G10 replay
    if sub_rtstruct is not None and sub_dose is not None and sub_geom is not None:
        g10_score, g10_notes = gate_G10_replay(sub_dir, sub_rtstruct, sub_dose, sub_geom)
    else:
        g10_score, g10_notes = 0, ["skipped: RTDOSE or RTSTRUCT not loaded"]
    report["gates"]["G10"] = {"score": g10_score, "max": 10, "notes": g10_notes}

    total = g1_score + g2_score + g3_score + g4_score + g5_score + g6_score + g7_score + g8_score + g9_score + g10_score
    report["total_score_before_g0"] = total
    if g0_ok:
        report["total_score"] = total
    else:
        report["total_score"] = 0
    report["pass"] = g0_ok and total >= 70

    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report = run(args.submission, args.reference)

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))

    print("\n=== Gate summary ===")
    for g, info in report["gates"].items():
        if "score" in info:
            print(f"  {g}: {info['score']}/{info['max']}")
        else:
            print(f"  {g}: {'OK' if info['ok'] else 'FAIL'}")
    print(f"  TOTAL: {report.get('total_score', 0)}/100  {'PASS' if report.get('pass') else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
