"""Detect floor-slab elevations from an OBJ mesh.

Returns a sorted list of Z values where horizontal faces concentrate. Uses
numpy only — no scipy or scikit dependencies, so it imports cleanly inside
Blender's bundled Python.
"""

from __future__ import annotations

import numpy as np


def parse_obj_mesh(obj_path: str):
    """Parse vertices and (triangulated) face indices from an OBJ file.

    Returns:
      verts: (Nv, 3) float array.
      faces: (Nf, 3) int array (0-based indices, fan-triangulated for n>3).
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                p = line.split()[1:]
                idx = [int(t.split("/")[0]) - 1 for t in p]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def detect_floors_with_weights(
    obj_path: str,
    bin_size_mm: float = 100.0,
    horizontal_nz_thresh: float = 0.9,
    min_peak_ratio: float = 0.05,
    merge_radius_mm: float = 500.0,
    up_axis: str = "Z",
) -> list[tuple[float, float]]:
    """Same as detect_floors but returns (z, weight) tuples sorted by Z.

    The weight reflects horizontal-face area mass at that peak — used by
    callers to find the dominant slab (typically the Hall ceiling).
    """
    floors_with_w = _detect_floors_impl(
        obj_path, bin_size_mm, horizontal_nz_thresh, min_peak_ratio, merge_radius_mm,
        up_axis=up_axis,
    )
    return floors_with_w


def detect_floors(
    obj_path: str,
    bin_size_mm: float = 100.0,
    horizontal_nz_thresh: float = 0.9,
    min_peak_ratio: float = 0.05,
    merge_radius_mm: float = 500.0,
) -> list[float]:
    """Find Z coordinates of floor-like horizontal slabs.

    Algorithm:
      1. Triangulate OBJ; compute per-face normal + area + centroid Z.
      2. Filter to nearly-horizontal faces (|nz| > horizontal_nz_thresh).
      3. Build weighted Z histogram (weights = face area), smooth with a 5-bin
         moving average.
      4. Detect local maxima where smoothed mass > min_peak_ratio * peak_max.
      5. Merge peaks within merge_radius_mm of each other (keep stronger).

    Returns sorted ascending list of Z elevations (same units as the OBJ).
    """
    return [z for z, _ in _detect_floors_impl(
        obj_path, bin_size_mm, horizontal_nz_thresh, min_peak_ratio, merge_radius_mm
    )]


def _detect_floors_impl(
    obj_path: str,
    bin_size_mm: float,
    horizontal_nz_thresh: float,
    min_peak_ratio: float,
    merge_radius_mm: float,
    up_axis: str = "Z",
) -> list[tuple[float, float]]:
    verts, faces = parse_obj_mesh(obj_path)
    if len(faces) == 0:
        return []
    # Column of the OBJ vertex array that corresponds to the source's up axis.
    up_col = {"X": 0, "Y": 1, "Z": 2}[up_axis.upper()]

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area2 = np.linalg.norm(cross, axis=1)
    valid = area2 > 1e-6
    nz = np.zeros(len(faces))
    nz[valid] = cross[valid, up_col] / area2[valid]
    areas = area2 / 2.0
    centroids_z = (v0[:, up_col] + v1[:, up_col] + v2[:, up_col]) / 3.0

    horizontal_mask = np.abs(nz) > horizontal_nz_thresh
    hz = centroids_z[horizontal_mask]
    ha = areas[horizontal_mask]
    if len(hz) == 0:
        return []

    z_min = float(verts[:, up_col].min())
    z_max = float(verts[:, up_col].max())
    if z_max - z_min < bin_size_mm:
        return [(z_min, 1.0)]
    n_bins = max(50, int((z_max - z_min) / bin_size_mm))
    hist, edges = np.histogram(hz, bins=n_bins, range=(z_min, z_max), weights=ha)

    k = 5
    kernel = np.ones(k) / k
    smoothed = np.convolve(hist, kernel, mode="same")

    threshold = float(smoothed.max()) * min_peak_ratio
    raw_peaks: list[tuple[float, float]] = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= threshold and smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            z = (edges[i] + edges[i + 1]) / 2.0
            raw_peaks.append((float(z), float(smoothed[i])))

    raw_peaks.sort(key=lambda p: p[0])
    merged: list[tuple[float, float]] = []
    for z, w in raw_peaks:
        if merged and z - merged[-1][0] < merge_radius_mm:
            if w > merged[-1][1]:
                merged[-1] = (z, w)
        else:
            merged.append((z, w))
    return merged


def split_hall_tower(
    floors_with_w: list[tuple[float, float]],
) -> tuple[list[float], list[float], float]:
    """Split detected floors into (hall_mezzanines, tower_floors, hall_ceiling_z).

    Strategy: the dominant horizontal slab (highest weight) is treated as the
    Hall ceiling — for Halle-style buildings the workshop roof has by far the
    largest horizontal-face area. Hall mezzanines are floors strictly below
    that ceiling; Tower floors are strictly above.

    Returns the ceiling Z separately so callers can render it if useful.
    """
    if not floors_with_w:
        return [], [], 0.0
    # Find max-weight peak — this is the Hall ceiling
    ceiling_z, _ = max(floors_with_w, key=lambda p: p[1])
    sorted_z = sorted(z for z, _ in floors_with_w)
    hall_mezz = [z for z in sorted_z if z < ceiling_z - 1e-3]
    tower = [z for z in sorted_z if z > ceiling_z + 1e-3]
    return hall_mezz, tower, ceiling_z


def select_plan_cuts(
    obj_z_min: float,
    floors_with_w: list[tuple[float, float]],
    floor_offset_mm: float = 1500.0,
) -> dict[str, float]:
    """Pick 4 plan cut heights from detected floors+weights.

    Convention:
      - hall_ground:   z_min + floor_offset (always present)
      - hall_first:    hall_mezz[0] + floor_offset (lowest hall mezzanine)
      - hall_second:   hall_mezz[1] + floor_offset (second hall mezzanine)
      - tower_typical: tower_floors[0] + floor_offset (lowest tower floor —
                       most likely to be inside the tower envelope rather than
                       at the parapet/roof level)
    """
    hall_mezz, tower, _ceiling = split_hall_tower(floors_with_w)
    cuts: dict[str, float] = {"hall_ground": obj_z_min + floor_offset_mm}
    if len(hall_mezz) >= 1:
        cuts["hall_first"] = hall_mezz[0] + floor_offset_mm
    if len(hall_mezz) >= 2:
        cuts["hall_second"] = hall_mezz[1] + floor_offset_mm
    if tower:
        cuts["tower_typical"] = tower[0] + floor_offset_mm
    return cuts


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: detect_floors.py <obj_path>")
        sys.exit(1)
    obj_path = sys.argv[1]
    floors_w = detect_floors_with_weights(obj_path)
    print(f"Detected {len(floors_w)} floor slabs (z mm, weight):")
    for z, w in floors_w:
        print(f"  z = {z:>10.1f} mm  ({z / 1000:5.2f} m)   weight = {w:.1f}")
    hall_mezz, tower, ceiling = split_hall_tower(floors_w)
    print(f"\nHall ceiling (max-weight peak): z = {ceiling:.1f} mm")
    print(f"Hall mezzanines: {hall_mezz}")
    print(f"Tower floors:    {tower}")
    verts, _ = parse_obj_mesh(obj_path)
    z_min = float(verts[:, 2].min())
    print(f"\nz_min = {z_min:.1f} mm")
    cuts = select_plan_cuts(z_min, floors_w)
    print(f"\nPlan cut heights:")
    for name, z in cuts.items():
        print(f"  {name:>14s} : {z:>10.1f} mm")
