# -*- coding: utf-8 -*-
"""
verify_stl.py -- Core evaluation script (runs on remote Windows VM)

Compares the agent's simulated stock model (agent_sim.stl) against the expert
reference stock model (reference_sim.stl) using point-to-mesh distance.

Strategy:
    - Sample N points uniformly from the agent's mesh surface
    - For each point, find the closest point on the reference mesh surface
      (using trimesh.proximity.closest_point -- does NOT require watertight meshes)
    - Compute what fraction of points fall within tolerance bands
    - Produce a weighted score in [0, 1]

Design note: We use unsigned closest-point distance rather than signed distance
because (a) PowerMill STL exports are often not watertight, which breaks
signed_distance, and (b) closest_point is significantly faster.

Requirements:
    pip install trimesh numpy

Usage:
    python verify_stl.py --agent PATH_TO_AGENT_STL --reference PATH_TO_REF_STL

Output:
    JSON to stdout, e.g.:
    {"score": 0.87, "mean_dist_mm": 0.21, "ratio_perfect": 0.83, ...}

Exit codes:
    0 = success (score printed to stdout as JSON)
    1 = error (missing files, mesh load failure, etc.)
"""

import sys
import json
import argparse
import numpy as np

try:
    import trimesh
except ImportError:
    print('{"error": "trimesh not installed -- run: pip install trimesh"}', flush=True)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Scoring thresholds (millimeters)
# ---------------------------------------------------------------------------
# Points within TOLERANCE_PERFECT mm of the reference surface are "perfect".
# Points within TOLERANCE_ACCEPTABLE mm get partial credit.
# Points beyond TOLERANCE_ACCEPTABLE are penalized (contribute 0).
TOLERANCE_PERFECT = 0.3  # mm
TOLERANCE_ACCEPTABLE = 2.0  # mm

# Number of surface points to sample for comparison
N_SAMPLE_POINTS = 10_000

# Weights for the final score formula:
# score = WEIGHT_PERFECT * ratio_perfect + WEIGHT_ACCEPTABLE * ratio_acceptable
WEIGHT_PERFECT = 0.70
WEIGHT_ACCEPTABLE = 0.30


def compare_stls(agent_stl: str, reference_stl: str) -> dict:
    """Compare two STL meshes and return a scoring dict.

    1. Load both meshes (handles Scene objects by concatenating geometries)
    2. Sample N_SAMPLE_POINTS from the agent surface
    3. For each sample, find the closest point on the reference surface
    4. Compute distance statistics and final score

    Args:
        agent_stl: path to agent's stock model STL
        reference_stl: path to expert reference STL
    Returns:
        dict with score, distance stats, and parameters
    """
    print(f"Loading reference: {reference_stl}", file=sys.stderr)
    ref_mesh = trimesh.load_mesh(reference_stl)

    print(f"Loading agent: {agent_stl}", file=sys.stderr)
    agent_mesh = trimesh.load_mesh(agent_stl)

    # Handle Scene objects (trimesh may return Scene if STL has multiple bodies)
    if not isinstance(ref_mesh, trimesh.Trimesh):
        ref_mesh = trimesh.util.concatenate(list(ref_mesh.geometry.values()))
    if not isinstance(agent_mesh, trimesh.Trimesh):
        agent_mesh = trimesh.util.concatenate(list(agent_mesh.geometry.values()))

    # Sample points uniformly from the agent's surface
    print(f"Sampling {N_SAMPLE_POINTS} points from agent surface...", file=sys.stderr)
    sample_pts, _ = trimesh.sample.sample_surface(agent_mesh, N_SAMPLE_POINTS)

    # Compute closest-point distances from sampled points to reference mesh
    # Returns: (closest_points, distances, triangle_ids)
    print(f"Computing closest-point distances to reference...", file=sys.stderr)
    _, distances, _ = trimesh.proximity.closest_point(ref_mesh, sample_pts)

    # ---------------------------------------------------------------------------
    # Distance statistics
    # ---------------------------------------------------------------------------
    mean_dist = float(np.mean(distances))
    median_dist = float(np.median(distances))
    max_dist = float(np.max(distances))
    ratio_perfect = float(np.mean(distances <= TOLERANCE_PERFECT))
    ratio_acceptable = float(np.mean(distances <= TOLERANCE_ACCEPTABLE))

    # ---------------------------------------------------------------------------
    # Final score: weighted combination of tolerance fractions, clamped to [0, 1]
    # ---------------------------------------------------------------------------
    score = WEIGHT_PERFECT * ratio_perfect + WEIGHT_ACCEPTABLE * ratio_acceptable
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "mean_dist_mm": round(mean_dist, 4),
        "median_dist_mm": round(median_dist, 4),
        "max_dist_mm": round(max_dist, 4),
        "ratio_perfect": round(ratio_perfect, 4),  # fraction within 0.3mm
        "ratio_acceptable": round(ratio_acceptable, 4),  # fraction within 2.0mm
        "n_sample_points": N_SAMPLE_POINTS,
        "tolerance_perfect_mm": TOLERANCE_PERFECT,
        "tolerance_acceptable_mm": TOLERANCE_ACCEPTABLE,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare agent STL vs reference STL and output JSON score"
    )
    parser.add_argument("--agent", required=True, help="Path to agent_sim.stl")
    parser.add_argument("--reference", required=True, help="Path to reference_sim.stl")
    args = parser.parse_args()

    import os

    for path, label in [(args.agent, "agent STL"), (args.reference, "reference STL")]:
        if not os.path.exists(path):
            result = {"score": 0.0, "error": f"{label} not found: {path}"}
            print(json.dumps(result))
            sys.exit(1)

    try:
        result = compare_stls(args.agent, args.reference)
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        result = {"score": 0.0, "error": str(e)}
        print(json.dumps(result))
        sys.exit(1)


if __name__ == "__main__":
    main()
