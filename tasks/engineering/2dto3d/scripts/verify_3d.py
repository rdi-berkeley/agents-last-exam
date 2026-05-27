"""
verify_3d.py — Core evaluation script (runs on remote Windows VM)

Compares the agent's extracted features JSON against the ground-truth features
JSON using a three-dimensional scoring system:

  1. Global geometry (20%): volume + bounding box origin alignment
  2. Feature quantity (30%): hole histogram comparison
  3. Feature precision (50%): hole position + axis matching

Requirements:
    numpy, scipy

Usage:
    python verify_3d.py --agent PATH_TO_AGENT_JSON --reference PATH_TO_GT_JSON

Output:
    JSON to stdout, e.g.:
    {"score": 0.85, "volume_error_ratio": 0.02, "matched_features": 8, ...}

Exit codes:
    0 = success (score printed to stdout as JSON)
    1 = error (missing files, parse failure, etc.)
"""

import sys
import json
import argparse
import os

try:
    import numpy as np
    from scipy.spatial.distance import cdist
except ImportError:
    print(
        json.dumps({"error": "numpy/scipy not installed -- run: pip install numpy scipy"}),
        flush=True,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_GLOBAL = 0.20  # 20% volume + origin
WEIGHT_QUANTITY = 0.30  # 30% hole histogram
WEIGHT_PRECISION = 0.50  # 50% hole position + axis

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
VOLUME_ERROR_THRESHOLD = 0.05  # > 5% volume error → 0 for volume component
POS_ERROR_MAX = 2.0  # > 2mm position error → match failure
POS_ERROR_PERFECT = 0.1  # < 0.1mm position error → perfect score
AXIS_ANGLE_MAX = 15.0  # > 15° axis error → match failure


def _vec_angle(v1, v2):
    """Compute angle (degrees) between two vectors. Axis is undirected (0-90°)."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 90.0

    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    if angle > 90:
        angle = 180 - angle
    return angle


def evaluate_global(gt_data, agent_data):
    """Dimension 1: Global geometry compliance (max 20 points internal)."""
    gt_geom = gt_data["geometry"]
    stu_geom = agent_data["geometry"]

    # Volume score (10 pts)
    vol_gt = gt_geom["volume"]
    vol_stu = stu_geom["volume"]

    if vol_gt == 0:
        vol_diff = 1.0 if vol_stu != 0 else 0.0
    else:
        vol_diff = abs(vol_stu - vol_gt) / abs(vol_gt)

    if vol_diff <= 0.01:
        vol_score = 10.0
    elif vol_diff >= VOLUME_ERROR_THRESHOLD:
        vol_score = 0.0
    else:
        vol_score = 10.0 * (1 - (vol_diff - 0.01) / (VOLUME_ERROR_THRESHOLD - 0.01))

    # Origin alignment score (10 pts)
    bbox_min_gt = np.array(gt_geom["bbox_min"])
    bbox_min_stu = np.array(stu_geom["bbox_min"])
    origin_dist = np.linalg.norm(bbox_min_gt - bbox_min_stu)

    if origin_dist <= 0.1:
        origin_score = 10.0
    elif origin_dist >= 2.0:
        origin_score = 0.0
    else:
        origin_score = 10.0 * (1 - origin_dist / 2.0)

    return {
        "score": round(vol_score + origin_score, 2),
        "max_score": 20.0,
        "details": {
            "volume_error_ratio": round(vol_diff, 4),
            "origin_deviation": round(origin_dist, 4),
        },
    }


def evaluate_quantity(gt_data, agent_data):
    """Dimension 2: Feature quantity completeness (max 30 points internal)."""
    gt_hist = gt_data["features"]["hole_histogram"]
    stu_hist = agent_data["features"]["hole_histogram"]

    total_items = sum(gt_hist.values())
    if total_items == 0:
        return {"score": 30.0, "max_score": 30.0, "details": "No holes in GT"}

    score_per_item = 30.0 / total_items
    current_score = 30.0

    missing_log = []
    extra_log = []

    # Missing holes
    for dia, count in gt_hist.items():
        stu_count = stu_hist.get(dia, 0)
        diff = count - stu_count
        if diff > 0:
            current_score -= diff * score_per_item
            missing_log.append(f"Missing {diff} holes of Dia {dia}mm")

    # Extra holes (half penalty)
    for dia, count in stu_hist.items():
        gt_count = gt_hist.get(dia, 0)
        diff = count - gt_count
        if diff > 0:
            current_score -= diff * score_per_item * 0.5
            extra_log.append(f"Extra {diff} holes of Dia {dia}mm")

    return {
        "score": round(max(0, current_score), 2),
        "max_score": 30.0,
        "details": {"missing": missing_log, "extra": extra_log},
    }


def evaluate_precision(gt_data, agent_data):
    """Dimension 3: Feature position and precision (max 50 points internal)."""
    gt_holes = gt_data["features"]["holes_details"]
    stu_holes = agent_data["features"]["holes_details"]

    if not gt_holes:
        return {"score": 50.0, "max_score": 50.0, "details": "No holes to check"}
    if not stu_holes:
        return {
            "score": 0.0,
            "max_score": 50.0,
            "details": "No holes found in agent file",
        }

    score_per_hole = 50.0 / len(gt_holes)
    total_score = 0.0
    matched_pairs = []
    total_pos_error = 0.0
    total_axis_error = 0.0

    # Build distance matrix
    gt_locs = np.array([h["location"] for h in gt_holes])
    stu_locs = np.array([h["location"] for h in stu_holes])
    dist_matrix = cdist(gt_locs, stu_locs, metric="euclidean")

    # Greedy matching
    for i, gt_h in enumerate(gt_holes):
        candidates = []
        for j, stu_h in enumerate(stu_holes):
            if abs(gt_h["diameter"] - stu_h["diameter"]) < 0.2:
                candidates.append(j)

        if not candidates:
            continue

        best_match_idx = -1
        best_dist = float("inf")
        for j in candidates:
            dist = dist_matrix[i, j]
            if dist < best_dist:
                best_dist = dist
                best_match_idx = j

        if best_match_idx != -1 and best_dist < POS_ERROR_MAX:
            stu_h = stu_holes[best_match_idx]
            axis_angle = _vec_angle(gt_h["axis"], stu_h["axis"])

            if axis_angle < AXIS_ANGLE_MAX:
                # Position score (70% weight)
                if best_dist <= POS_ERROR_PERFECT:
                    pos_score = 1.0
                else:
                    pos_score = max(
                        0, 1.0 - (best_dist - 0.1) / (POS_ERROR_MAX - 0.1)
                    )

                # Axis score (30% weight)
                ang_score = max(0, 1.0 - axis_angle / AXIS_ANGLE_MAX)

                hole_score = score_per_hole * (0.7 * pos_score + 0.3 * ang_score)
                total_score += hole_score

                matched_pairs.append(
                    {
                        "gt_idx": i,
                        "stu_idx": best_match_idx,
                        "pos_error": round(best_dist, 4),
                        "axis_error": round(axis_angle, 2),
                    }
                )
                total_pos_error += best_dist
                total_axis_error += axis_angle

    num_matched = len(matched_pairs)
    avg_pos_error = total_pos_error / num_matched if num_matched > 0 else 0

    return {
        "score": round(total_score, 2),
        "max_score": 50.0,
        "details": {
            "total_gt_features": len(gt_holes),
            "matched_features": num_matched,
            "avg_position_error": round(avg_pos_error, 4),
            "avg_axis_error": round(
                total_axis_error / num_matched if num_matched else 0, 2
            ),
        },
    }


def compare(gt_data, agent_data):
    """Run all three evaluation dimensions and produce final score."""
    global_res = evaluate_global(gt_data, agent_data)
    quant_res = evaluate_quantity(gt_data, agent_data)
    prec_res = evaluate_precision(gt_data, agent_data)

    # Internal scores are on 0-100 scale; normalize to 0-1
    raw_total = global_res["score"] + quant_res["score"] + prec_res["score"]
    final_score = raw_total / 100.0
    final_score = max(0.0, min(1.0, final_score))

    return {
        "score": round(final_score, 4),
        "volume_error_ratio": global_res["details"].get("volume_error_ratio", None),
        "origin_deviation": global_res["details"].get("origin_deviation", None),
        "matched_features": prec_res["details"].get("matched_features", None)
        if isinstance(prec_res["details"], dict)
        else None,
        "total_gt_features": prec_res["details"].get("total_gt_features", None)
        if isinstance(prec_res["details"], dict)
        else None,
        "raw_score_100": round(raw_total, 1),
        "breakdown": {
            "global_geometry": global_res,
            "feature_quantity": quant_res,
            "feature_precision": prec_res,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare agent features JSON vs ground-truth features JSON"
    )
    parser.add_argument(
        "--agent", required=True, help="Path to agent_features.json"
    )
    parser.add_argument(
        "--reference", required=True, help="Path to gt_features.json"
    )
    args = parser.parse_args()

    for path, label in [
        (args.agent, "agent features"),
        (args.reference, "reference features"),
    ]:
        if not os.path.exists(path):
            result = {"score": 0.0, "error": f"{label} not found: {path}"}
            print(json.dumps(result))
            sys.exit(1)

    try:
        with open(args.agent, "r") as f:
            agent_data = json.load(f)
        with open(args.reference, "r") as f:
            gt_data = json.load(f)

        result = compare(gt_data, agent_data)
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        result = {"score": 0.0, "error": str(e)}
        print(json.dumps(result))
        sys.exit(1)


if __name__ == "__main__":
    main()
