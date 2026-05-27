"""Evaluator verifier for MBMD task. Runs on the VM at eval time.

Compares agent solution.npz against ground_truth.npz using RMSE over FOV mask.
Prints JSON to stdout with score and metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PASS_WATER_RMSE = 0.150
PASS_BONE_RMSE = 0.050
STRETCH_WATER_RMSE = 0.050
STRETCH_BONE_RMSE = 0.025


def _fov_mask(N: int, pad: int = 2) -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(N) - (N - 1) / 2,
        np.arange(N) - (N - 1) / 2,
        indexing="ij",
    )
    return np.sqrt(y * y + x * x) < (N / 2 - pad)


def _rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    diff = (pred - truth)[mask]
    return float(np.sqrt((diff * diff).mean()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True, type=Path)
    ap.add_argument("--ground_truth", required=True, type=Path)
    args = ap.parse_args()

    if not args.solution.exists():
        json.dump({"score": 0.0, "error": "solution file not found"}, sys.stdout)
        return 0

    try:
        sol = np.load(args.solution)
    except Exception as e:
        json.dump({"score": 0.0, "error": f"cannot load solution: {e}"}, sys.stdout)
        return 0

    if "water_fraction" not in sol or "bone_fraction" not in sol:
        json.dump(
            {"score": 0.0, "error": "solution missing water_fraction or bone_fraction"},
            sys.stdout,
        )
        return 0

    gt = np.load(args.ground_truth)
    water_est = np.clip(sol["water_fraction"].astype(np.float32), 0.0, 1.0)
    bone_est = np.clip(sol["bone_fraction"].astype(np.float32), 0.0, 1.0)
    water_gt = gt["water_fraction_gt"].astype(np.float32)
    bone_gt = gt["bone_fraction_gt"].astype(np.float32)

    if water_est.shape != water_gt.shape or bone_est.shape != bone_gt.shape:
        json.dump(
            {
                "score": 0.0,
                "error": f"shape mismatch: {water_est.shape}/{bone_est.shape} vs {water_gt.shape}/{bone_gt.shape}",
            },
            sys.stdout,
        )
        return 0

    N = water_est.shape[0]
    fov = _fov_mask(N)

    water_rmse = _rmse(water_est, water_gt, fov)
    bone_rmse = _rmse(bone_est, bone_gt, fov)

    passed = water_rmse <= PASS_WATER_RMSE and bone_rmse <= PASS_BONE_RMSE
    stretch = water_rmse <= STRETCH_WATER_RMSE and bone_rmse <= STRETCH_BONE_RMSE

    if stretch:
        score = 1.0
    elif passed:
        score = 0.8
    else:
        score = 0.0

    report = {
        "score": score,
        "water_rmse_fov": water_rmse,
        "bone_rmse_fov": bone_rmse,
        "passed": passed,
        "stretch_passed": stretch,
    }

    json.dump(report, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
