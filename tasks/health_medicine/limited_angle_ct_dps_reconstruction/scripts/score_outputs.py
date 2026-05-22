#!/usr/bin/env python
"""Local scorer for limited_angle_ct_dps_reconstruction."""

from __future__ import annotations

import io
import json

import numpy as np
from scipy.ndimage import uniform_filter

DATA_RANGE = 0.04
PSNR_THRESHOLD = 32.0
SSIM_THRESHOLD = 0.90
CROP_SIZE = 480


def central_crop(x: np.ndarray, n: int = CROP_SIZE) -> np.ndarray:
    h, w = x.shape
    if h < n or w < n:
        raise ValueError(f"array too small for crop {n}: got {x.shape}")
    y0, x0 = (h - n) // 2, (w - n) // 2
    return x[y0 : y0 + n, x0 : x0 + n]


def compute_psnr(reference: np.ndarray, recon: np.ndarray, data_range: float = DATA_RANGE) -> float:
    mse = float(np.mean((reference.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def compute_ssim(reference: np.ndarray, recon: np.ndarray, data_range: float = DATA_RANGE) -> float:
    # Match skimage.metrics.structural_similarity defaults used by the raw scorer:
    # grayscale image, win_size=7, gaussian_weights=False, sample_covariance=True.
    win_size = 7
    pad = (win_size - 1) // 2
    cov_norm = (win_size**2) / float((win_size**2) - 1)
    x = reference.astype(np.float64)
    y = recon.astype(np.float64)

    ux = uniform_filter(x, size=win_size)
    uy = uniform_filter(y, size=win_size)
    uxx = uniform_filter(x * x, size=win_size)
    uyy = uniform_filter(y * y, size=win_size)
    uxy = uniform_filter(x * y, size=win_size)

    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * ux * uy + c1) * (2 * vxy + c2)
    den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    ssim_map = num / den
    if pad > 0:
        ssim_map = ssim_map[pad:-pad, pad:-pad]
    return float(np.mean(ssim_map))


def score_reconstruction_arrays(recon: np.ndarray, reference: np.ndarray) -> tuple[float, dict]:
    if recon.shape != (512, 512):
        return 0.0, {"error": f"shape mismatch: expected (512, 512), got {recon.shape}"}
    if reference.shape != (512, 512):
        return 0.0, {"error": f"hidden reference shape drift: {reference.shape}"}
    if not np.issubdtype(recon.dtype, np.number):
        return 0.0, {"error": f"non-numeric dtype: {recon.dtype}"}
    if not np.all(np.isfinite(recon)):
        return 0.0, {"error": "reconstruction contains non-finite values"}

    recon_crop = central_crop(recon.astype(np.float32))
    reference_crop = central_crop(reference.astype(np.float32))
    psnr_val = compute_psnr(reference_crop, recon_crop)
    ssim_val = compute_ssim(reference_crop, recon_crop)
    passed = psnr_val >= PSNR_THRESHOLD and ssim_val >= SSIM_THRESHOLD
    details = {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "thresholds": {
            "psnr": PSNR_THRESHOLD,
            "ssim": SSIM_THRESHOLD,
            "data_range": DATA_RANGE,
            "crop_size": CROP_SIZE,
        },
        "passed": passed,
    }
    return (1.0 if passed else 0.0), details


def score_reconstruction_bytes(output_bytes: bytes, reference_bytes: bytes) -> tuple[float, dict]:
    if not output_bytes:
        return 0.0, {"error": "empty reconstruction output"}
    if not reference_bytes:
        return 0.0, {"error": "empty hidden reference"}
    try:
        recon = np.load(io.BytesIO(output_bytes))
        reference = np.load(io.BytesIO(reference_bytes))
    except Exception as exc:
        return 0.0, {"error": f"numpy load failed: {exc}"}
    return score_reconstruction_arrays(recon, reference)


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()

    score, details = score_reconstruction_arrays(
        np.load(args.recon).astype(np.float32),
        np.load(args.reference).astype(np.float32),
    )
    print(json.dumps({"score": score, **details}, indent=2))


if __name__ == "__main__":
    main()
