"""Local image verification helpers for VTK rendering outputs."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class RenderingMetrics:
    width: int
    height: int
    mse: float
    mae: float
    psnr: float
    ssim: float


def _decode_image(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def _grayscale(image: np.ndarray) -> np.ndarray:
    return (
        0.299 * image[:, :, 0].astype(np.float32)
        + 0.587 * image[:, :, 1].astype(np.float32)
        + 0.114 * image[:, :, 2].astype(np.float32)
    ) / 255.0


def _resize_to_reference(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image, mode="RGB")
    resized = pil_image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _gaussian_blur(image: np.ndarray, radius: float = 1.5) -> np.ndarray:
    pil_image = Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_a = _gaussian_blur(a)
    mu_b = _gaussian_blur(b)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a_sq = _gaussian_blur(a * a) - mu_a_sq
    sigma_b_sq = _gaussian_blur(b * b) - mu_b_sq
    sigma_ab = _gaussian_blur(a * b) - mu_ab
    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    score_map = numerator / denominator
    return float(np.mean(score_map))


def compute_rendering_metrics(image_bytes: bytes, reference_bytes: bytes) -> RenderingMetrics:
    reference = _decode_image(reference_bytes)
    image = _decode_image(image_bytes)

    if image.shape[:2] != reference.shape[:2]:
        image = _resize_to_reference(image, reference.shape[1], reference.shape[0])

    image_gray = _grayscale(image)
    reference_gray = _grayscale(reference)
    diff = image_gray - reference_gray
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse == 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))
    ssim = _ssim(image_gray, reference_gray)
    return RenderingMetrics(
        width=int(reference.shape[1]),
        height=int(reference.shape[0]),
        mse=mse,
        mae=mae,
        psnr=psnr,
        ssim=ssim,
    )


def verify_rendering_output(
    image_bytes: bytes,
    reference_bytes: bytes,
    contract: dict[str, Any],
) -> dict[str, Any]:
    metrics = compute_rendering_metrics(image_bytes, reference_bytes)
    min_ssim = float(contract.get("min_ssim", 0.95))
    passed = metrics.ssim >= min_ssim
    return {
        "pass": passed,
        "min_ssim": min_ssim,
        "width": metrics.width,
        "height": metrics.height,
        "mse": metrics.mse,
        "mae": metrics.mae,
        "psnr": metrics.psnr,
        "ssim": metrics.ssim,
    }
