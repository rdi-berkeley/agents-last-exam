"""Local mask scorer for pathology_nuclei_segmentation_gui_1."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskScoreResult:
    score: float
    dice: float
    passed: bool
    reason: str


def _decode_image_bytes(raw: bytes) -> np.ndarray | None:
    if not raw:
        return None
    buf = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def _to_binary_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        mask = image
    elif image.ndim == 3:
        mask = image.max(axis=2)
    else:
        raise ValueError(f"unsupported image rank: {image.ndim}")
    return (mask > 0).astype(np.uint8)


def _dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    pred_sum = int(prediction.sum())
    ref_sum = int(reference.sum())
    if pred_sum == 0 or ref_sum == 0:
        return 0.0
    intersection = int((prediction & reference).sum())
    return (2.0 * intersection) / float(pred_sum + ref_sum)


def score_mask_bytes(
    *,
    agent_bytes: bytes,
    reference_bytes: bytes,
    threshold: float,
) -> MaskScoreResult:
    agent_image = _decode_image_bytes(agent_bytes)
    if agent_image is None:
        return MaskScoreResult(0.0, 0.0, False, "agent output is unreadable")

    reference_image = _decode_image_bytes(reference_bytes)
    if reference_image is None:
        return MaskScoreResult(0.0, 0.0, False, "reference mask is unreadable")

    agent_mask = _to_binary_mask(agent_image)
    reference_mask = _to_binary_mask(reference_image)

    if agent_mask.shape != reference_mask.shape:
        return MaskScoreResult(
            0.0,
            0.0,
            False,
            f"shape mismatch: {agent_mask.shape} vs {reference_mask.shape}",
        )

    if int(agent_mask.sum()) == 0:
        return MaskScoreResult(0.0, 0.0, False, "agent mask is empty")

    dice = _dice(agent_mask, reference_mask)
    passed = dice >= threshold
    return MaskScoreResult(
        1.0 if passed else 0.0,
        dice,
        passed,
        "ok" if passed else f"dice below threshold {threshold:.2f}",
    )
