"""Pure helpers for chroma-key evaluation."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any

import cv2
import numpy as np

ROI_INPUT_GATE_THRESHOLD = 0.3
SOFT_GATE_PASS_THRESHOLD = 0.5
HARD_QUALITY_FULL_FRAME_WEIGHT = 0.3
HARD_QUALITY_ROI_EDGE_WEIGHT = 0.7


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def remote_child(base: str, *parts: str) -> str:
    path = PureWindowsPath(base)
    for part in parts:
        path = path / part
    return str(path)


def ps_quote(text: str) -> str:
    return text.replace("'", "''")


def ps_literal(text: str) -> str:
    return f"'{ps_quote(text)}'"


def norm_to_xyxy(norm_bbox: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in norm_bbox]
    x1 = int(round(clamp(x1, 0.0, 1.0) * width))
    y1 = int(round(clamp(y1, 0.0, 1.0) * height))
    x2 = int(round(clamp(x2, 0.0, 1.0) * width))
    y2 = int(round(clamp(y2, 0.0, 1.0) * height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(clamp(x1, 0, max(0, width - 1)))
    y1 = int(clamp(y1, 0, max(0, height - 1)))
    x2 = int(clamp(x2, x1 + 1, width))
    y2 = int(clamp(y2, y1 + 1, height))
    return [x1, y1, x2, y2]


def xyxy_to_norm(xyxy: list[int], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 1.0, 1.0]
    return [
        clamp(x1 / width, 0.0, 1.0),
        clamp(y1 / height, 0.0, 1.0),
        clamp(x2 / width, 0.0, 1.0),
        clamp(y2 / height, 0.0, 1.0),
    ]


def xyxy_to_crop(xyxy: list[int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x = max(0, x1)
    y = max(0, y1)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return x, y, w, h


def legacy_bbox_to_norm(bbox: list[float], width: int, height: int) -> list[float]:
    if not bbox or len(bbox) != 4:
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(clamp(x1, 0, max(0, width - 1)))
    y1 = int(clamp(y1, 0, max(0, height - 1)))
    x2 = int(clamp(x2, x1 + 1, width))
    y2 = int(clamp(y2, y1 + 1, height))
    return xyxy_to_norm([x1, y1, x2, y2], width, height)


def load_breakpoints(point_file: Path, default_sample_count: int) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    payload: dict[str, Any] = {}
    points: list[dict[str, Any]] = []
    sample_count = max(1, int(default_sample_count))
    if not point_file.exists():
        return payload, points, sample_count
    try:
        payload = json.loads(point_file.read_text(encoding="utf-8-sig"))
        sample_count = max(1, int(payload.get("sample_count", sample_count)))
        raw_points = payload.get("breakpoints", payload.get("points", []))
        for item in raw_points:
            if not isinstance(item, dict):
                continue
            time_sec = safe_float(item.get("time_sec", 0.0), 0.0)
            if time_sec < 0:
                continue
            point = dict(item)
            point["time_sec"] = round(time_sec, 6)
            points.append(point)
    except Exception:
        return {}, [], sample_count
    points = sorted(points, key=lambda p: p["time_sec"])
    return payload, points[:sample_count], sample_count


def resolve_dual_bboxes(
    point: dict[str, Any],
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
) -> tuple[list[int], list[int], list[float], list[float]]:
    input_norm = point.get("input_bbox_norm")
    output_norm = point.get("output_bbox_norm")
    if not input_norm or len(input_norm) != 4 or not output_norm or len(output_norm) != 4:
        legacy = point.get("bbox_with_tolerance") or point.get("bbox")
        if legacy and len(legacy) == 4:
            output_norm = legacy_bbox_to_norm(legacy, output_width, output_height)
            input_norm = output_norm
        else:
            input_norm = [0.0, 0.0, 1.0, 1.0]
            output_norm = [0.0, 0.0, 1.0, 1.0]
    input_xyxy = norm_to_xyxy(input_norm, input_width, input_height)
    output_xyxy = norm_to_xyxy(output_norm, output_width, output_height)
    input_norm = xyxy_to_norm(input_xyxy, input_width, input_height)
    output_norm = xyxy_to_norm(output_xyxy, output_width, output_height)
    return input_xyxy, output_xyxy, input_norm, output_norm


def resize_like(src: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = target_shape
    if src.shape[0] == height and src.shape[1] == width:
        return src
    return cv2.resize(src, (width, height), interpolation=cv2.INTER_LINEAR)


def edge_iou_score(image_a: np.ndarray, image_b: np.ndarray) -> float:
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    edge_a = cv2.Canny(gray_a, 80, 160) > 0
    edge_b = cv2.Canny(gray_b, 80, 160) > 0
    union = np.logical_or(edge_a, edge_b).sum()
    if union <= 0:
        return 0.0
    inter = np.logical_and(edge_a, edge_b).sum()
    return clamp(float(inter / union), 0.0, 1.0)


def hist_corr_score(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray | None = None) -> float:
    hsv_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], mask, [32, 32], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], mask, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    return clamp((corr + 1.0) / 2.0, 0.0, 1.0)


def foreground_mask_non_green(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_low = np.array([35, 40, 40], dtype=np.uint8)
    green_high = np.array([90, 255, 255], dtype=np.uint8)
    green = cv2.inRange(hsv, green_low, green_high)
    fg = cv2.bitwise_not(green)
    kernel = np.ones((3, 3), dtype=np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fg


def foreground_mask_from_input_reference(input_roi: np.ndarray, reference_roi: np.ndarray) -> np.ndarray:
    reference_roi = resize_like(reference_roi, (input_roi.shape[0], input_roi.shape[1]))
    diff = cv2.absdiff(input_roi, reference_roi)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask = (diff_gray < 28).astype(np.uint8) * 255
    kernel_small = np.ones((3, 3), dtype=np.uint8)
    kernel_large = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    ratio = float((mask > 0).sum()) / float(mask.size)
    if ratio < 0.01 or ratio > 0.75:
        return foreground_mask_non_green(input_roi)
    return mask


def score_roi_input_foreground(
    output_roi: np.ndarray,
    input_roi: np.ndarray,
    reference_roi: np.ndarray | None = None,
) -> float:
    input_roi = resize_like(input_roi, (output_roi.shape[0], output_roi.shape[1]))
    if reference_roi is not None:
        reference_roi = resize_like(reference_roi, (output_roi.shape[0], output_roi.shape[1]))
        fg_mask = foreground_mask_from_input_reference(input_roi, reference_roi)
    else:
        fg_mask = foreground_mask_non_green(input_roi)
    fg_ratio = float((fg_mask > 0).sum()) / float(fg_mask.size)
    if fg_ratio < 0.02:
        return 0.0
    hist_score = hist_corr_score(output_roi, input_roi, mask=fg_mask)
    orb = cv2.ORB_create(nfeatures=500)
    gray_in = cv2.cvtColor(input_roi, cv2.COLOR_BGR2GRAY)
    gray_out = cv2.cvtColor(output_roi, cv2.COLOR_BGR2GRAY)
    key_in, des_in = orb.detectAndCompute(gray_in, fg_mask)
    key_out, des_out = orb.detectAndCompute(gray_out, fg_mask)
    orb_score = 0.0
    if des_in is not None and des_out is not None and key_in and key_out:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des_in, des_out)
        good = [m for m in matches if m.distance <= 60]
        orb_score = clamp(len(good) / max(len(key_in), len(key_out), 1), 0.0, 1.0)
    diff = cv2.absdiff(output_roi, input_roi)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    masked = diff_gray[fg_mask > 0]
    color_score = 0.0 if masked.size == 0 else clamp(1.0 - float(masked.mean()) / 255.0, 0.0, 1.0)
    return clamp(0.4 * hist_score + 0.4 * orb_score + 0.2 * color_score, 0.0, 1.0)


def compute_hard_quality_frame_score(full_frame_edge_iou: float, roi_edge_iou: float) -> float:
    return clamp(
        HARD_QUALITY_FULL_FRAME_WEIGHT * safe_float(full_frame_edge_iou, 0.0)
        + HARD_QUALITY_ROI_EDGE_WEIGHT * safe_float(roi_edge_iou, 0.0),
        0.0,
        1.0,
    )


def compute_final_frame_result(
    *,
    roi_input_cv: float,
    soft_frame_score: float,
    full_frame_edge_iou: float,
    roi_edge_iou: float,
) -> dict[str, float | bool]:
    roi_input_cv = clamp(safe_float(roi_input_cv, 0.0), 0.0, 1.0)
    soft_frame_score = clamp(safe_float(soft_frame_score, 0.0), 0.0, 1.0)
    full_frame_edge_iou = clamp(safe_float(full_frame_edge_iou, 0.0), 0.0, 1.0)
    roi_edge_iou = clamp(safe_float(roi_edge_iou, 0.0), 0.0, 1.0)

    roi_input_gate_passed = roi_input_cv >= ROI_INPUT_GATE_THRESHOLD
    hard_quality_frame_score = compute_hard_quality_frame_score(full_frame_edge_iou, roi_edge_iou)
    soft_gate_passed = soft_frame_score >= SOFT_GATE_PASS_THRESHOLD

    if not roi_input_gate_passed or not soft_gate_passed:
        final_frame_score = 0.0
    else:
        final_frame_score = hard_quality_frame_score

    return {
        "roi_input_cv": roi_input_cv,
        "roi_input_gate_passed": roi_input_gate_passed,
        "soft_frame_score": soft_frame_score,
        "soft_gate_passed": soft_gate_passed,
        "full_frame_edge_iou": full_frame_edge_iou,
        "roi_edge_iou": roi_edge_iou,
        "hard_quality_frame_score": hard_quality_frame_score,
        "final_frame_score": final_frame_score,
    }
