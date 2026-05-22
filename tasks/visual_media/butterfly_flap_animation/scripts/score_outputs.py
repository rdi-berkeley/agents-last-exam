#!/usr/bin/env python3
"""Deterministic scorer for visual_media/butterfly_flap_animation outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


PASS_THRESHOLD = 0.65


@dataclass
class VideoProbe:
    exists: bool
    readable: bool
    codec_name: str = ""
    container_ext: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_sec: float = 0.0


@dataclass
class MotionMetrics:
    detection_ratio: float
    width_amplitude_ratio: float
    estimated_flap_cycles: float
    y_turning_points: int
    y_range_ratio: float
    max_jump_ratio: float
    source_shape_iou: float


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    probe: VideoProbe
    metrics: MotionMetrics | None
    components: dict[str, float]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 6)
        return payload


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values.astype(float)
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values.astype(float), (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _probe_video(path: Path) -> VideoProbe:
    if not path.exists():
        return VideoProbe(exists=False, readable=False)
    codec_name = ""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        codec_name = result.stdout.strip()
    except Exception:
        codec_name = ""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return VideoProbe(exists=True, readable=False, codec_name=codec_name, container_ext=path.suffix.lower())
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoProbe(
        exists=True,
        readable=bool(fps and frame_count and width and height),
        codec_name=codec_name,
        container_ext=path.suffix.lower(),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=(frame_count / fps) if fps else 0.0,
    )


def _source_mask(input_image_path: Path) -> np.ndarray:
    image = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read input image: {input_image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray < 245).astype(np.uint8) * 255
    return mask


def _normalize_mask(mask: np.ndarray, out_size: tuple[int, int] = (256, 192)) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 50]
    if not contours:
        return None
    points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(points)
    if w <= 0 or h <= 0:
        return None
    crop = mask[y : y + h, x : x + w]
    resized = cv2.resize(crop, out_size, interpolation=cv2.INTER_NEAREST)
    return resized > 0


def _foreground_mask(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = (gray > 25).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    )
    return mask


def _track_foreground(video_path: Path) -> tuple[VideoProbe, dict[str, np.ndarray]]:
    probe = _probe_video(video_path)
    if not probe.readable:
        return probe, {}

    cap = cv2.VideoCapture(str(video_path))
    widths: list[float] = []
    centers_x: list[float] = []
    centers_y: list[float] = []
    areas: list[float] = []
    masks: list[np.ndarray] = []
    valid = 0
    total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        mask = _foreground_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]
        if not contours:
            widths.append(np.nan)
            centers_x.append(np.nan)
            centers_y.append(np.nan)
            areas.append(0.0)
            masks.append(mask)
            continue

        points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(points)
        widths.append(float(w))
        centers_x.append(float(x + w / 2.0))
        centers_y.append(float(y + h / 2.0))
        areas.append(float(sum(cv2.contourArea(contour) for contour in contours)))
        masks.append(mask)
        valid += 1
    cap.release()

    def fill(values: list[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.all(~np.isfinite(arr)):
            return np.asarray([], dtype=float)
        finite = np.isfinite(arr)
        idx = np.arange(arr.size)
        arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
        return arr

    return probe, {
        "widths": fill(widths),
        "centers_x": fill(centers_x),
        "centers_y": fill(centers_y),
        "areas": np.asarray(areas, dtype=float),
        "masks": masks,
        "detection_ratio": np.asarray([valid / max(1, total)], dtype=float),
    }


def _estimate_flap_cycles(widths: np.ndarray) -> float:
    if widths.size < 30 or float(np.nanstd(widths)) < 1e-6:
        return 0.0
    signal = _moving_average(widths, 5)
    signal = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(signal))
    if spectrum.size <= 1:
        return 0.0
    spectrum[0] = 0.0
    # Ignore implausibly slow drift and very high-frequency jitter.
    max_cycle_bin = max(2, min(spectrum.size - 1, int(round(widths.size / 12))))
    spectrum[max_cycle_bin + 1 :] = 0.0
    peak_bin = int(np.argmax(spectrum))
    # Symmetric wing perspective can make the measured silhouette width peak
    # twice during one physical open->closed->open cycle. Normalize that
    # common second harmonic back to the public flap-cycle count.
    return float(peak_bin / 2.0 if peak_bin > 5 else peak_bin)


def _count_turning_points(values: np.ndarray, min_distance: int = 10) -> int:
    if values.size < 20:
        return 0
    smooth = _moving_average(values, 9)
    value_range = float(np.nanmax(smooth) - np.nanmin(smooth))
    if value_range < 8:
        return 0
    prominence = max(8.0, 0.12 * value_range)
    delta = np.diff(smooth)
    sign = np.sign(delta)
    for idx in range(1, sign.size):
        if sign[idx] == 0:
            sign[idx] = sign[idx - 1]
    for idx in range(sign.size - 2, -1, -1):
        if sign[idx] == 0:
            sign[idx] = sign[idx + 1]

    extrema: list[int] = []
    for idx in range(1, sign.size):
        if idx < min_distance or idx >= smooth.size - min_distance:
            continue
        window = smooth[idx - min_distance : idx + min_distance + 1]
        center = smooth[idx]
        if sign[idx - 1] > 0 and sign[idx] < 0 and center - np.min(window) >= prominence:
            extrema.append(idx)
        elif sign[idx - 1] < 0 and sign[idx] > 0 and np.max(window) - center >= prominence:
            extrema.append(idx)
    deduped: list[int] = []
    for idx in extrema:
        if not deduped or idx - deduped[-1] >= min_distance:
            deduped.append(idx)
    return len(deduped)


def _shape_iou(input_image_path: Path, masks: list[np.ndarray], widths: np.ndarray) -> float:
    source = _normalize_mask(_source_mask(input_image_path))
    if source is None or not masks or widths.size == 0:
        return 0.0
    top_count = min(5, widths.size)
    top_indices = np.argsort(widths)[-top_count:]
    scores: list[float] = []
    for idx in top_indices:
        candidate = _normalize_mask(masks[int(idx)])
        if candidate is None:
            continue
        intersection = np.logical_and(source, candidate).sum()
        union = np.logical_or(source, candidate).sum()
        if union:
            scores.append(float(intersection / union))
    return float(np.mean(scores)) if scores else 0.0


def score_video(output_video_path: Path, input_image_path: Path, pass_threshold: float = PASS_THRESHOLD) -> ScoreResult:
    probe, tracks = _track_foreground(output_video_path)
    if not probe.exists:
        return ScoreResult(0.0, False, "missing output video", "missing_output", probe, None, {})
    if not probe.readable:
        return ScoreResult(0.0, False, "unreadable output video", "unreadable_output", probe, None, {})
    if probe.container_ext != ".mp4" or probe.codec_name != "h264":
        return ScoreResult(
            0.0,
            False,
            "output must be an H.264 MP4",
            "codec",
            probe,
            None,
            {},
        )

    timing_score = _clip01(1.0 - abs(probe.fps - 30.0) / 6.0) * _clip01(
        min((probe.duration_sec - 3.5) / 0.5, (6.0 - probe.duration_sec) / 0.5, 1.0)
    )
    if timing_score <= 0.0:
        return ScoreResult(0.0, False, "frame rate or duration outside accepted range", "timing", probe, None, {})

    widths = tracks.get("widths", np.asarray([], dtype=float))
    centers_y = tracks.get("centers_y", np.asarray([], dtype=float))
    centers_x = tracks.get("centers_x", np.asarray([], dtype=float))
    masks = tracks.get("masks", [])
    detection_ratio = float(tracks.get("detection_ratio", np.asarray([0.0]))[0])
    if detection_ratio < 0.85 or widths.size < 30:
        metrics = MotionMetrics(detection_ratio, 0.0, 0.0, 0, 0.0, 1.0, 0.0)
        return ScoreResult(0.0, False, "butterfly foreground not detected in enough frames", "detection", probe, metrics, {})

    width_p05, width_p95 = np.percentile(widths, [5, 95])
    width_amp_ratio = float((width_p95 - width_p05) / max(1.0, np.median(widths)))
    estimated_cycles = _estimate_flap_cycles(widths)
    cycle_score = _clip01(1.0 - abs(estimated_cycles - 4.0) / 2.0)
    amplitude_score = _clip01((width_amp_ratio - 0.05) / 0.12)
    wing_score = 0.65 * cycle_score + 0.35 * amplitude_score

    y_turns = _count_turning_points(centers_y)
    y_range_ratio = float((np.percentile(centers_y, 95) - np.percentile(centers_y, 5)) / max(1, probe.height))
    xy = np.column_stack([_moving_average(centers_x, 5), _moving_average(centers_y, 5)])
    jumps = np.linalg.norm(np.diff(xy, axis=0), axis=1) if xy.shape[0] > 5 else np.asarray([0.0])
    max_jump_ratio = float(np.nanmax(jumps) / max(1, max(probe.width, probe.height)))
    y_turn_score = _clip01(y_turns / 2.0)
    y_range_score = _clip01(y_range_ratio / 0.12)
    smoothness_score = _clip01(1.0 - max(0.0, max_jump_ratio - 0.03) / 0.08)
    path_score = (
        0.45 * y_turn_score + 0.35 * y_range_score + 0.20 * smoothness_score
        if y_range_ratio >= 0.03
        else 0.0
    )

    iou = _shape_iou(input_image_path, masks, widths)
    texture_score = _clip01((iou - 0.22) / 0.40)
    detection_score = _clip01((detection_ratio - 0.85) / 0.15)

    components = {
        "timing": timing_score,
        "detection": detection_score,
        "wing_flap": wing_score,
        "path": path_score,
        "source_shape": texture_score,
    }
    score = (
        0.20 * components["timing"]
        + 0.15 * components["detection"]
        + 0.30 * components["wing_flap"]
        + 0.25 * components["path"]
        + 0.10 * components["source_shape"]
    )
    if components["wing_flap"] < 0.65:
        score = min(score, 0.55)
    if components["path"] < 0.65:
        score = min(score, 0.55)
    metrics = MotionMetrics(
        detection_ratio=detection_ratio,
        width_amplitude_ratio=width_amp_ratio,
        estimated_flap_cycles=estimated_cycles,
        y_turning_points=y_turns,
        y_range_ratio=y_range_ratio,
        max_jump_ratio=max_jump_ratio,
        source_shape_iou=iou,
    )
    passed = score >= pass_threshold
    reason = "passed" if passed else "score below pass threshold"
    return ScoreResult(float(score), passed, reason, None, probe, metrics, components)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--input-image", required=True, type=Path)
    parser.add_argument("--pass-threshold", type=float, default=PASS_THRESHOLD)
    args = parser.parse_args()

    result = score_video(args.output_video, args.input_image, args.pass_threshold)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
