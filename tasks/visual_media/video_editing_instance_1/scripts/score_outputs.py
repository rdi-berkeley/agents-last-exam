#!/usr/bin/env python
"""Deterministic scorer for visual_media/video_editing_instance_1."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from cv2_compat import import_cv2

cv2 = import_cv2()


PASS_THRESHOLD = 0.75


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
class RevealMetrics:
    start_luma_mean: float
    end_ssim: float
    end_color_similarity: float
    max_luma_drop: float
    peak_frame_ratio: float
    mean_peak_count: float
    expanding_edge_score: float


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    probe: VideoProbe
    metrics: RevealMetrics | None
    components: dict[str, float]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["score"] = round(float(self.score), 6)
        return payload


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_run(cmd: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, check=True, text=True, capture_output=True)
    except Exception:
        return None


def _probe_video(path: Path) -> VideoProbe:
    if not path.exists():
        return VideoProbe(exists=False, readable=False)

    codec_name = ""
    result = _safe_run(
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
        ]
    )
    if result is not None:
        codec_name = result.stdout.strip().lower()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return VideoProbe(
            exists=True,
            readable=False,
            codec_name=codec_name,
            container_ext=path.suffix.lower(),
        )
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


def _read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _read_target_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read input image: {path}")
    width, height = size
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _ssim_gray(image_a: np.ndarray, image_b: np.ndarray) -> float:
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = float(gray_a.mean())
    mu_b = float(gray_b.mean())
    var_a = float(gray_a.var())
    var_b = float(gray_b.var())
    cov = float(((gray_a - mu_a) * (gray_b - mu_b)).mean())
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if denominator <= 0:
        return 0.0
    return _clip01(numerator / denominator)


def _color_similarity(image_a: np.ndarray, image_b: np.ndarray) -> float:
    a = image_a.astype(np.float32)
    b = image_b.astype(np.float32)
    channel_mae = float(np.mean(np.abs(a - b)) / 255.0)
    hsv_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2HSV).astype(np.float32)
    saturation_mae = float(np.mean(np.abs(hsv_a[:, :, 1] - hsv_b[:, :, 1])) / 255.0)
    return _clip01(1.0 - (0.65 * channel_mae + 0.35 * saturation_mae))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return values.astype(float)
    kernel = np.ones(window, dtype=float) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values.astype(float), (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _estimate_alpha(frame: np.ndarray, target: np.ndarray) -> np.ndarray:
    frame_luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    target_luma = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float32)
    denom = np.maximum(target_luma, 20.0)
    alpha = np.clip(frame_luma / denom, 0.0, 1.0)
    return cv2.GaussianBlur(alpha, (7, 7), 0)


def _radial_profile(alpha: np.ndarray, bin_count: int = 180) -> tuple[np.ndarray, np.ndarray]:
    height, width = alpha.shape
    yy, xx = np.indices((height, width))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    radii = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_radius = float(radii.max())
    bins = np.minimum((radii / max_radius * (bin_count - 1)).astype(np.int32), bin_count - 1)
    sums = np.bincount(bins.ravel(), weights=alpha.ravel(), minlength=bin_count)
    counts = np.bincount(bins.ravel(), minlength=bin_count)
    profile = sums / np.maximum(counts, 1)
    radius_values = np.linspace(0.0, max_radius, bin_count)
    return radius_values, profile


def _count_radial_edges(alpha: np.ndarray) -> tuple[int, float | None]:
    radius_values, profile = _radial_profile(alpha)
    profile = _moving_average(profile, 7)
    # A circular reveal has alpha that drops outward at one or more circular fronts.
    grad = -np.diff(profile)
    if grad.size < 10:
        return 0, None
    threshold = max(0.025, 0.22 * float(np.max(grad)))
    peaks: list[int] = []
    min_sep = 7
    for idx in range(2, grad.size - 2):
        if radius_values[idx] < 18:
            continue
        if grad[idx] < threshold:
            continue
        if grad[idx] >= grad[idx - 1] and grad[idx] >= grad[idx + 1]:
            if not peaks or idx - peaks[-1] >= min_sep:
                peaks.append(idx)
            elif grad[idx] > grad[peaks[-1]]:
                peaks[-1] = idx
    if not peaks:
        return 0, None
    return len(peaks), float(radius_values[max(peaks)])


def _analyze(frames: list[np.ndarray], target: np.ndarray) -> RevealMetrics:
    gray_means = np.array(
        [float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()) for frame in frames],
        dtype=float,
    )
    drops = gray_means[:-1] - gray_means[1:] if gray_means.size > 1 else np.array([0.0])
    max_drop = float(np.max(drops)) if drops.size else 0.0

    start_luma = float(gray_means[0])
    end_ssim = _ssim_gray(frames[-1], target)
    end_color_similarity = _color_similarity(frames[-1], target)

    sample_indices = np.linspace(
        max(1, int(len(frames) * 0.18)),
        max(1, int(len(frames) * 0.82)),
        num=min(18, max(1, len(frames) // 4)),
        dtype=int,
    )
    peak_counts: list[int] = []
    outer_radii: list[float] = []
    for idx in sample_indices:
        alpha = _estimate_alpha(frames[int(idx)], target)
        count, outer_radius = _count_radial_edges(alpha)
        peak_counts.append(count)
        if outer_radius is not None:
            outer_radii.append(outer_radius)

    peak_frame_ratio = float(np.mean([count >= 2 for count in peak_counts])) if peak_counts else 0.0
    mean_peak_count = float(np.mean(peak_counts)) if peak_counts else 0.0
    expanding_edge_score = 0.0
    if len(outer_radii) >= 4:
        radii = np.asarray(outer_radii, dtype=float)
        span = float(np.max(radii) - np.min(radii))
        positive_steps = float(np.mean(np.diff(radii) >= -4.0))
        expanding_edge_score = _clip01((span / max(target.shape[:2])) * 2.0) * positive_steps

    return RevealMetrics(
        start_luma_mean=start_luma,
        end_ssim=end_ssim,
        end_color_similarity=end_color_similarity,
        max_luma_drop=max_drop,
        peak_frame_ratio=peak_frame_ratio,
        mean_peak_count=mean_peak_count,
        expanding_edge_score=expanding_edge_score,
    )


def score_video(
    video_path: Path, input_image_path: Path, pass_threshold: float = PASS_THRESHOLD
) -> ScoreResult:
    probe = _probe_video(video_path)
    empty_components = {
        "format": 0.0,
        "start_black": 0.0,
        "end_match": 0.0,
        "brightness_continuity": 0.0,
        "ripple_pattern": 0.0,
    }
    if not probe.exists:
        return ScoreResult(
            0.0, False, "missing output video", "missing_output", probe, None, empty_components
        )
    if not probe.readable:
        return ScoreResult(
            0.0,
            False,
            "unreadable output video",
            "unreadable_output",
            probe,
            None,
            empty_components,
        )

    frames = _read_frames(video_path)
    if not frames:
        return ScoreResult(
            0.0, False, "no decodable frames", "no_frames", probe, None, empty_components
        )
    target = _read_target_image(input_image_path, (probe.width, probe.height))
    metrics = _analyze(frames, target)

    codec_score = 1.0 if probe.codec_name == "h264" else 0.0
    ext_score = 1.0 if probe.container_ext == ".mp4" else 0.0
    fps_score = _clip01(1.0 - abs(probe.fps - 30.0) / 2.0)
    duration_score = (
        1.0
        if 1.0 <= probe.duration_sec <= 3.05
        else _clip01(1.0 - min(abs(probe.duration_sec - 2.0), 4.0) / 4.0)
    )
    format_score = 0.25 * codec_score + 0.2 * ext_score + 0.3 * fps_score + 0.25 * duration_score

    start_score = _clip01((28.0 - metrics.start_luma_mean) / 28.0)
    end_structure_score = _clip01((metrics.end_ssim - 0.90) / 0.08)
    end_color_score = _clip01((metrics.end_color_similarity - 0.92) / 0.08)
    end_score = 0.65 * end_structure_score + 0.35 * end_color_score
    continuity_score = _clip01((18.0 - metrics.max_luma_drop) / 18.0)
    ripple_score = (
        0.45 * metrics.peak_frame_ratio
        + 0.35 * _clip01(metrics.mean_peak_count / 2.5)
        + 0.20 * metrics.expanding_edge_score
    )

    components = {
        "format": format_score,
        "start_black": start_score,
        "end_match": end_score,
        "brightness_continuity": continuity_score,
        "ripple_pattern": ripple_score,
    }
    score = (
        0.20 * format_score
        + 0.20 * start_score
        + 0.25 * end_score
        + 0.15 * continuity_score
        + 0.20 * ripple_score
    )

    if metrics.end_ssim < 0.90 or metrics.end_color_similarity < 0.92:
        score = min(score, 0.65)
    if metrics.start_luma_mean > 45:
        score = min(score, 0.65)
    if ripple_score < 0.35:
        # A plain fade can satisfy format/start/end checks but misses the core task.
        # Treat missing ripple fronts as a task-level failure rather than partial credit.
        score = 0.0
    if codec_score <= 0 or fps_score < 0.5 or duration_score < 0.5:
        score = min(score, 0.70)

    score = _clip01(score)
    passed = score >= pass_threshold
    reason = "passed" if passed else "failed threshold or one or more required pattern checks"
    return ScoreResult(score, passed, reason, None, probe, metrics, components)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("input_image", type=Path)
    args = parser.parse_args()
    result = score_video(args.video, args.input_image)
    print(json.dumps(result.to_dict(), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
