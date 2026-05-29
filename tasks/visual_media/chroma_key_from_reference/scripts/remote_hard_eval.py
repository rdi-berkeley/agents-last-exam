from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROI_INPUT_GATE_THRESHOLD = 0.3
SOFT_GATE_PASS_THRESHOLD = 0.5
HARD_QUALITY_FULL_FRAME_WEIGHT = 0.3
HARD_QUALITY_ROI_EDGE_WEIGHT = 0.7


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_breakpoints(point_file: Path, default_sample_count: int) -> tuple[list[dict], int]:
    payload = read_json(point_file)
    sample_count = max(1, int(payload.get("sample_count", default_sample_count)))
    points: list[dict] = []
    raw_points = payload.get("breakpoints", payload.get("points", []))
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        try:
            time_sec = float(item.get("time_sec", 0.0))
        except Exception:
            continue
        if time_sec < 0:
            continue
        point = dict(item)
        point["time_sec"] = round(time_sec, 6)
        points.append(point)
    points = sorted(points, key=lambda p: p["time_sec"])
    return points[:sample_count], sample_count


def probe_video(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video size: {path}")
    return width, height, fps, frame_count


def default_sample_times(duration: float, sample_count: int) -> list[float]:
    sample_count = max(1, sample_count)
    if duration <= 0:
        return [float(i) for i in range(sample_count)]
    if sample_count == 1:
        return [round(duration * 0.5, 3)]
    start = duration * 0.1
    end = duration * 0.9
    step = (end - start) / (sample_count - 1)
    return [round(start + i * step, 3) for i in range(sample_count)]


def read_frame_at(video_path: Path, time_sec: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        max_time = max(0.0, (frame_count - 1) / max(fps, 1e-6))
        time_sec = min(max(0.0, time_sec), max_time)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(max(0.0, time_sec) * fps)))
        ok, frame = cap.read()
    if not ok and frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame at {time_sec}s: {video_path}")
    return frame


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


def legacy_bbox_to_norm(bbox: list[float], width: int, height: int) -> list[float]:
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


def resolve_dual_bboxes(point: dict, input_width: int, input_height: int, output_width: int, output_height: int):
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
    return input_xyxy, output_xyxy, xyxy_to_norm(input_xyxy, input_width, input_height), xyxy_to_norm(output_xyxy, output_width, output_height)


def crop(frame: np.ndarray, xyxy: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return frame[y1:y2, x1:x2]


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


def score_roi_input_foreground(output_roi: np.ndarray, input_roi: np.ndarray, reference_roi: np.ndarray) -> float:
    input_roi = resize_like(input_roi, (output_roi.shape[0], output_roi.shape[1]))
    reference_roi = resize_like(reference_roi, (output_roi.shape[0], output_roi.shape[1]))
    fg_mask = foreground_mask_from_input_reference(input_roi, reference_roi)
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
        HARD_QUALITY_FULL_FRAME_WEIGHT * float(full_frame_edge_iou)
        + HARD_QUALITY_ROI_EDGE_WEIGHT * float(roi_edge_iou),
        0.0,
        1.0,
    )


def write_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"failed to write png: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote hard eval for chroma-key tasks")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--reference-video", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--breakpoint-json", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    args = parser.parse_args()

    input_video = Path(args.input_video)
    reference_video = Path(args.reference_video)
    output_video = Path(args.output_video)
    point_file = Path(args.breakpoint_json)
    report_json = Path(args.report_json)
    frames_dir = Path(args.frames_dir)

    if not input_video.exists():
        raise FileNotFoundError(f"missing input video: {input_video}")
    if not reference_video.exists():
        raise FileNotFoundError(f"missing reference video: {reference_video}")
    if not output_video.exists():
        raise FileNotFoundError(f"missing output video: {output_video}")

    input_w, input_h, _, _ = probe_video(input_video)
    output_w, output_h, fps, frame_count = probe_video(output_video)
    duration = max(0.0, (frame_count - 1) / max(fps, 1e-6)) if frame_count > 0 else 0.0

    points, sample_count = load_breakpoints(point_file, args.sample_count)
    if not points:
        points = [{"index": i, "time_sec": t} for i, t in enumerate(default_sample_times(duration, sample_count))]

    details = []
    frame_pairs = []
    total_hard_quality = 0.0
    for idx, point in enumerate(points):
        time_sec = float(point["time_sec"])
        identifier = str(point.get("point_id") or f"{idx:03d}_{int(round(time_sec * 1000))}ms")
        input_bbox_xyxy, output_bbox_xyxy, input_norm, output_norm = resolve_dual_bboxes(
            point, input_w, input_h, output_w, output_h
        )

        in_frame = read_frame_at(input_video, time_sec)
        ref_frame = read_frame_at(reference_video, time_sec)
        out_frame = read_frame_at(output_video, time_sec)

        input_full_path = frames_dir / f"input_{identifier}.png"
        ref_full_path = frames_dir / f"reference_{identifier}.png"
        out_full_path = frames_dir / f"output_{identifier}.png"
        write_png(input_full_path, in_frame)
        write_png(ref_full_path, ref_frame)
        write_png(out_full_path, out_frame)

        full_frame_edge_iou = edge_iou_score(out_frame, ref_frame)
        out_roi = crop(out_frame, output_bbox_xyxy)
        ref_roi = crop(ref_frame, output_bbox_xyxy)
        in_roi = crop(in_frame, input_bbox_xyxy)

        roi_edge_iou = edge_iou_score(out_roi, ref_roi)
        roi_input = score_roi_input_foreground(out_roi, in_roi, ref_roi)
        hard_quality_frame_score = compute_hard_quality_frame_score(full_frame_edge_iou, roi_edge_iou)
        total_hard_quality += hard_quality_frame_score

        details.append(
            {
                "identifier": identifier,
                "index": int(point.get("index", idx)),
                "time_sec": time_sec,
                "input_bbox_norm": input_norm,
                "output_bbox_norm": output_norm,
                "full_frame_edge_iou": full_frame_edge_iou,
                "roi_edge_iou": roi_edge_iou,
                "roi_input_cv": roi_input,
                "roi_input_gate_passed": roi_input >= ROI_INPUT_GATE_THRESHOLD,
                "soft_gate_threshold": SOFT_GATE_PASS_THRESHOLD,
                "hard_quality_frame_score": hard_quality_frame_score,
            }
        )
        frame_pairs.append(
            {
                "identifier": identifier,
                "index": int(point.get("index", idx)),
                "time_sec": time_sec,
                "input_frame_path": str(input_full_path),
                "reference_frame_path": str(ref_full_path),
                "output_frame_path": str(out_full_path),
            }
        )

    hard_score = float(total_hard_quality / max(1, len(details)))
    payload = {
        "summary": {
            "hard_score": hard_score,
            "sample_count": len(details),
            "input_video": str(input_video),
            "reference_video": str(reference_video),
            "output_video": str(output_video),
            "weights": {
                "hard_quality_full_frame_edge_iou": HARD_QUALITY_FULL_FRAME_WEIGHT,
                "hard_quality_roi_edge_iou": HARD_QUALITY_ROI_EDGE_WEIGHT,
            },
            "thresholds": {
                "roi_input_cv_gate": ROI_INPUT_GATE_THRESHOLD,
                "soft_gate": SOFT_GATE_PASS_THRESHOLD,
            },
        },
        "details": details,
        "frame_pairs": frame_pairs,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "score": hard_score,
                "report_path": str(report_json),
                "metrics": {
                    "hard_quality_score": hard_score,
                    "sample_count": len(details),
                },
                "frame_pairs": frame_pairs,
                "details": details,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
