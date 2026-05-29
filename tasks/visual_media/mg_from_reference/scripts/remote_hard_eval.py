from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_sample_times(point_path: Path, default_count: int) -> list[float]:
    payload = read_json(point_path)
    sample_count = max(1, int(payload.get("sample_count", default_count)))
    raw = payload.get("breakpoints", payload.get("points", []))
    points: list[float] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            time_sec = float(item.get("time_sec", 0.0))
        except Exception:
            continue
        if time_sec >= 0:
            points.append(round(time_sec, 6))
    points = sorted(set(points))
    return points[:sample_count]


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


def luma_similarity_score(image_a: np.ndarray, image_b: np.ndarray) -> float:
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mae = float(np.mean(np.abs(gray_a - gray_b)) / 255.0)
    return clamp(1.0 - mae, 0.0, 1.0)


def score_frame(output_frame: np.ndarray, reference_frame: np.ndarray) -> float:
    edge_iou = edge_iou_score(output_frame, reference_frame)
    luma_similarity = luma_similarity_score(output_frame, reference_frame)
    return clamp(0.6 * edge_iou + 0.4 * luma_similarity, 0.0, 1.0)


def save_frame(frame: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        raise RuntimeError(f"failed to write frame: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote hard eval for MG tasks")
    parser.add_argument("--reference-video", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--breakpoint-json", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    args = parser.parse_args()

    reference_video = Path(args.reference_video)
    output_video = Path(args.output_video)
    point_path = Path(args.breakpoint_json)
    report_json = Path(args.report_json)
    frames_dir = Path(args.frames_dir)

    if not reference_video.exists():
        raise FileNotFoundError(f"missing reference video: {reference_video}")
    if not output_video.exists():
        raise FileNotFoundError(f"missing output video: {output_video}")

    _, _, fps, frame_count = probe_video(reference_video)
    duration = max(0.0, (frame_count - 1) / max(fps, 1e-6)) if frame_count > 0 else 0.0
    sample_times = load_sample_times(point_path, args.sample_count)
    if not sample_times:
        sample_times = default_sample_times(duration, args.sample_count)

    details = []
    frame_pairs = []
    total = 0.0
    for idx, time_sec in enumerate(sample_times):
        ref_frame = read_frame_at(reference_video, time_sec)
        out_frame = read_frame_at(output_video, time_sec)
        score = score_frame(out_frame, ref_frame)
        total += score

        ref_frame_path = frames_dir / f"reference_{idx:03d}.png"
        out_frame_path = frames_dir / f"output_{idx:03d}.png"
        save_frame(ref_frame, ref_frame_path)
        save_frame(out_frame, out_frame_path)

        details.append(
            {
                "index": idx,
                "time_sec": float(time_sec),
                "score": float(score),
                "metrics": {
                    "edge_iou": float(edge_iou_score(out_frame, ref_frame)),
                    "luma_similarity": float(luma_similarity_score(out_frame, ref_frame)),
                },
            }
        )
        frame_pairs.append(
            {
                "index": idx,
                "time_sec": float(time_sec),
                "reference_frame_path": str(ref_frame_path),
                "output_frame_path": str(out_frame_path),
            }
        )

    final_score = float(total / max(1, len(details)))
    report_payload = {
        "summary": {
            "final_score": final_score,
            "sample_count": len(details),
            "reference_video": str(reference_video),
            "output_video": str(output_video),
        },
        "details": details,
        "frame_pairs": frame_pairs,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    result_payload = {
        "score": final_score,
        "report_path": str(report_json),
        "metrics": {
            "sample_count": len(details),
            "reference_video": str(reference_video),
            "output_video": str(output_video),
        },
        "frame_pairs": frame_pairs,
    }
    print(json.dumps(result_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
