from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


SAMPLE_RATE = 48000
FRAME_RATE = 24
FCPXML_VERSION = "1.10"
EXPECTED_ANGLES = {
    "A": {"angle": "CAM_A_wide", "offset_frames": 0, "fallback": "scratch_audio_cross_correlation"},
    "B": {"angle": "CAM_B_close", "offset_frames": -3, "fallback": "scratch_audio_cross_correlation"},
    "C": {"angle": "CAM_C_profile", "offset_frames": 7, "fallback": "jam_timecode_from_continuity_log"},
}
REQUIRED_OUTPUTS = [
    "handoff/scene07_take03_multicam.fcpxml",
    "reports/sync_report.json",
    "sync_check/CAM_A_synced.mp4",
    "sync_check/CAM_B_synced.mp4",
    "sync_check/CAM_C_synced.mp4",
]


def _run_json(cmd: list[str]) -> dict[str, Any]:
    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(result.stdout)


def _ffprobe(path: Path) -> dict[str, Any]:
    return _run_json(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])


def _extract_audio_to_wav(path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "wav", str(tmp)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return tmp


def _detect_clap_sample(path: Path) -> int:
    wav_path = _extract_audio_to_wav(path)
    with wave.open(str(wav_path), "rb") as wav:
        data = wav.readframes(wav.getnframes())
    best = 0
    best_val = 0
    for i in range(0, len(data), 2):
        value = abs(int.from_bytes(data[i : i + 2], "little", signed=True))
        if value > best_val:
            best_val = value
            best = i // 2
    return best


def _read_wav_window(path: Path, start: int, frames: int) -> list[float]:
    with wave.open(str(path), "rb") as wav:
        wav.setpos(max(0, start))
        data = wav.readframes(frames)
    return [int.from_bytes(data[i : i + 2], "little", signed=True) / 32768.0 for i in range(0, len(data), 2)]


def _correlation(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    den_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
    if den_a == 0.0 or den_b == 0.0:
        return 0.0
    return num / (den_a * den_b)


def _video_frame_md5(path: Path, frames: int = 6) -> str:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-frames:v", str(frames), "-f", "framemd5", "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    lines = [line for line in result.stdout.decode().splitlines() if line and not line.startswith("#")]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str, *, score: int = 0) -> dict[str, Any]:
    return {
        "score": score,
        "final_score": 0.0,
        "passed": False,
        "hard_failures": [message],
        "details": [],
    }


def score_submission(submission_dir: Path, reference_dir: Path) -> dict[str, Any]:
    submission = submission_dir.resolve()
    reference = reference_dir.resolve()
    targets_path = reference / "objective_targets.json"
    if not targets_path.exists():
        return _fail(f"missing evaluator reference: {targets_path}")
    targets = json.loads(targets_path.read_text())
    reference_media = reference / "media"

    score = 0
    hard_failures: list[str] = []
    details: list[str] = []

    missing = [rel for rel in REQUIRED_OUTPUTS if not (submission / rel).exists()]
    if missing:
        return _fail(f"missing required outputs: {missing}")
    score += 10

    try:
        root = ET.parse(submission / "handoff" / "scene07_take03_multicam.fcpxml").getroot()
    except Exception as exc:
        return _fail(f"failed to parse FCPXML: {exc}")

    if root.tag == "fcpxml" and root.get("version") == FCPXML_VERSION:
        score += 8
    else:
        hard_failures.append("FCPXML root/version mismatch")

    media = root.find("./resources/media[@id='r10']")
    resource_multicam = None if media is None else media.find("./multicam")
    multicam_angles = [] if resource_multicam is None else resource_multicam.findall("./mc-angle")
    multicams = root.findall("./library/event/project/sequence/spine/mc-clip")
    assets = {item.get("id"): item for item in root.findall("./resources/asset")}
    angle_names = {item.get("angleID") for item in multicam_angles}
    expected_angle_names = {item["angle"] for item in EXPECTED_ANGLES.values()}
    if (
        media is None
        or resource_multicam is None
        or len(multicams) != 1
        or multicams[0].get("name") != "scene07_take03_multicam"
        or multicams[0].get("ref") != "r10"
        or angle_names != expected_angle_names
    ):
        hard_failures.append("FCPXML must contain one media-resource multicam and one timeline mc-clip structure")
        sources = []
    else:
        sources = multicams[0].findall("./mc-source")

    by_camera = {item.get("cameraID"): item for item in sources}
    xml_ok = True
    for camera_id, expected in EXPECTED_ANGLES.items():
        item = by_camera.get(camera_id)
        if item is None:
            xml_ok = False
            continue
        if item.get("angleID") != expected["angle"] or item.get("scene") != "07" or item.get("take") != "03":
            xml_ok = False
        asset = assets.get(item.get("ref"))
        if asset is None or not asset.get("name", "").startswith(f"CAM_{camera_id}_"):
            xml_ok = False
        if item.get("offset") != f"{expected['offset_frames']}/24s":
            xml_ok = False
    if xml_ok and len(sources) == 3:
        score += 22
    else:
        hard_failures.append("Multicam source metadata or offsets are wrong")

    try:
        report = json.loads((submission / "reports" / "sync_report.json").read_text())
    except Exception as exc:
        return _fail(f"failed to parse sync_report.json: {exc}", score=score)

    angle_report = {item.get("camera_id"): item for item in report.get("angles", [])}
    report_ok = True
    for camera_id, expected in EXPECTED_ANGLES.items():
        item = angle_report.get(camera_id)
        if not item:
            report_ok = False
            continue
        if item.get("angle_name") != expected["angle"]:
            report_ok = False
        if int(item.get("applied_offset_frames", 999)) != expected["offset_frames"]:
            report_ok = False
        if item.get("fallback_method") != expected["fallback"]:
            report_ok = False
        hidden_proxy_sample = int(targets["expected_proxy_clap_sample"])
        if abs(int(item.get("final_proxy_clap_sample", 0)) - hidden_proxy_sample) > 120:
            report_ok = False
        if int(item.get("applied_offset_frames", 999)) != int(targets["offset_frames"][camera_id]):
            report_ok = False
        expected_scratch = targets["scratch_clap_samples"][camera_id]
        observed_scratch = item.get("scratch_clap_sample")
        if expected_scratch is None:
            if observed_scratch is not None:
                report_ok = False
        elif abs(int(observed_scratch) - int(expected_scratch)) > 120:
            report_ok = False
        if abs(float(item.get("drift_ppm", 999)) - float(targets["drift_ppm"][camera_id])) > 1.0:
            report_ok = False
    if abs(int(report.get("external_clap_sample", 0)) - int(targets["external_clap_sample"])) > 120:
        report_ok = False
    for filename in [f"CAM_{camera_id}_synced.mp4" for camera_id in EXPECTED_ANGLES]:
        proxy = submission / "sync_check" / filename
        if not proxy.exists() or report.get("proxy_hashes", {}).get(filename) != _sha256(proxy):
            report_ok = False
    if report_ok:
        score += 25
    else:
        details.append("sync_report.json fields are missing or outside tolerance")

    probe_ok = True
    sync_ok = True
    provenance_ok = True
    external_wav = reference_media / "audio" / "scene07_take03_prod.wav"
    prod_window = _read_wav_window(external_wav, int(targets["external_clap_sample"]) - 3000, 7000)
    for camera_id in EXPECTED_ANGLES:
        proxy = submission / "sync_check" / f"CAM_{camera_id}_synced.mp4"
        try:
            info = _ffprobe(proxy)
        except Exception:
            probe_ok = False
            sync_ok = False
            provenance_ok = False
            continue
        streams = info.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
        if video.get("avg_frame_rate") not in {"24/1", "24000/1000"}:
            probe_ok = False
        if int(audio.get("sample_rate", 0)) != SAMPLE_RATE:
            probe_ok = False
        try:
            clap = _detect_clap_sample(proxy)
            proxy_wav = _extract_audio_to_wav(proxy)
        except Exception:
            sync_ok = False
            continue
        if abs(clap - int(targets["expected_proxy_clap_sample"])) > 160:
            sync_ok = False
        proxy_window = _read_wav_window(proxy_wav, clap - 3000, 7000)
        signed_corr = _correlation(prod_window, proxy_window)
        envelope_corr = _correlation([abs(x) for x in prod_window], [abs(x) for x in proxy_window])
        if max(abs(signed_corr), envelope_corr) < 0.45:
            sync_ok = False
        source_video = reference_media / "camera" / f"CAM_{camera_id}_scene07_take03_proxy.mp4"
        try:
            if _video_frame_md5(source_video) != _video_frame_md5(proxy):
                provenance_ok = False
        except Exception:
            provenance_ok = False

    if probe_ok:
        score += 13
    else:
        hard_failures.append("Proxy media frame rate or sample rate mismatch")
    if sync_ok:
        score += 20
    else:
        hard_failures.append("Rendered proxy production-audio clap is not aligned")
    if not provenance_ok:
        hard_failures.append("Rendered proxy video does not match the declared camera source")

    source_hashes = report.get("source_hashes", {})
    if all(source_hashes.get(name) == digest for name, digest in targets["source_hashes"].items()):
        score += 2
    else:
        details.append("Source hash report mismatch")

    passed = score >= 85 and not hard_failures
    return {
        "score": min(100, score),
        "final_score": 1.0 if passed else 0.0,
        "passed": passed,
        "hard_failures": hard_failures,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()
    result = score_submission(Path(args.submission_dir), Path(args.reference_dir))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
