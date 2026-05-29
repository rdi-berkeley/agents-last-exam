"""Local evaluator for visual_media/VCV_modular_sound_design."""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import subprocess
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

REQUIRED_FILES = ("beat.vcv", "beat.wav", "kick.wav", "snare.wav", "hihat.wav")
ALLOWED_PLUGINS = {"Core", "Fundamental", "VCV-Recorder"}
EXPECTED_RECORDER_FILES = {"beat.wav", "kick.wav", "snare.wav", "hihat.wav"}
EXPECTED_QUARTER_SEC = 60.0 / 110.0


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = float(self.score)
        payload["passed"] = bool(self.passed)
        return payload


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _range_score(value: float, *, low: float, high: float, tolerance: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return _clip01(1.0 - (low - value) / tolerance)
    return _clip01(1.0 - (value - high) / tolerance)


def _at_least_score(value: float, target: float, tolerance: float) -> float:
    if value >= target:
        return 1.0
    return _clip01(1.0 - (target - value) / tolerance)


def _at_most_score(value: float, target: float, tolerance: float) -> float:
    if value <= target:
        return 1.0
    return _clip01(1.0 - (value - target) / tolerance)


def _basename_from_path(raw: str | None) -> str:
    if not raw:
        return ""
    text = raw.replace("\\", "/")
    return text.rsplit("/", 1)[-1].lower()


def _mean_power_spectrum(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.abs(librosa.stft(audio, n_fft=4096, hop_length=1024))
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=4096)
    mean_power = spectrum.mean(axis=1)
    total = float(mean_power.sum())
    if total <= 0:
        return freqs, mean_power
    return freqs, mean_power / total


def _band_ratio(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(power[mask].sum())


def _peak_band_ratio(freqs: np.ndarray, power: np.ndarray, center: float, width: float = 25.0) -> float:
    return _band_ratio(freqs, power, center - width, center + width)


def _decode_vcv_patch(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()

    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        pass

    try:
        decompressed = subprocess.run(
            ["zstd", "-q", "-d", "-c", str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except FileNotFoundError as exc:
        raise RuntimeError("`zstd` CLI is required to decode Rack 2 `.vcv` files") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to decompress beat.vcv: {exc.stderr.decode('utf-8', 'ignore')}") from exc

    try:
        return json.loads(decompressed.decode("utf-8"))
    except Exception:
        pass

    with tarfile.open(fileobj=io.BytesIO(decompressed), mode="r:") as archive:
        for member_name in ("./patch.json", "patch.json"):
            member = archive.extractfile(member_name)
            if member is None:
                continue
            return json.load(member)

    raise RuntimeError("beat.vcv did not contain a readable patch.json payload")


def _read_audio(path: Path) -> dict[str, Any]:
    data, sample_rate = sf.read(path, always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    duration = float(len(mono) / sample_rate) if sample_rate else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sample_rate)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=mono)))
    freqs, power = _mean_power_spectrum(mono, sample_rate)
    return {
        "audio": mono,
        "sample_rate": int(sample_rate),
        "channels": int(data.shape[1]),
        "duration": duration,
        "peak": peak,
        "rms": rms,
        "centroid": centroid,
        "flatness": flatness,
        "freqs": freqs,
        "power": power,
    }


def _rms_peak_times(audio: np.ndarray, sample_rate: int, *, threshold_ratio: float, min_gap: float) -> list[float]:
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=256)[0]
    times = librosa.times_like(rms, sr=sample_rate, hop_length=256)
    threshold = float(np.max(rms) * threshold_ratio) if len(rms) else 0.0
    peak_indices: list[int] = []
    for index in range(1, len(rms) - 1):
        if rms[index] <= threshold:
            continue
        if rms[index] <= rms[index - 1] or rms[index] < rms[index + 1]:
            continue
        if peak_indices and float(times[index] - times[peak_indices[-1]]) < min_gap:
            continue
        peak_indices.append(index)
    return [float(times[index]) for index in peak_indices]


def _onset_times(audio: np.ndarray, sample_rate: int) -> list[float]:
    values = librosa.onset.onset_detect(
        y=audio,
        sr=sample_rate,
        units="time",
        backtrack=False,
        pre_max=20,
        post_max=20,
        pre_avg=100,
        post_avg=100,
        wait=10,
        delta=0.2,
    )
    return [float(value) for value in values]


def _median_interval(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return None
    return float(np.median(diffs))


def _decay_score(audio: np.ndarray, sample_rate: int, peaks: list[float]) -> float:
    if not peaks:
        return 0.0
    passes = 0
    checked = 0
    window = max(1, int(0.03 * sample_rate))
    horizon = int(0.36 * sample_rate)
    for peak in peaks[:8]:
        start = int(peak * sample_rate)
        if start >= len(audio):
            continue
        segment = np.abs(audio[start : min(len(audio), start + horizon)])
        if len(segment) < window * 4:
            continue
        envelope = []
        for offset in range(0, len(segment) - window + 1, window):
            chunk = segment[offset : offset + window]
            envelope.append(float(np.sqrt(np.mean(chunk**2))))
        if len(envelope) < 4:
            continue
        checked += 1
        head = envelope[0]
        tail = max(envelope[-1], 1e-6)
        downward_steps = sum(1 for a, b in zip(envelope, envelope[1:]) if b <= a)
        if head / tail >= 1.8 and downward_steps >= max(2, len(envelope) - 3):
            passes += 1
    if checked == 0:
        return 0.0
    return float(passes / checked)


def _score_patch_structure(patch: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    modules = patch.get("modules", [])
    plugins = {str(module.get("plugin", "")) for module in modules}
    model_counts: dict[str, int] = {}
    recorder_basenames: set[str] = set()
    seq3_step_count = None
    gate_module_count = 0

    for module in modules:
        plugin = str(module.get("plugin", ""))
        model = str(module.get("model", ""))
        key = f"{plugin}/{model}"
        model_counts[key] = model_counts.get(key, 0) + 1
        if model == "Recorder":
            basename = _basename_from_path((module.get("data") or {}).get("path"))
            if basename:
                recorder_basenames.add(basename)
        if model == "SEQ3":
            for param in module.get("params", []):
                if param.get("id") == 3:
                    try:
                        seq3_step_count = float(param.get("value"))
                    except Exception:
                        seq3_step_count = None
        if model == "Gates":
            gate_module_count += 1

    allowed_plugins_score = 1.0 if plugins.issubset(ALLOWED_PLUGINS) else 0.0
    count_checks = {
        "noise": _clip01(model_counts.get("Fundamental/Noise", 0) / 2.0),
        "adsr": _clip01(model_counts.get("Fundamental/ADSR", 0) / 3.0),
        "vco": _clip01(model_counts.get("Fundamental/VCO", 0) / 2.0),
        "vcf": _clip01(model_counts.get("Fundamental/VCF", 0) / 1.0),
        "seq3": _clip01(model_counts.get("Fundamental/SEQ3", 0) / 1.0),
        "recorders": _clip01(model_counts.get("VCV-Recorder/Recorder", 0) / 4.0),
        "gates": _clip01(gate_module_count / 4.0),
    }
    recorder_names_score = float(len(recorder_basenames & EXPECTED_RECORDER_FILES) / len(EXPECTED_RECORDER_FILES))
    seq3_length_score = 0.0
    if seq3_step_count is not None:
        seq3_length_score = _range_score(seq3_step_count, low=4.0, high=4.0, tolerance=1.0)

    components = {
        "allowed_plugins": allowed_plugins_score,
        "noise_modules": count_checks["noise"],
        "adsr_modules": count_checks["adsr"],
        "vco_modules": count_checks["vco"],
        "vcf_modules": count_checks["vcf"],
        "seq3_presence": count_checks["seq3"],
        "seq3_step_count": seq3_length_score,
        "gate_modules": count_checks["gates"],
        "recorders": count_checks["recorders"],
        "recorder_filenames": recorder_names_score,
    }
    score = float(statistics.mean(components.values()))
    details = {
        "plugins": sorted(plugins),
        "model_counts": model_counts,
        "recorder_basenames": sorted(recorder_basenames),
        "seq3_step_count": seq3_step_count,
        "components": components,
    }
    return score, details


def score_output_dir(output_dir: str | Path) -> ScoreResult:
    output_dir = Path(output_dir)
    details: dict[str, Any] = {"output_dir": str(output_dir)}

    missing = [name for name in REQUIRED_FILES if not (output_dir / name).exists()]
    if missing:
        return ScoreResult(0.0, False, f"missing required files: {', '.join(missing)}", {"missing_files": missing})

    patch_path = output_dir / "beat.vcv"
    try:
        patch = _decode_vcv_patch(patch_path)
    except Exception as exc:
        return ScoreResult(0.0, False, f"failed to parse beat.vcv: {exc}", {"patch_error": str(exc)})

    patch_plugins = {
        str(module.get("plugin", ""))
        for module in patch.get("modules", [])
    }
    if not patch_plugins.issubset(ALLOWED_PLUGINS):
        return ScoreResult(
            0.0,
            False,
            "beat.vcv used disallowed plugins",
            {"plugins": sorted(patch_plugins)},
        )

    audio_info: dict[str, dict[str, Any]] = {}
    sample_rates = set()
    durations = []
    for filename in ("kick.wav", "snare.wav", "hihat.wav", "beat.wav"):
        try:
            info = _read_audio(output_dir / filename)
        except Exception as exc:
            return ScoreResult(0.0, False, f"failed to read {filename}: {exc}", {"audio_error": filename})
        if info["duration"] <= 20.0:
            return ScoreResult(
                0.0,
                False,
                f"{filename} was not longer than 20 seconds",
                {"filename": filename, "duration": info["duration"]},
            )
        audio_info[filename] = info
        sample_rates.add(info["sample_rate"])
        durations.append(info["duration"])

    if len(sample_rates) != 1:
        return ScoreResult(
            0.0,
            False,
            "WAV exports used inconsistent sample rates",
            {"sample_rates": sorted(sample_rates)},
        )

    structure_score, structure_details = _score_patch_structure(patch)
    details["patch"] = structure_details

    kick = audio_info["kick.wav"]
    snare = audio_info["snare.wav"]
    hihat = audio_info["hihat.wav"]
    beat = audio_info["beat.wav"]

    kick_low_ratio = _band_ratio(kick["freqs"], kick["power"], 0, 200)
    kick_high_ratio = _band_ratio(kick["freqs"], kick["power"], 1000, 12000)
    snare_low_tone = _peak_band_ratio(snare["freqs"], snare["power"], 140)
    snare_mid_tone = _peak_band_ratio(snare["freqs"], snare["power"], 300)
    snare_noise_ratio = _band_ratio(snare["freqs"], snare["power"], 1000, 12000)
    hihat_low_ratio = _band_ratio(hihat["freqs"], hihat["power"], 0, 500)
    hihat_high_ratio = _band_ratio(hihat["freqs"], hihat["power"], 4000, 12000)

    timbre_components = {
        "kick_low_end": _at_least_score(kick_low_ratio, 0.55, 0.20),
        "kick_limited_highs": _at_most_score(kick_high_ratio, 0.08, 0.08),
        "kick_centroid": _at_most_score(kick["centroid"], 2000.0, 1200.0),
        "snare_140hz": _at_least_score(snare_low_tone, 0.01, 0.02),
        "snare_300hz": _at_least_score(snare_mid_tone, 0.01, 0.02),
        "snare_noise": _at_least_score(snare_noise_ratio, 0.20, 0.15),
        "snare_centroid": _range_score(snare["centroid"], low=700.0, high=4500.0, tolerance=600.0),
        "hihat_highs": _at_least_score(hihat_high_ratio, 0.18, 0.10),
        "hihat_cutoff": _at_most_score(hihat_low_ratio, 0.10, 0.08),
        "hihat_centroid": _at_least_score(hihat["centroid"], 4000.0, 2500.0),
        "hihat_flatness": _at_least_score(hihat["flatness"], 0.25, 0.15),
    }
    timbre_score = float(statistics.mean(timbre_components.values()))
    details["timbre"] = {
        "components": timbre_components,
        "kick_low_ratio": kick_low_ratio,
        "kick_high_ratio": kick_high_ratio,
        "snare_140_ratio": snare_low_tone,
        "snare_300_ratio": snare_mid_tone,
        "snare_noise_ratio": snare_noise_ratio,
        "hihat_low_ratio": hihat_low_ratio,
        "hihat_high_ratio": hihat_high_ratio,
    }

    kick_peaks = _rms_peak_times(kick["audio"], kick["sample_rate"], threshold_ratio=0.03, min_gap=0.25)
    snare_onsets = _onset_times(snare["audio"], snare["sample_rate"])
    hihat_onsets = _onset_times(hihat["audio"], hihat["sample_rate"])

    hihat_interval = _median_interval(hihat_onsets)
    hihat_expected_count = hihat["duration"] / EXPECTED_QUARTER_SEC
    rhythm_components = {
        "kick_repeating_hits": _at_least_score(len(kick_peaks), max(12.0, kick["duration"] / 1.5), 10.0),
        "snare_repeating_hits": _at_least_score(len(snare_onsets), max(8.0, snare["duration"] / 4.0), 6.0),
        "hihat_count": _range_score(len(hihat_onsets), low=0.6 * hihat_expected_count, high=1.4 * hihat_expected_count, tolerance=10.0),
        "hihat_interval": _range_score(
            hihat_interval if hihat_interval is not None else 0.0,
            low=0.45,
            high=0.65,
            tolerance=0.12,
        ),
    }
    rhythm_score = float(statistics.mean(rhythm_components.values()))
    details["rhythm"] = {
        "components": rhythm_components,
        "kick_peak_count": len(kick_peaks),
        "snare_onset_count": len(snare_onsets),
        "hihat_onset_count": len(hihat_onsets),
        "hihat_interval": hihat_interval,
    }

    decay_components = {
        "kick_decay": _decay_score(kick["audio"], kick["sample_rate"], kick_peaks),
        "snare_decay": _decay_score(snare["audio"], snare["sample_rate"], snare_onsets),
        "hihat_decay": _decay_score(hihat["audio"], hihat["sample_rate"], hihat_onsets),
    }
    decay_score = float(statistics.mean(decay_components.values()))
    details["decay"] = {"components": decay_components}

    stem_duration_median = float(statistics.median(durations[:3]))
    duration_spread = max(durations) - min(durations)
    mix_components = {
        "beat_duration_matches_stems": _at_most_score(abs(beat["duration"] - stem_duration_median), 1.5, 1.5),
        "duration_spread": _at_most_score(duration_spread, 2.5, 2.0),
        "beat_not_clipped": _at_most_score(beat["peak"], 0.98, 0.05),
        "beat_not_silent": _at_least_score(beat["rms"], 0.005, 0.01),
    }
    mix_score = float(statistics.mean(mix_components.values()))
    details["mix"] = {
        "components": mix_components,
        "durations": {name: audio_info[name]["duration"] for name in audio_info},
        "sample_rates": {name: audio_info[name]["sample_rate"] for name in audio_info},
        "peaks": {name: audio_info[name]["peak"] for name in audio_info},
    }

    details["audio_summary"] = {
        name: {
            "duration": info["duration"],
            "sample_rate": info["sample_rate"],
            "channels": info["channels"],
            "peak": info["peak"],
            "rms": info["rms"],
            "centroid": info["centroid"],
            "flatness": info["flatness"],
        }
        for name, info in audio_info.items()
    }

    final_score = (
        0.40 * structure_score
        + 0.20 * rhythm_score
        + 0.25 * timbre_score
        + 0.10 * decay_score
        + 0.05 * mix_score
    )
    details["component_scores"] = {
        "structure": structure_score,
        "rhythm": rhythm_score,
        "timbre": timbre_score,
        "decay": decay_score,
        "mix": mix_score,
    }

    passed = final_score >= 0.85
    reason = "passed" if passed else "score below pass threshold"
    return ScoreResult(float(final_score), passed, reason, details)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score VCV modular sound design outputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-path", default=None,
                        help="Write JSON result to this file instead of stdout")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = score_output_dir(args.output_dir)
        payload = result.to_dict()
    except Exception as exc:
        payload = {"score": 0.0, "passed": False, "reason": str(exc),
                   "details": {}, "error": str(exc)}

    text = json.dumps(payload, ensure_ascii=True, indent=2)

    if args.result_path:
        import os
        os.makedirs(os.path.dirname(args.result_path) or ".", exist_ok=True)
        with open(args.result_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)

    return 0 if payload.get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
