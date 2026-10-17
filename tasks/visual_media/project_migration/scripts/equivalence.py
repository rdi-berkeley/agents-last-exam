from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf


FULL_SCALE = 1 << 32
PCM_CODECS = {"pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be", "pcm_s32le", "pcm_s32be", "flac"}


@dataclass
class PCM:
    samples: np.ndarray
    rate: int
    channels: int
    bits: int
    half_step: int
    sha256: str


def read_pcm(path: Path) -> PCM:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,channel_layout,bits_per_sample,bits_per_raw_sample,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=10,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("Requires one audio stream")
    stream = streams[0]
    if stream.get("codec_name") not in PCM_CODECS:
        raise ValueError("Unsupported codec: only signed integer PCM or lossless FLAC")
    bits = int(stream.get("bits_per_raw_sample") or stream.get("bits_per_sample") or 0)
    channels = int(stream["channels"])
    rate = int(stream["sample_rate"])
    if bits not in (16, 24, 32) or channels not in (1, 2):
        raise ValueError("Supports 16/24/32-bit mono or stereo PCM only")
    if stream.get("channel_layout") not in (None, "unknown", "mono" if channels == 1 else "stereo"):
        raise ValueError("Unsupported channel layout")
    if not 0 < float(stream.get("duration", 0)) <= 60 or not 0 < rate <= 192000:
        raise ValueError("Requires a nonempty at-most-60-second bounded audio window")
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-threads",
            "1",
            "-c:a",
            "pcm_s32le",
            "-f",
            "s32le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=15,
    )
    if len(decoded.stdout) % (4 * channels):
        raise ValueError("Malformed decoded PCM")
    samples = np.frombuffer(decoded.stdout, "<i4").astype(np.int64) * 2
    if not 0 < len(samples) <= 60 * rate * channels:
        raise ValueError("Decoded window is empty or exceeds the bound")
    half_step = 1 << (32 - bits)
    if np.any(samples % (2 * half_step)):
        raise ValueError("Decoded samples disagree with the declared integer precision")
    if not np.any(samples):
        raise ValueError("Silence is not an automatic timbre-equivalence positive")
    if np.any(samples == -FULL_SCALE) or np.any(samples == FULL_SCALE - 2 * half_step):
        raise ValueError("Saturated PCM endpoints have no bounded rounding cell")
    return PCM(
        samples.reshape(-1, channels),
        rate,
        channels,
        bits,
        half_step,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def gain_interval(reference: PCM, candidate: PCM):
    if (reference.rate, reference.channels, reference.samples.shape) != (
        candidate.rate,
        candidate.channels,
        candidate.samples.shape,
    ):
        return None
    lower_num, lower_den = 0, 1
    upper_num, upper_den = None, None
    for source, target in zip(reference.samples.flat, candidate.samples.flat, strict=True):
        source, target = int(source), int(target)
        if source < 0:
            source, target = -source, -target
        if source == 0:
            next_num = abs(target) - candidate.half_step
            next_den = reference.half_step
        else:
            next_num = target - candidate.half_step
            next_den = source + reference.half_step
            upper_candidate_num = target + candidate.half_step
            upper_candidate_den = source - reference.half_step
            if (
                upper_num is None
                or upper_candidate_num * upper_den < upper_num * upper_candidate_den
            ):
                upper_num, upper_den = upper_candidate_num, upper_candidate_den
        if next_num * lower_den > lower_num * next_den:
            lower_num, lower_den = next_num, next_den
        if upper_num is not None and (
            upper_num <= 0 or lower_num * upper_den > upper_num * lower_den
        ):
            return None
    if upper_num is None:
        return None
    return Fraction(lower_num, lower_den), Fraction(upper_num, upper_den)


def compare_float_quantization(reference_path: Path, candidate_path: Path) -> dict:
    infos = [sf.info(path) for path in (reference_path, candidate_path)]
    floating = [index for index, info in enumerate(infos) if info.subtype in ("FLOAT", "DOUBLE")]
    if len(floating) != 1:
        raise ValueError("Requires one lossless float stream and one signed integer PCM stream")
    float_index = floating[0]
    float_info = infos[float_index]
    integer_info = infos[1 - float_index]
    if float_info.format not in ("WAV", "WAVEX", "RF64"):
        raise ValueError("Float quantization witness requires lossless WAV")
    if (float_info.samplerate, float_info.channels, float_info.frames) != (
        integer_info.samplerate,
        integer_info.channels,
        integer_info.frames,
    ):
        raise ValueError("Quantization must preserve rate, frame count and channel layout")
    paths = (reference_path, candidate_path)
    integer = read_pcm(paths[1 - float_index])
    samples, _ = sf.read(paths[float_index], dtype="float64", always_2d=True)
    if not np.isfinite(samples).all() or np.any(np.abs(samples) >= 1):
        raise ValueError("Float witness must be finite and unsaturated")
    if np.all(samples == samples[0]) or np.all(integer.samples == integer.samples[0]):
        raise ValueError("Constant DC is not a timbre-equivalence positive")
    centers = integer.samples.astype(np.float64) / FULL_SCALE
    half_step = integer.half_step / FULL_SCALE
    if not np.all((samples >= centers - half_step) & (samples <= centers + half_step)):
        return {"equivalent": False, "reason": "Float samples leave the integer PCM rounding cells"}
    return {
        "equivalent": True,
        "reason": "Every lossless float sample lies in its corresponding integer PCM rounding cell",
        "shared_gain": 1.0,
        "shared_gain_exact": "1",
        "quantization_bound_full_scale": half_step,
        "quantization_bound_in_final_lsb": 0.5,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        "integer_bits": integer.bits,
        "float_subtype": float_info.subtype,
        "sample_rate": integer.rate,
        "channels": integer.channels,
        "frames": samples.shape[0],
        "compared_samples": samples.size,
        "resampled": False,
        "time_aligned": False,
        "channel_gains_or_mixing": False,
    }


def compare_pcm(
    reference_path: Path,
    candidate_path: Path,
    *,
    normalization_source: Path | None = None,
    normalization_gain: float | None = None,
    allow_float_quantization: bool = False,
) -> dict:
    """Prove shared positive gain consistency within signed-PCM nearest-rounding cells."""
    try:
        if allow_float_quantization and any(
            sf.info(path).subtype in ("FLOAT", "DOUBLE")
            for path in (reference_path, candidate_path)
        ):
            if normalization_source is not None or normalization_gain is not None:
                raise ValueError("Float quantization does not accept a gain-normalization witness")
            return compare_float_quantization(reference_path, candidate_path)
        reference = read_pcm(reference_path)
        candidate = read_pcm(candidate_path)
        if (normalization_source is None) != (normalization_gain is None):
            raise ValueError("Normalization witness requires both the retained source and its gain")
        base = candidate
        stage_gain = Fraction(1)
        if normalization_source is not None:
            if not math.isfinite(normalization_gain) or normalization_gain <= 0:
                raise ValueError("Normalization must use one finite positive scalar")
            base = read_pcm(normalization_source)
            if (base.rate, base.channels, base.samples.shape) != (
                candidate.rate,
                candidate.channels,
                candidate.samples.shape,
            ):
                raise ValueError("Normalization changes rate, frame count or channel layout")
            stage_gain = Fraction(normalization_gain)
            for source, target in zip(base.samples.flat, candidate.samples.flat, strict=True):
                residual = abs(
                    int(target) * stage_gain.denominator - int(source) * stage_gain.numerator
                )
                if residual > candidate.half_step * stage_gain.denominator:
                    raise ValueError(
                        "Retained normalization witness does not reproduce candidate within final rounding"
                    )
        interval = gain_interval(reference, base)
        if interval is None:
            return {
                "equivalent": False,
                "reason": "No shared positive gain fits all samples and channels within PCM rounding cells",
            }
        gain = (interval[0] + interval[1]) / 2
        final_gain = gain * stage_gain
        error_bound = (
            Fraction(base.half_step) * stage_gain + Fraction(reference.half_step) * final_gain
        )
        if normalization_source is not None:
            error_bound += candidate.half_step
        for source, target in zip(reference.samples.flat, candidate.samples.flat, strict=True):
            residual = abs(Fraction(int(target)) - final_gain * int(source))
            if residual > error_bound:
                raise ValueError("Composed quantization bound was violated")
        return {
            "equivalent": True,
            "reason": "One positive gain satisfies every time/channel sample within analytic PCM rounding bounds",
            "shared_gain": float(final_gain),
            "shared_gain_exact": str(final_gain),
            "base_gain_interval_exact": [str(bound) for bound in interval],
            "quantization_bound_full_scale": float(error_bound / FULL_SCALE),
            "quantization_bound_in_final_lsb": float(error_bound / (2 * candidate.half_step)),
            "normalization_witness_verified": normalization_source is not None,
            "normalization_source_sha256": base.sha256
            if normalization_source is not None
            else None,
            "reference_sha256": reference.sha256,
            "candidate_sha256": candidate.sha256,
            "reference_bits": reference.bits,
            "base_bits": base.bits,
            "candidate_bits": candidate.bits,
            "sample_rate": reference.rate,
            "channels": reference.channels,
            "frames": reference.samples.shape[0],
            "compared_samples": reference.samples.size,
            "resampled": False,
            "time_aligned": False,
            "channel_gains_or_mixing": False,
        }
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        return {
            "equivalent": False,
            "reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__,
        }
