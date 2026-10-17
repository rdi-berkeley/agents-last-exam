"""Deterministic, time-resolved audio similarity for bounded music excerpts."""

from functools import lru_cache
import math

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix
import soundfile as sf


VERSION = "spectral-envelope-v1"
SAMPLE_RATE = 22050
HOP = 220
FFT_SIZES = (1024, 4096)
BANDS = 64
SPECTRAL_SCALE_DB = 8.0
ENVELOPE_SCALE = 0.5
MAX_SECONDS = 60


def read_audio(path):
    info = sf.info(path)
    if not (
        0 < info.frames <= MAX_SECONDS * info.samplerate
        and 8000 <= info.samplerate <= 192000
        and info.channels in (1, 2)
    ):
        raise ValueError("Expected a bounded nonempty mono/stereo audio passage")
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    if not np.isfinite(samples).all():
        raise ValueError("Nonfinite decoded audio")
    return samples, rate


@lru_cache(maxsize=len(FFT_SIZES))
def filter_bank(size):
    frequencies = np.fft.rfftfreq(size, 1 / SAMPLE_RATE)
    highest = 2595 * np.log10(1 + SAMPLE_RATE / 1400)
    edges = 700 * (10 ** (np.linspace(0, highest, BANDS + 2) / 2595) - 1)
    rising = (frequencies[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None]
    falling = (edges[2:, None] - frequencies[None, :]) / np.diff(edges)[1:, None]
    return csr_matrix(np.maximum(0, np.minimum(rising, falling)))


def features(samples, rate, *, frames=None):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Invalid decoded audio")
    if np.all(samples == samples[0]):
        return None
    samples = samples - samples.mean(axis=0)
    samples /= np.abs(samples).max()
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        samples = signal.resample_poly(samples, SAMPLE_RATE // divisor, rate // divisor, axis=0)
    if frames is not None:
        samples = np.pad(samples[:frames], ((0, max(0, frames - len(samples))), (0, 0)))
    result = []
    for size in FFT_SIZES:
        padded = np.pad(samples, ((0, max(0, size - len(samples))), (0, 0)))
        _, times, spectrum = signal.stft(
            padded.T,
            SAMPLE_RATE,
            nperseg=size,
            noverlap=size - HOP,
            boundary="zeros",
            padded=True,
        )
        power = np.mean(np.abs(spectrum) ** 2, axis=0)
        bands = gaussian_filter1d(filter_bank(size) @ power, 1, axis=1)
        bands = bands[:, times <= len(samples) / SAMPLE_RATE]
        energy = bands.sum(axis=0)
        envelope = np.sqrt(energy / max(float(energy.mean()), 1e-30))
        shape = 10 * np.log10(np.maximum(bands / np.maximum(energy, 1e-30), 1e-5))
        result.append((shape, envelope))
    return result


def compare_features(reference, candidate):
    if reference is None:
        raise ValueError("Reference contains only silence or constant DC")
    if candidate is None:
        return {"score": 0.0, "route": "silent_candidate"}
    distances = []
    for (before_shape, before_envelope), (after_shape, after_envelope) in zip(
        reference, candidate, strict=True
    ):
        if before_shape.shape != after_shape.shape:
            raise ValueError("Passages must cover the same fixed time interval")
        active = (before_envelope > before_envelope.max() * 0.01) & (
            after_envelope > after_envelope.max() * 0.01
        )
        spectral = (
            float(np.abs(before_shape[:, active] - after_shape[:, active]).mean())
            if active.any()
            else 50.0
        )
        envelope = float(np.mean(np.abs(before_envelope - after_envelope)))
        distances.append({"spectral_db": spectral, "envelope_error": envelope})
    spectral = sum(item["spectral_db"] for item in distances) / len(distances)
    envelope = sum(item["envelope_error"] for item in distances) / len(distances)
    score = math.exp(-((spectral / SPECTRAL_SCALE_DB) ** 2 + (envelope / ENVELOPE_SCALE) ** 2))
    return {
        "score": score,
        "route": VERSION,
        "spectral_db": spectral,
        "envelope_error": envelope,
        "resolutions": distances,
    }


def compare_audio(reference, reference_rate, candidate, candidate_rate):
    duration = len(reference) / reference_rate
    if abs(duration - len(candidate) / candidate_rate) > 2 / min(reference_rate, candidate_rate):
        raise ValueError("Candidate does not cover the reference passage")
    expected = round(duration * SAMPLE_RATE)

    return compare_features(
        features(reference, reference_rate, frames=expected),
        features(candidate, candidate_rate, frames=expected),
    )
