"""Remote audio scoring for vocal_synthesis.

Compares a reproduced vocal WAV against a ground-truth WAV using
Mel-Spectrogram MSE. Runs on the Windows VM to avoid downloading
large WAV files.

Usage:
    python score_audio_remote.py \
        --agent-wav  C:\path\to\reproduced_vocal.wav \
        --ref-wav    C:\path\to\reproduced_vocal_ground_truth.wav \
        --result-path C:\path\to\result.json
"""

import argparse
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Constants (must match main.py)
# ---------------------------------------------------------------------------
MEL_SR = 22050
MEL_N_FFT = 2048
MEL_HOP_LENGTH = 512
MEL_N_MELS = 128
ALPHA = 0.002105


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def read_wav_rms_db(path):
    """Compute RMS level in dB from a WAV file. Returns -120 for silence."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    rms = np.sqrt(np.mean(data ** 2))
    if rms == 0:
        return -120.0
    return float(20 * np.log10(rms))


def compute_mel_mse(agent_path, ref_path):
    """Compute MSE between Mel-spectrograms of two WAV files."""
    import numpy as np
    import librosa

    agent_y, _ = librosa.load(agent_path, sr=MEL_SR, mono=True)
    ref_y, _ = librosa.load(ref_path, sr=MEL_SR, mono=True)

    agent_mel = librosa.feature.melspectrogram(
        y=agent_y, sr=MEL_SR, n_fft=MEL_N_FFT,
        hop_length=MEL_HOP_LENGTH, n_mels=MEL_N_MELS,
    )
    ref_mel = librosa.feature.melspectrogram(
        y=ref_y, sr=MEL_SR, n_fft=MEL_N_FFT,
        hop_length=MEL_HOP_LENGTH, n_mels=MEL_N_MELS,
    )

    agent_mel_db = librosa.power_to_db(agent_mel, ref=np.max)
    ref_mel_db = librosa.power_to_db(ref_mel, ref=np.max)

    min_len = min(agent_mel_db.shape[1], ref_mel_db.shape[1])
    if min_len == 0:
        return float("inf")
    agent_mel_db = agent_mel_db[:, :min_len]
    ref_mel_db = ref_mel_db[:, :min_len]

    mse = float(np.mean((agent_mel_db - ref_mel_db) ** 2))
    return mse


def mse_to_score(mse, alpha=ALPHA):
    """Map MSE to [0, 1] score via inverse quadratic: 1 / (1 + alpha * mse^2)."""
    if mse == float("inf") or math.isnan(mse):
        return 0.0
    return 1.0 / (1.0 + alpha * mse * mse)


def main():
    parser = argparse.ArgumentParser(
        description="Score reproduced vocal against ground truth (Mel-MSE)."
    )
    parser.add_argument("--agent-wav", required=True, help="Path to reproduced_vocal.wav")
    parser.add_argument("--ref-wav", required=True, help="Path to ground truth WAV")
    parser.add_argument("--result-path", required=True, help="Where to write JSON output")
    args = parser.parse_args()

    try:
        agent_path = os.path.normpath(args.agent_wav)
        ref_path = os.path.normpath(args.ref_wav)
        result_path = os.path.normpath(args.result_path)

        if not os.path.isfile(agent_path):
            raise FileNotFoundError(f"Agent WAV not found: {agent_path}")
        if not os.path.isfile(ref_path):
            raise FileNotFoundError(f"Reference WAV not found: {ref_path}")

        file_size = os.path.getsize(agent_path)
        log(f"Agent WAV size: {file_size} bytes")

        log("Computing RMS dB...")
        rms_db = read_wav_rms_db(agent_path)
        log(f"RMS dB: {rms_db:.2f}")

        log("Computing Mel-MSE...")
        mel_mse = compute_mel_mse(agent_path, ref_path)
        log(f"Mel-MSE: {mel_mse:.6f}")

        score = mse_to_score(mel_mse)
        log(f"Score: {score:.4f}")

        result = {
            "file_size": file_size,
            "rms_db": round(rms_db, 4),
            "mel_mse": round(mel_mse, 6) if not math.isinf(mel_mse) else None,
            "score": round(score, 6),
        }

        os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        log(f"Results written to {result_path}")

    except Exception as e:
        log(f"ERROR: {e}")
        error_result = {"error": str(e)}
        try:
            result_path = os.path.normpath(args.result_path)
            os.makedirs(os.path.dirname(result_path) or ".", exist_ok=True)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(error_result, f, indent=2)
        except Exception as write_err:
            log(f"Failed to write error result: {write_err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
