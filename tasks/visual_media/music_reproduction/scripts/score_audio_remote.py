"""Remote audio scoring for music_reproduction.

Processes stems for dynamics (RMS contour correlation) and timbre
(MFCC cosine similarity). Also checks the mixdown file.
Runs on the Windows VM to avoid downloading large WAV files.

Usage:
    python score_audio_remote.py \
        --agent-stems-dir  C:\path\to\output\stems \
        --ref-stems-dir    C:\path\to\reference\stems \
        --mixdown-path     C:\path\to\output\mixdown.wav \
        --pairings-json    '[{"ref_stem": "Piano.wav", "agent_stem": "Piano.wav"}]' \
        --result-path      C:\path\to\result.json
"""

import argparse
import json
import math
import os
import sys

# Timbre threshold (must match main.py: 0.80 for music_reproduction)
TIMBRE_THRESHOLD = 0.80


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_audio_mono(path):
    """Load audio file as mono float32 array + sample rate."""
    import librosa

    data, sr = librosa.load(path, sr=None, mono=True)
    return data, sr


def read_audio_rms_db(path):
    """Compute RMS level in dB from an audio file. Returns -120 for silence."""
    import numpy as np

    data, sr = load_audio_mono(path)
    rms = np.sqrt(np.mean(data ** 2))
    if rms == 0:
        return -120.0
    return float(20 * np.log10(rms))


def compute_rms_contour(path, window_ms=100, hop_ms=50):
    """Compute windowed RMS energy contour from an audio file."""
    import numpy as np

    data, sr = load_audio_mono(path)
    win_samples = int(sr * window_ms / 1000)
    hop_samples = int(sr * hop_ms / 1000)
    if win_samples <= 0 or hop_samples <= 0 or len(data) < win_samples:
        return np.array([])
    frames = []
    for start in range(0, len(data) - win_samples + 1, hop_samples):
        frame = data[start: start + win_samples]
        frames.append(np.sqrt(np.mean(frame ** 2)))
    return np.array(frames)


def compute_dynamics_audio(agent_path, ref_path):
    """Pearson correlation of RMS loudness contours, mapped to [0, 1]."""
    import numpy as np
    from scipy.stats import pearsonr

    agent_rms = compute_rms_contour(agent_path)
    ref_rms = compute_rms_contour(ref_path)
    min_len = min(len(agent_rms), len(ref_rms))
    if min_len < 3:
        return 0.5  # insufficient data
    agent_rms = agent_rms[:min_len]
    ref_rms = ref_rms[:min_len]
    if np.std(ref_rms) == 0 or np.std(agent_rms) == 0:
        return 1.0 if np.std(ref_rms) == 0 and np.std(agent_rms) == 0 else 0.0
    corr, _ = pearsonr(agent_rms, ref_rms)
    if np.isnan(corr):
        return 0.5
    return max(0.0, (corr + 1.0) / 2.0)


def compute_timbre_similarity(agent_path, ref_path):
    """Cosine similarity of MFCC embeddings (39-dim: 13 MFCCs + deltas)."""
    import numpy as np
    import librosa

    def _mfcc_embedding(wav_path):
        data, sr = librosa.load(wav_path, sr=22050, mono=True)
        if len(data) < 2048:
            return np.zeros(39)
        mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2])  # 39 x T
        return features.mean(axis=1)  # 39-dim

    agent_emb = _mfcc_embedding(agent_path)
    ref_emb = _mfcc_embedding(ref_path)

    norm_product = np.linalg.norm(agent_emb) * np.linalg.norm(ref_emb)
    if norm_product < 1e-8:
        return 0.0
    cos_sim = float(np.dot(agent_emb, ref_emb) / norm_product)
    # Threshold: >= 0.80 -> full credit, else linear from 0
    if cos_sim >= TIMBRE_THRESHOLD:
        return 1.0
    return max(0.0, cos_sim / TIMBRE_THRESHOLD)


def main():
    parser = argparse.ArgumentParser(
        description="Score music reproduction stems (dynamics + timbre)."
    )
    parser.add_argument("--agent-stems-dir", required=True, help="Path to agent output stems/ directory")
    parser.add_argument("--ref-stems-dir", required=True, help="Path to reference stems/ directory")
    parser.add_argument("--mixdown-path", required=True, help="Path to mixdown.wav")
    parser.add_argument("--pairings-json", required=True,
                        help='JSON string of track pairings: [{"ref_stem": "...", "agent_stem": "..."}, ...]')
    parser.add_argument("--result-path", required=True, help="Where to write JSON output")
    args = parser.parse_args()

    try:
        agent_dir = os.path.normpath(args.agent_stems_dir)
        ref_dir = os.path.normpath(args.ref_stems_dir)
        mixdown_path = os.path.normpath(args.mixdown_path)
        result_path = os.path.normpath(args.result_path)

        pairings_arg = args.pairings_json
        if os.path.isfile(pairings_arg):
            with open(pairings_arg, "r", encoding="utf-8") as f:
                pairings = json.load(f)
        else:
            pairings = json.loads(pairings_arg)

        # --- Mixdown check ---
        mixdown_info = {"file_size": 0, "rms_db": -120.0}
        if os.path.isfile(mixdown_path):
            mixdown_info["file_size"] = os.path.getsize(mixdown_path)
            mixdown_info["rms_db"] = round(read_audio_rms_db(mixdown_path), 4)
            log(f"Mixdown: size={mixdown_info['file_size']} rms_db={mixdown_info['rms_db']:.2f}")
        else:
            log(f"WARNING: Mixdown not found at {mixdown_path}")

        # --- Agent stems: list, compute RMS, identify non-silent ---
        agent_stems_info = {}
        num_non_silent = 0

        if os.path.isdir(agent_dir):
            agent_wav_files = [f for f in os.listdir(agent_dir) if f.lower().endswith(".wav")]
            log(f"Found {len(agent_wav_files)} WAV files in agent stems dir")

            for fname in agent_wav_files:
                fpath = os.path.join(agent_dir, fname)
                rms_db = read_audio_rms_db(fpath)
                is_silent = rms_db <= -60.0
                agent_stems_info[fname] = {
                    "rms_db": round(rms_db, 4),
                    "is_silent": is_silent,
                }
                if not is_silent:
                    num_non_silent += 1
                log(f"  {fname}: rms_db={rms_db:.2f} silent={is_silent}")
        else:
            log(f"WARNING: Agent stems directory not found: {agent_dir}")

        log(f"Non-silent stems: {num_non_silent}")

        # --- Process pairings ---
        pair_scores = []
        for pairing in pairings:
            ref_stem = pairing["ref_stem"]
            agent_stem = pairing["agent_stem"]

            ref_path = os.path.join(ref_dir, ref_stem)
            agent_path = os.path.join(agent_dir, agent_stem)

            dynamics_val = 0.5
            timbre_val = 0.0

            if not os.path.isfile(ref_path):
                log(f"  WARNING: ref stem not found: {ref_path}")
                pair_scores.append({
                    "ref_stem": ref_stem,
                    "agent_stem": agent_stem,
                    "dynamics": round(dynamics_val, 4),
                    "timbre": round(timbre_val, 4),
                })
                continue

            if not os.path.isfile(agent_path):
                log(f"  WARNING: agent stem not found: {agent_path}")
                pair_scores.append({
                    "ref_stem": ref_stem,
                    "agent_stem": agent_stem,
                    "dynamics": round(dynamics_val, 4),
                    "timbre": round(timbre_val, 4),
                })
                continue

            try:
                dynamics_val = compute_dynamics_audio(agent_path, ref_path)
                log(f"  Dynamics: ref='{ref_stem}' agent='{agent_stem}' corr={dynamics_val:.4f}")
            except Exception as e:
                log(f"  Dynamics eval failed for {ref_stem}: {e}")
                dynamics_val = 0.5

            try:
                timbre_val = compute_timbre_similarity(agent_path, ref_path)
                log(f"  Timbre: ref='{ref_stem}' agent='{agent_stem}' similarity={timbre_val:.4f}")
            except Exception as e:
                log(f"  Timbre eval failed for {ref_stem}: {e}")
                timbre_val = 0.0

            pair_scores.append({
                "ref_stem": ref_stem,
                "agent_stem": agent_stem,
                "dynamics": round(dynamics_val, 4),
                "timbre": round(timbre_val, 4),
            })

        result = {
            "mixdown": mixdown_info,
            "agent_stems": agent_stems_info,
            "num_non_silent": num_non_silent,
            "pair_scores": pair_scores,
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
