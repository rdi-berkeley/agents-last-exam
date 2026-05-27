"""Remote audio scoring for project_migration.

Processes stem WAV files for quality checks and timbral similarity.
Runs on the Windows VM to avoid downloading large WAV files.

Usage:
    python score_audio_remote.py \
        --agent-stems-dir  C:\path\to\output\stems \
        --ref-stems-dir    C:\path\to\reference\stems \
        --result-path      C:\path\to\result.json
"""

import argparse
import json
import os
import sys

# MFCC cosine similarity threshold (must match main.py)
TIMBRE_THRESHOLD = 0.70


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


def check_audio_quality(path):
    """Check WAV for clipping. Returns True if audio passes quality checks."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    peak = np.max(np.abs(data))
    return bool(peak < 1.0)


def extract_track_name(filename):
    """Extract the track/instrument name from a Cubase export filename.

    Cubase exports stems as: "Project - 0001 - 乐器 - Track Name.wav"
    This extracts "Track Name". Falls back to full basename if pattern not found.
    """
    base = os.path.splitext(filename)[0].strip()
    marker = "乐器 - "  # 乐器 -
    idx = base.find(marker)
    if idx >= 0:
        return base[idx + len(marker):].strip()
    marker_en = "Instrument - "
    idx = base.find(marker_en)
    if idx >= 0:
        return base[idx + len(marker_en):].strip()
    return base


def find_stem_file(stem_files, target_name):
    """Fuzzy-match a stem name to a filename in the list.

    Matching priority:
    1. Exact match on extracted track names
    2. Exact match on full basenames
    3. Substring match on extracted track names
    4. Substring match on full basenames
    """
    target_track = extract_track_name(target_name).lower()
    target_base = os.path.splitext(target_name)[0].lower().strip()

    candidates = []
    for f in stem_files:
        track = extract_track_name(f).lower()
        base = os.path.splitext(f)[0].lower().strip()
        candidates.append((f, track, base))

    # Pass 1: exact match on extracted track names
    for f, track, base in candidates:
        if track == target_track:
            return f

    # Pass 2: exact match on full basenames
    for f, track, base in candidates:
        if base == target_base:
            return f

    # Pass 3: substring match on extracted track names
    for f, track, base in candidates:
        if target_track in track or track in target_track:
            return f

    # Pass 4: substring match on full basenames
    for f, track, base in candidates:
        if target_base in base or base in target_base:
            return f

    return None


def compute_timbre_similarity(agent_path, ref_path):
    """Cosine similarity of MFCC embeddings (39-dim: 13 MFCCs + deltas).

    Returns a score in [0, 1]. Threshold: >= TIMBRE_THRESHOLD -> full credit,
    linearly scaled below.
    """
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
    if cos_sim >= TIMBRE_THRESHOLD:
        return 1.0
    return max(0.0, cos_sim / TIMBRE_THRESHOLD)


def main():
    parser = argparse.ArgumentParser(
        description="Score project migration stems (quality + timbral similarity)."
    )
    parser.add_argument("--agent-stems-dir", required=True, help="Path to agent output stems/ directory")
    parser.add_argument("--ref-stems-dir", required=True, help="Path to reference stems/ directory")
    parser.add_argument("--result-path", required=True, help="Where to write JSON output")
    args = parser.parse_args()

    try:
        agent_dir = os.path.normpath(args.agent_stems_dir)
        ref_dir = os.path.normpath(args.ref_stems_dir)
        result_path = os.path.normpath(args.result_path)

        if not os.path.isdir(agent_dir):
            raise FileNotFoundError(f"Agent stems directory not found: {agent_dir}")
        if not os.path.isdir(ref_dir):
            raise FileNotFoundError(f"Reference stems directory not found: {ref_dir}")

        # --- Agent stems: list, compute RMS, filter silent, check clipping ---
        agent_wav_files = [f for f in os.listdir(agent_dir) if f.lower().endswith(".wav")]
        log(f"Found {len(agent_wav_files)} WAV files in agent stems dir")

        agent_stems_info = {}
        for fname in agent_wav_files:
            fpath = os.path.join(agent_dir, fname)
            rms_db = read_wav_rms_db(fpath)
            is_silent = rms_db <= -60.0
            quality_ok = True
            if not is_silent:
                quality_ok = check_audio_quality(fpath)
            agent_stems_info[fname] = {
                "rms_db": round(rms_db, 4),
                "is_silent": is_silent,
                "quality_ok": quality_ok if not is_silent else False,
            }
            log(f"  {fname}: rms_db={rms_db:.2f} silent={is_silent} quality_ok={quality_ok}")

        non_silent_stems = [f for f, info in agent_stems_info.items() if not info["is_silent"]]
        log(f"Non-silent stems: {len(non_silent_stems)}")

        # --- Reference stems ---
        ref_wav_files = [f for f in os.listdir(ref_dir) if f.lower().endswith(".wav")]
        log(f"Found {len(ref_wav_files)} WAV files in reference stems dir")

        # --- Match ref stems to agent stems and compute timbre similarity ---
        matches = []
        timbre_scores = []
        num_valid = 0

        for ref_name in ref_wav_files:
            ref_base = os.path.splitext(ref_name)[0]
            matched = find_stem_file(non_silent_stems, ref_base)

            if matched and matched in agent_stems_info:
                quality_ok = agent_stems_info[matched]["quality_ok"]
                if quality_ok:
                    num_valid += 1

                # Compute timbre similarity
                agent_path = os.path.join(agent_dir, matched)
                ref_path = os.path.join(ref_dir, ref_name)
                try:
                    sim = compute_timbre_similarity(agent_path, ref_path)
                    log(f"  Timbre: ref='{ref_name}' agent='{matched}' similarity={sim:.4f}")
                except Exception as e:
                    log(f"  Timbre eval failed for {ref_name}: {e}")
                    sim = 0.0

                timbre_scores.append(sim)
                matches.append({
                    "ref": ref_name,
                    "agent": matched,
                    "quality_ok": quality_ok,
                    "timbre_similarity": round(sim, 4),
                })
            else:
                timbre_scores.append(0.0)
                matches.append({
                    "ref": ref_name,
                    "agent": None,
                    "quality_ok": False,
                    "timbre_similarity": None,
                })
                log(f"  Timbre: ref='{ref_name}' -- no matching agent stem")

        avg_timbre = sum(timbre_scores) / len(timbre_scores) if timbre_scores else 0.0

        result = {
            "agent_stems": agent_stems_info,
            "num_non_silent": len(non_silent_stems),
            "ref_stems": ref_wav_files,
            "matches": matches,
            "num_valid": num_valid,
            "num_expected": len(ref_wav_files),
            "avg_timbre": round(avg_timbre, 4),
        }

        log(f"num_valid={num_valid} num_expected={len(ref_wav_files)} avg_timbre={avg_timbre:.4f}")

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
