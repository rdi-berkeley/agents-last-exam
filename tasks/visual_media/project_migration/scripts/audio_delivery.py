"""Stream full deliveries and select reference-only, fixed audio passages."""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf


WINDOW_SECONDS = 12
QUANTILES = (0.25, 0.5, 0.75)


def inspect_audio(path: Path) -> dict:
    with sf.SoundFile(path) as stream:
        result = {
            "frames": stream.frames,
            "rate": stream.samplerate,
            "channels": stream.channels,
            "subtype": stream.subtype,
            "duration": stream.frames / stream.samplerate,
            "finite": True,
            "peak": 0.0,
        }
        energy = 0.0
        bins = []
        for block in stream.blocks(blocksize=max(1, stream.samplerate // 10), always_2d=True):
            if not np.isfinite(block).all():
                result["finite"] = False
                continue
            result["peak"] = max(result["peak"], float(np.abs(block).max()))
            energy += float(np.square(block).sum())
            bins.append(float(np.sqrt(np.square(block).mean())))
    result["rms"] = (energy / max(1, result["frames"] * result["channels"])) ** 0.5
    result["valid"] = bool(
        result["frames"]
        and result["finite"]
        and result["peak"] < 1
        and result["rate"] >= 44100
        and result["channels"] in (1, 2)
    )
    result["audible"] = result["rms"] > 0
    activity_floor = max(bins, default=0) * 0.001
    active = [index for index, level in enumerate(bins) if level > activity_floor]
    starts = []
    for quantile in QUANTILES:
        if not active:
            break
        center = (active[min(len(active) - 1, int(quantile * len(active)))] + 0.5) / 10
        start = round(
            max(0, min(center - WINDOW_SECONDS / 2, result["duration"] - WINDOW_SECONDS)), 6
        )
        if start not in starts:
            starts.append(start)
    result["passages"] = [
        {"start": start, "duration": min(WINDOW_SECONDS, result["duration"] - start)}
        for start in starts
    ]
    return result


def extract_passage(source: Path, target: Path, start: float, duration: float):
    with sf.SoundFile(source) as stream:
        stream.seek(round(start * stream.samplerate))
        audio = stream.read(round(duration * stream.samplerate), always_2d=True)
        sf.write(target, audio, stream.samplerate, subtype=stream.subtype)


def stem_name(path: Path) -> str:
    name = path.stem
    for marker in ("乐器 - ", "Instrument - "):
        if marker in name:
            name = name.split(marker, 1)[1]
    return name.strip().casefold()


def check_delivery(output: Path, references: Path, evidence: Path) -> dict:
    evidence.mkdir(parents=True, exist_ok=True)
    reference_files = sorted([*references.glob("*.wav"), *references.glob("*.flac")])
    if not reference_files:
        raise RuntimeError("Reference stems are unavailable; evaluator infrastructure failure")
    names = [stem_name(path) for path in reference_files]
    if len(names) != len(set(names)):
        raise RuntimeError("Ambiguous evaluator reference stems")
    candidates = {}
    for path in sorted((output / "stems").glob("*.wav")):
        candidates.setdefault(stem_name(path), []).append(path)
    stems = []
    for index, reference in enumerate(reference_files):
        info = inspect_audio(reference)
        if not info["finite"] or not info["frames"]:
            raise RuntimeError(f"Invalid evaluator reference: {reference.name}")
        matches = candidates.get(stem_name(reference), [])
        candidate = matches[0] if len(matches) == 1 else None
        record = {
            "reference": reference.name,
            "reference_audio": info,
            "candidate": candidate.name if candidate else None,
            "valid": False,
            "passages": [],
        }
        if candidate:
            try:
                actual = inspect_audio(candidate)
                record["candidate_audio"] = actual
                record["valid"] = (
                    actual["valid"]
                    and actual["duration"] + 1 / actual["rate"] >= info["duration"]
                    and (actual["audible"] or not info["audible"])
                )
                if not record["valid"]:
                    record["failure"] = "Invalid, clipped, silent or incomplete full-length stem"
            except (RuntimeError, ValueError, OSError) as exc:
                record["failure"] = str(exc)
        else:
            record["failure"] = "Missing or ambiguous stem name"
        if record["valid"] and info["audible"]:
            for passage_index, passage in enumerate(info["passages"]):
                paths = [
                    evidence / f"{index:03d}-{passage_index}-{side}.wav"
                    for side in ("reference", "candidate")
                ]
                for source, target in zip((reference, candidate), paths, strict=True):
                    extract_passage(source, target, **passage)
                record["passages"].append(
                    {
                        **passage,
                        "reference_path": str(paths[0]),
                        "candidate_path": str(paths[1]),
                        "sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
                    }
                )
        stems.append(record)
    mix = {"valid": False, "failure": "Missing mixdown.wav"}
    if (output / "mixdown.wav").is_file():
        try:
            mix = inspect_audio(output / "mixdown.wav")
            mix["valid"] = (
                mix["valid"]
                and mix["audible"]
                and mix["duration"] + 1 / mix["rate"]
                >= max(record["reference_audio"]["duration"] for record in stems)
            )
        except (RuntimeError, ValueError, OSError) as exc:
            mix = {"valid": False, "failure": str(exc)}
    return {
        "expected_stems": len(stems),
        "valid_stems": sum(record["valid"] for record in stems),
        "delivery_score": sum(record["valid"] for record in stems) / len(stems),
        "mix": mix,
        "stems": stems,
        "passage_rule": "Three 12s windows at 25/50/75% of reference-active 100ms bins; activity above -60dB relative to loudest reference bin; duplicate windows merged",
    }
