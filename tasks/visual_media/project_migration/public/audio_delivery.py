"""Audio inspection and extraction interfaces for native Ardour export helpers."""

from pathlib import Path

import numpy as np
import soundfile as sf


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
        for block in stream.blocks(blocksize=max(1, stream.samplerate), always_2d=True):
            if not np.isfinite(block).all():
                result["finite"] = False
                continue
            result["peak"] = max(result["peak"], float(np.abs(block).max()))
            energy += float(np.square(block).sum())
    result["rms"] = (energy / max(1, result["frames"] * result["channels"])) ** 0.5
    result["audible"] = result["rms"] > 0
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
