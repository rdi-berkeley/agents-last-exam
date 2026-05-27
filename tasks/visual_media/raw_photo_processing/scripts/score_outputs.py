"""Score Lightroom PNG exports for raw_photo_processing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_FILES = ["003.png", "015.png", "016.png", "017.png", "018.png", "019.png"]
EXPECTED_COPYRIGHT = "Copyright 2026 AgentHLE Benchmark. All rights reserved."
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAE_THRESHOLD = 2.0


def _image_files(path: Path) -> set[str]:
    if not path.exists() or not path.is_dir():
        return set()
    return {p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}


def _detect_image_format(path: Path) -> tuple[str | None, str | None]:
    try:
        with Image.open(path) as image:
            return (image.format or "").upper() or None, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _metadata_text(image: Image.Image) -> str:
    values: list[str] = []
    for key in ("xmp", "XML:com.adobe.xmp", "copyright", "Copyright"):
        value = image.info.get(key)
        if isinstance(value, bytes):
            values.append(value.decode("utf-8", "ignore"))
        elif value is not None:
            values.append(str(value))
    return "\n".join(values)


def _score_one(agent_path: Path, reference_path: Path) -> dict[str, object]:
    with Image.open(agent_path) as agent_img, Image.open(reference_path) as ref_img:
        agent_format = (agent_img.format or "").upper()
        if agent_format != "PNG":
            return {
                "file": agent_path.name,
                "passed": False,
                "reason": "not_png",
                "format": agent_format or None,
            }
        agent_rgb = agent_img.convert("RGB")
        ref_rgb = ref_img.convert("RGB")
        if agent_rgb.size != ref_rgb.size:
            return {"file": agent_path.name, "passed": False, "reason": "size_mismatch"}
        agent_arr = np.asarray(agent_rgb, dtype=np.float32)
        ref_arr = np.asarray(ref_rgb, dtype=np.float32)
        mae = float(np.mean(np.abs(agent_arr - ref_arr)))
        if not math.isfinite(mae):
            return {"file": agent_path.name, "passed": False, "reason": "non_finite_mae"}
        metadata_ok = EXPECTED_COPYRIGHT in _metadata_text(agent_img)
        passed = mae <= MAE_THRESHOLD and metadata_ok
        reason = "ok" if passed else ("metadata_missing" if mae <= MAE_THRESHOLD else "pixel_mae_high")
        return {
            "file": agent_path.name,
            "passed": passed,
            "reason": reason,
            "mae": mae,
            "metadata_ok": metadata_ok,
        }


def score_dirs(agent_dir: Path, reference_dir: Path) -> dict[str, object]:
    agent_files = _image_files(agent_dir)
    expected = set(EXPECTED_FILES)
    missing = sorted(expected - agent_files)
    extra = sorted(agent_files - expected)
    if missing or extra:
        return {
            "score": 0.0,
            "passed": False,
            "hard_fail": True,
            "missing": missing,
            "extra": extra,
            "per_file": [],
        }

    invalid_formats = []
    for name in EXPECTED_FILES:
        detected_format, error = _detect_image_format(agent_dir / name)
        if error or detected_format != "PNG":
            invalid_formats.append({"file": name, "format": detected_format, "error": error})
    if invalid_formats:
        return {
            "score": 0.0,
            "passed": False,
            "hard_fail": True,
            "reason": "invalid_png_format",
            "invalid_formats": invalid_formats,
            "missing": [],
            "extra": [],
            "per_file": [],
        }

    per_file = [_score_one(agent_dir / name, reference_dir / name) for name in EXPECTED_FILES]
    passed_count = sum(1 for item in per_file if item["passed"])
    return {
        "score": passed_count / len(EXPECTED_FILES),
        "passed": passed_count == len(EXPECTED_FILES),
        "hard_fail": False,
        "missing": [],
        "extra": [],
        "per_file": per_file,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()
    payload = score_dirs(Path(args.agent_dir), Path(args.reference_dir))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
