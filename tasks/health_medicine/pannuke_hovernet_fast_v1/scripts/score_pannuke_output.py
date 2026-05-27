from __future__ import annotations

import argparse
import ast
import json
import os
import struct
import zlib
from pathlib import Path
from zipfile import ZipFile


EXPECTED_SHAPE = (2722, 256, 256, 6)
EXPECTED_DTYPE = "<f8"
REFERENCE_MEMBER = "output/predictions/masks.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def npy_header_from_file(path: Path) -> tuple[tuple[int, ...], str]:
    with path.open("rb") as handle:
        return npy_header_from_handle(handle)


def npy_header_from_handle(handle) -> tuple[tuple[int, ...], str]:
    magic = handle.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("not a NumPy .npy file")
    major, minor = struct.unpack("BB", handle.read(2))
    if (major, minor) == (1, 0):
        header_len = struct.unpack("<H", handle.read(2))[0]
    elif major in {2, 3}:
        header_len = struct.unpack("<I", handle.read(4))[0]
    else:
        raise ValueError(f"unsupported npy version {(major, minor)}")
    header = ast.literal_eval(handle.read(header_len).decode("latin1").strip())
    return tuple(header["shape"]), header["descr"]


def reference_info(reference_zip: Path) -> dict[str, int]:
    with ZipFile(reference_zip) as zf:
        info = zf.getinfo(REFERENCE_MEMBER)
        return {"crc": info.CRC, "file_size": info.file_size}


def candidate_zip_info(path: Path) -> dict[str, int] | None:
    try:
        with ZipFile(path) as zf:
            info = zf.getinfo(REFERENCE_MEMBER)
            with zf.open(REFERENCE_MEMBER) as handle:
                shape, dtype = npy_header_from_handle(handle)
            return {"crc": info.CRC, "file_size": info.file_size, "shape": shape, "dtype": dtype}
    except Exception:
        return None


def crc32_file(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return value & 0xFFFFFFFF


def locate_candidate(output_dir: Path) -> tuple[str, Path] | tuple[None, None]:
    zip_path = output_dir / "output.zip"
    if zip_path.exists():
        return "zip", zip_path
    masks_path = output_dir / "predictions" / "masks.npy"
    if masks_path.exists():
        return "npy", masks_path
    return None, None


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    reference_zip = Path(args.reference_dir).resolve() / "reference_outputs" / "output.zip"
    report = {
        "score": 0.0,
        "passed": False,
        "checks": [],
        "notes": [],
    }
    if not reference_zip.exists():
        report["notes"].append("missing reference output.zip")
        print(json.dumps(report, indent=2))
        return 0

    kind, candidate = locate_candidate(output_dir)
    if candidate is None:
        report["notes"].append("missing output.zip or predictions/masks.npy")
        print(json.dumps(report, indent=2))
        return 0

    ref = reference_info(reference_zip)
    format_score = 0.0
    exact_score = 0.0
    try:
        if kind == "zip":
            info = candidate_zip_info(candidate)
            if not info:
                raise ValueError("zip does not contain expected masks member")
            shape_ok = tuple(info["shape"]) == EXPECTED_SHAPE
            dtype_ok = info["dtype"] == EXPECTED_DTYPE
            exact_ok = info["crc"] == ref["crc"] and info["file_size"] == ref["file_size"]
            format_score = 0.2 if shape_ok and dtype_ok else 0.0
            exact_score = 0.8 if exact_ok else 0.0
            report["checks"].extend(
                [
                    {"check": "zip_member_present", "passed": True},
                    {"check": "shape", "passed": shape_ok},
                    {"check": "dtype", "passed": dtype_ok},
                    {"check": "reference_crc", "passed": exact_ok},
                ]
            )
        else:
            shape, dtype = npy_header_from_file(candidate)
            shape_ok = shape == EXPECTED_SHAPE
            dtype_ok = dtype == EXPECTED_DTYPE
            size_ok = os.path.getsize(candidate) == ref["file_size"]
            exact_ok = size_ok and crc32_file(candidate) == ref["crc"]
            format_score = 0.2 if shape_ok and dtype_ok else 0.0
            exact_score = 0.8 if exact_ok else 0.0
            report["checks"].extend(
                [
                    {"check": "shape", "passed": shape_ok},
                    {"check": "dtype", "passed": dtype_ok},
                    {"check": "file_size", "passed": size_ok},
                    {"check": "reference_crc", "passed": exact_ok},
                ]
            )
    except Exception as exc:
        report["notes"].append(str(exc))

    manifest = output_dir / "run_manifest.json"
    manifest_score = 0.0
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_ok = payload.get("model_mode") == "hovernet_fast"
        except Exception:
            manifest_ok = False
        manifest_score = 0.0 if not manifest_ok else 0.0
        report["checks"].append({"check": "manifest_model_mode_hovernet_fast", "passed": manifest_ok})

    report["score"] = round(format_score + exact_score + manifest_score, 4)
    report["passed"] = report["score"] >= 1.0
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
