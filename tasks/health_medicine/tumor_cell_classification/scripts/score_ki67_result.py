"""Score Ki67 tumor-cell classification outputs."""

from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "total_cells",
    "positive_count",
    "negative_count",
    "ki67_index_percent",
    "hotspot_x",
    "hotspot_y",
    "hotspot_width",
    "hotspot_height",
    "notes",
]


@dataclass
class Ki67ScoreResult:
    score: float
    passed: bool
    reasons: list[str]
    ki67_index_percent: float | None = None


def _parse_json(data: bytes, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        parsed = json.loads(data.decode("utf-8-sig"))
    except Exception as exc:
        return None, [f"{label}:invalid_json:{exc}"]
    if not isinstance(parsed, dict):
        return None, [f"{label}:not_object"]
    return parsed, []


def _parse_int(value: Any, field: str, reasons: list[str]) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        reasons.append(f"{field}:not_integer")
        return None
    return value


def _parse_float(value: Any, field: str, reasons: list[str]) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        reasons.append(f"{field}:not_numeric")
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        reasons.append(f"{field}:not_finite")
        return None
    return parsed


def _validate_png(data: bytes) -> tuple[tuple[int, int] | None, list[str]]:
    reasons: list[str] = []
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None, ["overlay:not_png"]

    offset = 8
    dims: tuple[int, int] | None = None
    idat = bytearray()
    ihdr: tuple[int, int, int, int, int, int, int] | None = None
    saw_iend = False
    saw_ihdr = False
    saw_idat = False
    idat_closed = False
    chunk_index = 0

    while offset < len(data):
        if offset + 8 > len(data):
            reasons.append("overlay:truncated_chunk_header")
            break
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        offset += 8
        if offset + length + 4 > len(data):
            reasons.append("overlay:truncated_chunk_data")
            break
        chunk_data = data[offset : offset + length]
        offset += length
        expected_crc = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            reasons.append("overlay:crc_mismatch")
            break

        if chunk_type == b"IHDR":
            if chunk_index != 0:
                reasons.append("overlay:ihdr_not_first")
                break
            if saw_ihdr or length != 13:
                reasons.append("overlay:invalid_ihdr")
                break
            saw_ihdr = True
            width, height, bit_depth, color_type, compression, filter_method, interlace = (
                struct.unpack(">IIBBBBB", chunk_data)
            )
            dims = (width, height)
            ihdr = (width, height, bit_depth, color_type, compression, filter_method, interlace)
        elif chunk_type == b"IDAT":
            if not saw_ihdr:
                reasons.append("overlay:idat_before_ihdr")
                break
            if idat_closed:
                reasons.append("overlay:split_idat_sequence")
                break
            saw_idat = True
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            saw_iend = True
            break
        elif saw_idat:
            idat_closed = True
        chunk_index += 1

    if offset != len(data) and saw_iend:
        trailing = data[offset:]
        if trailing.strip(b"\x00\r\n\t "):
            reasons.append("overlay:trailing_data")
    if not saw_ihdr:
        reasons.append("overlay:missing_ihdr")
    if not idat:
        reasons.append("overlay:missing_idat")
    else:
        try:
            decoded = zlib.decompress(bytes(idat))
        except zlib.error:
            reasons.append("overlay:idat_not_decodable")
        else:
            if ihdr is not None:
                width, height, bit_depth, color_type, compression, filter_method, interlace = ihdr
                valid_depths = {
                    0: {1, 2, 4, 8, 16},
                    2: {8, 16},
                    3: {1, 2, 4, 8},
                    4: {8, 16},
                    6: {8, 16},
                }
                channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
                if color_type not in valid_depths or bit_depth not in valid_depths[color_type]:
                    reasons.append("overlay:unsupported_color_type_or_depth")
                if compression != 0 or filter_method != 0:
                    reasons.append("overlay:unsupported_png_methods")
                if interlace != 0:
                    reasons.append("overlay:unsupported_interlace")
                if not reasons:
                    bits_per_row = width * channels[color_type] * bit_depth
                    bytes_per_row = (bits_per_row + 7) // 8
                    expected_len = height * (1 + bytes_per_row)
                    if len(decoded) != expected_len:
                        reasons.append("overlay:decoded_scanline_length_mismatch")
    if not saw_iend:
        reasons.append("overlay:missing_iend")
    return dims, reasons


def _validate_overlay(data: bytes) -> list[str]:
    reasons: list[str] = []
    dims, png_reasons = _validate_png(data)
    reasons.extend(png_reasons)
    if dims is None or png_reasons:
        return reasons
    width, height = dims
    if width < 100 or height < 100:
        reasons.append("overlay:too_small")
    if len(data) < 1024:
        reasons.append("overlay:file_too_small")
    return reasons


def score_ki67_result_bytes(
    *,
    result_bytes: bytes,
    overlay_bytes: bytes,
    reference_bytes: bytes,
) -> Ki67ScoreResult:
    result, reasons = _parse_json(result_bytes, "result")
    reference, ref_reasons = _parse_json(reference_bytes, "reference")
    reasons.extend(ref_reasons)
    if result is None or reference is None:
        return Ki67ScoreResult(score=0.0, passed=False, reasons=reasons)

    for field in REQUIRED_FIELDS:
        if field not in result:
            reasons.append(f"missing_field:{field}")

    total = _parse_int(result.get("total_cells"), "total_cells", reasons)
    positive = _parse_int(result.get("positive_count"), "positive_count", reasons)
    negative = _parse_int(result.get("negative_count"), "negative_count", reasons)
    reported_index = _parse_float(result.get("ki67_index_percent"), "ki67_index_percent", reasons)

    hotspot_x = _parse_int(result.get("hotspot_x"), "hotspot_x", reasons)
    hotspot_y = _parse_int(result.get("hotspot_y"), "hotspot_y", reasons)
    hotspot_width = _parse_int(result.get("hotspot_width"), "hotspot_width", reasons)
    hotspot_height = _parse_int(result.get("hotspot_height"), "hotspot_height", reasons)
    notes = result.get("notes")
    if not isinstance(notes, str) or not notes.strip():
        reasons.append("notes:missing_or_empty")

    if total is not None and total <= 0:
        reasons.append("total_cells:not_positive")
    for field, value in [("positive_count", positive), ("negative_count", negative)]:
        if value is not None and value < 0:
            reasons.append(f"{field}:negative")
    if total is not None and positive is not None and negative is not None:
        if positive + negative != total:
            reasons.append("counts:do_not_sum_to_total")
        if positive > total or negative > total:
            reasons.append("counts:exceed_total")
        expected_index = positive / total * 100.0 if total > 0 else None
        if expected_index is not None and reported_index is not None:
            if abs(reported_index - expected_index) > 0.05:
                reasons.append("ki67_index_percent:inconsistent_with_counts")

    slide_width = int(reference.get("slide_width", 0))
    slide_height = int(reference.get("slide_height", 0))
    if all(v is not None for v in [hotspot_x, hotspot_y, hotspot_width, hotspot_height]):
        assert hotspot_x is not None
        assert hotspot_y is not None
        assert hotspot_width is not None
        assert hotspot_height is not None
        if hotspot_x < 0 or hotspot_y < 0:
            reasons.append("hotspot:negative_origin")
        if hotspot_width <= 0 or hotspot_height <= 0:
            reasons.append("hotspot:non_positive_size")
        if slide_width and hotspot_x + hotspot_width > slide_width:
            reasons.append("hotspot:exceeds_slide_width")
        if slide_height and hotspot_y + hotspot_height > slide_height:
            reasons.append("hotspot:exceeds_slide_height")

    target = _parse_float(reference.get("target_ki67_index_percent"), "target", reasons)
    tolerance = _parse_float(reference.get("tolerance_percentage_points"), "tolerance", reasons)
    if target is not None and tolerance is not None and reported_index is not None:
        if abs(reported_index - target) > tolerance:
            reasons.append("ki67_index_percent:outside_tolerance")

    reasons.extend(_validate_overlay(overlay_bytes))

    passed = not reasons
    return Ki67ScoreResult(
        score=1.0 if passed else 0.0,
        passed=passed,
        reasons=reasons[:50],
        ki67_index_percent=reported_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    score = score_ki67_result_bytes(
        result_bytes=Path(args.result).read_bytes(),
        overlay_bytes=Path(args.overlay).read_bytes(),
        reference_bytes=Path(args.reference).read_bytes(),
    )
    print(json.dumps(asdict(score), indent=2))


if __name__ == "__main__":
    main()
