"""Score KiCad PCB layout outputs for engineering/pcb_layout_kicad_1."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

EXPECTED_HOLE_SPAN_X_MM = 1.575 * 25.4
EXPECTED_HOLE_SPAN_Y_MM = 1.26 * 25.4
HOLE_SPAN_TOLERANCE_MM = 0.005 * 25.4
MIN_ROUTED_SEGMENTS_FOR_FALLBACK = 20
MIN_STITCHING_VIAS_FOR_FALLBACK = 2


def _extract_blocks(text: str, block_name: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines(keepends=True)
    pattern = re.compile(rf"^\s*\({re.escape(block_name)}\b")
    in_block = False
    depth = 0
    current: list[str] = []

    for line in lines:
        if not in_block and pattern.match(line):
            in_block = True
            depth = 0
            current = []
        if in_block:
            current.append(line)
            depth += line.count("(") - line.count(")")
            if depth <= 0:
                blocks.append("".join(current))
                in_block = False
                current = []
    return blocks


def _mounting_hole_positions(text: str) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    for block in _extract_blocks(text, "footprint"):
        if "MountingHole" not in block:
            continue
        match = re.search(r"\(at\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)", block)
        if not match:
            continue
        positions.append((float(match.group(1)), float(match.group(2))))
    return positions


def _span_ok(positions: list[tuple[float, float]]) -> bool:
    if len(positions) != 4:
        return False
    xs = [pos[0] for pos in positions]
    ys = [pos[1] for pos in positions]
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    return (
        math.isclose(span_x, EXPECTED_HOLE_SPAN_X_MM, abs_tol=HOLE_SPAN_TOLERANCE_MM)
        and math.isclose(span_y, EXPECTED_HOLE_SPAN_Y_MM, abs_tol=HOLE_SPAN_TOLERANCE_MM)
    )


def _gnd_zone_layers(text: str) -> set[str]:
    layers: set[str] = set()
    for block in _extract_blocks(text, "zone"):
        if '(net "GND")' not in block and '(net_name "GND")' not in block:
            continue
        if "(filled_polygon" not in block:
            continue
        if '(layer "F.Cu")' in block:
            layers.add("F.Cu")
        if '(layer "B.Cu")' in block:
            layers.add("B.Cu")
    return layers


def structural_checks(text: str) -> dict[str, Any]:
    holes = _mounting_hole_positions(text)
    zone_layers = _gnd_zone_layers(text)
    segment_count = len(_extract_blocks(text, "segment"))
    via_count = len(_extract_blocks(text, "via"))
    checks = {
        "edge_cuts": '(layer "Edge.Cuts")' in text,
        "gnd_zone_f_cu": "F.Cu" in zone_layers,
        "gnd_zone_b_cu": "B.Cu" in zone_layers,
        "mounting_hole_count": len(holes) == 4,
        "mounting_hole_spacing": _span_ok(holes),
        "fallback_routing_segments": segment_count >= MIN_ROUTED_SEGMENTS_FOR_FALLBACK,
        "fallback_stitching_vias": via_count >= MIN_STITCHING_VIAS_FOR_FALLBACK,
    }
    return {
        "checks": checks,
        "all_required": all(checks.values()),
        "mounting_hole_positions_mm": holes,
        "gnd_zone_layers": sorted(zone_layers),
        "segment_count": segment_count,
        "via_count": via_count,
    }


def parse_drc_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {
            "available": False,
            "parse_error": "no DRC JSON was produced",
            "violation_count": None,
            "unconnected_count": None,
            "passes": False,
        }
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "available": False,
            "parse_error": f"invalid DRC JSON: {exc}",
            "violation_count": None,
            "unconnected_count": None,
            "passes": False,
        }

    violations = data.get("violations") or []
    unconnected = data.get("unconnected_items") or data.get("unconnected") or []
    violation_count = len(violations) if isinstance(violations, list) else 0
    unconnected_count = len(unconnected) if isinstance(unconnected, list) else 0
    return {
        "available": True,
        "parse_error": None,
        "violation_count": violation_count,
        "unconnected_count": unconnected_count,
        "passes": violation_count == 0 and unconnected_count == 0,
    }


def score_from_text(
    pcb_text: str,
    *,
    drc_json_text: str | None = None,
    drc_unavailable_reason: str | None = None,
    allow_structural_fallback: bool = False,
) -> dict[str, Any]:
    structural = structural_checks(pcb_text)
    drc = parse_drc_json(drc_json_text)
    if drc_unavailable_reason:
        drc["available"] = False
        drc["unavailable_reason"] = drc_unavailable_reason

    if drc["available"]:
        if not drc["passes"]:
            score = 0.0
            reason = "drc_failed"
        elif structural["all_required"]:
            score = 1.0
            reason = "drc_and_structure_passed"
        else:
            score = 0.5
            reason = "drc_passed_structure_failed"
    elif allow_structural_fallback:
        # Current admin VM has KiCad 9.0.8, while the submitted fixtures are
        # KiCad 10-format. This fallback keeps Stage 2 fixture replay useful
        # until Stage 4 installs a compatible KiCad for authoritative DRC.
        score = 1.0 if structural["all_required"] else 0.0
        reason = "drc_unavailable_structural_fallback"
    else:
        score = 0.0
        reason = "drc_unavailable"

    return {
        "score": score,
        "reason": reason,
        "drc": drc,
        "structural": structural,
    }


def score_file(
    path: Path,
    drc_json_path: Path | None = None,
    drc_unavailable_reason: str | None = None,
    allow_structural_fallback: bool = False,
) -> dict[str, Any]:
    pcb_text = path.read_text(encoding="utf-8", errors="replace")
    drc_json_text = None
    if drc_json_path and drc_json_path.exists():
        drc_json_text = drc_json_path.read_text(encoding="utf-8", errors="replace")
    return score_from_text(
        pcb_text,
        drc_json_text=drc_json_text,
        drc_unavailable_reason=drc_unavailable_reason,
        allow_structural_fallback=allow_structural_fallback,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcb", required=True, type=Path)
    parser.add_argument("--drc-json", type=Path)
    parser.add_argument("--drc-unavailable-reason")
    parser.add_argument("--allow-structural-fallback", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            score_file(
                args.pcb,
                drc_json_path=args.drc_json,
                drc_unavailable_reason=args.drc_unavailable_reason,
                allow_structural_fallback=args.allow_structural_fallback,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
