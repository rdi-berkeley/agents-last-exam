"""Replay millimetre tool-tip motions against an axis-aligned rectangular stock."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess


def _position(values):
    point = tuple(float(value) for value in values)
    if len(point) != 3 or not all(map(math.isfinite, point)):
        raise ValueError("Stock replay requires finite three-axis coordinates")
    return point


def replay_stock(
    stock,
    controllers,
    motions,
    output,
    *,
    resolution_mm,
    tool_tolerance_mm,
    native_binary,
    library_root,
    timeout_s=300,
):
    from native_profiles import cutter_profile

    for value in (resolution_mm, tool_tolerance_mm):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Native stock tolerances must be finite and positive")
    if stock.isNull() or not stock.isValid() or stock.Volume <= 0:
        raise ValueError("Native stock must be valid and nonempty")
    bounds = stock.BoundBox
    if not math.isclose(
        stock.Volume, bounds.XLength * bounds.YLength * bounds.ZLength, rel_tol=1e-10
    ):
        raise ValueError("Native stock simulation requires the original rectangular blank")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    native_binary = Path(native_binary).resolve(strict=True)
    manifest = json.loads((Path(library_root) / "original-tools.json").read_text())
    entries = {entry["id"]: entry for entry in manifest["tools"]}
    profiles = {
        number: cutter_profile(controller, entries[controller.OriginalToolId], tool_tolerance_mm)
        for number, controller in controllers.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    body = output.with_suffix(".motions.txt")
    native_input = output.with_suffix(".input.txt")
    if body.exists() or native_input.exists():
        raise FileExistsError("Native stock replay evidence already exists")
    digest = hashlib.sha256()
    report = {
        "motion_count": 0,
        "cut_motion_count": 0,
        "rapid_motion_count": 0,
        "resolution_mm": resolution_mm,
        "tool_tolerance_mm": tool_tolerance_mm,
        "maximum_chord_error_mm": 0.0,
        "safety_checked": False,
        "tool_changes": [],
        "backend": "CAMotics native CutSim",
        "tool_profiles": profiles,
    }
    with body.open("x") as stream:
        _write_motions(motions, controllers, stream, report, digest)
    with native_input.open("x") as stream:
        stock_bounds = [
            bounds.XMin,
            bounds.YMin,
            bounds.ZMin,
            bounds.XMax,
            bounds.YMax,
            bounds.ZMax,
        ]
        stream.write(" ".join(map(str, stock_bounds)) + "\n")
        stream.write(f"{resolution_mm}\n{len(profiles)}\n")
        for number, profile in profiles.items():
            stream.write(f"{number} {len(profile['components'])}\n")
            for component in profile["components"]:
                stream.write(" ".join(map(str, component)) + "\n")
        stream.write(f"{report['motion_count']}\n")
        with body.open() as source:
            shutil.copyfileobj(source, stream)
    with native_input.open() as source, output.with_suffix(".native.log").open("x") as log:
        result = subprocess.run(
            [str(native_binary), str(output)],
            stdin=source,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    native_counts = json.loads(result.stdout.strip().splitlines()[-1])
    if (
        native_counts["motion_count"] != report["motion_count"]
        or native_counts["feed_count"] != report["cut_motion_count"]
    ):
        raise RuntimeError("Native stock backend consumed different motion counts")
    with output.open("rb") as stream:
        stream.seek(80)
        facets = struct.unpack("<I", stream.read(4))[0]
        if not facets or output.stat().st_size != 84 + 50 * facets:
            raise RuntimeError("Native stock backend returned an incomplete STL")
        stream.seek(0)
        output_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with native_binary.open("rb") as stream:
        backend_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report.update(
        motions_sha256=digest.hexdigest(),
        output_sha256=output_hash,
        backend_sha256=backend_hash,
        native_counts=native_counts,
        output=str(output),
        output_facets=facets,
    )
    return report


def _write_motions(motions, controllers, stream, report, digest):
    previous_position = None
    previous_gauge = None
    active_tool = None
    for motion in motions:
        start = _position(motion["start_mm"])
        end = _position(motion["end_mm"])
        rapid = motion["rapid"]
        if type(rapid) is not bool:
            raise ValueError("Stock motion must distinguish rapid from cutting feed")
        tool = motion["tool"]
        if type(tool) is not int or (tool not in controllers and not (tool == 0 and rapid)):
            raise ValueError("Motion tool is absent from the original controller mapping")
        gauge_keys = ("gauge_start_mm", "gauge_end_mm", "tool_gauge_offset_mm")
        gauge_present = [key in motion for key in gauge_keys]
        gauge_start = gauge_end = calibration = None
        if any(gauge_present):
            if not all(gauge_present):
                raise ValueError("Stock replay requires complete gauge calibration metadata")
            gauge_start, gauge_end, calibration = (_position(motion[key]) for key in gauge_keys)
            for tip, gauge in ((start, gauge_start), (end, gauge_end)):
                calibrated = tuple(gauge[axis] - calibration[axis] for axis in range(3))
                if math.dist(tip, calibrated) > 1e-7:
                    raise ValueError("Tool tip does not match the declared gauge calibration")
        if previous_position is not None:
            if tool != active_tool and previous_gauge is not None and gauge_start is not None:
                distance = math.dist(previous_gauge, gauge_start)
            else:
                distance = math.dist(previous_position, start)
            if distance > 1e-7:
                raise ValueError("Stock motion stream is discontinuous")
        chord_error = float(motion.get("chord_error_mm", 0))
        if not math.isfinite(chord_error) or chord_error < 0:
            raise ValueError("Invalid motion approximation error")
        if tool != active_tool:
            active_tool = tool
            report["tool_changes"].append({"motion": report["motion_count"], "tool": tool})
        if rapid:
            report["rapid_motion_count"] += 1
        else:
            report["cut_motion_count"] += 1
        stream.write(" ".join(map(str, [tool, int(rapid), *start, *end])) + "\n")
        digest.update(
            json.dumps(
                {
                    "start_mm": start,
                    "end_mm": end,
                    "tool": tool,
                    "rapid": rapid,
                    "chord_error_mm": chord_error,
                    "gauge_start_mm": gauge_start,
                    "gauge_end_mm": gauge_end,
                    "tool_gauge_offset_mm": calibration,
                },
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        report["motion_count"] += 1
        report["maximum_chord_error_mm"] = max(report["maximum_chord_error_mm"], chord_error)
        previous_position = end
        previous_gauge = gauge_end
    report["final_position_mm"] = previous_position
