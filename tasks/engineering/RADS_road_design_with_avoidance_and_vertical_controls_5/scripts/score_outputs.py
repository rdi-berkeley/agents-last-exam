"""Local scorer for RADS road-design TSV submissions."""
from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

RUBRIC_WEIGHTS = {
    "output_tsv_present_readable_and_named": 0.04,
    "alignment_existence_and_naming": 0.04,
    "start_point_accuracy": 0.06,
    "end_point_accuracy": 0.06,
    "horizontal_geometry_compliance": 0.12,
    "spiral_compliance": 0.04,
    "plan_avoidance_of_control_objects": 0.20,
    "profile_existence_and_source_profile_type_correctness": 0.04,
    "must_pass_vertical_point_compliance": 0.15,
    "elevation_zone_compliance": 0.10,
    "grade_compliance": 0.07,
    "vertical_curve_compliance": 0.05,
    "tsv_format_and_sampling_consistency": 0.01,
    "tsv_geometry_elevation_consistency_with_final_alignment_and_profile": 0.07,
}


def _coerce_value(text: str) -> Any:
    value = text.strip()
    if not value:
        return ""
    if value.startswith("{") or value.startswith("["):
        return json.loads(value)
    lowered = value.lower()
    if lowered in frozenset({"true", "false"}):
        return lowered == "true"
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_required_columns(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return [*("station", "x", "y", "z")]


def parse_instance_spec_text(text: str) -> dict[str, Any]:
    parsed = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = _coerce_value(value)
    parsed["required_columns"] = _normalize_required_columns(
        parsed.get("required_columns", "station,x,y,z")
    )
    if "output_tsv" in parsed and "output_tsv_filename" not in parsed:
        parsed["output_tsv_filename"] = str(parsed["output_tsv"]).strip()
    if "sample_interval" in parsed and "sample_interval_m" not in parsed:
        parsed["sample_interval_m"] = parsed["sample_interval"]
    if "min_grade" in parsed and "min_grade_pct" not in parsed:
        parsed["min_grade_pct"] = parsed["min_grade"]
    if "max_grade" in parsed and "max_grade_pct" not in parsed:
        parsed["max_grade_pct"] = parsed["max_grade"]
    return parsed


def load_fixture_metrics_text(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    payload = json.loads(text)
    if isinstance(payload, dict):
        return payload
    return {}


def resolve_mode_fixture_manifest(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    if not payload:
        return {}
    if isinstance(payload.get("manifest"), dict):
        base = dict(payload["manifest"])
    elif isinstance(payload.get("base_manifest"), dict):
        base = dict(payload["base_manifest"])
    else:
        base = {
            key: value
            for key, value in payload.items()
            if key not in frozenset({"output_test_pos", "output_test_neg", "modes"})
        }
    mode_override = {}
    if isinstance(payload.get("modes"), dict) and isinstance(
        payload["modes"].get(mode), dict
    ):
        mode_override = payload["modes"][mode]
    elif isinstance(payload.get(mode), dict):
        mode_override = payload[mode]
    return _deep_merge(base, mode_override)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pair_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _path_length(points: list[dict[str, float]]) -> float:
    return sum(
        _pair_distance(points[i], points[i + 1]) for i in range(len(points) - 1)
    )


def _point_error(point: dict[str, float], expected: dict[str, Any] | None) -> float | None:
    if not expected:
        return None
    if "x" not in expected or "y" not in expected:
        return None
    return math.hypot(
        point["x"] - float(expected["x"]),
        point["y"] - float(expected["y"]),
    )


def _value_within_two_stage_tolerance(error: float | None, tolerance: float | None) -> float:
    if error is None:
        if tolerance is not None:
            return 0.0
        return 1.0
    tol = tolerance if (tolerance is not None and tolerance > 0) else 0.0
    if tol <= 0:
        if error <= 0:
            return 1.0
        return 0.0
    if error <= tol:
        return 1.0
    if error <= 2.0 * tol:
        return 0.5
    return 0.0


def _station_value(row: dict[str, float]) -> float:
    return float(row["station"])


def _interpolate_axis(points: list[dict[str, float]], station: float, axis: str) -> float | None:
    if not points:
        return None
    if station <= _station_value(points[0]):
        return float(points[0][axis])
    if station >= _station_value(points[-1]):
        return float(points[-1][axis])
    for index in range(len(points) - 1):
        left = points[index]
        right = points[index + 1]
        left_station = _station_value(left)
        right_station = _station_value(right)
        if left_station <= station <= right_station:
            span = right_station - left_station
            if abs(span) < 1e-9:
                return float(left[axis])
            ratio = (station - left_station) / span
            return float(left[axis]) + ratio * (float(right[axis]) - float(left[axis]))
    return None


def _turning_and_radius(a: dict[str, float], b: dict[str, float], c: dict[str, float]) -> tuple[float, float | None]:
    v1x = b["x"] - a["x"]
    v1y = b["y"] - a["y"]
    v2x = c["x"] - b["x"]
    v2y = c["y"] - b["y"]
    len1 = math.hypot(v1x, v1y)
    len2 = math.hypot(v2x, v2y)
    if len1 < 1e-9 or len2 < 1e-9:
        return (0.0, None)
    det = v1x * v2y - v1y * v2x
    dot = v1x * v2x + v1y * v2y
    turning = math.atan2(det, dot)
    ab = _pair_distance(a, b)
    bc = _pair_distance(b, c)
    ac = _pair_distance(a, c)
    area2 = abs(
        (b["x"] - a["x"]) * (c["y"] - a["y"]) - (b["y"] - a["y"]) * (c["x"] - a["x"])
    )
    if area2 < 1e-9:
        return (turning, None)
    radius = ab * bc * ac / (2.0 * area2)
    if math.isfinite(radius):
        return (turning, radius)
    return (turning, None)


def _analyze_horizontal_geometry(points: list[dict[str, float]]) -> dict[str, Any]:
    if len(points) < 3:
        return {
            "curve_count": 0,
            "min_radius_m": None,
            "curve_segments": [],
            "spiral_count": 0,
            "min_spiral_length_m": 0.0,
        }
    samples = []
    for index in range(1, len(points) - 1):
        turning, radius = _turning_and_radius(
            points[index - 1], points[index], points[index + 1]
        )
        station = points[index]["station"]
        if not radius or radius <= 0:
            curvature = 0.0
        else:
            curvature = math.copysign(1.0 / radius, turning)
        if abs(curvature) < 1e-9:
            sign = 0
        elif curvature > 0:
            sign = 1
        else:
            sign = -1
        samples.append(
            {
                "station": station,
                "turning": turning,
                "radius": radius,
                "curvature": curvature,
                "sign": sign,
            }
        )
    curve_segments = []
    current = None
    angle_threshold = math.radians(1.5)
    for sample in samples:
        active = (
            abs(sample["turning"]) >= angle_threshold
            and sample["radius"] is not None
        )
        if active:
            if current is None or current["sign"] != sample["sign"]:
                current = {
                    "sign": sample["sign"],
                    "start_station": sample["station"],
                    "end_station": sample["station"],
                    "radii": [sample["radius"]],
                    "curvature_trace": [abs(sample["curvature"])],
                }
                curve_segments.append(current)
            else:
                current["end_station"] = sample["station"]
                current["radii"].append(sample["radius"])
                current["curvature_trace"].append(abs(sample["curvature"]))
        else:
            current = None
    filtered_segments = []
    for segment in curve_segments:
        length_m = max(
            0.0, float(segment["end_station"]) - float(segment["start_station"])
        )
        if length_m <= 0.0 and len(segment["radii"]) < 2:
            continue
        filtered_segments.append(
            {
                "sign": segment["sign"],
                "start_station": float(segment["start_station"]),
                "end_station": float(segment["end_station"]),
                "length_m": length_m,
                "min_radius_m": min(
                    float(radius) for radius in segment["radii"] if radius
                ),
                "curvature_trace": list(segment["curvature_trace"]),
            }
        )
    spiral_count = 0
    min_spiral_length = 0.0
    for segment in filtered_segments:
        trace = segment["curvature_trace"]
        if len(trace) < 3:
            continue
        if not max(trace) > 0:
            continue
        rising = (
            trace[0] < trace[len(trace) // 2] and trace[len(trace) // 2] <= trace[-1]
        )
        falling = (
            trace[0] >= trace[len(trace) // 2] and trace[len(trace) // 2] > trace[-1]
        )
        if not rising and not falling:
            continue
        spiral_count += 1
        length_m = max(segment["length_m"], 0.0)
        if min_spiral_length == 0.0 or length_m < min_spiral_length:
            min_spiral_length = length_m
    min_radius = None
    if filtered_segments:
        min_radius = min(
            segment["min_radius_m"]
            for segment in filtered_segments
            if segment["min_radius_m"] is not None
        )
    return {
        "curve_count": len(filtered_segments),
        "min_radius_m": min_radius,
        "curve_segments": filtered_segments,
        "spiral_count": spiral_count,
        "min_spiral_length_m": min_spiral_length,
    }


def _analyze_vertical_geometry(points: list[dict[str, float]]) -> dict[str, Any]:
    if len(points) < 2:
        return {
            "grades_pct": [],
            "transition_count": 0,
            "max_transition_length_m": 0.0,
        }
    grades_pct = []
    for index in range(len(points) - 1):
        left = points[index]
        right = points[index + 1]
        ds = right["station"] - left["station"]
        if abs(ds) < 1e-9:
            continue
        grade_pct = (right["z"] - left["z"]) / ds * 100.0
        grades_pct.append({"station": right["station"], "grade_pct": grade_pct})
    if len(grades_pct) < 2:
        return {
            "grades_pct": grades_pct,
            "transition_count": 0,
            "max_transition_length_m": 0.0,
        }
    change_threshold = 0.25
    transitions = []
    start_station = None
    for index in range(1, len(grades_pct)):
        previous = grades_pct[index - 1]["grade_pct"]
        current = grades_pct[index]["grade_pct"]
        if abs(current - previous) >= change_threshold:
            if start_station is None:
                start_station = grades_pct[index - 1]["station"]
        else:
            if start_station is not None:
                transitions.append((start_station, grades_pct[index]["station"]))
                start_station = None
    if start_station is not None:
        transitions.append((start_station, grades_pct[-1]["station"]))
    max_length = 0.0
    for start_station, end_station in transitions:
        max_length = max(max_length, end_station - start_station)
    return {
        "grades_pct": grades_pct,
        "transition_count": len(transitions),
        "max_transition_length_m": max_length,
    }


def inspect_tsv_bytes(
    tsv_bytes,
    required_columns,
    sample_interval,
    output_filename,
    expected_filename,
) -> dict[str, Any]:
    result = {
        "exists": bool(tsv_bytes),
        "readable": False,
        "output_filename": output_filename,
        "expected_filename": expected_filename,
        "filename_score": 1.0 if output_filename == expected_filename else 0.0,
        "format_score": 0.0,
        "sampling_score": 0.0,
        "alignment_extractable": False,
        "header": [],
        "row_count": 0,
        "errors": [],
        "points": [],
        "metadata": {},
        "path_length_m": 0.0,
        "stations_monotonic": False,
    }
    if not tsv_bytes:
        result["errors"].append("missing tsv")
        return result
    text = tsv_bytes.decode("utf-8", errors="replace")
    logical_lines = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#") and ":" in stripped:
            meta_key, meta_value = stripped[1:].split(":", 1)
            result["metadata"][meta_key.strip()] = meta_value.strip()
        else:
            logical_lines.append(raw_line)
    reader = csv.reader(
        io.StringIO("\n".join(logical_lines)), delimiter="\t"
    )
    rows = list(reader)
    if not rows:
        result["errors"].append("empty tsv")
        return result
    header = [cell.strip() for cell in rows[0]]
    result["header"] = header
    normalized_header = [cell.lower() for cell in header]
    expected_header = [cell.lower() for cell in required_columns]
    if normalized_header == expected_header:
        result["format_score"] = 1.0
    elif all(column in normalized_header for column in expected_header):
        result["format_score"] = 0.5
    else:
        result["errors"].append(
            f"header mismatch: expected {expected_header}, got {normalized_header}"
        )
        return result
    indices = {}
    for column in ("station", "x", "y", "z"):
        if column not in normalized_header:
            result["errors"].append(f"required column missing: {column}")
            return result
        indices[column] = normalized_header.index(column)
    points = []
    previous_station = None
    monotonic = True
    for row_num, row in enumerate(rows[1:], start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) < len(header):
            result["errors"].append(f"row {row_num} has fewer columns than header")
            return result
        try:
            station = float(row[indices["station"]].strip())
            x = float(row[indices["x"]].strip())
            y = float(row[indices["y"]].strip())
            z = float(row[indices["z"]].strip())
        except Exception:
            result["errors"].append(f"row {row_num} has non-numeric station/x/y/z")
            return result
        if previous_station is not None and station <= previous_station:
            monotonic = False
        previous_station = station
        points.append({"station": station, "x": x, "y": y, "z": z})
    result["row_count"] = len(points)
    result["points"] = points
    result["stations_monotonic"] = monotonic
    if len(points) < 2:
        result["errors"].append("tsv must contain at least two numeric rows")
        return result
    result["readable"] = True
    result["path_length_m"] = _path_length(points)
    result["alignment_extractable"] = monotonic and result["path_length_m"] > 0.0
    if not monotonic:
        result["errors"].append("stations are not strictly increasing")
    if sample_interval is None:
        result["sampling_score"] = 1.0
        return result
    expected = float(sample_interval)
    tolerance = max(0.25, expected * 0.02)
    diffs = [
        points[i + 1]["station"] - points[i]["station"]
        for i in range(len(points) - 1)
    ]
    if not diffs:
        result["sampling_score"] = 0.0
        return result
    ratio = sum(1 for diff in diffs if abs(diff - expected) <= tolerance) / len(diffs)
    if ratio >= 0.999:
        result["sampling_score"] = 1.0
        return result
    if ratio >= 0.8:
        result["sampling_score"] = 0.5
        return result
    result["sampling_score"] = 0.0
    result["errors"].append(
        f"station intervals deviate from required sample interval {expected}"
    )
    return result


def _fraction_score(passed: int, total: int) -> float:
    if total <= 0:
        return 1.0
    ratio = passed / total
    if ratio >= 0.999:
        return 1.0
    if ratio >= 0.9:
        return 0.5
    return 0.0


def _mean(values: list[float]) -> float:
    if values:
        return sum(values) / len(values)
    return 1.0


def _observed_reference_consistency(
    points,
    fixture_manifest,
) -> tuple:
    reference_samples = fixture_manifest.get("reference_samples") or []
    if not reference_samples:
        return (1.0, 1.0, {"reference_samples_used": 0})
    xy_tolerance = _safe_float(fixture_manifest.get("reference_xy_tolerance_m"), 1.0)
    z_tolerance = _safe_float(fixture_manifest.get("reference_z_tolerance_m"), 0.5)
    xy_scores = []
    z_scores = []
    xy_errors = []
    z_errors = []
    for sample in reference_samples:
        station = _safe_float(sample.get("station"))
        agent_x = _interpolate_axis(points, station, "x")
        agent_y = _interpolate_axis(points, station, "y")
        agent_z = _interpolate_axis(points, station, "z")
        if agent_x is None or agent_y is None or agent_z is None:
            xy_scores.append(0.0)
            z_scores.append(0.0)
            continue
        xy_error = math.hypot(
            agent_x - _safe_float(sample.get("x")),
            agent_y - _safe_float(sample.get("y")),
        )
        z_error = abs(agent_z - _safe_float(sample.get("z")))
        xy_errors.append(xy_error)
        z_errors.append(z_error)
        xy_scores.append(_value_within_two_stage_tolerance(xy_error, xy_tolerance))
        z_scores.append(_value_within_two_stage_tolerance(z_error, z_tolerance))
    diagnostics = {
        "reference_samples_used": len(reference_samples),
        "max_reference_xy_error_m": max(xy_errors) if xy_errors else None,
        "max_reference_z_error_m": max(z_errors) if z_errors else None,
    }
    return (_mean(xy_scores), _mean(z_scores), diagnostics)


def score_rads_road_design_submission(
    instance_spec,
    fixture_manifest,
    tsv_inspection,
    mode,
) -> dict[str, Any]:
    required_columns = instance_spec.get("required_columns", [*("station", "x", "y", "z")])
    expected_filename = instance_spec.get("output_tsv_filename", "alignment_profile.tsv")
    points = list(tsv_inspection.get("points") or [])

    effective_manifest = _deep_merge(instance_spec, fixture_manifest)

    start_point = {
        "x": _safe_float(
            effective_manifest.get("start_point_x",
                                   effective_manifest.get("start_point", {}).get("x"))
        ),
        "y": _safe_float(
            effective_manifest.get("start_point_y",
                                   effective_manifest.get("start_point", {}).get("y"))
        ),
    }
    end_point = {
        "x": _safe_float(
            effective_manifest.get("end_point_x",
                                   effective_manifest.get("end_point", {}).get("x"))
        ),
        "y": _safe_float(
            effective_manifest.get("end_point_y",
                                   effective_manifest.get("end_point", {}).get("y"))
        ),
    }
    start_tol = _safe_float(
        effective_manifest.get(
            "start_point_tolerance_m",
            effective_manifest.get("start_point", {}).get("tolerance_m"),
        ),
        0.5,
    )
    end_tol = _safe_float(
        effective_manifest.get(
            "end_point_tolerance_m",
            effective_manifest.get("end_point", {}).get("tolerance_m"),
        ),
        0.5,
    )

    geometry = _analyze_horizontal_geometry(points)
    vertical = _analyze_vertical_geometry(points)
    xy_consistency, z_consistency, reference_diagnostics = _observed_reference_consistency(
        points, effective_manifest
    )

    # output_tsv_present_readable_and_named
    output_component = 0.0
    if tsv_inspection["exists"] and tsv_inspection["readable"]:
        output_component = 1.0 if tsv_inspection["filename_score"] >= 1.0 else 0.5
    elif tsv_inspection["exists"]:
        output_component = 0.5

    # alignment_existence_and_naming
    alignment_component = 1.0 if tsv_inspection["alignment_extractable"] else 0.0

    # start_point_accuracy
    start_error = (
        _point_error(points[0], start_point) if points else None
    )
    # end_point_accuracy
    end_error = (
        _point_error(points[-1], end_point) if points else None
    )
    start_component = _value_within_two_stage_tolerance(start_error, start_tol)
    end_component = _value_within_two_stage_tolerance(end_error, end_tol)

    # horizontal_geometry_compliance
    min_curve_count = int(_safe_float(effective_manifest.get("min_curve_count"), 0.0))
    min_curve_radius = _safe_float(effective_manifest.get("min_curve_radius_m"), 0.0)
    radius_tolerance = max(1.0, min_curve_radius * 0.05) if min_curve_radius > 0 else 1.0

    curve_count_score = 1.0
    if min_curve_count > 0:
        if geometry["curve_count"] >= min_curve_count:
            curve_count_score = 1.0
        elif geometry["curve_count"] == max(0, min_curve_count - 1):
            curve_count_score = 0.5
        else:
            curve_count_score = 0.0

    radius_score = 1.0
    if min_curve_radius > 0:
        observed_radius = _safe_float(geometry.get("min_radius_m"), 0.0)
        if observed_radius >= min_curve_radius:
            radius_score = 1.0
        elif observed_radius >= min_curve_radius - radius_tolerance:
            radius_score = 0.5
        else:
            radius_score = 0.0
    horizontal_component = _mean([curve_count_score, radius_score])

    # spiral_compliance
    require_spirals = bool(effective_manifest.get("require_spirals", False))
    min_spiral_length = _safe_float(effective_manifest.get("min_spiral_length_m"), 0.0)
    if not require_spirals:
        spiral_component = 1.0
    else:
        expected_spiral_count = int(
            _safe_float(
                effective_manifest.get("expected_spiral_count"),
                max(1, geometry["curve_count"]),
            )
        )
        if geometry["spiral_count"] >= expected_spiral_count:
            count_score = 1.0
        elif geometry["spiral_count"] > 0:
            count_score = 0.5
        else:
            count_score = 0.0
        length_score = 1.0
        if min_spiral_length > 0:
            observed_spiral_length = _safe_float(
                geometry.get("min_spiral_length_m"), 0.0
            )
            if observed_spiral_length >= min_spiral_length:
                length_score = 1.0
            elif observed_spiral_length >= min_spiral_length * 0.9:
                length_score = 0.5
            else:
                length_score = 0.0
        spiral_component = _mean([count_score, length_score])

    # plan_avoidance_of_control_objects
    control_object_scores = []
    control_object_diagnostics = []
    hard_fail_reasons = []
    for control_object in (
        effective_manifest.get("control_objects") or []
    ):
        radius = _safe_float(control_object.get("radius_m"), 0.0)
        setback = _safe_float(
            control_object.get("setback_m", control_object.get("required_clearance_m")),
            0.0,
        )
        tolerance = _safe_float(control_object.get("tolerance_m"), 0.5)
        min_clearance = None
        for point in points:
            clearance = (
                math.hypot(
                    point["x"] - _safe_float(control_object.get("x")),
                    point["y"] - _safe_float(control_object.get("y")),
                )
                - radius
            )
            if min_clearance is None or clearance < min_clearance:
                min_clearance = clearance
        violation = max(0.0, setback - _safe_float(min_clearance, 0.0))
        if violation <= 0:
            score = 1.0
        elif violation <= tolerance:
            score = 0.5
        else:
            score = 0.0
        if violation > 2.0 * tolerance:
            hard_fail_reasons.append(
                f"control_object_violation:{control_object.get('name', 'unnamed')}"
            )
        control_object_scores.append(score)
        control_object_diagnostics.append(
            {
                "name": control_object.get("name", "unnamed"),
                "required_clearance_m": setback,
                "min_clearance_m": min_clearance,
                "violation_m": violation,
            }
        )
    plan_avoidance_component = (
        _mean(control_object_scores) if control_object_scores else 1.0
    )

    # must_pass_vertical_point_compliance
    must_pass_scores = []
    must_pass_diagnostics = []
    for point_spec in (
        effective_manifest.get("must_pass_vertical_points") or []
    ):
        station = _safe_float(point_spec.get("station"))
        expected_elevation = _safe_float(point_spec.get("elevation"))
        tolerance = _safe_float(point_spec.get("tolerance_m"), 0.5)
        observed_elevation = _interpolate_axis(points, station, "z")
        error = (
            abs(observed_elevation - expected_elevation)
            if observed_elevation is not None
            else None
        )
        score = _value_within_two_stage_tolerance(error, tolerance)
        if error is not None and error > 2.0 * tolerance:
            hard_fail_reasons.append(
                f"must_pass_vertical_point_violation:{station}"
            )
        must_pass_scores.append(score)
        must_pass_diagnostics.append(
            {
                "station": station,
                "expected_elevation": expected_elevation,
                "observed_elevation": observed_elevation,
                "error_m": error,
            }
        )
    must_pass_component = _mean(must_pass_scores) if must_pass_scores else 1.0

    # elevation_zone_compliance
    zone_scores = []
    for zone in (effective_manifest.get("elevation_zones") or []):
        zone_points = [
            point
            for point in points
            if _safe_float(zone.get("start_station")) <= point["station"]
            <= _safe_float(zone.get("end_station"))
        ]
        if not zone_points:
            zone_scores.append(0.0)
            continue
        within = 0
        for point in zone_points:
            if (
                _safe_float(zone.get("min_elevation"), -1e18) <= point["z"]
                <= _safe_float(zone.get("max_elevation"), 1e18)
            ):
                within += 1
        zone_scores.append(_fraction_score(within, len(zone_points)))
    elevation_zone_component = _mean(zone_scores) if zone_scores else 1.0

    # grade_compliance
    min_grade_pct = effective_manifest.get("min_grade_pct")
    max_grade_pct = effective_manifest.get("max_grade_pct")
    grade_tolerance_pct = _safe_float(effective_manifest.get("grade_tolerance_pct"), 0.25)
    grade_scores = []
    if min_grade_pct is not None or max_grade_pct is not None:
        for grade_sample in vertical["grades_pct"]:
            grade_value = grade_sample["grade_pct"]
            lower_ok = True
            upper_ok = True
            lower_partial = True
            upper_partial = True
            if min_grade_pct is not None:
                lower_ok = grade_value >= _safe_float(min_grade_pct)
                lower_partial = grade_value >= _safe_float(min_grade_pct) - grade_tolerance_pct
            if max_grade_pct is not None:
                upper_ok = grade_value <= _safe_float(max_grade_pct)
                upper_partial = grade_value <= _safe_float(max_grade_pct) + grade_tolerance_pct
            if lower_ok and upper_ok:
                grade_scores.append(1.0)
            elif lower_partial and upper_partial:
                grade_scores.append(0.5)
            else:
                grade_scores.append(0.0)
    grade_component = _mean(grade_scores) if grade_scores else 1.0

    # vertical_curve_compliance
    vertical_curve_constraints = (
        effective_manifest.get("vertical_curve_constraints") or {}
    )
    if vertical_curve_constraints:
        required_count = int(
            _safe_float(vertical_curve_constraints.get("required_count"), 1.0)
        )
        min_length_m = _safe_float(vertical_curve_constraints.get("min_length_m"), 0.0)
        tolerance_m = _safe_float(vertical_curve_constraints.get("tolerance_m"), 5.0)
        if vertical["transition_count"] >= required_count:
            count_score = 1.0
        elif vertical["transition_count"] > 0:
            count_score = 0.5
        else:
            count_score = 0.0
        if min_length_m <= 0:
            length_score = 1.0
        elif vertical["max_transition_length_m"] >= min_length_m:
            length_score = 1.0
        elif vertical["max_transition_length_m"] >= min_length_m - tolerance_m:
            length_score = 0.5
        else:
            length_score = 0.0
        vertical_curve_component = _mean([count_score, length_score])
    else:
        vertical_curve_component = 1.0

    # profile_existence_and_source_profile_type_correctness
    profile_type = str(
        effective_manifest.get("profile_type", "design")
    ).strip().lower()
    if points:
        profile_exists = any(
            abs(points[i + 1]["z"] - points[i]["z"]) > 1e-9
            for i in range(len(points) - 1)
        ) or bool(points)
    else:
        profile_exists = False

    if profile_type == "existing_ground":
        profile_component = _mean(
            [1.0 if profile_exists else 0.0, z_consistency]
        )
    else:
        profile_component = _mean(
            [
                1.0 if profile_exists else 0.0,
                must_pass_component,
                elevation_zone_component,
                grade_component,
                vertical_curve_component,
            ]
        )

    # tsv_format_and_sampling_consistency
    tsv_consistency_component = _mean(
        [
            tsv_inspection.get("format_score", 0.0),
            tsv_inspection.get("sampling_score", 0.0),
        ]
    )

    # tsv_geometry_elevation_consistency_with_final_alignment_and_profile
    geometry_elevation_component = _mean([xy_consistency, z_consistency])

    components = {
        "output_tsv_present_readable_and_named": output_component,
        "alignment_existence_and_naming": alignment_component,
        "start_point_accuracy": start_component,
        "end_point_accuracy": end_component,
        "horizontal_geometry_compliance": horizontal_component,
        "spiral_compliance": spiral_component,
        "plan_avoidance_of_control_objects": plan_avoidance_component,
        "profile_existence_and_source_profile_type_correctness": profile_component,
        "must_pass_vertical_point_compliance": must_pass_component,
        "elevation_zone_compliance": elevation_zone_component,
        "grade_compliance": grade_component,
        "vertical_curve_compliance": vertical_curve_component,
        "tsv_format_and_sampling_consistency": tsv_consistency_component,
        "tsv_geometry_elevation_consistency_with_final_alignment_and_profile": geometry_elevation_component,
    }

    # hard-fail checks
    if not tsv_inspection["exists"]:
        hard_fail_reasons.append("missing_tsv")
    if tsv_inspection["exists"] and not tsv_inspection["readable"]:
        hard_fail_reasons.append("unreadable_tsv")
    if tsv_inspection["readable"] and not all(
        c.lower() in [h.lower() for h in tsv_inspection["header"]]
        for c in required_columns
    ):
        hard_fail_reasons.append("missing_required_columns")
    if tsv_inspection["readable"] and not tsv_inspection["alignment_extractable"]:
        hard_fail_reasons.append("no_valid_alignment_extractable")

    weighted_components = {
        key: RUBRIC_WEIGHTS[key] * value for key, value in components.items()
    }
    raw_score = sum(weighted_components.values())
    final_score = 0.0 if hard_fail_reasons else raw_score

    return {
        "mode": mode,
        "score": round(min(max(final_score, 0.0), 1.0), 6),
        "raw_score_before_hard_fail": round(min(max(raw_score, 0.0), 1.0), 6),
        "hard_fail_reasons": sorted(set(hard_fail_reasons)),
        "weights": RUBRIC_WEIGHTS,
        "components": components,
        "weighted_components": weighted_components,
        "observed": {
            "path_length_m": tsv_inspection.get("path_length_m"),
            "curve_count": geometry["curve_count"],
            "min_radius_m": geometry["min_radius_m"],
            "spiral_count": geometry["spiral_count"],
            "min_spiral_length_m": geometry["min_spiral_length_m"],
            "grade_sample_count": len(vertical["grades_pct"]),
            "vertical_transition_count": vertical["transition_count"],
            "max_vertical_transition_length_m": vertical["max_transition_length_m"],
            "start_point_error_m": start_error,
            "end_point_error_m": end_error,
            "control_objects": control_object_diagnostics,
            "must_pass_vertical_points": must_pass_diagnostics,
            "reference_consistency": reference_diagnostics,
        },
        "tsv": {
            "exists": tsv_inspection["exists"],
            "readable": tsv_inspection["readable"],
            "output_filename": tsv_inspection["output_filename"],
            "expected_filename": expected_filename,
            "row_count": tsv_inspection["row_count"],
            "header": tsv_inspection["header"],
            "stations_monotonic": tsv_inspection["stations_monotonic"],
            "errors": list(tsv_inspection["errors"]),
        },
    }
