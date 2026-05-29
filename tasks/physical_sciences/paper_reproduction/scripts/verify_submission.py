"""Local verifiers for the physical_sciences/paper_reproduction benchmark family."""

from __future__ import annotations

import argparse
import json
import math
import sys
from io import BytesIO
from typing import Any

from PIL import Image, ImageStat

PNG_MIN_WIDTH = 400
PNG_MIN_HEIGHT = 400

HEAVY_NEUTRINO_EXPECTED_TASK_ID = "heavy_neutrino_xsec"
HEAVY_NEUTRINO_EXPECTED_PROCESS = "pp -> mu± N"
HEAVY_NEUTRINO_EXPECTED_MASSES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
HEAVY_NEUTRINO_EXPECTED_ENERGIES = ("7", "8", "14")
HEAVY_NEUTRINO_PASS_RELATIVE_TOLERANCE = 0.12

ZPRIME_PROD_EXPECTED_TASK_ID = "zprime_bl_production"
ZPRIME_PROD_EXPECTED_OBSERVABLE = "sigma(pp -> Z')"
ZPRIME_PROD_PANEL_SPECS = {
    "2.0": {
        "m_zprime_TeV": 2.0,
        "g1_prime_grid": [round(step * 0.01, 2) for step in range(21)],
        "gtilde_keys": ("0.0", "-0.05", "-0.1"),
    },
    "3.0": {
        "m_zprime_TeV": 3.0,
        "g1_prime_grid": [round(step * 0.05, 2) for step in range(15)],
        "gtilde_keys": ("0.0", "-0.3", "-0.6"),
    },
}
ZPRIME_PROD_PASS_RELATIVE_TOLERANCE = 0.18

ZPRIME_EXCL_EXPECTED_TASK_ID = "zprime_bl_exclusion"
ZPRIME_EXCL_EXPECTED_OBSERVABLE = "2 sigma LHC exclusion contour"
ZPRIME_EXCL_PANEL_SPECS = {
    "2.0": {"m_zprime_TeV": 2.0, "x_range": [-0.6, 0.4], "y_range": [0.0, 0.6]},
    "2.5": {"m_zprime_TeV": 2.5, "x_range": [-1.0, 0.6], "y_range": [0.0, 0.75]},
    "3.0": {"m_zprime_TeV": 3.0, "x_range": [-1.0, 0.8], "y_range": [0.0, 0.9]},
}
ZPRIME_EXCL_RESAMPLE_POINTS = 256
ZPRIME_EXCL_PASS_P95_DISTANCE = 0.05
ZPRIME_EXCL_PASS_MEAN_DISTANCE = 0.025

ZPRIME_DIMUON_EXPECTED_TASK_ID = "zprime_dimuon_scan"
ZPRIME_DIMUON_EXPECTED_OBSERVABLE = "sigma(pp -> Z' -> mu+ mu-)"
ZPRIME_DIMUON_EXPECTED_MASSES = [200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
ZPRIME_DIMUON_EXPECTED_CURVES = ("zprime_ssm", "zprime_psi")
ZPRIME_DIMUON_PASS_RELATIVE_TOLERANCE = 0.18


def _failure(reason: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "passed": False,
        "reason": reason,
        "max_relative_error": None,
        "mean_relative_error": None,
        "per_group_max_relative_error": {},
        "png_ok": False,
    }


def _load_json_bytes(raw: bytes) -> dict[str, Any]:
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("submission must be a JSON object")
    return payload


def _validate_png(png_bytes: bytes) -> tuple[bool, str, dict[str, int]]:
    try:
        image = Image.open(BytesIO(png_bytes))
        image.load()
    except Exception as exc:
        return False, f"png_open_failed:{type(exc).__name__}", {}

    width, height = image.size
    if width < PNG_MIN_WIDTH or height < PNG_MIN_HEIGHT:
        return False, "png_too_small", {"width": width, "height": height}

    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    if stat.stddev[0] < 1.0:
        return False, "png_nearly_blank", {"width": width, "height": height}

    return True, "ok", {"width": width, "height": height}


def _validate_positive_numeric_list(
    values: Any,
    *,
    expected_length: int,
    label: str,
) -> None:
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(f"length_mismatch:{label}")
    for value in values:
        if not isinstance(value, (int, float)):
            raise ValueError(f"non_numeric_value:{label}")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"non_positive_value:{label}")


def _validate_nonnegative_numeric_list(
    values: Any,
    *,
    expected_length: int,
    label: str,
) -> None:
    if not isinstance(values, list) or len(values) != expected_length:
        raise ValueError(f"length_mismatch:{label}")
    saw_positive = False
    for value in values:
        if not isinstance(value, (int, float)):
            raise ValueError(f"non_numeric_value:{label}")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"negative_value:{label}")
        if float(value) > 0:
            saw_positive = True
    if not saw_positive:
        raise ValueError(f"all_zero_values:{label}")


def _validate_exact_range(
    values: Any,
    *,
    expected: list[float],
    label: str,
) -> None:
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"range_mismatch:{label}")
    for got, want in zip(values, expected, strict=True):
        if not isinstance(got, (int, float)) or abs(float(got) - float(want)) > 1e-6:
            raise ValueError(f"range_mismatch:{label}")


def _score_numeric_groups(
    *,
    grouped_candidate: dict[str, list[float]],
    grouped_reference: dict[str, list[float]],
    pass_relative_tolerance: float,
    mismatch_reason: str,
) -> dict[str, Any]:
    relative_errors: list[float] = []
    per_group_max: dict[str, float] = {}
    for group_name, candidate_values in grouped_candidate.items():
        reference_values = grouped_reference[group_name]
        errors = []
        for candidate, reference in zip(candidate_values, reference_values, strict=True):
            rel = abs(float(candidate) - float(reference)) / float(reference)
            errors.append(rel)
            relative_errors.append(rel)
        per_group_max[group_name] = max(errors)

    mean_relative_error = sum(relative_errors) / len(relative_errors)
    max_relative_error = max(relative_errors)
    numeric_score = sum(max(0.0, 1.0 - (err / 0.25)) for err in relative_errors) / len(relative_errors)
    score = round(0.9 * numeric_score + 0.1, 6)
    passed = max_relative_error <= pass_relative_tolerance

    return {
        "score": 1.0 if passed else score,
        "passed": passed,
        "reason": "ok" if passed else mismatch_reason,
        "max_relative_error": round(max_relative_error, 6),
        "mean_relative_error": round(mean_relative_error, 6),
        "per_group_max_relative_error": {
            group_name: round(value, 6) for group_name, value in per_group_max.items()
        },
    }


def _validate_heavy_neutrino_schema(payload: dict[str, Any]) -> None:
    if payload.get("task_id") != HEAVY_NEUTRINO_EXPECTED_TASK_ID:
        raise ValueError("task_id_mismatch")
    if payload.get("process") != HEAVY_NEUTRINO_EXPECTED_PROCESS:
        raise ValueError("process_mismatch")
    if payload.get("masses_GeV") != HEAVY_NEUTRINO_EXPECTED_MASSES:
        raise ValueError("mass_grid_mismatch")

    cross_sections = payload.get("cross_sections_fb")
    if not isinstance(cross_sections, dict):
        raise ValueError("cross_sections_fb_missing")
    if tuple(cross_sections.keys()) != HEAVY_NEUTRINO_EXPECTED_ENERGIES:
        raise ValueError("energy_keys_mismatch")

    for energy in HEAVY_NEUTRINO_EXPECTED_ENERGIES:
        _validate_positive_numeric_list(
            cross_sections.get(energy),
            expected_length=len(HEAVY_NEUTRINO_EXPECTED_MASSES),
            label=energy,
        )


def score_submission_payload(
    agent_payload: dict[str, Any],
    reference_payload: dict[str, Any],
    *,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    _validate_heavy_neutrino_schema(reference_payload)
    _validate_heavy_neutrino_schema(agent_payload)

    png_ok, png_reason, png_meta = _validate_png(figure_png_bytes)
    if not png_ok:
        return {**_failure(png_reason), "png_meta": png_meta}

    grouped_candidate = agent_payload["cross_sections_fb"]
    grouped_reference = reference_payload["cross_sections_fb"]
    payload = _score_numeric_groups(
        grouped_candidate=grouped_candidate,
        grouped_reference=grouped_reference,
        pass_relative_tolerance=HEAVY_NEUTRINO_PASS_RELATIVE_TOLERANCE,
        mismatch_reason="cross_section_mismatch",
    )
    payload["png_ok"] = True
    payload["png_meta"] = png_meta
    return payload


def score_submission_bytes(
    *,
    agent_json_bytes: bytes,
    reference_json_bytes: bytes,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    agent_payload = _load_json_bytes(agent_json_bytes)
    reference_payload = _load_json_bytes(reference_json_bytes)
    return score_submission_payload(
        agent_payload,
        reference_payload,
        figure_png_bytes=figure_png_bytes,
    )


def _validate_zprime_production_schema(payload: dict[str, Any]) -> None:
    if payload.get("task_id") != ZPRIME_PROD_EXPECTED_TASK_ID:
        raise ValueError("task_id_mismatch")
    if payload.get("observable") != ZPRIME_PROD_EXPECTED_OBSERVABLE:
        raise ValueError("observable_mismatch")

    panels = payload.get("panels")
    if not isinstance(panels, dict):
        raise ValueError("panels_missing")
    if tuple(panels.keys()) != tuple(ZPRIME_PROD_PANEL_SPECS.keys()):
        raise ValueError("panel_keys_mismatch")

    for panel_key, panel_spec in ZPRIME_PROD_PANEL_SPECS.items():
        panel_payload = panels.get(panel_key)
        if not isinstance(panel_payload, dict):
            raise ValueError(f"panel_missing:{panel_key}")
        if float(panel_payload.get("m_zprime_TeV")) != panel_spec["m_zprime_TeV"]:
            raise ValueError(f"panel_mass_mismatch:{panel_key}")
        if panel_payload.get("g1_prime_grid") != panel_spec["g1_prime_grid"]:
            raise ValueError(f"g1_grid_mismatch:{panel_key}")

        cross_sections = panel_payload.get("cross_sections_pb")
        if not isinstance(cross_sections, dict):
            raise ValueError(f"cross_sections_pb_missing:{panel_key}")
        if tuple(cross_sections.keys()) != panel_spec["gtilde_keys"]:
            raise ValueError(f"gtilde_keys_mismatch:{panel_key}")

        expected_length = len(panel_spec["g1_prime_grid"])
        for gtilde_key in panel_spec["gtilde_keys"]:
            _validate_positive_numeric_list(
                cross_sections.get(gtilde_key),
                expected_length=expected_length,
                label=f"{panel_key}:{gtilde_key}",
            )


def _validate_contour_points(
    contour_points: Any,
    *,
    panel_key: str,
    x_range: list[float],
    y_range: list[float],
) -> None:
    if not isinstance(contour_points, list) or len(contour_points) < 40:
        raise ValueError(f"contour_points_invalid:{panel_key}")

    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    x_lo = x_range[0] - 0.05 * x_span
    x_hi = x_range[1] + 0.05 * x_span
    y_lo = y_range[0] - 0.05 * y_span
    y_hi = y_range[1] + 0.05 * y_span

    for point in contour_points:
        if not isinstance(point, dict):
            raise ValueError(f"contour_point_invalid:{panel_key}")
        gtilde = point.get("gtilde")
        g1_prime = point.get("g1_prime")
        if not isinstance(gtilde, (int, float)) or not isinstance(g1_prime, (int, float)):
            raise ValueError(f"contour_point_invalid:{panel_key}")
        if not math.isfinite(float(gtilde)) or not math.isfinite(float(g1_prime)):
            raise ValueError(f"contour_point_invalid:{panel_key}")
        if not (x_lo <= float(gtilde) <= x_hi and y_lo <= float(g1_prime) <= y_hi):
            raise ValueError(f"contour_point_out_of_bounds:{panel_key}")


def _validate_zprime_exclusion_schema(payload: dict[str, Any]) -> None:
    if payload.get("task_id") != ZPRIME_EXCL_EXPECTED_TASK_ID:
        raise ValueError("task_id_mismatch")
    if payload.get("observable") != ZPRIME_EXCL_EXPECTED_OBSERVABLE:
        raise ValueError("observable_mismatch")

    panels = payload.get("panels")
    if not isinstance(panels, dict):
        raise ValueError("panels_missing")
    if tuple(panels.keys()) != tuple(ZPRIME_EXCL_PANEL_SPECS.keys()):
        raise ValueError("panel_keys_mismatch")

    for panel_key, panel_spec in ZPRIME_EXCL_PANEL_SPECS.items():
        panel_payload = panels.get(panel_key)
        if not isinstance(panel_payload, dict):
            raise ValueError(f"panel_missing:{panel_key}")
        if float(panel_payload.get("m_zprime_TeV")) != panel_spec["m_zprime_TeV"]:
            raise ValueError(f"panel_mass_mismatch:{panel_key}")
        _validate_exact_range(panel_payload.get("x_range"), expected=panel_spec["x_range"], label=f"{panel_key}:x")
        _validate_exact_range(panel_payload.get("y_range"), expected=panel_spec["y_range"], label=f"{panel_key}:y")
        _validate_contour_points(
            panel_payload.get("contour_points"),
            panel_key=panel_key,
            x_range=panel_spec["x_range"],
            y_range=panel_spec["y_range"],
        )


def _panel_points(payload: dict[str, Any], panel_key: str) -> list[tuple[float, float]]:
    return [
        (float(point["gtilde"]), float(point["g1_prime"]))
        for point in payload["panels"][panel_key]["contour_points"]
    ]


def _validate_zprime_dimuon_schema(payload: dict[str, Any]) -> None:
    if payload.get("task_id") != ZPRIME_DIMUON_EXPECTED_TASK_ID:
        raise ValueError("task_id_mismatch")
    if payload.get("observable") != ZPRIME_DIMUON_EXPECTED_OBSERVABLE:
        raise ValueError("observable_mismatch")
    if payload.get("masses_GeV") != ZPRIME_DIMUON_EXPECTED_MASSES:
        raise ValueError("mass_grid_mismatch")

    curves = payload.get("cross_sections_pb")
    if not isinstance(curves, dict):
        raise ValueError("cross_sections_pb_missing")
    if tuple(curves.keys()) != ZPRIME_DIMUON_EXPECTED_CURVES:
        raise ValueError("curve_keys_mismatch")

    for curve_name in ZPRIME_DIMUON_EXPECTED_CURVES:
        _validate_positive_numeric_list(
            curves.get(curve_name),
            expected_length=len(ZPRIME_DIMUON_EXPECTED_MASSES),
            label=curve_name,
        )


def _resample_closed_polyline(points: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    if len(points) < 3:
        raise ValueError("contour_too_short")

    work_points = list(points)
    if math.hypot(work_points[0][0] - work_points[-1][0], work_points[0][1] - work_points[-1][1]) <= 1e-9:
        work_points = work_points[:-1]
    if len(work_points) < 3:
        raise ValueError("contour_too_short")

    segments = list(zip(work_points, work_points[1:] + [work_points[0]], strict=True))
    lengths = [math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in segments]
    total_length = sum(lengths)
    if total_length <= 0:
        raise ValueError("contour_zero_length")

    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)

    samples: list[tuple[float, float]] = []
    target_positions = [(step / count) * total_length for step in range(count)]
    seg_idx = 0
    for position in target_positions:
        while seg_idx < len(lengths) - 1 and cumulative[seg_idx + 1] < position:
            seg_idx += 1
        start = cumulative[seg_idx]
        end = cumulative[seg_idx + 1]
        (ax, ay), (bx, by) = segments[seg_idx]
        if end - start <= 1e-12:
            samples.append((ax, ay))
            continue
        t = (position - start) / (end - start)
        samples.append((ax + t * (bx - ax), ay + t * (by - ay)))
    return samples


def _symmetric_nearest_distance(
    candidate: list[tuple[float, float]],
    reference: list[tuple[float, float]],
) -> tuple[float, float]:
    def directed(source: list[tuple[float, float]], target: list[tuple[float, float]]) -> list[float]:
        distances = []
        for px, py in source:
            distances.append(min(math.hypot(px - qx, py - qy) for qx, qy in target))
        return distances

    cand_to_ref = directed(candidate, reference)
    ref_to_cand = directed(reference, candidate)
    combined_mean = max(sum(cand_to_ref) / len(cand_to_ref), sum(ref_to_cand) / len(ref_to_cand))
    sorted_cand = sorted(cand_to_ref)
    sorted_ref = sorted(ref_to_cand)
    p95_index = int(round(0.95 * (len(sorted_cand) - 1)))
    combined_p95 = max(sorted_cand[p95_index], sorted_ref[p95_index])
    return combined_mean, combined_p95


def score_exclusion_contour_submission_payload(
    agent_payload: dict[str, Any],
    reference_payload: dict[str, Any],
    *,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    _validate_zprime_exclusion_schema(reference_payload)
    _validate_zprime_exclusion_schema(agent_payload)

    png_ok, png_reason, png_meta = _validate_png(figure_png_bytes)
    if not png_ok:
        return {**_failure(png_reason), "png_meta": png_meta}

    per_group_mean: dict[str, float] = {}
    per_group_p95: dict[str, float] = {}
    panel_scores: list[float] = []
    max_mean = 0.0
    max_p95 = 0.0

    for panel_key, panel_spec in ZPRIME_EXCL_PANEL_SPECS.items():
        x_min, x_max = panel_spec["x_range"]
        y_min, y_max = panel_spec["y_range"]
        x_span = x_max - x_min
        y_span = y_max - y_min

        candidate_points = _panel_points(agent_payload, panel_key)
        reference_points = _panel_points(reference_payload, panel_key)
        candidate_norm = _resample_closed_polyline(
            [((x - x_min) / x_span, (y - y_min) / y_span) for x, y in candidate_points],
            ZPRIME_EXCL_RESAMPLE_POINTS,
        )
        reference_norm = _resample_closed_polyline(
            [((x - x_min) / x_span, (y - y_min) / y_span) for x, y in reference_points],
            ZPRIME_EXCL_RESAMPLE_POINTS,
        )
        mean_distance, p95_distance = _symmetric_nearest_distance(candidate_norm, reference_norm)
        per_group_mean[panel_key] = round(mean_distance, 6)
        per_group_p95[panel_key] = round(p95_distance, 6)
        max_mean = max(max_mean, mean_distance)
        max_p95 = max(max_p95, p95_distance)
        panel_scores.append(max(0.0, 1.0 - (p95_distance / 0.15)))

    passed = max_mean <= ZPRIME_EXCL_PASS_MEAN_DISTANCE and max_p95 <= ZPRIME_EXCL_PASS_P95_DISTANCE
    score = round(0.9 * (sum(panel_scores) / len(panel_scores)) + 0.1, 6)
    return {
        "score": 1.0 if passed else score,
        "passed": passed,
        "reason": "ok" if passed else "exclusion_contour_mismatch",
        "max_relative_error": round(max_p95, 6),
        "mean_relative_error": round(max_mean, 6),
        "per_group_max_relative_error": per_group_p95,
        "per_group_mean_distance": per_group_mean,
        "png_ok": True,
        "png_meta": png_meta,
    }


def score_exclusion_contour_submission_bytes(
    *,
    agent_json_bytes: bytes,
    reference_json_bytes: bytes,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    agent_payload = _load_json_bytes(agent_json_bytes)
    reference_payload = _load_json_bytes(reference_json_bytes)
    return score_exclusion_contour_submission_payload(
        agent_payload,
        reference_payload,
        figure_png_bytes=figure_png_bytes,
    )


def score_zprime_mass_scan_submission_payload(
    agent_payload: dict[str, Any],
    reference_payload: dict[str, Any],
    *,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    _validate_zprime_dimuon_schema(reference_payload)
    _validate_zprime_dimuon_schema(agent_payload)

    png_ok, png_reason, png_meta = _validate_png(figure_png_bytes)
    if not png_ok:
        return {**_failure(png_reason), "png_meta": png_meta}

    payload = _score_numeric_groups(
        grouped_candidate=agent_payload["cross_sections_pb"],
        grouped_reference=reference_payload["cross_sections_pb"],
        pass_relative_tolerance=ZPRIME_DIMUON_PASS_RELATIVE_TOLERANCE,
        mismatch_reason="zprime_mass_scan_mismatch",
    )
    payload["png_ok"] = True
    payload["png_meta"] = png_meta
    return payload


def score_zprime_mass_scan_submission_bytes(
    *,
    agent_json_bytes: bytes,
    reference_json_bytes: bytes,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    agent_payload = _load_json_bytes(agent_json_bytes)
    reference_payload = _load_json_bytes(reference_json_bytes)
    return score_zprime_mass_scan_submission_payload(
        agent_payload,
        reference_payload,
        figure_png_bytes=figure_png_bytes,
    )


def score_zprime_grid_submission_payload(
    agent_payload: dict[str, Any],
    reference_payload: dict[str, Any],
    *,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    _validate_zprime_production_schema(reference_payload)
    _validate_zprime_production_schema(agent_payload)

    png_ok, png_reason, png_meta = _validate_png(figure_png_bytes)
    if not png_ok:
        return {**_failure(png_reason), "png_meta": png_meta}

    grouped_candidate: dict[str, list[float]] = {}
    grouped_reference: dict[str, list[float]] = {}
    for panel_key, panel_spec in ZPRIME_PROD_PANEL_SPECS.items():
        candidate_panel = agent_payload["panels"][panel_key]["cross_sections_pb"]
        reference_panel = reference_payload["panels"][panel_key]["cross_sections_pb"]
        for gtilde_key in panel_spec["gtilde_keys"]:
            group_name = f"{panel_key}:{gtilde_key}"
            grouped_candidate[group_name] = candidate_panel[gtilde_key]
            grouped_reference[group_name] = reference_panel[gtilde_key]

    payload = _score_numeric_groups(
        grouped_candidate=grouped_candidate,
        grouped_reference=grouped_reference,
        pass_relative_tolerance=ZPRIME_PROD_PASS_RELATIVE_TOLERANCE,
        mismatch_reason="zprime_curve_mismatch",
    )
    payload["png_ok"] = True
    payload["png_meta"] = png_meta
    return payload


def score_zprime_grid_submission_bytes(
    *,
    agent_json_bytes: bytes,
    reference_json_bytes: bytes,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    agent_payload = _load_json_bytes(agent_json_bytes)
    reference_payload = _load_json_bytes(reference_json_bytes)
    return score_zprime_grid_submission_payload(
        agent_payload,
        reference_payload,
        figure_png_bytes=figure_png_bytes,
    )


def _validate_sampled_curves_schema(
    payload: dict[str, Any],
    *,
    reference_payload: dict[str, Any],
) -> None:
    if payload.get("task_id") != reference_payload.get("task_id"):
        raise ValueError("task_id_mismatch")
    if payload.get("observable") != reference_payload.get("observable"):
        raise ValueError("observable_mismatch")
    if payload.get("x_values") != reference_payload.get("x_values"):
        raise ValueError("x_values_mismatch")

    reference_series = reference_payload.get("series")
    payload_series = payload.get("series")
    if not isinstance(reference_series, dict) or not isinstance(payload_series, dict):
        raise ValueError("series_missing")
    if tuple(payload_series.keys()) != tuple(reference_series.keys()):
        raise ValueError("series_keys_mismatch")

    expected_length = len(reference_payload["x_values"])
    for series_name in reference_series:
        _validate_positive_numeric_list(
            payload_series.get(series_name),
            expected_length=expected_length,
            label=series_name,
        )


def score_sampled_curves_submission_payload(
    agent_payload: dict[str, Any],
    reference_payload: dict[str, Any],
    *,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    _validate_sampled_curves_schema(reference_payload, reference_payload=reference_payload)
    _validate_sampled_curves_schema(agent_payload, reference_payload=reference_payload)

    png_ok, png_reason, png_meta = _validate_png(figure_png_bytes)
    if not png_ok:
        return {**_failure(png_reason), "png_meta": png_meta}

    payload = _score_numeric_groups(
        grouped_candidate=agent_payload["series"],
        grouped_reference=reference_payload["series"],
        pass_relative_tolerance=0.2,
        mismatch_reason="sampled_curve_mismatch",
    )
    payload["png_ok"] = True
    payload["png_meta"] = png_meta
    return payload


def score_sampled_curves_submission_bytes(
    *,
    agent_json_bytes: bytes,
    reference_json_bytes: bytes,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    agent_payload = _load_json_bytes(agent_json_bytes)
    reference_payload = _load_json_bytes(reference_json_bytes)
    return score_sampled_curves_submission_payload(
        agent_payload,
        reference_payload,
        figure_png_bytes=figure_png_bytes,
    )


def _validate_panel_histograms_schema(
    payload: dict[str, Any],
    *,
    reference_payload: dict[str, Any],
) -> None:
    if payload.get("task_id") != reference_payload.get("task_id"):
        raise ValueError("task_id_mismatch")
    if payload.get("observable") != reference_payload.get("observable"):
        raise ValueError("observable_mismatch")

    reference_panels = reference_payload.get("panels")
    payload_panels = payload.get("panels")
    if not isinstance(reference_panels, dict) or not isinstance(payload_panels, dict):
        raise ValueError("panels_missing")
    if tuple(payload_panels.keys()) != tuple(reference_panels.keys()):
        raise ValueError("panel_keys_mismatch")

    for panel_key, panel_reference in reference_panels.items():
        panel_payload = payload_panels.get(panel_key)
        if not isinstance(panel_payload, dict):
            raise ValueError(f"panel_missing:{panel_key}")
        if panel_payload.get("x_values") != panel_reference.get("x_values"):
            raise ValueError(f"x_values_mismatch:{panel_key}")
        reference_series = panel_reference.get("series")
        payload_series = panel_payload.get("series")
        if not isinstance(reference_series, dict) or not isinstance(payload_series, dict):
            raise ValueError(f"series_missing:{panel_key}")
        if tuple(payload_series.keys()) != tuple(reference_series.keys()):
            raise ValueError(f"series_keys_mismatch:{panel_key}")
        expected_length = len(panel_reference["x_values"])
        for series_name in reference_series:
            _validate_nonnegative_numeric_list(
                payload_series.get(series_name),
                expected_length=expected_length,
                label=f"{panel_key}:{series_name}",
            )


def score_panel_histograms_submission_payload(
    agent_payload: dict[str, Any],
    reference_payload: dict[str, Any],
    *,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    _validate_panel_histograms_schema(reference_payload, reference_payload=reference_payload)
    _validate_panel_histograms_schema(agent_payload, reference_payload=reference_payload)

    png_ok, png_reason, png_meta = _validate_png(figure_png_bytes)
    if not png_ok:
        return {**_failure(png_reason), "png_meta": png_meta}

    grouped_candidate: dict[str, list[float]] = {}
    grouped_reference: dict[str, list[float]] = {}
    for panel_key in reference_payload["panels"]:
        for series_name in reference_payload["panels"][panel_key]["series"]:
            group_name = f"{panel_key}:{series_name}"
            grouped_candidate[group_name] = agent_payload["panels"][panel_key]["series"][series_name]
            grouped_reference[group_name] = reference_payload["panels"][panel_key]["series"][series_name]

    payload = _score_numeric_groups(
        grouped_candidate=grouped_candidate,
        grouped_reference=grouped_reference,
        pass_relative_tolerance=0.18,
        mismatch_reason="panel_histogram_mismatch",
    )
    payload["png_ok"] = True
    payload["png_meta"] = png_meta
    return payload


def score_panel_histograms_submission_bytes(
    *,
    agent_json_bytes: bytes,
    reference_json_bytes: bytes,
    figure_png_bytes: bytes,
) -> dict[str, Any]:
    agent_payload = _load_json_bytes(agent_json_bytes)
    reference_payload = _load_json_bytes(reference_json_bytes)
    return score_panel_histograms_submission_payload(
        agent_payload,
        reference_payload,
        figure_png_bytes=figure_png_bytes,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--png", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "heavy_neutrino_xsec",
            "zprime_bl_exclusion",
            "zprime_bl_production",
            "zprime_dimuon_scan",
            "vector_lq_u1_exclusion",
            "scalar_lq_mej",
            "vector_lq_eta_hist",
            "vector_lq_reach",
        ),
        default="heavy_neutrino_xsec",
    )
    args = parser.parse_args()

    try:
        if args.mode == "zprime_bl_exclusion":
            payload = score_exclusion_contour_submission_bytes(
                agent_json_bytes=open(args.agent, "rb").read(),
                reference_json_bytes=open(args.ref, "rb").read(),
                figure_png_bytes=open(args.png, "rb").read(),
            )
        elif args.mode == "zprime_bl_production":
            payload = score_zprime_grid_submission_bytes(
                agent_json_bytes=open(args.agent, "rb").read(),
                reference_json_bytes=open(args.ref, "rb").read(),
                figure_png_bytes=open(args.png, "rb").read(),
            )
        elif args.mode == "zprime_dimuon_scan":
            payload = score_zprime_mass_scan_submission_bytes(
                agent_json_bytes=open(args.agent, "rb").read(),
                reference_json_bytes=open(args.ref, "rb").read(),
                figure_png_bytes=open(args.png, "rb").read(),
            )
        elif args.mode == "vector_lq_eta_hist":
            payload = score_panel_histograms_submission_bytes(
                agent_json_bytes=open(args.agent, "rb").read(),
                reference_json_bytes=open(args.ref, "rb").read(),
                figure_png_bytes=open(args.png, "rb").read(),
            )
        elif args.mode in {
            "vector_lq_u1_exclusion",
            "scalar_lq_mej",
            "vector_lq_reach",
        }:
            payload = score_sampled_curves_submission_bytes(
                agent_json_bytes=open(args.agent, "rb").read(),
                reference_json_bytes=open(args.ref, "rb").read(),
                figure_png_bytes=open(args.png, "rb").read(),
            )
        else:
            payload = score_submission_bytes(
                agent_json_bytes=open(args.agent, "rb").read(),
                reference_json_bytes=open(args.ref, "rb").read(),
                figure_png_bytes=open(args.png, "rb").read(),
            )
    except FileNotFoundError as exc:
        payload = _failure(f"missing_file:{exc.filename}")
    except json.JSONDecodeError as exc:
        payload = _failure(f"json_error:{exc}")
    except ValueError as exc:
        payload = _failure(str(exc))
    except Exception as exc:  # pragma: no cover
        payload = _failure(f"unexpected_error:{type(exc).__name__}:{exc}")

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
