"""Aggregate evaluator-owned monitor extractions without evaluating model adequacy."""

import hashlib
import json
import math
from pathlib import Path

from .verification import CASES


def summarize_controllers(observations):
    summaries = {}
    for table, prefix, quantity in (
        ("stations", "DBC", "max_settlement_mm"),
        ("profiles", "CX", "max_disp_mm"),
    ):
        grouped = {}
        for row in observations[table]:
            point, displacement = row["xyz_m"], row["u_m"]
            if (
                len(point) != 3
                or len(displacement) != 3
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in point + displacement
                )
            ):
                raise ValueError("Invalid independent physical monitor field")
            value = (
                max(0.0, -displacement[2]) if table == "stations" else math.hypot(*displacement[:2])
            ) * 1000
            grouped.setdefault(row["monitor"], []).append((value, tuple(point)))
        if set(grouped) != {f"{prefix}{number}" for number in range(1, 11)}:
            raise ValueError("Independent extraction lacks the complete monitor set")
        maxima = []
        for monitor, rows in grouped.items():
            points = {point for _, point in rows}
            if table == "stations" and len(points) != 1:
                raise ValueError("DBC observation must be one physical station")
            if table == "profiles":
                if len({point[:2] for point in points}) != 1:
                    raise ValueError("CX observation must remain on its registered profile")
                elevations = [point[2] for point in points]
                if [min(elevations), max(elevations)] != observations["wall_range_z_m"]:
                    raise ValueError("Independent CX profile lacks wall top/toe coverage")
            maximum = max(value for value, _ in rows)
            locations = sorted({point for value, point in rows if value == maximum})
            maxima.append(
                {
                    "monitor": monitor,
                    "maximum_mm": maximum,
                    "xyz_m": [list(point) for point in locations],
                }
            )
        maxima.sort(key=lambda item: (-item["maximum_mm"], item["monitor"]))
        maximum = maxima[0]["maximum_mm"]
        if not math.isclose(maximum, observations[quantity], rel_tol=0, abs_tol=1e-8):
            raise ValueError("Independent summary disagrees with its monitor fields")
        summaries[quantity] = {
            "maximum_mm": maximum,
            "controllers": [row for row in maxima if row["maximum_mm"] == maximum],
            "next_distinct_monitor": maxima[1],
            "controller_margin_mm": maximum - maxima[1]["maximum_mm"],
            "monitor_count": len(grouped),
        }
    return summaries


def build_controller_evidence(records, baseline_input_hashes):
    cases, evidence_hashes = {}, {}
    for record in records:
        label = record["label"]
        if label in cases or record["case"] not in CASES:
            raise ValueError("Duplicate evidence label or unknown support height")
        artifacts = {}
        for name in ("terminal", "observations"):
            artifact = record[name]
            path = Path(artifact["path"])
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != artifact["sha256"]:
                raise ValueError("Retained observation evidence hash changed: " + str(path))
            artifacts[name] = json.loads(content)
            evidence_hashes[str(path)] = digest
        terminal = artifacts["terminal"]
        extraction = terminal["extraction"]
        if (
            terminal["case"] != record["case"]
            or terminal["input_hashes"] != record["input_hashes"]
            or terminal.get("native_exit_code") != 0
            or not terminal.get("source_fact_audit", {}).get("passed")
            or not extraction.get("native_replay_and_reader_completed")
        ):
            raise ValueError("Controller evidence requires matching audited native completion")
        if record["kind"] not in ("baseline", "diagnostic"):
            raise ValueError("Unknown controller evidence scope")
        if record["kind"] == "baseline" and record["input_hashes"] != baseline_input_hashes:
            raise ValueError("Baseline controller evidence concerns a different model")
        controllers = summarize_controllers(artifacts["observations"])
        for quantity, values in controllers.items():
            if not math.isclose(
                values["maximum_mm"], extraction[quantity], rel_tol=0, abs_tol=1e-8
            ):
                raise ValueError("Controller evidence differs from native terminal extraction")
        cases[label] = {
            "case": record["case"],
            "kind": record["kind"],
            "input_hashes": record["input_hashes"],
            "native_fields_sha256": extraction["native_hashes"],
            "controllers": controllers,
            "profile_extraction": artifacts["observations"]["profile_extrema"],
        }
    baseline_cases = [row["case"] for row in cases.values() if row["kind"] == "baseline"]
    if sorted(baseline_cases) != sorted(CASES):
        raise ValueError("Controller evidence requires each of the five baseline heights once")
    return {
        "format": "support-controller-evidence-1",
        "records": records,
        "baseline_input_hashes": baseline_input_hashes,
        "evidence_sha256": evidence_hashes,
        "cases": cases,
        "limits": (
            "Observed controllers, physical locations and margins in retained native results. "
            "Margins are not discretization error bounds or predictions for untested meshes. "
            "DBC values are discrete station observations. CX values concern the full registered "
            "vertical profiles, not maxima over all wall XY locations or the soil continuum. "
            "Native profile extraction completeness and FE approximation adequacy are distinct."
        ),
    }
