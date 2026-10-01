"""Physical and first-order optimality checks for the full-MEE fuel problem."""

from __future__ import annotations

import math

import numpy as np

from tasks.engineering.aerospace_low_thrust_trajectory.scripts import mee_dynamics as model
from tasks.engineering.aerospace_low_thrust_trajectory.scripts import cartesian_check

VERSION = "full_mee_j2_minimum_fuel_v2"
DEFECT_LIMIT = 1e-4
QUADRATURE_LIMIT = 1e-5
GAP_INTEGRAL_LIMIT = 1e-4
GAP_MAX_LIMIT = 2e-3
FUEL_ALLOWANCE_KG = 1.0


class EvaluatorReferenceError(RuntimeError):
    """The evaluator-controlled reference cannot grade this contract."""


def validate_incumbent(reference):
    incumbent = reference.get("tier3") if isinstance(reference, dict) else None
    if not isinstance(incumbent, dict) or incumbent.get("formulation") != VERSION:
        raise EvaluatorReferenceError("tier3 reference is not a verified v2 full-MEE/J2 incumbent")
    mass = incumbent.get("final_mass_kg")
    if (
        isinstance(mass, bool)
        or not isinstance(mass, (int, float))
        or not math.isfinite(mass)
        or not model.M0 - model.THRUST * model.TF / model.VE <= mass <= model.M0
    ):
        raise EvaluatorReferenceError("tier3 reference mass is outside its physical range")
    return incumbent


def _step(values, left_control, right_control, interval):
    def derivative(values, control):
        state_rate, costate_rate = model.canonical_rates(values[..., :7], values[..., 7:], control)
        return np.concatenate((state_rate, costate_rate), axis=-1)

    midpoint = (left_control + right_control) / 2
    first = derivative(values, left_control)
    second = derivative(values + interval * first / 2, midpoint)
    third = derivative(values + interval * second / 2, midpoint)
    fourth = derivative(values + interval * third, right_control)
    return values + interval * (first + 2 * second + 2 * third + fourth) / 6


def integrated_defects(trajectory, control):
    """All-interval Richardson-extrapolated canonical defects and error estimates."""
    values = np.column_stack(
        (trajectory[:, 1:8] / model.SCALE, trajectory[:, 8:15] * model.SCALE / model.M0)
    )
    defects = np.zeros(14)
    errors = np.zeros(14)
    costate_scale = np.maximum(1.0, np.max(np.abs(values[:, 7:]), axis=0))
    scale = np.r_[np.ones(7), costate_scale]
    for start in range(0, len(values) - 1, 4096):
        stop = min(start + 4096, len(values) - 1)
        left = values[start:stop]
        interval = np.diff(trajectory[start : stop + 1, 0])[:, None] / model.TF
        left_control, right_control = control[start:stop, 1:], control[start + 1 : stop + 1, 1:]
        midpoint = (left_control + right_control) / 2
        full = _step(left, left_control, right_control, interval)
        half = _step(left, left_control, midpoint, interval / 2)
        halves = _step(half, midpoint, right_control, interval / 2)
        correction = (halves - full) / 15
        extrapolated = halves + correction
        defects += np.sum(np.abs(values[start + 1 : stop + 1] - extrapolated) / scale, axis=0)
        errors += np.sum(np.abs(correction) / scale, axis=0)
    return defects, errors


def optimality_metrics(trajectory, control):
    state = trajectory[:, 1:8] / model.SCALE
    costate = trajectory[:, 8:15] * model.SCALE / model.M0
    derivative, adjoint = model.canonical_rates(state, costate, control[:, 1:])
    hamiltonian = np.sum(costate * derivative, axis=1)
    best, _ = model.minimizing_control(state, costate)
    minimum = np.sum(costate * model.normalized_rates(state, best), axis=1)
    gap = np.maximum(0, hamiltonian - minimum)
    times = trajectory[:, 0] / model.TF
    interval = np.diff(times)[:, None]
    midpoint_state = (state[:-1] + state[1:]) / 2 + interval * (
        derivative[:-1] - derivative[1:]
    ) / 8
    midpoint_costate = (costate[:-1] + costate[1:]) / 2 + interval * (
        adjoint[:-1] - adjoint[1:]
    ) / 8
    midpoint_control = (control[:-1, 1:] + control[1:, 1:]) / 2
    midpoint_hamiltonian = np.sum(
        midpoint_costate * model.normalized_rates(midpoint_state, midpoint_control), axis=1
    )
    midpoint_best, _ = model.minimizing_control(midpoint_state, midpoint_costate)
    midpoint_minimum = np.sum(
        midpoint_costate * model.normalized_rates(midpoint_state, midpoint_best), axis=1
    )
    midpoint_gap = np.maximum(0, midpoint_hamiltonian - midpoint_minimum)
    gap_integral = np.sum(np.diff(times) * (gap[:-1] + 4 * midpoint_gap + gap[1:]) / 6)
    all_hamiltonians = np.r_[hamiltonian, midpoint_hamiltonian]
    boundary = np.r_[
        (state[-1, :5] - model.TARGET) / np.r_[model.TARGET[0], np.ones(4)],
        costate[-1, 5],
        costate[-1, 6] + 1,
    ]
    return {
        "constraint_norm": float(np.linalg.norm(boundary)),
        "hamiltonian_initial": float(hamiltonian[0] * model.M0 / model.TF),
        "hamiltonian_final": float(hamiltonian[-1] * model.M0 / model.TF),
        "hamiltonian_range_normalized": float(np.ptp(all_hamiltonians)),
        "hamiltonian_scale_normalized": float(np.max(np.abs(all_hamiltonians))),
        "hamiltonian_gap_integral": float(gap_integral),
        "hamiltonian_gap_max": float(max(np.max(gap), np.max(midpoint_gap))),
    }


def check(results, reference, trajectory, control):
    """Return failures; references supply only a versioned fuel-quality incumbent."""
    incumbent = validate_incumbent(reference)
    if trajectory.ndim != 2 or trajectory.shape[1] != 15 or not 1000 <= len(trajectory) <= 2000000:
        return [
            f"tier3_trajectory.npy requires 1000..2000000 rows and 15 columns, got {trajectory.shape}"
        ]
    if control.shape != (len(trajectory), 4):
        return [f"tier3_control.npy has invalid shape {control.shape}"]
    if not np.isrealobj(trajectory) or not np.isrealobj(control):
        return ["tier3 arrays must be real"]
    if not np.all(np.isfinite(trajectory)) or not np.all(np.isfinite(control)):
        return ["tier3 arrays contain non-finite values"]
    times = trajectory[:, 0]
    if np.any(np.diff(times) <= 0) or not np.array_equal(times, control[:, 0]):
        return ["tier3 times must be strictly increasing and identical in both arrays"]
    if abs(times[0]) > 1e-8 or abs(times[-1] - model.TF) > 1e-6:
        return ["tier3 time domain must be exactly 0 to 25920000 seconds"]
    state = trajectory[:, 1:8]
    eccentricity = np.hypot(state[:, 1], state[:, 2])
    radius = state[:, 0] / (
        1 + state[:, 1] * np.cos(state[:, 5]) + state[:, 2] * np.sin(state[:, 5])
    )
    if np.any(state[:, 0] <= 0) or np.any(eccentricity >= 1) or np.any(radius <= model.RE):
        return ["tier3 trajectory must remain elliptic and outside Earth"]
    if np.any(state[:, 6] <= 0) or np.any(np.diff(state[:, 6]) > 1e-7):
        return ["tier3 mass must be positive and non-increasing"]
    if np.max(np.abs((state[0] - model.INITIAL) / model.SCALE)) > 1e-9:
        return ["tier3 initial MEE/mass state is wrong"]
    if np.max(np.linalg.norm(control[:, 1:], axis=1)) > 1 + 1e-10:
        return ["tier3 control norm exceeds unit magnitude"]
    if np.any(np.diff(state[:, 5]) <= 0) or np.max(np.diff(state[:, 5])) > 0.2:
        return ["tier3 unwrapped longitude increments must be in (0, 0.2] radians"]
    if (
        np.max(np.diff(times)) > 3600
        or np.max(np.linalg.norm(np.diff(control[:, 1:], axis=0), axis=1)) > 0.05
    ):
        return ["tier3 refine output: gaps <=3600 s and adjacent control changes <=0.05 required"]
    candidate = results.get("tier3", {})
    if not isinstance(candidate, dict) or candidate.get("formulation") != VERSION:
        return [f"tier3 formulation must be {VERSION}"]

    def number(section, key):
        value = section.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"tier3 field must be finite numeric: {key}")
        return float(value)

    failures = []
    final = state[-1]
    final_mass = float(final[6])
    fuel = model.M0 - final_mass
    final_ecc = float(eccentricity[-1])
    scalars = {
        "final_sma_km": (float(final[0] / (1 - final_ecc**2) / 1000), 1e-3),
        "final_eccentricity": (final_ecc, 1e-8),
        "final_inclination_deg": (
            float(np.rad2deg(2 * np.arctan(np.hypot(final[3], final[4])))),
            1e-6,
        ),
        "final_mass_kg": (final_mass, 1e-5),
        "fuel_consumed_kg": (fuel, 1e-5),
        "dv_total_m_s": (float(model.VE * np.log(model.M0 / final_mass)), 1e-3),
        "transfer_time_days": (model.TF / 86400.0, 1e-10),
    }
    for key, (expected, tolerance) in scalars.items():
        if abs(number(candidate, key) - expected) > tolerance:
            failures.append(f"tier3 {key} does not match the computed trajectory value")
    reference_mass = number(incumbent, "final_mass_kg")
    if final_mass < reference_mass - FUEL_ALLOWANCE_KG:
        failures.append("tier3 fuel exceeds the independently computed incumbent by more than 1 kg")
    if candidate.get("shooting_converged") is not True:
        failures.append("tier3 solver did not report convergence")
    initial_costates = np.asarray(candidate.get("initial_costates", []))
    if (
        initial_costates.shape != (7,)
        or not np.issubdtype(initial_costates.dtype, np.number)
        or not np.all(np.isfinite(initial_costates))
    ):
        failures.append("tier3 initial_costates must contain all seven finite SI costates")
    elif np.max(np.abs((initial_costates - trajectory[0, 8:]) * model.SCALE / model.M0)) > 1e-8:
        failures.append("tier3 initial_costates do not match the trajectory")
    metrics = optimality_metrics(trajectory, control)
    if not all(math.isfinite(value) for value in metrics.values()):
        return failures + ["tier3 non-finite physical/optimality diagnostics"]
    if metrics["constraint_norm"] > 1e-4:
        failures.append("tier3 recomputed terminal state/transversality norm exceeds 1e-4")
    if abs(number(candidate, "constraint_violation_norm") - metrics["constraint_norm"]) > 1e-7:
        failures.append("tier3 reported constraint norm differs from recomputed boundary residual")
    for key in ("hamiltonian_initial", "hamiltonian_final"):
        if abs(number(candidate, key) - metrics[key]) > 1e-10:
            failures.append(f"tier3 {key} differs from the recomputed Hamiltonian")
    if (
        metrics["hamiltonian_range_normalized"]
        > 1e-4 + 0.01 * metrics["hamiltonian_scale_normalized"]
    ):
        failures.append("tier3 full-history Hamiltonian drift exceeds tolerance")
    if (
        metrics["hamiltonian_gap_integral"] > GAP_INTEGRAL_LIMIT
        or metrics["hamiltonian_gap_max"] > GAP_MAX_LIMIT
    ):
        failures.append("tier3 control violates Pontryagin minimization including throttle/coast")
    defects, errors = integrated_defects(trajectory, control)
    if not np.all(np.isfinite(defects)) or not np.all(np.isfinite(errors)):
        failures.append("tier3 non-finite canonical propagation defects")
    elif np.max(errors) > QUADRATURE_LIMIT:
        failures.append("tier3 output mesh too coarse for canonical propagation certification")
    if np.max(defects[:7]) > DEFECT_LIMIT:
        failures.append("tier3 full MEE/J2 state or throttle mass-flow dynamics fail")
    if np.max(defects[7:]) > DEFECT_LIMIT:
        failures.append("tier3 seven-costate adjoint dynamics fail")
    if not failures:
        replay = cartesian_check.replay(trajectory, control)
        if not replay["success"]:
            failures.append("tier3 independent full-duration Cartesian replay failed")
        elif (
            not all(math.isfinite(value) for value in replay.values())
            or replay["terminal_orbit_norm"] > 1e-4
            or replay["position_discrepancy"] > 1e-3
            or replay["velocity_discrepancy"] > 1e-3
            or replay["mass_discrepancy_normalized"] > 1e-4
        ):
            failures.append(
                "tier3 independent Cartesian replay violates the endpoint or saved-state consistency limits"
            )
    return failures
