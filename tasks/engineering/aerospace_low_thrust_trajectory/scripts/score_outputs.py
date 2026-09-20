"""Scoring helpers for aerospace_low_thrust_trajectory."""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from tasks.engineering.aerospace_low_thrust_trajectory.scripts.tier3_contract import (
    EvaluatorReferenceError,
    validate_incumbent,
)


MU = 3.986004418e14
A_GEO_KM = 42164.0
A_LEO_M = 6678.0e3
M0 = 2000.0
THRUST = 0.5
ISP = 3000.0
G0 = 9.80665
VE = ISP * G0
I_LEO_DEG = 28.5

REQUIRED_FILES = (
    "results.json",
    "tier2_trajectory.npy",
    "tier3_trajectory.npy",
    "tier3_control.npy",
)


@dataclass(frozen=True)
class ScoreReport:
    score: float
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "failures": list(self.failures),
        }


def _load_json(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("results.json payload must be bytes")
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("results.json must contain a JSON object")
    return data


def _load_npy(payload: bytes) -> np.ndarray:
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("NumPy payload must be bytes")
    data = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(data, np.ndarray):
        data.close()
        raise ValueError("Expected a NumPy array, not an archive")
    return data


def _number(data: dict[str, Any], path: str) -> float:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"missing numeric field: {path}")
        current = current[part]
    if not isinstance(current, (int, float)) or isinstance(current, bool):
        raise ValueError(f"field must be numeric: {path}")
    value = float(current)
    if not math.isfinite(value):
        raise ValueError(f"field must be finite: {path}")
    return value


def _bool(data: dict[str, Any], path: str) -> bool:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"missing boolean field: {path}")
        current = current[part]
    if not isinstance(current, bool):
        raise ValueError(f"field must be boolean: {path}")
    return current


def _rel_err(value: float, expected: float) -> float:
    return abs(value - expected) / max(abs(expected), 1e-12)


def _close(value: float, expected: float, *, atol: float, rtol: float = 0.0) -> bool:
    return abs(value - expected) <= atol + rtol * abs(expected)


def _orbital_elements_from_cartesian(row: np.ndarray) -> tuple[float, float, float]:
    r_vec = np.asarray(row[1:4], dtype=float)
    v_vec = np.asarray(row[4:7], dtype=float)
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    if r <= 0.0 or h <= 0.0:
        raise ValueError("invalid Cartesian trajectory state")
    energy = 0.5 * v * v - MU / r
    a = -MU / (2.0 * energy)
    e_vec = ((v * v - MU / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / MU
    ecc = np.linalg.norm(e_vec)
    inc = math.degrees(math.acos(float(np.clip(h_vec[2] / h, -1.0, 1.0))))
    return a / 1000.0, float(ecc), inc


def _mee_inclination_deg(h: float, k: float) -> float:
    return math.degrees(2.0 * math.atan(math.hypot(h, k)))


def _check_required(candidate: dict[str, bytes]) -> list[str]:
    failures: list[str] = []
    for name in REQUIRED_FILES:
        if name not in candidate:
            failures.append(f"missing candidate file: {name}")
    return failures


def _check_tier1(results: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    checks = {
        "tier1.transfer_orbit_sma_km": (1e-6, 0.0),
        "tier1.dv1_m_s": (1.0, 0.0),
        "tier1.dv2_m_s": (1.0, 0.0),
        "tier1.dv_total_m_s": (1.0, 0.0),
        "tier1.transfer_time_s": (1.0, 0.0),
        "tier1.transfer_time_hours": (1e-3, 0.0),
        "tier1.v_circular_leo_m_s": (1.0, 0.0),
        "tier1.v_circular_geo_m_s": (1.0, 0.0),
    }
    for path, (atol, rtol) in checks.items():
        if not _close(_number(results, path), _number(reference, path), atol=atol, rtol=rtol):
            failures.append(f"{path} outside tolerance")
    return failures


def _interp_columns(source: np.ndarray, target_t: np.ndarray, columns: list[int]) -> np.ndarray:
    return np.column_stack([np.interp(target_t, source[:, 0], source[:, col]) for col in columns])


def _tier2_derivative(time: float, state: np.ndarray) -> np.ndarray:
    position = state[:3]
    velocity = state[3:6]
    radius = np.linalg.norm(position)
    speed = np.linalg.norm(velocity)
    mass = state[6]
    if radius <= 0.0 or speed <= 0.0 or mass <= 0.0:
        raise ValueError("tier2 trajectory has invalid state for dynamics check")
    acceleration = -MU * position / radius**3 + (THRUST / mass) * velocity / speed
    return np.concatenate((velocity, acceleration, [-THRUST / VE]))


def _propagate_tier2(initial: np.ndarray, final_time: float) -> np.ndarray:
    if final_time == initial[0]:
        return initial[1:].copy()
    solution = solve_ivp(
        _tier2_derivative,
        (float(initial[0]), float(final_time)),
        initial[1:],
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        max_step=500.0,
        t_eval=[float(final_time)],
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise ValueError("tier2 dynamics propagation failed")
    return solution.y[:, -1]


def _tier2_dynamics_residuals(traj: np.ndarray) -> np.ndarray:
    """Return position/velocity propagation defects in acceleration units."""
    residuals = []
    for index in np.unique(np.linspace(0, traj.shape[0] - 2, min(80, traj.shape[0] - 1), dtype=int)):
        previous = traj[index]
        following = traj[index + 1]
        interval = float(following[0] - previous[0])
        if interval <= 0.0:
            raise ValueError("tier2 trajectory has non-positive propagation interval")
        propagated = _propagate_tier2(previous, float(following[0]))
        position_defect = 2.0 * np.linalg.norm(following[1:4] - propagated[:3]) / interval**2
        velocity_defect = np.linalg.norm(following[4:7] - propagated[3:6]) / interval
        residuals.append((position_defect, velocity_defect))
    return np.asarray(residuals)


def _check_tier2(
    results: dict[str, Any],
    reference: dict[str, Any],
    traj: np.ndarray,
    reference_traj: np.ndarray,
) -> list[str]:
    failures: list[str] = []
    if traj.ndim != 2 or traj.shape[1] != 8 or traj.shape[0] < 1000:
        return [f"tier2_trajectory.npy has invalid shape {traj.shape}"]
    if not np.all(np.isfinite(traj)):
        return ["tier2 trajectory contains non-finite values"]
    if np.any(np.diff(traj[:, 0]) <= 0.0):
        return ["tier2 trajectory time must be strictly increasing"]
    if abs(float(traj[0, 0])) > 1e-6:
        failures.append("tier2 trajectory does not start at t=0")
    if abs(float(traj[0, 7]) - M0) > 1e-3:
        failures.append("tier2 trajectory initial mass is not 2000 kg")
    expected_r0 = np.array([A_LEO_M, 0.0, 0.0])
    v_circ = math.sqrt(MU / A_LEO_M)
    expected_v0 = np.array([
        0.0,
        v_circ * math.cos(math.radians(I_LEO_DEG)),
        v_circ * math.sin(math.radians(I_LEO_DEG)),
    ])
    if np.linalg.norm(traj[0, 1:4] - expected_r0) > 1.0:
        failures.append("tier2 trajectory initial position is wrong")
    if np.linalg.norm(traj[0, 4:7] - expected_v0) > 1e-2:
        failures.append("tier2 trajectory initial velocity is wrong")
    if np.any(traj[:, 7] <= 0.0) or np.any(np.diff(traj[:, 7]) > 1e-6):
        failures.append("tier2 mass must remain positive and non-increasing")
    if np.max(np.abs(traj[:, 7] - (M0 - THRUST * traj[:, 0] / VE))) > 2.0:
        failures.append("tier2 mass history does not satisfy constant-thrust mass flow")

    final_sma = _number(results, "tier2.final_sma_km")
    final_ecc = _number(results, "tier2.final_eccentricity")
    dv_total = _number(results, "tier2.dv_total_m_s")
    transfer_time = _number(results, "tier2.transfer_time_s")
    final_mass = _number(results, "tier2.final_mass_kg")
    fuel = _number(results, "tier2.fuel_consumed_kg")
    edelbaum = _number(results, "tier2.edelbaum_dv_m_s")

    if _rel_err(final_sma, A_GEO_KM) > 0.01:
        failures.append("tier2 final SMA is outside 1% GEO tolerance")
    if final_ecc > 0.01:
        failures.append("tier2 final eccentricity is too high")
    if abs(dv_total - edelbaum) / edelbaum > 0.10:
        failures.append("tier2 delta-v is not within 10% of Edelbaum")
    if not _close(dv_total, _number(reference, "tier2.dv_total_m_s"), atol=250.0, rtol=0.05):
        failures.append("tier2 delta-v is implausible relative to hidden reference")
    expected_mass = M0 - fuel
    rocket_mass = M0 * math.exp(-dv_total / VE)
    if abs(final_mass - expected_mass) > 2.0:
        failures.append("tier2 final mass and fuel consumed are inconsistent")
    if abs(final_mass - rocket_mass) > 15.0:
        failures.append("tier2 final mass is inconsistent with rocket equation")
    if abs(float(traj[-1, 0]) - transfer_time) > max(10.0, 0.002 * transfer_time):
        failures.append("tier2 trajectory final time does not match results.json")
    if abs(float(traj[-1, 7]) - final_mass) > 2.0:
        failures.append("tier2 trajectory final mass does not match results.json")

    try:
        a_from_state, ecc_from_state, inc_from_state = _orbital_elements_from_cartesian(traj[-1])
    except ValueError as exc:
        failures.append(str(exc))
    else:
        if _rel_err(a_from_state, final_sma) > 0.01:
            failures.append("tier2 final state SMA does not match results.json")
        if abs(ecc_from_state - final_ecc) > 0.02:
            failures.append("tier2 final state eccentricity does not match results.json")
        if abs(inc_from_state - _number(results, "tier2.final_inclination_deg")) > 1.0:
            failures.append("tier2 final state inclination does not match results.json")

    if reference_traj.ndim == 2 and reference_traj.shape[1] == 8:
        if _rel_err(float(traj[-1, 0]), float(reference_traj[-1, 0])) > 0.03:
            failures.append("tier2 transfer duration is implausible relative to hidden reference")
        if failures:
            return failures
        sample_t = np.linspace(max(traj[0, 0], reference_traj[0, 0]), min(traj[-1, 0], reference_traj[-1, 0]), 25)
        sample_indices = np.searchsorted(traj[:, 0], sample_t).clip(0, traj.shape[0] - 1)
        sample_indices[traj[sample_indices, 0] > reference_traj[-1, 0]] -= 1
        sample_t = traj[sample_indices, 0]
        cand_sample = traj[sample_indices, 1:]
        reference_indices = (np.searchsorted(reference_traj[:, 0], sample_t, side="right") - 1).clip(0)
        ref_sample = np.asarray([
            _propagate_tier2(reference_traj[index], float(time))
            for index, time in zip(reference_indices, sample_t)
        ])
        pos_err_km = np.linalg.norm(cand_sample[:, 0:3] - ref_sample[:, 0:3], axis=1) / 1000.0
        vel_err = np.linalg.norm(cand_sample[:, 3:6] - ref_sample[:, 3:6], axis=1)
        mass_err = np.abs(cand_sample[:, 6] - ref_sample[:, 6])
        if float(np.median(pos_err_km)) > 2500.0:
            failures.append("tier2 trajectory position history is implausible relative to hidden reference")
        if float(np.median(vel_err)) > 350.0:
            failures.append("tier2 trajectory velocity history is implausible relative to hidden reference")
        if float(np.median(mass_err)) > 10.0:
            failures.append("tier2 trajectory mass history is implausible relative to hidden reference")

    if failures:
        return failures
    residuals = _tier2_dynamics_residuals(traj)
    if np.any(np.median(residuals, axis=0) > 0.005) or np.max(residuals) > 0.08:
        failures.append("tier2 trajectory does not satisfy gravity+tangential-thrust dynamics")

    return failures


def _check_tier3(
    results: dict[str, Any],
    reference: dict[str, Any],
    trajectory: np.ndarray,
    control: np.ndarray,
    reference_trajectory: np.ndarray,
    reference_control: np.ndarray,
) -> list[str]:
    from tasks.engineering.aerospace_low_thrust_trajectory.scripts.tier3_contract import check

    _validate_reference_tier3(reference, reference_trajectory, reference_control)
    return check(results, reference, trajectory, control)


def _validate_reference_tier3(reference, trajectory, control):
    validate_incumbent(reference)
    if (
        trajectory.ndim != 2
        or trajectory.shape[1] != 15
        or not 1000 <= len(trajectory) <= 2000000
        or control.shape != (len(trajectory), 4)
    ):
        raise EvaluatorReferenceError("tier3 v2 reference requires a 15-column trajectory and aligned control array")
    for array in (trajectory, control):
        if not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array) or not np.all(np.isfinite(array)):
            raise EvaluatorReferenceError("tier3 reference arrays must be finite real numbers")
    if not np.array_equal(trajectory[:, 0], control[:, 0]) or np.any(np.diff(trajectory[:, 0]) <= 0):
        raise EvaluatorReferenceError("tier3 reference timestamps must be aligned and strictly increasing")
    if abs(float(trajectory[-1, 7]) - _number(reference, "tier3.final_mass_kg")) > 1e-5:
        raise EvaluatorReferenceError("tier3 reference incumbent mass differs from its trajectory")


def validate_reference(reference_files):
    """Parse and validate evaluator-owned data before inspecting candidate files."""
    if not isinstance(reference_files, dict):
        raise EvaluatorReferenceError("Evaluator reference bundle must be a file mapping")
    missing = [name for name in REQUIRED_FILES if name not in reference_files]
    if missing:
        raise EvaluatorReferenceError(f"Missing evaluator reference files: {', '.join(missing)}")
    try:
        results = _load_json(reference_files["results.json"])
        tier2 = _load_npy(reference_files["tier2_trajectory.npy"])
        tier3 = _load_npy(reference_files["tier3_trajectory.npy"])
        control = _load_npy(reference_files["tier3_control.npy"])
        _check_tier1(results, results)
        for key in (
            "final_sma_km", "final_eccentricity", "final_inclination_deg", "dv_total_m_s",
            "transfer_time_s", "final_mass_kg", "fuel_consumed_kg", "edelbaum_dv_m_s",
        ):
            _number(results, "tier2." + key)
        for key in (
            "final_sma_km", "final_eccentricity", "final_inclination_deg", "dv_total_m_s",
            "transfer_time_days", "final_mass_kg", "fuel_consumed_kg", "constraint_violation_norm",
            "hamiltonian_initial", "hamiltonian_final",
        ):
            _number(results, "tier3." + key)
        if not _bool(results, "tier3.shooting_converged"):
            raise ValueError("Reference does not report convergence")
        if (
            tier2.ndim != 2 or tier2.shape[1] != 8 or len(tier2) < 1000
            or not np.issubdtype(tier2.dtype, np.number) or not np.isrealobj(tier2)
            or not np.all(np.isfinite(tier2)) or np.any(np.diff(tier2[:, 0]) <= 0)
        ):
            raise ValueError("Invalid Tier 2 reference trajectory")
        _validate_reference_tier3(results, tier3, control)
    except (ValueError, TypeError, EOFError, OSError, zipfile.BadZipFile) as error:
        raise EvaluatorReferenceError(f"Malformed evaluator reference: {error}") from error
    return results, tier2, tier3, control


def score_submission(candidate_files: dict[str, bytes], reference_files: dict[str, bytes]) -> ScoreReport:
    reference_results, reference_tier2_traj, reference_tier3_traj, reference_tier3_control = validate_reference(reference_files)
    failures = _check_required(candidate_files)
    if failures:
        return ScoreReport(score=0.0, passed=False, failures=tuple(failures))

    try:
        candidate_results = _load_json(candidate_files["results.json"])
        tier2_traj = _load_npy(candidate_files["tier2_trajectory.npy"])
        tier3_traj = _load_npy(candidate_files["tier3_trajectory.npy"])
        tier3_control = _load_npy(candidate_files["tier3_control.npy"])
    except (ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        return ScoreReport(score=0.0, passed=False, failures=(str(exc),))
    for array in (tier2_traj, tier3_traj, tier3_control):
        if not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array):
            return ScoreReport(score=0.0, passed=False, failures=("Candidate arrays must be real numeric arrays",))
    try:
        failures.extend(_check_tier1(candidate_results, reference_results))
        failures.extend(
            _check_tier2(candidate_results, reference_results, tier2_traj, reference_tier2_traj)
        )
        failures.extend(
            _check_tier3(
                candidate_results,
                reference_results,
                tier3_traj,
                tier3_control,
                reference_tier3_traj,
                reference_tier3_control,
            )
        )
    except (ValueError, ArithmeticError) as exc:
        failures.append(str(exc))

    passed = not failures
    return ScoreReport(score=1.0 if passed else 0.0, passed=passed, failures=tuple(failures))
