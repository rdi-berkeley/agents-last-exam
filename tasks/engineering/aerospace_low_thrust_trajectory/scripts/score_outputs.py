"""Scoring helpers for aerospace_low_thrust_trajectory."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


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
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("results.json must contain a JSON object")
    return data


def _load_npy(payload: bytes) -> np.ndarray:
    return np.load(io.BytesIO(payload), allow_pickle=False)


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


def _check_required(candidate: dict[str, bytes], reference: dict[str, bytes]) -> list[str]:
    failures: list[str] = []
    for name in REQUIRED_FILES:
        if name not in candidate:
            failures.append(f"missing candidate file: {name}")
        if name not in reference:
            failures.append(f"missing reference file: {name}")
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
        failures.append("tier2 trajectory contains non-finite values")
    if np.any(np.diff(traj[:, 0]) < -1e-9):
        failures.append("tier2 trajectory time is not monotonic")
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
        sample_t = np.linspace(max(traj[0, 0], reference_traj[0, 0]), min(traj[-1, 0], reference_traj[-1, 0]), 25)
        cand_sample = _interp_columns(traj, sample_t, [1, 2, 3, 4, 5, 6, 7])
        ref_sample = _interp_columns(reference_traj, sample_t, [1, 2, 3, 4, 5, 6, 7])
        pos_err_km = np.linalg.norm(cand_sample[:, 0:3] - ref_sample[:, 0:3], axis=1) / 1000.0
        vel_err = np.linalg.norm(cand_sample[:, 3:6] - ref_sample[:, 3:6], axis=1)
        mass_err = np.abs(cand_sample[:, 6] - ref_sample[:, 6])
        if float(np.median(pos_err_km)) > 2500.0:
            failures.append("tier2 trajectory position history is implausible relative to hidden reference")
        if float(np.median(vel_err)) > 350.0:
            failures.append("tier2 trajectory velocity history is implausible relative to hidden reference")
        if float(np.median(mass_err)) > 10.0:
            failures.append("tier2 trajectory mass history is implausible relative to hidden reference")

    residuals: list[float] = []
    sample_count = min(80, max(10, traj.shape[0] // 200))
    for idx in np.linspace(2, traj.shape[0] - 3, sample_count, dtype=int):
        previous = traj[idx - 1]
        current = traj[idx]
        following = traj[idx + 1]
        dt = float(following[0] - previous[0])
        if dt <= 0.0:
            failures.append("tier2 trajectory has non-positive finite-difference interval")
            break
        fd_acc = (following[4:7] - previous[4:7]) / dt
        r_vec = current[1:4]
        v_vec = current[4:7]
        mass = float(current[7])
        r_norm = np.linalg.norm(r_vec)
        v_norm = np.linalg.norm(v_vec)
        if r_norm <= 0.0 or v_norm <= 0.0 or mass <= 0.0:
            failures.append("tier2 trajectory has invalid state for dynamics check")
            break
        model_acc = -MU * r_vec / r_norm**3 + (THRUST / mass) * (v_vec / v_norm)
        residuals.append(float(np.linalg.norm(fd_acc - model_acc)))
    if residuals and (float(np.median(residuals)) > 0.005 or max(residuals) > 0.08):
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
    failures: list[str] = []
    if trajectory.ndim != 2 or trajectory.shape[1] != 14 or trajectory.shape[0] < 1000:
        return [f"tier3_trajectory.npy has invalid shape {trajectory.shape}"]
    if control.ndim != 2 or control.shape[1] != 4 or control.shape[0] != trajectory.shape[0]:
        return [f"tier3_control.npy has invalid shape {control.shape}"]
    if not np.all(np.isfinite(trajectory)) or not np.all(np.isfinite(control)):
        failures.append("tier3 arrays contain non-finite values")
    if np.any(np.diff(trajectory[:, 0]) < -1e-9) or np.any(np.diff(control[:, 0]) < -1e-9):
        failures.append("tier3 times are not monotonic")
    if np.max(np.abs(trajectory[:, 0] - control[:, 0])) > 1.0:
        failures.append("tier3 trajectory/control times are not aligned")
    control_norm = np.linalg.norm(control[:, 1:4], axis=1)
    if np.max(control_norm) > 1.01:
        failures.append("tier3 control norm exceeds unit magnitude")
    active_fraction = float(np.mean(control_norm > 0.5))
    if active_fraction < 0.55:
        failures.append("tier3 control is inactive for too much of the transfer")
    unit_like_fraction = float(np.mean((control_norm > 0.95) | (control_norm < 0.05)))
    if unit_like_fraction < 0.80:
        failures.append("tier3 control magnitudes are not mostly bang-bang/unit or coast")

    final_sma = _number(results, "tier3.final_sma_km")
    final_ecc = _number(results, "tier3.final_eccentricity")
    final_inc = _number(results, "tier3.final_inclination_deg")
    dv_total = _number(results, "tier3.dv_total_m_s")
    final_mass = _number(results, "tier3.final_mass_kg")
    fuel = _number(results, "tier3.fuel_consumed_kg")
    transfer_days = _number(results, "tier3.transfer_time_days")
    constraint_norm = _number(results, "tier3.constraint_violation_norm")

    if _rel_err(final_sma, A_GEO_KM) > 0.001:
        failures.append("tier3 final SMA is outside 0.1% GEO tolerance")
    if final_ecc > 0.005:
        failures.append("tier3 final eccentricity is too high")
    if final_inc > 0.1:
        failures.append("tier3 final inclination is too high")
    if not _bool(results, "tier3.shooting_converged"):
        failures.append("tier3 shooting did not converge")
    if constraint_norm > 1e-4:
        failures.append("tier3 shooting constraint norm is too high")
    if abs(transfer_days - 300.0) > 0.5:
        failures.append("tier3 transfer time is not 300 days")
    if abs(final_mass - (M0 - fuel)) > 2.0:
        failures.append("tier3 final mass and fuel consumed are inconsistent")
    rocket_mass = M0 * math.exp(-dv_total / VE)
    if abs(final_mass - rocket_mass) > 20.0:
        failures.append("tier3 final mass is inconsistent with rocket equation")
    if not _close(dv_total, _number(reference, "tier3.dv_total_m_s"), atol=400.0, rtol=0.06):
        failures.append("tier3 delta-v is implausible relative to hidden reference")

    h0 = _number(results, "tier3.hamiltonian_initial")
    hf = _number(results, "tier3.hamiltonian_final")
    denom = max(abs(h0), abs(hf), 1.0)
    if abs(hf - h0) / denom > 0.01:
        failures.append("tier3 Hamiltonian drift is too high")

    if abs(float(trajectory[-1, 0]) / 86400.0 - transfer_days) > 0.5:
        failures.append("tier3 trajectory final time does not match results.json")
    if _rel_err(float(trajectory[-1, 1]) / 1000.0, final_sma) > 0.001:
        failures.append("tier3 final trajectory p/SMA does not match results.json")
    if abs(math.hypot(float(trajectory[-1, 2]), float(trajectory[-1, 3])) - final_ecc) > 0.01:
        failures.append("tier3 final MEE eccentricity does not match results.json")
    if abs(_mee_inclination_deg(float(trajectory[-1, 4]), float(trajectory[-1, 5])) - final_inc) > 0.2:
        failures.append("tier3 final MEE inclination does not match results.json")
    if abs(float(trajectory[-1, 7]) - final_mass) > 5.0:
        failures.append("tier3 final trajectory mass does not match results.json")

    expected_initial = np.array([
        A_LEO_M,
        0.0,
        0.0,
        math.tan(math.radians(I_LEO_DEG) / 2.0),
        0.0,
        0.0,
        M0,
    ])
    candidate_initial = trajectory[0, [1, 2, 3, 4, 5, 6, 7]]
    if np.linalg.norm(candidate_initial - expected_initial) > 1e-3:
        failures.append("tier3 trajectory initial MEE/mass state is wrong")

    if (
        reference_trajectory.ndim == 2
        and reference_trajectory.shape[1] == 14
        and reference_control.ndim == 2
        and reference_control.shape[1] == 4
    ):
        if _rel_err(float(trajectory[-1, 0]), float(reference_trajectory[-1, 0])) > 0.01:
            failures.append("tier3 transfer duration is implausible relative to hidden reference")
        sample_t = np.linspace(
            max(trajectory[0, 0], reference_trajectory[0, 0]),
            min(trajectory[-1, 0], reference_trajectory[-1, 0]),
            50,
        )
        cand_state = _interp_columns(trajectory, sample_t, [1, 2, 3, 4, 5, 7])
        ref_state = _interp_columns(reference_trajectory, sample_t, [1, 2, 3, 4, 5, 7])
        p_err_km = np.abs(cand_state[:, 0] - ref_state[:, 0]) / 1000.0
        fg_err = np.linalg.norm(cand_state[:, 1:3] - ref_state[:, 1:3], axis=1)
        hk_err = np.linalg.norm(cand_state[:, 3:5] - ref_state[:, 3:5], axis=1)
        mass_err = np.abs(cand_state[:, 5] - ref_state[:, 5])
        if float(np.median(p_err_km)) > 1500.0:
            failures.append("tier3 p history is implausible relative to hidden reference")
        if float(np.median(fg_err)) > 0.02:
            failures.append("tier3 eccentricity-vector history is implausible relative to hidden reference")
        if float(np.median(hk_err)) > 0.03:
            failures.append("tier3 inclination-vector history is implausible relative to hidden reference")
        if float(np.median(mass_err)) > 25.0:
            failures.append("tier3 mass history is implausible relative to hidden reference")

        cand_control = _interp_columns(control, sample_t, [1, 2, 3])
        ref_control = _interp_columns(reference_control, sample_t, [1, 2, 3])
        control_err = np.linalg.norm(cand_control - ref_control, axis=1)
        ref_norm = np.linalg.norm(ref_control, axis=1)
        active_mask = ref_norm > 0.5
        if active_mask.any() and float(np.median(control_err[active_mask])) > 0.45:
            failures.append("tier3 active control direction is implausible relative to hidden reference")
        if abs(active_fraction - float(np.mean(ref_norm > 0.5))) > 0.15:
            failures.append("tier3 thrust active fraction is implausible relative to hidden reference")

    return failures


def score_submission(candidate_files: dict[str, bytes], reference_files: dict[str, bytes]) -> ScoreReport:
    failures = _check_required(candidate_files, reference_files)
    if failures:
        return ScoreReport(score=0.0, passed=False, failures=tuple(failures))

    try:
        candidate_results = _load_json(candidate_files["results.json"])
        reference_results = _load_json(reference_files["results.json"])
        tier2_traj = _load_npy(candidate_files["tier2_trajectory.npy"])
        tier3_traj = _load_npy(candidate_files["tier3_trajectory.npy"])
        tier3_control = _load_npy(candidate_files["tier3_control.npy"])
        reference_tier2_traj = _load_npy(reference_files["tier2_trajectory.npy"])
        reference_tier3_traj = _load_npy(reference_files["tier3_trajectory.npy"])
        reference_tier3_control = _load_npy(reference_files["tier3_control.npy"])
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
    except Exception as exc:
        failures.append(str(exc))

    passed = not failures
    return ScoreReport(score=1.0 if passed else 0.0, passed=passed, failures=tuple(failures))
