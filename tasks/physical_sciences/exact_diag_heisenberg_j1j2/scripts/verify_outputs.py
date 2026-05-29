"""Local verifier for exact_diag_heisenberg_j1j2."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

NUM_SITES = 16
NUM_UP_SPINS = 8
GROUND_STATE_DIM = 12870
NUM_Q_POINTS = 16
NUM_OMEGA_POINTS = 500
J1 = 1.0
J2 = 0.55
FULL_SCORE = 1.0
PARTIAL_SCORE = 2.0 / 3.0
FAIL_SCORE = 0.0
ENERGY_TOL = 1e-6
RESIDUAL_TOL = 1e-8
NORM_TOL = 1e-6
CORRELATION_TOL = 1e-5
STRUCTURE_FACTOR_TOL = 1e-5
STATIC_SUM_RULE_TOL = 1e-4
NONNEGATIVE_TOL = 1e-8
OMEGA_MAX_TOL = 0.5
OMEGA_GRID_UNIFORM_TOL = 1e-8
SUM_RULE_REL_TOL = 0.05
SUM_RULE_RESULTS_TOL = 0.01
PEAK_POSITION_TOL = 0.6
RESULTS_TOL = 1e-6
EXPECTED_LANCZOS_STEPS = 200

REQUIRED_OUTPUT_FILES = (
    "ground_state.npz",
    "correlations.npz",
    "dynamical_sf.npz",
    "results.json",
)

REQUIRED_RESULTS_KEYS = {
    "tier1": {"E0", "E1", "spin_gap", "E0_per_site", "hilbert_dim", "num_sites"},
    "tier2": {
        "S_q_pi_pi",
        "S_q_0_0",
        "correlation_sum_rule",
        "C_nearest_neighbor",
        "C_next_nearest_neighbor",
    },
    "tier3": {
        "omega_max",
        "eta",
        "num_lanczos_steps",
        "num_omega_points",
        "num_q_points",
        "sum_rule_max_error",
    },
}


@dataclass
class TierResult:
    passed: bool
    reason: str
    details: dict[str, Any]


@dataclass
class ScoreResult:
    score: float
    tier1_passed: bool
    tier2_passed: bool
    tier3_passed: bool
    tier1_reason: str
    tier2_reason: str
    tier3_reason: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VerificationError(RuntimeError):
    """Raised when an output artifact cannot be verified cleanly."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, reason: str, **details: Any) -> TierResult | None:
    if condition:
        return None
    return TierResult(False, reason, _json_ready(details))


def _q_key(q_point: np.ndarray) -> tuple[float, float]:
    return (round(float(q_point[0]), 8), round(float(q_point[1]), 8))


def _build_q_index_map(candidate_q: np.ndarray) -> dict[tuple[float, float], int]:
    index_map: dict[tuple[float, float], int] = {}
    for idx, q_point in enumerate(candidate_q):
        key = _q_key(q_point)
        if key in index_map:
            raise ValueError(f"duplicate q-point {key}")
        index_map[key] = idx
    return index_map


def _reorder_by_q_points(reference_q: np.ndarray, candidate_q: np.ndarray, values: np.ndarray) -> np.ndarray:
    if candidate_q.shape != reference_q.shape:
        raise ValueError(f"q_points shape mismatch: {candidate_q.shape} vs {reference_q.shape}")
    index_map = _build_q_index_map(candidate_q)
    order = []
    for q_point in reference_q:
        key = _q_key(q_point)
        if key not in index_map:
            raise ValueError(f"missing q-point {key}")
        order.append(index_map[key])
    return values[np.asarray(order)]


def _reorder_coefficients_by_q_points(
    reference_q: np.ndarray,
    candidate_q: np.ndarray,
    coefficients: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if candidate_q.shape != reference_q.shape:
        raise ValueError(f"q_points shape mismatch: {candidate_q.shape} vs {reference_q.shape}")
    index_map = _build_q_index_map(candidate_q)
    reordered: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ref_idx, q_point in enumerate(reference_q):
        key = _q_key(q_point)
        if key not in index_map:
            raise ValueError(f"missing q-point {key}")
        reordered[ref_idx] = coefficients[index_map[key]]
    return reordered


@lru_cache(maxsize=1)
def _basis_states() -> tuple[np.ndarray, dict[int, int]]:
    states = [state for state in range(1 << NUM_SITES) if state.bit_count() == NUM_UP_SPINS]
    basis = np.asarray(states, dtype=np.uint32)
    state_to_index = {int(state): idx for idx, state in enumerate(states)}
    return basis, state_to_index


@lru_cache(maxsize=1)
def _bonds() -> tuple[tuple[int, int, float], ...]:
    def site(x: int, y: int) -> int:
        return y * 4 + x

    bonds: list[tuple[int, int, float]] = []
    for y in range(4):
        for x in range(4):
            i = site(x, y)
            bonds.append((i, site((x + 1) % 4, y), J1))
            bonds.append((i, site(x, (y + 1) % 4), J1))
            bonds.append((i, site((x + 1) % 4, (y + 1) % 4), J2))
            bonds.append((i, site((x - 1) % 4, (y + 1) % 4), J2))
    return tuple(bonds)


def apply_hamiltonian(vector: np.ndarray) -> np.ndarray:
    basis, state_to_index = _basis_states()
    bonds = _bonds()
    out = np.zeros(vector.shape[0], dtype=np.complex128)
    for basis_index, state in enumerate(basis):
        amplitude = vector[basis_index]
        diagonal = 0.0
        state_int = int(state)
        for i, j, coupling in bonds:
            spin_i = 1 if (state_int >> i) & 1 else -1
            spin_j = 1 if (state_int >> j) & 1 else -1
            diagonal += coupling * (0.25 if spin_i == spin_j else -0.25)
            if spin_i != spin_j:
                flipped = state_int ^ ((1 << i) | (1 << j))
                out[state_to_index[flipped]] += 0.5 * coupling * amplitude
        out[basis_index] += diagonal * amplitude
    return out


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    except Exception as exc:
        raise VerificationError(f"failed to load npz archive {path}: {exc}") from exc


def _validate_results_schema(results: dict[str, Any]) -> TierResult | None:
    for section, keys in REQUIRED_RESULTS_KEYS.items():
        if section not in results:
            return TierResult(False, f"results.json missing section {section}", {})
        missing = sorted(keys - set(results[section]))
        if missing:
            return TierResult(False, f"results.json missing keys in {section}", {"missing": missing})
    return None


def _extract_tier3_coefficients(
    dynamical_data: dict[str, np.ndarray],
) -> tuple[bool, dict[int, tuple[np.ndarray, np.ndarray]]]:
    coeffs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    has_flat = True
    for idx in range(NUM_Q_POINTS):
        a_key = f"a_{idx}"
        b_key = f"b_{idx}"
        if a_key not in dynamical_data or b_key not in dynamical_data:
            has_flat = False
            break
        coeffs[idx] = (dynamical_data[a_key], dynamical_data[b_key])
    if has_flat:
        return True, coeffs

    if "lanczos_coefficients" not in dynamical_data:
        return False, {}

    packed = dynamical_data["lanczos_coefficients"]
    if packed.dtype.names is None:
        return False, {}
    if packed.shape[0] != NUM_Q_POINTS:
        return False, {}
    if "a" not in packed.dtype.names or "b" not in packed.dtype.names:
        return False, {}
    for idx in range(NUM_Q_POINTS):
        a_values = np.asarray(packed[idx]["a"], dtype=np.float64).reshape(-1)
        b_values = np.asarray(packed[idx]["b"], dtype=np.float64).reshape(-1)
        if "a_len" in packed.dtype.names:
            a_len = int(packed[idx]["a_len"])
        else:
            a_len = len(a_values)
        if "b_len" in packed.dtype.names:
            b_len = int(packed[idx]["b_len"])
        else:
            b_len = len(b_values)
        coeffs[idx] = (
            np.asarray(a_values[:a_len], dtype=np.float64),
            np.asarray(b_values[:b_len], dtype=np.float64),
        )
    return True, coeffs


def _max_lanczos_steps(coefficients: dict[int, tuple[np.ndarray, np.ndarray]]) -> int:
    return max((len(values[0]) for values in coefficients.values()), default=0)


def _coefficient_length_contract_violations(
    submitted: dict[int, tuple[np.ndarray, np.ndarray]],
    submitted_structure_factor: np.ndarray,
    expected_lanczos_steps: int,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for idx in range(NUM_Q_POINTS):
        submitted_a, submitted_b = submitted.get(idx, (np.array([]), np.array([])))
        a_len = len(submitted_a)
        b_len = len(submitted_b)
        if abs(float(submitted_structure_factor[idx])) < 1e-10:
            allowed_a_lens = {0, expected_lanczos_steps}
            allowed_b_lens = {0, expected_lanczos_steps - 1, expected_lanczos_steps}
        else:
            allowed_a_lens = {expected_lanczos_steps}
            allowed_b_lens = {expected_lanczos_steps - 1, expected_lanczos_steps}
        if a_len not in allowed_a_lens or b_len not in allowed_b_lens:
            violations.append(
                {
                    "q_index": idx,
                    "submitted_a_len": a_len,
                    "submitted_b_len": b_len,
                    "allowed_a_lens": sorted(allowed_a_lens),
                    "allowed_b_lens": sorted(allowed_b_lens),
                    "structure_factor": float(submitted_structure_factor[idx]),
                }
            )
    return violations


def evaluate_tier1(
    output_dir: Path,
    reference_dir: Path,
    results_json: dict[str, Any],
) -> TierResult:
    try:
        output = _load_npz(output_dir / "ground_state.npz")
        reference = _load_npz(reference_dir / "ground_state.npz")
    except VerificationError as exc:
        return TierResult(False, str(exc), {})
    required = {"E0", "E1", "spin_gap", "ground_state_vector"}
    missing = sorted(required - set(output))
    if missing:
        return TierResult(False, "ground_state.npz missing required keys", {"missing": missing})

    try:
        vector = np.asarray(output["ground_state_vector"])
        e0 = float(output["E0"])
        e1 = float(output["E1"])
        spin_gap = float(output["spin_gap"])
    except (TypeError, ValueError) as exc:
        return TierResult(False, "Tier 1 payload contains non-numeric values", {"error": str(exc)})
    if vector.shape != (GROUND_STATE_DIM,):
        return TierResult(False, "ground_state_vector has wrong shape", {"shape": vector.shape})
    if not np.all(np.isfinite(vector)):
        return TierResult(False, "ground_state_vector contains non-finite values", {})

    if not all(np.isfinite([e0, e1, spin_gap])):
        return TierResult(False, "Tier 1 scalars contain non-finite values", {})

    norm = float(np.linalg.norm(vector))
    failed = _require(abs(norm - 1.0) <= NORM_TOL, "ground_state_vector is not normalized", norm=norm)
    if failed:
        return failed

    residual = float(np.linalg.norm(apply_hamiltonian(vector) - e0 * vector))
    failed = _require(
        residual < RESIDUAL_TOL,
        "Tier 1 Hamiltonian residual too large",
        residual=residual,
        tolerance=RESIDUAL_TOL,
    )
    if failed:
        return failed

    reference_errors = {
        "E0_abs_error": abs(e0 - float(reference["E0"])),
        "E1_abs_error": abs(e1 - float(reference["E1"])),
        "spin_gap_abs_error": abs(spin_gap - float(reference["spin_gap"])),
    }
    failed = _require(
        max(reference_errors.values()) <= ENERGY_TOL,
        "Tier 1 energies differ from hidden reference",
        tolerance=ENERGY_TOL,
        **reference_errors,
    )
    if failed:
        return failed

    schema_failed = _validate_results_schema(results_json)
    if schema_failed:
        return schema_failed

    tier1_results = results_json["tier1"]
    result_errors = {
        "E0_abs_error": abs(float(tier1_results["E0"]) - e0),
        "E1_abs_error": abs(float(tier1_results["E1"]) - e1),
        "spin_gap_abs_error": abs(float(tier1_results["spin_gap"]) - spin_gap),
        "E0_per_site_abs_error": abs(float(tier1_results["E0_per_site"]) - (e0 / NUM_SITES)),
    }
    if int(tier1_results["hilbert_dim"]) != GROUND_STATE_DIM or int(tier1_results["num_sites"]) != NUM_SITES:
        return TierResult(
            False,
            "results.json Tier 1 metadata is inconsistent",
            {
                "hilbert_dim": tier1_results["hilbert_dim"],
                "num_sites": tier1_results["num_sites"],
            },
        )
    failed = _require(
        max(result_errors.values()) <= RESULTS_TOL,
        "results.json Tier 1 summary is inconsistent with ground_state.npz",
        tolerance=RESULTS_TOL,
        **result_errors,
    )
    if failed:
        return failed

    return TierResult(
        True,
        "Tier 1 passed",
        {
            "residual": residual,
            "norm": norm,
            **reference_errors,
            **result_errors,
        },
    )


def evaluate_tier2(
    output_dir: Path,
    reference_dir: Path,
    results_json: dict[str, Any],
) -> TierResult:
    try:
        output = _load_npz(output_dir / "correlations.npz")
        reference = _load_npz(reference_dir / "correlations.npz")
    except VerificationError as exc:
        return TierResult(False, str(exc), {})
    required = {"correlation_matrix", "q_points", "structure_factor"}
    missing = sorted(required - set(output))
    if missing:
        return TierResult(False, "correlations.npz missing required keys", {"missing": missing})

    try:
        correlation_matrix = np.asarray(output["correlation_matrix"], dtype=np.float64)
        q_points = np.asarray(output["q_points"], dtype=np.float64)
        structure_factor = np.asarray(output["structure_factor"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        return TierResult(False, "Tier 2 payload contains non-numeric values", {"error": str(exc)})

    if correlation_matrix.shape != (NUM_SITES, NUM_SITES):
        return TierResult(False, "correlation_matrix has wrong shape", {"shape": correlation_matrix.shape})
    if q_points.shape != (NUM_Q_POINTS, 2):
        return TierResult(False, "q_points has wrong shape", {"shape": q_points.shape})
    if structure_factor.shape != (NUM_Q_POINTS,):
        return TierResult(False, "structure_factor has wrong shape", {"shape": structure_factor.shape})
    if not np.all(np.isfinite(correlation_matrix)) or not np.all(np.isfinite(structure_factor)):
        return TierResult(False, "Tier 2 outputs contain non-finite values", {})

    try:
        reordered_sf = _reorder_by_q_points(reference["q_points"], q_points, structure_factor)
    except ValueError as exc:
        return TierResult(False, "Tier 2 q_points do not match hidden reference", {"error": str(exc)})
    diag_error = float(np.max(np.abs(np.diag(correlation_matrix) - 0.75)))
    sum_rule_value = float(np.sum(reordered_sf))
    sum_rule_error = abs(sum_rule_value - 12.0)
    min_sf = float(np.min(reordered_sf))

    failed = _require(
        diag_error <= CORRELATION_TOL,
        "Tier 2 self-correlation diagonal is inconsistent",
        diag_error=diag_error,
        tolerance=CORRELATION_TOL,
    )
    if failed:
        return failed
    failed = _require(
        sum_rule_error <= STATIC_SUM_RULE_TOL,
        "Tier 2 static sum rule failed",
        sum_rule_error=sum_rule_error,
        tolerance=STATIC_SUM_RULE_TOL,
    )
    if failed:
        return failed
    failed = _require(
        min_sf >= -NONNEGATIVE_TOL,
        "Tier 2 structure factor is not non-negative",
        min_structure_factor=min_sf,
        tolerance=NONNEGATIVE_TOL,
    )
    if failed:
        return failed

    corr_error = float(np.max(np.abs(correlation_matrix - reference["correlation_matrix"])))
    sf_error = float(np.max(np.abs(reordered_sf - reference["structure_factor"])))
    failed = _require(
        corr_error <= CORRELATION_TOL and sf_error <= STRUCTURE_FACTOR_TOL,
        "Tier 2 outputs differ from hidden reference",
        correlation_max_abs_error=corr_error,
        structure_factor_max_abs_error=sf_error,
        correlation_tolerance=CORRELATION_TOL,
        structure_factor_tolerance=STRUCTURE_FACTOR_TOL,
    )
    if failed:
        return failed

    tier2_results = results_json["tier2"]
    pi_pi_idx = int(np.argmax(np.all(np.isclose(reference["q_points"], np.array([np.pi, np.pi])), axis=1)))
    zero_zero_idx = int(np.argmax(np.all(np.isclose(reference["q_points"], np.array([0.0, 0.0])), axis=1)))
    tier2_errors = {
        "S_q_pi_pi_abs_error": abs(float(tier2_results["S_q_pi_pi"]) - float(reordered_sf[pi_pi_idx])),
        "S_q_0_0_abs_error": abs(float(tier2_results["S_q_0_0"]) - float(reordered_sf[zero_zero_idx])),
        "correlation_sum_rule_abs_error": abs(float(tier2_results["correlation_sum_rule"]) - sum_rule_value),
        "C_nearest_neighbor_abs_error": abs(float(tier2_results["C_nearest_neighbor"]) - float(correlation_matrix[0, 1])),
        "C_next_nearest_neighbor_abs_error": abs(float(tier2_results["C_next_nearest_neighbor"]) - float(correlation_matrix[0, 5])),
    }
    failed = _require(
        max(tier2_errors.values()) <= RESULTS_TOL,
        "results.json Tier 2 summary is inconsistent with correlations.npz",
        tolerance=RESULTS_TOL,
        **tier2_errors,
    )
    if failed:
        return failed

    return TierResult(
        True,
        "Tier 2 passed",
        {
            "diag_error": diag_error,
            "sum_rule_error": sum_rule_error,
            "correlation_max_abs_error": corr_error,
            "structure_factor_max_abs_error": sf_error,
            **tier2_errors,
        },
    )


def evaluate_tier3(
    output_dir: Path,
    reference_dir: Path,
    results_json: dict[str, Any],
    submitted_structure_factor: np.ndarray,
    expected_lanczos_steps: int,
) -> TierResult:
    try:
        output = _load_npz(output_dir / "dynamical_sf.npz")
        reference = _load_npz(reference_dir / "dynamical_sf.npz")
    except VerificationError as exc:
        return TierResult(False, str(exc), {})
    required = {"q_points", "omega_grid", "S_q_omega", "eta", "omega_max"}
    missing = sorted(required - set(output))
    if missing:
        return TierResult(False, "dynamical_sf.npz missing required keys", {"missing": missing})

    coeff_ok, coefficients = _extract_tier3_coefficients(output)
    if not coeff_ok:
        return TierResult(
            False,
            "Tier 3 Lanczos coefficient payload missing or malformed",
            {"accepted": ["lanczos_coefficients", "a_<q_index>/b_<q_index>"]},
        )
    try:
        q_points = np.asarray(output["q_points"], dtype=np.float64)
        omega_grid = np.asarray(output["omega_grid"], dtype=np.float64)
        sqw = np.asarray(output["S_q_omega"], dtype=np.float64)
        eta = float(output["eta"])
        omega_max = float(output["omega_max"])
    except (TypeError, ValueError) as exc:
        return TierResult(False, "Tier 3 payload contains non-numeric values", {"error": str(exc)})

    if q_points.shape != (NUM_Q_POINTS, 2):
        return TierResult(False, "Tier 3 q_points has wrong shape", {"shape": q_points.shape})
    if omega_grid.shape != (NUM_OMEGA_POINTS,):
        return TierResult(False, "Tier 3 omega_grid has wrong shape", {"shape": omega_grid.shape})
    if sqw.shape != (NUM_Q_POINTS, NUM_OMEGA_POINTS):
        return TierResult(False, "Tier 3 S_q_omega has wrong shape", {"shape": sqw.shape})
    if not np.all(np.isfinite(sqw)) or not np.all(np.isfinite(omega_grid)):
        return TierResult(False, "Tier 3 outputs contain non-finite values", {})

    try:
        reordered_sqw = _reorder_by_q_points(reference["q_points"], q_points, sqw)
        reordered_coefficients = _reorder_coefficients_by_q_points(
            reference["q_points"],
            q_points,
            coefficients,
        )
    except ValueError as exc:
        return TierResult(False, "Tier 3 q_points do not match hidden reference", {"error": str(exc)})
    min_sqw = float(np.min(reordered_sqw))
    failed = _require(
        min_sqw >= -NONNEGATIVE_TOL,
        "Tier 3 spectrum is not non-negative",
        min_sqw=min_sqw,
        tolerance=NONNEGATIVE_TOL,
    )
    if failed:
        return failed

    if not np.all(np.diff(omega_grid) > 0):
        return TierResult(False, "omega_grid is not strictly increasing", {})
    if abs(float(omega_grid[0])) > RESULTS_TOL:
        return TierResult(False, "omega_grid does not start at zero", {"omega_0": float(omega_grid[0])})
    failed = _require(
        abs(omega_max - float(reference["omega_max"])) <= OMEGA_MAX_TOL,
        "omega_max differs too much from hidden reference",
        omega_max=omega_max,
        reference_omega_max=float(reference["omega_max"]),
        tolerance=OMEGA_MAX_TOL,
    )
    if failed:
        return failed
    failed = _require(abs(eta - 0.1) <= RESULTS_TOL, "eta differs from required value", eta=eta)
    if failed:
        return failed

    uniform_grid_error = float(
        np.max(np.abs(omega_grid - np.linspace(0.0, omega_max, NUM_OMEGA_POINTS)))
    )
    failed = _require(
        uniform_grid_error <= OMEGA_GRID_UNIFORM_TOL,
        "omega_grid is not uniformly spaced from 0 to omega_max",
        uniform_grid_error=uniform_grid_error,
        tolerance=OMEGA_GRID_UNIFORM_TOL,
    )
    if failed:
        return failed

    integrated = np.trapezoid(reordered_sqw, omega_grid, axis=1)
    submitted_sf = np.asarray(submitted_structure_factor, dtype=np.float64)
    rel_errors = []
    for expected, observed in zip(submitted_sf, integrated):
        if abs(expected) < 1e-12:
            rel_errors.append(abs(observed - expected))
        else:
            rel_errors.append(abs(observed - expected) / abs(expected))
    rel_errors_arr = np.asarray(rel_errors, dtype=np.float64)
    sum_rule_max_error = float(np.max(rel_errors_arr))
    failed = _require(
        sum_rule_max_error <= SUM_RULE_REL_TOL,
        "Tier 3 static sum-rule check failed",
        sum_rule_max_error=sum_rule_max_error,
        tolerance=SUM_RULE_REL_TOL,
    )
    if failed:
        return failed

    reference_sqw = reference["S_q_omega"]
    reference_active = np.max(reference_sqw, axis=1) > 1e-4
    peak_position_error = 0.0
    active_count = 0
    for idx, is_active in enumerate(reference_active):
        if not is_active:
            continue
        ref_peak = float(reference["omega_grid"][int(np.argmax(reference_sqw[idx]))])
        pred_peak = float(omega_grid[int(np.argmax(reordered_sqw[idx]))])
        peak_position_error = max(peak_position_error, abs(pred_peak - ref_peak))
        active_count += 1
    failed = _require(
        peak_position_error <= PEAK_POSITION_TOL,
        "Tier 3 peak positions differ too much from hidden reference",
        peak_position_error=peak_position_error,
        tolerance=PEAK_POSITION_TOL,
        active_q_points=active_count,
    )
    if failed:
        return failed

    tier3_results = results_json["tier3"]
    max_lanczos_steps = _max_lanczos_steps(reordered_coefficients)
    failed = _require(
        max_lanczos_steps == expected_lanczos_steps,
        "Tier 3 Lanczos coefficient length does not match the required step count",
        max_lanczos_steps=max_lanczos_steps,
        expected_lanczos_steps=expected_lanczos_steps,
    )
    if failed:
        return failed
    length_mismatches = _coefficient_length_contract_violations(
        reordered_coefficients,
        submitted_structure_factor,
        expected_lanczos_steps,
    )
    failed = _require(
        not length_mismatches,
        "Tier 3 Lanczos coefficient lengths violate the visible contract",
        mismatches=length_mismatches,
    )
    if failed:
        return failed
    tier3_errors = {
        "omega_max_abs_error": abs(float(tier3_results["omega_max"]) - omega_max),
        "eta_abs_error": abs(float(tier3_results["eta"]) - eta),
        "sum_rule_max_error_abs_error": abs(
            float(tier3_results["sum_rule_max_error"]) - sum_rule_max_error
        ),
    }
    metadata_match = (
        int(tier3_results["num_lanczos_steps"]) == expected_lanczos_steps
        and int(tier3_results["num_omega_points"]) == NUM_OMEGA_POINTS
        and int(tier3_results["num_q_points"]) == NUM_Q_POINTS
    )
    failed = _require(
        metadata_match and max(tier3_errors.values()) <= SUM_RULE_RESULTS_TOL,
        "results.json Tier 3 summary is inconsistent with dynamical_sf.npz",
        max_lanczos_steps=max_lanczos_steps,
        expected_lanczos_steps=expected_lanczos_steps,
        reported_num_lanczos_steps=tier3_results["num_lanczos_steps"],
        reported_num_omega_points=tier3_results["num_omega_points"],
        reported_num_q_points=tier3_results["num_q_points"],
        tolerance=SUM_RULE_RESULTS_TOL,
        **tier3_errors,
    )
    if failed:
        return failed

    return TierResult(
        True,
        "Tier 3 passed",
        {
            "sum_rule_max_error": sum_rule_max_error,
            "peak_position_error": peak_position_error,
            "max_lanczos_steps": max_lanczos_steps,
            **tier3_errors,
        },
    )


def score_submission(
    output_dir: Path,
    reference_dir: Path,
    metadata_path: Path | None = None,
) -> ScoreResult:
    try:
        missing = [name for name in REQUIRED_OUTPUT_FILES if not (output_dir / name).exists()]
        if missing:
            return ScoreResult(
                score=FAIL_SCORE,
                tier1_passed=False,
                tier2_passed=False,
                tier3_passed=False,
                tier1_reason="missing required output files",
                tier2_reason="not evaluated",
                tier3_reason="not evaluated",
                details={"missing_files": missing},
            )

        metadata: dict[str, Any] = {}
        if metadata_path and metadata_path.exists():
            metadata = _load_json(metadata_path)
        expected_lanczos_steps = int(
            metadata.get("tier_contract", {}).get("tier3", {}).get(
                "num_lanczos_steps",
                EXPECTED_LANCZOS_STEPS,
            )
        )

        results_json = _load_json(output_dir / "results.json")
        schema_failed = _validate_results_schema(results_json)
        if schema_failed:
            return ScoreResult(
                score=FAIL_SCORE,
                tier1_passed=False,
                tier2_passed=False,
                tier3_passed=False,
                tier1_reason=schema_failed.reason,
                tier2_reason="not evaluated",
                tier3_reason="not evaluated",
                details=schema_failed.details,
            )

        tier1 = evaluate_tier1(output_dir, reference_dir, results_json)
        if not tier1.passed:
            return ScoreResult(
                score=FAIL_SCORE,
                tier1_passed=False,
                tier2_passed=False,
                tier3_passed=False,
                tier1_reason=tier1.reason,
                tier2_reason="not evaluated",
                tier3_reason="not evaluated",
                details={"tier1": tier1.details},
            )

        tier2 = evaluate_tier2(output_dir, reference_dir, results_json)
        if not tier2.passed:
            return ScoreResult(
                score=FAIL_SCORE,
                tier1_passed=True,
                tier2_passed=False,
                tier3_passed=False,
                tier1_reason=tier1.reason,
                tier2_reason=tier2.reason,
                tier3_reason="not evaluated",
                details={"tier1": tier1.details, "tier2": tier2.details},
            )

        submitted_corr = _load_npz(output_dir / "correlations.npz")
        reference_corr = _load_npz(reference_dir / "correlations.npz")
        reordered_sf = _reorder_by_q_points(
            reference_corr["q_points"],
            submitted_corr["q_points"],
            submitted_corr["structure_factor"],
        )
        tier3 = evaluate_tier3(
            output_dir,
            reference_dir,
            results_json,
            reordered_sf,
            expected_lanczos_steps,
        )
        score = FULL_SCORE if tier3.passed else PARTIAL_SCORE
        return ScoreResult(
            score=score,
            tier1_passed=True,
            tier2_passed=True,
            tier3_passed=tier3.passed,
            tier1_reason=tier1.reason,
            tier2_reason=tier2.reason,
            tier3_reason=tier3.reason,
            details={
                "tier1": tier1.details,
                "tier2": tier2.details,
                "tier3": tier3.details,
            },
        )
    except Exception as exc:
        return ScoreResult(
            score=FAIL_SCORE,
            tier1_passed=False,
            tier2_passed=False,
            tier3_passed=False,
            tier1_reason="verifier error",
            tier2_reason="not evaluated",
            tier3_reason="not evaluated",
            details={"exception": str(exc)},
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--metadata")
    args = parser.parse_args()

    result = score_submission(
        output_dir=Path(args.output_dir),
        reference_dir=Path(args.reference_dir),
        metadata_path=Path(args.metadata) if args.metadata else None,
    )
    print(json.dumps(_json_ready(result.to_dict()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
