"""Scoring logic for computing_math/particle_filter_nonlinear_tracking."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SEED = 24601

TIER1_NUM_PARTICLES = 1000
TIER2_NUM_PARTICLES = 5000
TIER3_NUM_PARTICLES = 50000
TIER3_NUM_BACKWARD_TRAJECTORIES = 5000

TIER1_MEAN_THRESHOLD = 0.20
TIER1_VAR_THRESHOLD = 0.35
TIER2_POS_RMSE_THRESHOLD = 1.5
TIER2_VEL_RMSE_THRESHOLD = 0.3
TIER2_MEAN_ESS_THRESHOLD = 1000.0
TIER3_FILTER_RMSE_THRESHOLD = 3.0
TIER3_MEAN_ESS_THRESHOLD = 500.0

REQUIRED_OUTPUT_FILES = (
    "pf_solver.py",
    "tier1_results.npz",
    "tier2_results.npz",
    "results.json",
)
OPTIONAL_OUTPUT_FILES = ("tier3_results.npz",)
REQUIRED_REFERENCE_FILES = ("tier2_results.npz", "tier3_results.npz")


@dataclass
class TierReport:
    passed: bool
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "metrics": self.metrics,
            "notes": self.notes,
        }


@dataclass
class ScoreReport:
    score: float
    solver_present: bool
    results_json_present: bool
    results_json_valid: bool
    tier1: TierReport
    tier2: TierReport
    tier3: TierReport
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "solver_present": self.solver_present,
            "results_json_present": self.results_json_present,
            "results_json_valid": self.results_json_valid,
            "tier1": self.tier1.to_dict(),
            "tier2": self.tier2.to_dict(),
            "tier3": self.tier3.to_dict(),
            "notes": self.notes,
        }


def _load_npz(blob: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _load_json(blob: bytes) -> dict[str, Any]:
    return json.loads(blob.decode("utf-8"))


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _kalman_filter_tier1(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = 0.9
    Q = 1.0
    H = 1.0
    R = 0.5
    x0_mean = 0.0
    x0_var = 2.0
    timesteps = observations.shape[0]

    means = np.zeros(timesteps, dtype=float)
    variances = np.zeros(timesteps, dtype=float)
    m_pred = x0_mean
    p_pred = x0_var

    for t in range(timesteps):
        if t > 0:
            m_pred = A * means[t - 1]
            p_pred = A**2 * variances[t - 1] + Q
        s = p_pred + R
        k = p_pred / s
        innovation = observations[t] - H * m_pred
        means[t] = m_pred + k * innovation
        variances[t] = (1.0 - k * H) * p_pred

    return means, variances


def _generate_tier1_truth() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    A = 0.9
    Q = 1.0
    R = 0.5
    x0_mean = 0.0
    x0_var = 2.0
    timesteps = 100

    true_states = np.zeros(timesteps, dtype=float)
    observations = np.zeros(timesteps, dtype=float)

    x = rng.normal(x0_mean, np.sqrt(x0_var))
    true_states[0] = x
    observations[0] = x + rng.normal(0.0, np.sqrt(R))

    for t in range(1, timesteps):
        x = A * x + rng.normal(0.0, np.sqrt(Q))
        true_states[t] = x
        observations[t] = x + rng.normal(0.0, np.sqrt(R))

    return true_states, observations


def _generate_tier2_truth_fresh() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    dt = 1.0
    q = 0.01
    sigma_bearing = 0.02
    sigma_range = 0.5
    sensor = np.array([0.0, 0.0], dtype=float)
    timesteps = 200
    x0_true = np.array([5.0, 5.0, 0.3, -0.1], dtype=float)
    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    G = np.array(
        [[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]],
        dtype=float,
    )
    sqrt_q = np.sqrt(q)

    true_states = np.zeros((timesteps, 4), dtype=float)
    observations = np.zeros((timesteps, 2), dtype=float)

    state = x0_true.copy()
    for t in range(timesteps):
        if t > 0:
            z = sqrt_q * rng.standard_normal(2)
            state = F @ state + G @ z
        true_states[t] = state
        dx = state[0] - sensor[0]
        dy = state[1] - sensor[1]
        observations[t, 0] = np.arctan2(dy, dx) + rng.normal(0.0, sigma_bearing)
        observations[t, 1] = np.sqrt(dx**2 + dy**2) + rng.normal(0.0, sigma_range)

    return true_states, observations


def _coordinated_turn_f(omega: float, dt: float) -> np.ndarray:
    if abs(omega) < 1e-8:
        return np.array(
            [[1, 0, dt, 0, 0], [0, 1, 0, dt, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
            dtype=float,
        )
    s = np.sin(omega * dt)
    c = np.cos(omega * dt)
    return np.array(
        [
            [1, 0, s / omega, -(1 - c) / omega, 0],
            [0, 1, (1 - c) / omega, s / omega, 0],
            [0, 0, c, -s, 0],
            [0, 0, s, c, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=float,
    )


def _generate_tier3_truth_fresh() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    q_diag = np.array([0.01, 0.01, 0.005, 0.005, 0.001], dtype=float)
    nu = 5
    sigma_bearing = 0.02
    bias_drift_std = 0.003
    sensor_positions = (np.array([0.0, 0.0]), np.array([47.3, 0.0]))
    timesteps = 200
    dt = 1.0

    state = np.array([10.0, 10.0, 0.5, 0.2, 0.03], dtype=float)
    bias = np.array([0.0, 0.0], dtype=float)
    true_states = np.zeros((timesteps, 5), dtype=float)
    observations = np.zeros(timesteps, dtype=float)

    for t in range(timesteps):
        if t > 0:
            F = _coordinated_turn_f(state[4], dt)
            chi2_sample = rng.chisquare(nu)
            scale = np.sqrt(nu / chi2_sample)
            noise = scale * rng.standard_normal(5) * np.sqrt(q_diag)
            state = F @ state + noise
            bias[0] += rng.normal(0.0, bias_drift_std)
            bias[1] += rng.normal(0.0, bias_drift_std)

        true_states[t] = state
        sensor_id = 0 if t % 2 == 0 else 1
        sensor = sensor_positions[sensor_id]
        dx = state[0] - sensor[0]
        dy = state[1] - sensor[1]
        observations[t] = np.arctan2(dy, dx) + bias[sensor_id] + rng.normal(0.0, sigma_bearing)

    return true_states, observations


def _allclose(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-8, rtol: float = 1e-8) -> bool:
    return a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol)


def _finite_array(name: str, value: np.ndarray) -> str | None:
    if not np.all(np.isfinite(value)):
        return f"{name} contains non-finite values"
    return None


def _validate_covariances(name: str, covs: np.ndarray) -> str | None:
    if covs.ndim != 3 or covs.shape[1] != covs.shape[2]:
        return f"{name} must be a stack of square covariance matrices"
    if not np.allclose(covs, np.swapaxes(covs, 1, 2), atol=1e-6, rtol=1e-6):
        return f"{name} must be symmetric"
    eigvals = np.linalg.eigvalsh(covs)
    if np.min(eigvals) < -1e-6:
        return f"{name} must be positive semidefinite"
    return None


def _require_shape(arrays: dict[str, np.ndarray], name: str, shape: tuple[int, ...]) -> str | None:
    if name not in arrays:
        return f"missing `{name}`"
    if arrays[name].shape != shape:
        return f"`{name}` has shape {arrays[name].shape}, expected {shape}"
    return None


def _match_truth(
    candidate_true_states: np.ndarray,
    candidate_observations: np.ndarray,
    fresh_true_states: np.ndarray,
    fresh_observations: np.ndarray,
    reference_true_states: np.ndarray,
    reference_observations: np.ndarray,
) -> tuple[str | None, np.ndarray | None]:
    if _allclose(candidate_true_states, fresh_true_states) and _allclose(candidate_observations, fresh_observations):
        return "fresh_seed_per_tier", fresh_true_states
    if _allclose(candidate_true_states, reference_true_states) and _allclose(
        candidate_observations, reference_observations
    ):
        return "shared_reference_rng", reference_true_states
    return None, None


def _score_tier1(arrays: dict[str, np.ndarray]) -> TierReport:
    required_shapes = {
        "true_states": (100,),
        "observations": (100,),
        "kf_means": (100,),
        "kf_vars": (100,),
        "pf_means": (100,),
        "pf_vars": (100,),
        "ess_trajectory": (100,),
    }
    for key, shape in required_shapes.items():
        error = _require_shape(arrays, key, shape)
        if error:
            return TierReport(False, error)

    truth_states, truth_obs = _generate_tier1_truth()
    if not _allclose(arrays["true_states"], truth_states) or not _allclose(arrays["observations"], truth_obs):
        return TierReport(False, "tier1 true_states/observations do not match the published seed and model")

    for key in required_shapes:
        error = _finite_array(key, arrays[key])
        if error:
            return TierReport(False, error)
    if np.any(arrays["kf_vars"] <= 0) or np.any(arrays["pf_vars"] < 0):
        return TierReport(False, "tier1 variances must be positive / non-negative")
    if np.any(arrays["ess_trajectory"] < 1.0) or np.any(arrays["ess_trajectory"] > TIER1_NUM_PARTICLES + 1e-6):
        return TierReport(False, "tier1 ESS must stay within [1, 1000]")

    expected_kf_means, expected_kf_vars = _kalman_filter_tier1(arrays["observations"])
    if not _allclose(arrays["kf_means"], expected_kf_means, atol=1e-6, rtol=1e-6):
        return TierReport(False, "tier1 kf_means do not match the exact Kalman filter for the submitted observations")
    if not _allclose(arrays["kf_vars"], expected_kf_vars, atol=1e-6, rtol=1e-6):
        return TierReport(False, "tier1 kf_vars do not match the exact Kalman filter for the submitted observations")

    abs_errors_mean = np.abs(arrays["pf_means"] - arrays["kf_means"])
    rel_errors_var = np.abs(arrays["pf_vars"] - arrays["kf_vars"]) / np.maximum(arrays["kf_vars"], 1e-12)
    max_abs_error_mean = float(np.max(abs_errors_mean))
    max_rel_error_var = float(np.max(rel_errors_var))

    if "abs_errors_mean" in arrays and not _allclose(arrays["abs_errors_mean"], abs_errors_mean, atol=1e-6, rtol=1e-6):
        return TierReport(False, "tier1 abs_errors_mean is inconsistent with pf_means vs kf_means")
    if "rel_errors_var" in arrays and not _allclose(arrays["rel_errors_var"], rel_errors_var, atol=1e-6, rtol=1e-6):
        return TierReport(False, "tier1 rel_errors_var is inconsistent with pf_vars vs kf_vars")
    if "max_abs_error_mean" in arrays and not np.isclose(
        float(arrays["max_abs_error_mean"]), max_abs_error_mean, atol=1e-6, rtol=1e-6
    ):
        return TierReport(False, "tier1 max_abs_error_mean is inconsistent with the submitted trajectories")
    if "max_rel_error_var" in arrays and not np.isclose(
        float(arrays["max_rel_error_var"]), max_rel_error_var, atol=1e-6, rtol=1e-6
    ):
        return TierReport(False, "tier1 max_rel_error_var is inconsistent with the submitted trajectories")

    passed = max_abs_error_mean < TIER1_MEAN_THRESHOLD and max_rel_error_var < TIER1_VAR_THRESHOLD
    return TierReport(
        passed,
        "tier1 passed" if passed else "tier1 thresholds not met",
        metrics={
            "max_abs_error_mean": max_abs_error_mean,
            "max_rel_error_var": max_rel_error_var,
            "mean_ess": float(np.mean(arrays["ess_trajectory"])),
        },
    )


def _score_tier2(arrays: dict[str, np.ndarray], reference_arrays: dict[str, np.ndarray]) -> TierReport:
    required_shapes = {
        "true_states": (200, 4),
        "observations": (200, 2),
        "filter_means": (200, 4),
        "filter_covs": (200, 4, 4),
        "ess_trajectory": (200,),
        "rmse_position": (200,),
        "rmse_velocity": (200,),
    }
    for key, shape in required_shapes.items():
        error = _require_shape(arrays, key, shape)
        if error:
            return TierReport(False, error)
    for key in ("overall_rmse_pos", "overall_rmse_vel", "num_resampling_events"):
        if key not in arrays or arrays[key].shape != ():
            return TierReport(False, f"missing scalar `{key}`")

    fresh_true_states, fresh_observations = _generate_tier2_truth_fresh()
    truth_mode, matched_truth = _match_truth(
        arrays["true_states"],
        arrays["observations"],
        fresh_true_states,
        fresh_observations,
        reference_arrays["true_states"],
        reference_arrays["observations"],
    )
    if truth_mode is None or matched_truth is None:
        return TierReport(False, "tier2 true_states/observations do not match either accepted deterministic interpretation")

    for key in required_shapes:
        error = _finite_array(key, arrays[key])
        if error:
            return TierReport(False, error)
    error = _validate_covariances("tier2 filter_covs", arrays["filter_covs"])
    if error:
        return TierReport(False, error)
    if np.any(arrays["ess_trajectory"] < 1.0) or np.any(arrays["ess_trajectory"] > TIER2_NUM_PARTICLES + 1e-6):
        return TierReport(False, "tier2 ESS must stay within [1, 5000]")

    rmse_position = np.sqrt(np.sum((arrays["filter_means"][:, :2] - matched_truth[:, :2]) ** 2, axis=1))
    rmse_velocity = np.sqrt(np.sum((arrays["filter_means"][:, 2:] - matched_truth[:, 2:]) ** 2, axis=1))
    overall_rmse_pos = float(np.mean(rmse_position))
    overall_rmse_vel = float(np.mean(rmse_velocity))
    mean_ess = float(np.mean(arrays["ess_trajectory"]))
    try:
        num_resampling_events = int(arrays["num_resampling_events"])
    except (TypeError, ValueError, OverflowError) as exc:
        return TierReport(False, f"tier2 num_resampling_events is not an integer: {exc}")

    if not _allclose(arrays["rmse_position"], rmse_position, atol=1e-6, rtol=1e-6):
        return TierReport(False, "tier2 rmse_position is inconsistent with filter_means vs true_states")
    if not _allclose(arrays["rmse_velocity"], rmse_velocity, atol=1e-6, rtol=1e-6):
        return TierReport(False, "tier2 rmse_velocity is inconsistent with filter_means vs true_states")
    if not np.isclose(float(arrays["overall_rmse_pos"]), overall_rmse_pos, atol=1e-6, rtol=1e-6):
        return TierReport(False, "tier2 overall_rmse_pos is inconsistent with rmse_position")
    if not np.isclose(float(arrays["overall_rmse_vel"]), overall_rmse_vel, atol=1e-6, rtol=1e-6):
        return TierReport(False, "tier2 overall_rmse_vel is inconsistent with rmse_velocity")
    if num_resampling_events < 0 or num_resampling_events > 200:
        return TierReport(False, "tier2 num_resampling_events must stay within [0, 200]")

    passed = (
        overall_rmse_pos < TIER2_POS_RMSE_THRESHOLD
        and overall_rmse_vel < TIER2_VEL_RMSE_THRESHOLD
        and mean_ess > TIER2_MEAN_ESS_THRESHOLD
    )
    return TierReport(
        passed,
        "tier2 passed" if passed else "tier2 thresholds not met",
        metrics={
            "overall_rmse_pos": overall_rmse_pos,
            "overall_rmse_vel": overall_rmse_vel,
            "mean_ess": mean_ess,
            "num_resampling_events": float(num_resampling_events),
        },
        notes=[f"truth_mode={truth_mode}"],
    )


def _score_tier3(
    arrays: dict[str, np.ndarray] | None,
    reference_arrays: dict[str, np.ndarray],
) -> TierReport:
    if arrays is None:
        return TierReport(False, "tier3_results.npz missing")

    required_shapes = {
        "true_states": (200, 5),
        "observations": (200,),
        "filter_means": (200, 5),
        "filter_covs": (200, 5, 5),
        "smoother_means": (200, 5),
        "smoother_covs": (200, 5, 5),
        "ess_trajectory": (200,),
        "rmse_filter_pos": (200,),
        "rmse_smoother_pos": (200,),
    }
    for key, shape in required_shapes.items():
        error = _require_shape(arrays, key, shape)
        if error:
            return TierReport(False, error)
    for key in ("overall_rmse_filter_pos", "overall_rmse_smoother_pos"):
        if key not in arrays or arrays[key].shape != ():
            return TierReport(False, f"missing scalar `{key}`")

    fresh_true_states, fresh_observations = _generate_tier3_truth_fresh()
    truth_mode, matched_truth = _match_truth(
        arrays["true_states"],
        arrays["observations"],
        fresh_true_states,
        fresh_observations,
        reference_arrays["true_states"],
        reference_arrays["observations"],
    )
    if truth_mode is None or matched_truth is None:
        return TierReport(False, "tier3 true_states/observations do not match either accepted deterministic interpretation")

    for key in required_shapes:
        error = _finite_array(key, arrays[key])
        if error:
            return TierReport(False, error)
    for name in ("filter_covs", "smoother_covs"):
        error = _validate_covariances(f"tier3 {name}", arrays[name])
        if error:
            return TierReport(False, error)
    if np.any(arrays["ess_trajectory"] < 1.0) or np.any(arrays["ess_trajectory"] > TIER3_NUM_PARTICLES + 1e-6):
        return TierReport(False, "tier3 ESS must stay within [1, 50000]")

    rmse_filter_pos = np.sqrt(np.sum((arrays["filter_means"][:, :2] - matched_truth[:, :2]) ** 2, axis=1))
    rmse_smoother_pos = np.sqrt(np.sum((arrays["smoother_means"][:, :2] - matched_truth[:, :2]) ** 2, axis=1))
    overall_rmse_filter_pos = float(np.mean(rmse_filter_pos))
    overall_rmse_smoother_pos = float(np.mean(rmse_smoother_pos))
    mean_ess = float(np.mean(arrays["ess_trajectory"]))

    if not _allclose(arrays["rmse_filter_pos"], rmse_filter_pos, atol=1e-6, rtol=1e-6):
        return TierReport(False, "tier3 rmse_filter_pos is inconsistent with filter_means vs true_states")
    if not _allclose(arrays["rmse_smoother_pos"], rmse_smoother_pos, atol=1e-6, rtol=1e-6):
        return TierReport(False, "tier3 rmse_smoother_pos is inconsistent with smoother_means vs true_states")
    if not np.isclose(
        float(arrays["overall_rmse_filter_pos"]), overall_rmse_filter_pos, atol=1e-6, rtol=1e-6
    ):
        return TierReport(False, "tier3 overall_rmse_filter_pos is inconsistent with rmse_filter_pos")
    if not np.isclose(
        float(arrays["overall_rmse_smoother_pos"]), overall_rmse_smoother_pos, atol=1e-6, rtol=1e-6
    ):
        return TierReport(False, "tier3 overall_rmse_smoother_pos is inconsistent with rmse_smoother_pos")

    passed = (
        overall_rmse_filter_pos < TIER3_FILTER_RMSE_THRESHOLD
        and overall_rmse_smoother_pos < overall_rmse_filter_pos
        and mean_ess > TIER3_MEAN_ESS_THRESHOLD
    )
    return TierReport(
        passed,
        "tier3 passed" if passed else "tier3 thresholds not met",
        metrics={
            "overall_rmse_filter_pos": overall_rmse_filter_pos,
            "overall_rmse_smoother_pos": overall_rmse_smoother_pos,
            "mean_ess": mean_ess,
        },
        notes=[f"truth_mode={truth_mode}"],
    )


def _validate_results_json(
    results_json: dict[str, Any] | None,
    tier1: TierReport,
    tier2: TierReport,
    tier3: TierReport,
    *,
    tier3_present: bool,
) -> tuple[bool, list[str]]:
    if results_json is None:
        return False, ["results.json is missing or invalid JSON"]
    if not isinstance(results_json, dict):
        return False, ["results.json is not a JSON object"]
    notes: list[str] = []

    def _compare_metric(
        section_name: str,
        section: dict[str, Any],
        result_key: str,
        metric_name: str,
        metrics: dict[str, float],
    ) -> tuple[bool, str]:
        if result_key not in section:
            return False, f"{section_name} missing `{result_key}`"
        if metric_name not in metrics:
            return False, f"results.json cross-check unavailable: missing computed `{metric_name}`"
        try:
            value = float(section[result_key])
        except (TypeError, ValueError, OverflowError) as exc:
            return False, f"{section_name} `{result_key}` is not a numeric value: {exc}"
        if not np.isclose(value, metrics[metric_name], atol=1e-6, rtol=1e-6):
            return False, f"{section_name} `{result_key}` does not match corresponding npz metric"
        return True, ""

    for key in ("tier1", "tier2"):
        if key not in results_json or not isinstance(results_json[key], dict):
            return False, [f"results.json missing `{key}` section"]

    if not tier1.passed:
        return False, ["tier1 npz validation failed; skipping results.json cross-check"]

    tier1_json = results_json["tier1"]
    for key, metric_name in (
        ("max_abs_error_mean", "max_abs_error_mean"),
        ("max_rel_error_variance", "max_rel_error_var"),
    ):
        ok, msg = _compare_metric("results.json tier1", tier1_json, key, metric_name, tier1.metrics)
        if not ok:
            return False, [msg]

    if not tier2.passed:
        return False, ["tier2 npz validation failed; skipping results.json cross-check"]

    tier2_json = results_json["tier2"]
    for key in ("overall_rmse_pos", "overall_rmse_vel", "num_resampling_events"):
        if key not in tier2_json:
            return False, [f"results.json tier2 missing `{key}`"]
    for key, metric_name in (
        ("overall_rmse_pos", "overall_rmse_pos"),
        ("overall_rmse_vel", "overall_rmse_vel"),
    ):
        ok, msg = _compare_metric("results.json tier2", tier2_json, key, metric_name, tier2.metrics)
        if not ok:
            return False, [msg]
    if "num_resampling_events" not in tier2_json:
        return False, ["results.json tier2 missing `num_resampling_events`"]
    if "num_resampling_events" not in tier2.metrics:
        return False, ["results.json cross-check unavailable: missing computed `num_resampling_events`"]
    try:
        submitted_num_resampling = int(tier2_json["num_resampling_events"])
    except (TypeError, ValueError, OverflowError) as exc:
        return False, [f"results.json tier2 `num_resampling_events` is not an integer: {exc}"]
    if submitted_num_resampling != int(tier2.metrics["num_resampling_events"]):
        return False, ["results.json tier2 num_resampling_events does not match tier2_results.npz"]
    if "mean_ess" in tier2_json:
        try:
            submitted_mean_ess = float(tier2_json["mean_ess"])
        except (TypeError, ValueError, OverflowError) as exc:
            return False, [f"results.json tier2 mean_ess is not numeric: {exc}"]
        if not np.isclose(submitted_mean_ess, tier2.metrics["mean_ess"], atol=1e-6, rtol=1e-6):
            return False, ["results.json tier2 mean_ess does not match tier2_results.npz"]

    if tier3_present:
        if not tier3.passed:
            notes.append("tier3 npz validation failed; skipping tier3 results.json cross-check")
        elif "tier3" not in results_json or not isinstance(results_json["tier3"], dict):
            return False, ["results.json missing tier3 section even though tier3_results.npz is present"]
        else:
            tier3_json = results_json["tier3"]
            for key, metric_name in (
                ("overall_rmse_filter_pos", "overall_rmse_filter_pos"),
                ("overall_rmse_smoother_pos", "overall_rmse_smoother_pos"),
            ):
                ok, msg = _compare_metric("results.json tier3", tier3_json, key, metric_name, tier3.metrics)
                if not ok:
                    return False, [msg]
            if "mean_ess" in tier3_json:
                try:
                    submitted_mean_ess = float(tier3_json["mean_ess"])
                except (TypeError, ValueError, OverflowError) as exc:
                    return False, [f"results.json tier3 mean_ess is not numeric: {exc}"]
                if not np.isclose(submitted_mean_ess, tier3.metrics["mean_ess"], atol=1e-6, rtol=1e-6):
                    return False, ["results.json tier3 mean_ess does not match tier3_results.npz"]
    elif "tier3" in results_json:
        notes.append("results.json includes tier3 section without tier3_results.npz; tier3 treated as failed")

    return True, notes


def score_submission(output_files: dict[str, bytes], reference_files: dict[str, bytes]) -> ScoreReport:
    missing_outputs = [name for name in REQUIRED_OUTPUT_FILES if name not in output_files]
    if missing_outputs:
        return ScoreReport(
            score=0.0,
            solver_present="pf_solver.py" in output_files,
            results_json_present="results.json" in output_files,
            results_json_valid=False,
            tier1=TierReport(False, f"missing required output files: {', '.join(missing_outputs)}"),
            tier2=TierReport(False, "missing required output files"),
            tier3=TierReport(False, "missing required output files"),
            notes=[],
        )

    missing_refs = [name for name in REQUIRED_REFERENCE_FILES if name not in reference_files]
    if missing_refs:
        return ScoreReport(
            score=0.0,
            solver_present=True,
            results_json_present=True,
            results_json_valid=False,
            tier1=TierReport(False, "reference files missing during evaluation"),
            tier2=TierReport(False, "reference files missing during evaluation"),
            tier3=TierReport(False, "reference files missing during evaluation"),
            notes=[f"missing reference files: {', '.join(missing_refs)}"],
        )

    solver_present = bool(output_files["pf_solver.py"].strip())
    if not solver_present:
        return ScoreReport(
            score=0.0,
            solver_present=False,
            results_json_present=True,
            results_json_valid=False,
            tier1=TierReport(False, "pf_solver.py is empty"),
            tier2=TierReport(False, "pf_solver.py is empty"),
            tier3=TierReport(False, "pf_solver.py is empty"),
        )

    try:
        tier1_arrays = _load_npz(output_files["tier1_results.npz"])
        tier2_arrays = _load_npz(output_files["tier2_results.npz"])
        tier3_arrays = _load_npz(output_files["tier3_results.npz"]) if "tier3_results.npz" in output_files else None
        ref_tier2_arrays = _load_npz(reference_files["tier2_results.npz"])
        ref_tier3_arrays = _load_npz(reference_files["tier3_results.npz"])
    except Exception as exc:
        return ScoreReport(
            score=0.0,
            solver_present=True,
            results_json_present="results.json" in output_files,
            results_json_valid=False,
            tier1=TierReport(False, f"failed to parse npz files: {exc}"),
            tier2=TierReport(False, f"failed to parse npz files: {exc}"),
            tier3=TierReport(False, f"failed to parse npz files: {exc}"),
        )

    try:
        results_json = _load_json(output_files["results.json"])
        results_json_present = True
    except Exception as exc:
        results_json = None
        results_json_present = True
        json_error = str(exc)
    else:
        json_error = ""

    try:
        tier1_report = _score_tier1(tier1_arrays)
        tier2_report = _score_tier2(tier2_arrays, ref_tier2_arrays)
        tier3_report = _score_tier3(tier3_arrays, ref_tier3_arrays)
    except Exception as exc:
        return ScoreReport(
            score=0.0,
            solver_present=True,
            results_json_present=results_json is not None,
            results_json_valid=False,
            tier1=TierReport(False, f"failed to evaluate tier1: {exc}"),
            tier2=TierReport(False, f"failed to evaluate tier2: {exc}"),
            tier3=TierReport(False, f"failed to evaluate tier3: {exc}"),
            notes=[f"tiered scoring failed: {exc}"],
        )

    results_json_valid, json_notes = _validate_results_json(
        results_json,
        tier1_report,
        tier2_report,
        tier3_report,
        tier3_present=tier3_arrays is not None,
    )
    notes = list(json_notes)
    if json_error:
        notes.append(f"results.json parse error: {json_error}")

    if not results_json_valid or not tier1_report.passed or not tier2_report.passed:
        score = 0.0
    elif tier3_report.passed:
        score = 1.0
    else:
        score = 0.5

    return ScoreReport(
        score=score,
        solver_present=solver_present,
        results_json_present=results_json_present,
        results_json_valid=results_json_valid,
        tier1=tier1_report,
        tier2=tier2_report,
        tier3=tier3_report,
        notes=notes,
    )
