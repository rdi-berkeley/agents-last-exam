"""3-step scoring for ml_natural_gradient_gmm.

Used by main.py.evaluate() — receives in-memory byte blobs for agent output and
for the hidden reference fixture, returns a score in {0.0, 0.5, 1.0} plus
structured diagnostics.

Step 1: NLL accuracy + NGD superiority (per-tier tolerance vs reference).
Step 2: Trajectory consistency — recompute NLL from the agent's own trajectory
         snapshots and compare to agent's own reported nll_history; shape/count
         checks; 90% monotonicity gate.
Step 3: Fisher + gradient-check cross-check — agent's gradient_check.json must
         have all_passed=true with relative errors < tolerance; for Tier 1,
         independently recompute the regularized Fisher condition number at the
         reported initial parameters via the same quadrature the spec prescribes
         and require < 1% relative error vs the agent's reported value.

Scoring rubric (submission §verification_method):
- FULL PASS (1.0): all three steps pass for all three tiers.
- PARTIAL (0.5): Tiers 1 and 2 fully pass; Tier 3 missing or fails any step.
- FAIL (0.0): Tier 1 fails any step, gradient check fails, or fewer than 2 tiers pass.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import numpy as np
from numpy.polynomial.legendre import leggauss

NLL_TOLERANCE = {"tier1": 0.5, "tier2": 0.5, "tier3": 1.0}
NLL_RECOMPUTE_TOL = 1e-6
MONOTONIC_FRACTION = 0.90
GRADCHECK_TOL = 1e-4
GRADCHECK_CROSSCHECK_REL_TOL = 1e-3
GRADCHECK_SEED = 999
GRADCHECK_N_INDICES = 5
FISHER_COND_REL_TOL = 0.01

TIER_SPECS = {
    "tier1": {"K": 3, "D": 2, "N_data": 1000, "n_iterations": 300, "n_snapshots": 31},
    "tier2": {"K": 5, "D": 4, "N_data": 5000, "n_iterations": 500, "n_snapshots": 51},
    "tier3": {"K": 8, "D": 6, "N_data": 10000, "n_iterations": 500, "n_snapshots": 51},
}


def n_params_for(K: int, D: int) -> int:
    return K * D + K * D * (D + 1) // 2 + (K - 1)


def unpack_params(theta: np.ndarray, K: int, D: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_mean = K * D
    n_chol_per = D * (D + 1) // 2
    n_chol = K * n_chol_per
    means = theta[:n_mean].reshape(K, D)
    chol_params = theta[n_mean : n_mean + n_chol].reshape(K, n_chol_per)
    betas = theta[n_mean + n_chol :]
    return means, chol_params, betas


def chol_to_L(chol_vec: np.ndarray, D: int) -> np.ndarray:
    """Column-major packing of lower triangle; diagonal stored as logs."""
    L = np.zeros((D, D))
    idx = 0
    for j in range(D):
        L[j, j] = math.exp(chol_vec[idx])
        idx += 1
        for i in range(j + 1, D):
            L[i, j] = chol_vec[idx]
            idx += 1
    return L


def betas_to_weights(betas: np.ndarray, K: int) -> np.ndarray:
    full = np.zeros(K)
    full[: K - 1] = betas
    m = full.max()
    e = np.exp(full - m)
    return e / e.sum()


def log_gaussian_batch(X: np.ndarray, mu: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Vectorized log N(x | mu, L L^T) for X of shape (N, D)."""
    D = mu.shape[0]
    diff = X - mu[None, :]
    v = np.linalg.solve(L, diff.T).T
    maha = np.einsum("nd,nd->n", v, v)
    log_det = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (D * np.log(2 * np.pi) + log_det + maha)


def compute_nll(theta: np.ndarray, data: np.ndarray, K: int, D: int) -> float:
    means, chol_params, betas = unpack_params(theta, K, D)
    Ls = [chol_to_L(chol_params[k], D) for k in range(K)]
    ws = betas_to_weights(betas, K)
    log_comp = np.stack([log_gaussian_batch(data, means[k], Ls[k]) for k in range(K)], axis=1)
    log_comp += np.log(ws + 1e-300)[None, :]
    log_px = np.logaddexp.reduce(log_comp, axis=1)
    return float(-log_px.mean())


def score_log_p(x: np.ndarray, theta: np.ndarray, K: int, D: int) -> np.ndarray:
    """Score vector ∂ log p(x|θ) / ∂θ for a single point x. Used by Fisher recompute."""
    means, chol_params, betas = unpack_params(theta, K, D)
    Ls = np.stack([chol_to_L(chol_params[k], D) for k in range(K)])
    ws = betas_to_weights(betas, K)

    log_each = np.empty(K)
    diffs = np.empty((K, D))
    vs = np.empty((K, D))
    for k in range(K):
        diff = x - means[k]
        v = np.linalg.solve(Ls[k], diff)
        diffs[k] = diff
        vs[k] = v
        maha = v @ v
        log_det = 2.0 * np.log(np.diag(Ls[k])).sum()
        log_each[k] = -0.5 * (D * np.log(2 * np.pi) + log_det + maha)
    log_w = np.log(ws + 1e-300)
    log_joint = log_each + log_w
    m = log_joint.max()
    r = np.exp(log_joint - m)
    r /= r.sum()
    nparam = n_params_for(K, D)
    grad = np.zeros(nparam)
    n_mean = K * D
    n_chol_per = D * (D + 1) // 2

    for k in range(K):
        L = Ls[k]
        v = vs[k]
        Lt_inv_v = np.linalg.solve(L.T, v)
        grad[k * D : (k + 1) * D] += r[k] * Lt_inv_v
        dlog_dL = -np.linalg.solve(L.T, np.eye(D)) + np.outer(Lt_inv_v, v)
        dlog_dL = np.tril(dlog_dL)
        idx = n_mean + k * n_chol_per
        for j in range(D):
            grad[idx] += r[k] * (dlog_dL[j, j] * L[j, j])
            idx += 1
            for i in range(j + 1, D):
                grad[idx] += r[k] * dlog_dL[i, j]
                idx += 1
    betas_idx = n_mean + K * n_chol_per
    for k in range(K - 1):
        grad[betas_idx + k] = r[k] - ws[k]
    return grad


def _load_json(blob: bytes) -> dict:
    return json.loads(blob.decode("utf-8"))


def _load_npy(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


@dataclass
class Blobs:
    """In-memory file tree: {relative_path: bytes}. Missing paths allowed."""

    files: Dict[str, bytes] = field(default_factory=dict)

    def get(self, path: str) -> bytes | None:
        return self.files.get(path)

    def has(self, path: str) -> bool:
        return path in self.files and self.files[path] is not None

    def add(self, path: str, data: bytes) -> None:
        self.files[path] = data


def _initial_params(data: np.ndarray, K: int, D: int) -> np.ndarray:
    """Reproduce the spec's canonical initialization (seed=123)."""
    rng_init = np.random.default_rng(123)
    data_mean = np.mean(data, axis=0)
    n_mean = K * D
    n_chol_per = D * (D + 1) // 2
    theta = np.zeros(n_params_for(K, D))
    for k in range(K):
        theta[k * D : (k + 1) * D] = data_mean + 0.1 * rng_init.standard_normal(D)
    # chol block: zeros (identity L, log-diag=0)
    # beta block: zeros
    return theta


def _tier1_fisher_cond_at_init(data: np.ndarray, lam: float, n_quad: int = 100) -> float:
    """Independent quadrature Fisher for Tier 1 at initial parameters, with regularization."""
    K, D = 3, 2
    theta0 = _initial_params(data, K, D)
    means, chol_params, betas = unpack_params(theta0, K, D)
    Ls = [chol_to_L(chol_params[k], D) for k in range(K)]
    ws = betas_to_weights(betas, K)
    centers = np.stack(means)
    sigmas = np.array([np.sqrt(max(np.diag(Ls[k] @ Ls[k].T).max(), 0.0)) for k in range(K)])
    max_sigma = float(sigmas.max())
    lows = centers.min(axis=0) - 5.0 * max_sigma
    highs = centers.max(axis=0) + 5.0 * max_sigma
    nodes_x, weights_x = leggauss(n_quad)
    nodes_y, weights_y = leggauss(n_quad)
    ax = 0.5 * (highs[0] - lows[0]) * nodes_x + 0.5 * (highs[0] + lows[0])
    ay = 0.5 * (highs[1] - lows[1]) * nodes_y + 0.5 * (highs[1] + lows[1])
    wx = 0.5 * (highs[0] - lows[0]) * weights_x
    wy = 0.5 * (highs[1] - lows[1]) * weights_y
    nparam = n_params_for(K, D)
    F = np.zeros((nparam, nparam))
    for i in range(n_quad):
        for j in range(n_quad):
            x = np.array([ax[i], ay[j]])
            log_comp = np.array(
                [log_gaussian_batch(x[None, :], means[k], Ls[k])[0] + math.log(ws[k]) for k in range(K)]
            )
            mx = log_comp.max()
            p = math.exp(mx) * np.exp(log_comp - mx).sum()
            s = score_log_p(x, theta0, K, D)
            F += wx[i] * wy[j] * p * np.outer(s, s)
    F_reg = F + lam * np.eye(nparam)
    eigs = np.linalg.eigvalsh(F_reg)
    return float(eigs.max() / max(eigs.min(), 1e-300))


def step1_nll(results: dict, ref_results: dict, tier: str) -> dict:
    if tier not in results:
        return {"passed": False, "reason": f"tier missing in results.json: {tier}"}
    r = results[tier]
    ref = ref_results.get(tier, {})
    final_ngd = r.get("final_nll_ngd")
    final_gd = r.get("final_nll_gd")
    if final_ngd is None or not math.isfinite(final_ngd):
        return {"passed": False, "reason": f"{tier}.final_nll_ngd non-finite"}
    if final_gd is None or not math.isfinite(final_gd):
        return {"passed": False, "reason": f"{tier}.final_nll_gd non-finite"}
    if not (final_ngd < final_gd):
        return {
            "passed": False,
            "reason": f"{tier}: NGD did not beat GD ({final_ngd:.4f} >= {final_gd:.4f})",
        }
    ref_ngd = ref.get("final_nll_ngd")
    tol = NLL_TOLERANCE[tier]
    if ref_ngd is not None and abs(final_ngd - ref_ngd) > tol:
        return {
            "passed": False,
            "reason": f"{tier}: final_nll_ngd={final_ngd:.4f} outside ±{tol} of ref={ref_ngd:.4f}",
        }
    return {
        "passed": True,
        "final_nll_ngd": final_ngd,
        "final_nll_gd": final_gd,
        "reason": f"{tier}: NGD {final_ngd:.4f} < GD {final_gd:.4f}, within ±{tol} of ref",
    }


def step2_trajectory(agent: Blobs, data: np.ndarray, tier: str) -> dict:
    # Only NGD trajectories are cross-checked here; GD histories are consumed by Step 1
    # (final_nll_gd) and have no reference ground truth for snapshot-level replay.
    spec = TIER_SPECS[tier]
    K, D = spec["K"], spec["D"]
    n_snap = spec["n_snapshots"]
    nparam_expected = n_params_for(K, D)
    nll_ngd_blob = agent.get(f"nll_history/nll_ngd_{tier}.npy")
    traj_ngd_blob = agent.get(f"trajectories/params_ngd_{tier}.npy")
    if nll_ngd_blob is None or traj_ngd_blob is None:
        return {"passed": False, "reason": f"{tier}: missing nll_history or trajectories"}
    try:
        nll_hist = _load_npy(nll_ngd_blob)
        traj = _load_npy(traj_ngd_blob)
    except Exception as exc:
        return {"passed": False, "reason": f"{tier}: cannot load trajectory arrays: {exc}"}
    if nll_hist.shape != (spec["n_iterations"] + 1,):
        return {
            "passed": False,
            "reason": f"{tier}: nll_ngd shape {nll_hist.shape}, expected ({spec['n_iterations'] + 1},)",
        }
    if traj.shape != (n_snap, nparam_expected):
        return {
            "passed": False,
            "reason": f"{tier}: traj shape {traj.shape}, expected ({n_snap}, {nparam_expected})",
        }
    # Recompute NLL at each snapshot and compare to reported nll_history[10*i]
    recomputed = np.array([compute_nll(traj[i], data, K, D) for i in range(n_snap)])
    reported = nll_hist[:: 10][:n_snap]
    if reported.shape != recomputed.shape:
        return {
            "passed": False,
            "reason": f"{tier}: reported snapshot shape {reported.shape} vs recomputed {recomputed.shape}",
        }
    if not np.all(np.isfinite(recomputed)):
        return {"passed": False, "reason": f"{tier}: recomputed NLL contains non-finite values"}
    max_abs_diff = float(np.max(np.abs(recomputed - reported)))
    if max_abs_diff > NLL_RECOMPUTE_TOL:
        return {
            "passed": False,
            "reason": f"{tier}: max |recomputed - reported| NLL = {max_abs_diff:.3e} > {NLL_RECOMPUTE_TOL}",
        }
    # Monotonicity on full NLL history
    diffs = np.diff(nll_hist)
    frac_non_increasing = float(np.mean(diffs <= 0))
    if frac_non_increasing < MONOTONIC_FRACTION:
        return {
            "passed": False,
            "reason": f"{tier}: NLL non-increasing fraction {frac_non_increasing:.3f} < {MONOTONIC_FRACTION}",
        }
    return {
        "passed": True,
        "max_recompute_diff": max_abs_diff,
        "frac_non_increasing": frac_non_increasing,
        "reason": f"{tier}: snapshot consistency + monotonicity OK",
    }


def _independent_analytic_grad_at_init(data_tier1: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    """Evaluator's own analytic NLL-gradient entries at Tier 1 init for the given parameter indices."""
    K, D = 3, 2
    theta0 = _initial_params(data_tier1, K, D)
    indices = list(indices)
    accum = np.zeros(len(indices))
    N = data_tier1.shape[0]
    for n in range(N):
        s = score_log_p(data_tier1[n], theta0, K, D)
        for j, idx in enumerate(indices):
            accum[j] += s[idx]
    return -(accum / N)


def step3_gradient_and_fisher(agent: Blobs, data_tier1: np.ndarray, results: dict) -> dict:
    gc_blob = agent.get("gradient_check.json")
    if gc_blob is None:
        return {"passed": False, "reason": "missing gradient_check.json"}
    try:
        gc = _load_json(gc_blob)
    except Exception as exc:
        return {"passed": False, "reason": f"gradient_check.json unparseable: {exc}"}
    if not gc.get("all_passed", False):
        return {"passed": False, "reason": "gradient_check.all_passed is not true"}
    rel_errs = gc.get("relative_errors", [])
    if not rel_errs or any((e is None or e > GRADCHECK_TOL) for e in rel_errs):
        return {
            "passed": False,
            "reason": f"gradient_check relative_errors violate tolerance: {rel_errs}",
        }
    # Cross-check agent's reported analytic gradient against an independent evaluator
    # computation at the same Tier 1 init, preventing a cheat path where the agent
    # forges all_passed=true with fabricated numbers. Also require the 5 indices to
    # match the spec's deterministic seed=999 draw.
    reported_indices = gc.get("parameter_indices")
    reported_analytic = gc.get("analytic_gradient")
    if reported_indices is None or reported_analytic is None:
        return {
            "passed": False,
            "reason": "gradient_check.json missing parameter_indices or analytic_gradient",
        }
    if (
        len(reported_indices) != GRADCHECK_N_INDICES
        or len(reported_analytic) != GRADCHECK_N_INDICES
    ):
        return {
            "passed": False,
            "reason": (
                f"gradient_check arity mismatch: indices={len(reported_indices)} "
                f"analytic={len(reported_analytic)} expected={GRADCHECK_N_INDICES}"
            ),
        }
    expected_indices = set(
        int(i) for i in np.random.default_rng(GRADCHECK_SEED).choice(
            n_params_for(3, 2), size=GRADCHECK_N_INDICES, replace=False
        )
    )
    reported_set = set(int(i) for i in reported_indices)
    if reported_set != expected_indices:
        return {
            "passed": False,
            "reason": (
                f"gradient_check.parameter_indices {sorted(reported_set)} do not match "
                f"spec-required set {sorted(expected_indices)} (seed=999 draw)"
            ),
        }
    try:
        indep_analytic = _independent_analytic_grad_at_init(data_tier1, reported_indices)
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"independent analytic-gradient recomputation raised: {exc}",
        }
    reported_analytic_arr = np.asarray(reported_analytic, dtype=float)
    denom = np.maximum(np.abs(indep_analytic), 1e-12)
    cross_rel = np.abs(reported_analytic_arr - indep_analytic) / denom
    if not np.all(np.isfinite(cross_rel)):
        return {"passed": False, "reason": "gradient cross-check produced non-finite residuals"}
    if np.max(cross_rel) > GRADCHECK_CROSSCHECK_REL_TOL:
        return {
            "passed": False,
            "reason": (
                "gradient_check.analytic_gradient disagrees with evaluator's independent "
                f"computation: max rel_err={float(np.max(cross_rel)):.3e} > {GRADCHECK_CROSSCHECK_REL_TOL}"
            ),
        }
    # Fisher condition number cross-check for Tier 1 at initial parameters (snapshot index 0).
    fc_blob = agent.get("fisher_cond/fisher_cond_tier1.npy")
    if fc_blob is None:
        return {"passed": False, "reason": "missing fisher_cond/fisher_cond_tier1.npy"}
    fc = _load_npy(fc_blob)
    if fc.ndim != 1 or fc.size == 0:
        return {"passed": False, "reason": f"fisher_cond_tier1 malformed shape {fc.shape}"}
    if not np.all(np.isfinite(fc)) or not np.all(fc > 0):
        return {"passed": False, "reason": "fisher_cond contains non-finite or non-positive entries"}
    reported_cond0 = float(fc[0])
    lam = results.get("tier1", {}).get("fisher_lambda", 1e-4)
    try:
        indep_cond0 = _tier1_fisher_cond_at_init(data_tier1, lam=lam, n_quad=100)
    except Exception as exc:
        return {
            "passed": False,
            "reason": f"independent Fisher recomputation raised: {exc}",
        }
    rel = abs(indep_cond0 - reported_cond0) / max(indep_cond0, 1e-300)
    if rel > FISHER_COND_REL_TOL:
        return {
            "passed": False,
            "reason": (
                f"Tier 1 Fisher cond (init): agent={reported_cond0:.3e} vs independent={indep_cond0:.3e}"
                f" rel_err={rel:.3f} > {FISHER_COND_REL_TOL}"
            ),
        }
    return {
        "passed": True,
        "grad_rel_errs": rel_errs,
        "grad_crosscheck_max_rel_err": float(np.max(cross_rel)),
        "fisher_cond_rel_err": rel,
        "reason": (
            f"gradient check OK (cross-check max_rel_err={float(np.max(cross_rel)):.2e}) "
            f"and Tier 1 Fisher cond within {FISHER_COND_REL_TOL:.0%}"
        ),
    }


def score_all(agent: Blobs, data_tier1: np.ndarray, data_tier2: np.ndarray, data_tier3: np.ndarray) -> dict:
    """Return {'score': float, 'steps': {...}, 'reason': str}."""
    res_blob = agent.get("results.json")
    if res_blob is None:
        return {"score": 0.0, "reason": "missing results.json", "steps": {}}
    try:
        results = _load_json(res_blob)
    except Exception as exc:
        return {"score": 0.0, "reason": f"results.json unparseable: {exc}", "steps": {}}
    ref_blob = agent.get("__ref__results.json")
    ref_results = {}
    if ref_blob is not None:
        try:
            ref_results = _load_json(ref_blob)
        except Exception:
            ref_results = {}

    steps: dict[str, Any] = {}
    tier_pass: dict[str, bool] = {}
    data_by_tier = {"tier1": data_tier1, "tier2": data_tier2, "tier3": data_tier3}

    grad_fisher = step3_gradient_and_fisher(agent, data_tier1, results)
    steps["gradient_and_fisher"] = grad_fisher
    grad_ok = grad_fisher["passed"]

    for tier in ("tier1", "tier2", "tier3"):
        s1 = step1_nll(results, ref_results, tier)
        s2 = step2_trajectory(agent, data_by_tier[tier], tier)
        steps[f"{tier}_step1"] = s1
        steps[f"{tier}_step2"] = s2
        tier_pass[tier] = bool(s1["passed"] and s2["passed"] and grad_ok)

    t1, t2, t3 = tier_pass["tier1"], tier_pass["tier2"], tier_pass["tier3"]
    if t1 and t2 and t3:
        score = 1.0
        reason = "FULL PASS"
    elif t1 and t2 and not t3:
        score = 0.5
        reason = "PARTIAL PASS (T3 failed)"
    else:
        score = 0.0
        if not grad_ok:
            reason = f"FAIL — gradient/Fisher: {grad_fisher['reason']}"
        elif not t1:
            s1 = steps["tier1_step1"]
            s2 = steps["tier1_step2"]
            reason = f"FAIL — Tier 1 failed: step1={s1['reason']} | step2={s2['reason']}"
        else:
            reason = "FAIL — fewer than 2 tiers passed"
    return {"score": score, "reason": reason, "tier_pass": tier_pass, "steps": steps}


def collect_outputs(read_bytes: Any, output_dir: str) -> Blobs:
    """Collect the agent-expected output tree via a read_bytes(path) callable.

    read_bytes should be a callable that, given an absolute path, returns the
    file bytes or None/b'' if missing. main.py wires this to session.read_bytes
    with try/except.
    """
    paths = [
        "results.json",
        "gradient_check.json",
        "nll_history/nll_ngd_tier1.npy",
        "nll_history/nll_ngd_tier2.npy",
        "nll_history/nll_ngd_tier3.npy",
        "nll_history/nll_gd_tier1.npy",
        "nll_history/nll_gd_tier2.npy",
        "nll_history/nll_gd_tier3.npy",
        "trajectories/params_ngd_tier1.npy",
        "trajectories/params_ngd_tier2.npy",
        "trajectories/params_ngd_tier3.npy",
        "trajectories/params_gd_tier1.npy",
        "trajectories/params_gd_tier2.npy",
        "trajectories/params_gd_tier3.npy",
        "fisher_cond/fisher_cond_tier1.npy",
        "fisher_cond/fisher_cond_tier2.npy",
        "fisher_cond/fisher_cond_tier3.npy",
    ]
    blobs = Blobs()
    for rel in paths:
        abs_path = f"{output_dir.rstrip('/')}/{rel}"
        try:
            data = read_bytes(abs_path)
        except Exception:
            data = None
        if data:
            blobs.add(rel, data)
    return blobs
