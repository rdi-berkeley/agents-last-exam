"""Objective-preserving inference for the pinned PyDESeq2 0.4.12 API."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from pydeseq2.default_inference import DefaultInference
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from .coefficients import fit_convex_regions as fit_regions
from .coefficients import fit_regions as fit_all_regions


def stable_nb_nll(counts, means, dispersion):
    """Evaluate the same NB likelihood with a stable large-shape gamma ratio."""
    shape = np.longdouble(1) / dispersion
    counts_extended = counts.astype(np.longdouble)
    means_extended = means.astype(np.longdouble)
    if shape >= 10:

        def remainder(value):
            reciprocal = 1 / value
            squared = reciprocal**2
            coefficients = [
                1 / 12,
                -1 / 360,
                1 / 1260,
                -1 / 1680,
                1 / 1188,
                -691 / 360360,
                1 / 156,
                -3617 / 122400,
            ]
            return reciprocal * np.polynomial.polynomial.polyval(squared, coefficients)

        ratio = (
            (shape + counts_extended - 0.5) * np.log1p(counts_extended / shape)
            - counts_extended
            + remainder(shape + counts_extended)
            - remainder(shape)
        )
    else:
        ratio = (
            gammaln(np.asarray(shape + counts_extended, dtype=float))
            - gammaln(float(shape))
            - counts_extended * np.log(shape)
        )
    return float(
        np.sum(
            gammaln(counts + 1)
            - ratio
            + (counts_extended + shape) * np.log1p(means_extended / shape)
            - counts_extended * np.log(means_extended)
        )
    )


def fit_dispersion(counts, design, means, center, minimum, maximum, variance, cr_reg, prior_reg):
    if prior_reg and (variance is None or not np.isfinite(variance) or variance <= 0):
        raise ValueError("The MAP objective requires a positive finite prior variance")

    def objective(log_dispersion):
        dispersion = np.exp(log_dispersion)
        value = stable_nb_nll(counts, means, dispersion)
        if cr_reg:
            weights = means / (1 + means * dispersion)
            value += 0.5 * np.linalg.slogdet((design.T * weights) @ design)[1]
        if prior_reg:
            value += (log_dispersion - np.log(center)) ** 2 / (2 * variance)
        return float(value)

    lower, upper = np.log(minimum), np.log(maximum)
    result = minimize_scalar(
        objective, bounds=(lower, upper), method="bounded", options={"xatol": 1e-12}
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Dispersion fit failed: {result.message}")
    candidates = [(result.fun, result.x), (objective(lower), lower), (objective(upper), upper)]
    _, best = min(candidates)
    return np.exp(best), True


def fit_coefficients(
    counts, factors, design, dispersion, min_mu, beta_tol, min_beta, max_beta, optimizer, maxiter
):
    if (min_mu, min_beta, max_beta) != (0.5, -30, 30):
        raise ValueError("The coefficient optimizer is defined for the pinned bounds only")
    initial = np.linalg.lstsq(design, np.log(counts / factors + 0.1), rcond=None)[0]
    regions = fit_regions(counts, design, factors, dispersion, initial)
    if not regions:
        raise RuntimeError("No feasible coefficient region")
    best = regions[0]
    if best["success"] and np.isfinite(best["kkt_residual"]) and best["kkt_residual"] > 1e-5:
        expanded = fit_all_regions(counts, design, factors, dispersion, initial)
        if expanded and expanded[0]["objective"] <= best["objective"]:
            best = expanded[0]
    certificate = [best["objective"], best["kkt_residual"], best["minimum_slack"], *best["beta"]]
    if (
        not best["success"]
        or not np.all(np.isfinite(certificate))
        or best["kkt_residual"] > 1e-5
        or best["minimum_slack"] < -1e-7
    ):
        raise RuntimeError(
            f"Coefficient fallback is not certified: {best}; counts={counts.tolist()}; "
            f"dispersion={dispersion}; factors={factors.tolist()}"
        )
    beta = np.asarray(best["beta"])
    raw_means = factors * np.exp(design @ beta)
    means = np.maximum(raw_means, min_mu)
    weights = means / (1 + means * dispersion)
    ridge = np.eye(design.shape[1]) * 1e-6
    hats = weights * np.einsum(
        "ij,jk,ki->i", design, np.linalg.inv((design.T * weights) @ design + ridge), design.T
    )
    return beta, raw_means, hats, True


class ObjectivePreservingInference(DefaultInference):
    def alpha_mle(
        self,
        counts,
        design_matrix,
        mu,
        alpha_hat,
        min_disp,
        max_disp,
        prior_disp_var=None,
        cr_reg=True,
        prior_reg=False,
        optimizer="L-BFGS-B",
    ):
        with parallel_backend(self._backend, inner_max_num_threads=1):
            results = Parallel(n_jobs=self.n_cpus, batch_size=self._batch_size)(
                delayed(fit_dispersion)(
                    counts[:, index],
                    design_matrix,
                    mu[:, index],
                    alpha_hat[index],
                    min_disp,
                    max_disp,
                    prior_disp_var,
                    cr_reg,
                    prior_reg,
                )
                for index in range(counts.shape[1])
            )
        return tuple(np.asarray(values) for values in zip(*results))

    def irls(
        self,
        counts,
        size_factors,
        design_matrix,
        disp,
        min_mu,
        beta_tol,
        min_beta=-30,
        max_beta=30,
        optimizer="L-BFGS-B",
        maxiter=250,
    ):
        with parallel_backend(self._backend, inner_max_num_threads=1):
            results = Parallel(n_jobs=self.n_cpus, batch_size=self._batch_size)(
                delayed(fit_coefficients)(
                    counts[:, index],
                    size_factors,
                    design_matrix,
                    disp[index],
                    min_mu,
                    beta_tol,
                    min_beta,
                    max_beta,
                    optimizer,
                    maxiter,
                )
                for index in range(counts.shape[1])
            )
        beta, means, hats, converged = (np.asarray(values) for values in zip(*results))
        return beta, means.T, hats.T, converged
