"""Bounded clipped-mean negative-binomial coefficient optimization."""

import itertools

import numpy as np
from scipy.linalg import qr
from scipy.optimize import LinearConstraint, linprog, minimize, nnls, root


def coefficient_objective(beta, counts, design, factors, dispersion, active=None):
    raw_means = factors * np.exp(design @ beta)
    if active is None:
        active = raw_means > 0.5
    means = np.where(active, raw_means, 0.5)
    saturated = np.maximum(counts, 0.5)
    means_extended = means.astype(np.longdouble)
    saturated_extended = saturated.astype(np.longdouble)
    difference = (means_extended - saturated_extended) / (1 / dispersion + saturated_extended)
    loss = np.sum(
        (counts + 1 / dispersion) * np.log1p(difference)
        - counts * np.log(means_extended / saturated_extended)
    ) + 0.5e-6 * np.sum(beta**2)
    gradient = (((means - counts) / (1 + dispersion * means)) * active) @ design
    gradient += 1e-6 * beta
    return float(loss), gradient


def fit_regions(counts, design, factors, dispersion, initial):
    """Enumerate the eight-sample clipped-mean regions with bounded convex fits."""
    if len(counts) != 8:
        raise ValueError("This diagnostic enumerates this task's eight samples only")
    fits = []
    threshold = np.log(0.5 / factors)
    for pattern in itertools.product([False, True], repeat=len(counts)):
        active = np.asarray(pattern)
        signs = np.where(active, 1, -1)
        constraint_matrix = signs[:, None] * design
        lower = signs * threshold
        result = minimize(
            coefficient_objective,
            np.clip(initial, -30, 30),
            args=(counts, design, factors, dispersion, active),
            jac=True,
            method="SLSQP",
            bounds=[(-30, 30)] * design.shape[1],
            constraints=[LinearConstraint(constraint_matrix, lower, np.inf)],
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        slack = constraint_matrix @ result.x - lower
        feasible = float(slack.min()) >= -1e-7
        if not feasible:
            continue
        gradient = coefficient_objective(result.x, counts, design, factors, dispersion, active)[1]
        all_constraints = np.vstack(
            [constraint_matrix, np.eye(len(initial)), -np.eye(len(initial))]
        )
        all_slack = np.concatenate([slack, result.x + 30, 30 - result.x])
        binding = all_constraints[all_slack < 1e-6]
        residual = nnls(binding.T, gradient)[1] if len(binding) else np.linalg.norm(gradient)
        fits.append(
            {
                "pattern": active.tolist(),
                "success": bool(result.success),
                "message": result.message,
                "objective": coefficient_objective(result.x, counts, design, factors, dispersion)[
                    0
                ],
                "region_objective": float(result.fun),
                "kkt_residual": float(residual),
                "minimum_slack": float(slack.min()),
                "beta": result.x.tolist(),
            }
        )
    return sorted(fits, key=lambda fit: fit["objective"])


def fit_convex_regions(counts, design, factors, dispersion, initial):
    """Fit all admissible positive-count regions; zero counts use a convex epigraph."""
    if len(counts) != 8:
        raise ValueError("This diagnostic is restricted to the task's eight samples")
    positive = np.flatnonzero(counts > 0)
    zero = np.flatnonzero(counts == 0)
    num_beta = design.shape[1]
    num_variables = num_beta + len(zero)
    candidate = minimize(
        coefficient_objective,
        np.clip(initial, -30, 30),
        args=(counts, design, factors, dispersion),
        jac=True,
        method="SLSQP",
        bounds=[(-30, 30)] * num_beta,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    initial = candidate.x
    upper_bound = coefficient_objective(initial, counts, design, factors, dispersion)[0]
    floor_loss = np.array(
        [
            coefficient_objective(
                np.array([0.0]), np.array([count]), np.array([[0.0]]), np.array([0.5]), dispersion
            )[0]
            for count in counts
        ]
    )
    forced = positive[floor_loss[positive] > upper_bound + 1e-10]
    optional = np.setdiff1d(positive, forced)
    threshold = np.log(0.5 / factors)
    initial_variables = np.concatenate(
        [
            initial,
            np.maximum(np.log(factors[zero]) + design[zero] @ initial, np.log(0.5)),
        ]
    )
    bounds = [(-30, 30)] * num_beta + [(np.log(0.5), None)] * len(zero)
    fits = []
    for pattern in itertools.product([False, True], repeat=len(optional)):
        active = np.zeros(len(counts), dtype=bool)
        active[forced] = True
        active[optional] = pattern
        signs = np.where(active[positive], 1, -1)
        constraints = np.zeros((len(counts), num_variables))
        constraints[: len(positive), :num_beta] = signs[:, None] * design[positive]
        constraints[len(positive) :, :num_beta] = -design[zero]
        constraints[len(positive) :, num_beta:] = np.eye(len(zero))
        lower = np.concatenate([signs * threshold[positive], np.log(factors[zero])])

        def objective(variables):
            beta = variables[:num_beta]
            means = np.where(active, factors * np.exp(design @ beta), 0.5)
            means[zero] = np.exp(variables[num_beta:])
            saturated = np.maximum(counts, 0.5).astype(np.longdouble)
            extended = means.astype(np.longdouble)
            delta = (extended - saturated) / (1 / dispersion + saturated)
            loss = np.sum(
                (counts + 1 / dispersion) * np.log1p(delta) - counts * np.log(extended / saturated)
            )
            loss += 0.5e-6 * np.sum(beta**2)
            score = (means - counts) / (1 + dispersion * means)
            gradient = np.concatenate([(score * active) @ design + 1e-6 * beta, score[zero]])
            return float(loss), gradient

        result = minimize(
            objective,
            initial_variables,
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=[LinearConstraint(constraints, lower, np.inf)],
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if not result.success or np.min(constraints @ result.x - lower) < -1e-7:
            feasible = linprog(
                np.zeros(num_variables),
                A_ub=-constraints,
                b_ub=-lower,
                bounds=bounds,
                method="highs",
            )
            if feasible.status == 2:
                continue
            if not feasible.success:
                raise RuntimeError(f"Region feasibility unresolved: {feasible.message}")
            result = minimize(
                objective,
                feasible.x,
                jac=True,
                method="SLSQP",
                bounds=bounds,
                constraints=[LinearConstraint(constraints, lower, np.inf)],
                options={"ftol": 1e-12, "maxiter": 2000},
            )
        slack = constraints @ result.x - lower
        bound_constraints = np.vstack([np.eye(num_variables), -np.eye(num_variables)[:num_beta]])
        bound_slack = np.concatenate(
            [result.x[:num_beta] + 30, result.x[num_beta:] - np.log(0.5), 30 - result.x[:num_beta]]
        )
        all_constraints = np.vstack([constraints, bound_constraints])
        all_slack = np.concatenate([slack, bound_slack])
        binding = all_constraints[all_slack < 1e-6]
        binding_lower = (all_constraints @ result.x - all_slack)[all_slack < 1e-6]
        if len(binding):
            _, _, pivots = qr(binding.T, pivoting=True, mode="economic")
            independent = pivots[: np.linalg.matrix_rank(binding)]
            binding, binding_lower = binding[independent], binding_lower[independent]

        def hessian(variables):
            beta = variables[:num_beta]
            means = np.where(active, factors * np.exp(design @ beta), 0.5)
            means[zero] = np.exp(variables[num_beta:])
            curvature = means * (1 + dispersion * counts) / (1 + dispersion * means) ** 2
            matrix = np.zeros((num_variables, num_variables))
            matrix[:num_beta, :num_beta] = (design.T * (curvature * active)) @ design
            matrix[:num_beta, :num_beta] += np.eye(num_beta) * 1e-6
            matrix[num_beta:, num_beta:] = np.diag(curvature[zero])
            return matrix

        def equations(values):
            variables, multipliers = values[:num_variables], values[num_variables:]
            return np.concatenate(
                [
                    objective(variables)[1] - binding.T @ multipliers,
                    binding @ variables - binding_lower,
                ]
            )

        def jacobian(values):
            return np.block(
                [
                    [hessian(values[:num_variables]), -binding.T],
                    [binding, np.zeros((len(binding), len(binding)))],
                ]
            )

        multipliers = np.linalg.lstsq(binding.T, objective(result.x)[1], rcond=None)[0]
        refined = root(
            equations,
            np.concatenate([result.x, multipliers]),
            jac=jacobian,
            method="hybr",
            options={"xtol": 1e-10},
        )
        refined_variables = refined.x[:num_variables]
        refined_slack = all_constraints @ (refined_variables - result.x) + all_slack
        if (
            np.all(np.isfinite(refined_variables))
            and refined_slack.min() >= -1e-7
            and objective(refined_variables)[0] <= result.fun + 1e-9
        ):
            result.x = refined_variables
            result.fun = objective(result.x)[0]
            all_slack = refined_slack
            binding = all_constraints[all_slack < 1e-6]
        gradient = objective(result.x)[1]
        residual = nnls(binding.T, gradient)[1] if len(binding) else np.linalg.norm(gradient)
        if residual > 1e-7:
            refined = minimize(
                objective,
                result.x,
                jac=True,
                hess=hessian,
                method="trust-constr",
                bounds=bounds,
                constraints=[LinearConstraint(constraints, lower, np.inf)],
                options={
                    "gtol": 1e-12,
                    "xtol": 1e-12,
                    "barrier_tol": 1e-12,
                    "initial_barrier_parameter": 1e-14,
                    "initial_barrier_tolerance": 1e-12,
                    "maxiter": 2000,
                },
            )
            refined_slack = all_constraints @ (refined.x - result.x) + all_slack
            if (
                refined.success
                and refined_slack.min() >= -1e-7
                and refined.fun <= result.fun + 1e-7
            ):
                result = refined
                all_slack = refined_slack
                binding = all_constraints[all_slack < 1e-6]
                gradient = objective(result.x)[1]
                residual = (
                    nnls(binding.T, gradient)[1] if len(binding) else np.linalg.norm(gradient)
                )
        beta = result.x[:num_beta]
        fits.append(
            {
                "pattern": active.tolist(),
                "success": bool(result.success),
                "message": result.message,
                "objective": coefficient_objective(beta, counts, design, factors, dispersion)[0],
                "region_objective": float(result.fun),
                "kkt_residual": float(residual),
                "minimum_slack": float(all_slack.min()),
                "beta": beta.tolist(),
            }
        )
    return sorted(fits, key=lambda fit: fit["objective"])
