"""Full osculating MEE/J2 dynamics and normalized minimum-fuel adjoints."""

from __future__ import annotations

import numpy as np

MU = 3.986004418e14
RE = 6378137.0
J2 = 1.08263e-3
THRUST = 0.5
VE = 29419.95
M0 = 2000.0
TF = 300.0 * 86400.0
SCALE = np.array([RE, 1.0, 1.0, 1.0, 1.0, 1.0, M0])
INITIAL = np.array([6678000.0, 0.0, 0.0, np.tan(np.deg2rad(28.5) / 2), 0.0, 0.0, M0])
TARGET = np.array([42164000.0 / RE, 0.0, 0.0, 0.0, 0.0])


def geometry(state):
    """Return the six-row RTN acceleration matrix and J2 acceleration in SI."""
    semilatus, ecc_cos, ecc_sin, inc_cos, inc_sin, longitude, mass = np.moveaxis(state, -1, 0)
    sine, cosine = np.sin(longitude), np.cos(longitude)
    weight = 1 + ecc_cos * cosine + ecc_sin * sine
    squared = 1 + inc_cos**2 + inc_sin**2
    zeta = inc_cos * sine - inc_sin * cosine
    eta = inc_cos * cosine + inc_sin * sine
    root = np.sqrt(semilatus / MU)
    matrix = np.zeros(state.shape[:-1] + (6, 3), dtype=state.dtype)
    matrix[..., 0, 1] = 2 * semilatus * root / weight
    matrix[..., 1, 0] = root * sine
    matrix[..., 1, 1] = root * ((weight + 1) * cosine + ecc_cos) / weight
    matrix[..., 1, 2] = -root * ecc_sin * zeta / weight
    matrix[..., 2, 0] = -root * cosine
    matrix[..., 2, 1] = root * ((weight + 1) * sine + ecc_sin) / weight
    matrix[..., 2, 2] = root * ecc_cos * zeta / weight
    matrix[..., 3, 2] = root * squared * cosine / (2 * weight)
    matrix[..., 4, 2] = root * squared * sine / (2 * weight)
    matrix[..., 5, 2] = root * zeta / weight
    coefficient = MU * RE**2 / (semilatus / weight) ** 4
    perturbation = np.stack(
        (
            -1.5 * coefficient * (1 - 12 * zeta**2 / squared**2),
            -12 * coefficient * zeta * eta / squared**2,
            -6 * coefficient * zeta * (1 - inc_cos**2 - inc_sin**2) / squared**2,
        ),
        axis=-1,
    )
    drift = np.sqrt(MU * semilatus) * (weight / semilatus) ** 2
    return matrix, perturbation, drift


def rates(state, control, *, j2=J2, thrust=THRUST):
    """SI time derivatives; control is the RTN thrust fraction, norm in [0,1]."""
    state = np.asarray(state)
    control = np.asarray(control)
    matrix, perturbation, drift = geometry(state)
    acceleration = j2 * perturbation + thrust * control / state[..., 6, None]
    derivative = np.empty_like(state)
    derivative[..., :6] = np.einsum("...ij,...j->...i", matrix, acceleration)
    derivative[..., 5] += drift
    derivative[..., 6] = -thrust * np.linalg.norm(control, axis=-1) / VE
    return derivative


def normalized_rates(state, control, *, duration=None, thrust=THRUST, j2=J2):
    duration = TF if duration is None else duration
    return rates(state * SCALE, control, j2=j2, thrust=thrust) * duration / SCALE


def minimizing_control(state, costate, *, duration=None, thrust=THRUST, epsilon=0.0):
    """Minimize H, with optional entropy regularization per normalized time."""
    duration = TF if duration is None else duration
    physical = state * SCALE
    matrix, _, _ = geometry(physical)
    primer = np.einsum("...ij,...i->...j", matrix, costate[..., :6] / SCALE[:6])
    primer *= (duration * thrust / physical[..., 6])[..., None]
    magnitude = np.linalg.norm(primer, axis=-1)
    coefficient = -magnitude - costate[..., 6] * duration * thrust / (M0 * VE)
    if epsilon:
        throttle = 0.5 * (1 - np.tanh(coefficient / (2 * epsilon)))
    else:
        throttle = (coefficient < 0).astype(float)
    direction = -primer / np.maximum(magnitude[..., None], 1e-100)
    return throttle[..., None] * direction, coefficient


def canonical_rates(state, costate, control, *, duration=None, thrust=THRUST, j2=J2):
    """Normalized state and adjoint rates with control held fixed in partials."""
    derivative = normalized_rates(state, control, duration=duration, thrust=thrust, j2=j2)
    adjoint = np.empty_like(costate)
    for column in range(7):
        shifted = state.astype(complex)
        shifted[..., column] += 1e-25j
        partial = (
            normalized_rates(shifted, control, duration=duration, thrust=thrust, j2=j2).imag / 1e-25
        )
        adjoint[..., column] = -np.sum(costate * partial, axis=-1)
    return derivative, adjoint


def cartesian(state):
    """Convert osculating MEE to ECI position and velocity without angle inversions."""
    semilatus, ecc_cos, ecc_sin, inc_cos, inc_sin, longitude, mass = np.moveaxis(state, -1, 0)
    squared = 1 + inc_cos**2 + inc_sin**2
    first = (
        np.stack((1 + inc_cos**2 - inc_sin**2, 2 * inc_cos * inc_sin, -2 * inc_sin), axis=-1)
        / squared[..., None]
    )
    second = (
        np.stack((2 * inc_cos * inc_sin, 1 - inc_cos**2 + inc_sin**2, 2 * inc_cos), axis=-1)
        / squared[..., None]
    )
    sine, cosine = np.sin(longitude), np.cos(longitude)
    radius = semilatus / (1 + ecc_cos * cosine + ecc_sin * sine)
    position = radius[..., None] * (cosine[..., None] * first + sine[..., None] * second)
    velocity = np.sqrt(MU / semilatus)[..., None] * (
        -(sine + ecc_sin)[..., None] * first + (cosine + ecc_cos)[..., None] * second
    )
    return position, velocity
