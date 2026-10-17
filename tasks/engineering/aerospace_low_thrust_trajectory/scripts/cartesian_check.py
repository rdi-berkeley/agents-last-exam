"""Independent full-duration Cartesian verification of saved RTN controls."""

import math

import numpy as np
from scipy.integrate import ode

from tasks.engineering.aerospace_low_thrust_trajectory.scripts import mee_dynamics as model


def inertial_rates(time, state, control_times, controls, constants):
    mu, oblateness, thrust, exhaust = constants
    index = max(0, min(len(control_times) - 2, int(np.searchsorted(control_times, time)) - 1))
    fraction = (time - control_times[index]) / (control_times[index + 1] - control_times[index])
    radial_control, transverse_control, normal_control = controls[index] + fraction * (
        controls[index + 1] - controls[index]
    )
    position_x, position_y, position_z, velocity_x, velocity_y, velocity_z, mass = state
    radius = math.sqrt(position_x**2 + position_y**2 + position_z**2)
    momentum_x = position_y * velocity_z - position_z * velocity_y
    momentum_y = position_z * velocity_x - position_x * velocity_z
    momentum_z = position_x * velocity_y - position_y * velocity_x
    momentum = math.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2)
    normal_x, normal_y, normal_z = (
        momentum_x / momentum,
        momentum_y / momentum,
        momentum_z / momentum,
    )
    radial_x, radial_y, radial_z = position_x / radius, position_y / radius, position_z / radius
    transverse_x = normal_y * radial_z - normal_z * radial_y
    transverse_y = normal_z * radial_x - normal_x * radial_z
    transverse_z = normal_x * radial_y - normal_y * radial_x
    vertical = position_z**2 / radius**2
    central = -mu / radius**3
    j2 = oblateness / radius**5
    acceleration_x = (central + j2 * (5 * vertical - 1)) * position_x + thrust / mass * (
        radial_control * radial_x + transverse_control * transverse_x + normal_control * normal_x
    )
    acceleration_y = (central + j2 * (5 * vertical - 1)) * position_y + thrust / mass * (
        radial_control * radial_y + transverse_control * transverse_y + normal_control * normal_y
    )
    acceleration_z = (central + j2 * (5 * vertical - 3)) * position_z + thrust / mass * (
        radial_control * radial_z + transverse_control * transverse_z + normal_control * normal_z
    )
    return [
        velocity_x,
        velocity_y,
        velocity_z,
        acceleration_x,
        acceleration_y,
        acceleration_z,
        -thrust
        / exhaust
        * math.sqrt(radial_control**2 + transverse_control**2 + normal_control**2),
    ]


def replay(trajectory, control):
    """Replay from the prescribed initial state and return physical diagnostics."""
    indices = np.unique(np.linspace(0, len(trajectory) - 1, min(2001, len(trajectory)), dtype=int))
    position, velocity = model.cartesian(model.INITIAL)
    constants = (model.MU, 1.5 * model.J2 * model.MU * model.RE**2, model.THRUST, model.VE)
    scale = np.r_[np.full(3, model.RE), np.full(3, np.sqrt(model.MU / model.RE)), model.M0]
    times = np.clip(trajectory[:, 0], 0, model.TF)
    interval = 0
    evaluations = 0
    surface_crossed = False

    def derivative(time, state):
        nonlocal evaluations
        evaluations += 1
        return np.asarray(
            inertial_rates(
                time,
                state * scale,
                times[interval : interval + 2],
                control[interval : interval + 2, 1:],
                constants,
            )
        ) / scale

    def surface(time, state):
        nonlocal surface_crossed
        surface_crossed = bool(np.dot(state[:3], state[:3]) <= 1 or state[6] <= 0)
        return -1 if surface_crossed else 0

    integrator = ode(derivative).set_integrator(
        "dop853", rtol=3e-11, atol=3e-11, max_step=300, first_step=300, nsteps=10000
    )
    initial = np.r_[position, velocity, model.M0]
    integrator.set_initial_value(initial / scale, 0)
    integrator.set_solout(surface)
    sampled = np.empty((len(indices), 7))
    sampled[0] = initial
    next_sample = 1
    for interval in range(len(times) - 1):
        state = integrator.integrate(times[interval + 1])
        if surface_crossed or not integrator.successful() or not np.all(np.isfinite(state)):
            return {"success": False, "message": "Cartesian replay failed or reached Earth's surface"}
        if interval + 1 == indices[next_sample]:
            sampled[next_sample] = state * scale
            next_sample += 1
    positions, velocities = sampled[:, :3], sampled[:, 3:6]
    momentum = np.cross(positions[-1], velocities[-1])
    magnitude = np.linalg.norm(momentum)
    inc_cos = -momentum[1] / (magnitude + momentum[2])
    inc_sin = momentum[0] / (magnitude + momentum[2])
    squared = 1 + inc_cos**2 + inc_sin**2
    first = np.array([1 + inc_cos**2 - inc_sin**2, 2 * inc_cos * inc_sin, -2 * inc_sin]) / squared
    second = np.array([2 * inc_cos * inc_sin, 1 - inc_cos**2 + inc_sin**2, 2 * inc_cos]) / squared
    eccentricity = np.cross(velocities[-1], momentum) / model.MU - positions[-1] / np.linalg.norm(
        positions[-1]
    )
    elements = np.r_[
        magnitude**2 / (model.MU * model.RE),
        eccentricity @ first,
        eccentricity @ second,
        inc_cos,
        inc_sin,
    ]
    reference_position, reference_velocity = model.cartesian(trajectory[indices, 1:8])
    return {
        "success": True,
        "rhs_evaluations": evaluations,
        "terminal_orbit_norm": float(
            np.linalg.norm((elements - model.TARGET) / np.r_[model.TARGET[0], np.ones(4)])
        ),
        "position_discrepancy": float(
            np.max(
                np.linalg.norm(positions - reference_position, axis=1)
                / np.linalg.norm(positions, axis=1)
            )
        ),
        "velocity_discrepancy": float(
            np.max(
                np.linalg.norm(velocities - reference_velocity, axis=1)
                / np.linalg.norm(velocities, axis=1)
            )
        ),
        "mass_discrepancy_normalized": float(
            np.max(abs(sampled[:, 6] - trajectory[indices, 7])) / model.M0
        ),
    }
