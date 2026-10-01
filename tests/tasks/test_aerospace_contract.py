import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tasks.engineering.aerospace_low_thrust_trajectory.scripts import score_outputs as scorer


@pytest.fixture(scope="module")
def independent_spiral():
    radius = 8.0e6
    inclination = 0.31
    speed = math.sqrt(3.986004418e14 / radius)
    initial = [
        radius,
        0.0,
        0.0,
        0.0,
        speed * math.cos(inclination),
        speed * math.sin(inclination),
        2000.0,
    ]

    def derivative(time, state):
        position = state[:3]
        velocity = state[3:6]
        mass = 2000.0 - 0.5 * time / (3000.0 * 9.80665)
        acceleration = -3.986004418e14 * position / np.dot(
            position, position
        ) ** 1.5 + 0.5 * velocity / (mass * math.sqrt(np.dot(velocity, velocity)))
        return [*velocity, *acceleration, -0.5 / (3000.0 * 9.80665)]

    solution = solve_ivp(
        derivative,
        (0.0, 60000.0),
        initial,
        method="RK45",
        rtol=2e-12,
        atol=1e-12,
        dense_output=True,
    )
    assert solution.success
    return solution


@pytest.mark.parametrize("count", [7, 101, 1501])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_integrated_dynamics_accept_independent_spiral_on_different_grids(
    independent_spiral, count, nonuniform
):
    fraction = np.linspace(0.0, 1.0, count)
    times = 60000.0 * (fraction**2 if nonuniform else fraction)
    trajectory = np.column_stack((times, independent_spiral.sol(times).T))
    residuals = scorer._tier2_dynamics_residuals(trajectory)
    assert np.max(residuals) < 0.005


def test_propagator_agrees_with_analytic_kepler_orbit(monkeypatch):
    monkeypatch.setattr(scorer, "THRUST", 0.0)
    radius = 9.0e6
    speed = math.sqrt(scorer.MU / radius)
    frequency = speed / radius
    times = np.array([0.0, 0.17, 1.1, 3.4]) * 2.0 * math.pi / frequency
    phase = frequency * times
    trajectory = np.column_stack(
        (
            times,
            radius * np.cos(phase),
            radius * np.sin(phase),
            np.zeros(4),
            -speed * np.sin(phase),
            speed * np.cos(phase),
            np.zeros(4),
            np.full(4, 2000.0),
        )
    )
    assert np.max(scorer._tier2_dynamics_residuals(trajectory)) < 1e-5


def test_propagation_checks_position_and_velocity(independent_spiral):
    times = np.linspace(0.0, 60000.0, 1501)
    trajectory = np.column_stack((times, independent_spiral.sol(times).T))
    wrong_position = trajectory.copy()
    wrong_position[1, 1] += 10000.0
    wrong_velocity = trajectory.copy()
    wrong_velocity[1, 4] += 100.0
    assert np.max(scorer._tier2_dynamics_residuals(wrong_position)) > 0.08
    assert np.max(scorer._tier2_dynamics_residuals(wrong_velocity)) > 0.08


def test_rotated_spiral_is_not_tied_to_reference_phase(independent_spiral):
    times = np.linspace(0.0, 60000.0, 21)
    trajectory = np.column_stack((times, independent_spiral.sol(times).T))
    rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    trajectory[:, 1:4] = trajectory[:, 1:4] @ rotation.T
    trajectory[:, 4:7] = trajectory[:, 4:7] @ rotation.T
    assert np.max(scorer._tier2_dynamics_residuals(trajectory)) < 0.005


def test_mass_propagation_matches_constant_thrust_law(independent_spiral):
    initial = np.concatenate(([0.0], independent_spiral.y[:, 0]))
    propagated = scorer._propagate_tier2(initial, 60000.0)
    assert propagated[6] == pytest.approx(2000.0 - 0.5 * 60000.0 / (3000.0 * 9.80665), abs=1e-9)


@pytest.mark.parametrize("reference_count", [7, 23])
def test_history_check_propagates_sparse_reference_states(
    independent_spiral, monkeypatch, reference_count
):
    times = np.linspace(0.0, 60000.0, 1000)
    trajectory = np.column_stack((times, independent_spiral.sol(times).T))
    reference_times = np.linspace(0.0, 60000.0, reference_count)
    reference_trajectory = np.column_stack(
        (reference_times, independent_spiral.sol(reference_times).T)
    )
    position, velocity = trajectory[-1, 1:4], trajectory[-1, 4:7]
    radius = np.linalg.norm(position)
    energy = np.dot(velocity, velocity) / 2.0 - 3.986004418e14 / radius
    semimajor_axis = -3.986004418e14 / (2.0 * energy) / 1000.0
    eccentricity = np.linalg.norm(
        np.cross(velocity, np.cross(position, velocity)) / 3.986004418e14 - position / radius
    )
    mass = float(trajectory[-1, 7])
    delta_v = 3000.0 * 9.80665 * math.log(2000.0 / mass)
    results = {
        "tier2": {
            "final_sma_km": float(semimajor_axis),
            "final_eccentricity": float(eccentricity),
            "final_inclination_deg": math.degrees(0.31),
            "dv_total_m_s": delta_v,
            "transfer_time_s": 60000.0,
            "final_mass_kg": mass,
            "fuel_consumed_kg": 2000.0 - mass,
            "edelbaum_dv_m_s": delta_v,
        }
    }
    monkeypatch.setattr(scorer, "A_LEO_M", 8.0e6)
    monkeypatch.setattr(scorer, "I_LEO_DEG", math.degrees(0.31))
    monkeypatch.setattr(scorer, "A_GEO_KM", semimajor_axis)
    assert scorer._check_tier2(results, results, trajectory, reference_trajectory) == []


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_states_fail_before_integration(value):
    trajectory = np.zeros((1000, 8))
    trajectory[400, 2] = value
    assert scorer._check_tier2({}, {}, trajectory, trajectory) == [
        "tier2 trajectory contains non-finite values"
    ]


@pytest.mark.parametrize("duplicate", [False, True])
def test_nonincreasing_times_fail_before_integration(duplicate):
    trajectory = np.zeros((1000, 8))
    trajectory[:, 0] = np.arange(1000)
    trajectory[100, 0] = trajectory[99, 0] if duplicate else trajectory[98, 0]
    assert scorer._check_tier2({}, {}, trajectory, trajectory) == [
        "tier2 trajectory time must be strictly increasing"
    ]


def test_nonpositive_propagation_interval_is_rejected():
    with pytest.raises(ValueError, match="non-positive propagation interval"):
        scorer._tier2_dynamics_residuals(np.zeros((2, 8)))


def test_integration_failure_is_not_accepted(monkeypatch, independent_spiral):
    class FailedSolution:
        success = False

    monkeypatch.setattr(scorer, "solve_ivp", lambda *args, **kwargs: FailedSolution())
    initial = np.concatenate(([0.0], independent_spiral.y[:, 0]))
    with pytest.raises(ValueError, match="propagation failed"):
        scorer._propagate_tier2(initial, 100.0)
