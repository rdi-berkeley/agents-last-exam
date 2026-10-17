import asyncio
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tasks.engineering.aerospace_low_thrust_trajectory.scripts import mee_dynamics as model
from tasks.engineering.aerospace_low_thrust_trajectory.scripts import tier3_contract as contract
from tasks.engineering.aerospace_low_thrust_trajectory.scripts import cartesian_check


def independent_cartesian_rates(time, state, control, j2=model.J2):
    position, velocity, mass = state[:3], state[3:6], state[6]
    radius = np.linalg.norm(position)
    vertical = position[2] / radius
    oblateness = 1.5 * j2 * model.MU * model.RE**2 / radius**5
    perturbation = (
        oblateness
        * position
        * np.array([5 * vertical**2 - 1, 5 * vertical**2 - 1, 5 * vertical**2 - 3])
    )
    radial = position / radius
    normal = np.cross(position, velocity)
    normal /= np.linalg.norm(normal)
    transverse = np.cross(normal, radial)
    thrust = model.THRUST / mass * np.column_stack((radial, transverse, normal)) @ control
    return np.r_[
        velocity,
        -model.MU * position / radius**3 + perturbation + thrust,
        -model.THRUST * np.linalg.norm(control) / model.VE,
    ]


@pytest.mark.parametrize("control", [[0, 0, 0], [0.2, 0.6, -0.3], [0, 0, 1]])
@pytest.mark.parametrize("j2", [0.0, model.J2])
def test_mee_rates_match_independent_cartesian_j2(control, j2):
    state = np.array([9e6, 0.12, -0.08, 0.19, -0.13, 1.7, 1700.0])
    derivative = model.rates(state, np.array(control), j2=j2)
    shifted = state.astype(complex) + 1e-25j * derivative
    position, velocity = model.cartesian(state)
    position_rate, velocity_rate = model.cartesian(shifted)
    actual = np.r_[position_rate.imag / 1e-25, velocity_rate.imag / 1e-25, derivative[6]]
    expected = independent_cartesian_rates(0, np.r_[position, velocity, state[6]], control, j2)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=2e-12)


@pytest.mark.parametrize("control", [[0, 0, 0], [0.15, 0.7, -0.45]])
def test_multi_revolution_propagation_agrees_with_cartesian(control):
    initial = model.INITIAL.copy()
    position, velocity = model.cartesian(initial)
    times = np.linspace(0, 30000, 151)
    mee = solve_ivp(
        lambda time, state: model.rates(state, control),
        [0, times[-1]],
        initial,
        method="DOP853",
        rtol=2e-12,
        atol=1e-12,
        t_eval=times,
    )
    cart = solve_ivp(
        lambda time, state: independent_cartesian_rates(time, state, control),
        [0, times[-1]],
        np.r_[position, velocity, initial[6]],
        method="RK45",
        rtol=2e-12,
        atol=1e-12,
        t_eval=times,
    )
    assert mee.success and cart.success
    positions, velocities = model.cartesian(mee.y.T)
    np.testing.assert_allclose(positions, cart.y[:3].T, atol=0.02, rtol=0)
    np.testing.assert_allclose(velocities, cart.y[3:6].T, atol=2e-5, rtol=0)


def test_longitude_normal_term_is_required():
    state = np.array([9e6, 0.12, -0.08, 0.19, -0.13, 1.7, 1700.0])
    _, _, kepler_rate = model.geometry(state)
    assert abs(model.rates(state, [0, 0, 1])[5] - kepler_rate) > 1e-9


def test_adjoint_is_hamiltonian_gradient_and_longitude_is_not_constant():
    state = model.INITIAL / model.SCALE
    state[5] = 0.7
    costate = np.array([-0.12, 0.08, -0.06, 0.3, 0.04, 0.001, -0.85])
    control = np.array([0.2, 0.5, -0.3])
    _, adjoint = model.canonical_rates(state, costate, control)
    for column in range(7):
        offset = np.zeros(7)
        offset[column] = 1e-6
        difference = (
            model.normalized_rates(state + offset, control)
            - model.normalized_rates(state - offset, control)
        ) / 2e-6
        assert adjoint[column] == pytest.approx(-costate @ difference, abs=1e-6, rel=1e-6)
    assert abs(adjoint[5]) > 1e-3


def test_primer_control_minimizes_hamiltonian_including_coast():
    generator = np.random.default_rng(293)
    state = model.INITIAL / model.SCALE
    for costate in generator.normal(size=(15, 7)):
        control, _ = model.minimizing_control(state, costate)
        best = costate @ model.normalized_rates(state, control)
        alternatives = generator.normal(size=(500, 3))
        alternatives /= np.linalg.norm(alternatives, axis=1)[:, None]
        alternatives *= generator.uniform(size=(500, 1))
        alternatives[0] = 0
        objective = model.normalized_rates(np.broadcast_to(state, (500, 7)), alternatives) @ costate
        assert np.min(objective) >= best - 1e-10


def test_throttle_changes_fuel_objective():
    state = model.INITIAL.copy()
    coast = model.rates(state, [0, 0, 0])
    half = model.rates(state, [0, 0.5, 0])
    full = model.rates(state, [0, 1, 0])
    assert coast[6] == 0
    assert half[6] == full[6] / 2
    assert full[6] < 0


def test_burning_adjoint_propagation_with_independent_finite_difference_jacobian():
    initial = model.INITIAL / model.SCALE
    costate = np.array([-0.12, 0.08, -0.06, 0.3, 0.04, 0.001, -0.85])
    control = np.array([0.2, 0.7, -0.3])
    times = np.linspace(0, 2000.0, 1001)

    def derivative(time, values):
        state, adjoint = values[:7], values[7:]
        physical_rate = model.normalized_rates(state, control)
        jacobian = np.empty((7, 7))
        for column in range(7):
            offset = np.zeros(7)
            offset[column] = 1e-6
            jacobian[:, column] = (
                model.normalized_rates(state + offset, control)
                - model.normalized_rates(state - offset, control)
            ) / 2e-6
        return np.r_[physical_rate, -jacobian.T @ adjoint] / model.TF

    solution = solve_ivp(
        derivative,
        [0, times[-1]],
        np.r_[initial, costate],
        t_eval=times,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )
    assert solution.success
    trajectory = np.column_stack(
        (times, solution.y[:7].T * model.SCALE, solution.y[7:].T * model.M0 / model.SCALE)
    )
    controls = np.column_stack((times, np.broadcast_to(control, (len(times), 3))))
    defects, errors = contract.integrated_defects(trajectory, controls)
    assert np.max(defects) < 1e-6
    assert np.max(errors) < 1e-8
    wrong_control = controls.copy()
    wrong_control[:, 1:] *= -1
    wrong_defects, _ = contract.integrated_defects(trajectory, wrong_control)
    assert np.max(wrong_defects[:7]) > contract.DEFECT_LIMIT


@pytest.fixture
def independent_coast_certificate(monkeypatch):
    duration = 12000.0
    times = np.linspace(0, duration, 1501)
    position, velocity = model.cartesian(model.INITIAL)
    solution = solve_ivp(
        lambda time, state: independent_cartesian_rates(time, state, [0, 0, 0]),
        [0, duration],
        np.r_[position, velocity, model.M0],
        method="RK45",
        rtol=3e-13,
        atol=1e-12,
        t_eval=times,
    )
    assert solution.success
    position, velocity = solution.y[:3].T, solution.y[3:6].T
    momentum = np.cross(position, velocity)
    magnitude = np.linalg.norm(momentum, axis=1)
    normal = momentum / magnitude[:, None]
    inc_cos = -normal[:, 1] / (1 + normal[:, 2])
    inc_sin = normal[:, 0] / (1 + normal[:, 2])
    squared = 1 + inc_cos**2 + inc_sin**2
    first = (
        np.column_stack((1 + inc_cos**2 - inc_sin**2, 2 * inc_cos * inc_sin, -2 * inc_sin))
        / squared[:, None]
    )
    second = (
        np.column_stack((2 * inc_cos * inc_sin, 1 - inc_cos**2 + inc_sin**2, 2 * inc_cos))
        / squared[:, None]
    )
    eccentricity = (
        np.cross(velocity, momentum) / model.MU
        - position / np.linalg.norm(position, axis=1)[:, None]
    )
    longitude = np.unwrap(
        np.arctan2(np.sum(position * second, axis=1), np.sum(position * first, axis=1))
    )
    state = np.column_stack(
        (
            magnitude**2 / model.MU,
            np.sum(eccentricity * first, axis=1),
            np.sum(eccentricity * second, axis=1),
            inc_cos,
            inc_sin,
            longitude,
            np.full(len(times), model.M0),
        )
    )
    trajectory = np.column_stack((times, state, np.zeros((len(times), 7))))
    trajectory[:, -1] = -1
    control = np.column_stack((times, np.zeros((len(times), 3))))
    monkeypatch.setattr(model, "TF", duration)
    monkeypatch.setattr(model, "TARGET", state[-1, :5] / model.SCALE[:5])
    final = state[-1]
    ecc = np.hypot(final[1], final[2])
    result = {
        "tier3": {
            "formulation": contract.VERSION,
            "shooting_converged": True,
            "final_sma_km": final[0] / (1 - ecc**2) / 1000,
            "final_eccentricity": ecc,
            "final_inclination_deg": np.rad2deg(2 * np.arctan(np.hypot(final[3], final[4]))),
            "final_mass_kg": model.M0,
            "fuel_consumed_kg": 0.0,
            "dv_total_m_s": 0.0,
            "transfer_time_days": duration / 86400,
            "initial_costates": [0.0] * 6 + [-1.0],
            "constraint_violation_norm": 0.0,
            "hamiltonian_initial": 0.0,
            "hamiltonian_final": 0.0,
        }
    }
    return result, trajectory, control


def test_complete_checker_accepts_independent_j2_coast_optimum(independent_coast_certificate):
    results, trajectory, control = independent_coast_certificate
    assert contract.check(results, results, trajectory, control) == []


def test_global_cartesian_replay_with_independent_physical_coast(independent_coast_certificate):
    _, trajectory, control = independent_coast_certificate
    replay = cartesian_check.replay(trajectory, control)
    assert replay["success"]
    assert replay["terminal_orbit_norm"] < 1e-8
    assert replay["position_discrepancy"] < 1e-7
    control[:, 2] = 1
    wrong = cartesian_check.replay(trajectory, control)
    assert wrong["success"]
    assert wrong["terminal_orbit_norm"] > 1e-4


def test_global_replay_errors_use_actual_radius_and_speed(independent_coast_certificate):
    _, trajectory, control = independent_coast_certificate
    trajectory[777, 1] *= 1.02
    result = cartesian_check.replay(trajectory, control)
    assert result["position_discrepancy"] == pytest.approx(0.02, abs=1e-7)
    assert result["velocity_discrepancy"] == pytest.approx(1 - 1 / np.sqrt(1.02), abs=1e-7)


def test_global_replay_resolves_narrow_throttle_pulses(independent_coast_certificate):
    _, trajectory, control = independent_coast_certificate
    control[[4, 600, 1494], 2] = 1
    result = cartesian_check.replay(trajectory, control)
    exact_fuel = 3 * (trajectory[1, 0] - trajectory[0, 0]) * model.THRUST / model.VE
    assert result["success"]
    assert result["mass_discrepancy_normalized"] == pytest.approx(exact_fuel / model.M0, abs=1e-12)


@pytest.mark.parametrize("control", [[0, 0, 0], [0.2, 0.6, -0.3], [0, 0, 1]])
def test_global_cartesian_rhs_matches_independent_equations(control):
    state = np.array([9e6, 0.12, -0.08, 0.19, -0.13, 1.7, 1700.0])
    position, velocity = model.cartesian(state)
    cartesian = np.r_[position, velocity, state[6]]
    constants = (model.MU, 1.5 * model.J2 * model.MU * model.RE**2, model.THRUST, model.VE)
    actual = cartesian_check.inertial_rates(
        0.5, cartesian, np.array([0.0, 1.0]), np.array([control, control]), constants
    )
    np.testing.assert_allclose(
        actual, independent_cartesian_rates(0.5, cartesian, control), rtol=1e-12, atol=1e-10
    )


def test_v2_reference_cannot_relabel_legacy_arrays(independent_coast_certificate):
    from tasks.engineering.aerospace_low_thrust_trajectory.scripts import score_outputs as scorer

    results, trajectory, control = independent_coast_certificate
    with pytest.raises(contract.EvaluatorReferenceError, match="15-column"):
        scorer._check_tier3(results, results, trajectory, control, trajectory[:, :14], control)


@pytest.mark.parametrize(
    "column,amount,expected",
    [
        (1, 10000.0, "state"),
        (6, 0.001, "state"),
        (7, -0.2, "mass"),
        (8, 1e-5, "adjoint"),
        (13, 1.0, "adjoint"),
    ],
)
def test_complete_checker_rejects_interior_fabrication(
    independent_coast_certificate, column, amount, expected
):
    results, trajectory, control = independent_coast_certificate
    trajectory[777, column] += amount
    failures = contract.check(results, results, trajectory, control)
    assert any(expected in failure for failure in failures), failures


def test_complete_checker_rejects_thrust_without_fuel(independent_coast_certificate):
    results, trajectory, control = independent_coast_certificate
    control[:, 2] = 1.0
    failures = contract.check(results, results, trajectory, control)
    assert any("Pontryagin" in failure for failure in failures)
    assert any("mass-flow" in failure for failure in failures)


@pytest.mark.parametrize("bad", ["duplicate", "nan", "legacy", "transversality", "claim"])
def test_complete_checker_rejects_false_certificates(independent_coast_certificate, bad):
    results, trajectory, control = independent_coast_certificate
    reference = copy.deepcopy(results)
    if bad == "duplicate":
        trajectory[500, 0] = trajectory[499, 0]
    elif bad == "nan":
        trajectory[500, 9] = np.nan
    elif bad == "legacy":
        results["tier3"]["formulation"] = "orbit_averaged_velocity"
    elif bad == "transversality":
        trajectory[-1, -1] = 0
    else:
        results["tier3"]["hamiltonian_initial"] = 1.0
    assert contract.check(results, reference, trajectory, control)


def test_main_entrypoint_with_independently_computed_three_tier_fixture(
    independent_coast_certificate, monkeypatch
):
    from tasks.engineering.aerospace_low_thrust_trajectory import main
    from tasks.engineering.aerospace_low_thrust_trajectory.scripts import score_outputs as scorer

    results, tier3, control = independent_coast_certificate
    position, velocity = model.cartesian(model.INITIAL)
    initial = np.r_[position, velocity, model.M0]

    def derivative(time, state):
        position, velocity, mass = state[:3], state[3:6], state[6]
        acceleration = -model.MU * position / np.linalg.norm(
            position
        ) ** 3 + model.THRUST * velocity / (mass * np.linalg.norm(velocity))
        return np.r_[velocity, acceleration, -model.THRUST / model.VE]

    solution = solve_ivp(
        derivative,
        [0, model.TF],
        initial,
        t_eval=tier3[:, 0],
        method="RK45",
        rtol=3e-13,
        atol=1e-12,
    )
    assert solution.success
    tier2 = np.column_stack((solution.t, solution.y.T))
    position, velocity, mass = solution.y[:3, -1], solution.y[3:6, -1], solution.y[6, -1]
    momentum = np.cross(position, velocity)
    target = -model.MU / (np.dot(velocity, velocity) - 2 * model.MU / np.linalg.norm(position))
    ecc = np.linalg.norm(
        np.cross(velocity, momentum) / model.MU - position / np.linalg.norm(position)
    )
    monkeypatch.setattr(scorer, "A_GEO_KM", target / 1000)
    initial_radius = model.INITIAL[0]
    initial_speed, final_speed = np.sqrt(model.MU / initial_radius), np.sqrt(model.MU / target)
    transfer_axis = (initial_radius + target) / 2
    first = initial_speed * (np.sqrt(2 * target / (initial_radius + target)) - 1)
    second = final_speed * (1 - np.sqrt(2 * initial_radius / (initial_radius + target)))
    transfer_time = np.pi * np.sqrt(transfer_axis**3 / model.MU)
    results["tier1"] = dict(
        transfer_orbit_sma_km=transfer_axis / 1000,
        dv1_m_s=first,
        dv2_m_s=second,
        dv_total_m_s=first + second,
        transfer_time_s=transfer_time,
        transfer_time_hours=transfer_time / 3600,
        v_circular_leo_m_s=initial_speed,
        v_circular_geo_m_s=final_speed,
    )
    results["tier2"] = dict(
        final_sma_km=target / 1000,
        final_eccentricity=ecc,
        final_inclination_deg=28.5,
        dv_total_m_s=model.VE * np.log(model.M0 / mass),
        transfer_time_s=model.TF,
        final_mass_kg=mass,
        fuel_consumed_kg=model.M0 - mass,
        edelbaum_dv_m_s=initial_speed - final_speed,
    )
    files = {"results.json": json.dumps(results).encode()}
    for name, values in [
        ("tier2_trajectory.npy", tier2),
        ("tier3_trajectory.npy", tier3),
        ("tier3_control.npy", control),
    ]:
        buffer = io.BytesIO()
        np.save(buffer, values)
        files[name] = buffer.getvalue()

    reference_files = files.copy()

    class Session:
        async def file_exists(self, name):
            return name.removeprefix("reference/") in (reference_files if name.startswith("reference/") else files)

        async def directory_exists(self, name):
            return False

        async def read_bytes(self, name):
            if name.startswith("reference/"):
                return reference_files[name.removeprefix("reference/")]
            return files[name]

    paths = {name: name for name in files}
    configuration = SimpleNamespace(metadata={"candidate_files": paths, "reference_files": {name: "reference/" + name for name in files}})
    assert scorer.score_submission(files, files).failures == ()
    assert asyncio.run(main.evaluate(configuration, Session())) == [1.0]
    files["tier3_trajectory.npy"] = b"not a NumPy file"
    assert asyncio.run(main.evaluate(configuration, Session())) == [0.0]
