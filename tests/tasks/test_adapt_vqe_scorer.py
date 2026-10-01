from __future__ import annotations

import asyncio
import copy
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import minimize

from tasks.physical_sciences.adapt_vqe_molecular_energy.scripts import score_outputs as scorer

DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "task-data-hf/extracted/physical_sciences/adapt_vqe_molecular_energy/base"
)
if not (DATA_ROOT / "reference/results.json").is_file():
    pytest.skip("ADAPT integration tests require the gated task-data reference", allow_module_level=True)
REFERENCE = json.loads((DATA_ROOT / "reference/results.json").read_text())
OBSERVED = {
    "tier1": {
        "molecule": "H2",
        "energy_ha": -1.137306035753322,
        "method": "VQE",
        "n_parameters": 1,
    },
    "tier2": {
        "molecule": "LiH",
        "energy_ha": -7.879460726200755,
        "method": "ADAPT-VQE",
        "adapt_iterations": 1,
        "n_parameters": 1,
        "operator_sequence": ["D(0,1->8,9)"],
    },
    "tier3": {
        "molecule": "BeH2",
        "energy_ha": -15.553995261531787,
        "method": "ADAPT-VQE",
        "adapt_iterations": 1,
        "n_parameters": 1,
        "operator_sequence": ["D(0,1->8,9)"],
    },
}


def independent_problem(filename):
    data = json.loads((DATA_ROOT / "input" / filename).read_text())
    count, electrons = data["n_qubits"], data["n_electrons"]
    basis = list(itertools.combinations(range(count), electrons))
    lookup = {occupied: index for index, occupied in enumerate(basis)}
    hamiltonian = np.zeros((len(basis), len(basis)), complex)
    for column, occupied in enumerate(basis):
        for word, coefficient in data["pauli_terms"].items():
            bits = [int(orbital in occupied) for orbital in range(count)]
            phase = complex(coefficient)
            for orbital, symbol in enumerate(word):
                if symbol == "Z":
                    phase *= (-1) ** bits[orbital]
                elif symbol in ("X", "Y"):
                    if symbol == "Y":
                        phase *= 1j * (-1) ** bits[orbital]
                    bits[orbital] = 1 - bits[orbital]
            target = tuple(index for index, bit in enumerate(bits) if bit)
            if target in lookup:
                hamiltonian[lookup[target], column] += phase
    assert np.max(np.abs(hamiltonian.imag)) < 1e-12
    pool = []
    for occupied in range(electrons):
        for virtual in range(electrons, count):
            pool.append((f"S({occupied}->{virtual})", [(occupied, False), (virtual, True)]))
    for occupied_first, occupied_second in itertools.combinations(range(electrons), 2):
        for virtual_first, virtual_second in itertools.combinations(range(electrons, count), 2):
            pool.append(
                (
                    f"D({occupied_first},{occupied_second}->{virtual_first},{virtual_second})",
                    [
                        (occupied_first, False),
                        (occupied_second, False),
                        (virtual_second, True),
                        (virtual_first, True),
                    ],
                )
            )
    generators = []
    for name, actions in pool:
        excitation = np.zeros_like(hamiltonian.real)
        for column, occupied in enumerate(basis):
            remaining, sign = list(occupied), 1
            for orbital, create in actions:
                if (orbital in remaining) == create:
                    break
                sign *= (-1) ** sum(item < orbital for item in remaining)
                if create:
                    remaining.append(orbital)
                    remaining.sort()
                else:
                    remaining.remove(orbital)
            else:
                excitation[lookup[tuple(remaining)], column] = sign
        generators.append((name, excitation - excitation.T))
    indices = [sum(2 ** (count - 1 - orbital) for orbital in occupied) for occupied in basis]
    return data, hamiltonian.real, generators, indices


def honest_solve(filename):
    data, hamiltonian, pool, indices = independent_problem(filename)
    initial = np.zeros(len(indices))
    initial[0] = 1
    chosen, sequence, parameters, history = [], [], np.empty(0), []
    state = initial.copy()

    def objective(values):
        current = initial.copy()
        jacobian = np.zeros((len(indices), len(values)))
        for index, (theta, generator) in enumerate(zip(values, chosen, strict=True)):
            square = generator @ generator
            rotation = (
                np.eye(len(indices)) + np.sin(theta) * generator + (1 - np.cos(theta)) * square
            )
            derivative = (np.cos(theta) * generator + np.sin(theta) * square) @ current
            current = rotation @ current
            jacobian = rotation @ jacobian
            jacobian[:, index] = derivative
        return float(current @ hamiltonian @ current), 2 * jacobian.T @ hamiltonian @ current

    for iteration in range(64):
        gradients = np.array(
            [2 * (hamiltonian @ state) @ generator @ state for _, generator in pool]
        )
        selected = int(np.argmax(np.abs(gradients)))
        maximum = float(abs(gradients[selected]))
        if maximum < 5e-4:
            break
        name, generator = pool[selected]
        sequence.append(name)
        chosen.append(generator)
        result = minimize(
            objective,
            np.append(parameters, 0.0),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-8, "maxiter": 300},
        )
        parameters = result.x
        assert np.max(np.abs(objective(parameters)[1])) < 1e-5
        state = initial.copy()
        for theta, operator in zip(parameters, chosen, strict=True):
            state = expm(theta * operator) @ state
        history.append(
            {
                "iteration": iteration + 1,
                "selected": name,
                "maximum_pool_gradient": maximum,
                "selected_gradient": float(gradients[selected]),
                "energy": float(state @ hamiltonian @ state),
                "optimizer_success": bool(result.success),
                "joint_gradient_max": float(np.max(np.abs(objective(parameters)[1]))),
            }
        )
    else:
        raise AssertionError("honest ADAPT fixture did not converge")
    payload = {
        "molecule": data["molecule"],
        "energy_ha": float(state @ hamiltonian @ state),
        "method": "VQE" if data["molecule"] == "H2" else "ADAPT-VQE",
        "n_parameters": len(parameters),
        "operator_sequence": sequence,
        "parameters": parameters.tolist(),
    }
    if data["molecule"] != "H2":
        payload["adapt_iterations"] = len(parameters)
    audit = {
        "history": history,
        "final_maximum_pool_gradient": maximum,
        "state_norm": float(state @ state),
        "sector_dimension": len(indices),
        "energy": payload["energy_ha"],
        "sector_exact_energy": float(np.linalg.eigvalsh(hamiltonian)[0]),
    }
    return payload, audit, state, indices


@pytest.fixture(scope="module")
def models():
    return {
        tier: scorer._hamiltonian_matrix(DATA_ROOT / "input" / filename)
        for tier, filename in scorer.HAMILTONIAN_FILES.items()
    }


@pytest.fixture(scope="module")
def honest():
    return {tier: honest_solve(filename) for tier, filename in scorer.HAMILTONIAN_FILES.items()}


@pytest.mark.parametrize("tier", scorer.TIERS)
def test_honest_public_hamiltonian_solution_passes(tier, honest, models):
    payload, audit, sector_state, indices = honest[tier]
    passed, reasons = scorer._score_tier(tier, {tier: payload}, REFERENCE, models[tier])
    assert passed, reasons
    assert abs(audit["sector_exact_energy"] - REFERENCE[tier]["exact_energy_ha"]) < 1e-9
    assert audit["final_maximum_pool_gradient"] < 5e-4
    assert audit["state_norm"] == pytest.approx(1, abs=1e-12)
    full_state = np.zeros(models[tier][0].shape[0], complex)
    full_state[indices] = sector_state
    assert np.vdot(full_state, models[tier][0] @ full_state).real == pytest.approx(
        payload["energy_ha"], abs=1e-11
    )
    replay = np.zeros_like(full_state)
    replay[indices[0]] = 1
    for operator, theta in zip(payload["operator_sequence"], payload["parameters"], strict=True):
        replay = scorer._apply_excitation(replay, models[tier][1], operator, theta)
    np.testing.assert_allclose(replay, full_state, atol=1e-11)


@pytest.mark.parametrize(
    "operator,actions",
    [
        ("S(0->2)", [(2, True), (0, False)]),
        ("S(3->1)", [(1, True), (3, False)]),
        ("D(0,1->2,3)", [(2, True), (3, True), (1, False), (0, False)]),
        ("D(1,0->2,3)", [(2, True), (3, True), (0, False), (1, False)]),
    ],
)
def test_rotations_match_independent_full_jordan_wigner_expm(operator, actions):
    ladders = []
    for orbital in range(4):
        matrix = np.array([[1.0]])
        for position in range(4):
            factor = (
                np.diag([1.0, -1.0])
                if position < orbital
                else np.array([[0.0, 1.0], [0.0, 0.0]])
                if position == orbital
                else np.eye(2)
            )
            matrix = np.kron(matrix, factor)
        ladders.append(matrix)
    product = np.eye(16)
    for orbital, create in actions:
        product = product @ (ladders[orbital].T if create else ladders[orbital])
    generator = product - product.T
    state = np.arange(16) + 1j * np.arange(16)[::-1]
    state /= np.linalg.norm(state)
    for theta in (-1.2, 0.0, 0.37, math.pi / 2):
        np.testing.assert_allclose(
            scorer._apply_excitation(state, 4, operator, theta),
            expm(theta * generator) @ state,
            atol=1e-12,
        )


def test_replay_does_not_reject_better_than_zero_start_stationary_point():
    hamiltonian = np.zeros((16, 16), complex)
    hamiltonian[12, 12], hamiltonian[3, 3] = 1, -1
    assert scorer._reported_ansatz_energy(
        hamiltonian, 4, 2, ["D(0,1->2,3)"], [math.pi / 2]
    ) == pytest.approx(-1)
    assert "minimize" not in Path(scorer.__file__).read_text()
    payload = {
        "molecule": "H2",
        "method": "VQE",
        "energy_ha": -1.0,
        "n_parameters": 1,
        "operator_sequence": ["D(0,1->2,3)"],
        "parameters": [math.pi / 2],
    }
    assert scorer._score_tier(
        "tier1", {"tier1": payload}, {"tier1": {"exact_energy_ha": -1.0}}, (hamiltonian, 4, 2)
    )[0]


def test_consistency_tolerance_does_not_relax_scientific_accuracy():
    hamiltonian = np.eye(16) * (-1 + 2e-4)
    payload = {
        "molecule": "H2",
        "method": "VQE",
        "energy_ha": -1.0,
        "n_parameters": 1,
        "operator_sequence": ["D(0,1->2,3)"],
        "parameters": [0.0],
    }
    passed, reasons = scorer._score_tier(
        "tier1", {"tier1": payload}, {"tier1": {"exact_energy_ha": -1.0}}, (hamiltonian, 4, 2)
    )
    assert not passed and any("replayed energy error" in reason for reason in reasons)


def test_repeated_generators_are_allowed(honest, models):
    payload = copy.deepcopy(honest["tier1"][0])
    payload["operator_sequence"] *= 2
    payload["parameters"] = [value / 2 for value in payload["parameters"]] * 2
    payload["n_parameters"] *= 2
    assert scorer._score_tier("tier1", {"tier1": payload}, REFERENCE, models["tier1"])[0]


@pytest.mark.parametrize("count", [True, 0, 1.5, "1", None, 999])
def test_adapt_iterations_must_match_actual_state(count, honest, models):
    payload = copy.deepcopy(honest["tier2"][0])
    payload["adapt_iterations"] = count
    assert not scorer._score_tier("tier2", {"tier2": payload}, REFERENCE, models["tier2"])[0]


@pytest.mark.parametrize("tier", scorer.TIERS)
def test_observed_diagonalization_with_fabricated_metadata_rejected(tier, models):
    passed, reasons = scorer._score_tier(tier, OBSERVED, REFERENCE, models[tier])
    assert not passed and any("parameters" in reason for reason in reasons)


@pytest.mark.parametrize("tier", ("tier2", "tier3"))
def test_observed_one_operator_even_with_optimal_angle_fails(tier, models):
    payload = copy.deepcopy(OBSERVED)
    hamiltonian, count, electrons = models[tier]
    initial = np.zeros(2**count, complex)
    initial[sum(1 << (count - 1 - orbital) for orbital in range(electrons))] = 1
    excited = scorer._apply_excitation(
        initial, count, payload[tier]["operator_sequence"][0], math.pi / 2
    )
    basis = np.column_stack([initial, excited])
    values, vectors = np.linalg.eigh((basis.conj().T @ hamiltonian @ basis).real)
    payload[tier]["parameters"] = [float(np.arctan2(vectors[1, 0], vectors[0, 0]))]
    assert values[0] - REFERENCE[tier]["exact_energy_ha"] > scorer.TIERS[tier]["threshold_ha"]
    passed, reasons = scorer._score_tier(tier, payload, REFERENCE, models[tier])
    assert not passed and any("replayed" in reason for reason in reasons)


@pytest.mark.parametrize(
    "field,value",
    [
        ("parameters", None),
        ("parameters", []),
        ("parameters", [float("nan")]),
        ("parameters", [float("inf")]),
        ("parameters", [True]),
        ("parameters", ["0.1"]),
        ("parameters", [1e300, 0]),
        ("operator_sequence", ["S(0->0)"]),
        ("operator_sequence", ["S(0->4)"]),
        ("operator_sequence", ["D(0,0->2,3)"]),
        ("operator_sequence", ["S(bad)"]),
        ("operator_sequence", [{}]),
        ("n_parameters", True),
        ("method", []),
        ("energy_ha", float("nan")),
    ],
)
def test_invalid_candidate_fields_are_normal_failures(field, value, honest, models):
    payload = copy.deepcopy(honest["tier1"][0])
    payload[field] = value
    assert not scorer._score_tier("tier1", {"tier1": payload}, REFERENCE, models["tier1"])[0]


def test_reported_parameters_not_reoptimized_and_thresholds_unchanged(honest, models):
    assert [spec["threshold_ha"] for spec in scorer.TIERS.values()] == [1e-4, 1.6e-3, 1.6e-3]
    payload = copy.deepcopy(honest["tier1"][0])
    payload["parameters"] = [0.0] * payload["n_parameters"]
    assert not scorer._score_tier("tier1", {"tier1": payload}, REFERENCE, models["tier1"])[0]


def test_order_and_parameter_sign_are_not_ignored(honest, models):
    payload = copy.deepcopy(honest["tier2"][0])
    correct = scorer._reported_ansatz_energy(
        *models["tier2"], payload["operator_sequence"], payload["parameters"]
    )
    wrong_order = scorer._reported_ansatz_energy(
        *models["tier2"], payload["operator_sequence"][::-1], payload["parameters"]
    )
    wrong_sign = scorer._reported_ansatz_energy(
        *models["tier2"], payload["operator_sequence"], [-value for value in payload["parameters"]]
    )
    assert abs(correct - wrong_order) > 5e-4
    assert abs(correct - wrong_sign) > 5e-4


def test_cli_candidate_boundary_and_controlled_reference_precedence(
    tmp_path, monkeypatch, models, capsys
):
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(REFERENCE))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_outputs.py",
            "--output-dir",
            str(tmp_path),
            "--reference-file",
            str(reference),
            "--input-dir",
            str(DATA_ROOT / "input"),
        ],
    )
    monkeypatch.setattr(
        scorer,
        "_hamiltonian_matrix",
        lambda path: models[
            next(
                tier for tier, filename in scorer.HAMILTONIAN_FILES.items() if filename == path.name
            )
        ],
    )
    for invalid in (b"\xff", b"{", b"null", b"[]", b"9" * 5000):
        (tmp_path / "results.json").write_bytes(invalid)
        assert scorer.main() == 0
        assert json.loads(capsys.readouterr().out)["score"] == 0
    (tmp_path / "results.json").unlink()
    assert scorer.main() == 0
    for invalid_reference in ("{", "null", "{}", '{"tier1":{"exact_energy_ha":"bad"}}'):
        reference.write_text(invalid_reference)
        with pytest.raises((ValueError, json.JSONDecodeError)):
            scorer.main()
    reference.unlink()
    with pytest.raises(FileNotFoundError):
        scorer.main()


@pytest.mark.parametrize(
    "content",
    [
        None,
        "{",
        "{}",
        '{"n_qubits":4,"n_electrons":2,"n_pauli_terms":1,"pauli_terms":{"IIII":true}}',
    ],
)
def test_bad_controlled_hamiltonian_raises(tmp_path, content):
    path = tmp_path / "hamiltonian.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises((FileNotFoundError, ValueError)):
        scorer._hamiltonian_matrix(path)


def test_oversized_operator_index_is_candidate_failure():
    assert scorer._excitation_pairs(4, "S(" + "9" * 5000 + "->2)") == []


def test_actual_cli_honest_positive_and_observed_negative(tmp_path, honest):
    command = [
        sys.executable,
        scorer.__file__,
        "--output-dir",
        str(tmp_path),
        "--reference-file",
        str(DATA_ROOT / "reference/results.json"),
        "--input-dir",
        str(DATA_ROOT / "input"),
    ]
    for payload, expected in (
        ({tier: record[0] for tier, record in honest.items()}, 1.0),
        (OBSERVED, 0.0),
    ):
        (tmp_path / "results.json").write_text(json.dumps(payload))
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["score"] == expected


@pytest.mark.parametrize(
    "response,missing_reference",
    [
        ({"return_code": 1, "stderr": "reference parse failed"}, False),
        ({"return_code": 0, "stdout": "invalid"}, False),
        ({"return_code": 0, "stdout": '{"score":true}'}, False),
        ({}, True),
    ],
)
def test_main_propagates_evaluator_infrastructure(response, missing_reference):
    from tasks.physical_sciences.adapt_vqe_molecular_energy import main

    class Session:
        async def file_exists(self, path):
            return not missing_reference

        async def create_dir(self, path):
            return None

        async def write_file(self, path, contents):
            return None

        async def run_command(self, command, check):
            return response

    session = Session()
    session.interface = session
    config = SimpleNamespace(metadata=main.config.to_metadata())
    with pytest.raises((RuntimeError, FileNotFoundError)):
        asyncio.run(main.evaluate(config, session))
