#!/usr/bin/env python
"""Scorer for physical_sciences/adapt_vqe_molecular_energy."""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


TIERS = {
    "tier1": {
        "molecule": "H2",
        "threshold_ha": 1.0e-4,
        "allowed_methods": {"VQE", "ADAPT-VQE"},
        "require_adapt": False,
    },
    "tier2": {
        "molecule": "LiH",
        "threshold_ha": 1.6e-3,
        "allowed_methods": {"ADAPT-VQE"},
        "require_adapt": True,
    },
    "tier3": {
        "molecule": "BeH2",
        "threshold_ha": 1.6e-3,
        "allowed_methods": {"ADAPT-VQE"},
        "require_adapt": True,
    },
}

SINGLE_RE = re.compile(r"^S\((\d+)->(\d+)\)$")
DOUBLE_RE = re.compile(r"^D\((\d+),(\d+)->(\d+),(\d+)\)$")
PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}
HAMILTONIAN_FILES = {
    "tier1": "h2_hamiltonian.json",
    "tier2": "lih_hamiltonian.json",
    "tier3": "beh2_hamiltonian.json",
}
ANSATZ_TOLERANCE_HA = 5.0e-4


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_number(value: Any) -> bool:
    try:
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        )
    except OverflowError:
        return False


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _hamiltonian_matrix(path: Path) -> tuple[np.ndarray, int, int]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid controlled Hamiltonian: {path}")
    n_qubits = payload.get("n_qubits")
    n_electrons = payload.get("n_electrons")
    if (
        not _is_positive_int(n_qubits)
        or not _is_positive_int(n_electrons)
        or n_electrons >= n_qubits
    ):
        raise ValueError(f"invalid controlled orbital counts: {path}")
    terms = payload.get("pauli_terms")
    if not isinstance(terms, dict) or not terms or payload.get("n_pauli_terms") != len(terms):
        raise ValueError(f"invalid controlled Pauli terms: {path}")
    hamiltonian = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for pauli_string, coefficient in terms.items():
        if (
            len(pauli_string) != n_qubits
            or any(symbol not in PAULI for symbol in pauli_string)
            or not _is_number(coefficient)
        ):
            raise ValueError(f"invalid controlled Pauli term: {path}")
        term = np.array([[1]], dtype=complex)
        for symbol in pauli_string:
            term = np.kron(term, PAULI[symbol])
        hamiltonian += float(coefficient) * term
    return hamiltonian, n_qubits, n_electrons


def _excitation_pairs(n_qubits: int, operator: str) -> list[tuple[int, int, float]]:
    match = SINGLE_RE.fullmatch(operator)
    if match:
        try:
            occupied, virtual = (int(value) for value in match.groups())
        except ValueError:
            return []
        orbitals = [(virtual, True), (occupied, False)]
    else:
        match = DOUBLE_RE.fullmatch(operator)
        if not match:
            return []
        try:
            occupied_first, occupied_second, virtual_first, virtual_second = (
                int(value) for value in match.groups()
            )
        except ValueError:
            return []
        orbitals = [
            (virtual_first, True),
            (virtual_second, True),
            (occupied_second, False),
            (occupied_first, False),
        ]

    if any(orbital >= n_qubits for orbital, _ in orbitals) or len(
        {orbital for orbital, _ in orbitals}
    ) != len(orbitals):
        return []

    bit_positions = [(n_qubits - 1 - orbital, create) for orbital, create in orbitals]
    pairs: list[tuple[int, int, float]] = []
    for source in range(2**n_qubits):
        state = source
        sign = 1.0
        for bit_position, create in reversed(bit_positions):
            occupied = (state >> bit_position) & 1
            if (create and occupied) or ((not create) and not occupied):
                break
            sign *= -1.0 if (state >> (bit_position + 1)).bit_count() % 2 else 1.0
            state = state | (1 << bit_position) if create else state & ~(1 << bit_position)
        else:
            if state != source:
                pairs.append((source, state, sign))
    return pairs


def _apply_excitation(state: np.ndarray, n_qubits: int, operator: str, theta: float) -> np.ndarray:
    result = state.copy()
    seen: set[tuple[int, int]] = set()
    cosine = math.cos(theta)
    sine = math.sin(theta)
    for source, target, sign in _excitation_pairs(n_qubits, operator):
        low, high = sorted((source, target))
        pair = (low, high)
        if pair in seen:
            continue
        seen.add(pair)
        generator_high_low = sign if target == high else -sign
        low_value, high_value = result[low], result[high]
        result[low] = cosine * low_value - generator_high_low * sine * high_value
        result[high] = generator_high_low * sine * low_value + cosine * high_value
    return result


def _reported_ansatz_energy(
    hamiltonian: np.ndarray,
    n_qubits: int,
    n_electrons: int,
    operator_sequence: list[str],
    parameters: list[float],
) -> float:
    hf_index = sum(1 << (n_qubits - 1 - orbital) for orbital in range(n_electrons))

    state = np.zeros(2**n_qubits, dtype=complex)
    state[hf_index] = 1.0
    for operator, theta in zip(operator_sequence, parameters, strict=True):
        state = _apply_excitation(state, n_qubits, operator, float(theta))
    energy = float(np.real(np.vdot(state, hamiltonian @ state)))
    if not math.isfinite(energy) or not np.isclose(np.vdot(state, state), 1.0, atol=1e-10, rtol=0):
        raise RuntimeError("state replay failed numerical integrity check")
    return energy


def _score_tier(
    tier_name: str,
    output_payload: Dict[str, Any],
    reference_payload: Dict[str, Any],
    hamiltonian_data: tuple[np.ndarray, int, int],
) -> Tuple[bool, List[str]]:
    spec = TIERS[tier_name]
    reasons: List[str] = []
    payload = output_payload.get(tier_name)
    if not isinstance(payload, dict):
        return False, [f"{tier_name}: missing object payload"]

    if payload.get("molecule") != spec["molecule"]:
        reasons.append(
            f"{tier_name}: molecule={payload.get('molecule')!r} expected {spec['molecule']!r}"
        )

    energy = payload.get("energy_ha")
    target = reference_payload[tier_name]["exact_energy_ha"]
    if not _is_number(energy):
        reasons.append(f"{tier_name}: energy_ha not numeric ({energy!r})")
    elif abs(float(energy) - float(target)) > spec["threshold_ha"]:
        reasons.append(
            f"{tier_name}: energy error {abs(float(energy) - float(target)):.6e} "
            f"> {spec['threshold_ha']:.6e}"
        )

    method = payload.get("method")
    if not isinstance(method, str) or method not in spec["allowed_methods"]:
        reasons.append(f"{tier_name}: method={method!r} not in {sorted(spec['allowed_methods'])}")

    n_parameters = payload.get("n_parameters")
    if not _is_positive_int(n_parameters):
        reasons.append(f"{tier_name}: n_parameters={n_parameters!r} must be positive int")

    hamiltonian, n_qubits, n_electrons = hamiltonian_data
    operator_sequence = payload.get("operator_sequence")
    if not isinstance(operator_sequence, list) or not operator_sequence:
        reasons.append(f"{tier_name}: operator_sequence must be a non-empty list")
    else:
        for index, operator in enumerate(operator_sequence):
            if not isinstance(operator, str) or not _excitation_pairs(n_qubits, operator):
                reasons.append(f"{tier_name}: operator_sequence[{index}] invalid ({operator!r})")
                break
        if len(operator_sequence) != n_parameters:
            reasons.append(f"{tier_name}: operator_sequence length != n_parameters")

    parameters = payload.get("parameters")
    if (
        not isinstance(parameters, list)
        or not parameters
        or not all(_is_number(value) for value in parameters)
    ):
        reasons.append(f"{tier_name}: parameters must be a non-empty list of finite real numbers")
    elif len(parameters) != n_parameters:
        reasons.append(f"{tier_name}: parameters length != n_parameters")

    if spec["require_adapt"] or method == "ADAPT-VQE":
        adapt_iterations = payload.get("adapt_iterations")
        if not _is_positive_int(adapt_iterations):
            reasons.append(
                f"{tier_name}: adapt_iterations={adapt_iterations!r} must be positive int"
            )

        if (
            isinstance(adapt_iterations, int)
            and isinstance(n_parameters, int)
            and isinstance(operator_sequence, list)
            and len(operator_sequence) != adapt_iterations
        ):
            reasons.append(
                f"{tier_name}: len(operator_sequence)={len(operator_sequence)} "
                f"!= adapt_iterations={adapt_iterations}"
            )
        if (
            isinstance(adapt_iterations, int)
            and isinstance(n_parameters, int)
            and adapt_iterations != n_parameters
        ):
            reasons.append(
                f"{tier_name}: adapt_iterations={adapt_iterations} != n_parameters={n_parameters}"
            )

    if not reasons:
        replayed_energy = _reported_ansatz_energy(
            hamiltonian, n_qubits, n_electrons, operator_sequence, parameters
        )
        if abs(float(energy) - replayed_energy) > ANSATZ_TOLERANCE_HA:
            reasons.append(
                f"{tier_name}: reported energy {float(energy):.10f} differs from "
                f"replayed state energy {replayed_energy:.10f} by more than {ANSATZ_TOLERANCE_HA:.6e} Ha"
            )
        if abs(replayed_energy - float(target)) > spec["threshold_ha"]:
            reasons.append(
                f"{tier_name}: replayed energy error {abs(replayed_energy - float(target)):.6e} "
                f"> {spec['threshold_ha']:.6e}"
            )

    return len(reasons) == 0, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reference_file = Path(args.reference_file)
    input_dir = Path(args.input_dir)
    output_file = output_dir / "results.json"

    reference_payload = _load_json(reference_file)
    if not isinstance(reference_payload, dict):
        raise ValueError("controlled reference is not an object")
    for tier_name in TIERS:
        reference_tier = reference_payload.get(tier_name)
        if not isinstance(reference_tier, dict) or not _is_number(
            reference_tier.get("exact_energy_ha")
        ):
            raise ValueError(f"controlled reference energy invalid: {tier_name}")
    hamiltonians = {
        tier_name: _hamiltonian_matrix(input_dir / filename)
        for tier_name, filename in HAMILTONIAN_FILES.items()
    }

    try:
        output_payload = _load_json(output_file)
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        print(
            json.dumps(
                {"score": 0.0, "passed": {}, "reasons": [f"unparseable results.json: {exc}"]}
            )
        )
        return 0

    if not isinstance(output_payload, dict):
        print(
            json.dumps({"score": 0.0, "passed": {}, "reasons": ["results.json is not an object"]})
        )
        return 0

    passed: Dict[str, bool] = {}
    reasons: List[str] = []
    for tier_name in ("tier1", "tier2", "tier3"):
        ok, tier_reasons = _score_tier(
            tier_name, output_payload, reference_payload, hamiltonians[tier_name]
        )
        passed[tier_name] = ok
        reasons.extend(tier_reasons)

    score = sum(1 for ok in passed.values() if ok) / 3.0
    print(json.dumps({"score": score, "passed": passed, "reasons": reasons}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
