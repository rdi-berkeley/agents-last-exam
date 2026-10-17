"""Compare the declared experiment's detector-error distributions."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import stim


def circuit_signature(text: str) -> dict:
    circuit = stim.Circuit(text)
    dimensions = {
        "qubits": circuit.num_qubits,
        "measurements": circuit.num_measurements,
        "detectors": circuit.num_detectors,
        "observables": circuit.num_observables,
    }
    if not all(0 < value <= 200_000 for value in dimensions.values()):
        raise ValueError("Circuit dimensions outside the declared experiment")
    effects = defaultdict(list)
    for instruction in circuit.detector_error_model(decompose_errors=False).flattened():
        if instruction.type == "error":
            targets = tuple(sorted(str(target) for target in instruction.targets_copy()))
            effects[targets].append(instruction.args_copy()[0])
    terms = []
    for targets, probabilities in sorted(effects.items()):
        parity = 1.0
        for probability in sorted(probabilities):
            parity *= 1 - 2 * probability
        terms.append([targets, format((1 - parity) / 2, ".12g")])
    digest = hashlib.sha256(json.dumps(terms, separators=(",", ":")).encode()).hexdigest()
    return {**dimensions, "detector_error_distribution_sha256": digest}
