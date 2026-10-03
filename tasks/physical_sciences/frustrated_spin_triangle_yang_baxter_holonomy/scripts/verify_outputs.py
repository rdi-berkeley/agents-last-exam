"""Forward verifier for the frustrated spin-triangle pulse-synthesis task."""

from __future__ import annotations

import argparse
import cmath
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MATRIX_DIM = 8
MIN_PULSES = 2
MAX_PULSES = 96
TOLERANCE = 1e-8
MAX_DRIVE = 0.25
PULSE_DURATION = 0.25
MIN_TRIANGLE_BOND_ACTION = 0.05
PAULIS = (
    [[1.0 + 0j, 0j], [0j, 1.0 + 0j]],
    [[0j, 1.0 + 0j], [1.0 + 0j, 0j]],
    [[0j, -1j], [1j, 0j]],
    [[1.0 + 0j, 0j], [0j, -1.0 + 0j]],
)


@dataclass
class ScoreResult:
    score: float
    correct: int
    total: int
    results: dict[str, dict[str, Any]]
    diagnostic_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def mat_zero(n: int = MATRIX_DIM) -> list[list[complex]]:
    return [[0j for _ in range(n)] for _ in range(n)]


def mat_eye(n: int = MATRIX_DIM) -> list[list[complex]]:
    result = mat_zero(n)
    for index in range(n):
        result[index][index] = 1.0 + 0j
    return result


def mat_add(
    left: list[list[complex]],
    right: list[list[complex]],
    left_scale: complex = 1.0 + 0j,
    right_scale: complex = 1.0 + 0j,
) -> list[list[complex]]:
    n = len(left)
    return [
        [left_scale * left[row][column] + right_scale * right[row][column] for column in range(n)]
        for row in range(n)
    ]


def mat_mul(left: list[list[complex]], right: list[list[complex]]) -> list[list[complex]]:
    n = len(left)
    result = mat_zero(n)
    for row in range(n):
        for inner in range(n):
            value = left[row][inner]
            if value == 0j:
                continue
            for column in range(n):
                result[row][column] += value * right[inner][column]
    return result


def mat_dag(matrix: list[list[complex]]) -> list[list[complex]]:
    n = len(matrix)
    return [[matrix[column][row].conjugate() for column in range(n)] for row in range(n)]


def mat_trace(matrix: list[list[complex]]) -> complex:
    return sum(matrix[index][index] for index in range(len(matrix)))


def mat_frob_norm(matrix: list[list[complex]]) -> float:
    return math.sqrt(
        sum(value.real * value.real + value.imag * value.imag for row in matrix for value in row)
    )


def mat_comm(left: list[list[complex]], right: list[list[complex]]) -> list[list[complex]]:
    return mat_add(mat_mul(left, right), mat_mul(right, left), 1.0, -1.0)


def kron2(left: list[list[complex]], right: list[list[complex]]) -> list[list[complex]]:
    left_dim = len(left)
    right_dim = len(right)
    result = mat_zero(left_dim * right_dim)
    for left_row in range(left_dim):
        for left_column in range(left_dim):
            for right_row in range(right_dim):
                for right_column in range(right_dim):
                    result[left_row * right_dim + right_row][
                        left_column * right_dim + right_column
                    ] = left[left_row][left_column] * right[right_row][right_column]
    return result


def kron3(
    first: list[list[complex]],
    second: list[list[complex]],
    third: list[list[complex]],
) -> list[list[complex]]:
    return kron2(kron2(first, second), third)


def u3(theta: float, phi: float, lam: float) -> list[list[complex]]:
    cosine = math.cos(theta / 2.0)
    sine = math.sin(theta / 2.0)
    exp_phi = cmath.exp(1j * phi)
    exp_lam = cmath.exp(1j * lam)
    exp_sum = cmath.exp(1j * (phi + lam))
    return [
        [cosine + 0j, -exp_lam * sine],
        [exp_phi * sine, exp_sum * cosine],
    ]


def expm_hermitian_8x8(hamiltonian: list[list[complex]], duration: float) -> list[list[complex]]:
    """Compute exp(-i H t) with scaling, an 18-term Taylor series, and squaring."""
    max_column_sum = max(
        sum(abs(hamiltonian[row][column]) for row in range(MATRIX_DIM)) * abs(duration)
        for column in range(MATRIX_DIM)
    )
    squarings = 0
    if max_column_sum > 0.125:
        squarings = max(0, math.ceil(math.log2(max_column_sum / 0.125)))
    scale = 1.0 / (2**squarings)

    scaled = [
        [-1j * duration * scale * hamiltonian[row][column] for column in range(MATRIX_DIM)]
        for row in range(MATRIX_DIM)
    ]
    result = mat_eye()
    term = mat_eye()
    for order in range(1, 19):
        term = mat_mul(term, scaled)
        inverse_order = 1.0 / float(order)
        for row in range(MATRIX_DIM):
            for column in range(MATRIX_DIM):
                term[row][column] *= inverse_order
                result[row][column] += term[row][column]

    for _ in range(squarings):
        result = mat_mul(result, result)
    return result


def decode_matrix(raw_matrix: Any) -> list[list[complex]]:
    if not isinstance(raw_matrix, list) or len(raw_matrix) != MATRIX_DIM:
        raise ValueError("matrix must have 8 rows")
    decoded: list[list[complex]] = []
    for raw_row in raw_matrix:
        if not isinstance(raw_row, list) or len(raw_row) != MATRIX_DIM:
            raise ValueError("matrix rows must have 8 entries")
        row: list[complex] = []
        for raw_value in raw_row:
            if not isinstance(raw_value, list) or len(raw_value) != 2:
                raise ValueError("matrix entries must be [real, imag] pairs")
            real = float(raw_value[0])
            imaginary = float(raw_value[1])
            if not math.isfinite(real) or not math.isfinite(imaginary):
                raise ValueError("matrix entries must be finite")
            row.append(complex(real, imaginary))
        decoded.append(row)
    return decoded


def _decode_pulses(pulses: Any) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not isinstance(pulses, list) or not MIN_PULSES <= len(pulses) <= MAX_PULSES:
        count = len(pulses) if isinstance(pulses, list) else 0
        return None, f"invalid pulse count: {count}"

    decoded: list[dict[str, Any]] = []
    for index, pulse in enumerate(pulses):
        if not isinstance(pulse, dict):
            return None, f"pulse {index} must be an object"
        frame_codes = pulse.get("frame_codes")
        bond_drives = pulse.get("bond_drives")
        if (
            not isinstance(frame_codes, list)
            or len(frame_codes) != 3
            or any(type(value) is not int or value not in range(4) for value in frame_codes)
        ):
            return None, f"pulse {index} frame_codes must contain 3 integers in [0, 3]"
        if not isinstance(bond_drives, list) or len(bond_drives) != 3:
            return None, f"pulse {index} bond_drives must contain 3 numbers"
        try:
            drive_values = [float(value) for value in bond_drives]
            duration = float(pulse["duration"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return None, f"pulse {index} is malformed: {exc}"
        all_values = drive_values + [duration]
        if not all(math.isfinite(value) for value in all_values):
            return None, f"pulse {index} contains a non-finite number"
        if duration != PULSE_DURATION:
            return None, f"pulse {index} duration must equal {PULSE_DURATION}"
        if any(abs(value) > MAX_DRIVE for value in drive_values):
            return None, f"pulse {index} bond drive is out of bounds"
        if sum(value != 0.0 for value in drive_values) > 1:
            return None, f"pulse {index} may drive at most one bond"
        decoded.append(
            {
                "frame_codes": frame_codes,
                "bond_drives": drive_values,
                "duration": duration,
            }
        )
    return decoded, None


def noise_coefficients(
    sequence: list[list[list[list[complex]]]],
) -> tuple[list[list[list[complex]]], dict[tuple[int, int], list[list[complex]]]]:
    """Return coefficients of the linear and quadratic noise-amplitude polynomials."""
    count = len(sequence[0])
    first = [mat_zero() for _ in range(count)]
    second = {(a, b): mat_zero() for a in range(count) for b in range(a, count)}
    for current in sequence:
        for (a, b), accumulated in second.items():
            delta = mat_comm(current[a], first[b])
            if a != b:
                delta = mat_add(delta, mat_comm(current[b], first[a]))
            second[a, b] = mat_add(accumulated, delta, 1.0, 0.5)
        first = [mat_add(previous, value) for previous, value in zip(first, current)]
    return first, second


def evaluate_scenario(scenario: dict[str, Any], submission: Any) -> tuple[bool, float, str]:
    if not isinstance(submission, dict) or "pulses" not in submission:
        return False, 0.0, "missing pulses list"
    pulses, pulse_error = _decode_pulses(submission["pulses"])
    if pulse_error is not None or pulses is None:
        return False, 0.0, pulse_error or "invalid pulses"

    amplitudes = submission.get("bond_amplitudes")
    if not isinstance(amplitudes, list) or len(amplitudes) != 3:
        return False, 0.0, "bond_amplitudes must contain 3 shared magnitudes"
    try:
        amplitudes = [float(value) for value in amplitudes]
    except (TypeError, ValueError, OverflowError):
        return False, 0.0, "invalid bond_amplitudes"
    if not all(math.isfinite(value) and 0.0 <= value <= MAX_DRIVE for value in amplitudes):
        return False, 0.0, "bond_amplitudes must be finite and within [0, 0.25]"
    for index, pulse in enumerate(pulses):
        for bond, value in enumerate(pulse["bond_drives"]):
            if value != 0.0 and abs(value) != amplitudes[bond]:
                return False, 0.0, f"pulse {index} must use the shared bond_amplitudes"

    calibration = submission.get("frame_calibration")
    if not isinstance(calibration, list) or len(calibration) != 9:
        return False, 0.0, "frame_calibration must contain 9 finite angles"
    try:
        calibration = [float(value) for value in calibration]
    except (TypeError, ValueError, OverflowError):
        return False, 0.0, "invalid frame_calibration"
    if not all(math.isfinite(value) and abs(value) <= 1e6 for value in calibration):
        return False, 0.0, "frame_calibration must be finite and within [-1e6, 1e6]"
    site_calibrations = [u3(*calibration[index : index + 3]) for index in range(0, 9, 3)]

    try:
        h_drift = decode_matrix(scenario["H_drift"])
        h_ex_12 = decode_matrix(scenario["H_ex_12"])
        h_ex_23 = decode_matrix(scenario["H_ex_23"])
        h_ex_31 = decode_matrix(scenario["H_ex_31"])
        h_leak_12 = decode_matrix(scenario["H_leak_12"])
        h_leak_23 = decode_matrix(scenario["H_leak_23"])
        h_leak_31 = decode_matrix(scenario["H_leak_31"])
        noise = decode_matrix(scenario["E_noise"])
        target = decode_matrix(scenario["U_target"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, 0.0, f"invalid scenario data: {exc}"

    for bond_index, bond_name in enumerate(("12", "23", "31")):
        action = sum(abs(pulse["bond_drives"][bond_index]) * pulse["duration"] for pulse in pulses)
        if action < MIN_TRIANGLE_BOND_ACTION:
            return (
                False,
                0.0,
                f"bond {bond_name} exchange action {action:.4f} < {MIN_TRIANGLE_BOND_ACTION:.2f}",
            )

    total_unitary = mat_eye()
    noise_channels = [noise]
    for site in range(3):
        for pauli in PAULIS[1:]:
            factors = [PAULIS[0], PAULIS[0], PAULIS[0]]
            factors[site] = pauli
            noise_channels.append(kron3(*factors))
    noise_sequence = []
    frame_cache = {}
    transverse_leakage = [mat_zero() for _ in range(3)]

    for pulse in pulses:
        frame_codes = tuple(pulse["frame_codes"])
        drives = pulse["bond_drives"]
        duration = pulse["duration"]
        if frame_codes not in frame_cache:
            frame = kron3(
                *(mat_mul(site, PAULIS[code]) for site, code in zip(site_calibrations, frame_codes))
            )
            transformed_noise = [
                [
                    [duration * value for value in row]
                    for row in mat_mul(mat_dag(frame), mat_mul(channel, frame))
                ]
                for channel in noise_channels
            ]
            frame_cache[frame_codes] = (frame, transformed_noise)
        local_frame, transformed_noise = frame_cache[frame_codes]
        noise_sequence.append(transformed_noise)
        local_frame_dag = mat_dag(local_frame)

        exchange = mat_add(
            mat_add(h_ex_12, h_ex_23, drives[0], drives[1]),
            h_ex_31,
            1.0,
            drives[2],
        )
        raw_hamiltonian = mat_add(h_drift, exchange)
        dressed_hamiltonian = mat_mul(local_frame_dag, mat_mul(raw_hamiltonian, local_frame))

        for bond, leakage in enumerate((h_leak_12, h_leak_23, h_leak_31)):
            if drives[bond] != 0.0:
                dressed_leakage = mat_mul(local_frame_dag, mat_mul(leakage, local_frame))
                transverse_leakage[bond] = mat_add(
                    transverse_leakage[bond], dressed_leakage, 1.0, duration * drives[bond]
                )

        segment_unitary = expm_hermitian_8x8(dressed_hamiltonian, duration)
        total_unitary = mat_mul(segment_unitary, total_unitary)

    omega_1, omega_2 = noise_coefficients(noise_sequence)
    overlap = mat_trace(mat_mul(mat_dag(target), total_unitary))
    fidelity = (overlap.real**2 + overlap.imag**2) / float(MATRIX_DIM**2)
    metrics = {
        "infidelity": abs(1.0 - fidelity),
        "omega_1": max(mat_frob_norm(value) for value in omega_1),
        "omega_2": max(mat_frob_norm(value) for value in omega_2.values()),
        "leakage": max(mat_frob_norm(value) for value in transverse_leakage),
    }
    criteria = (
        ("omega_1", 0.06, "||Omega_1||"),
        ("omega_2", 0.06, "||Omega_2||"),
        ("leakage", 0.08, "||Leak||"),
        ("infidelity", 0.80, "1-F"),
    )
    scenario_score = sum(weight for key, weight, _ in criteria if metrics[key] <= TOLERANCE)
    failures = [
        f"{label}={metrics[key]:.2e}>{TOLERANCE:.1e}"
        for key, _, label in criteria
        if metrics[key] > TOLERANCE
    ]
    if failures:
        return (
            False,
            round(scenario_score, 4),
            (f"Incomplete (weighted diagnostic {scenario_score:.2f}): " + ", ".join(failures)),
        )
    return (
        True,
        1.0,
        (
            "Verified: "
            f"1-F={metrics['infidelity']:.2e}, "
            f"||Omega_1||={metrics['omega_1']:.2e}, "
            f"||Omega_2||={metrics['omega_2']:.2e}, "
            f"||Leak||={metrics['leakage']:.2e}"
        ),
    )


def score_submission(scenarios: Any, submission: Any) -> ScoreResult:
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("scenarios must be a non-empty object")
    if not isinstance(submission, dict):
        submission = {}

    results: dict[str, dict[str, Any]] = {}
    passed = 0
    diagnostic_total = 0.0
    for scenario_id, scenario in scenarios.items():
        valid, scenario_score, message = evaluate_scenario(
            scenario, submission.get(scenario_id, {})
        )
        passed += int(valid)
        diagnostic_total += scenario_score
        results[scenario_id] = {
            "valid": valid,
            "score": float(valid),
            "diagnostic_score": scenario_score,
            "message": message,
        }
    return ScoreResult(
        score=passed / float(len(scenarios)),
        correct=passed,
        total=len(scenarios),
        results=results,
        diagnostic_score=diagnostic_total / float(len(scenarios)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--scenarios")
    args = parser.parse_args()

    scenarios_path = (
        Path(args.scenarios)
        if args.scenarios
        else Path(__file__).resolve().with_name("scenarios.json")
    )
    try:
        scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
        submission = json.loads(Path(args.answers).read_text(encoding="utf-8"))
        result = score_submission(scenarios, submission)
    except Exception as exc:  # noqa: BLE001 - CLI diagnostics; host grading propagates errors.
        result = ScoreResult(
            score=0.0,
            correct=0,
            total=12,
            results={"error": {"valid": False, "score": 0.0, "message": str(exc)}},
        )

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    print(f"Weighted diagnostic (not benchmark score): {result.diagnostic_score:.4f}")
    print(f"Score: {result.score:.4f} ({result.correct}/{result.total} PASS)")


if __name__ == "__main__":
    main()
