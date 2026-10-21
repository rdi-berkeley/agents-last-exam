"""Build deterministic visible inputs and author-only verification witnesses."""

from __future__ import annotations

import cmath
import math
import random
from typing import Any

try:
    from .verify_outputs import (
        expm_hermitian_8x8,
        kron3,
        mat_add,
        mat_dag,
        mat_eye,
        mat_mul,
        u3,
    )
except ImportError:
    from verify_outputs import (  # type: ignore[no-redef]
        expm_hermitian_8x8,
        kron3,
        mat_add,
        mat_dag,
        mat_eye,
        mat_mul,
        u3,
    )

TASK_SPEC = r"""# Constrained gate synthesis in frustrated spin triangles

## Goal

For each of the 12 scenarios in `scenarios.json`, construct a piecewise-constant
local-frame and exchange-drive schedule with between 2 and 96 pulse segments.
The schedule must synthesize the supplied three-qubit target gate while
cancelling transverse exchange leakage and the first two static-noise Magnus
terms for ten independently weighted noise channels. This is a synthetic
control-compilation problem with the exact model specified below.

The three-qubit Hilbert space has dimension 8. Every scenario supplies raw
laboratory-frame 8 by 8 matrices as grids of `[real, imag]` pairs:

- `H_drift`
- `H_ex_12`, `H_ex_23`, and `H_ex_31`
- `H_leak_12`, `H_leak_23`, and `H_leak_31`
- `E_noise`
- `U_target`

The local spin-orbit gauge is not supplied separately. It must be inferred from
the visible operators if your construction needs it.

## Required output

Write `output/answers.json` as an object with keys `S1` through `S12`. Each value
must contain one `frame_calibration`, three shared `bond_amplitudes`, and a
`pulses` list:

```json
{
  "S1": {
    "frame_calibration": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "bond_amplitudes": [0.1, 0.0, 0.0],
    "pulses": [
      {
        "frame_codes": [0, 0, 0],
        "bond_drives": [0.1, 0.0, 0.0],
        "duration": 0.25
      },
      {
        "frame_codes": [1, 1, 1],
        "bond_drives": [0.0, 0.0, 0.0],
        "duration": 0.25
      }
    ]
  }
}
```

The example illustrates the schema, not a valid solution. The nine calibration
angles define three U3 matrices `C1`, `C2`, `C3`, shared by every pulse in that
scenario. Angles must be finite and lie in `[-1e6, 1e6]`. You choose these
calibrations; they are not given. Each pulse contains:

- `frame_codes`: three integers in `[0, 3]`, selecting `I, X, Y, Z` respectively.
  Spin `q` uses `Cq P_code`, in that order. Arbitrary per-pulse rotations are
  unavailable: the controller supports four calibrated frames per spin.
- `bond_drives`: finite `[J12, J23, J31]`, each in `[-0.25, 0.25]`.
  At most one bond may have a nonzero drive in any pulse; idle pulses are allowed.
  For bond `b`, every drive must equal exactly `0`, `+bond_amplitudes[b]`, or
  `-bond_amplitudes[b]`. The three amplitude magnitudes are chosen once per
  scenario, must be finite and in `[0, 0.25]`, and cannot vary between pulses.
  Reuse the same JSON numeric magnitude for every activation of a bond.
- `duration`: exactly `0.25` for every pulse.

For every scenario, every bond must have total exchange action
`sum_k(abs(J_ab,k) * duration_k) >= 0.05`.

## Verification

Run:

```bash
python3 input/check.py --answers output/answers.json
```

A scenario passes only when all four conditions hold with tolerance `1e-8`:

1. Gate infidelity `1 - |Tr(U_target^dagger U_total)|^2 / 64`.
2. Maximum Frobenius norm of the three bond-resolved integrated leakage operators.
3. Maximum Frobenius norm of the ten first-order noise coefficients.
4. Maximum Frobenius norm of all 55 second-order noise coefficients, including
   cross-channel terms. The channels are `E_noise` and the nine laboratory-frame
   single-spin Pauli operators, with independent static amplitudes.

Each scenario scores 1 only if its control constraints and all four physical
criteria pass; otherwise it scores 0. The benchmark score is the fraction of
the 12 scenarios that pass completely. A noise-cancelling schedule producing
the wrong gate is not a completed gate-synthesis solution.

For diagnosis only, the checker also reports a weighted criterion score:
gate 0.80, leakage 0.08, first order 0.06, second order 0.06. This diagnostic
does not contribute to the benchmark score. See `calculus.md` and `check.py`
for the exact forward model.
"""

CALCULUS_SPEC = r"""# Mathematical specification

For pulse segment `k`, define

`W_k = (C1 P_code1,k) tensor (C2 P_code2,k) tensor (C3 P_code3,k)`,

where the shared `Cq = U3(theta_q, phi_q, lambda_q)` are defined by
`frame_calibration`, and `P_0 = I`, `P_1 = X`, `P_2 = Y`, `P_3 = Z`.
Here `X = [[0,1],[1,0]]`, `Y = [[0,-i],[i,0]]`, and `Z = diag(1,-1)`.

The local calibration matrix is

```
U3(theta, phi, lambda) =
  [[cos(theta/2),               -exp(i lambda) sin(theta/2)],
   [exp(i phi) sin(theta/2), exp(i(phi+lambda)) cos(theta/2)]].
```

The dressed Hamiltonian and time-ordered propagator are

```
H_k = W_k^dagger (H_drift + J12 H_ex_12 + J23 H_ex_23 + J31 H_ex_31) W_k
U_total = exp(-i H_L tau_L) ... exp(-i H_1 tau_1).
```

The three transverse leakage integrals are tested separately:

```
L_ab = sum_k tau_k J_ab,k W_k^dagger H_leak_ab W_k.
```

Let `E_0 = E_noise`. Let `E_1,...,E_9` be `XII,YII,ZII,IXI,IYI,IZI,IIX,IIY,IIZ`
in the laboratory frame. With independent static amplitudes `a_r`, the error is
`E(a) = sum_r a_r E_r`. Define `B_r,k = tau_k W_k^dagger E_r W_k`.
The coefficient-wise cancellation conditions are

```
A_r = sum_k B_r,k
Q_rr = 0.5 sum_{j<k} [B_r,k, B_r,j]
Q_rs = 0.5 sum_{j<k} ([B_r,k, B_s,j] + [B_s,k, B_r,j]), r < s.
```

Thus `Omega_1(a) = sum_r a_r A_r` and
`Omega_2(a) = sum_r a_r^2 Q_rr + sum_{r<s} a_r a_s Q_rs`.
Every coefficient must have Frobenius norm at most `1e-8`. No averaging over
noise directions is used. These are the task's idealized frame-level error
functionals; they do not include finite-width frame switches or transport
through the ideal gate propagator.

All 12 scenarios have feasible schedules under these constraints. The verifier
checks the physical matrices and constraints, not agreement with a reference
pulse sequence. Global phases of local frames and final gates are irrelevant.
All norms are Frobenius norms. The verifier is the executable specification.
"""


def su2_mul(left: list[list[complex]], right: list[list[complex]]) -> list[list[complex]]:
    return [
        [
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ],
        [
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ],
    ]


def decompose_u3(unitary: list[list[complex]]) -> list[float]:
    determinant = unitary[0][0] * unitary[1][1] - unitary[0][1] * unitary[1][0]
    phase = cmath.phase(determinant) / 2.0
    special = [[value * cmath.exp(-1j * phase) for value in row] for row in unitary]
    cosine = min(1.0, max(0.0, abs(special[0][0])))
    sine = min(1.0, max(0.0, abs(special[1][0])))
    theta = 2.0 * math.atan2(sine, cosine)
    alpha = cmath.phase(special[0][0]) if cosine > 1e-12 else 0.0
    beta = cmath.phase(special[1][0]) if sine > 1e-12 else 0.0
    if cosine <= 1e-12:
        alpha = -cmath.phase(special[1][1])
    return [theta, beta - alpha, -beta - alpha]


def encode_matrix(matrix: list[list[complex]]) -> list[list[list[float]]]:
    return [[[value.real, value.imag] for value in row] for row in matrix]


def _scenario_configs() -> list[tuple[Any, ...]]:
    return [
        (
            "S1",
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0.0,
            0.0,
            0.0,
            [1.25, 1.10, 1.35],
            [math.pi / 4.0, math.pi / 6.0, -math.pi / 8.0],
        ),
        (
            "S2",
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0.0,
            0.0,
            0.0,
            [1.40, 1.20, 1.15],
            [math.pi / 3.0, -math.pi / 5.0, math.pi / 4.0],
        ),
        (
            "S3",
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            0.0,
            0.0,
            0.0,
            [1.15, 1.30, 1.25],
            [math.pi / 6.0, math.pi / 4.0, math.pi / 5.0],
        ),
        (
            "S4",
            [0.46, -0.32, 0.21],
            [-0.39, 0.41, -0.18],
            [0.0, 0.0, 0.0],
            0.45,
            -0.30,
            -0.15,
            [1.30, 1.15, 1.40],
            [math.pi / 4.0, math.pi / 6.0, -math.pi / 8.0],
        ),
        (
            "S5",
            [-0.54, 0.37, -0.29],
            [0.48, -0.35, 0.24],
            [0.52, -0.38, 0.27],
            -0.62,
            0.28,
            0.34,
            [1.20, 1.35, 1.10],
            [math.pi / 3.0, -math.pi / 5.0, math.pi / 4.0],
        ),
        (
            "S6",
            [0.42, 0.31, -0.25],
            [-0.38, 0.52, 0.19],
            [0.55, -0.41, 0.33],
            0.52,
            -0.35,
            -0.17,
            [1.45, 1.25, 1.30],
            [math.pi / 4.0, math.pi / 5.0, math.pi / 6.0],
        ),
        (
            "S7",
            [-0.51, 0.44, 0.28],
            [0.63, -0.29, -0.41],
            [-0.39, 0.48, -0.22],
            -0.68,
            0.41,
            0.27,
            [1.35, 1.50, 1.20],
            [2.0 * math.pi / 5.0, -math.pi / 4.0, math.pi / 3.0],
        ),
        (
            "S8",
            [0.64, -0.52, 0.37],
            [-0.47, -0.61, 0.29],
            [0.58, 0.33, -0.49],
            0.74,
            -0.48,
            -0.26,
            [1.25, 1.40, 1.55],
            [3.0 * math.pi / 8.0, math.pi / 3.0, -math.pi / 5.0],
        ),
        (
            "S9",
            [-0.72, 0.39, -0.54],
            [0.51, 0.68, -0.36],
            [-0.66, -0.42, 0.57],
            -0.81,
            0.53,
            0.28,
            [1.50, 1.15, 1.35],
            [5.0 * math.pi / 12.0, -3.0 * math.pi / 8.0, math.pi / 4.0],
        ),
        (
            "S10",
            [0.83, -0.61, 0.45],
            [-0.69, 0.47, 0.58],
            [0.75, -0.53, -0.38],
            0.89,
            -0.57,
            -0.32,
            [1.30, 1.45, 1.25],
            [math.pi / 3.0, 5.0 * math.pi / 12.0, -math.pi / 6.0],
        ),
        (
            "S11",
            [-0.67, -0.74, 0.52],
            [0.78, -0.55, 0.43],
            [-0.82, 0.61, -0.47],
            -0.95,
            0.62,
            0.33,
            [1.55, 1.30, 1.40],
            [7.0 * math.pi / 12.0, -math.pi / 3.0, 3.0 * math.pi / 8.0],
        ),
        (
            "S12",
            [0.91, 0.58, -0.66],
            [-0.84, -0.49, 0.71],
            [0.69, 0.77, -0.53],
            1.04,
            -0.68,
            -0.36,
            [1.40, 1.60, 1.50],
            [math.pi / 2.0, math.pi / 4.0, -5.0 * math.pi / 12.0],
        ),
    ]


def build_fixtures() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    identity = [[1.0 + 0j, 0j], [0j, 1.0 + 0j]]
    pauli_x = [[0j, 1.0 + 0j], [1.0 + 0j, 0j]]
    pauli_y = [[0j, -1j], [1j, 0j]]
    pauli_z = [[1.0 + 0j, 0j], [0j, -1.0 + 0j]]

    group_8 = [
        (identity, identity, identity),
        (pauli_x, pauli_x, pauli_x),
        (pauli_z, identity, identity),
        (pauli_y, pauli_x, pauli_x),
        (identity, pauli_z, identity),
        (pauli_x, pauli_y, pauli_x),
        (pauli_z, pauli_z, identity),
        (pauli_y, pauli_y, pauli_x),
    ]
    group_16 = []
    for first, second, third in group_8:
        group_16.append((first, second, third))
        group_16.append((first, second, su2_mul(third, pauli_z)))

    scenarios: dict[str, dict[str, Any]] = {}
    author_answers: dict[str, dict[str, Any]] = {}
    tier1_answers: dict[str, dict[str, Any]] = {}
    tier2_answers: dict[str, dict[str, Any]] = {}

    for (
        scenario_id,
        euler_1,
        euler_2,
        euler_3,
        dm_1,
        dm_2,
        dm_3,
        gammas,
        target_angles,
    ) in _scenario_configs():
        gauge_1 = su2_mul(u3(*euler_1), u3(0.0, 0.0, dm_1))
        gauge_2 = su2_mul(u3(*euler_2), u3(0.0, 0.0, dm_2))
        gauge_3 = su2_mul(u3(*euler_3), u3(0.0, 0.0, dm_3))
        gauge = kron3(gauge_1, gauge_2, gauge_3)
        gauge_dag = mat_dag(gauge)

        xx_i = kron3(pauli_x, pauli_x, identity)
        yy_i = kron3(pauli_y, pauli_y, identity)
        zz_i = kron3(pauli_z, pauli_z, identity)
        i_xx = kron3(identity, pauli_x, pauli_x)
        i_yy = kron3(identity, pauli_y, pauli_y)
        i_zz = kron3(identity, pauli_z, pauli_z)
        x_i_x = kron3(pauli_x, identity, pauli_x)
        y_i_y = kron3(pauli_y, identity, pauli_y)
        z_i_z = kron3(pauli_z, identity, pauli_z)

        scenario_number = int(scenario_id[1:])
        xz_i = kron3(pauli_x, pauli_z, identity)
        yz_i = kron3(pauli_y, pauli_z, identity)
        zx_i = kron3(pauli_z, pauli_x, identity)
        zy_i = kron3(pauli_z, pauli_y, identity)
        i_xz = kron3(identity, pauli_x, pauli_z)
        i_yz = kron3(identity, pauli_y, pauli_z)
        i_zx = kron3(identity, pauli_z, pauli_x)
        i_zy = kron3(identity, pauli_z, pauli_y)
        x_i_z = kron3(pauli_x, identity, pauli_z)
        y_i_z = kron3(pauli_y, identity, pauli_z)
        z_i_x = kron3(pauli_z, identity, pauli_x)
        z_i_y = kron3(pauli_z, identity, pauli_y)

        leak_12_base = mat_add(
            mat_add(mat_add(xx_i, yy_i, 1.0, 0.63), mat_add(xz_i, yz_i, 0.44, 0.37)),
            mat_add(zx_i, zy_i, 0.51, -0.49),
        )
        leak_23_base = mat_add(
            mat_add(mat_add(i_xx, i_yy, 1.0, 0.58), mat_add(i_xz, i_yz, 0.41, -0.35)),
            mat_add(i_zx, i_zy, -0.47, 0.52),
        )
        leak_31_base = mat_add(
            mat_add(mat_add(x_i_x, y_i_y, 1.0, 0.67), mat_add(x_i_z, y_i_z, -0.39, 0.43)),
            mat_add(z_i_x, z_i_y, 0.48, -0.45),
        )
        eta = 0.85
        exchange_12_base = mat_add(
            mat_add(leak_12_base, zz_i, 1.0, gammas[0]), mat_add(xx_i, yy_i, eta, -eta)
        )
        exchange_23_base = mat_add(
            mat_add(leak_23_base, i_zz, 1.0, gammas[1]), mat_add(i_xx, i_yy, eta, -eta)
        )
        exchange_31_base = mat_add(
            mat_add(leak_31_base, z_i_z, 1.0, gammas[2]), mat_add(x_i_x, y_i_y, eta, -eta)
        )

        zii = kron3(pauli_z, identity, identity)
        izi = kron3(identity, pauli_z, identity)
        iiz = kron3(identity, identity, pauli_z)
        omega_1 = 1.15 + 0.17 * scenario_number
        omega_2 = -0.85 - 0.13 * scenario_number
        omega_3 = 1.65 - 0.11 * scenario_number
        drift_base = mat_add(
            mat_add(zii, izi, 0.5 * omega_1, 0.5 * omega_2),
            iiz,
            1.0,
            0.5 * omega_3,
        )

        noise_base = mat_add(
            mat_add(
                mat_add(zii, izi, 0.40, -0.35),
                mat_add(iiz, xx_i, 0.30, 0.45),
            ),
            mat_add(
                mat_add(i_xx, kron3(pauli_z, pauli_y, identity), -0.38, 0.42),
                kron3(identity, pauli_z, pauli_y),
                1.0,
                -0.33,
            ),
        )

        def dress(
            matrix: list[list[complex]],
            frame: list[list[complex]] = gauge,
            frame_dag: list[list[complex]] = gauge_dag,
        ) -> list[list[complex]]:
            return mat_mul(frame, mat_mul(matrix, frame_dag))

        drift = dress(drift_base)
        exchange_12 = dress(exchange_12_base)
        exchange_23 = dress(exchange_23_base)
        exchange_31 = dress(exchange_31_base)
        leak_12 = dress(leak_12_base)
        leak_23 = dress(leak_23_base)
        leak_31 = dress(leak_31_base)
        noise = dress(noise_base)

        duration = 0.25
        drives = [target_angles[index] / (32.0 * gammas[index] * duration) for index in range(3)]
        first_half = (
            [(element, [drives[0], 0.0, 0.0]) for element in group_16]
            + [(element, [0.0, drives[1], 0.0]) for element in group_16]
            + [(element, [0.0, 0.0, drives[2]]) for element in group_16]
        )
        # Permuting a balanced multiset preserves its first-order averages;
        # reversal cancels its second-order coefficients for every noise channel.
        random.Random(0xA1E_2026 + scenario_number).shuffle(first_half)
        schedule = first_half + list(reversed(first_half))

        target = mat_eye()
        author_pulses = []
        tier1_pulses = []
        tier2_pulses = []
        for (first, second, third), segment_drives in schedule:
            codes = []
            for operator in (first, second, third):
                overlaps = [
                    abs(
                        sum(
                            operator[row][column].conjugate() * pauli[row][column]
                            for row in range(2)
                            for column in range(2)
                        )
                    )
                    for pauli in (identity, pauli_x, pauli_y, pauli_z)
                ]
                codes.append(max(range(4), key=overlaps.__getitem__))
            author_frame = (
                decompose_u3(su2_mul(gauge_1, first))
                + decompose_u3(su2_mul(gauge_2, second))
                + decompose_u3(su2_mul(gauge_3, third))
            )
            author_pulses.append(
                {
                    "frame_codes": codes,
                    "bond_drives": segment_drives,
                    "duration": duration,
                }
            )
            local_frame = kron3(
                u3(*author_frame[0:3]),
                u3(*author_frame[3:6]),
                u3(*author_frame[6:9]),
            )
            exchange = mat_add(
                mat_add(
                    exchange_12,
                    exchange_23,
                    segment_drives[0],
                    segment_drives[1],
                ),
                exchange_31,
                1.0,
                segment_drives[2],
            )
            hamiltonian = mat_mul(
                mat_dag(local_frame), mat_mul(mat_add(drift, exchange), local_frame)
            )
            target = mat_mul(expm_hermitian_8x8(hamiltonian, duration), target)

            tier1_pulses.append(
                {
                    "frame_codes": codes,
                    "bond_drives": segment_drives,
                    "duration": duration,
                }
            )
            tier2_pulses.append(
                {
                    "frame_codes": codes,
                    "bond_drives": segment_drives,
                    "duration": duration,
                }
            )

        scenarios[scenario_id] = {
            "scenario_id": scenario_id,
            "H_drift": encode_matrix(drift),
            "H_ex_12": encode_matrix(exchange_12),
            "H_ex_23": encode_matrix(exchange_23),
            "H_ex_31": encode_matrix(exchange_31),
            "H_leak_12": encode_matrix(leak_12),
            "H_leak_23": encode_matrix(leak_23),
            "H_leak_31": encode_matrix(leak_31),
            "E_noise": encode_matrix(noise),
            "U_target": encode_matrix(target),
        }
        author_answers[scenario_id] = {
            "frame_calibration": decompose_u3(gauge_1)
            + decompose_u3(gauge_2)
            + decompose_u3(gauge_3),
            "pulses": author_pulses,
            "bond_amplitudes": [abs(value) for value in drives],
        }
        tier1_answers[scenario_id] = {
            "frame_calibration": [0.0] * 9,
            "pulses": tier1_pulses,
            "bond_amplitudes": [abs(value) for value in drives],
        }
        tier2_answers[scenario_id] = {
            "frame_calibration": decompose_u3(gauge_1) + decompose_u3(gauge_2) + [0.0, 0.0, dm_3],
            "pulses": tier2_pulses,
            "bond_amplitudes": [abs(value) for value in drives],
        }

    return scenarios, author_answers, tier1_answers, tier2_answers
