#!/usr/bin/env python3
"""Scorer for physical_sciences/qchem_hartree_fock_scf.

Reads the agent's output tree and the reference tree; emits a single JSON
payload on stdout with `score`, per-molecule pass flags, and a list of
human-readable failure reasons. Uploaded into the eval temp dir at runtime
and invoked by `main.evaluate`.

Pass conditions (all must hold for a molecule to count toward `n_passed/3`):
  1. `total_energy` matches reference within 1e-6 Hartree.
  2. `eri_<mol>.npy` matches reference element-wise to <= 1e-7.
  3. ERI tensor is 8-fold permutation-symmetric to <= 1e-10.
  4. `overlap_<mol>.npy` diagonal == 1.0 within 1e-10.
  5. `density_<mol>.npy`: |tr(P @ S) - n_e| < 1e-6.
  6. `density_<mol>.npy`: max|P @ S @ P - 2*P| < 1e-6 (RHF idempotency).
  7. Reported `nuclear_repulsion_energy` matches the closed-form value
     `Sum_{A<B} Z_A Z_B / R_AB` derived from spec geometries to <= 1e-10.
  8. `converged == True` and `n_scf_iterations <= 200`.

Hard-gate (auto score=0):
  * `output/results.json` missing or unparseable.
  * Any of the nine `.npy` files missing or wrong shape.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

MOLECULES = ("LiH", "H2O", "C2H2")
N_BASIS = {"LiH": 6, "H2O": 7, "C2H2": 12}
N_ELECTRONS = {"LiH": 4, "H2O": 10, "C2H2": 14}

# Closed-form nuclear repulsion energies. Computed from the explicit Cartesian
# coordinates listed in `input/problem_spec.md` § "Molecules" using
# V_nn = Sum_{A<B} Z_A * Z_B / |R_A - R_B|.  These are NOT read from reference
# data — they are derivable purely from the spec geometry that the agent itself
# is given, so leaking them via this scorer would not give the agent any
# information it cannot already compute.
GEOMETRIES: dict[str, list[tuple[float, tuple[float, float, float]]]] = {
    "LiH": [
        (3.0, (0.000000, 0.000000, 0.000000)),
        (1.0, (3.015000, 0.000000, 0.000000)),
    ],
    "H2O": [
        (8.0, (0.00000000, 0.00000000, 0.0)),
        (1.0, (1.43233673, -0.99825468, 0.0)),
        (1.0, (-1.43233673, -0.99825468, 0.0)),
    ],
    "C2H2": [
        (6.0, (-1.13738, 0.0, 0.0)),
        (6.0, (1.13738, 0.0, 0.0)),
        (1.0, (-3.15738, 0.0, 0.0)),
        (1.0, (3.15738, 0.0, 0.0)),
    ],
}


def _compute_vnn(atoms: list[tuple[float, tuple[float, float, float]]]) -> float:
    total = 0.0
    for i in range(len(atoms)):
        Zi, ri = atoms[i]
        for j in range(i + 1, len(atoms)):
            Zj, rj = atoms[j]
            dx = ri[0] - rj[0]
            dy = ri[1] - rj[1]
            dz = ri[2] - rj[2]
            R = math.sqrt(dx * dx + dy * dy + dz * dz)
            total += Zi * Zj / R
    return total


NUCLEAR_REPULSION_REF: dict[str, float] = {
    mol: _compute_vnn(atoms) for mol, atoms in GEOMETRIES.items()
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_eri_symmetry(eri: np.ndarray, tol: float = 1e-10) -> tuple[bool, str | None]:
    if np.max(np.abs(eri - eri.transpose(1, 0, 2, 3))) > tol:
        return False, "eri_sym_munu"
    if np.max(np.abs(eri - eri.transpose(0, 1, 3, 2))) > tol:
        return False, "eri_sym_lamsig"
    if np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))) > tol:
        return False, "eri_sym_swap"
    return True, None


def _score_molecule(
    mol: str,
    out_results: dict,
    ref_results: dict,
    out_dir: Path,
    ref_dir: Path,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    n = N_BASIS[mol]
    n_e = N_ELECTRONS[mol]

    out_payload = out_results.get(mol)
    ref_payload = ref_results.get(mol)
    if not isinstance(out_payload, dict):
        return False, [f"{mol}: missing payload in output results.json"]
    if not isinstance(ref_payload, dict):
        return False, [f"{mol}: missing payload in reference results.json"]

    eri_path = out_dir / "eri_tensors" / f"eri_{mol}.npy"
    den_path = out_dir / "density_matrices" / f"density_{mol}.npy"
    ovl_path = out_dir / "overlap_matrices" / f"overlap_{mol}.npy"
    ref_eri_path = ref_dir / "eri_tensors" / f"eri_{mol}.npy"

    for label, path, expected_shape in (
        ("eri", eri_path, (n, n, n, n)),
        ("density", den_path, (n, n)),
        ("overlap", ovl_path, (n, n)),
    ):
        if not path.exists():
            return False, [f"{mol}: missing {label} array at {path}"]
        try:
            arr = np.load(path)
        except Exception as exc:  # noqa: BLE001 - report any load failure
            return False, [f"{mol}: failed to load {label} array ({exc})"]
        if arr.shape != expected_shape:
            return False, [f"{mol}: {label} shape {arr.shape} != {expected_shape}"]
        if label == "eri":
            eri = arr
        elif label == "density":
            den = arr
        elif label == "overlap":
            ovl = arr

    if not ref_eri_path.exists():
        return False, [f"{mol}: reference ERI missing at {ref_eri_path}"]
    ref_eri = np.load(ref_eri_path)

    # 1. total_energy
    out_E = out_payload.get("total_energy")
    ref_E = ref_payload.get("total_energy")
    if not isinstance(out_E, (int, float)):
        reasons.append(f"{mol}: total_energy not numeric ({out_E!r})")
    elif not isinstance(ref_E, (int, float)):
        reasons.append(f"{mol}: reference total_energy missing")
    elif abs(out_E - ref_E) > 1e-6:
        reasons.append(f"{mol}: total_energy off by {abs(out_E - ref_E):.3e} > 1e-6")

    # 2. ERI element-wise vs reference
    eri_diff = float(np.max(np.abs(eri - ref_eri)))
    if eri_diff > 1e-7:
        reasons.append(f"{mol}: ERI max-abs error {eri_diff:.3e} > 1e-7")

    # 3. ERI 8-fold symmetry
    sym_ok, sym_label = _check_eri_symmetry(eri)
    if not sym_ok:
        reasons.append(f"{mol}: {sym_label} symmetry violated")

    # 4. overlap diagonal
    ovl_diag_err = float(np.max(np.abs(np.diag(ovl) - 1.0)))
    if ovl_diag_err > 1e-10:
        reasons.append(f"{mol}: overlap diagonal off by {ovl_diag_err:.3e} > 1e-10")

    # 5. tr(P @ S) == n_electrons
    tr_PS = float(np.trace(den @ ovl))
    if abs(tr_PS - n_e) > 1e-6:
        reasons.append(f"{mol}: tr(P·S)={tr_PS:.6f}, expected {n_e}")

    # 6. P @ S @ P == 2*P
    idem_err = float(np.max(np.abs(den @ ovl @ den - 2.0 * den)))
    if idem_err > 1e-6:
        reasons.append(f"{mol}: idempotency residual {idem_err:.3e} > 1e-6")

    # 7. nuclear repulsion closed form
    out_vnn = out_payload.get("nuclear_repulsion_energy")
    expected_vnn = NUCLEAR_REPULSION_REF[mol]
    if not isinstance(out_vnn, (int, float)):
        reasons.append(f"{mol}: nuclear_repulsion_energy not numeric ({out_vnn!r})")
    elif abs(out_vnn - expected_vnn) > 1e-10:
        reasons.append(
            f"{mol}: V_nuc={out_vnn:.10f} differs from closed-form "
            f"{expected_vnn:.10f} by {abs(out_vnn - expected_vnn):.3e}"
        )

    # 8. converged & iter cap
    converged = out_payload.get("converged")
    n_iter = out_payload.get("n_scf_iterations")
    if converged is not True:
        reasons.append(f"{mol}: converged={converged!r} (must be True)")
    if not isinstance(n_iter, int) or n_iter > 200 or n_iter < 1:
        reasons.append(f"{mol}: n_scf_iterations={n_iter!r} (must be 1..200)")

    return (len(reasons) == 0), reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ref_dir = Path(args.reference_dir)

    out_results_path = out_dir / "results.json"
    ref_results_path = ref_dir / "results.json"

    if not out_results_path.exists():
        print(json.dumps({
            "score": 0.0,
            "passed": [],
            "reasons": [f"missing output results.json at {out_results_path}"],
        }))
        return 0
    if not ref_results_path.exists():
        # Reference missing is an evaluator-environment bug, not the agent's
        # fault; report score 0 but flag the cause loudly.
        print(json.dumps({
            "score": 0.0,
            "passed": [],
            "reasons": [f"reference results.json missing at {ref_results_path}"],
        }))
        return 0

    try:
        out_results = _load_json(out_results_path)
    except Exception as exc:  # noqa: BLE001 - any parse error is hard fail
        print(json.dumps({
            "score": 0.0,
            "passed": [],
            "reasons": [f"output results.json unparseable: {exc}"],
        }))
        return 0
    ref_results = _load_json(ref_results_path)

    if not isinstance(out_results, dict):
        print(json.dumps({
            "score": 0.0,
            "passed": [],
            "reasons": ["output results.json is not an object"],
        }))
        return 0

    per_mol_passed: dict[str, bool] = {}
    all_reasons: list[str] = []
    for mol in MOLECULES:
        ok, reasons = _score_molecule(mol, out_results, ref_results, out_dir, ref_dir)
        per_mol_passed[mol] = ok
        all_reasons.extend(reasons)

    n_passed = sum(1 for v in per_mol_passed.values() if v)
    score = n_passed / len(MOLECULES)
    print(json.dumps({
        "score": score,
        "passed": per_mol_passed,
        "reasons": all_reasons,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
