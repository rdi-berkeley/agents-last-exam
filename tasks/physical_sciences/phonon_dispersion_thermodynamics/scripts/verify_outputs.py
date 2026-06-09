"""Local verifier for phonon_dispersion_thermodynamics outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

NPZ_SPECS = {
    "diatomic_1d.npz": {
        "keys": ("q_points", "omega_acoustic", "omega_optical"),
        "tolerances": {
            "q_points": (1e-10, 1e-12),
            "omega_acoustic": (1e-8, 1e-10),
            "omega_optical": (1e-8, 1e-10),
        },
    },
    "dispersion_2d.npz": {
        "keys": ("q_points", "q_distance", "frequencies"),
        "tolerances": {
            "q_points": (1e-10, 1e-12),
            "q_distance": (1e-10, 1e-12),
            "frequencies": (1e-6, 1e-8),
        },
    },
    "dos.npz": {
        "keys": ("omega_bins", "dos"),
        "tolerances": {
            "omega_bins": (1e-10, 1e-12),
            "dos": (2e-3, 1e-5),
        },
    },
    "thermodynamics.npz": {
        "keys": ("temperatures", "energy", "heat_capacity", "free_energy"),
        "tolerances": {
            "temperatures": (1e-10, 1e-12),
            "energy": (2e-3, 1e-5),
            "heat_capacity": (2e-3, 1e-5),
            "free_energy": (2e-3, 1e-5),
        },
    },
}

RESULTS_SPEC = {
    "tier1": {
        "optical_frequency_at_gamma": (1e-8, 1e-10),
        "max_acoustic_at_gamma": (1e-8, 1e-10),
        "num_q_points": None,
    },
    "tier2": {
        "num_q_points": None,
        "num_branches": None,
        "freq_range": (1e-6, 1e-8),
        "negative_eigenvalue_count": None,
        "acoustic_at_gamma": (1e-8, 1e-10),
    },
    "tier3": {
        "omega_max": (1e-6, 1e-8),
        "dos_integral": (2e-3, 1e-5),
        "cv_high_T": (2e-3, 1e-5),
        "cv_low_T": (2e-3, 1e-5),
        "num_temperatures": None,
        "q_grid_size": None,
    },
}

REQUIRED_FILES = tuple(NPZ_SPECS) + ("results.json",)


def _fail(failures: list[str], message: str) -> None:
    failures.append(message)


def _compare_array(
    name: str,
    key: str,
    agent: np.ndarray,
    reference: np.ndarray,
    *,
    rtol: float,
    atol: float,
    failures: list[str],
) -> None:
    if agent.shape != reference.shape:
        _fail(
            failures,
            f"{name}:{key} shape mismatch ({agent.shape} != {reference.shape})",
        )
        return
    if not np.allclose(agent, reference, rtol=rtol, atol=atol, equal_nan=False):
        max_abs = float(np.max(np.abs(agent - reference)))
        _fail(
            failures,
            f"{name}:{key} values differ (rtol={rtol}, atol={atol}, max_abs_diff={max_abs})",
        )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _compare_npz(name: str, output_path: Path, reference_path: Path, failures: list[str]) -> None:
    agent_payload = _load_npz(output_path)
    reference_payload = _load_npz(reference_path)
    expected_keys = tuple(NPZ_SPECS[name]["keys"])

    if tuple(sorted(agent_payload)) != tuple(sorted(expected_keys)):
        _fail(failures, f"{name} keys mismatch in agent output: {tuple(sorted(agent_payload))}")
        return
    if tuple(sorted(reference_payload)) != tuple(sorted(expected_keys)):
        _fail(
            failures,
            f"{name} keys mismatch in reference payload: {tuple(sorted(reference_payload))}",
        )
        return

    for key in expected_keys:
        rtol, atol = NPZ_SPECS[name]["tolerances"][key]
        _compare_array(
            name,
            key,
            np.asarray(agent_payload[key]),
            np.asarray(reference_payload[key]),
            rtol=rtol,
            atol=atol,
            failures=failures,
        )


def _compare_scalar(
    path: str,
    agent_value: Any,
    reference_value: Any,
    *,
    tolerance: tuple[float, float] | None,
    failures: list[str],
) -> None:
    if tolerance is None:
        if agent_value != reference_value:
            _fail(failures, f"{path} mismatch ({agent_value!r} != {reference_value!r})")
        return

    rtol, atol = tolerance
    if isinstance(reference_value, list):
        if not isinstance(agent_value, list):
            _fail(failures, f"{path} should be a list")
            return
        if len(agent_value) != len(reference_value):
            _fail(failures, f"{path} length mismatch ({len(agent_value)} != {len(reference_value)})")
            return
        for idx, (agent_item, reference_item) in enumerate(zip(agent_value, reference_value)):
            if not math.isclose(float(agent_item), float(reference_item), rel_tol=rtol, abs_tol=atol):
                _fail(
                    failures,
                    f"{path}[{idx}] mismatch ({agent_item!r} != {reference_item!r})",
                )
        return

    if not math.isclose(float(agent_value), float(reference_value), rel_tol=rtol, abs_tol=atol):
        _fail(failures, f"{path} mismatch ({agent_value!r} != {reference_value!r})")


def _compare_results(output_path: Path, reference_path: Path, failures: list[str]) -> None:
    agent = json.loads(output_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    if tuple(sorted(agent)) != tuple(sorted(RESULTS_SPEC)):
        _fail(failures, f"results.json top-level keys mismatch: {tuple(sorted(agent))}")
        return
    if tuple(sorted(reference)) != tuple(sorted(RESULTS_SPEC)):
        _fail(failures, f"reference results.json top-level keys mismatch: {tuple(sorted(reference))}")
        return

    for tier_name, fields in RESULTS_SPEC.items():
        agent_tier = agent.get(tier_name)
        reference_tier = reference.get(tier_name)
        if not isinstance(agent_tier, dict) or not isinstance(reference_tier, dict):
            _fail(failures, f"{tier_name} should be a JSON object")
            continue
        if tuple(sorted(agent_tier)) != tuple(sorted(fields)):
            _fail(failures, f"{tier_name} keys mismatch: {tuple(sorted(agent_tier))}")
            continue
        for key, tolerance in fields.items():
            _compare_scalar(
                f"{tier_name}.{key}",
                agent_tier.get(key),
                reference_tier.get(key),
                tolerance=tolerance,
                failures=failures,
            )


def evaluate_output_tree(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    failures: list[str] = []

    for name in REQUIRED_FILES:
        output_path = output_dir / name
        reference_path = reference_dir / name
        if not output_path.exists():
            _fail(failures, f"missing output file: {name}")
            continue
        if not reference_path.exists():
            _fail(failures, f"missing reference file: {name}")
            continue

        if name.endswith(".npz"):
            try:
                _compare_npz(name, output_path, reference_path, failures)
            except Exception as exc:
                _fail(failures, f"{name} could not be compared: {exc}")
        else:
            try:
                _compare_results(output_path, reference_path, failures)
            except Exception as exc:
                _fail(failures, f"{name} could not be compared: {exc}")

    return {
        "score": 1.0 if not failures else 0.0,
        "passed": not failures,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_output_tree(args.output_dir, args.reference_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

