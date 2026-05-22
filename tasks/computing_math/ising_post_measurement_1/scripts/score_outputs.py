"""Local scorer for ising_post_measurement_1 outputs."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from variant_specs import VariantSpec, get_variant

STATE_INFIDELITY_TOLERANCE = 1e-4
MAX_ABS_ERROR_TOLERANCE = 1e-2
REQUIRED_CORRELATOR_KEYS = ("Z_one_body", "X_one_body")


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reason: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["score"] = float(self.score)
        payload["passed"] = bool(self.passed)
        return payload


def _fail(reason: str, **metrics: Any) -> ScoreResult:
    return ScoreResult(score=0.0, passed=False, reason=reason, metrics=metrics)


def _load_npy(blob: bytes, *, label: str) -> np.ndarray:
    try:
        return np.load(io.BytesIO(blob), allow_pickle=False)
    except Exception as exc:  # pragma: no cover - defensive surface
        raise ValueError(f"{label} is not a valid .npy file: {exc}") from exc


def _load_npz(blob: bytes, *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except Exception as exc:  # pragma: no cover - defensive surface
        raise ValueError(f"{label} is not a valid .npz file: {exc}") from exc


def _ensure_numeric_and_finite(name: str, array: np.ndarray) -> None:
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype={array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")


def _max_abs_error(agent: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(agent - reference)))


def _state_infidelity(agent: np.ndarray, reference: np.ndarray) -> float:
    agent_norm_sq = float(np.vdot(agent, agent).real)
    reference_norm_sq = float(np.vdot(reference, reference).real)
    if agent_norm_sq <= 0.0:
        raise ValueError("critical_state.npy has zero norm")
    if reference_norm_sq <= 0.0:
        raise ValueError("reference critical_state.npy has zero norm")
    fidelity = (abs(np.vdot(reference, agent)) ** 2) / (reference_norm_sq * agent_norm_sq)
    fidelity = min(max(float(fidelity), 0.0), 1.0)
    return 1.0 - fidelity


def score_submission_bytes(
    *,
    variant: VariantSpec,
    agent_payloads: dict[str, bytes],
    reference_payloads: dict[str, bytes],
) -> ScoreResult:
    metrics: dict[str, Any] = {"variant_name": variant.variant_name}

    for output_name in variant.required_outputs:
        if output_name not in agent_payloads:
            return _fail(f"Missing required agent output: {output_name}", **metrics)
        if output_name not in reference_payloads:
            return _fail(f"Missing required hidden reference: {output_name}", **metrics)

    expected_dim = variant.expected_dimension

    try:
        critical_agent = _load_npy(agent_payloads["critical_state.npy"], label="critical_state.npy")
        critical_reference = _load_npy(
            reference_payloads["critical_state.npy"], label="reference critical_state.npy"
        )
        _ensure_numeric_and_finite("critical_state.npy", critical_agent)
        _ensure_numeric_and_finite("reference critical_state.npy", critical_reference)
        if critical_agent.shape != (expected_dim,):
            return _fail(
                f"critical_state.npy has wrong shape: expected {(expected_dim,)}, got {critical_agent.shape}",
                **metrics,
            )
        if critical_reference.shape != (expected_dim,):
            return _fail(
                (
                    "reference critical_state.npy has wrong shape: "
                    f"expected {(expected_dim,)}, got {critical_reference.shape}"
                ),
                **metrics,
            )
        state_infidelity = _state_infidelity(critical_agent, critical_reference)
        metrics["state_infidelity"] = state_infidelity
        if state_infidelity > STATE_INFIDELITY_TOLERANCE:
            return _fail(
                (
                    "critical_state.npy exceeds state-infidelity tolerance: "
                    f"{state_infidelity:.6e} > {STATE_INFIDELITY_TOLERANCE:.6e}"
                ),
                **metrics,
            )

        probs_agent = _load_npy(agent_payloads["post_probs.npy"], label="post_probs.npy")
        probs_reference = _load_npy(
            reference_payloads["post_probs.npy"], label="reference post_probs.npy"
        )
        _ensure_numeric_and_finite("post_probs.npy", probs_agent)
        _ensure_numeric_and_finite("reference post_probs.npy", probs_reference)
        if probs_agent.shape != (expected_dim,):
            return _fail(
                f"post_probs.npy has wrong shape: expected {(expected_dim,)}, got {probs_agent.shape}",
                **metrics,
            )
        if probs_reference.shape != (expected_dim,):
            return _fail(
                (
                    "reference post_probs.npy has wrong shape: "
                    f"expected {(expected_dim,)}, got {probs_reference.shape}"
                ),
                **metrics,
            )
        prob_max_abs_error = _max_abs_error(probs_agent, probs_reference)
        metrics["post_probs_max_abs_error"] = prob_max_abs_error
        if prob_max_abs_error > MAX_ABS_ERROR_TOLERANCE:
            return _fail(
                (
                    "post_probs.npy exceeds max-abs-error tolerance: "
                    f"{prob_max_abs_error:.6e} > {MAX_ABS_ERROR_TOLERANCE:.6e}"
                ),
                **metrics,
            )

        rdm_agent = _load_npy(agent_payloads["rdm_site1.npy"], label="rdm_site1.npy")
        rdm_reference = _load_npy(
            reference_payloads["rdm_site1.npy"], label="reference rdm_site1.npy"
        )
        _ensure_numeric_and_finite("rdm_site1.npy", rdm_agent)
        _ensure_numeric_and_finite("reference rdm_site1.npy", rdm_reference)
        expected_rdm_shape = (expected_dim, 2, 2)
        if rdm_agent.shape != expected_rdm_shape:
            return _fail(
                f"rdm_site1.npy has wrong shape: expected {expected_rdm_shape}, got {rdm_agent.shape}",
                **metrics,
            )
        if rdm_reference.shape != expected_rdm_shape:
            return _fail(
                (
                    "reference rdm_site1.npy has wrong shape: "
                    f"expected {expected_rdm_shape}, got {rdm_reference.shape}"
                ),
                **metrics,
            )
        rdm_max_abs_error = _max_abs_error(rdm_agent, rdm_reference)
        metrics["rdm_site1_max_abs_error"] = rdm_max_abs_error
        if rdm_max_abs_error > MAX_ABS_ERROR_TOLERANCE:
            return _fail(
                (
                    "rdm_site1.npy exceeds max-abs-error tolerance: "
                    f"{rdm_max_abs_error:.6e} > {MAX_ABS_ERROR_TOLERANCE:.6e}"
                ),
                **metrics,
            )

        if variant.requires_correlators:
            correlators_agent = _load_npz(
                agent_payloads["correlators.npz"], label="correlators.npz"
            )
            correlators_reference = _load_npz(
                reference_payloads["correlators.npz"], label="reference correlators.npz"
            )
            for key in REQUIRED_CORRELATOR_KEYS:
                if key not in correlators_agent:
                    return _fail(f"correlators.npz missing key: {key}", **metrics)
                if key not in correlators_reference:
                    return _fail(f"reference correlators.npz missing key: {key}", **metrics)
                _ensure_numeric_and_finite(f"correlators.npz:{key}", correlators_agent[key])
                _ensure_numeric_and_finite(
                    f"reference correlators.npz:{key}", correlators_reference[key]
                )
                expected_corr_shape = (expected_dim, variant.n_qubits)
                if correlators_agent[key].shape != expected_corr_shape:
                    return _fail(
                        (
                            f"correlators.npz:{key} has wrong shape: expected "
                            f"{expected_corr_shape}, got {correlators_agent[key].shape}"
                        ),
                        **metrics,
                    )
                if correlators_reference[key].shape != expected_corr_shape:
                    return _fail(
                        (
                            f"reference correlators.npz:{key} has wrong shape: expected "
                            f"{expected_corr_shape}, got {correlators_reference[key].shape}"
                        ),
                        **metrics,
                    )
            correlator_max_abs_error = max(
                _max_abs_error(correlators_agent[key], correlators_reference[key])
                for key in REQUIRED_CORRELATOR_KEYS
            )
            metrics["correlators_max_abs_error"] = correlator_max_abs_error
            if correlator_max_abs_error > MAX_ABS_ERROR_TOLERANCE:
                return _fail(
                    (
                        "correlators.npz exceeds max-abs-error tolerance: "
                        f"{correlator_max_abs_error:.6e} > {MAX_ABS_ERROR_TOLERANCE:.6e}"
                    ),
                    **metrics,
                )

    except ValueError as exc:
        return _fail(str(exc), **metrics)

    return ScoreResult(
        score=1.0, passed=True, reason="all required outputs passed", metrics=metrics
    )


def _load_payloads_from_dir(directory: Path, variant: VariantSpec) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for output_name in variant.required_outputs:
        payloads[output_name] = (directory / output_name).read_bytes()
    return payloads


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="Canonical variant name")
    parser.add_argument("--agent-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    variant = get_variant(args.variant)
    result = score_submission_bytes(
        variant=variant,
        agent_payloads=_load_payloads_from_dir(args.agent_dir, variant),
        reference_payloads=_load_payloads_from_dir(args.reference_dir, variant),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
