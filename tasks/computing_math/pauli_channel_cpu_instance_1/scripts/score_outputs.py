"""Scoring helpers for pauli_channel_cpu_instance_1."""

import io
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    from .variant_specs import VARIANT, VariantSpec
except ImportError:  # pragma: no cover - direct script execution fallback
    from variant_specs import VARIANT, VariantSpec

SUMMARY_REQUIRED_KEYS = {
    "schema_version",
    "variant_id",
    "n_qubits",
    "num_random_states",
    "state_support_size",
    "num_trajectories",
    "rng_spec",
    "state_seed",
    "trajectory_seed",
    "channel_generation",
    "channel_terms",
    "density_nnz_per_state",
}
NPZ_REQUIRED_KEYS = {"row", "col", "data_real", "data_imag", "shape"}
TRACE_TOLERANCE = 1e-6
PERFECT_THRESHOLD = 5e-4


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


def _load_json(blob: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(blob.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return value


def _load_npz(blob: bytes, *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except Exception as exc:
        raise ValueError(f"{label} is not a valid .npz file: {exc}") from exc


def _validate_summary(summary: dict[str, Any], variant: VariantSpec) -> None:
    missing = SUMMARY_REQUIRED_KEYS - set(summary)
    if missing:
        raise ValueError(f"summary.json missing keys: {sorted(missing)}")

    expected_pairs = {
        "schema_version": 1,
        "variant_id": variant.variant_name,
        "n_qubits": variant.n_qubits,
        "num_random_states": variant.num_random_states,
        "state_support_size": variant.state_support_size,
        "num_trajectories": variant.num_trajectories,
        "state_seed": variant.state_seed,
        "trajectory_seed": variant.trajectory_seed,
    }
    for key, expected in expected_pairs.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"summary.json metadata mismatch for {key}: expected {expected!r}, got {summary.get(key)!r}"
            )

    rng_spec = summary.get("rng_spec")
    if rng_spec != {
        "library": "numpy",
        "api": "default_rng",
        "bit_generator": "PCG64",
    }:
        raise ValueError(f"summary.json rng_spec mismatch: {rng_spec!r}")

    channel_generation = summary.get("channel_generation")
    if not isinstance(channel_generation, dict):
        raise ValueError("summary.json channel_generation must be an object")
    if channel_generation.get("seed") != variant.channel_seed:
        raise ValueError(
            "summary.json channel_generation.seed mismatch: "
            f"expected {variant.channel_seed}, got {channel_generation.get('seed')!r}"
        )

    channel_terms = summary.get("channel_terms")
    if not isinstance(channel_terms, list) or not channel_terms:
        raise ValueError("summary.json channel_terms must be a non-empty list")
    prob_sum = 0.0
    for idx, term in enumerate(channel_terms):
        if not isinstance(term, dict):
            raise ValueError(f"summary.json channel_terms[{idx}] must be an object")
        probability = term.get("probability")
        if not isinstance(probability, (int, float)):
            raise ValueError(f"summary.json channel_terms[{idx}].probability must be numeric")
        if probability < 0.0:
            raise ValueError(f"summary.json channel_terms[{idx}].probability must be non-negative")
        prob_sum += float(probability)
    if not math.isclose(prob_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"summary.json probabilities sum to {prob_sum:.12f}, expected 1.0")

    density_nnz = summary.get("density_nnz_per_state")
    if not isinstance(density_nnz, list) or len(density_nnz) != variant.num_random_states:
        raise ValueError(
            "summary.json density_nnz_per_state must have "
            f"{variant.num_random_states} entries"
        )


def _validate_density_payload(
    payload: dict[str, np.ndarray],
    *,
    label: str,
    variant: VariantSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], float]:
    if set(payload) != NPZ_REQUIRED_KEYS:
        raise ValueError(
            f"{label} keys mismatch: expected {sorted(NPZ_REQUIRED_KEYS)}, got {sorted(payload)}"
        )

    row = np.asarray(payload["row"])
    col = np.asarray(payload["col"])
    data_real = np.asarray(payload["data_real"])
    data_imag = np.asarray(payload["data_imag"])
    shape = np.asarray(payload["shape"])

    if row.ndim != 1 or col.ndim != 1 or data_real.ndim != 1 or data_imag.ndim != 1:
        raise ValueError(f"{label} row/col/data arrays must be 1D")
    if not (row.shape == col.shape == data_real.shape == data_imag.shape):
        raise ValueError(f"{label} row/col/data array shapes must match")
    if shape.shape != (2,):
        raise ValueError(f"{label} shape must be length-2, got {shape.shape}")

    matrix_shape = (int(shape[0]), int(shape[1]))
    expected_shape = (variant.matrix_dim, variant.matrix_dim)
    if matrix_shape != expected_shape:
        raise ValueError(f"{label} matrix shape mismatch: expected {expected_shape}, got {matrix_shape}")

    if not np.issubdtype(row.dtype, np.integer) or not np.issubdtype(col.dtype, np.integer):
        raise ValueError(f"{label} row/col must be integer arrays")
    if np.any(row < 0) or np.any(col < 0) or np.any(row >= variant.matrix_dim) or np.any(col >= variant.matrix_dim):
        raise ValueError(f"{label} row/col indices out of range")
    if not np.issubdtype(data_real.dtype, np.number) or not np.issubdtype(data_imag.dtype, np.number):
        raise ValueError(f"{label} data arrays must be numeric")
    if not np.all(np.isfinite(data_real)) or not np.all(np.isfinite(data_imag)):
        raise ValueError(f"{label} data arrays contain non-finite values")

    data = data_real.astype(np.float64) + 1j * data_imag.astype(np.float64)
    diag_mask = row == col
    trace = float(np.sum(data[diag_mask]).real)
    if not math.isclose(trace, 1.0, rel_tol=0.0, abs_tol=TRACE_TOLERANCE):
        raise ValueError(f"{label} trace is {trace:.8f}, expected 1.0 +/- {TRACE_TOLERANCE}")
    return row.astype(np.int64), col.astype(np.int64), data, matrix_shape, trace


def _coo_to_dense_on_union_support(
    candidate_triplet: tuple[np.ndarray, np.ndarray, np.ndarray],
    reference_triplet: tuple[np.ndarray, np.ndarray, np.ndarray],
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    cand_row, cand_col, cand_data = candidate_triplet
    ref_row, ref_col, ref_data = reference_triplet
    support = np.unique(
        np.concatenate((cand_row, cand_col, ref_row, ref_col), dtype=np.int64)
    )
    support_map = np.full(dim, -1, dtype=np.int64)
    support_map[support] = np.arange(support.shape[0], dtype=np.int64)

    candidate = np.zeros((support.shape[0], support.shape[0]), dtype=np.complex128)
    reference = np.zeros((support.shape[0], support.shape[0]), dtype=np.complex128)
    np.add.at(candidate, (support_map[cand_row], support_map[cand_col]), cand_data)
    np.add.at(reference, (support_map[ref_row], support_map[ref_col]), ref_data)
    return candidate, reference


def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    hermitian = (matrix + matrix.conj().T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(hermitian)
    clipped = np.clip(eigvals.real, 0.0, None)
    return (eigvecs * np.sqrt(clipped)) @ eigvecs.conj().T


def _fidelity(candidate: np.ndarray, reference: np.ndarray) -> float:
    sqrt_candidate = _matrix_sqrt_psd(candidate)
    product = sqrt_candidate @ reference @ sqrt_candidate
    product = (product + product.conj().T) / 2.0
    eigvals = np.linalg.eigvalsh(product)
    eigvals = np.clip(eigvals.real, 0.0, None)
    return float(np.square(np.sum(np.sqrt(eigvals))).real)


def score_submission_bytes(
    *,
    variant: VariantSpec,
    candidate_summary_blob: bytes,
    candidate_density_payloads: dict[str, bytes],
    reference_summary_blob: bytes,
    reference_density_payloads: dict[str, bytes],
) -> ScoreResult:
    metrics: dict[str, Any] = {"variant_name": variant.variant_name}

    try:
        candidate_summary = _load_json(candidate_summary_blob, label="candidate summary.json")
        _validate_summary(candidate_summary, variant)
        reference_summary = _load_json(reference_summary_blob, label="reference summary.json")
        _validate_summary(reference_summary, variant)
    except ValueError as exc:
        return _fail(str(exc), **metrics)

    missing_candidate = sorted(set(variant.density_filenames) - set(candidate_density_payloads))
    if missing_candidate:
        return _fail(f"missing candidate density files: {missing_candidate[:5]}", **metrics)
    missing_reference = sorted(set(variant.density_filenames) - set(reference_density_payloads))
    if missing_reference:
        return _fail(f"missing reference density files: {missing_reference[:5]}", **metrics)

    infidelities: list[float] = []
    traces: list[float] = []

    for density_name in variant.density_filenames:
        try:
            candidate_payload = _load_npz(
                candidate_density_payloads[density_name],
                label=f"candidate {density_name}",
            )
            reference_payload = _load_npz(
                reference_density_payloads[density_name],
                label=f"reference {density_name}",
            )
            cand_row, cand_col, cand_data, _, cand_trace = _validate_density_payload(
                candidate_payload,
                label=f"candidate {density_name}",
                variant=variant,
            )
            ref_row, ref_col, ref_data, _, _ = _validate_density_payload(
                reference_payload,
                label=f"reference {density_name}",
                variant=variant,
            )
        except ValueError as exc:
            return _fail(str(exc), density_name=density_name, **metrics)

        candidate_dense, reference_dense = _coo_to_dense_on_union_support(
            (cand_row, cand_col, cand_data),
            (ref_row, ref_col, ref_data),
            variant.matrix_dim,
        )
        fidelity = min(max(_fidelity(candidate_dense, reference_dense), 0.0), 1.0)
        infidelity = 1.0 - fidelity
        infidelities.append(float(infidelity))
        traces.append(float(cand_trace))

    average_infidelity = float(np.mean(infidelities)) if infidelities else 1.0
    worst_index = int(np.argmax(infidelities)) if infidelities else -1
    worst_infidelity = float(max(infidelities)) if infidelities else 1.0

    metrics.update(
        {
            "average_infidelity": average_infidelity,
            "worst_state": variant.density_filenames[worst_index] if worst_index >= 0 else None,
            "worst_infidelity": worst_infidelity,
            "min_trace": float(min(traces)) if traces else None,
            "max_trace": float(max(traces)) if traces else None,
        }
    )

    if average_infidelity <= PERFECT_THRESHOLD:
        return ScoreResult(
            score=1.0,
            passed=True,
            reason="average_infidelity within threshold",
            metrics=metrics,
        )

    score = math.exp(-(average_infidelity - PERFECT_THRESHOLD) / PERFECT_THRESHOLD)
    return ScoreResult(
        score=float(score),
        passed=False,
        reason="average_infidelity above threshold",
        metrics=metrics,
    )
