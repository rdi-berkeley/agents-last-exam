"""Scorer for computing_math/linreg_optimal_estimator."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

RISK_RATIO_TOLERANCE = 1.5


def _read_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _read_vector_bytes(data: bytes, label: str) -> np.ndarray:
    try:
        arr = np.loadtxt(io.BytesIO(data), delimiter=",")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{label} could not be parsed as numeric CSV: {exc}") from exc
    return np.asarray(arr, dtype=float).reshape(-1)


def _read_matrix_bytes(data: bytes, label: str) -> np.ndarray:
    try:
        arr = np.loadtxt(io.BytesIO(data), delimiter=",")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{label} could not be parsed as numeric CSV: {exc}") from exc
    return np.asarray(arr, dtype=float)


def population_excess_risk(
    w_hat: np.ndarray,
    w_star: np.ndarray,
    eigenbasis: np.ndarray,
    lambdas: np.ndarray,
) -> float:
    diff = w_hat.reshape(-1) - w_star.reshape(-1)
    coords = eigenbasis.T @ diff
    return float(np.sum(lambdas.reshape(-1) * coords**2))


def score_submission_bytes(
    *,
    submission_w_hat: bytes | None,
    submission_json: bytes | None,
    metadata_json: bytes,
    lambdas_csv: bytes,
    hidden_u_csv: bytes,
    w_star_csv: bytes,
    reference_output_json: bytes | None = None,
    reference_w_hat_csv: bytes | None = None,
    risk_ratio_tolerance: float = RISK_RATIO_TOLERANCE,
) -> dict[str, Any]:
    """Return a verifier-style report plus a binary score."""
    report: dict[str, Any] = {
        "passed": False,
        "score": 0.0,
        "errors": [],
        "risk_ratio_tolerance": float(risk_ratio_tolerance),
    }

    try:
        if submission_w_hat is None:
            raise ValueError("Missing submission_w_hat.csv")
        if submission_json is None:
            raise ValueError("Missing submission.json")

        metadata = _read_json_bytes(metadata_json, "instance_metadata.json")
        submission_meta = _read_json_bytes(submission_json, "submission.json")
        estimator_name = submission_meta.get("estimator_name")
        if not isinstance(estimator_name, str) or not estimator_name.strip():
            raise ValueError("submission.json must include a nonempty estimator_name string")

        d = int(metadata["d"])
        lambdas = _read_vector_bytes(lambdas_csv, "lambdas.csv")
        w_star = _read_vector_bytes(w_star_csv, "w_star.csv")
        eigenbasis = _read_matrix_bytes(hidden_u_csv, "U_hidden.csv")
        w_hat = _read_vector_bytes(submission_w_hat, "submission_w_hat.csv")

        if w_hat.shape[0] != d:
            raise ValueError(f"submission_w_hat.csv has length {w_hat.shape[0]}, expected d={d}")
        if lambdas.shape[0] != d:
            raise ValueError(f"lambdas.csv has length {lambdas.shape[0]}, expected d={d}")
        if w_star.shape[0] != d:
            raise ValueError(f"w_star.csv has length {w_star.shape[0]}, expected d={d}")
        if eigenbasis.shape != (d, d):
            raise ValueError(f"U_hidden.csv has shape {eigenbasis.shape}, expected {(d, d)}")
        if not np.all(np.isfinite(w_hat)):
            raise ValueError("submission_w_hat.csv contains NaN or infinite values")

        risk = population_excess_risk(w_hat, w_star, eigenbasis, lambdas)
        null_risk = population_excess_risk(np.zeros_like(w_star), w_star, eigenbasis, lambdas)

        reference_risk: float | None = None
        if reference_output_json is not None:
            ref_payload = _read_json_bytes(reference_output_json, "reference_output.json")
            raw_ref_risk = ref_payload.get("population_excess_risk")
            if raw_ref_risk is not None:
                reference_risk = float(raw_ref_risk)
        if reference_risk is None:
            if reference_w_hat_csv is None:
                raise ValueError("Missing both reference_output.json risk and reference_w_hat.csv")
            ref_w_hat = _read_vector_bytes(reference_w_hat_csv, "reference_w_hat.csv")
            reference_risk = population_excess_risk(ref_w_hat, w_star, eigenbasis, lambdas)

        threshold = max(risk_ratio_tolerance * reference_risk, 1e-10)
        passed = bool(risk <= threshold)
        report.update(
            {
                "submission_estimator_name": estimator_name,
                "population_excess_risk": risk,
                "reference_population_excess_risk": reference_risk,
                "null_estimator_excess_risk": null_risk,
                "risk_ratio_to_reference": risk / reference_risk if reference_risk > 0 else None,
                "threshold": threshold,
                "passed": passed,
                "score": 1.0 if passed else 0.0,
            }
        )
    except Exception as exc:  # noqa: BLE001
        report["errors"].append(str(exc))

    return report


def _read_optional(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def score_submission_paths(public_dir: Path, reference_dir: Path, submission_dir: Path) -> dict[str, Any]:
    return score_submission_bytes(
        submission_w_hat=_read_optional(submission_dir / "submission_w_hat.csv"),
        submission_json=_read_optional(submission_dir / "submission.json"),
        metadata_json=(public_dir / "instance_metadata.json").read_bytes(),
        lambdas_csv=(public_dir / "lambdas.csv").read_bytes(),
        hidden_u_csv=(reference_dir / "U_hidden.csv").read_bytes(),
        w_star_csv=(reference_dir / "w_star.csv").read_bytes(),
        reference_output_json=_read_optional(reference_dir / "reference_output.json"),
        reference_w_hat_csv=_read_optional(reference_dir / "reference_w_hat.csv"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score linreg_optimal_estimator output.")
    parser.add_argument("--public-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = score_submission_paths(args.public_dir, args.reference_dir, args.submission_dir)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
