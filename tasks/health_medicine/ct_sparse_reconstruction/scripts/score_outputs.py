"""Scorer for the CT sparse reconstruction benchmark."""

from __future__ import annotations

import io
import json
import math
from typing import Any

import numpy as np


TIER_SPECS = {
    "tier1": {
        "shape": (128, 128),
        "threshold": 20.0,
        "path": "reconstructions/tier1_recon.npy",
        "required": True,
    },
    "tier2": {
        "shape": (128, 128),
        "threshold": 20.0,
        "path": "reconstructions/tier2_recon.npy",
        "required": True,
    },
    "tier3": {
        "shape": (128, 128),
        "threshold": 22.0,
        "path": "reconstructions/tier3_recon.npy",
        "required": False,
    },
}


def compute_psnr(reference: np.ndarray, recon: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse <= 1e-15:
        return float("inf")
    peak = float(np.max(np.abs(reference)))
    if peak <= 0.0:
        return 0.0
    return float(20.0 * np.log10(peak / math.sqrt(mse)))


def _load_npy(raw: bytes, label: str) -> np.ndarray:
    if not raw:
        raise ValueError(f"{label} is empty")
    try:
        return np.load(io.BytesIO(raw))
    except Exception as exc:  # pragma: no cover - defensive surface for VM data
        raise ValueError(f"{label} is not a readable .npy file: {exc}") from exc


def _load_results(raw: bytes) -> dict[str, Any]:
    if not raw:
        raise ValueError("results.json is empty")
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"results.json is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("results.json must contain a JSON object")
    return data


def _validate_array(arr: np.ndarray, expected_shape: tuple[int, int], label: str) -> str | None:
    if arr.shape != expected_shape:
        return f"{label} shape mismatch: expected {expected_shape}, got {arr.shape}"
    if not np.issubdtype(arr.dtype, np.number):
        return f"{label} has non-numeric dtype {arr.dtype}"
    if not np.all(np.isfinite(arr)):
        return f"{label} contains non-finite values"
    return None


def score_submission_bytes(
    *,
    result_bytes: bytes,
    reconstruction_bytes: dict[str, bytes],
    reference_bytes: dict[str, bytes],
) -> tuple[float, dict[str, Any]]:
    """Return an AgentHLE-normalized score and a detailed report."""

    errors: list[str] = []
    warnings: list[str] = []
    tiers: dict[str, Any] = {}

    try:
        results = _load_results(result_bytes)
    except ValueError as exc:
        return 0.0, {"passed": False, "errors": [str(exc)], "tiers": {}}

    tier2_report = results.get("tier2", {})
    fbp_psnr = None
    if isinstance(tier2_report, dict):
        fbp_psnr = tier2_report.get("fbp_psnr_db")
    try:
        fbp_psnr_float = float(fbp_psnr)
    except (TypeError, ValueError):
        fbp_psnr_float = float("nan")
    tier2_fbp_claim_ok = math.isfinite(fbp_psnr_float) and fbp_psnr_float < 20.0
    if not tier2_fbp_claim_ok:
        errors.append("results.json must report tier2.fbp_psnr_db as a finite value below 20.0")

    for tier_name, spec in TIER_SPECS.items():
        try:
            recon = _load_npy(reconstruction_bytes.get(tier_name, b""), f"{tier_name} reconstruction")
            reference = _load_npy(reference_bytes.get(tier_name, b""), f"{tier_name} reference")
        except ValueError as exc:
            if spec.get("required", True):
                errors.append(str(exc))
            else:
                warnings.append(str(exc))
            tiers[tier_name] = {"passed": False, "error": str(exc)}
            continue

        shape_error = _validate_array(recon, spec["shape"], f"{tier_name} reconstruction")
        reference_error = _validate_array(reference, spec["shape"], f"{tier_name} reference")
        if shape_error or reference_error:
            message = shape_error or reference_error or "invalid array"
            if spec.get("required", True):
                errors.append(message)
            else:
                warnings.append(message)
            tiers[tier_name] = {"passed": False, "error": message}
            continue

        psnr = compute_psnr(reference, recon)
        passed = bool(psnr > float(spec["threshold"]))
        tiers[tier_name] = {
            "passed": passed,
            "psnr_db": psnr,
            "threshold_db": float(spec["threshold"]),
            "shape": list(recon.shape),
        }
        if not passed:
            message = f"{tier_name} PSNR {psnr:.3f} dB does not exceed {float(spec['threshold']):.1f} dB"
            if spec.get("required", True):
                errors.append(message)
            else:
                warnings.append(message)

    tier1_ok = bool(tiers.get("tier1", {}).get("passed"))
    tier2_ok = bool(tiers.get("tier2", {}).get("passed")) and tier2_fbp_claim_ok
    tier3_ok = bool(tiers.get("tier3", {}).get("passed"))

    if tier1_ok and tier2_ok and tier3_ok:
        score = 1.0
    elif tier1_ok and tier2_ok:
        score = 0.7
    else:
        score = 0.0

    report = {
        "passed": score == 1.0,
        "score": score,
        "tiers": tiers,
        "tier2_fbp_claim_ok": tier2_fbp_claim_ok,
        "reported_tier2_fbp_psnr_db": fbp_psnr_float if math.isfinite(fbp_psnr_float) else None,
        "errors": errors,
        "warnings": warnings,
    }
    return score, report


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()

    reconstruction_bytes = {}
    reference_bytes = {}
    for tier_name, spec in TIER_SPECS.items():
        recon_path = args.output_dir / spec["path"]
        ref_path = args.reference_dir / "phantoms" / f"{tier_name}_phantom.npy"
        reconstruction_bytes[tier_name] = recon_path.read_bytes() if recon_path.exists() else b""
        reference_bytes[tier_name] = ref_path.read_bytes() if ref_path.exists() else b""

    results_path = args.output_dir / "results.json"
    score, report = score_submission_bytes(
        result_bytes=results_path.read_bytes() if results_path.exists() else b"",
        reconstruction_bytes=reconstruction_bytes,
        reference_bytes=reference_bytes,
    )
    print(json.dumps({"normalized_score": score, **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
