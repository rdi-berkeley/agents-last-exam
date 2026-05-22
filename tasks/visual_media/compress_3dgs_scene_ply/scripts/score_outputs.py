#!/usr/bin/env python
"""Local scorer for compress_3dgs_scene_ply."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class ScoreResult:
    score: float
    passed: bool
    reason: str
    hard_gate: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hard_fail(reason: str, details: dict[str, Any] | None = None) -> ScoreResult:
    return ScoreResult(
        score=0.0,
        passed=False,
        reason=reason,
        hard_gate=reason,
        details=details or {},
    )


def load_json_bytes(payload: str | bytes, *, label: str) -> dict[str, Any]:
    text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
    try:
        parsed = json.loads(text)
    except Exception as exc:
        raise ValueError(f"{label} unreadable: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to an object")
    return parsed


def count_ply_vertices(payload: bytes) -> int:
    for raw_line in payload.splitlines():
        line = raw_line.decode("ascii", errors="ignore").strip()
        if line.startswith("element vertex "):
            return int(line.split()[-1])
        if line == "end_header":
            break
    raise ValueError("ply vertex count missing")


def decode_rgb(payload: bytes) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def grayscale(image: np.ndarray) -> np.ndarray:
    return (
        0.299 * image[:, :, 0].astype(np.float32)
        + 0.587 * image[:, :, 1].astype(np.float32)
        + 0.114 * image[:, :, 2].astype(np.float32)
    ) / 255.0


def gaussian_blur(image: np.ndarray, radius: float = 1.5) -> np.ndarray:
    del radius
    kernel = np.array(
        [0.00102838, 0.00759876, 0.03600077, 0.10936069, 0.21300554, 0.26601172,
         0.21300554, 0.10936069, 0.03600077, 0.00759876, 0.00102838],
        dtype=np.float32,
    )
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, image)
    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, blurred)
    return blurred.astype(np.float32)


def ssim_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_ref = gaussian_blur(reference)
    mu_cand = gaussian_blur(candidate)
    mu_ref_sq = mu_ref * mu_ref
    mu_cand_sq = mu_cand * mu_cand
    mu_mix = mu_ref * mu_cand
    sigma_ref_sq = gaussian_blur(reference * reference) - mu_ref_sq
    sigma_cand_sq = gaussian_blur(candidate * candidate) - mu_cand_sq
    sigma_mix = gaussian_blur(reference * candidate) - mu_mix
    numerator = (2 * mu_mix + c1) * (2 * sigma_mix + c2)
    denominator = (mu_ref_sq + mu_cand_sq + c1) * (sigma_ref_sq + sigma_cand_sq + c2)
    return float(np.clip(np.mean(numerator / denominator), 0.0, 1.0))


def compute_metrics(candidate_bytes: bytes, reference_bytes: bytes) -> dict[str, float]:
    candidate = decode_rgb(candidate_bytes)
    reference = decode_rgb(reference_bytes)
    ref_gray = grayscale(reference)
    cand_gray = grayscale(candidate)
    diff = ref_gray - cand_gray
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse == 0 else float(20.0 * np.log10(1.0 / np.sqrt(mse)))
    ssim = ssim_score(ref_gray, cand_gray)
    return {"psnr": psnr, "ssim": ssim}


def _approx_equal(lhs: float, rhs: float, tol: float) -> bool:
    return abs(lhs - rhs) <= tol


def score_output_bundle(
    *,
    output_vertex_count: int,
    results_json: str | bytes,
    compression_report_json: str | bytes,
    eval_contract_json: str | bytes,
    rendered_images: dict[str, bytes],
    reference_images: dict[str, bytes],
) -> ScoreResult:
    try:
        contract = load_json_bytes(eval_contract_json, label="eval_contract.json")
        results = load_json_bytes(results_json, label="results.json")
        compression_report = load_json_bytes(compression_report_json, label="compression_report.json")
    except ValueError as exc:
        return hard_fail(str(exc))

    output_count = output_vertex_count

    key = f"ours_{contract['iteration']}"
    if key not in results or not isinstance(results[key], dict):
        return hard_fail("missing_results_key", {"expected_key": key})
    payload = results[key]

    required_avg_fields = ("PSNR", "SSIM", "LPIPS")
    missing = [field for field in required_avg_fields if field not in payload]
    if missing:
        return hard_fail("missing_result_fields", {"missing": missing})

    try:
        reported_psnr = float(payload["PSNR"])
        reported_ssim = float(payload["SSIM"])
        reported_lpips = float(payload["LPIPS"])
    except Exception as exc:
        return hard_fail("reported_metric_not_numeric", {"error": str(exc)})

    if not math.isfinite(reported_psnr) or not math.isfinite(reported_ssim):
        return hard_fail("reported_metric_not_finite")
    if not math.isfinite(reported_lpips) or reported_lpips < 0.0:
        return hard_fail("reported_lpips_invalid")

    holdout_names = contract["holdout_filenames"]
    missing_renders = [name for name in holdout_names if name not in rendered_images]
    missing_reference = [name for name in holdout_names if name not in reference_images]
    if missing_renders:
        return hard_fail("missing_rendered_test_views", {"missing": missing_renders})
    if missing_reference:
        return hard_fail("missing_reference_test_views", {"missing": missing_reference})

    original_count = int(contract["original_gaussian_count"])
    reduction_fraction = 1.0 - (output_count / original_count)
    min_reduction = float(contract["min_reduction_fraction"])
    if reduction_fraction < min_reduction:
        return hard_fail(
            "insufficient_gaussian_reduction",
            {
                "original_gaussian_count": original_count,
                "output_gaussian_count": output_count,
                "reduction_fraction": reduction_fraction,
                "required_min_reduction_fraction": min_reduction,
            },
        )

    if compression_report.get("gaussian_count_before") != original_count:
        return hard_fail(
            "compression_report_before_count_mismatch",
            {"reported": compression_report.get("gaussian_count_before"), "expected": original_count},
        )
    if compression_report.get("gaussian_count_after") != output_count:
        return hard_fail(
            "compression_report_after_count_mismatch",
            {"reported": compression_report.get("gaussian_count_after"), "expected": output_count},
        )

    computed_psnr = []
    computed_ssim = []
    per_view_details = {}
    for name in holdout_names:
        metrics = compute_metrics(rendered_images[name], reference_images[name])
        computed_psnr.append(metrics["psnr"])
        computed_ssim.append(metrics["ssim"])
        per_view_details[name] = metrics

    avg_psnr = float(np.mean(computed_psnr))
    avg_ssim = float(np.mean(computed_ssim))

    baseline = contract["baseline_results"]
    min_psnr = float(baseline["PSNR"]) - float(contract["psnr_drop_limit_db"])
    min_ssim = float(baseline["SSIM"]) * float(contract["ssim_factor"])
    if avg_psnr < min_psnr:
        return hard_fail("psnr_below_threshold", {"computed": avg_psnr, "required_min": min_psnr})
    if avg_ssim < min_ssim:
        return hard_fail("ssim_below_threshold", {"computed": avg_ssim, "required_min": min_ssim})

    return ScoreResult(
        score=1.0,
        passed=True,
        reason="passed",
        hard_gate=None,
        details={
            "iteration": contract["iteration"],
            "original_gaussian_count": original_count,
            "output_gaussian_count": output_count,
            "reduction_fraction": reduction_fraction,
            "average_psnr": avg_psnr,
            "average_ssim": avg_ssim,
            "reported_psnr": reported_psnr,
            "reported_ssim": reported_ssim,
            "reported_lpips": reported_lpips,
            "per_view": per_view_details,
            "thresholds": {"min_psnr": min_psnr, "min_ssim": min_ssim},
        },
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_dir)
    contract = load_json_bytes(_read_text(reference_dir / "eval_contract.json"), label="eval_contract.json")
    rendered = {
        name: _read_bytes(output_dir / "rendered_test_views" / name)
        for name in contract["holdout_filenames"]
        if (output_dir / "rendered_test_views" / name).exists()
    }
    reference = {
        name: _read_bytes(reference_dir / "test_images" / name)
        for name in contract["holdout_filenames"]
        if (reference_dir / "test_images" / name).exists()
    }
    ply_path = output_dir / f"point_cloud_{contract['iteration']}_compressed.ply"
    vertex_count = count_ply_vertices(_read_bytes(ply_path))
    result = score_output_bundle(
        output_vertex_count=vertex_count,
        results_json=_read_text(output_dir / "results.json"),
        compression_report_json=_read_text(output_dir / "compression_report.json"),
        eval_contract_json=_read_text(reference_dir / "eval_contract.json"),
        rendered_images=rendered,
        reference_images=reference,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
