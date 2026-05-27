from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any

import numpy as np

REQUIRED_FILES = (
    "results.json",
    "tier1_decoded.npy",
    "tier2_decoded.npy",
    "ber_curve.npy",
)

EXPECTED_TIER1_SHAPE = (100, 19)
EXPECTED_TIER1_DTYPE = np.dtype(np.int32)
EXPECTED_TIER2_SHAPE = (10000, 223)
EXPECTED_TIER2_DTYPE = np.dtype(np.uint8)
EXPECTED_BER_SHAPE = (12, 2)
EXPECTED_BER_DTYPE = np.dtype(np.float64)
EXPECTED_SNR_POINTS = np.arange(12, dtype=np.float64)
TIER3_ABS_TOL = np.array(
    [2e-3, 2e-3, 2e-3, 1e-3, 1e-3, 5e-4, 1e-5, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8],
    dtype=np.float64,
)


@dataclass
class ScoreReport:
    score: float
    missing_files: list[str]
    tier1_pass: bool
    tier2_pass: bool
    tier3_pass: bool
    tier1_correct_rows: int
    tier2_correct_rows: int
    tier3_max_abs_error: float | None
    results_json_valid: bool
    tier1_results_ok: bool
    tier2_results_ok: bool
    tier3_results_ok: bool
    results_json_errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_npy(payload: bytes) -> np.ndarray:
    return np.load(BytesIO(payload), allow_pickle=False)


def _parse_results_json(payload: bytes) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        return None, [f"results.json is not valid UTF-8 JSON: {exc}"]

    errors: list[str] = []
    for key in ("tier1", "tier2", "tier3"):
        if not isinstance(data.get(key), dict):
            errors.append(f"results.json missing object field {key!r}")
    return data if not errors else None, errors


def _validate_results_json(
    results_data: dict[str, Any],
    tier1_correct_rows: int,
    tier2_correct_rows: int,
    ber_curve: np.ndarray | None,
) -> tuple[dict[str, list[str]], list[str]]:
    per_tier_errors: dict[str, list[str]] = {"tier1": [], "tier2": [], "tier3": []}
    general_errors: list[str] = []

    tier1 = results_data["tier1"]
    if tier1.get("num_vectors") != 100:
        per_tier_errors["tier1"].append("tier1.num_vectors must equal 100")
    if tier1.get("num_decoded_successfully") != tier1_correct_rows:
        per_tier_errors["tier1"].append("tier1.num_decoded_successfully does not match actual decoded rows")
    if tier1.get("all_correct") is not (tier1_correct_rows == 100):
        per_tier_errors["tier1"].append("tier1.all_correct does not match actual decoded rows")

    tier2 = results_data["tier2"]
    if tier2.get("num_codewords") != 10000:
        per_tier_errors["tier2"].append("tier2.num_codewords must equal 10000")
    if tier2.get("num_decoded_successfully") != tier2_correct_rows:
        per_tier_errors["tier2"].append("tier2.num_decoded_successfully does not match actual decoded rows")
    if tier2.get("all_correct") is not (tier2_correct_rows == 10000):
        per_tier_errors["tier2"].append("tier2.all_correct does not match actual decoded rows")

    tier3 = results_data["tier3"]
    if tier3.get("snr_points_db") != EXPECTED_SNR_POINTS.astype(int).tolist():
        per_tier_errors["tier3"].append("tier3.snr_points_db must list 0..11")
    if tier3.get("codewords_per_snr") != 50000:
        per_tier_errors["tier3"].append("tier3.codewords_per_snr must equal 50000")
    if tier3.get("seed") != 54321:
        per_tier_errors["tier3"].append("tier3.seed must equal 54321")
    if tier3.get("total_info_bits_per_snr") != 89200000:
        per_tier_errors["tier3"].append("tier3.total_info_bits_per_snr must equal 89200000")
    ber_values = tier3.get("ber_values")
    if not isinstance(ber_values, list) or len(ber_values) != 12:
        per_tier_errors["tier3"].append("tier3.ber_values must be a list of length 12")
    elif ber_curve is None:
        per_tier_errors["tier3"].append("tier3.ber_values cannot be verified because ber_curve.npy is malformed")
    else:
        if not np.allclose(np.asarray(ber_values, dtype=np.float64), ber_curve[:, 1], atol=1e-12):
            per_tier_errors["tier3"].append("tier3.ber_values does not match ber_curve.npy")

    return per_tier_errors, general_errors


def score_submission(
    agent_outputs: dict[str, bytes],
    reference_outputs: dict[str, bytes],
) -> ScoreReport:
    missing_files = [name for name in REQUIRED_FILES if name not in agent_outputs]
    if missing_files:
        return ScoreReport(
            score=0.0,
            missing_files=missing_files,
            tier1_pass=False,
            tier2_pass=False,
            tier3_pass=False,
            tier1_correct_rows=0,
            tier2_correct_rows=0,
            tier3_max_abs_error=None,
            results_json_valid=False,
            tier1_results_ok=False,
            tier2_results_ok=False,
            tier3_results_ok=False,
            results_json_errors=["missing required output files"],
        )

    try:
        tier1_agent = _load_npy(agent_outputs["tier1_decoded.npy"])
        tier2_agent = _load_npy(agent_outputs["tier2_decoded.npy"])
        ber_agent = _load_npy(agent_outputs["ber_curve.npy"])
        tier1_ref = _load_npy(reference_outputs["tier1_decoded.npy"])
        tier2_ref = _load_npy(reference_outputs["tier2_decoded.npy"])
        ber_ref = _load_npy(reference_outputs["ber_curve.npy"])
    except Exception as exc:
        return ScoreReport(
            score=0.0,
            missing_files=[],
            tier1_pass=False,
            tier2_pass=False,
            tier3_pass=False,
            tier1_correct_rows=0,
            tier2_correct_rows=0,
            tier3_max_abs_error=None,
            results_json_valid=False,
            tier1_results_ok=False,
            tier2_results_ok=False,
            tier3_results_ok=False,
            results_json_errors=[f"failed to load numpy outputs: {exc}"],
        )

    tier1_correct_rows = 0
    tier1_pass = False
    if tier1_agent.shape == EXPECTED_TIER1_SHAPE and tier1_agent.dtype == EXPECTED_TIER1_DTYPE:
        tier1_correct_rows = int(np.all(tier1_agent.astype(np.int64) == tier1_ref.astype(np.int64), axis=1).sum())
        tier1_pass = tier1_correct_rows == 100

    tier2_correct_rows = 0
    tier2_pass = False
    if tier2_agent.shape == EXPECTED_TIER2_SHAPE and tier2_agent.dtype == EXPECTED_TIER2_DTYPE:
        tier2_correct_rows = int(np.all(tier2_agent.astype(np.int64) == tier2_ref.astype(np.int64), axis=1).sum())
        tier2_pass = tier2_correct_rows == 10000

    tier3_max_abs_error: float | None = None
    tier3_pass = False
    if ber_agent.shape == EXPECTED_BER_SHAPE and ber_agent.dtype == EXPECTED_BER_DTYPE:
        snr_ok = np.allclose(ber_agent[:, 0], EXPECTED_SNR_POINTS, rtol=0.0, atol=0.0)
        ber_vals = ber_agent[:, 1].astype(np.float64)
        ref_vals = ber_ref[:, 1].astype(np.float64)
        tier3_max_abs_error = float(np.max(np.abs(ber_vals - ref_vals)))
        tol_ok = np.all(np.abs(ber_vals - ref_vals) <= TIER3_ABS_TOL)
        shape_ok = bool(
            np.all(ber_vals >= 0.0)
            and np.all(ber_vals <= 1.0)
            and np.all(np.diff(ber_vals) <= 1e-12)
        )
        tier3_pass = bool(snr_ok and tol_ok and shape_ok)

    results_data, results_errors = _parse_results_json(agent_outputs["results.json"])
    per_tier_results_errors = {"tier1": [], "tier2": [], "tier3": []}
    if results_data is not None:
        per_tier_results_errors, general_errors = _validate_results_json(
            results_data,
            tier1_correct_rows=tier1_correct_rows,
            tier2_correct_rows=tier2_correct_rows,
            ber_curve=ber_agent if ber_agent.shape == EXPECTED_BER_SHAPE and ber_agent.dtype == EXPECTED_BER_DTYPE else None,
        )
        results_errors.extend(general_errors)
        for tier_name in ("tier1", "tier2", "tier3"):
            results_errors.extend(per_tier_results_errors[tier_name])

    results_json_valid = not results_errors
    tier1_results_ok = not per_tier_results_errors["tier1"] and results_data is not None
    tier2_results_ok = not per_tier_results_errors["tier2"] and results_data is not None
    tier3_results_ok = not per_tier_results_errors["tier3"] and results_data is not None
    tier1_pass = tier1_pass and tier1_results_ok
    tier2_pass = tier2_pass and tier2_results_ok
    tier3_pass = tier3_pass and tier3_results_ok

    if tier1_pass and tier2_pass and tier3_pass:
        score = 1.0
    elif tier1_pass and tier2_pass:
        score = 0.5
    else:
        score = 0.0

    return ScoreReport(
        score=score,
        missing_files=[],
        tier1_pass=tier1_pass,
        tier2_pass=tier2_pass,
        tier3_pass=tier3_pass,
        tier1_correct_rows=tier1_correct_rows,
        tier2_correct_rows=tier2_correct_rows,
        tier3_max_abs_error=tier3_max_abs_error,
        results_json_valid=results_json_valid,
        tier1_results_ok=tier1_results_ok,
        tier2_results_ok=tier2_results_ok,
        tier3_results_ok=tier3_results_ok,
        results_json_errors=results_errors,
    )
