"""Scoring logic for business_finance/american_option_pricing_ls."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

REQUIRED_OUTPUT_FILES = ("results.json", "exercise_boundary_tier2.npy")
REQUIRED_REFERENCE_FILES = ("results.json", "exercise_boundary_tier2.npy")

TIER1_TOLERANCES = {
    "bs_price": 1e-4,
    "bs_delta": 1e-4,
    "bs_gamma": 1e-4,
    "bs_vega": 1e-2,
    "bs_theta": 1e-4,
}
TIER2_PRICE_TOLERANCE = 0.20
TIER3_PRICE_TOLERANCE = 0.30
GREEK_FACTOR_LIMIT = 3.0
BOUNDARY_MIN = 70.0
BOUNDARY_MAX = 115.0


@dataclass
class TierReport:
    passed: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "metrics": self.metrics,
            "notes": self.notes,
        }


@dataclass
class ScoreReport:
    score: float
    results_json_present: bool
    boundary_present: bool
    results_json_valid: bool
    tier1: TierReport
    tier2: TierReport
    tier3: TierReport
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "results_json_present": self.results_json_present,
            "boundary_present": self.boundary_present,
            "results_json_valid": self.results_json_valid,
            "tier1": self.tier1.to_dict(),
            "tier2": self.tier2.to_dict(),
            "tier3": self.tier3.to_dict(),
            "notes": self.notes,
        }


def _load_json(blob: bytes) -> dict[str, Any]:
    return json.loads(blob.decode("utf-8"))


def _load_npy(blob: bytes) -> np.ndarray:
    with io.BytesIO(blob) as handle:
        return np.load(handle, allow_pickle=False)


def _ok(reason: str, **metrics: Any) -> TierReport:
    return TierReport(True, reason, metrics=metrics)


def _fail(reason: str, **metrics: Any) -> TierReport:
    return TierReport(False, reason, metrics=metrics)


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _float_list(value: Any, label: str, expected_len: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_len:
        raise ValueError(f"{label} must be a list of length {expected_len}")
    return [_float(item, f"{label}[{idx}]") for idx, item in enumerate(value)]


def _validate_constants(section: dict[str, Any], expected: dict[str, Any], prefix: str) -> None:
    for key, expected_value in expected.items():
        if key not in section:
            raise ValueError(f"missing {prefix}.{key}")
        actual = section[key]
        if isinstance(expected_value, float):
            if abs(_float(actual, f"{prefix}.{key}") - expected_value) > 1e-9:
                raise ValueError(f"{prefix}.{key} must equal {expected_value}")
        else:
            if actual != expected_value:
                raise ValueError(f"{prefix}.{key} must equal {expected_value}")


def _validate_tier1(results: dict[str, Any], reference: dict[str, Any]) -> TierReport:
    try:
        tier = results["tier1"]
        ref = reference["tier1"]
    except KeyError as exc:
        return _fail(f"missing {exc.args[0]} in tier1")

    try:
        _validate_constants(
            tier,
            {"S0": 100.0, "K": 100.0, "r": 0.05, "sigma": 0.2, "T": 1.0, "mc_n_paths": 100000},
            "tier1",
        )
        metric_errors: dict[str, float] = {}
        for key, tol in TIER1_TOLERANCES.items():
            actual = _float(tier[key], f"tier1.{key}")
            target = _float(ref[key], f"reference.tier1.{key}")
            error = abs(actual - target)
            metric_errors[f"{key}_abs_error"] = error
            if error > tol:
                return _fail(f"tier1 {key} differs from reference by {error:.6g} > {tol}", **metric_errors)

        mc_price = _float(tier["mc_price"], "tier1.mc_price")
        mc_stderr = _float(tier["mc_stderr"], "tier1.mc_stderr")
        if mc_stderr <= 0:
            return _fail("tier1.mc_stderr must be positive")
        if not _bool(tier["mc_within_2se"], "tier1.mc_within_2se"):
            return _fail("tier1.mc_within_2se must be true")

        bs_price = _float(tier["bs_price"], "tier1.bs_price")
        mc_gap = abs(mc_price - bs_price)
        if mc_gap > 2.0 * mc_stderr + 1e-12:
            return _fail("tier1.mc_price is not within 2 standard errors of tier1.bs_price", mc_gap=mc_gap)

        return _ok("tier1 checks passed", mc_gap=mc_gap, **metric_errors)
    except (KeyError, ValueError) as exc:
        return _fail(str(exc))


def _validate_boundary(boundary: np.ndarray, tier2: dict[str, Any]) -> TierReport:
    if boundary.shape != (100,):
        return _fail(f"exercise_boundary_tier2.npy has shape {boundary.shape}, expected (100,)")
    if boundary.dtype.kind not in {"f", "i"}:
        return _fail("exercise_boundary_tier2.npy must be numeric")

    finite_mask = np.isfinite(boundary)
    finite_values = boundary[finite_mask]
    if finite_values.size < 80:
        return _fail("exercise boundary must contain at least 80 finite entries")

    non_zero_finite = finite_values[np.abs(finite_values) > 1e-12]
    if non_zero_finite.size < 80:
        return _fail("exercise boundary must contain at least 80 non-zero finite entries")
    if np.min(non_zero_finite) < BOUNDARY_MIN or np.max(non_zero_finite) > BOUNDARY_MAX:
        return _fail(
            "exercise boundary non-zero finite entries must stay inside the published range",
            boundary_min=float(np.min(non_zero_finite)),
            boundary_max=float(np.max(non_zero_finite)),
        )

    sampled = boundary[::10]
    sampled_finite = sampled[np.isfinite(sampled)]
    sampled_non_zero = sampled_finite[np.abs(sampled_finite) > 1e-12]
    if sampled_non_zero.size < 7:
        return _fail("exercise boundary sampled points must contain at least 7 non-zero finite entries")

    monotonic_violations = int(np.sum(np.diff(sampled_non_zero) < -1e-6))
    if monotonic_violations > 2:
        return _fail("exercise boundary sample is too non-monotone", monotonic_violations=monotonic_violations)

    last_ten_median = float(np.nanmedian(boundary[-10:]))
    if last_ten_median <= 95.0:
        return _fail("exercise boundary tail median must exceed 95", last_ten_median=last_ten_median)

    sample_json = tier2.get("exercise_boundary_sample")
    if not isinstance(sample_json, list) or len(sample_json) != 10:
        return _fail("tier2.exercise_boundary_sample must be a list of length 10")
    for idx, (json_value, npy_value) in enumerate(zip(sample_json, sampled)):
        if json_value is None:
            if np.isfinite(npy_value):
                return _fail(f"tier2.exercise_boundary_sample[{idx}] is null but boundary sample is finite")
            continue
        try:
            parsed = _float(json_value, f"tier2.exercise_boundary_sample[{idx}]")
        except ValueError as exc:
            return _fail(str(exc))
        if not np.isfinite(npy_value) or abs(parsed - float(npy_value)) > 1e-8:
            return _fail(f"tier2.exercise_boundary_sample[{idx}] does not match exercise_boundary_tier2.npy")

    return _ok(
        "tier2 boundary checks passed",
        monotonic_violations=monotonic_violations,
        last_ten_median=last_ten_median,
        finite_entries=int(finite_values.size),
        non_zero_entries=int(non_zero_finite.size),
    )


def _validate_tier2(results: dict[str, Any], reference: dict[str, Any], boundary: np.ndarray) -> TierReport:
    try:
        tier = results["tier2"]
        ref = reference["tier2"]
    except KeyError as exc:
        return _fail(f"missing {exc.args[0]} in tier2")

    try:
        _validate_constants(
            tier,
            {
                "S0": 100.0,
                "K": 110.0,
                "r": 0.05,
                "sigma": 0.2,
                "T": 1.0,
                "n_paths": 100000,
                "n_steps": 100,
                "poly_degree": 3,
            },
            "tier2",
        )
        bs_european_put = _float(tier["bs_european_put"], "tier2.bs_european_put")
        bs_error = abs(bs_european_put - _float(ref["bs_european_put"], "reference.tier2.bs_european_put"))
        if bs_error > 1e-4:
            return _fail("tier2.bs_european_put does not match the Black-Scholes reference", bs_error=bs_error)

        american_put_price = _float(tier["american_put_price"], "tier2.american_put_price")
        american_price_error = abs(american_put_price - _float(ref["american_put_price"], "reference.tier2.american_put_price"))
        if american_price_error > TIER2_PRICE_TOLERANCE:
            return _fail(
                "tier2.american_put_price is outside the allowed tolerance",
                american_price_error=american_price_error,
            )

        american_put_se = _float(tier["american_put_se"], "tier2.american_put_se")
        if american_put_se >= 0.05:
            return _fail("tier2.american_put_se must be < 0.05", american_put_se=american_put_se)

        european_mc_from_paths = _float(tier["european_mc_from_paths"], "tier2.european_mc_from_paths")
        early_exercise_premium = _float(tier["early_exercise_premium"], "tier2.early_exercise_premium")
        premium_consistency = abs((american_put_price - european_mc_from_paths) - early_exercise_premium)
        if premium_consistency > 1e-6:
            return _fail("tier2.early_exercise_premium is inconsistent with tier2 prices", premium_consistency=premium_consistency)
        if early_exercise_premium <= 0.5:
            return _fail("tier2.early_exercise_premium must exceed 0.5", early_exercise_premium=early_exercise_premium)

        if not _bool(tier["premium_positive"], "tier2.premium_positive"):
            return _fail("tier2.premium_positive must be true")
        if not _bool(tier["bs_underestimates"], "tier2.bs_underestimates"):
            return _fail("tier2.bs_underestimates must be true")

        boundary_report = _validate_boundary(boundary, tier)
        if not boundary_report.passed:
            return boundary_report

        return _ok(
            "tier2 checks passed",
            american_price_error=american_price_error,
            bs_error=bs_error,
            early_exercise_premium=early_exercise_premium,
            american_put_se=american_put_se,
            **boundary_report.metrics,
        )
    except (KeyError, ValueError) as exc:
        return _fail(str(exc))


def _validate_tier3(results: dict[str, Any], reference: dict[str, Any]) -> TierReport:
    if "tier3" not in results:
        return _fail("missing tier3")
    try:
        tier = results["tier3"]
        ref = reference["tier3"]
        _validate_constants(
            tier,
            {
                "n_assets": 5,
                "S0": [100.0, 100.0, 100.0, 100.0, 100.0],
                "sigma": [0.18, 0.22, 0.25, 0.20, 0.28],
                "weights": [0.25, 0.20, 0.20, 0.15, 0.20],
                "K": 95.0,
                "r": 0.05,
                "T": 1.0,
                "rho": 0.3,
                "n_paths": 200000,
                "n_steps": 100,
                "poly_degree": 3,
            },
            "tier3",
        )
        american_price = _float(tier["american_basket_put_price"], "tier3.american_basket_put_price")
        ref_american_price = _float(ref["american_basket_put_price"], "reference.tier3.american_basket_put_price")
        american_price_error = abs(american_price - ref_american_price)
        if american_price_error > TIER3_PRICE_TOLERANCE:
            return _fail(
                "tier3.american_basket_put_price is outside the allowed tolerance",
                american_price_error=american_price_error,
            )

        american_se = _float(tier["american_basket_put_se"], "tier3.american_basket_put_se")
        if american_se <= 0.0 or american_se >= 0.05:
            return _fail("tier3.american_basket_put_se must be in (0, 0.05)", american_basket_put_se=american_se)

        european_mc = _float(tier["european_basket_put_mc"], "tier3.european_basket_put_mc")
        early_exercise_premium = _float(tier["early_exercise_premium"], "tier3.early_exercise_premium")
        premium_consistency = abs((american_price - european_mc) - early_exercise_premium)
        if premium_consistency > 1e-6:
            return _fail("tier3.early_exercise_premium is inconsistent with basket prices", premium_consistency=premium_consistency)
        if early_exercise_premium <= 0.0:
            return _fail("tier3.early_exercise_premium must be positive", early_exercise_premium=early_exercise_premium)

        deltas = _float_list(tier["deltas"], "tier3.deltas", 5)
        vegas = _float_list(tier["vegas"], "tier3.vegas", 5)
        if not all(delta < 0.0 for delta in deltas):
            return _fail("tier3 deltas must all be negative")
        if not all(vega > 0.0 for vega in vegas):
            return _fail("tier3 vegas must all be positive")

        ref_deltas = _float_list(ref["deltas"], "reference.tier3.deltas", 5)
        ref_vegas = _float_list(ref["vegas"], "reference.tier3.vegas", 5)
        for idx, (actual, target) in enumerate(zip(deltas, ref_deltas)):
            ratio = abs(actual) / max(abs(target), 1e-12)
            if ratio < 1.0 / GREEK_FACTOR_LIMIT or ratio > GREEK_FACTOR_LIMIT:
                return _fail(f"tier3 delta[{idx}] is outside the allowed factor-{GREEK_FACTOR_LIMIT} band", ratio=ratio)
        for idx, (actual, target) in enumerate(zip(vegas, ref_vegas)):
            ratio = abs(actual) / max(abs(target), 1e-12)
            if ratio < 1.0 / GREEK_FACTOR_LIMIT or ratio > GREEK_FACTOR_LIMIT:
                return _fail(f"tier3 vega[{idx}] is outside the allowed factor-{GREEK_FACTOR_LIMIT} band", ratio=ratio)

        weights = _float_list(tier["weights"], "tier3.weights", 5)
        s0 = _float_list(tier["S0"], "tier3.S0", 5)
        implied_weighted_delta_sum = float(sum(w * s * d for w, s, d in zip(weights, s0, deltas)))
        weighted_delta_sum = _float(tier["weighted_delta_sum"], "tier3.weighted_delta_sum")
        if abs(implied_weighted_delta_sum - weighted_delta_sum) > 1e-6:
            return _fail(
                "tier3.weighted_delta_sum is inconsistent with weights, S0, and deltas",
                weighted_delta_sum_error=abs(implied_weighted_delta_sum - weighted_delta_sum),
            )

        return _ok(
            "tier3 checks passed",
            american_price_error=american_price_error,
            american_basket_put_se=american_se,
            early_exercise_premium=early_exercise_premium,
        )
    except (KeyError, ValueError) as exc:
        return _fail(str(exc))


def score_submission(output_payloads: dict[str, bytes], reference_payloads: dict[str, bytes]) -> ScoreReport:
    notes: list[str] = []
    missing_reference = [name for name in REQUIRED_REFERENCE_FILES if name not in reference_payloads]
    if missing_reference:
        return ScoreReport(
            score=0.0,
            results_json_present="results.json" in output_payloads,
            boundary_present="exercise_boundary_tier2.npy" in output_payloads,
            results_json_valid=False,
            tier1=_fail("reference data missing"),
            tier2=_fail("reference data missing"),
            tier3=_fail("reference data missing"),
            notes=[f"missing reference files: {missing_reference}"],
        )

    reference_results = _load_json(reference_payloads["results.json"])
    reference_boundary = _load_npy(reference_payloads["exercise_boundary_tier2.npy"])
    if reference_boundary.shape != (100,):
        raise ValueError("hidden reference boundary must have shape (100,)")

    results_json_present = "results.json" in output_payloads
    boundary_present = "exercise_boundary_tier2.npy" in output_payloads
    if not results_json_present:
        return ScoreReport(
            score=0.0,
            results_json_present=False,
            boundary_present=boundary_present,
            results_json_valid=False,
            tier1=_fail("missing results.json"),
            tier2=_fail("missing results.json"),
            tier3=_fail("missing results.json"),
            notes=["results.json is required"],
        )

    try:
        results = _load_json(output_payloads["results.json"])
    except Exception as exc:
        return ScoreReport(
            score=0.0,
            results_json_present=True,
            boundary_present=boundary_present,
            results_json_valid=False,
            tier1=_fail("results.json is not valid JSON"),
            tier2=_fail("results.json is not valid JSON"),
            tier3=_fail("results.json is not valid JSON"),
            notes=[str(exc)],
        )

    if not boundary_present:
        boundary = np.empty((0,), dtype=float)
        notes.append("exercise_boundary_tier2.npy missing")
    else:
        try:
            boundary = _load_npy(output_payloads["exercise_boundary_tier2.npy"])
        except Exception as exc:
            return ScoreReport(
                score=0.0,
                results_json_present=True,
                boundary_present=True,
                results_json_valid=True,
                tier1=_validate_tier1(results, reference_results),
                tier2=_fail("exercise_boundary_tier2.npy could not be loaded"),
                tier3=_fail("tier3 not evaluated because boundary loading failed"),
                notes=[str(exc)],
            )

    tier1 = _validate_tier1(results, reference_results)
    tier2 = _fail("missing exercise_boundary_tier2.npy") if not boundary_present else _validate_tier2(results, reference_results, boundary)
    tier3 = _validate_tier3(results, reference_results)

    if tier1.passed and tier2.passed and tier3.passed:
        score = 1.0
    elif tier1.passed and tier2.passed:
        score = 0.5
    else:
        score = 0.0

    if boundary_present and boundary.shape == reference_boundary.shape:
        boundary_mae = float(np.nanmean(np.abs(boundary - reference_boundary)))
        notes.append(f"boundary_mae_vs_reference={boundary_mae:.6f}")

    return ScoreReport(
        score=score,
        results_json_present=results_json_present,
        boundary_present=boundary_present,
        results_json_valid=True,
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
        notes=notes,
    )
