#!/usr/bin/env python3
"""Scorer for the Gillespie gene regulatory network task.

Reads the agent output tree and hidden Tier 1/2 references, then prints one JSON
payload with `score` and failure reasons. Uses only the Python standard library.
"""

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

BANNED_IMPORTS = {
    "scipy",
    "gillespy2",
    "biosimulator",
    "biosimulators",
    "stochpy",
    "copasi",
    "stochkit",
}

TIER2_PARAMS = {
    "alpha": 50.0,
    "n": 2.73,
    "K": 20.0,
    "gamma_A": 0.047,
    "gamma_B": 0.083,
    "gamma_C": 0.061,
    "basal": 1.0,
    "num_trajectories": 10,
    "events_per_trajectory": 1000000,
    "seeds": list(range(42, 52)),
}


def _json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _close(actual: Any, expected: Any, abs_tol: float, rel_tol: float = 0.0) -> bool:
    if not _is_number(actual) or not _is_number(expected):
        return False
    actual_f = float(actual)
    expected_f = float(expected)
    return abs(actual_f - expected_f) <= max(abs_tol, rel_tol * max(1.0, abs(expected_f)))


def _prob_triplet(values: Iterable[Any], tol: float = 1e-6) -> bool:
    nums = [float(v) for v in values if _is_number(v)]
    return len(nums) == 3 and all(0.0 <= v <= 1.0 for v in nums) and abs(sum(nums) - 1.0) <= tol


def _add_check(points: List[float], reasons: List[str], condition: bool, weight: float, reason: str) -> None:
    if condition:
        points.append(weight)
    else:
        reasons.append(reason)


def _scan_source(path: Path) -> Tuple[bool, float, List[str]]:
    reasons: List[str] = []
    if not path.exists():
        return False, 0.0, [f"missing solver source at {path}"]
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except Exception as exc:
        return False, 0.0, [f"could not parse solver source: {exc}"]

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0].lower())
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0].lower())
    banned = sorted(imported & BANNED_IMPORTS)
    if banned:
        return False, 0.0, [f"banned import(s) in solver source: {', '.join(banned)}"]

    score = 0.06
    if "default_rng" in text:
        score += 0.02
    else:
        reasons.append("solver source does not mention numpy.random.default_rng")
    if "gillespie" in text.lower() or "propensit" in text.lower():
        score += 0.02
    else:
        reasons.append("solver source does not appear to implement SSA propensities")
    return True, min(0.10, score), reasons


def _score_tier1(output: Dict[str, Any], reference: Dict[str, Any]) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    points: List[float] = []
    out = output.get("birth_death")
    ref = reference.get("birth_death")
    if not isinstance(out, dict) or not isinstance(ref, dict):
        return 0.0, ["tier1: missing birth_death object"]

    _add_check(points, reasons, out.get("num_events") == 100000, 0.03, "tier1: num_events != 100000")
    _add_check(points, reasons, out.get("seed") == 42, 0.02, "tier1: seed != 42")
    _add_check(points, reasons, _close(out.get("theoretical_mean"), 100.0, 1e-12), 0.02, "tier1: theoretical_mean incorrect")
    _add_check(points, reasons, _close(out.get("theoretical_std"), 10.0, 1e-12), 0.02, "tier1: theoretical_std incorrect")
    _add_check(points, reasons, _close(out.get("mean_population"), ref.get("mean_population"), 3.0, 0.02), 0.04, "tier1: mean population not reproducible")
    _add_check(points, reasons, _close(out.get("std_population"), ref.get("std_population"), 2.0, 0.10), 0.03, "tier1: std population not reproducible")
    _add_check(points, reasons, _is_number(out.get("final_population")) and 50 <= int(out["final_population"]) <= 150, 0.02, "tier1: final_population implausible")
    _add_check(points, reasons, _is_number(out.get("ks_statistic")) and 0.0 <= float(out["ks_statistic"]) <= 0.2, 0.01, "tier1: ks_statistic implausible")
    _add_check(points, reasons, _is_number(out.get("ks_pvalue")) and float(out["ks_pvalue"]) > 0.01, 0.01, "tier1: ks_pvalue <= 0.01")
    return sum(points), reasons


def _score_tier2(output: Dict[str, Any], reference: Dict[str, Any]) -> Tuple[float, List[str], bool]:
    reasons: List[str] = []
    points: List[float] = []

    params = output.get("parameters")
    if not isinstance(params, dict):
        return 0.0, ["tier2: missing parameters"], True
    params_ok = True
    for key, expected in TIER2_PARAMS.items():
        actual = params.get(key)
        if isinstance(expected, list):
            ok = actual == expected
        else:
            ok = _close(actual, expected, 1e-12)
        if not ok:
            params_ok = False
            reasons.append(f"tier2: parameter {key} mismatch")
    _add_check(points, reasons, params_ok, 0.06, "tier2: parameter block mismatch")

    traj = output.get("trajectory_stats")
    ref_traj = reference.get("trajectory_stats")
    if not isinstance(traj, list) or len(traj) != 10:
        return 0.0, reasons + ["tier2: trajectory_stats must contain 10 entries"], True
    if not isinstance(ref_traj, list) or len(ref_traj) != 10:
        return 0.0, reasons + ["tier2: reference trajectory_stats malformed"], True

    schema_ok = True
    close_count = 0
    total_close = 0
    for idx, row in enumerate(traj):
        if not isinstance(row, dict):
            schema_ok = False
            continue
        if row.get("seed") != 42 + idx:
            schema_ok = False
        if not _prob_triplet(
            [row.get("fraction_basin_A"), row.get("fraction_basin_B"), row.get("fraction_basin_C")],
            tol=2e-5,
        ):
            schema_ok = False
        ref_row = ref_traj[idx]
        for key in ("mean_A", "mean_B", "mean_C", "std_A", "std_B", "std_C"):
            total_close += 1
            if _close(row.get(key), ref_row.get(key), abs_tol=8.0, rel_tol=0.08):
                close_count += 1
        for key in ("fraction_basin_A", "fraction_basin_B", "fraction_basin_C"):
            total_close += 1
            if _close(row.get(key), ref_row.get(key), abs_tol=0.08, rel_tol=0.0):
                close_count += 1
    _add_check(points, reasons, schema_ok, 0.07, "tier2: trajectory schema/seeds/fractions invalid")
    _add_check(points, reasons, close_count >= int(0.85 * total_close), 0.11, "tier2: trajectory statistics too far from reference")

    ens = output.get("ensemble_stats")
    ref_ens = reference.get("ensemble_stats")
    ens_ok = isinstance(ens, dict) and isinstance(ref_ens, dict)
    if ens_ok:
        for key in ("grand_mean_A", "grand_mean_B", "grand_mean_C", "grand_std_A", "grand_std_B", "grand_std_C"):
            if not _close(ens.get(key), ref_ens.get(key), abs_tol=10.0, rel_tol=0.08):
                ens_ok = False
                reasons.append(f"tier2: ensemble {key} mismatch")
    _add_check(points, reasons, ens_ok, 0.05, "tier2: ensemble stats invalid")

    acf = output.get("autocorrelation")
    acf_ok = isinstance(acf, dict)
    if acf_ok:
        lags = acf.get("lag_values")
        acf_ok = (
            isinstance(lags, list)
            and len(lags) == 50
            and all(_is_number(x) for x in lags)
            and all(float(lags[i]) < float(lags[i + 1]) for i in range(49))
        )
        for key in ("acf_A", "acf_B", "acf_C"):
            vals = acf.get(key)
            acf_ok = acf_ok and isinstance(vals, list) and len(vals) == 50 and all(_is_number(v) and -1.05 <= float(v) <= 1.05 for v in vals)
    _add_check(points, reasons, acf_ok, 0.04, "tier2: autocorrelation arrays invalid")

    tri = output.get("tristability_evidence")
    tri_ok = isinstance(tri, dict) and isinstance(tri.get("num_basins_visited"), int)
    if tri_ok:
        fracs = tri.get("basin_fractions_ensemble")
        tri_ok = isinstance(fracs, dict) and _prob_triplet(
            [fracs.get("A_dominant"), fracs.get("B_dominant"), fracs.get("C_dominant")],
            tol=1e-5,
        )
    _add_check(points, reasons, tri_ok, 0.03, "tier2: tristability evidence invalid")
    return sum(points), reasons, False


def _linspace(start: float, stop: float, count: int) -> List[float]:
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def _score_tier3(output: Dict[str, Any]) -> Tuple[float, List[str], bool]:
    reasons: List[str] = []
    points: List[float] = []
    scan = output.get("bifurcation_scan")
    comp = output.get("tau_leaping_comparison")
    if not isinstance(scan, dict) or not isinstance(comp, dict):
        return 0.0, ["tier3: missing bifurcation_scan or tau_leaping_comparison"], True

    alphas = scan.get("alpha_values")
    expected_alphas = _linspace(10.0, 100.0, 30)
    alpha_ok = isinstance(alphas, list) and len(alphas) == 30 and all(
        _close(a, b, 1e-8) for a, b in zip(alphas, expected_alphas)
    )
    if not alpha_ok:
        return 0.0, ["tier3: alpha_values must be 30 linearly spaced values from 10 to 100"], True
    points.append(0.06)

    rows = scan.get("basin_fractions")
    if not isinstance(rows, list) or len(rows) != 30:
        return 0.0, ["tier3: basin_fractions must contain 30 rows"], True
    row_ok = True
    for expected_alpha, row in zip(expected_alphas, rows):
        if not isinstance(row, dict):
            row_ok = False
            continue
        if not _close(row.get("alpha"), expected_alpha, 1e-8):
            row_ok = False
        if not _prob_triplet([row.get("fraction_A"), row.get("fraction_B"), row.get("fraction_C")], tol=1e-5):
            row_ok = False
        for key in ("mean_A", "mean_B", "mean_C", "std_A", "std_B", "std_C"):
            if not _is_number(row.get(key)) or float(row[key]) < 0.0:
                row_ok = False
    _add_check(points, reasons, row_ok, 0.14, "tier3: bifurcation rows invalid")

    eps_ok = _close(comp.get("epsilon"), 0.03, 1e-12)
    _add_check(points, reasons, eps_ok, 0.03, "tier3: epsilon != 0.03")

    points_list = comp.get("comparison_points")
    expected_points = [10.0, 30.0, 50.0, 70.0, 100.0]
    if not isinstance(points_list, list) or len(points_list) != 5:
        return 0.0, reasons + ["tier3: comparison_points must contain 5 rows"], True
    comp_ok = True
    for expected_alpha, row in zip(expected_points, points_list):
        if not isinstance(row, dict) or not _close(row.get("alpha"), expected_alpha, 1e-8):
            comp_ok = False
            continue
        for species in ("A", "B", "C"):
            exact = row.get(f"exact_mean_{species}")
            tau = row.get(f"tauleap_mean_{species}")
            rel = row.get(f"relative_error_mean_{species}")
            if not _is_number(exact) or float(exact) < 0.0:
                comp_ok = False
            if tau is not None and (not _is_number(tau) or float(tau) < 0.0):
                comp_ok = False
            if rel is not None and (not _is_number(rel) or not (0.0 <= float(rel) < 0.05)):
                comp_ok = False
        if not _is_number(row.get("relative_error_mean_A")) or float(row["relative_error_mean_A"]) >= 0.05:
            comp_ok = False
        if not _is_number(row.get("speedup_factor")) or float(row["speedup_factor"]) <= 1.0:
            comp_ok = False
    _add_check(points, reasons, comp_ok, 0.12, "tier3: tau-leaping comparison invalid")
    return sum(points), reasons, False


def _fail(reasons: List[str]) -> int:
    print(json.dumps({"score": 0.0, "reasons": reasons}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ref_dir = Path(args.reference_dir)
    paths = {
        "tier1": out_dir / "tier1_results.json",
        "tier2": out_dir / "tier2_results.json",
        "tier3": out_dir / "tier3_results.json",
    }
    ref_paths = {
        "tier1": ref_dir / "tier1_results.json",
        "tier2": ref_dir / "tier2_results.json",
    }
    missing = [str(path) for path in list(paths.values()) + list(ref_paths.values()) if not path.exists()]
    if missing:
        return _fail(["missing required file(s): " + ", ".join(missing)])

    try:
        tier1 = _json(paths["tier1"])
        tier2 = _json(paths["tier2"])
        tier3 = _json(paths["tier3"])
        ref1 = _json(ref_paths["tier1"])
        ref2 = _json(ref_paths["tier2"])
    except Exception as exc:
        return _fail([f"unparseable JSON: {exc}"])

    source_ok, source_score, source_reasons = _scan_source(out_dir / "gillespie_solver.py")
    if not source_ok:
        return _fail(source_reasons)

    tier1_score, tier1_reasons = _score_tier1(tier1, ref1)
    tier2_score, tier2_reasons, tier2_hard = _score_tier2(tier2, ref2)
    tier3_score, tier3_reasons, tier3_hard = _score_tier3(tier3)
    if tier2_hard or tier3_hard:
        return _fail(tier2_reasons + tier3_reasons + source_reasons)

    score = source_score + tier1_score + tier2_score + tier3_score
    reasons = source_reasons + tier1_reasons + tier2_reasons + tier3_reasons
    print(
        json.dumps(
            {
                "score": max(0.0, min(1.0, score)),
                "components": {
                    "source": source_score,
                    "tier1": tier1_score,
                    "tier2": tier2_score,
                    "tier3": tier3_score,
                },
                "reasons": reasons,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
