"""Benchmark-side verifier for the lid-driven cavity benchmark.

This script is used by the benchmark evaluator and admin replay tooling. It
reads the submitted output bundle (or benchmark fixtures) and compares the
final solution fields to the hidden reference solution bundle. It also verifies
that the submitted mesh-study evidence is internally consistent, rather than
trusting a self-reported convergence rate. The normalized visible contract now
follows the submitter-confirmed v3 package with one benchmark-side tightening:

- agent outputs must include verifiable mesh-study field snapshots
- `convergence_rate_u` must match the slope recomputed from those snapshots

- Reynolds numbers: Re200, Re750, Re1500
- required results.json fields:
  final_grid, nx, ny, dx, dy, convergence_rate_u
- steady_state_criterion is not required
- max_div is evaluator-side diagnostic metadata only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RE_CASES = ("Re200", "Re750", "Re1500")
THRESHOLDS = {"Re200": 0.01, "Re750": 0.02, "Re1500": 0.03}
MIN_FINEST_GRID = {"Re200": 65, "Re750": 129, "Re1500": 129}
MIN_MESH_STUDY_LEVELS = 3
CONVERGENCE_RATE_TOL = 0.15


@dataclass(frozen=True)
class CaseResult:
    case: str
    n: int
    max_abs_err_u: float
    max_abs_err_v: float
    max_abs_err: float
    max_div: float
    bc_max_err: float
    passed: bool


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def _interp_1d(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    # np.interp expects 1D ascending x. Our grids are uniform [0,1], so safe.
    return np.interp(x_tgt, x_src, y_src)


def _centerline_at_half_coord(values: np.ndarray, coord_axis: int) -> np.ndarray:
    """Extract a centerline at coordinate 0.5 from a collocated node grid.

    For odd N (e.g. 129), 0.5 is exactly on index (N-1)/2.
    For even N, linearly interpolate between the two middle indices.

    coord_axis selects which axis is the spatial coordinate along which 0.5 lies:
    - for u(y,x) vertical centerline, coord_axis=1 (x)
    - for v(y,x) horizontal centerline, coord_axis=0 (y)
    """

    n = values.shape[coord_axis]
    if n < 2:
        raise ValueError(f"invalid grid size: {n}")

    if (n - 1) % 2 == 0:
        mid = (n - 1) // 2
        if coord_axis == 1:
            return values[:, mid]
        return values[mid, :]

    # Even N: x=0.5 is between indices k and k+1 where x_k=k/(N-1)
    k = (n - 2) // 2
    x0 = k / (n - 1)
    x1 = (k + 1) / (n - 1)
    t = (0.5 - x0) / (x1 - x0)
    if coord_axis == 1:
        return (1.0 - t) * values[:, k] + t * values[:, k + 1]
    return (1.0 - t) * values[k, :] + t * values[k + 1, :]


def _max_divergence(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> float:
    # Compute divergence on interior nodes using central differences.
    du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx)
    dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dy)
    div = du_dx + dv_dy
    return float(np.max(np.abs(div)))


def _bc_max_error(u: np.ndarray, v: np.ndarray) -> float:
    # Ignore the two top corners for u because the lid velocity is discontinuous there.
    top_u_err = np.max(np.abs(u[-1, 1:-1] - 1.0)) if u.shape[1] > 2 else float("inf")
    bottom_u_err = float(np.max(np.abs(u[0, :] - 0.0)))
    left_u_err = float(np.max(np.abs(u[:-1, 0] - 0.0)))
    right_u_err = float(np.max(np.abs(u[:-1, -1] - 0.0)))

    top_v_err = float(np.max(np.abs(v[-1, :] - 0.0)))
    bottom_v_err = float(np.max(np.abs(v[0, :] - 0.0)))
    left_v_err = float(np.max(np.abs(v[:, 0] - 0.0)))
    right_v_err = float(np.max(np.abs(v[:, -1] - 0.0)))

    return float(
        max(
            top_u_err,
            bottom_u_err,
            left_u_err,
            right_u_err,
            top_v_err,
            bottom_v_err,
            left_v_err,
            right_v_err,
        )
    )


def _load_field_pair(flow_dir: Path, case: str) -> tuple[np.ndarray, np.ndarray]:
    u_path = flow_dir / f"u_{case}.npy"
    v_path = flow_dir / f"v_{case}.npy"
    _require_file(u_path)
    _require_file(v_path)
    u = np.load(u_path)
    v = np.load(v_path)
    if u.ndim != 2 or v.ndim != 2 or u.shape != v.shape:
        raise ValueError(f"invalid shapes for {case}: u{u.shape} v{v.shape}")
    if u.shape[0] != u.shape[1]:
        raise ValueError(f"grid must be square for {case}: {u.shape}")
    return u.astype(np.float64, copy=False), v.astype(np.float64, copy=False)


def _parse_final_grid(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if "x" not in text:
        return None
    left, right = text.split("x", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def _validate_case_metadata_against_output(
    case: str,
    obj: dict[str, Any],
    agent_u: np.ndarray,
    agent_v: np.ndarray,
) -> list[str]:
    errors: list[str] = []
    n_y, n_x = agent_u.shape
    if agent_v.shape != agent_u.shape:
        errors.append(f"{case}: u/v shape mismatch {agent_u.shape} vs {agent_v.shape}")
        return errors

    parsed_grid = _parse_final_grid(obj.get("final_grid"))
    if parsed_grid != (n_x, n_y):
        errors.append(f"{case}: final_grid does not match saved field shape {agent_u.shape}")

    try:
        nx = int(obj.get("nx"))
        ny = int(obj.get("ny"))
    except Exception:
        pass
    else:
        if (nx, ny) != (n_x, n_y):
            errors.append(f"{case}: nx/ny do not match saved field shape {agent_u.shape}")

    expected_spacing = 1.0 / (n_x - 1)
    try:
        dx = float(obj.get("dx"))
        dy = float(obj.get("dy"))
    except Exception:
        pass
    else:
        if not np.isclose(dx, expected_spacing, atol=1e-12, rtol=0.0):
            errors.append(f"{case}: dx={dx} does not match grid spacing {expected_spacing}")
        if not np.isclose(dy, expected_spacing, atol=1e-12, rtol=0.0):
            errors.append(f"{case}: dy={dy} does not match grid spacing {expected_spacing}")

    return errors


def _collect_mesh_study_fields(
    case: str,
    mesh_case_dir: Path,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[str]]:
    errors: list[str] = []
    pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    if not mesh_case_dir.exists():
        return {}, [f"{case}: missing mesh study directory {mesh_case_dir}"]

    u_paths = sorted(mesh_case_dir.glob("u_*.npy"))
    if not u_paths:
        return {}, [f"{case}: no mesh-study u-fields found under {mesh_case_dir}"]

    for u_path in u_paths:
        suffix = u_path.stem.removeprefix("u_")
        try:
            grid_n = int(suffix)
        except ValueError:
            errors.append(f"{case}: invalid mesh-study filename {u_path.name}")
            continue
        v_path = mesh_case_dir / f"v_{grid_n}.npy"
        if not v_path.exists():
            errors.append(f"{case}: missing paired mesh-study file {v_path.name}")
            continue
        try:
            u, v = _load_field_pair(mesh_case_dir, str(grid_n))
        except Exception as exc:
            errors.append(f"{case}: invalid mesh-study field {grid_n}: {exc}")
            continue
        pairs[grid_n] = (u, v)

    if len(pairs) < MIN_MESH_STUDY_LEVELS:
        errors.append(f"{case}: need at least {MIN_MESH_STUDY_LEVELS} mesh-study grid levels")

    return pairs, errors


def _compute_convergence_rate_from_mesh_study(
    case: str,
    mesh_fields: dict[int, tuple[np.ndarray, np.ndarray]],
    final_u: np.ndarray,
    final_v: np.ndarray,
) -> tuple[float | None, dict[str, Any], list[str]]:
    errors: list[str] = []
    if not mesh_fields:
        return None, {}, [f"{case}: missing mesh-study fields"]

    finest_n = max(mesh_fields)
    if finest_n != int(final_u.shape[0]):
        errors.append(
            f"{case}: finest mesh-study grid {finest_n} does not match final field grid {final_u.shape[0]}"
        )
        return None, {}, errors

    fine_u, fine_v = mesh_fields[finest_n]
    if not np.allclose(fine_u, final_u, atol=1e-12, rtol=1e-10):
        errors.append(f"{case}: finest mesh-study u-field does not match final output field")
    if not np.allclose(fine_v, final_v, atol=1e-12, rtol=1e-10):
        errors.append(f"{case}: finest mesh-study v-field does not match final output field")
    if errors:
        return None, {}, errors

    x_fine = np.linspace(0.0, 1.0, finest_n)
    fine_u_line = _centerline_at_half_coord(final_u, coord_axis=1)

    hs: list[float] = []
    errs: list[float] = []
    study_details: dict[str, Any] = {"grids": sorted(mesh_fields), "coarse_errors": {}}
    for grid_n in sorted(mesh_fields):
        if grid_n == finest_n:
            continue
        coarse_u, _ = mesh_fields[grid_n]
        coarse_x = np.linspace(0.0, 1.0, grid_n)
        coarse_u_line = _centerline_at_half_coord(coarse_u, coord_axis=1)
        err = float(np.max(np.abs(_interp_1d(coarse_x, coarse_u_line, x_fine) - fine_u_line)))
        if not np.isfinite(err) or err <= 0.0:
            errors.append(f"{case}: non-positive mesh-study error for grid {grid_n}")
            continue
        hs.append(1.0 / (grid_n - 1))
        errs.append(err)
        study_details["coarse_errors"][str(grid_n)] = err

    if len(errs) < 2:
        errors.append(f"{case}: need at least two coarse mesh-study levels to fit convergence rate")
        return None, study_details, errors

    rate = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
    study_details["computed_convergence_rate_u"] = rate
    study_details["finest_grid"] = finest_n
    return rate, study_details, errors


def _score_case(
    case: str,
    agent_u: np.ndarray,
    agent_v: np.ndarray,
    ref_u: np.ndarray,
    ref_v: np.ndarray,
) -> CaseResult:
    n = int(agent_u.shape[0])
    if n < MIN_FINEST_GRID[case]:
        return CaseResult(
            case=case,
            n=n,
            max_abs_err_u=float("inf"),
            max_abs_err_v=float("inf"),
            max_abs_err=float("inf"),
            max_div=float("inf"),
            bc_max_err=float("inf"),
            passed=False,
        )

    dx = 1.0 / (n - 1)
    dy = 1.0 / (n - 1)

    # Extract centerlines at x=0.5 and y=0.5.
    u_line = _centerline_at_half_coord(agent_u, coord_axis=1)
    v_line = _centerline_at_half_coord(agent_v, coord_axis=0)

    n_ref = int(ref_u.shape[0])
    x_ref = np.linspace(0.0, 1.0, n_ref)
    x_agent = np.linspace(0.0, 1.0, n)

    ref_u_line = _centerline_at_half_coord(ref_u, coord_axis=1)
    ref_v_line = _centerline_at_half_coord(ref_v, coord_axis=0)

    # Compare on the reference grid for determinism.
    u_line_i = _interp_1d(x_agent, u_line, x_ref)
    v_line_i = _interp_1d(x_agent, v_line, x_ref)

    max_u = float(np.max(np.abs(u_line_i - ref_u_line)))
    max_v = float(np.max(np.abs(v_line_i - ref_v_line)))
    max_err = float(max(max_u, max_v))

    max_div = _max_divergence(agent_u, agent_v, dx=dx, dy=dy)
    bc_err = _bc_max_error(agent_u, agent_v)

    # Boundary and divergence are hard gates.
    passed = (
        max_err <= THRESHOLDS[case]
        and max_div < 0.01
        and bc_err < 0.05
    )
    return CaseResult(
        case=case,
        n=n,
        max_abs_err_u=max_u,
        max_abs_err_v=max_v,
        max_abs_err=max_err,
        max_div=max_div,
        bc_max_err=bc_err,
        passed=bool(passed),
    )


def _validate_results_json(
    results: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for case in RE_CASES:
        if case not in results or not isinstance(results[case], dict):
            errors.append(f"missing results[{case}] object")
            continue
        obj = results[case]
        for k in ("final_grid", "nx", "ny", "dx", "dy", "convergence_rate_u"):
            if k not in obj:
                errors.append(f"{case}: missing {k}")
        try:
            nx = int(obj.get("nx"))
            ny = int(obj.get("ny"))
        except Exception:
            errors.append(f"{case}: nx and ny must be integers")
        else:
            if nx <= 1 or ny <= 1:
                errors.append(f"{case}: nx and ny must be > 1")
        try:
            dx = float(obj.get("dx"))
            dy = float(obj.get("dy"))
        except Exception:
            errors.append(f"{case}: dx and dy must be numbers")
        else:
            if not (dx > 0.0 and dy > 0.0):
                errors.append(f"{case}: dx and dy must be > 0")
        final_grid = obj.get("final_grid")
        if not isinstance(final_grid, str) or not final_grid.strip():
            errors.append(f"{case}: final_grid must be a non-empty string")
        try:
            rate = float(obj.get("convergence_rate_u"))
        except Exception:
            errors.append(f"{case}: convergence_rate_u must be a number")
        else:
            if not (rate > 0.8):
                errors.append(f"{case}: convergence_rate_u must be > 0.8 (got {rate})")
    return errors


def score_output_bundle(
    *,
    mode: str,
    input_dir: Path,
    reference_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _require_file(input_dir / "problem_spec.md")
    _require_file(reference_dir / "evaluator_policy.json")

    ref_flow = reference_dir / "reference_solver" / "flow_fields"
    if not ref_flow.exists():
        return {"score": 0.0, "error": f"reference_flow_missing: {ref_flow}"}

    flow_dir = output_dir / "flow_fields"
    results_path = output_dir / "results.json"
    if not flow_dir.exists():
        return {"score": 0.0, "error": f"missing_flow_fields_dir: {flow_dir}"}
    if not results_path.exists():
        return {"score": 0.0, "error": f"missing_results_json: {results_path}"}

    try:
        results = _read_json(results_path)
    except Exception as exc:
        return {"score": 0.0, "error": f"bad_results_json: {exc}"}

    json_errors = _validate_results_json(results)
    if json_errors:
        return {"score": 0.0, "error": "results_json_invalid", "details": json_errors}

    case_details: dict[str, Any] = {}
    passed_all = True
    mesh_root = output_dir / "mesh_study"
    for case in RE_CASES:
        try:
            agent_u, agent_v = _load_field_pair(flow_dir, case)
            meta_errors = _validate_case_metadata_against_output(case, results[case], agent_u, agent_v)
            mesh_fields, mesh_errors = _collect_mesh_study_fields(case, mesh_root / case)
            computed_rate, mesh_details, rate_errors = _compute_convergence_rate_from_mesh_study(
                case,
                mesh_fields,
                agent_u,
                agent_v,
            )
            ref_u, ref_v = _load_field_pair(ref_flow, case)
            res = _score_case(case, agent_u, agent_v, ref_u, ref_v)
        except Exception as exc:
            passed_all = False
            case_details[case] = {"passed": False, "error": str(exc)}
            continue
        reported_rate = float(results[case]["convergence_rate_u"])
        rate_mismatch = (
            computed_rate is None
            or not np.isclose(reported_rate, computed_rate, atol=CONVERGENCE_RATE_TOL, rtol=0.0)
        )
        case_passed = res.passed and not meta_errors and not mesh_errors and not rate_errors and not rate_mismatch
        case_details[case] = {
            "passed": case_passed,
            "n": res.n,
            "max_abs_err_u": res.max_abs_err_u,
            "max_abs_err_v": res.max_abs_err_v,
            "max_abs_err": res.max_abs_err,
            "max_div": res.max_div,
            "bc_max_err": res.bc_max_err,
            "threshold": THRESHOLDS[case],
            "reported_convergence_rate_u": reported_rate,
            "computed_convergence_rate_u": computed_rate,
            "mesh_study": mesh_details,
        }
        detail_errors: list[str] = []
        detail_errors.extend(meta_errors)
        detail_errors.extend(mesh_errors)
        detail_errors.extend(rate_errors)
        if rate_mismatch:
            detail_errors.append(
                f"{case}: reported convergence_rate_u={reported_rate} does not match computed rate {computed_rate}"
            )
        if detail_errors:
            case_details[case]["details"] = detail_errors
        passed_all = passed_all and case_passed

    return {
        "score": 1.0 if passed_all else 0.0,
        "mode": mode,
        "passed": bool(passed_all),
        "cases": case_details,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, help="output | output_test_pos | output_test_neg")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    mode = str(args.mode)
    input_dir = Path(args.input_dir)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)

    try:
        payload = score_output_bundle(
            mode=mode,
            input_dir=input_dir,
            reference_dir=reference_dir,
            output_dir=output_dir,
        )
    except Exception as exc:
        payload = {"score": 0.0, "error": f"staging_missing: {exc}"}
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
