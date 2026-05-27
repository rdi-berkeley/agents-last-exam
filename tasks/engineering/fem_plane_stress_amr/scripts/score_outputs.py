"""Evaluator for the FEM plane-stress AMR task."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SIGMA_PEAK = 47.3e6
TIER3_CYCLE_FIELDS = {
    "displacement": ("tier3_cycle{cycle}_displacement.npy", 2, 2),
    "nodes": ("tier3_cycle{cycle}_nodes.npy", 2, 2),
    "elements": ("tier3_cycle{cycle}_elements.npy", None, None),
    "eta": ("tier3_cycle{cycle}_eta.npy", 1, None),
    "von_mises": ("tier3_cycle{cycle}_von_mises.npy", 1, None),
}
REFERENCE_OUTPUT_FILES = {
    "tier1_results.json",
    "tier1_displacement.npy",
    "tier1_nodes.npy",
    "tier1_elements.npy",
    "tier2_results.json",
    "tier2_displacement.npy",
    "tier2_stress.npy",
    "tier2_von_mises.npy",
    "tier2_nodes.npy",
    "tier2_elements.npy",
    "tier3_results.json",
    "tier3_final_displacement.npy",
    "tier3_final_nodes.npy",
    "tier3_final_elements.npy",
    "tier3_final_eta.npy",
    "tier3_final_von_mises.npy",
}


def _relative_error(value: float, target: float, floor: float = 1.0) -> float:
    return abs(value - target) / max(abs(target), floor)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_array(path: Path, ndim: int | None = None, last_dim: int | None = None) -> np.ndarray:
    arr = np.load(path)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path.name} contains non-finite values")
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{path.name} ndim {arr.ndim} != {ndim}")
    if last_dim is not None and (arr.ndim == 0 or arr.shape[-1] != last_dim):
        raise ValueError(f"{path.name} last dim {arr.shape[-1] if arr.ndim else None} != {last_dim}")
    return arr


def _triangle_areas(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    verts = nodes[elements[:, :3]]
    v1 = verts[:, 1] - verts[:, 0]
    v2 = verts[:, 2] - verts[:, 0]
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])


def _mesh_valid(nodes: np.ndarray, elements: np.ndarray, min_nodes_per_elem: int) -> bool:
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        return False
    if elements.ndim != 2 or elements.shape[1] < min_nodes_per_elem:
        return False
    if elements.size == 0 or elements.min() < 0 or elements.max() >= len(nodes):
        return False
    return bool(np.all(_triangle_areas(nodes, elements) > 0.0))


def _l_shape_ok(nodes: np.ndarray, tol: float = 3e-2) -> bool:
    removed_block = (nodes[:, 0] > 1.0 + tol) & (nodes[:, 1] > 1.0 + tol)
    return not bool(np.any(removed_block))


def _clamped_ok(nodes: np.ndarray, disp: np.ndarray, mask: np.ndarray, tol: float = 1e-10) -> bool:
    if not np.any(mask):
        return False
    return bool(np.max(np.abs(disp[mask])) <= tol)


def _von_mises(stress: np.ndarray) -> np.ndarray:
    sx, sy, txy = stress[:, 0], stress[:, 1], stress[:, 2]
    return np.sqrt(np.maximum(sx * sx + sy * sy - sx * sy + 3.0 * txy * txy, 0.0))


def _load_reference(reference_dir: Path) -> dict[str, object]:
    missing = [name for name in REFERENCE_OUTPUT_FILES if not (reference_dir / name).exists()]
    if missing:
        raise FileNotFoundError("missing reference file(s): " + ", ".join(sorted(missing)))
    return {
        "tier1_results": _load_json(reference_dir / "tier1_results.json"),
        "tier1_nodes": _finite_array(reference_dir / "tier1_nodes.npy", 2, 2),
        "tier1_elements": np.load(reference_dir / "tier1_elements.npy"),
        "tier2_results": _load_json(reference_dir / "tier2_results.json"),
        "tier2_nodes": _finite_array(reference_dir / "tier2_nodes.npy", 2, 2),
        "tier2_elements": np.load(reference_dir / "tier2_elements.npy"),
        "tier2_von_mises": _finite_array(reference_dir / "tier2_von_mises.npy", 1),
        "tier3_results": _load_json(reference_dir / "tier3_results.json"),
        "tier3_final_nodes": _finite_array(reference_dir / "tier3_final_nodes.npy", 2, 2),
        "tier3_final_elements": np.load(reference_dir / "tier3_final_elements.npy"),
        "tier3_final_von_mises": _finite_array(reference_dir / "tier3_final_von_mises.npy", 1),
    }


def _tier3_cycle_outputs_ok(output_dir: Path, hist: list[dict]) -> tuple[bool, str]:
    if not hist:
        return False, "empty refinement history"
    for item in hist:
        try:
            cycle = int(item["cycle"])
            num_nodes = int(item["num_nodes"])
            num_elements = int(item["num_elements"])
            nodes = _finite_array(output_dir / TIER3_CYCLE_FIELDS["nodes"][0].format(cycle=cycle), 2, 2)
            disp = _finite_array(output_dir / TIER3_CYCLE_FIELDS["displacement"][0].format(cycle=cycle), 2, 2)
            elems = np.load(output_dir / TIER3_CYCLE_FIELDS["elements"][0].format(cycle=cycle))
            eta = _finite_array(output_dir / TIER3_CYCLE_FIELDS["eta"][0].format(cycle=cycle), 1)
            vm = _finite_array(output_dir / TIER3_CYCLE_FIELDS["von_mises"][0].format(cycle=cycle), 1)
        except Exception as exc:
            return False, f"cycle output load failed: {exc}"
        shapes_ok = len(nodes) == len(disp) == num_nodes and len(elems) == len(eta) == len(vm) == num_elements
        if not shapes_ok:
            return False, f"cycle {cycle} shapes do not match reported history counts"
        if not _mesh_valid(nodes, elems, 6):
            return False, f"cycle {cycle} mesh is invalid"
    return True, ""


def _tier3_sif_fit_ok(output_dir: Path, res: dict, exponent: float) -> tuple[bool, str]:
    try:
        sif = _load_json(output_dir / "tier3_sif_fit.json")
        k_i = float(sif.get("K_I", np.nan))
        reported = float(res.get("stress_intensity_factor", np.nan))
        r_min = float(sif.get("r_min", np.nan))
        r_max = float(sif.get("r_max", np.nan))
        samples = int(sif.get("num_sample_points", -1))
        residual = float(sif.get("residual_norm", np.nan))
        lambda_used = float(sif.get("lambda_used", np.nan))
    except Exception as exc:
        return False, f"SIF fit load failed: {exc}"
    valid = (
        np.isfinite(k_i)
        and k_i > 0.0
        and _relative_error(reported, k_i) <= 0.05
        and np.isfinite(r_min)
        and np.isfinite(r_max)
        and 0.0 < r_min < r_max
        and samples >= 20
        and np.isfinite(residual)
        and residual >= 0.0
        and abs(lambda_used - exponent) <= 0.02
    )
    if not valid:
        return False, "invalid or inconsistent tier3_sif_fit.json"
    return True, ""


def score_directory(output_dir: Path, reference_dir: Path) -> dict:
    report = {"score": 0.0, "tier_scores": {}, "checks": {}, "messages": []}

    def ok(name: str, value: bool, message: str = "") -> bool:
        report["checks"][name] = bool(value)
        if not value and message:
            report["messages"].append(f"{name}: {message}")
        return bool(value)

    try:
        ref = _load_reference(reference_dir)
        ok("reference_load", True)
    except Exception as exc:
        ok("reference_load", False, str(exc))
        return report

    # Tier 1
    tier1 = 0.0
    try:
        res = _load_json(output_dir / "tier1_results.json")
        nodes = _finite_array(output_dir / "tier1_nodes.npy", 2, 2)
        disp = _finite_array(output_dir / "tier1_displacement.npy", 2, 2)
        elems = np.load(output_dir / "tier1_elements.npy")
        ref_res = ref["tier1_results"]
        analytic = float(ref_res["analytic_displacement"])
        right_edge = np.isclose(nodes[:, 0], np.max(nodes[:, 0]))
        tip_from_field = float(np.max(np.abs(disp[right_edge, 1]))) if np.any(right_edge) else math.nan
        reported_tip = abs(float(res.get("tip_displacement", np.nan)))
        reported_analytic = float(res.get("analytic_displacement", np.nan))
        rel_err = abs(tip_from_field - analytic) / analytic
        if ok("tier1_displacement", rel_err <= 0.05, f"relative error {rel_err:.4g}"):
            tier1 += 0.30
        if ok("tier1_report_matches_field", abs(reported_tip - tip_from_field) / analytic <= 0.02, "reported tip does not match displacement field"):
            tier1 += 0.10
        if ok("tier1_reported_analytic", abs(reported_analytic - analytic) / analytic <= 0.001, "analytic displacement mismatch"):
            tier1 += 0.05
        if ok("tier1_mesh", _mesh_valid(nodes, elems, 3), "invalid triangle mesh"):
            tier1 += 0.15
        if ok("tier1_shapes", len(nodes) == len(disp), "node/displacement length mismatch"):
            tier1 += 0.10
        count_ok = (
            int(res.get("num_nodes", -1)) == len(nodes)
            and int(res.get("num_elements", -1)) == len(elems)
            and len(elems) >= 320
            and len(nodes) >= 200
        )
        if ok("tier1_counts", count_ok, "reported counts or minimum mesh resolution invalid"):
            tier1 += 0.15
        if ok("tier1_clamped_bc", _clamped_ok(nodes, disp, np.isclose(nodes[:, 0], 0.0)), "x=0 nodes move"):
            tier1 += 0.10
        if ok("tier1_nonzero_field", np.linalg.norm(disp) > 0.25 * analytic, "displacement field is essentially zero"):
            tier1 += 0.05
    except Exception as exc:
        ok("tier1_load", False, str(exc))
    report["tier_scores"]["tier1"] = tier1
    tier1_critical = all(
        report["checks"].get(name, False)
        for name in [
            "tier1_displacement",
            "tier1_report_matches_field",
            "tier1_reported_analytic",
            "tier1_mesh",
            "tier1_shapes",
            "tier1_counts",
            "tier1_clamped_bc",
            "tier1_nonzero_field",
        ]
    )
    if not tier1_critical or tier1 < 0.70:
        report["score"] = 0.0
        return report

    # Tier 2
    tier2 = 0.0
    try:
        res = _load_json(output_dir / "tier2_results.json")
        nodes = _finite_array(output_dir / "tier2_nodes.npy", 2, 2)
        disp = _finite_array(output_dir / "tier2_displacement.npy", 2, 2)
        stress = _finite_array(output_dir / "tier2_stress.npy", 2, 3)
        vm = _finite_array(output_dir / "tier2_von_mises.npy", 1)
        elems = np.load(output_dir / "tier2_elements.npy")
        ref_res = ref["tier2_results"]
        energy = float(res.get("strain_energy", np.nan))
        energy_ref = float(ref_res["strain_energy"])
        energy_err = abs(energy - energy_ref) / energy_ref
        scf = float(res.get("stress_concentration_factor", np.nan))
        vm_calc = _von_mises(stress)
        vm_max = float(np.nanmax(vm))
        disp_max = float(np.nanmax(np.linalg.norm(disp, axis=1)))
        if ok("tier2_energy", energy_err <= 0.05, f"relative error {energy_err:.4g}"):
            tier2 += 0.20
        if ok("tier2_scf", scf > 3.0 and vm_max / SIGMA_PEAK > 3.0, "stress concentration too low"):
            tier2 += 0.15
        if ok("tier2_mesh", _mesh_valid(nodes, elems, 6), "invalid quadratic triangle mesh"):
            tier2 += 0.15
        shapes_ok = len(nodes) == len(disp) and len(stress) == len(vm) == len(elems)
        if ok("tier2_shapes", shapes_ok, "shape mismatch"):
            tier2 += 0.10
        denom = np.maximum(np.abs(vm_calc), 1.0)
        if ok("tier2_von_mises_consistency", np.nanmax(np.abs(vm - vm_calc) / denom) <= 0.02, "vm mismatch"):
            tier2 += 0.15
        clamp_mask = np.isclose(nodes[:, 0], 0.0) | np.isclose(nodes[:, 1], 0.0)
        if ok("tier2_clamped_bc", _clamped_ok(nodes, disp, clamp_mask, tol=1e-9), "clamped nodes move"):
            tier2 += 0.10
        report_ok = (
            int(res.get("num_nodes", -1)) == len(nodes)
            and int(res.get("num_elements", -1)) == len(elems)
            and _relative_error(float(res.get("max_von_mises_stress", np.nan)), vm_max) <= 0.03
            and _relative_error(float(res.get("max_displacement_magnitude", np.nan)), disp_max, floor=1e-12) <= 0.03
            and _relative_error(scf, vm_max / SIGMA_PEAK) <= 0.03
        )
        if ok("tier2_report_matches_fields", report_ok, "reported Tier 2 scalars do not match arrays"):
            tier2 += 0.10
        min_elems = max(400, int(0.5 * len(ref["tier2_elements"])))
        if ok("tier2_domain_and_resolution", len(elems) >= min_elems and _l_shape_ok(nodes), "insufficient L-shaped mesh"):
            tier2 += 0.05
    except Exception as exc:
        ok("tier2_load", False, str(exc))
    tier2_critical = all(
        report["checks"].get(name, False)
        for name in [
            "tier2_energy",
            "tier2_mesh",
            "tier2_shapes",
            "tier2_von_mises_consistency",
            "tier2_clamped_bc",
            "tier2_report_matches_fields",
            "tier2_domain_and_resolution",
        ]
    )
    if not tier2_critical:
        tier2 = 0.0
    report["tier_scores"]["tier2"] = tier2

    # Tier 3: full AMR output checks against the rev2 reference-backed task.
    tier3 = 0.0
    try:
        res = _load_json(output_dir / "tier3_results.json")
        ref_res = ref["tier3_results"]
        hist = res.get("refinement_history") or []
        if ok("tier3_history_len", len(hist) >= 3, "need at least 3 refinement entries"):
            tier3 += 0.07
        dofs = np.array([float(h.get("num_dof", np.nan)) for h in hist])
        elems_hist = np.array([float(h.get("num_elements", np.nan)) for h in hist])
        etas = np.array([float(h.get("eta_global", np.nan)) for h in hist])
        hist_order_ok = len(hist) >= 3 and np.all(np.diff(dofs) >= 0) and np.all(np.diff(elems_hist) >= 0) and dofs[-1] > dofs[0]
        if ok("tier3_history_order", hist_order_ok, "refinement history does not grow"):
            tier3 += 0.07
        if len(dofs) >= 3 and np.all(np.isfinite(dofs)) and np.all(np.isfinite(etas)) and np.all(dofs > 0) and np.all(etas > 0):
            slope = np.polyfit(np.log(dofs), np.log(etas), 1)[0]
            claimed = float(res.get("convergence_rate", np.nan))
            slope_ok = 0.5 <= abs(slope) <= 1.3 or 0.5 <= abs(claimed) <= 1.3
            if ok("tier3_convergence", slope_ok, f"slope={slope:.3g}, claimed={claimed:.3g}"):
                tier3 += 0.12
        exponent = float(res.get("singularity_exponent", np.nan))
        if ok("tier3_exponent", 0.52 <= exponent <= 0.57, f"lambda={exponent}"):
            tier3 += 0.07
        final_nodes = _finite_array(output_dir / "tier3_final_nodes.npy", 2, 2)
        final_elems = np.load(output_dir / "tier3_final_elements.npy")
        final_disp = _finite_array(output_dir / "tier3_final_displacement.npy", 2, 2)
        final_eta = _finite_array(output_dir / "tier3_final_eta.npy", 1)
        final_vm = _finite_array(output_dir / "tier3_final_von_mises.npy", 1)
        areas = _triangle_areas(final_nodes, final_elems)
        if ok("tier3_final_mesh", _mesh_valid(final_nodes, final_elems, 6), "invalid final mesh"):
            tier3 += 0.08
        shapes_ok = len(final_nodes) == len(final_disp) and len(final_elems) == len(final_eta) == len(final_vm)
        if ok("tier3_final_shapes", shapes_ok, "shape mismatch"):
            tier3 += 0.10
        if ok("tier3_refinement_evidence", areas.max() / areas.min() > 5.0, "little mesh grading"):
            tier3 += 0.07
        final_report_ok = (
            int(res.get("final_num_elements", -1)) == len(final_elems)
            and int(res.get("final_num_dof", -1)) == 2 * len(final_nodes)
            and int(hist[-1].get("num_elements", -1)) == len(final_elems)
            and int(hist[-1].get("num_nodes", -1)) == len(final_nodes)
            and int(hist[-1].get("num_dof", -1)) == 2 * len(final_nodes)
        )
        if ok("tier3_report_matches_fields", final_report_ok, "reported final counts do not match final arrays"):
            tier3 += 0.08
        ref_final_elems = len(ref["tier3_final_elements"])
        ref_final_vm = float(np.nanmax(ref["tier3_final_von_mises"]))
        final_vm_max = float(np.nanmax(final_vm))
        final_energy = float(hist[-1].get("strain_energy", np.nan)) if hist else math.nan
        ref_energy = float((ref_res.get("refinement_history") or [{}])[-1].get("strain_energy", np.nan))
        ref_scale_ok = (
            len(final_elems) >= max(150, int(0.75 * ref_final_elems))
            and _l_shape_ok(final_nodes)
            and final_vm_max >= 0.5 * ref_final_vm
            and 0.5 <= final_energy / ref_energy <= 2.0
            and np.linalg.norm(final_disp) > 0.0
        )
        if ok("tier3_reference_scale", ref_scale_ok, "final AMR output is too small or physically off-scale"):
            tier3 += 0.05
        clamp_mask = np.isclose(final_nodes[:, 0], 0.0) | np.isclose(final_nodes[:, 1], 0.0)
        if ok("tier3_clamped_bc", _clamped_ok(final_nodes, final_disp, clamp_mask, tol=1e-9), "clamped nodes move"):
            tier3 += 0.04
        cycle_ok, cycle_msg = _tier3_cycle_outputs_ok(output_dir, hist)
        if ok("tier3_cycle_outputs", cycle_ok, cycle_msg):
            tier3 += 0.15
        sif_ok, sif_msg = _tier3_sif_fit_ok(output_dir, res, exponent)
        if ok("tier3_sif_fit", sif_ok, sif_msg):
            tier3 += 0.10
    except Exception as exc:
        ok("tier3_load", False, str(exc))
    tier3_critical = all(
        report["checks"].get(name, False)
        for name in [
            "tier3_history_len",
            "tier3_history_order",
            "tier3_exponent",
            "tier3_final_mesh",
            "tier3_final_shapes",
            "tier3_refinement_evidence",
            "tier3_report_matches_fields",
            "tier3_reference_scale",
            "tier3_clamped_bc",
            "tier3_cycle_outputs",
            "tier3_sif_fit",
        ]
    )
    if not tier3_critical:
        tier3 = 0.0
    tier3 = float(max(0.0, min(1.0, tier3)))
    report["tier_scores"]["tier3"] = tier3

    total = 0.35 * tier1 + 0.40 * tier2 + 0.25 * tier3
    report["score"] = float(max(0.0, min(1.0, total)))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(score_directory(args.output_dir, args.reference_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
