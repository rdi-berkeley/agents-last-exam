"""Score outputs for engineering/bouligand_uniaxial_compression_abaqus."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_OUTPUT_FILES = (
    "abaqus_input_deck.inp",
    "simulated_force_displacement.csv",
    "simulation_summary.json",
    "verification_report.md",
)
EXPECTED_CURVE_COLUMNS = ["displacement_mm", "force_n"]
MIN_SIMULATED_CURVE_COVERAGE_MM = 0.25
HIDDEN_REFERENCE_MARGIN_PERCENT = 3.0
HIDDEN_REFERENCE_VISIBLE_GAP_PERCENT = 2.0
SUMMARY_MAPE_TOLERANCE_PERCENT = 0.05
REPORT_MAPE_TOLERANCE_PERCENT = 0.1
REPORT_THRESHOLD_TOLERANCE_PERCENT = 0.1
SUMMARY_WEIGHT_CURVE = 0.70
SUMMARY_WEIGHT_METADATA = 0.15
SUMMARY_WEIGHT_INP = 0.10
SUMMARY_WEIGHT_REPORT = 0.05


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: bool
    reasons: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


def _result(score: float, reasons: list[str], *, details: dict[str, Any] | None = None) -> ScoreResult:
    bounded = max(0.0, min(1.0, float(score)))
    return ScoreResult(
        score=bounded,
        passed=bounded >= 1.0,
        reasons=reasons,
        details=details or {},
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a top-level JSON object")
    return payload


def _parse_finite_number(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _read_curve_csv(path: Path) -> list[tuple[float, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_CURVE_COLUMNS:
            raise ValueError(
                f"{path.name} columns must be exactly {EXPECTED_CURVE_COLUMNS}, got {reader.fieldnames}"
            )
        rows: list[tuple[float, float]] = []
        last_displacement: float | None = None
        for row_index, row in enumerate(reader, start=2):
            displacement = _parse_finite_number(
                row.get("displacement_mm"),
                f"{path.name} row {row_index} displacement_mm",
            )
            force = _parse_finite_number(
                row.get("force_n"),
                f"{path.name} row {row_index} force_n",
            )
            if last_displacement is not None and displacement <= last_displacement:
                raise ValueError(
                    f"{path.name} displacement_mm must be strictly increasing; "
                    f"row {row_index} has {displacement} after {last_displacement}"
                )
            rows.append((displacement, force))
            last_displacement = displacement
    if len(rows) < 10:
        raise ValueError(f"{path.name} must contain at least 10 data rows")
    return rows


def _compute_mape(
    experimental_curve: list[tuple[float, float]],
    simulated_curve: list[tuple[float, float]],
) -> float:
    sim_x = [x for x, _ in simulated_curve]
    sim_y = [y for _, y in simulated_curve]

    def _interp(x: float) -> float:
        idx = bisect.bisect_left(sim_x, x)
        if idx <= 0:
            return sim_y[0]
        if idx >= len(sim_x):
            return sim_y[-1]
        x0, y0 = sim_x[idx - 1], sim_y[idx - 1]
        x1, y1 = sim_x[idx], sim_y[idx]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    common = [
        (x, experimental_force, _interp(x))
        for x, experimental_force in experimental_curve
        if sim_x[0] <= x <= sim_x[-1] and experimental_force != 0.0
    ]
    if not common:
        raise ValueError("simulated_force_displacement.csv has no common displacement range with experiment")
    return (
        sum(abs((simulated_force - experimental_force) / experimental_force) for _, experimental_force, simulated_force in common)
        / len(common)
        * 100.0
    )


def _curve_score(mape_percent: float, threshold_percent: float) -> float:
    if mape_percent <= threshold_percent:
        return 1.0
    if mape_percent >= 50.0:
        return 0.0
    return max(0.0, 1.0 - (mape_percent - threshold_percent) / (50.0 - threshold_percent))


def _summary_subscore(
    summary: dict[str, Any],
    visible_constants: dict[str, Any],
    *,
    computed_mape_percent: float,
) -> tuple[float, list[str], dict[str, Any]]:
    reasons: list[str] = []

    software_text = str(summary.get("software", ""))
    software_ok = "abaqus" in software_text.lower()
    if not software_ok:
        reasons.append("simulation_summary.json software must mention Abaqus")

    analysis_ok = summary.get("analysis_procedure") == "dynamic_explicit"
    if not analysis_ok:
        reasons.append("simulation_summary.json analysis_procedure must be dynamic_explicit")

    numeric_checks = (
        "density_tonne_per_mm3",
        "youngs_modulus_mpa",
        "poissons_ratio",
        "friction_coefficient",
        "loading_rate_mm_per_min",
        "specimen_diameter_mm",
        "specimen_height_mm",
        "error_threshold_percent",
    )
    constants_ok = True
    constant_mismatches: dict[str, dict[str, float]] = {}
    for key in numeric_checks:
        expected = _parse_finite_number(visible_constants.get(key), f"visible constant {key}")
        actual = _parse_finite_number(summary.get(key), f"summary field {key}")
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            constants_ok = False
            constant_mismatches[key] = {"expected": expected, "actual": actual}
    if not constants_ok:
        reasons.append("simulation_summary.json constants do not match input/canonical_benchmark_constants.json")

    reported_mape = _parse_finite_number(
        summary.get("computed_mape_percent"),
        "simulation_summary.json computed_mape_percent",
    )
    mape_ok = abs(reported_mape - computed_mape_percent) <= SUMMARY_MAPE_TOLERANCE_PERCENT
    if not mape_ok:
        reasons.append(
            "simulation_summary.json computed_mape_percent does not match the evaluator recomputation"
        )

    subscore = (
        0.10 * float(software_ok)
        + 0.10 * float(analysis_ok)
        + 0.50 * float(constants_ok)
        + 0.30 * float(mape_ok)
    )
    details = {
        "software_ok": software_ok,
        "analysis_ok": analysis_ok,
        "constants_ok": constants_ok,
        "mape_ok": mape_ok,
        "reported_mape_percent": reported_mape,
        "constant_mismatches": constant_mismatches,
    }
    return subscore, reasons, details


def _parse_first_friction(inp_text: str) -> float | None:
    prev = ""
    for raw_line in inp_text.splitlines():
        line = raw_line.strip()
        if prev == "*friction" and line:
            head = line.split(",", 1)[0].strip()
            try:
                return float(head)
            except ValueError:
                return None
        prev = line.lower()
    return None


def _count_deck_section_rows(inp_text: str, header: str) -> int:
    target = header.lower()
    active = False
    count = 0
    for raw_line in inp_text.splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("*"):
            active = lowered.startswith(target)
            continue
        if active and line:
            count += 1
    return count


def _inp_subscore(inp_text: str, visible_constants: dict[str, Any]) -> tuple[float, list[str], dict[str, Any]]:
    lowered = inp_text.lower()
    expected_friction = _parse_finite_number(
        visible_constants.get("friction_coefficient"),
        "visible constant friction_coefficient",
    )
    parsed_friction = _parse_first_friction(inp_text)
    node_row_count = _count_deck_section_rows(inp_text, "*node")
    element_row_count = _count_deck_section_rows(inp_text, "*element")
    nonempty_line_count = sum(1 for line in inp_text.splitlines() if line.strip())
    checks = {
        "has_step": "*step" in lowered,
        "has_dynamic_explicit": "*dynamic, explicit" in lowered,
        "has_material": "*material" in lowered,
        "has_density": "*density" in lowered,
        "has_elastic": "*elastic" in lowered,
        "has_surface_interaction": "*surface interaction" in lowered,
        "has_contact": "*contact" in lowered,
        "has_boundary": "*boundary" in lowered,
        "has_assembly": "*assembly" in lowered and "*instance" in lowered,
        "friction_matches": parsed_friction is not None
        and math.isclose(parsed_friction, expected_friction, rel_tol=1e-9, abs_tol=1e-9),
        "has_nontrivial_node_block": node_row_count >= 100,
        "has_nontrivial_element_block": element_row_count >= 100,
        "has_nontrivial_length": nonempty_line_count >= 500,
    }
    reasons = [name.replace("_", " ") for name, ok in checks.items() if not ok]
    if reasons:
        reasons = [f"abaqus_input_deck.inp missing or mismatching {reason}" for reason in reasons]
    return (
        sum(float(ok) for ok in checks.values()) / len(checks),
        reasons,
        {
            "parsed_friction": parsed_friction,
            "expected_friction": expected_friction,
            "node_row_count": node_row_count,
            "element_row_count": element_row_count,
            "nonempty_line_count": nonempty_line_count,
            "checks": checks,
        },
    )


def _report_subscore(
    report_text: str,
    *,
    computed_mape_percent: float,
    threshold_percent: float,
) -> tuple[float, list[str], dict[str, Any]]:
    lowered = report_text.lower()
    mape_match = re.search(r"mape[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*%", lowered)
    threshold_match = re.search(r"threshold[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*%", lowered)
    report_mape = float(mape_match.group(1)) if mape_match else None
    report_threshold = float(threshold_match.group(1)) if threshold_match else None
    report_conclusion_ok = (
        ("meets" in lowered or "passes" in lowered)
        if computed_mape_percent <= threshold_percent
        else ("fails" in lowered or "does not meet" in lowered)
    )

    checks = {
        "report_mape_matches": report_mape is not None
        and abs(report_mape - computed_mape_percent) <= REPORT_MAPE_TOLERANCE_PERCENT,
        "report_threshold_matches": report_threshold is not None
        and abs(report_threshold - threshold_percent) <= REPORT_THRESHOLD_TOLERANCE_PERCENT,
        "report_conclusion_matches": report_conclusion_ok,
    }
    reasons = [name.replace("_", " ") for name, ok in checks.items() if not ok]
    if reasons:
        reasons = [f"verification_report.md missing or mismatching {reason}" for reason in reasons]
    return (
        sum(float(ok) for ok in checks.values()) / len(checks),
        reasons,
        {
            "reported_mape_percent": report_mape,
            "reported_threshold_percent": report_threshold,
            "checks": checks,
        },
    )


def evaluate_submission(output_dir: Path, input_dir: Path, reference_dir: Path) -> ScoreResult:
    missing = [
        str(output_dir / name)
        for name in REQUIRED_OUTPUT_FILES
        if not (output_dir / name).exists()
    ]
    if missing:
        return _result(0.0, [f"missing required output files: {', '.join(missing)}"])

    input_contract_path = input_dir / "output_contract.json"
    input_constants_path = input_dir / "canonical_benchmark_constants.json"
    experimental_curve_path = input_dir / "experimental_force_displacement.csv"
    reference_summary_path = reference_dir / "reference_summary.json"
    scoring_contract_path = reference_dir / "scoring_contract.json"
    reference_curve_path = reference_dir / "reference_force_displacement.csv"

    hidden_missing = [
        str(path)
        for path in (
            input_contract_path,
            input_constants_path,
            experimental_curve_path,
            reference_summary_path,
            scoring_contract_path,
            reference_curve_path,
        )
        if not path.exists()
    ]
    if hidden_missing:
        return _result(0.0, [f"missing evaluator-side task data: {', '.join(hidden_missing)}"])

    try:
        output_contract = _load_json(input_contract_path)
        visible_constants = _load_json(input_constants_path)
        _load_json(reference_summary_path)
        _load_json(scoring_contract_path)
        simulated_curve = _read_curve_csv(output_dir / "simulated_force_displacement.csv")
        experimental_curve = _read_curve_csv(experimental_curve_path)
        reference_curve = _read_curve_csv(reference_curve_path)
        summary = _load_json(output_dir / "simulation_summary.json")
        inp_text = (output_dir / "abaqus_input_deck.inp").read_text(encoding="utf-8", errors="ignore")
        report_text = (output_dir / "verification_report.md").read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return _result(0.0, [f"failed to parse task inputs or outputs: {exc}"])

    required_summary_keys = output_contract.get("summary_required_keys")
    if not isinstance(required_summary_keys, list) or not all(
        isinstance(key, str) for key in required_summary_keys
    ):
        return _result(0.0, ["input/output_contract.json summary_required_keys is malformed"])
    missing_summary_keys = [key for key in required_summary_keys if key not in summary]
    if missing_summary_keys:
        return _result(
            0.0,
            [f"simulation_summary.json missing required keys: {missing_summary_keys}"],
        )

    try:
        threshold_percent = _parse_finite_number(
            visible_constants.get("error_threshold_percent"),
            "error_threshold_percent",
        )
        computed_mape_percent = _compute_mape(experimental_curve, simulated_curve)
        hidden_reference_mape_percent = _compute_mape(reference_curve, simulated_curve)
        public_curve_copy_mape_percent = _compute_mape(reference_curve, experimental_curve)
    except Exception as exc:
        return _result(0.0, [str(exc)])

    required_max_displacement = reference_curve[-1][0] - MIN_SIMULATED_CURVE_COVERAGE_MM
    if simulated_curve[-1][0] < required_max_displacement:
        return _result(
            0.0,
            [
                "simulated_force_displacement.csv does not cover the required compression range"
            ],
            details={
                "simulated_curve_max_displacement_mm": simulated_curve[-1][0],
                "required_min_displacement_mm": required_max_displacement,
            },
        )

    hidden_reference_limit = public_curve_copy_mape_percent - HIDDEN_REFERENCE_MARGIN_PERCENT
    if hidden_reference_mape_percent >= hidden_reference_limit:
        return _result(
            0.0,
            [
                "simulated_force_displacement.csv is not materially closer to the hidden reference run than the visible experimental curve"
            ],
            details={
                "hidden_reference_mape_percent": hidden_reference_mape_percent,
                "public_curve_copy_mape_percent": public_curve_copy_mape_percent,
                "required_hidden_reference_limit_percent": hidden_reference_limit,
            },
        )
    hidden_visible_gap_limit = computed_mape_percent + HIDDEN_REFERENCE_VISIBLE_GAP_PERCENT
    if hidden_reference_mape_percent > hidden_visible_gap_limit:
        return _result(
            0.0,
            [
                "simulated_force_displacement.csv appears derived from the visible experimental curve rather than a hidden-reference-consistent simulation"
            ],
            details={
                "hidden_reference_mape_percent": hidden_reference_mape_percent,
                "computed_mape_percent": computed_mape_percent,
                "allowed_hidden_reference_mape_percent": hidden_visible_gap_limit,
            },
        )

    curve_score = _curve_score(computed_mape_percent, threshold_percent)
    summary_score, summary_reasons, summary_details = _summary_subscore(
        summary,
        visible_constants,
        computed_mape_percent=computed_mape_percent,
    )
    inp_score, inp_reasons, inp_details = _inp_subscore(inp_text, visible_constants)
    report_score, report_reasons, report_details = _report_subscore(
        report_text,
        computed_mape_percent=computed_mape_percent,
        threshold_percent=threshold_percent,
    )

    reasons = summary_reasons + inp_reasons + report_reasons
    final_score = (
        SUMMARY_WEIGHT_CURVE * curve_score
        + SUMMARY_WEIGHT_METADATA * summary_score
        + SUMMARY_WEIGHT_INP * inp_score
        + SUMMARY_WEIGHT_REPORT * report_score
    )
    details = {
        "computed_mape_percent": computed_mape_percent,
        "hidden_reference_mape_percent": hidden_reference_mape_percent,
        "public_curve_copy_mape_percent": public_curve_copy_mape_percent,
        "hidden_reference_limit_percent": hidden_reference_limit,
        "allowed_hidden_reference_mape_percent": hidden_visible_gap_limit,
        "threshold_percent": threshold_percent,
        "curve_score": curve_score,
        "summary_score": summary_score,
        "inp_score": inp_score,
        "report_score": report_score,
        "summary_details": summary_details,
        "inp_details": inp_details,
        "report_details": report_details,
    }
    return _result(final_score, reasons, details=details)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_submission(
        output_dir=Path(args.output_dir),
        input_dir=Path(args.input_dir),
        reference_dir=Path(args.reference_dir),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
