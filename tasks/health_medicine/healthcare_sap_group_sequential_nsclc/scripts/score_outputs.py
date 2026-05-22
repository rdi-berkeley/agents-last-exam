"""Scoring logic for healthcare_sap_group_sequential_nsclc."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = {
    "SAP.md",
    "analysis.R",
    "sample_size.json",
    "boundaries.csv",
    "multiple_testing.json",
    "power_curve.csv",
    "power_curve.png",
    "boundary_plot.png",
}


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8")


def _load_json(raw: bytes) -> Any:
    return json.loads(_decode(raw))


def _load_csv(raw: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(_decode(raw).splitlines()))


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not numeric")
    return float(value)


def _png_ok(raw: bytes) -> bool:
    return len(raw) > 100 and raw.startswith(b"\x89PNG\r\n\x1a\n")


def _nonempty_required_files(output_files: dict[str, bytes]) -> list[str]:
    return sorted(name for name in REQUIRED_FILES if not output_files.get(name))


def evaluate_output_bundle(
    *,
    output_files: dict[str, bytes],
    reference_values_bytes: bytes,
) -> dict[str, Any]:
    issues: list[str] = []
    observed = set(output_files)
    missing = sorted(REQUIRED_FILES - observed)
    if missing:
        issues.append("missing required files: " + ", ".join(missing))
    empty = _nonempty_required_files(output_files)
    if empty:
        issues.append("required files are empty: " + ", ".join(empty))
    if issues:
        return {"score": 0.0, "passed": False, "issues": issues, "hard_gate_failed": True}

    ref = _load_json(reference_values_bytes)

    checks = {
        "sample_size": 0.0,
        "boundaries": 0.0,
        "multiple_testing": 0.0,
        "power_curve": 0.0,
        "plots": 0.0,
        "documents": 0.0,
    }

    try:
        sample_size = _load_json(output_files["sample_size.json"])
        pred_n = _to_float(sample_size.get("per_arm_n"))
        ref_n = _to_float(ref["sample_size"]["per_arm_n"])
        if abs(pred_n - ref_n) / ref_n <= 0.03:
            checks["sample_size"] += 0.65
        for key in ["events_required", "total_n", "adjusted_total_n"]:
            if int(round(_to_float(sample_size.get(key)))) == int(ref["sample_size"][key]):
                checks["sample_size"] += 0.08
        inflation = _to_float(sample_size.get("inflation_factor"))
        if abs(inflation - _to_float(ref["sample_size"]["inflation_factor"])) <= 0.01:
            checks["sample_size"] += 0.11
        checks["sample_size"] = min(1.0, checks["sample_size"])
    except Exception as exc:
        issues.append(f"sample_size.json invalid: {exc}")

    try:
        rows = _load_csv(output_files["boundaries.csv"])
        by_look = {int(row["look"]): row for row in rows if row.get("look")}
        passed = 0
        total = 0
        for ref_row in ref["boundaries"]:
            look = int(ref_row["look"])
            row = by_look.get(look)
            if row is None:
                issues.append(f"boundaries.csv missing look {look}")
                total += 4
                continue
            total += 4
            if abs(_to_float(row.get("info_fraction")) - _to_float(ref_row["info_fraction"])) <= 0.001:
                passed += 1
            if int(round(_to_float(row.get("events_at_look")))) == int(ref_row["events_at_look"]):
                passed += 1
            if abs(_to_float(row.get("efficacy_z_boundary")) - _to_float(ref_row["efficacy_z_boundary"])) <= 0.05:
                passed += 1
            futility = str(ref_row["futility_z_boundary"]).upper()
            if futility == "NA":
                observed_futility = str(row.get("futility_z_boundary", "")).strip().upper()
                if observed_futility in {"", "NA", "NAN", "NONE"}:
                    passed += 1
            elif abs(_to_float(row.get("futility_z_boundary")) - _to_float(ref_row["futility_z_boundary"])) <= 0.05:
                passed += 1
        checks["boundaries"] = passed / total if total else 0.0
    except Exception as exc:
        issues.append(f"boundaries.csv invalid: {exc}")

    try:
        mt = _load_json(output_files["multiple_testing.json"])
        observed_eps = {
            item.get("name"): item for item in mt.get("secondary_endpoints", [])
        }
        passed = 0
        total = 0
        for ref_ep in ref["multiple_testing"]["secondary_endpoints"]:
            total += 2
            ep = observed_eps.get(ref_ep["name"])
            if ep is None:
                issues.append(f"secondary endpoint {ref_ep['name']} missing")
                continue
            if int(round(_to_float(ep.get("hochberg_rank")))) == int(ref_ep["hochberg_rank"]):
                passed += 1
            if abs(round(_to_float(ep.get("adjusted_alpha")), 4) - _to_float(ref_ep["adjusted_alpha"])) <= 0.0001:
                passed += 1
        total += 1
        if abs(_to_float(mt.get("subgroup_bonferroni_alpha")) - _to_float(ref["multiple_testing"]["subgroup_bonferroni_alpha"])) <= 0.00001:
            passed += 1
        checks["multiple_testing"] = passed / total if total else 0.0
    except Exception as exc:
        issues.append(f"multiple_testing.json invalid: {exc}")

    try:
        pc = _load_csv(output_files["power_curve.csv"])
        if len(pc) >= 5 and {"hazard_ratio", "power"}.issubset(pc[0]):
            powers = [(_to_float(row["hazard_ratio"]), _to_float(row["power"])) for row in pc]
            in_range = all(0.0 <= power <= 1.0 for _, power in powers)
            has_target = any(abs(hr - 0.73) <= 0.001 and abs(power - 0.80) <= 0.08 for hr, power in powers)
            monotone = all(powers[i][1] >= powers[i + 1][1] - 1e-9 for i in range(len(powers) - 1))
            checks["power_curve"] = (float(in_range) + float(has_target) + float(monotone)) / 3.0
        else:
            issues.append("power_curve.csv missing required structure")
    except Exception as exc:
        issues.append(f"power_curve.csv invalid: {exc}")

    checks["plots"] = (
        float(_png_ok(output_files["power_curve.png"]))
        + float(_png_ok(output_files["boundary_plot.png"]))
    ) / 2.0

    try:
        sap = _decode(output_files["SAP.md"])
        analysis_r = _decode(output_files["analysis.R"])
        required_terms = ["sample size", "o'brien", "hochberg", "futility", "overall survival"]
        term_score = sum(term in sap.lower() for term in required_terms) / len(required_terms)
        checks["documents"] = (
            float(len(sap) >= 500)
            + term_score
            + float(len(analysis_r) >= 200 and "gsDesign" in analysis_r and "protocol.json" in analysis_r)
        ) / 3.0
    except Exception as exc:
        issues.append(f"document decode invalid: {exc}")

    weights = {
        "sample_size": 0.20,
        "boundaries": 0.25,
        "multiple_testing": 0.20,
        "power_curve": 0.10,
        "plots": 0.10,
        "documents": 0.15,
    }
    score = sum(weights[name] * checks[name] for name in weights)
    passed = score >= 0.85 and not issues
    return {
        "score": score,
        "passed": passed,
        "hard_gate_failed": False,
        "component_scores": checks,
        "issues": issues,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score healthcare_sap_group_sequential_nsclc output.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    args = parser.parse_args()
    output_files = {
        path.name: path.read_bytes()
        for path in args.output_dir.iterdir()
        if path.is_file()
    }
    result = evaluate_output_bundle(
        output_files=output_files,
        reference_values_bytes=(args.reference_dir / "reference_values.json").read_bytes(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
