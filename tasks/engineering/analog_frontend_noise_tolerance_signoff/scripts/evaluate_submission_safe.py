#!/usr/bin/env python
"""Safe scorer for analog front-end signoff submissions.

The submitted code receives scratch-copied scenario inputs, not paths adjacent
to hidden reference outputs.
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED = {
    "noise_summary.csv": [
        "case_id",
        "input_referred_noise_nv_sqrt_hz",
        "integrated_noise_uv_rms",
        "snr_db",
        "noise_status",
    ],
    "ac_stability.csv": [
        "case_id",
        "unity_gain_bandwidth_hz",
        "signal_bandwidth_hz",
        "phase_margin_deg",
        "gain_margin_db",
        "stability_status",
    ],
    "transient_settling.csv": [
        "case_id",
        "settling_time_us",
        "overshoot_pct",
        "final_gain_error_pct",
        "settling_status",
    ],
    "tolerance_corners.csv": [
        "case_id",
        "corner_id",
        "effective_gain_v_per_v",
        "worst_output_high_v",
        "worst_output_low_v",
        "corner_noise_uv_rms",
        "corner_gain_error_pct",
        "corner_phase_margin_deg",
        "corner_status",
    ],
}

NUMERIC_TOLERANCES = {
    "input_referred_noise_nv_sqrt_hz": ("rel_abs", 0.06, 0.08),
    "integrated_noise_uv_rms": ("rel_abs", 0.06, 0.08),
    "snr_db": ("abs", 0.35),
    "unity_gain_bandwidth_hz": ("rel", 0.08),
    "signal_bandwidth_hz": ("rel", 0.08),
    "phase_margin_deg": ("abs", 3.5),
    "gain_margin_db": ("abs", 1.5),
    "settling_time_us": ("rel", 0.08),
    "overshoot_pct": ("abs", 0.4),
    "final_gain_error_pct": ("abs", 0.08),
    "effective_gain_v_per_v": ("abs", 0.3),
    "worst_output_high_v": ("abs", 0.035),
    "worst_output_low_v": ("abs", 0.035),
    "corner_noise_uv_rms": ("rel_abs", 0.06, 0.08),
    "corner_gain_error_pct": ("abs", 0.08),
    "corner_phase_margin_deg": ("abs", 3.5),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: str, expected: str, column: str) -> bool:
    try:
        a = float(actual)
        e = float(expected)
    except ValueError:
        return False
    spec = NUMERIC_TOLERANCES[column]
    if spec[0] == "abs":
        return abs(a - e) <= spec[1]
    if spec[0] == "rel":
        return abs(a - e) <= max(abs(e) * spec[1], 1e-9)
    if spec[0] == "rel_abs":
        return abs(a - e) <= max(abs(e) * spec[1], spec[2])
    raise ValueError(spec[0])


def compare_csv(actual_path: Path, expected_path: Path, key_columns: list[str]) -> tuple[float, list[str]]:
    filename = expected_path.name
    if not actual_path.exists():
        return 0.0, [f"missing {filename}"]
    actual = read_csv(actual_path)
    expected = read_csv(expected_path)
    if not actual:
        return 0.0, [f"{filename} has no data rows"]
    if list(actual[0].keys()) != REQUIRED[filename]:
        return 0.0, [f"{filename} schema mismatch"]
    actual_map = {tuple(row[col] for col in key_columns): row for row in actual}
    expected_map = {tuple(row[col] for col in key_columns): row for row in expected}
    if len(actual_map) != len(actual):
        return 0.0, [f"{filename} duplicate keys"]

    pairs = [(actual_map[key], expected_row) for key, expected_row in expected_map.items() if key in actual_map]
    key_credit = len(pairs) / max(1, len(expected_map), len(actual_map))
    checks = 0
    ok = 0
    for actual_row, expected_row in pairs:
        for column in REQUIRED[filename]:
            if column in key_columns:
                continue
            checks += 1
            if column in NUMERIC_TOLERANCES:
                ok += int(close(actual_row[column], expected_row[column], column))
            else:
                ok += int(actual_row[column] == expected_row[column])
    field_credit = 1.0 if checks == 0 else ok / checks
    credit = 0.35 * key_credit + 0.65 * field_credit
    return credit, [] if math.isclose(credit, 1.0) else [f"{filename} matched {credit:.3f}"]


def check_design(path: Path, input_dir: Path) -> tuple[float, list[str]]:
    if not path.exists():
        return 0.0, ["missing design_choices.json"]
    try:
        design = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return 0.0, [f"design_choices.json invalid: {exc}"]

    allowed = {}
    for row in read_csv(input_dir / "component_bins.csv"):
        allowed[row["name"]] = set(row["allowed_values"].split("|"))

    def numeric_allowed(key: str, allowed_key: str, fmt: str) -> bool:
        try:
            return fmt.format(float(design.get(key))) in allowed[allowed_key]
        except (TypeError, ValueError):
            return False

    checks = [
        design.get("topology") == "three_opamp_instrumentation_amp_with_post_rc",
        str(design.get("gain_v_per_v")) in allowed["gain_v_per_v"]
        or numeric_allowed("gain_v_per_v", "gain_v_per_v", "{:.0f}"),
        numeric_allowed("input_resistor_ohm", "input_resistor_ohm", "{:.0f}"),
        numeric_allowed("feedback_resistor_ohm", "feedback_resistor_ohm", "{:.0f}"),
        str(design.get("filter_cap_nf")) in allowed["filter_cap_nf"]
        or numeric_allowed("filter_cap_nf", "filter_cap_nf", "{:.1f}"),
        str(design.get("comp_cap_pf")) in allowed["comp_cap_pf"]
        or numeric_allowed("comp_cap_pf", "comp_cap_pf", "{:.0f}"),
        str(design.get("bias_v")) in allowed["bias_v"]
        or numeric_allowed("bias_v", "bias_v", "{:.2f}"),
        isinstance(design.get("units"), dict),
    ]
    credit = sum(checks) / len(checks)
    return credit, [] if credit == 1.0 else ["design_choices.json missing allowed values, topology, or units"]


def compare_summary(actual_path: Path, expected_path: Path) -> tuple[float, list[str]]:
    if not actual_path.exists():
        return 0.0, ["missing signoff_summary.json"]
    try:
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return 0.0, [f"signoff_summary.json invalid: {exc}"]
    fields = [
        "scenario_id",
        "case_count",
        "corner_count",
        "limiting_noise_case",
        "limiting_phase_margin_case",
        "limiting_settling_case",
        "corner_fail_count",
        "overall_status",
    ]
    ok = sum(1 for field in fields if actual.get(field) == expected.get(field))
    return ok / len(fields), [] if ok == len(fields) else ["signoff_summary.json summary fields do not match"]


def check_netlist(path: Path) -> tuple[bool, list[str]]:
    if not path.exists() or path.stat().st_size == 0:
        return False, ["missing final_netlist.cir"]
    text = path.read_text(encoding="utf-8").lower()
    required_tokens = [
        "three_opamp_instrumentation_amp_with_post_rc",
        ".param gain=",
        "rf=475000",
        "cin=4.7n",
        "ccomp=33",
        "bias=1.65",
        "e_stage1",
        "e_stage2",
        "e_diff",
        "r_post",
        "c_post",
        ".noise",
        ".ac",
        ".tran",
        ".end",
    ]
    unresolved = ["{todo}", "todo", ".param gain={", " rf={", " cin={", " ccomp={", " bias={"]
    issues = []
    missing = [token for token in required_tokens if token not in text]
    if missing:
        issues.append("final_netlist.cir missing required topology/tokens: " + ", ".join(missing[:6]))
    if any(token in text for token in unresolved):
        issues.append("final_netlist.cir contains unresolved placeholders")
    return not issues, issues


def check_note(path: Path) -> tuple[float, list[str]]:
    if not path.exists():
        return 0.0, ["missing signoff_note.md"]
    text = path.read_text(encoding="utf-8").lower()
    headings = [
        "## architecture and gain choice",
        "## noise budget",
        "## ac stability",
        "## transient settling",
        "## tolerance corner limiting cases",
        "## ltspice compatibility",
        "## hidden scenario generalization",
    ]
    found = sum(1 for heading in headings if heading in text)
    length_ok = len(text) >= 900
    credit = (found / len(headings)) * (1.0 if length_ok else 0.65)
    return credit, [] if credit == 1.0 else ["signoff_note.md missing required headings or too short"]


def find_submission_script(submission_dir: Path) -> Path:
    candidates = [submission_dir / "run_signoff.py", submission_dir / "starter_project" / "run_signoff.py"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("run_signoff.py not found")


def run_submission(submission_dir: Path, input_dir: Path, output_dir: Path) -> None:
    script = find_submission_script(submission_dir)
    subprocess.run(
        [sys.executable, str(script), "--input-dir", str(input_dir), "--output-dir", str(output_dir)],
        cwd=str(script.parent),
        check=True,
        timeout=90,
    )


def score_scenario(outputs: Path, references: Path, input_dir: Path) -> tuple[float, list[str]]:
    score = 0.0
    issues = []
    schema_points = 10.0
    for filename, columns in REQUIRED.items():
        path = outputs / filename
        if not path.exists():
            schema_points -= 2.5
            continue
        rows = read_csv(path)
        if not rows or list(rows[0].keys()) != columns:
            schema_points -= 2.5
    netlist_ok, netlist_issues = check_netlist(outputs / "final_netlist.cir")
    if not netlist_ok:
        schema_points -= 5.0
    score += max(0.0, schema_points)
    issues.extend(netlist_issues)

    design_credit, design_issues = check_design(outputs / "design_choices.json", input_dir)
    score += 10.0 * design_credit
    issues.extend(design_issues)

    for filename, keys, weight in [
        ("noise_summary.csv", ["case_id"], 18.0),
        ("ac_stability.csv", ["case_id"], 18.0),
        ("transient_settling.csv", ["case_id"], 16.0),
        ("tolerance_corners.csv", ["case_id", "corner_id"], 18.0),
    ]:
        credit, csv_issues = compare_csv(outputs / filename, references / filename, keys)
        score += weight * credit
        issues.extend(csv_issues)

    summary_credit, summary_issues = compare_summary(outputs / "signoff_summary.json", references / "signoff_summary.json")
    score += 5.0 * summary_credit
    issues.extend(summary_issues)

    note_credit, note_issues = check_note(outputs / "signoff_note.md")
    score += 5.0 * note_credit
    issues.extend(note_issues)
    if not netlist_ok:
        score = min(score, 70.0)
    return score, issues


def copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    scenarios = [
        ("visible", reference_dir / "evaluator_only" / "visible_input", reference_dir / "reference_outputs" / "visible", 0.60),
        ("hidden", reference_dir / "evaluator_only" / "hidden_input", reference_dir / "reference_outputs" / "hidden", 0.40),
    ]

    safe_submission_dir = work_dir / "submission"
    copytree(submission_dir, safe_submission_dir)

    total = 0.0
    all_issues = []
    scenario_reports = {}
    for name, raw_input, refs, weight in scenarios:
        safe_input = work_dir / "inputs" / name
        output = work_dir / "outputs" / name
        copytree(raw_input, safe_input)
        output.mkdir(parents=True, exist_ok=True)
        try:
            run_submission(safe_submission_dir, safe_input, output)
            raw_score, issues = score_scenario(output, refs, safe_input)
        except Exception as exc:
            raw_score = 0.0
            issues = [f"{name} execution failed: {exc}"]
        weighted = raw_score * weight
        total += weighted
        all_issues.extend(f"{name}: {issue}" for issue in issues)
        scenario_reports[name] = {"raw_score": raw_score, "weighted_score": weighted, "issues": issues[:12]}

    report = {
        "score": round(total, 3),
        "pass": total >= 84.0,
        "threshold": 84.0,
        "scenarios": scenario_reports,
        "issues": all_issues[:40],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
