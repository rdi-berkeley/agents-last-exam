from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bundle_lib import PASS_THRESHOLD, canonical_cutoffs, canonical_policy, read_backup_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Moodle gradebook closeout submission.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=False)
    return parser.parse_args()


def gate(name: str, points: float, max_points: float, detail: str, passed: bool | None = None) -> dict:
    if passed is None:
        passed = points >= max_points
    return {
        "name": name,
        "points": round(points, 2),
        "max_points": max_points,
        "passed": bool(passed),
        "detail": detail,
    }


def exact_file_match(path_a: Path, path_b: Path) -> bool:
    return path_a.read_bytes() == path_b.read_bytes()


def find_gate(gates: list[dict], name: str) -> dict:
    for gate_item in gates:
        if gate_item["name"] == name:
            return gate_item
    raise KeyError(name)


def compare_csv_rows(submission_path: Path, reference_path: Path, key_columns: list[str]) -> float:
    submission = pd.read_csv(submission_path, keep_default_na=False)
    reference = pd.read_csv(reference_path, keep_default_na=False)
    if list(submission.columns) != list(reference.columns):
        return 0.0
    submission_lookup = submission.set_index(key_columns).to_dict("index")
    reference_lookup = reference.set_index(key_columns).to_dict("index")
    keys = sorted(set(reference_lookup) | set(submission_lookup))
    if not keys:
        return 1.0
    exact = 0
    for key in keys:
        if key in submission_lookup and key in reference_lookup and submission_lookup[key] == reference_lookup[key]:
            exact += 1
    return exact / len(keys)


def main() -> int:
    args = parse_args()
    submission_dir = args.submission.resolve()
    ground_truth_dir = args.ground_truth.resolve()

    required = [
        "corrected_course.mbz",
        "final_grade_export.csv",
        "final_grade_export.xml",
        "audit_report.csv",
        "audit_report.json",
        "exception_log.csv",
        "decisions.md",
        "oneroster_package/manifest.csv",
        "oneroster_package/users.csv",
        "oneroster_package/classes.csv",
        "oneroster_package/enrollments.csv",
        "oneroster_package/lineItems.csv",
        "oneroster_package/results.csv",
    ]
    missing = [name for name in required if not (submission_dir / name).exists()]
    gates: list[dict] = []
    if missing:
        report = {
            "score": 0.0,
            "passed": False,
            "pass_threshold": PASS_THRESHOLD,
            "gates": [gate("required_artifacts", 0, 0, f"Missing artifacts: {', '.join(missing)}", passed=False)],
        }
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        else:
            print(json.dumps(report, indent=2))
        return 0

    decisions_text = (submission_dir / "decisions.md").read_text(encoding="utf-8").strip()
    if not decisions_text:
        report = {
            "score": 0.0,
            "passed": False,
            "pass_threshold": PASS_THRESHOLD,
            "gates": [
                gate(
                    "required_artifacts",
                    0,
                    0,
                    "Missing or empty required artifact: decisions.md",
                    passed=False,
                )
            ],
        }
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        else:
            print(json.dumps(report, indent=2))
        return 0

    backup_state = read_backup_state(submission_dir / "corrected_course.mbz")
    _reference_contract = json.loads((ground_truth_dir / "reference_contract.json").read_text())

    policy_points = 0.0
    expected_policy = canonical_policy()
    policy = backup_state["policy"]
    if policy.get("category_weights") == expected_policy["category_weights"]:
        policy_points += 10
    if policy.get("drop_lowest") == expected_policy["drop_lowest"]:
        policy_points += 5
    if policy.get("empty_grade_behavior") == expected_policy["empty_grade_behavior"]:
        policy_points += 5
    gates.append(gate("category_weights_drop_empty", policy_points, 20, "Policy settings for weights, drop-lowest, and empty-grade behavior."))

    workflow_points = 0.0
    if policy.get("late_policy") == expected_policy["late_policy"]:
        workflow_points += 5
    if policy.get("override_precedence") == expected_policy["override_precedence"]:
        workflow_points += 5
    if policy.get("excused_behavior") == expected_policy["excused_behavior"]:
        workflow_points += 2.5
    if backup_state["section_cutoffs"] == canonical_cutoffs():
        workflow_points += 2.5
    gates.append(gate("late_override_excused_cutoffs", workflow_points, 15, "Late penalties, overrides, excused handling, and section-specific cutoffs."))

    final_match_ratio = compare_csv_rows(
        submission_dir / "final_grade_export.csv",
        ground_truth_dir / "final_grade_export.csv",
        ["sis_student_id", "section_id"],
    )
    final_xml_exact = exact_file_match(
        submission_dir / "final_grade_export.xml",
        ground_truth_dir / "final_grade_export.xml",
    )
    final_points = 20 * final_match_ratio + (5 if final_xml_exact else 0)
    gates.append(
        gate(
            "final_grade_export_exactness",
            final_points,
            25,
            "Final registrar CSV rows and XML export exactly match the reference.",
        )
    )

    registrar_points = 0.0
    registrar_points += 7 * final_match_ratio
    oneroster_ratio = (
        compare_csv_rows(
            submission_dir / "oneroster_package" / "results.csv",
            ground_truth_dir / "oneroster_package" / "results.csv",
            ["sourcedId"],
        )
        + float(exact_file_match(submission_dir / "oneroster_package" / "lineItems.csv", ground_truth_dir / "oneroster_package" / "lineItems.csv"))
        + float(exact_file_match(submission_dir / "oneroster_package" / "classes.csv", ground_truth_dir / "oneroster_package" / "classes.csv"))
        + float(exact_file_match(submission_dir / "oneroster_package" / "users.csv", ground_truth_dir / "oneroster_package" / "users.csv"))
        + float(exact_file_match(submission_dir / "oneroster_package" / "enrollments.csv", ground_truth_dir / "oneroster_package" / "enrollments.csv"))
        + float(exact_file_match(submission_dir / "oneroster_package" / "manifest.csv", ground_truth_dir / "oneroster_package" / "manifest.csv"))
    ) / 6.0
    registrar_points += 8 * oneroster_ratio
    gates.append(gate("registrar_and_oneroster_exports", registrar_points, 15, "Registrar CSV and OneRoster package correctness."))

    audit_ratio = compare_csv_rows(
        submission_dir / "audit_report.csv",
        ground_truth_dir / "audit_report.csv",
        ["sis_student_id", "section_id"],
    )
    audit_json_exact = exact_file_match(submission_dir / "audit_report.json", ground_truth_dir / "audit_report.json")
    audit_points = 10 * audit_ratio + (5 if audit_json_exact else 0)
    gates.append(gate("audit_honesty", audit_points, 15, "Audit report matches the reference recomputation."))

    exception_ratio = compare_csv_rows(
        submission_dir / "exception_log.csv",
        ground_truth_dir / "exception_log.csv",
        ["sis_student_id", "section_id"],
    )
    gates.append(gate("exception_log_exactness", 10 * exception_ratio, 10, "Exception log rows and reasons exactly match the reference."))

    decisions_present = decisions_text.startswith("#")
    gates.append(
        gate(
            "decisions_doc_present",
            0,
            0,
            "Decisions doc is present and non-empty; prose is not exact-matched.",
            passed=decisions_present,
        )
    )

    # Compare the editable id map and final grade flags exactly against the gold backup.
    import tempfile
    from bundle_lib import extract_backup_archive

    with tempfile.TemporaryDirectory() as subdir, tempfile.TemporaryDirectory() as refdir:
        sub_path = Path(subdir)
        ref_path = Path(refdir)
        extract_backup_archive(submission_dir / "corrected_course.mbz", sub_path)
        extract_backup_archive(ground_truth_dir / "corrected_course.mbz", ref_path)
        reference_backup_contract = json.loads((ref_path / "benchmark_contract.json").read_text(encoding="utf-8"))
        immutable_failures = []
        for relative_path in reference_backup_contract["immutable_hashes"]:
            submission_file = sub_path / relative_path
            reference_file = ref_path / relative_path
            if not submission_file.exists() or not reference_file.exists() or not exact_file_match(submission_file, reference_file):
                immutable_failures.append(relative_path)

        immutable_gate = gate(
            "immutable_backup_content",
            0,
            0,
            "Immutable backup files match the benchmark contract."
            if not immutable_failures
            else f"Changed immutable files: {', '.join(immutable_failures)}",
            passed=not immutable_failures,
        )
        gates.insert(0, immutable_gate)

        if not exact_file_match(sub_path / "integration" / "id_map.csv", ref_path / "integration" / "id_map.csv"):
            registrar_gate = find_gate(gates, "registrar_and_oneroster_exports")
            registrar_gate["points"] = round(max(0.0, registrar_gate["points"] - 4.0), 2)
            registrar_gate["detail"] += " ID map mismatch against gold backup."
            registrar_gate["passed"] = False
        if not exact_file_match(sub_path / "gradebook" / "final_grade_flags.csv", ref_path / "gradebook" / "final_grade_flags.csv"):
            workflow_gate = find_gate(gates, "late_override_excused_cutoffs")
            workflow_gate["points"] = round(max(0.0, workflow_gate["points"] - 2.5), 2)
            workflow_gate["detail"] += " Final grade lock flags not fully repaired."
            workflow_gate["passed"] = False

    hard_failed = not immutable_gate["passed"]
    score = 0.0 if hard_failed else round(sum(gate_item["points"] for gate_item in gates), 2)
    passed = (score >= PASS_THRESHOLD) and not hard_failed
    report = {
        "score": score,
        "passed": passed,
        "pass_threshold": PASS_THRESHOLD,
        "hard_failed": hard_failed,
        "gates": gates,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
