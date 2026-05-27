"""Score outputs for engineering/stess_pbd_branch_airflow."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_CYCLE_COLUMNS = ["cycle_step", "branch_id", "airflow_rate_m3s"]
CYCLE_SUMMARY_TOLERANCE_M3S = 0.05


def _result(
    score: float,
    reasons: list[str],
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "score": float(score),
        "passed": score >= 1.0,
        "reasons": reasons,
        "details": details or {},
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a top-level JSON object")
    return payload


def _parse_finite_number(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _read_cycle_csv(path: Path, expected_branch_id: str) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != EXPECTED_CYCLE_COLUMNS:
            raise ValueError(
                f"target_branch_cycle.csv columns must be exactly {EXPECTED_CYCLE_COLUMNS}, "
                f"got {reader.fieldnames}"
            )

        values: list[float] = []
        for row_index, row in enumerate(reader, start=2):
            branch_id = (row.get("branch_id") or "").strip()
            if branch_id != expected_branch_id:
                raise ValueError(
                    f"row {row_index} branch_id must be {expected_branch_id!r}, got {branch_id!r}"
                )
            _parse_finite_number(row.get("cycle_step"), f"row {row_index} cycle_step")
            values.append(
                _parse_finite_number(
                    row.get("airflow_rate_m3s"),
                    f"row {row_index} airflow_rate_m3s",
                )
            )

    if len(values) < 2:
        raise ValueError("target_branch_cycle.csv must contain at least two data rows")
    return values


def evaluate_submission(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "branch_airflow_summary.json"
    cycle_path = output_dir / "target_branch_cycle.csv"
    ground_truth_path = reference_dir / "ground_truth.json"
    evaluation_contract_path = reference_dir / "evaluation_contract.json"

    missing = [
        str(path)
        for path in [
            summary_path,
            cycle_path,
            ground_truth_path,
            evaluation_contract_path,
        ]
        if not path.exists()
    ]
    if missing:
        return _result(0.0, [f"missing required files: {', '.join(missing)}"])

    try:
        summary = _load_json(summary_path)
        ground_truth = _load_json(ground_truth_path)
        evaluation_contract = _load_json(evaluation_contract_path)
    except Exception as exc:
        return _result(0.0, [f"failed to parse json inputs: {exc}"])

    required_summary_keys = evaluation_contract.get("required_summary_keys", [])
    missing_keys = [key for key in required_summary_keys if key not in summary]
    if missing_keys:
        return _result(0.0, [f"summary missing required keys: {missing_keys}"])

    reasons: list[str] = []
    details: dict[str, Any] = {
        "expected_branch_id": ground_truth.get("target_branch_id"),
        "expected_target_variable": evaluation_contract.get("target_variable"),
        "expected_unit": evaluation_contract.get("unit"),
        "expected_sign_convention": evaluation_contract.get("sign_convention"),
    }

    expected_branch_id = str(ground_truth.get("target_branch_id"))
    if summary.get("target_branch_id") != expected_branch_id:
        reasons.append(
            f"target_branch_id must be {expected_branch_id!r}, got {summary.get('target_branch_id')!r}"
        )

    expected_target_variable = evaluation_contract.get("target_variable")
    if summary.get("target_variable") != expected_target_variable:
        reasons.append(
            "target_variable must be "
            f"{expected_target_variable!r}, got {summary.get('target_variable')!r}"
        )

    expected_unit = evaluation_contract.get("unit")
    if summary.get("unit") != expected_unit:
        reasons.append(f"unit must be {expected_unit!r}, got {summary.get('unit')!r}")

    expected_sign_convention = evaluation_contract.get("sign_convention")
    if summary.get("sign_convention") != expected_sign_convention:
        reasons.append(
            "sign_convention must be "
            f"{expected_sign_convention!r}, got {summary.get('sign_convention')!r}"
        )

    try:
        summary_maximum = _parse_finite_number(summary.get("cycle_maximum"), "cycle_maximum")
        summary_minimum = _parse_finite_number(summary.get("cycle_minimum"), "cycle_minimum")
    except ValueError as exc:
        return _result(0.0, [str(exc)], details=details)

    if summary_maximum < summary_minimum:
        reasons.append("cycle_maximum must be greater than or equal to cycle_minimum")

    try:
        cycle_values = _read_cycle_csv(cycle_path, expected_branch_id)
    except ValueError as exc:
        return _result(0.0, [str(exc)], details=details)

    cycle_maximum = max(cycle_values)
    cycle_minimum = min(cycle_values)
    details.update(
        {
            "cycle_file_maximum": cycle_maximum,
            "cycle_file_minimum": cycle_minimum,
            "reported_cycle_maximum": summary_maximum,
            "reported_cycle_minimum": summary_minimum,
            "hidden_cycle_maximum": ground_truth.get("cycle_maximum"),
            "hidden_cycle_minimum": ground_truth.get("cycle_minimum"),
        }
    )

    if abs(summary_maximum - cycle_maximum) > CYCLE_SUMMARY_TOLERANCE_M3S:
        reasons.append(
            "cycle_maximum is inconsistent with target_branch_cycle.csv "
            f"(reported {summary_maximum}, csv max {cycle_maximum})"
        )
    if abs(summary_minimum - cycle_minimum) > CYCLE_SUMMARY_TOLERANCE_M3S:
        reasons.append(
            "cycle_minimum is inconsistent with target_branch_cycle.csv "
            f"(reported {summary_minimum}, csv min {cycle_minimum})"
        )

    maximum_tolerance = _parse_finite_number(
        evaluation_contract.get("cycle_maximum_tolerance_m3s"),
        "cycle_maximum_tolerance_m3s",
    )
    minimum_tolerance = _parse_finite_number(
        evaluation_contract.get("cycle_minimum_tolerance_m3s"),
        "cycle_minimum_tolerance_m3s",
    )
    hidden_maximum = _parse_finite_number(ground_truth.get("cycle_maximum"), "ground_truth cycle_maximum")
    hidden_minimum = _parse_finite_number(ground_truth.get("cycle_minimum"), "ground_truth cycle_minimum")

    if abs(summary_maximum - hidden_maximum) > maximum_tolerance:
        reasons.append(
            f"cycle_maximum differs from hidden truth by more than {maximum_tolerance} m3/s"
        )
    if abs(summary_minimum - hidden_minimum) > minimum_tolerance:
        reasons.append(
            f"cycle_minimum differs from hidden truth by more than {minimum_tolerance} m3/s"
        )

    score = 1.0 if not reasons else 0.0
    return _result(score, reasons, details=details)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_submission(
        output_dir=Path(args.output_dir),
        reference_dir=Path(args.reference_dir),
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
