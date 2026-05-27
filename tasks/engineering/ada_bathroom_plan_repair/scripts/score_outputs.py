"""Local scorer for engineering/ada_bathroom_plan_repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_OUTPUT_FILES = (
    "layer_inventory.json",
    "extracted_original_layout.json",
    "violations_before.json",
    "repaired_layout.json",
    "changes.json",
)
REFERENCE_MAP = {
    "layer_inventory.json": "layer_inventory.json",
    "extracted_original_layout.json": "reference_original_layout.json",
    "violations_before.json": "expected_violations.json",
    "repaired_layout.json": "reference_repaired_layout.json",
    "changes.json": "changes.json",
}
FLOAT_TOLERANCE = 1e-6


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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object at the top level")
    return data


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers_close(left: Any, right: Any) -> bool:
    if not (_is_number(left) and _is_number(right)):
        return False
    left_f = float(left)
    right_f = float(right)
    return abs(left_f - right_f) <= FLOAT_TOLERANCE


def _compare_payload(
    expected: Any,
    actual: Any,
    path: str,
    reasons: list[str],
) -> None:
    if _numbers_close(expected, actual):
        return
    if type(expected) is not type(actual):
        reasons.append(f"{path}: type mismatch ({type(expected).__name__} != {type(actual).__name__})")
        return

    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            missing = sorted(expected.keys() - actual.keys())
            extra = sorted(actual.keys() - expected.keys())
            if missing:
                reasons.append(f"{path}: missing keys {missing}")
            if extra:
                reasons.append(f"{path}: unexpected keys {extra}")
        for key in expected.keys() & actual.keys():
            _compare_payload(
                expected[key],
                actual[key],
                f"{path}.{key}" if path else key,
                reasons,
            )
        return

    if isinstance(expected, list):
        if len(expected) != len(actual):
            reasons.append(f"{path}: list length mismatch {len(expected)} != {len(actual)}")
            return
        for index, (lhs, rhs) in enumerate(zip(expected, actual)):
            _compare_payload(lhs, rhs, f"{path}[{index}]", reasons)
        return

    if expected != actual:
        reasons.append(f"{path}: value mismatch expected {expected!r}, got {actual!r}")


def evaluate_submission(output_dir: Path, reference_dir: Path) -> dict[str, Any]:
    output_payloads: dict[str, Any] = {}
    reference_payloads: dict[str, Any] = {}

    for file_name in REQUIRED_OUTPUT_FILES:
        output_path = output_dir / file_name
        if not output_path.exists():
            return _result(0.0, [f"missing output file: {file_name}"])
        try:
            output_payloads[file_name] = _load_json(output_path)
        except Exception as exc:
            return _result(0.0, [f"failed to parse output/{file_name}: {exc}"])

    for file_name in REQUIRED_OUTPUT_FILES:
        reference_name = REFERENCE_MAP[file_name]
        ref_path = reference_dir / reference_name
        if not ref_path.exists():
            return _result(0.0, [f"missing reference file: {reference_name}"])
        try:
            reference_payloads[file_name] = _load_json(ref_path)
        except Exception as exc:
            return _result(0.0, [f"failed to parse reference/{reference_name}: {exc}"])

    reasons: list[str] = []
    for file_name in REQUIRED_OUTPUT_FILES:
        _compare_payload(
            reference_payloads[file_name],
            output_payloads[file_name],
            file_name,
            reasons,
        )

    if reasons:
        return _result(0.0, reasons)
    return _result(1.0, [])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
