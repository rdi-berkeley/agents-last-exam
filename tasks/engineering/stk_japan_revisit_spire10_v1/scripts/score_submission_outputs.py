"""Score STK Japan revisit outputs against the hidden reference tree."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


RAW_FOM_FILES = {
    "FOM_SimpleCoverage_Percent_Satisfied.txt": "pair",
    "FOM_Revisit_Avg_Grid_Stats.txt": "triple",
    "FOM_Revisit_Max_Grid_Stats.txt": "triple",
    "FOM_TimeAvgGap_Grid_Stats.txt": "triple",
    "FOM_Revisit_Avg_6h_Percent_Satisfied.txt": "pair",
}

JSON_METRIC_MAP = {
    ("simple_coverage", "percent_satisfied"): "simple_coverage_percent_satisfied",
    ("simple_coverage", "area_satisfied_km2"): "simple_coverage_area_satisfied_km2",
    ("average_revisit_time_sec", "minimum"): "average_revisit_min_sec",
    ("average_revisit_time_sec", "maximum"): "average_revisit_max_sec",
    ("average_revisit_time_sec", "average"): "average_revisit_avg_sec",
    ("maximum_revisit_time_sec", "minimum"): "maximum_revisit_min_sec",
    ("maximum_revisit_time_sec", "maximum"): "maximum_revisit_max_sec",
    ("maximum_revisit_time_sec", "average"): "maximum_revisit_avg_sec",
    ("time_average_gap_sec", "minimum"): "time_average_gap_min_sec",
    ("time_average_gap_sec", "maximum"): "time_average_gap_max_sec",
    ("time_average_gap_sec", "average"): "time_average_gap_avg_sec",
    ("average_revisit_le_6h", "percent_satisfied"): "average_revisit_le_6h_percent_satisfied",
    ("average_revisit_le_6h", "area_satisfied_km2"): "average_revisit_le_6h_area_satisfied_km2",
}


EXPECTED_JSON_SCHEMA = {
    "analysis_interval_utc": {
        "start": "str",
        "stop": "str",
    },
    "coverage_grid": {
        "resolution_deg": "num",
        "number_of_points": "num",
        "assets_required_for_valid_access": "num",
    },
    "simple_coverage": {
        "percent_satisfied": "num",
        "area_satisfied_km2": "num",
    },
    "average_revisit_time_sec": {
        "minimum": "num",
        "maximum": "num",
        "average": "num",
    },
    "maximum_revisit_time_sec": {
        "minimum": "num",
        "maximum": "num",
        "average": "num",
    },
    "time_average_gap_sec": {
        "minimum": "num",
        "maximum": "num",
        "average": "num",
    },
    "average_revisit_le_6h": {
        "percent_satisfied": "num",
        "area_satisfied_km2": "num",
    },
}


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _load_strict_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )


def _load_contract(reference_root: Path) -> dict:
    return _load_strict_json(reference_root / "evaluation_contract.json")


def _parse_finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be boolean")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is not finite")
    return parsed


def _load_csv_rows(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(
                f"{path.name} columns expected={expected_columns} got={reader.fieldnames}"
            )
        rows = list(reader)
    for index, row in enumerate(rows):
        if None in row:
            raise ValueError(f"{path.name} row {index} has overflow columns")
        if list(row.keys()) != expected_columns:
            raise ValueError(f"{path.name} row {index} keys drifted from header")
        if any(value is None for value in row.values()):
            raise ValueError(f"{path.name} row {index} has missing cells")
    return rows


def _load_metric_csv(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in _load_csv_rows(path, ["metric", "value"]):
        metric_name = row["metric"]
        if metric_name in metrics:
            raise ValueError(f"duplicate metric row: {metric_name}")
        metrics[metric_name] = _parse_finite_float(
            row["value"],
            label=f"{path.name} metric {metric_name}",
        )
    return metrics


def _validate_json_schema(payload: object, schema: object, *, label: str, prefix: str = "") -> None:
    if isinstance(schema, dict):
        if not isinstance(payload, dict):
            raise ValueError(f"{label} {prefix or '<root>'} is not an object")
        payload_keys = set(payload)
        schema_keys = set(schema)
        if payload_keys != schema_keys:
            raise ValueError(
                f"{label} {prefix or '<root>'} keys expected={sorted(schema_keys)} got={sorted(payload_keys)}"
            )
        for key, subschema in schema.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            _validate_json_schema(payload[key], subschema, label=label, prefix=next_prefix)
        return
    if schema == "num" and (isinstance(payload, bool) or not isinstance(payload, (int, float))):
        raise ValueError(f"{label} {prefix} is not numeric")
    if schema == "str" and not isinstance(payload, str):
        raise ValueError(f"{label} {prefix} is not a string")


def _load_metric_json_payload(path: Path) -> dict:
    payload = _load_strict_json(path)
    _validate_json_schema(payload, EXPECTED_JSON_SCHEMA, label=path.name)
    return payload


def _compare_float_map(
    actual: dict[str, float],
    expected: dict[str, float],
    tolerance: float,
    *,
    label: str,
) -> list[str]:
    reasons: list[str] = []
    actual_keys = set(actual)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        reasons.append(
            f"{label}: metric key mismatch actual={sorted(actual_keys)} expected={sorted(expected_keys)}"
        )
        return reasons
    for key in sorted(expected):
        if not math.isfinite(actual[key]) or not math.isfinite(expected[key]):
            reasons.append(f"{label}: metric {key} is not finite")
            continue
        if abs(actual[key] - expected[key]) > tolerance:
            reasons.append(
                f"{label}: metric {key} expected {expected[key]:.6f} got {actual[key]:.6f}"
            )
    return reasons


def _compare_json_payload(
    actual: object,
    expected: object,
    tolerance: float,
    *,
    label: str,
    schema: object,
    prefix: str = "",
) -> list[str]:
    reasons: list[str] = []
    if isinstance(schema, dict):
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            reasons.append(
                f"{label}: keys at {prefix or '<root>'} expected={sorted(expected_keys)} got={sorted(actual_keys)}"
            )
            return reasons
        for key, subschema in schema.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            reasons.extend(
                _compare_json_payload(
                    actual[key],
                    expected[key],
                    tolerance,
                    label=label,
                    schema=subschema,
                    prefix=next_prefix,
                )
            )
        return reasons
    if schema == "num":
        actual_num = _parse_finite_float(actual, label=f"{label} {prefix}")
        expected_num = _parse_finite_float(expected, label=f"{label} {prefix}")
        if abs(actual_num - expected_num) > tolerance:
            reasons.append(
                f"{label}: value at {prefix} expected={expected_num:.6f} got={actual_num:.6f}"
            )
        return reasons
    if schema == "str" and actual != expected:
        reasons.append(f"{label}: value at {prefix} expected={expected} got={actual}")
    return reasons


def _compare_audit_summary(
    actual_rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
    tolerance: float,
) -> list[str]:
    reasons: list[str] = []
    if len(actual_rows) != len(expected_rows):
        reasons.append(
            f"audit_access_summary.csv row count mismatch expected={len(expected_rows)} got={len(actual_rows)}"
        )
        return reasons

    actual_by_city = {row["city"]: row for row in actual_rows}
    expected_by_city = {row["city"]: row for row in expected_rows}
    if set(actual_by_city) != set(expected_by_city):
        reasons.append(
            "audit_access_summary.csv city set mismatch "
            f"expected={sorted(expected_by_city)} got={sorted(actual_by_city)}"
        )
        return reasons

    for city in sorted(expected_by_city):
        expected = expected_by_city[city]
        actual = actual_by_city[city]
        for key in ("interval_count",):
            if int(actual[key]) != int(expected[key]):
                reasons.append(
                    f"audit_access_summary.csv {actual['city']} {key} expected={expected[key]} got={actual[key]}"
                )
        for key in ("total_duration_sec", "min_duration_sec", "max_duration_sec"):
            actual_value = _parse_finite_float(
                actual[key],
                label=f"audit_access_summary.csv {actual['city']} {key}",
            )
            expected_value = _parse_finite_float(
                expected[key],
                label=f"audit_access_summary.csv {actual['city']} {key}",
            )
            if abs(actual_value - expected_value) > tolerance:
                reasons.append(
                    f"audit_access_summary.csv {actual['city']} {key} expected={expected[key]} got={actual[key]}"
                )
        for key in ("first_start_utcg", "last_stop_utcg"):
            if actual[key] != expected[key]:
                reasons.append(
                    f"audit_access_summary.csv {actual['city']} {key} expected={expected[key]} got={actual[key]}"
                )
    return reasons


def _compare_interval_rows(
    actual_rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
    tolerance: float,
    *,
    label: str,
    city_column: bool,
    ignore_order: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if len(actual_rows) != len(expected_rows):
        reasons.append(f"{label} row count mismatch expected={len(expected_rows)} got={len(actual_rows)}")
        return reasons

    keys = ["Start Time (UTCG)", "Stop Time (UTCG)", "Duration (sec)"]
    if city_column:
        keys = ["city", *keys]

    if ignore_order:
        def _sort_key(row: dict[str, str]) -> tuple[object, ...]:
            ordered: list[object] = []
            for key in keys:
                if key == "Duration (sec)":
                    ordered.append(_parse_finite_float(row[key], label=f"{label} sort duration"))
                else:
                    ordered.append(row[key])
            return tuple(ordered)

        actual_rows = sorted(actual_rows, key=_sort_key)
        expected_rows = sorted(expected_rows, key=_sort_key)

    for index, (expected, actual) in enumerate(zip(expected_rows, actual_rows, strict=True)):
        for key in keys:
            if key == "Duration (sec)":
                actual_duration = _parse_finite_float(
                    actual[key],
                    label=f"{label} row {index} duration",
                )
                expected_duration = _parse_finite_float(
                    expected[key],
                    label=f"{label} row {index} duration",
                )
                if abs(actual_duration - expected_duration) > tolerance:
                    reasons.append(
                        f"{label} row {index} duration expected={expected[key]} got={actual[key]}"
                    )
            elif actual[key] != expected[key]:
                reasons.append(
                    f"{label} row {index} {key} expected={expected[key]} got={actual[key]}"
                )
    return reasons


def _extract_fom_values(path: Path, mode: str) -> tuple[float, ...]:
    text = path.read_text(encoding="utf-8")
    numeric_lines = [
        line.strip()
        for line in text.splitlines()
        if re.fullmatch(r"[-0-9. ]+", line.strip()) and re.search(r"\d", line)
    ]
    if not numeric_lines:
        raise ValueError(f"could not parse numeric values from {path}")
    values = tuple(
        _parse_finite_float(token, label=f"{path.name} numeric token")
        for token in numeric_lines[-1].split()
    )
    if mode == "pair" and len(values) >= 2:
        return values[-2:]
    if mode == "triple" and len(values) >= 3:
        return values[-3:]
    raise ValueError(f"unexpected numeric layout in {path}: {values}")


def score_output_tree(output_root: Path, reference_root: Path) -> dict[str, object]:
    try:
        contract = _load_contract(reference_root)
        required_files = contract["required_output_files"]
        tolerance = float(contract["metric_abs_tolerance"])
        duration_tolerance = float(contract["duration_abs_tolerance_sec"])

        missing = [rel for rel in required_files if not (output_root / rel).exists()]
        if missing:
            return {
                "score": 0.0,
                "passed": False,
                "reasons": [f"missing required output files: {missing[:10]}"],
            }

        actual_files = {
            str(path.relative_to(output_root)).replace("\\", "/")
            for path in output_root.rglob("*")
            if path.is_file()
        }
        extra_files = sorted(actual_files - set(required_files))
        if extra_files:
            return {
                "score": 0.0,
                "passed": False,
                "reasons": [f"unexpected extra output files: {extra_files[:10]}"],
            }

        expected_dirs = {
            str(Path(rel).parent).replace("\\", "/")
            for rel in required_files
            if Path(rel).parent != Path(".")
        }
        actual_dirs = {
            str(path.relative_to(output_root)).replace("\\", "/")
            for path in output_root.rglob("*")
            if path.is_dir()
        }
        missing_dirs = sorted(expected_dirs - actual_dirs)
        extra_dirs = sorted(actual_dirs - expected_dirs)
        if missing_dirs or extra_dirs:
            return {
                "score": 0.0,
                "passed": False,
                "reasons": [
                    f"unexpected output directories: missing={missing_dirs[:10]} extra={extra_dirs[:10]}"
                ],
            }

        reasons: list[str] = []

        ref_outputs = reference_root / "reference_outputs"
        reasons.extend(
            _compare_float_map(
                _load_metric_csv(output_root / "derived_summaries" / "reference_metrics_summary.csv"),
                _load_metric_csv(ref_outputs / "derived_summaries" / "reference_metrics_summary.csv"),
                tolerance,
                label="reference_metrics_summary.csv",
            )
        )
        reasons.extend(
            _compare_json_payload(
                _load_metric_json_payload(
                    output_root / "derived_summaries" / "reference_metrics_summary.json"
                ),
                _load_metric_json_payload(
                    ref_outputs / "derived_summaries" / "reference_metrics_summary.json"
                ),
                tolerance,
                label="reference_metrics_summary.json",
                schema=EXPECTED_JSON_SCHEMA,
            )
        )
        reasons.extend(
            _compare_audit_summary(
                _load_csv_rows(
                    output_root / "derived_summaries" / "audit_access_summary.csv",
                    [
                        "city",
                        "interval_count",
                        "total_duration_sec",
                        "min_duration_sec",
                        "max_duration_sec",
                        "first_start_utcg",
                        "last_stop_utcg",
                    ],
                ),
                _load_csv_rows(
                    ref_outputs / "derived_summaries" / "audit_access_summary.csv",
                    [
                        "city",
                        "interval_count",
                        "total_duration_sec",
                        "min_duration_sec",
                        "max_duration_sec",
                        "first_start_utcg",
                        "last_stop_utcg",
                    ],
                ),
                duration_tolerance,
            )
        )
        reasons.extend(
            _compare_interval_rows(
                _load_csv_rows(
                    output_root / "derived_summaries" / "audit_access_intervals_combined.csv",
                    ["city", "Start Time (UTCG)", "Stop Time (UTCG)", "Duration (sec)"],
                ),
                _load_csv_rows(
                    ref_outputs / "derived_summaries" / "audit_access_intervals_combined.csv",
                    ["city", "Start Time (UTCG)", "Stop Time (UTCG)", "Duration (sec)"],
                ),
                duration_tolerance,
                label="audit_access_intervals_combined.csv",
                city_column=True,
                ignore_order=True,
            )
        )

        for city in contract["city_names"]:
            filename = f"Chain_{city}_Time_Ordered_Access.csv"
            reasons.extend(
                _compare_interval_rows(
                    _load_csv_rows(
                        output_root / "raw_stk_exports" / filename,
                        ["Start Time (UTCG)", "Stop Time (UTCG)", "Duration (sec)"],
                    ),
                    _load_csv_rows(
                        ref_outputs / "raw_stk_exports" / filename,
                        ["Start Time (UTCG)", "Stop Time (UTCG)", "Duration (sec)"],
                    ),
                    duration_tolerance,
                    label=filename,
                    city_column=False,
                )
            )

        for filename, mode in RAW_FOM_FILES.items():
            actual = _extract_fom_values(output_root / "raw_stk_exports" / filename, mode)
            expected = _extract_fom_values(ref_outputs / "raw_stk_exports" / filename, mode)
            for idx, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
                if abs(actual_value - expected_value) > tolerance:
                    reasons.append(
                        f"{filename} value[{idx}] expected={expected_value:.6f} got={actual_value:.6f}"
                    )

        passed = not reasons
        return {
            "score": 1.0 if passed else 0.0,
            "passed": passed,
            "reasons": reasons,
        }
    except Exception as exc:
        return {
            "score": 0.0,
            "passed": False,
            "reasons": [f"malformed or unreadable candidate output: {exc}"],
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = score_output_tree(args.output_root, args.reference_root)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
