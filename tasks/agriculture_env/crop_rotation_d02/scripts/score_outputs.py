"""Repo-owned scorer for agriculture_env/crop_rotation_d02."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

EXPECTED_CRS_EPSG = 2154
ELIGIBLE_LAYER = "eligible_units"
FLAGGED_LAYER = "flagged_units"
ELIGIBLE_FILE = "eligible_units.gpkg"
FLAGGED_FILE = "flagged_units.gpkg"
ID_COL = "id_lcp"
CORE_FIELDS = [
    "obs_years",
    "unique_crops_1524",
    "max_same_run_1524",
    "recent_unique_crops_1824",
    "is_strict_alternation_1824",
    "is_persistent_monocrop_1824",
    "violation_codes",
    "risk_class",
]
FIELD_WEIGHT = 10.0
FLAGGED_COUNT_WEIGHT = 5.0
FLAGGED_SET_WEIGHT = 15.0
TOTAL_SCORE = 100.0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def list_layers_safe(path: Path) -> tuple[list[str] | None, str | None]:
    try:
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                "SELECT table_name FROM gpkg_contents ORDER BY table_name"
            ).fetchall()
        return [str(row[0]) for row in rows], None
    except Exception as exc:
        return None, f"Could not list layers in {path}: {exc}"


def table_columns(path: Path, table_name: str) -> list[str]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
    return [str(row[1]) for row in rows]


def table_row_count(path: Path, table_name: str) -> int:
    with sqlite3.connect(path) as conn:
        row = conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}").fetchone()
    return int(row[0])


def table_crs_epsg(path: Path, table_name: str) -> int | None:
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            """
            SELECT s.organization, s.organization_coordsys_id, c.srs_id
            FROM gpkg_contents AS c
            LEFT JOIN gpkg_spatial_ref_sys AS s
              ON c.srs_id = s.srs_id
            WHERE c.table_name = ?
            """,
            (table_name,),
        ).fetchone()
    if row is None:
        return None
    organization, org_epsg, srs_id = row
    if organization == "EPSG" and org_epsg is not None:
        return int(org_epsg)
    if srs_id is not None:
        return int(srs_id)
    return None


def read_rows_safe(
    path: Path,
    table_name: str,
    selected_columns: list[str],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        columns = table_columns(path, table_name)
    except Exception as exc:
        return None, f"Could not inspect `{table_name}` in {path}: {exc}"

    missing = [column for column in selected_columns if column not in columns]
    if missing:
        return None, f"Missing required columns in `{table_name}` from {path}: {missing}"

    column_expr = ", ".join(_quote_identifier(column) for column in selected_columns)
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT {column_expr} FROM {_quote_identifier(table_name)}"
            ).fetchall()
        return [dict(row) for row in rows], None
    except Exception as exc:
        return None, f"Could not read `{table_name}` from {path}: {exc}"


def exact_match_accuracy(pred_values: list[Any], gt_values: list[Any]) -> float:
    if len(pred_values) != len(gt_values):
        raise ValueError("pred_values and gt_values must have the same length")
    if not pred_values:
        return 0.0
    matches = 0
    for pred_value, gt_value in zip(pred_values, gt_values):
        if pred_value == gt_value or (pred_value is None and gt_value is None):
            matches += 1
    return matches / len(pred_values)


def _rows_to_index(rows: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {row[ID_COL]: row for row in rows}


def evaluate_output_dirs(pred_dir: Path, gt_dir: Path) -> dict[str, Any]:
    pred_eligible_path = pred_dir / ELIGIBLE_FILE
    pred_flagged_path = pred_dir / FLAGGED_FILE
    gt_eligible_path = gt_dir / ELIGIBLE_FILE
    gt_flagged_path = gt_dir / FLAGGED_FILE

    report: dict[str, Any] = {
        "paths": {
            "pred_dir": str(pred_dir),
            "gt_dir": str(gt_dir),
            "pred_eligible": str(pred_eligible_path),
            "pred_flagged": str(pred_flagged_path),
            "gt_eligible": str(gt_eligible_path),
            "gt_flagged": str(gt_flagged_path),
        },
        "gate_passed": False,
        "gate_failures": [],
        "hard_gate_checks": {},
        "core_field_scores": {},
        "flagged_scores": {},
        "summary": {},
        "total_score": 0.0,
    }
    gate_failures: list[str] = report["gate_failures"]
    hard = report["hard_gate_checks"]

    gt_layers, gt_layers_err = list_layers_safe(gt_eligible_path)
    if gt_layers_err is not None or gt_layers != [ELIGIBLE_LAYER]:
        raise RuntimeError(f"Ground truth eligible file is invalid: {gt_layers_err or gt_layers}")
    gt_eligible_rows, gt_eligible_err = read_rows_safe(
        gt_eligible_path,
        ELIGIBLE_LAYER,
        [ID_COL, *CORE_FIELDS],
    )
    if gt_eligible_err is not None or gt_eligible_rows is None:
        raise RuntimeError(gt_eligible_err)

    gt_flagged_layers, gt_flagged_layers_err = list_layers_safe(gt_flagged_path)
    if gt_flagged_layers_err is not None or gt_flagged_layers != [FLAGGED_LAYER]:
        raise RuntimeError(
            f"Ground truth flagged file is invalid: {gt_flagged_layers_err or gt_flagged_layers}"
        )
    gt_flagged_rows, gt_flagged_err = read_rows_safe(gt_flagged_path, FLAGGED_LAYER, [ID_COL])
    if gt_flagged_err is not None or gt_flagged_rows is None:
        raise RuntimeError(gt_flagged_err)

    pred_eligible_rows: list[dict[str, Any]] | None = None
    pred_flagged_rows: list[dict[str, Any]] | None = None

    hard["eligible_exists"] = pred_eligible_path.exists()
    if not hard["eligible_exists"]:
        gate_failures.append(f"Missing file: {pred_eligible_path}")
    else:
        layers, err = list_layers_safe(pred_eligible_path)
        hard["eligible_readable"] = err is None
        if err is not None:
            gate_failures.append(err)
        hard["eligible_layer_name"] = layers == [ELIGIBLE_LAYER]
        if layers != [ELIGIBLE_LAYER]:
            gate_failures.append(
                f"{pred_eligible_path.name} must contain exactly one layer named "
                f"`{ELIGIBLE_LAYER}`; found {layers}"
            )
        if err is None and layers == [ELIGIBLE_LAYER]:
            pred_eligible_rows, pred_eligible_err = read_rows_safe(
                pred_eligible_path,
                ELIGIBLE_LAYER,
                [ID_COL, *CORE_FIELDS],
            )
            hard["eligible_open"] = pred_eligible_err is None
            if pred_eligible_err is not None:
                gate_failures.append(pred_eligible_err)

    hard["flagged_exists"] = pred_flagged_path.exists()
    if not hard["flagged_exists"]:
        gate_failures.append(f"Missing file: {pred_flagged_path}")
    else:
        layers, err = list_layers_safe(pred_flagged_path)
        hard["flagged_readable"] = err is None
        if err is not None:
            gate_failures.append(err)
        hard["flagged_layer_name"] = layers == [FLAGGED_LAYER]
        if layers != [FLAGGED_LAYER]:
            gate_failures.append(
                f"{pred_flagged_path.name} must contain exactly one layer named "
                f"`{FLAGGED_LAYER}`; found {layers}"
            )
        if err is None and layers == [FLAGGED_LAYER]:
            pred_flagged_rows, pred_flagged_err = read_rows_safe(
                pred_flagged_path,
                FLAGGED_LAYER,
                [ID_COL],
            )
            hard["flagged_open"] = pred_flagged_err is None
            if pred_flagged_err is not None:
                gate_failures.append(pred_flagged_err)

    gt_eligible_count = table_row_count(gt_eligible_path, ELIGIBLE_LAYER)
    gt_flagged_count = table_row_count(gt_flagged_path, FLAGGED_LAYER)

    if pred_eligible_rows is not None:
        eligible_columns = table_columns(pred_eligible_path, ELIGIBLE_LAYER)
        hard["eligible_crs_epsg_2154"] = (
            table_crs_epsg(pred_eligible_path, ELIGIBLE_LAYER) == EXPECTED_CRS_EPSG
        )
        if not hard["eligible_crs_epsg_2154"]:
            gate_failures.append(
                f"{ELIGIBLE_FILE} CRS must be EPSG:{EXPECTED_CRS_EPSG}; "
                f"found {table_crs_epsg(pred_eligible_path, ELIGIBLE_LAYER)}"
            )

        hard["id_lcp_exists"] = ID_COL in eligible_columns
        if not hard["id_lcp_exists"]:
            gate_failures.append(f"Column `{ID_COL}` is missing from {ELIGIBLE_FILE}")
        else:
            pred_ids = [row[ID_COL] for row in pred_eligible_rows]
            hard["id_lcp_no_nulls"] = all(value is not None for value in pred_ids)
            if not hard["id_lcp_no_nulls"]:
                gate_failures.append(f"Column `{ID_COL}` in {ELIGIBLE_FILE} contains null values")
            hard["id_lcp_no_duplicates"] = len(pred_ids) == len(set(pred_ids))
            if not hard["id_lcp_no_duplicates"]:
                gate_failures.append(f"Column `{ID_COL}` in {ELIGIBLE_FILE} contains duplicates")

        missing_core = [field for field in CORE_FIELDS if field not in eligible_columns]
        hard["all_8_fields_exist"] = not missing_core
        if missing_core:
            gate_failures.append(f"Missing required computed fields in {ELIGIBLE_FILE}: {missing_core}")

        pred_eligible_count = len(pred_eligible_rows)
        hard["eligible_feature_count_matches_gt"] = pred_eligible_count == gt_eligible_count
        if not hard["eligible_feature_count_matches_gt"]:
            gate_failures.append(
                f"{ELIGIBLE_FILE} feature count mismatch: predicted {pred_eligible_count}, "
                f"ground truth {gt_eligible_count}"
            )

        gt_ids = {row[ID_COL] for row in gt_eligible_rows}
        pred_ids = {row[ID_COL] for row in pred_eligible_rows}
        hard["eligible_id_lcp_set_matches_gt"] = pred_ids == gt_ids
        if not hard["eligible_id_lcp_set_matches_gt"]:
            gate_failures.append(f"{ELIGIBLE_FILE} `{ID_COL}` set does not exactly match ground truth")

    if pred_flagged_rows is not None:
        flagged_columns = table_columns(pred_flagged_path, FLAGGED_LAYER)
        hard["flagged_crs_epsg_2154"] = (
            table_crs_epsg(pred_flagged_path, FLAGGED_LAYER) == EXPECTED_CRS_EPSG
        )
        if not hard["flagged_crs_epsg_2154"]:
            gate_failures.append(
                f"{FLAGGED_FILE} CRS must be EPSG:{EXPECTED_CRS_EPSG}; "
                f"found {table_crs_epsg(pred_flagged_path, FLAGGED_LAYER)}"
            )
        if pred_eligible_rows is not None:
            hard["flagged_schema_matches_eligible"] = flagged_columns == eligible_columns
            if not hard["flagged_schema_matches_eligible"]:
                gate_failures.append(
                    f"{FLAGGED_FILE} must use the same schema and column order as "
                    f"{ELIGIBLE_FILE}"
                )

    if gate_failures:
        report["summary"] = {
            "pred_eligible_count": len(pred_eligible_rows) if pred_eligible_rows is not None else None,
            "gt_eligible_count": gt_eligible_count,
            "pred_flagged_count": len(pred_flagged_rows) if pred_flagged_rows is not None else None,
            "gt_flagged_count": gt_flagged_count,
        }
        return report

    report["gate_passed"] = True

    pred_core_by_id = _rows_to_index(pred_eligible_rows)
    gt_core_by_id = _rows_to_index(gt_eligible_rows)
    ordered_ids = sorted(gt_core_by_id.keys(), key=lambda value: str(value))

    core_total = 0.0
    for field in CORE_FIELDS:
        pred_values = [pred_core_by_id[row_id][field] for row_id in ordered_ids]
        gt_values = [gt_core_by_id[row_id][field] for row_id in ordered_ids]
        accuracy = exact_match_accuracy(pred_values, gt_values)
        field_score = FIELD_WEIGHT * accuracy
        core_total += field_score
        report["core_field_scores"][field] = {
            "accuracy": accuracy,
            "score": field_score,
            "max_score": FIELD_WEIGHT,
        }

    pred_flagged_count = len(pred_flagged_rows)
    flagged_count_score = FLAGGED_COUNT_WEIGHT if pred_flagged_count == gt_flagged_count else 0.0
    pred_flagged_ids = {row[ID_COL] for row in pred_flagged_rows}
    gt_flagged_ids = {row[ID_COL] for row in gt_flagged_rows}
    flagged_id_set_exact = pred_flagged_ids == gt_flagged_ids
    flagged_set_score = FLAGGED_SET_WEIGHT if flagged_id_set_exact else 0.0
    flagged_total = flagged_count_score + flagged_set_score

    report["flagged_scores"] = {
        "count_score": flagged_count_score,
        "count_max_score": FLAGGED_COUNT_WEIGHT,
        "count_exact": pred_flagged_count == gt_flagged_count,
        "pred_flagged_count": pred_flagged_count,
        "gt_flagged_count": gt_flagged_count,
        "id_set_score": flagged_set_score,
        "id_set_max_score": FLAGGED_SET_WEIGHT,
        "id_set_exact": flagged_id_set_exact,
    }
    report["summary"] = {
        "pred_eligible_count": len(pred_eligible_rows),
        "gt_eligible_count": gt_eligible_count,
        "pred_flagged_count": pred_flagged_count,
        "gt_flagged_count": gt_flagged_count,
        "core_total": core_total,
        "flagged_total": flagged_total,
        "max_total_score": TOTAL_SCORE,
    }
    report["total_score"] = core_total + flagged_total
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score crop rotation GeoPackage outputs.")
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_output_dirs(pred_dir=args.pred_dir, gt_dir=args.gt_dir)
    text = json.dumps(report, indent=2 if args.pretty else None, default=_json_default)
    print(text)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
