from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError

EXPECTED_CSV_COLUMNS = [
    "id_lcp",
    "rot_count",
    "valid_px",
    "mean_ndvi",
    "median_ndvi",
]
CSV_INT_FIELDS = ["rot_count", "valid_px"]
CSV_ROUNDED_FIELDS = ["mean_ndvi", "median_ndvi"]
TIFF_FILENAME = "ndvi.tif"
CSV_FILENAME = "polygon_ndvi_stats.csv"
TWOPLACES = Decimal("0.01")
SIX_DECIMAL_PATTERN = re.compile(r"^-?\d+(?:\.\d{6})$")


def load_csv_strict(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_int_series(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    valid = vals.notna() & np.isfinite(vals.to_numpy(dtype=float)) & (np.floor(vals) == vals)
    if valid.any():
        out.loc[valid] = vals.loc[valid].astype("int64")
    return out


def parse_rounded_token(token: Any) -> tuple[str, Any]:
    text = str(token).strip()
    if text == "NA":
        return ("NA", None)
    if text == "":
        return ("INVALID", None)
    try:
        dec = Decimal(text)
    except InvalidOperation:
        return ("INVALID", None)
    return ("NUM", dec.quantize(TWOPLACES, rounding=ROUND_HALF_UP))


def rounded_series_accuracy(pred: pd.Series, gt: pd.Series) -> float:
    correct = 0
    total = len(pred)
    for pred_value, gt_value in zip(pred.tolist(), gt.tolist()):
        pred_kind, pred_norm = parse_rounded_token(pred_value)
        gt_kind, gt_norm = parse_rounded_token(gt_value)
        if pred_kind == "NA" and gt_kind == "NA":
            correct += 1
        elif pred_kind == "NUM" and gt_kind == "NUM" and pred_norm == gt_norm:
            correct += 1
    return correct / total if total else 0.0


def round_half_up_int_array(arr: np.ndarray, decimals: int = 2) -> np.ndarray:
    factor = 10**decimals
    scaled = arr.astype(np.float64) * factor
    rounded = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return rounded.astype(np.int32)


def fail_report(gate_failures: list[str], hard_gate_checks: dict[str, bool]) -> dict[str, Any]:
    return {
        "gate_passed": False,
        "gate_failures": gate_failures,
        "hard_gate_checks": hard_gate_checks,
        "csv_scores": {},
        "tiff_scores": {},
        "summary": {"csv_score": 0.0, "tiff_score": 0.0},
        "total_points": 0.0,
        "score": 0.0,
    }


def evaluate_outputs(pred_dir: Path, gt_dir: Path) -> dict[str, Any]:
    pred_csv_path = pred_dir / CSV_FILENAME
    gt_csv_path = gt_dir / CSV_FILENAME
    pred_tif_path = pred_dir / TIFF_FILENAME
    gt_tif_path = gt_dir / TIFF_FILENAME

    gate_failures: list[str] = []
    hard_gate_checks: dict[str, bool] = {
        "pred_csv_exists_readable": False,
        "pred_tif_exists_readable": False,
        "gt_csv_exists_readable": False,
        "gt_tif_exists_readable": False,
    }

    try:
        pred_csv = load_csv_strict(pred_csv_path)
        hard_gate_checks["pred_csv_exists_readable"] = True
    except Exception as exc:
        gate_failures.append(f"Cannot read participant CSV: {exc}")
        pred_csv = None

    try:
        gt_csv = load_csv_strict(gt_csv_path)
        hard_gate_checks["gt_csv_exists_readable"] = True
    except Exception as exc:
        gate_failures.append(f"Cannot read ground-truth CSV: {exc}")
        gt_csv = None

    try:
        pred_tif = rasterio.open(pred_tif_path)
        hard_gate_checks["pred_tif_exists_readable"] = True
    except RasterioIOError as exc:
        gate_failures.append(f"Cannot read participant TIFF: {exc}")
        pred_tif = None

    try:
        gt_tif = rasterio.open(gt_tif_path)
        hard_gate_checks["gt_tif_exists_readable"] = True
    except RasterioIOError as exc:
        gate_failures.append(f"Cannot read ground-truth TIFF: {exc}")
        gt_tif = None

    if gate_failures:
        for dataset in (pred_tif, gt_tif):
            if dataset is not None:
                dataset.close()
        return fail_report(gate_failures, hard_gate_checks)

    assert pred_csv is not None
    assert gt_csv is not None
    assert pred_tif is not None
    assert gt_tif is not None

    hard_gate_checks["csv_columns_exact"] = list(pred_csv.columns) == EXPECTED_CSV_COLUMNS
    if not hard_gate_checks["csv_columns_exact"]:
        gate_failures.append(
            f"CSV columns must be exactly {EXPECTED_CSV_COLUMNS}; got {list(pred_csv.columns)}"
        )

    hard_gate_checks["id_lcp_exists"] = "id_lcp" in pred_csv.columns
    if not hard_gate_checks["id_lcp_exists"]:
        gate_failures.append("CSV is missing id_lcp column")
    else:
        pred_ids = pred_csv["id_lcp"].astype(str)
        gt_ids = gt_csv["id_lcp"].astype(str)
        sorted_ids = sorted(pred_ids.tolist())
        hard_gate_checks["id_lcp_sorted_lexicographic"] = pred_ids.tolist() == sorted_ids
        hard_gate_checks["id_lcp_no_nulls"] = not (pred_ids.str.strip() == "").any()
        hard_gate_checks["id_lcp_no_duplicates"] = not pred_ids.duplicated().any()
        hard_gate_checks["csv_row_count_matches_gt"] = len(pred_csv) == len(gt_csv)
        hard_gate_checks["id_lcp_set_matches_gt"] = set(pred_ids.tolist()) == set(
            gt_ids.tolist()
        )
        if not hard_gate_checks["id_lcp_sorted_lexicographic"]:
            gate_failures.append("CSV id_lcp values are not sorted lexicographically")
        if not hard_gate_checks["id_lcp_no_nulls"]:
            gate_failures.append("CSV id_lcp contains empty values")
        if not hard_gate_checks["id_lcp_no_duplicates"]:
            gate_failures.append("CSV id_lcp contains duplicates")
        if not hard_gate_checks["csv_row_count_matches_gt"]:
            gate_failures.append(
                f"CSV row count mismatch: pred={len(pred_csv)}, gt={len(gt_csv)}"
            )
        if not hard_gate_checks["id_lcp_set_matches_gt"]:
            gate_failures.append("CSV id_lcp set does not match ground truth exactly")

    hard_gate_checks["tiff_single_band"] = pred_tif.count == 1
    hard_gate_checks["tiff_dtype_float32"] = (
        pred_tif.dtypes[0] == "float32" if pred_tif.count >= 1 else False
    )
    hard_gate_checks["tiff_width_matches_gt"] = pred_tif.width == gt_tif.width
    hard_gate_checks["tiff_height_matches_gt"] = pred_tif.height == gt_tif.height
    hard_gate_checks["tiff_crs_matches_gt"] = pred_tif.crs == gt_tif.crs
    hard_gate_checks["tiff_transform_matches_gt"] = pred_tif.transform == gt_tif.transform

    if not hard_gate_checks["tiff_single_band"]:
        gate_failures.append(f"Participant ndvi.tif must be single-band; got {pred_tif.count}")
    if not hard_gate_checks["tiff_dtype_float32"]:
        dtype = pred_tif.dtypes[0] if pred_tif.count >= 1 else None
        gate_failures.append(f"Participant ndvi.tif dtype must be float32; got {dtype}")
    if not hard_gate_checks["tiff_width_matches_gt"]:
        gate_failures.append(f"TIFF width mismatch: pred={pred_tif.width}, gt={gt_tif.width}")
    if not hard_gate_checks["tiff_height_matches_gt"]:
        gate_failures.append(f"TIFF height mismatch: pred={pred_tif.height}, gt={gt_tif.height}")
    if not hard_gate_checks["tiff_crs_matches_gt"]:
        gate_failures.append(f"TIFF CRS mismatch: pred={pred_tif.crs}, gt={gt_tif.crs}")
    if not hard_gate_checks["tiff_transform_matches_gt"]:
        gate_failures.append("TIFF transform mismatch")

    if gate_failures:
        pred_tif.close()
        gt_tif.close()
        return fail_report(gate_failures, hard_gate_checks)

    pred_csv = pred_csv.sort_values("id_lcp").reset_index(drop=True)
    gt_csv = gt_csv.sort_values("id_lcp").reset_index(drop=True)
    merged = pred_csv.merge(gt_csv, on="id_lcp", suffixes=("_pred", "_gt"))

    csv_scores: dict[str, float] = {}
    for field in CSV_INT_FIELDS:
        pred_norm = normalize_int_series(merged[f"{field}_pred"])
        gt_norm = normalize_int_series(merged[f"{field}_gt"])
        csv_scores[field] = 10.0 * float((pred_norm == gt_norm).fillna(False).mean())

    for field in CSV_ROUNDED_FIELDS:
        acc = rounded_series_accuracy(merged[f"{field}_pred"], merged[f"{field}_gt"])
        csv_scores[field] = 20.0 * acc

    csv_score_total = sum(csv_scores.values())

    pred_arr = pred_tif.read(1)
    gt_arr = gt_tif.read(1)
    pred_tif.close()
    gt_tif.close()

    pred_finite = np.isfinite(pred_arr)
    gt_finite = np.isfinite(gt_arr)
    mask_score = 10.0 * float((pred_finite == gt_finite).mean())

    both_finite = pred_finite & gt_finite
    if both_finite.any():
        pred_q = round_half_up_int_array(pred_arr[both_finite], decimals=2)
        gt_q = round_half_up_int_array(gt_arr[both_finite], decimals=2)
        value_score = 30.0 * float((pred_q == gt_q).mean())
    else:
        value_score = 30.0

    format_ok = True
    for field in CSV_ROUNDED_FIELDS:
        values = pred_csv[field].astype(str).str.strip()
        ok = values.map(lambda value: value == "NA" or bool(SIX_DECIMAL_PATTERN.fullmatch(value)))
        if not bool(ok.all()):
            format_ok = False
            break
    format_penalty = 0.0 if format_ok else 10.0

    total_points = csv_score_total + mask_score + value_score - format_penalty
    return {
        "gate_passed": True,
        "gate_failures": [],
        "hard_gate_checks": hard_gate_checks,
        "csv_scores": csv_scores,
        "tiff_scores": {
            "mask_agreement": mask_score,
            "value_agreement": value_score,
        },
        "format_scores": {
            "six_decimal_format": format_ok,
            "format_penalty": format_penalty,
        },
        "summary": {
            "csv_score": csv_score_total,
            "tiff_score": mask_score + value_score,
            "format_penalty": format_penalty,
        },
        "total_points": total_points,
        "score": max(0.0, min(1.0, total_points / 100.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    report = evaluate_outputs(Path(args.pred_dir), Path(args.gt_dir))
    text = json.dumps(report, indent=2 if args.pretty else None, ensure_ascii=False)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
