"""Grader for engineering/phm2010_tool_wear.

Reference: c4_wear.csv with columns cut,flute_1,flute_2,flute_3 (flank wear in micrometres) for cuts 1..315.
Prediction: output/c4_wear_pred.csv with the same columns and exactly the same 315 cuts.
Score: MAE over all 945 values, mapped linearly: MAE <= 6 um -> 1.0, MAE >= 24 um -> 0.0.
Calibration on the real reference: a constant guess scores 0; averaging the two training cutters by cut index
(no signal processing) scores about 0.3; the 2010 challenge leaders reached single-digit MAE.
Hard gates (score 0): file missing, unparseable, wrong columns, missing or duplicate cuts, non-numeric values.
"""
from __future__ import annotations

import csv
import io
import math

COLUMNS = ["cut", "flute_1", "flute_2", "flute_3"]
FULL_CREDIT_MAE_UM = 6.0
ZERO_CREDIT_MAE_UM = 24.0


def parse_wear_csv(text: str, expect_cuts: int | None = None) -> dict[int, tuple[float, float, float]] | None:
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
    except Exception:
        return None
    if not rows:
        return None
    keys = {k.strip().lower() for k in rows[0].keys() if k}
    if not set(COLUMNS).issubset(keys):
        return None
    out: dict[int, tuple[float, float, float]] = {}
    for r in rows:
        r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
        try:
            cut = int(float(r["cut"])); vals = tuple(float(r[c]) for c in COLUMNS[1:])
        except (ValueError, TypeError, KeyError):
            return None
        if any(math.isnan(v) or math.isinf(v) for v in vals) or cut in out:
            return None
        out[cut] = vals
    if expect_cuts is not None and len(out) != expect_cuts:
        return None
    return out


def score_wear(pred: dict[int, tuple[float, float, float]] | None, ref: dict[int, tuple[float, float, float]]) -> dict:
    if not pred or set(pred) != set(ref):
        return {"score": 0.0, "mae_um": None, "rmse_um": None, "gate": "missing, unparseable or wrong cut set"}
    errs = [abs(p - t) for cut in ref for p, t in zip(pred[cut], ref[cut])]
    mae = sum(errs) / len(errs); rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
    if mae <= FULL_CREDIT_MAE_UM:
        score = 1.0
    elif mae >= ZERO_CREDIT_MAE_UM:
        score = 0.0
    else:
        score = 1.0 - (mae - FULL_CREDIT_MAE_UM) / (ZERO_CREDIT_MAE_UM - FULL_CREDIT_MAE_UM)
    return {"score": round(score, 4), "mae_um": round(mae, 3), "rmse_um": round(rmse, 3), "gate": "ok"}
