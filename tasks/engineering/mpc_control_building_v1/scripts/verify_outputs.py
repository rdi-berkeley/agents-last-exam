"""Verify MPC building-control task outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

COP = 3.0
STEP_HOURS = 0.25
EXPECTED_TRACE_ROWS = 28 * 24 * 4
REFERENCE_BASELINE = {
    "cooling_kwh": 14.27,
    "elec_kwh": 4.76,
    "cost_usd": 0.6947,
}
REQUIRED_DATA = {
    "baseline": "baseline_data.csv",
    "mpc_energy_saving": "mpc_energy_saving_data.csv",
    "mpc_demand_response": "mpc_demand_response_data.csv",
}
REQUIRED_FILES = [
    "mpc_actions_energy_saving.csv",
    "mpc_actions_demand_response.csv",
    "rc_log_energy_saving.json",
    "rc_log_demand_response.json",
    "metrics_comparison.csv",
    "results_summary.json",
]
DATA_COLUMNS = ["hour", "cooling_w", "t_zone", "setpoint"]
METRIC_COLUMNS = [
    "label",
    "cooling_kwh",
    "elec_kwh",
    "cost_usd",
    "peak_load_kw",
    "peak_hour_avg_kw",
    "discomfort_dh",
]


def _fail(reason: str) -> dict:
    return {"score": 0.0, "passed": False, "reason": reason}


def _price(hour: int) -> float:
    if 12 <= hour <= 16:
        return 0.12
    if 17 <= hour <= 19:
        return 0.25
    return 0.06


def _finite_float(value) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite value {value!r}")
    return out


def _read_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, (dict, list)):
        raise ValueError(f"{path.name} must contain a JSON object or list")
    return data


def _last_day(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < EXPECTED_TRACE_ROWS:
        raise ValueError(f"expected at least {EXPECTED_TRACE_ROWS} rows for July 1-July 28, found {len(df)}")
    return df.tail(96).copy()


def _metrics_from_timeseries(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    missing = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    day = _last_day(df)
    for column in DATA_COLUMNS:
        day[column] = pd.to_numeric(day[column], errors="coerce")
    if day[DATA_COLUMNS].isna().any().any():
        raise ValueError(f"{path.name} has non-numeric required data")
    hours = day["hour"].astype(int).clip(0, 23)
    cooling_kw = day["cooling_w"].abs() / 1000.0
    elec_kw = cooling_kw / COP
    discomfort = (
        (day["t_zone"] - (day["setpoint"] + 1.0)).clip(lower=0)
        + ((day["setpoint"] - 1.0) - day["t_zone"]).clip(lower=0)
    ).sum() * STEP_HOURS
    on_peak = cooling_kw[hours.between(17, 19)]
    return {
        "cooling_kwh": float((cooling_kw * STEP_HOURS).sum()),
        "elec_kwh": float((elec_kw * STEP_HOURS).sum()),
        "cost_usd": float(sum(elec_kw.iloc[i] * STEP_HOURS * _price(int(hours.iloc[i])) for i in range(len(day)))),
        "peak_load_kw": float(cooling_kw.max()),
        "peak_hour_avg_kw": float(on_peak.mean()) if len(on_peak) else float(cooling_kw.mean()),
        "discomfort_dh": float(discomfort),
    }


def _score_ratio(value: float, target: float, *, better: str, slack: float = 0.0) -> float:
    if target <= 0:
        return 0.0
    if better == "lower":
        if value <= target * (1.0 + slack):
            return 1.0
        return max(0.0, 1.0 - (value - target) / target)
    if better == "higher":
        if value >= target * (1.0 - slack):
            return 1.0
        return max(0.0, value / target)
    raise ValueError(better)


def _load_reported_metrics(path: Path) -> dict[str, dict[str, float]]:
    df = pd.read_csv(path)
    missing = [c for c in METRIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"metrics_comparison.csv missing columns: {missing}")
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        label = str(row["label"]).strip()
        out[label] = {c: _finite_float(row[c]) for c in METRIC_COLUMNS if c != "label"}
    for label in REQUIRED_DATA:
        if label not in out:
            raise ValueError(f"metrics_comparison.csv missing row {label}")
    return out


def _candidate_dicts(obj) -> list[dict]:
    if isinstance(obj, dict):
        out = [obj]
        if isinstance(obj.get("metrics"), dict):
            out.append(obj["metrics"])
        return out
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(_candidate_dicts(item))
        return out
    return []


def _rc_score(logs: list) -> float:
    vals = []
    for log in logs:
        for candidate in _candidate_dicts(log):
            for key in ("mae_test", "rmse_test", "test_rmse", "rmse_test_c"):
                if key in candidate:
                    vals.append(_finite_float(candidate[key]))
                    break
            if vals:
                break
    if not vals:
        return 0.0
    err = sum(vals) / len(vals)
    if err <= 0.5:
        return 1.0
    if err <= 1.0:
        return 0.85
    if err <= 1.5:
        return 0.65
    if err <= 2.5:
        return 0.25
    return 0.0


def verify(output_dir: Path, input_dir: Path, reference_dir: Path) -> dict:
    for name in ["SFH.idf", "Denver_current_TMY.epw", "task_spec.json"]:
        if not (input_dir / name).exists():
            return _fail(f"missing input asset visible to task: {name}")
    for name in list(REQUIRED_DATA.values()) + REQUIRED_FILES:
        if not (output_dir / name).exists():
            return _fail(f"missing required output file: {name}")

    try:
        metrics = {label: _metrics_from_timeseries(output_dir / filename) for label, filename in REQUIRED_DATA.items()}
        reported = _load_reported_metrics(output_dir / "metrics_comparison.csv")
        rc_logs = [
            _read_json(output_dir / "rc_log_energy_saving.json"),
            _read_json(output_dir / "rc_log_demand_response.json"),
        ]
        summary = _read_json(output_dir / "results_summary.json")
        actions_energy = pd.read_csv(output_dir / "mpc_actions_energy_saving.csv")
        actions_dr = pd.read_csv(output_dir / "mpc_actions_demand_response.csv")
    except Exception as exc:
        return _fail(f"failed to parse outputs: {exc}")

    if len(actions_energy) < 96 or len(actions_dr) < 96:
        return _fail("MPC action files must contain at least 96 July 28 timesteps")
    if not isinstance(summary, dict) or len(json.dumps(summary)) < 200:
        return _fail("results_summary.json is too sparse")

    for label, values in metrics.items():
        for key in ["cooling_kwh", "elec_kwh", "cost_usd", "peak_load_kw", "discomfort_dh"]:
            tolerance = max(0.05, abs(values[key]) * 0.08)
            if abs(values[key] - reported[label][key]) > tolerance:
                return _fail(f"reported {label}.{key} disagrees with time series")

    baseline = metrics["baseline"]
    for key, ref in REFERENCE_BASELINE.items():
        if abs(baseline[key] - ref) > 0.10 * ref:
            return _fail(f"baseline {key}={baseline[key]:.4f} outside 10% of reference {ref}")

    energy = metrics["mpc_energy_saving"]
    demand = metrics["mpc_demand_response"]
    if energy["cost_usd"] >= baseline["cost_usd"] or demand["cost_usd"] >= baseline["cost_usd"]:
        return _fail("MPC policies do not reduce tariff cost versus baseline")
    if demand["peak_hour_avg_kw"] >= baseline["peak_hour_avg_kw"]:
        return _fail("demand-response policy does not reduce on-peak average load versus baseline")

    baseline_score = 0.20
    rc_component = 0.15 * _rc_score(rc_logs)
    energy_component = 0.25 * (
        0.55 * _score_ratio(energy["cost_usd"], 0.4546, better="lower", slack=0.05)
        + 0.25 * _score_ratio(energy["cooling_kwh"], baseline["cooling_kwh"] * 0.85, better="lower")
        + 0.20 * _score_ratio(energy["discomfort_dh"], 4.0, better="lower", slack=0.0)
    )
    demand_component = 0.30 * (
        0.55 * _score_ratio(demand["peak_hour_avg_kw"], baseline["peak_hour_avg_kw"] * 0.55, better="lower")
        + 0.25 * _score_ratio(demand["cost_usd"], 0.36, better="lower", slack=0.05)
        + 0.20 * _score_ratio(demand["discomfort_dh"], 8.5, better="lower", slack=0.0)
    )
    reporting_component = 0.10
    score = max(0.0, min(1.0, baseline_score + rc_component + energy_component + demand_component + reporting_component))
    return {
        "score": round(score, 6),
        "passed": True,
        "baseline": {k: round(v, 6) for k, v in baseline.items()},
        "energy_saving": {k: round(v, 6) for k, v in energy.items()},
        "demand_response": {k: round(v, 6) for k, v in demand.items()},
        "rc_score": round(_rc_score(rc_logs), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()
    result = verify(Path(args.output_dir), Path(args.input_dir), Path(args.reference_dir))
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
