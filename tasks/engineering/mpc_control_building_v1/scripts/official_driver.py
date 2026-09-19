"""Deterministic benchmark driver for the building MPC task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

COP = 3.0
CP_AIR = 1006.0
STEP_HOURS = 0.25
FLOW_STATES = np.array([0.0, 0.1, 0.3], dtype=float)
SUPPLY_TEMP_C = 13.0
TRACE_ROWS = 28 * 24 * 4
DRIVER_VERSION = "ale-mpc-driver-v1"
DRIVER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_controller(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("ale_candidate_controller", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load controller module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("fit_model", "predict_next_temperature", "select_action"):
        if not callable(getattr(module, name, None)):
            raise ValueError(f"controller.py must define callable {name}()")
    return module


def _canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "t_outdoor": ("t_outdoor", "t_out"),
        "solar": ("solar", "solar_w_m2"),
        "occupancy": ("occupancy", "occupied"),
    }
    out = df.copy()
    for target, choices in aliases.items():
        if target in out:
            continue
        source = next((name for name in choices if name in out), None)
        if source is None:
            raise ValueError(f"canonical baseline missing {target}")
        out[target] = out[source]
    required = ["hour", "cooling_w", "t_zone", "setpoint", *aliases]
    missing = [name for name in required if name not in out]
    if missing:
        raise ValueError(f"canonical baseline missing columns: {missing}")
    if len(out) != TRACE_ROWS:
        raise ValueError(f"canonical baseline must contain {TRACE_ROWS} rows")
    for name in required:
        out[name] = pd.to_numeric(out[name], errors="raise")
    return out.reset_index(drop=True)


def _nearest_flow(cooling_w: float, t_zone: float) -> float:
    denominator = CP_AIR * max(t_zone - SUPPLY_TEMP_C, 0.5)
    estimated = max(0.0, cooling_w) / denominator
    return float(FLOW_STATES[np.argmin(np.abs(FLOW_STATES - estimated))])


def _training_rows(baseline: pd.DataFrame) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    train_end = 20 * 96
    for index in range(train_end - 1):
        current = baseline.iloc[index]
        rows.append(
            {
                "t_zone": float(current.t_zone),
                "t_outdoor": float(current.t_outdoor),
                "solar": float(current.solar),
                "occupancy": float(current.occupancy),
                "hour_of_day": float(current.hour) % 24.0,
                "flow_kg_s": _nearest_flow(float(current.cooling_w), float(current.t_zone)),
                "next_t_zone": float(baseline.iloc[index + 1].t_zone),
            }
        )
    return rows


def _observation(row: pd.Series, t_zone: float, index: int) -> dict[str, float]:
    return {
        "step": index,
        "hour_of_day": float(row.hour) % 24.0,
        "t_zone": float(t_zone),
        "t_outdoor": float(row.t_outdoor),
        "solar": float(row.solar),
        "occupancy": float(row.occupancy),
        "setpoint": float(row.setpoint),
        "price_usd_per_kwh": _price(float(row.hour) % 24.0),
    }


def _forecast(baseline: pd.DataFrame, index: int, horizon: int = 24) -> list[dict[str, float]]:
    rows = []
    for offset in range(horizon):
        row = baseline.iloc[min(index + offset, len(baseline) - 1)]
        rows.append(
            {
                "hour_of_day": float(row.hour) % 24.0,
                "t_outdoor": float(row.t_outdoor),
                "solar": float(row.solar),
                "occupancy": float(row.occupancy),
                "setpoint": float(row.setpoint),
                "price_usd_per_kwh": _price(float(row.hour) % 24.0),
            }
        )
    return rows


def _price(hour: float) -> float:
    if 17.0 <= hour < 20.0:
        return 0.25
    if 12.0 <= hour < 17.0:
        return 0.12
    return 0.06


def _validate_flow(value: Any) -> float:
    flow = float(value)
    if not math.isfinite(flow):
        raise ValueError("controller returned a non-finite flow")
    nearest = float(FLOW_STATES[np.argmin(np.abs(FLOW_STATES - flow))])
    if abs(flow - nearest) > 1e-8:
        raise ValueError(f"controller flow must be one of {FLOW_STATES.tolist()}: {flow}")
    return nearest


def _model_validation(module: ModuleType, model: Any, baseline: pd.DataFrame) -> dict[str, float]:
    errors = []
    start = 20 * 96
    end = 25 * 96
    for index in range(start, end - 1):
        row = baseline.iloc[index]
        observation = _observation(row, float(row.t_zone), index)
        observation["flow_kg_s"] = _nearest_flow(float(row.cooling_w), float(row.t_zone))
        predicted = float(module.predict_next_temperature(model, observation))
        if not math.isfinite(predicted):
            raise ValueError("predict_next_temperature returned a non-finite value")
        errors.append(predicted - float(baseline.iloc[index + 1].t_zone))
    values = np.asarray(errors, dtype=float)
    return {
        "rmse_test": float(np.sqrt(np.mean(values**2))),
        "mae_test": float(np.mean(np.abs(values))),
        "validation_steps": int(len(values)),
    }


def _simulate_policy(
    module: ModuleType,
    model: Any,
    baseline: pd.DataFrame,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = baseline.copy()
    output["controller"] = mode
    output["flow_kg_s"] = 0.0
    t_zone = float(baseline.iloc[0].t_zone)
    actions = []
    for index in range(len(output)):
        row = baseline.iloc[index]
        observation = _observation(row, t_zone, index)
        flow = _validate_flow(module.select_action(mode, observation, _forecast(baseline, index), model))
        cooling_w = flow * CP_AIR * max(t_zone - SUPPLY_TEMP_C, 0.0)
        output.at[index, "t_zone"] = t_zone
        output.at[index, "cooling_w"] = cooling_w
        output.at[index, "flow_kg_s"] = flow
        actions.append(
            {
                "timestep": index,
                "hour": float(row.hour),
                "t_zone": t_zone,
                "t_outdoor": float(row.t_outdoor),
                "setpoint": float(row.setpoint),
                "cooling_w": cooling_w,
                "flow_kg_s": flow,
            }
        )
        if index + 1 < len(output):
            canonical_next = float(baseline.iloc[index + 1].t_zone)
            canonical_now = float(row.t_zone)
            canonical_cooling = float(row.cooling_w)
            temperature_offset = t_zone - canonical_now
            t_zone = canonical_next + 0.88 * temperature_offset - 0.00012 * (
                cooling_w - canonical_cooling
            )
            t_zone = float(np.clip(t_zone, 15.0, 35.0))
    return output, pd.DataFrame(actions).tail(96).reset_index(drop=True)


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    day = df.tail(96).copy()
    hours = day["hour"].astype(float) % 24.0
    cooling_kw = day["cooling_w"].abs() / 1000.0
    elec_kw = cooling_kw / COP
    discomfort = (
        (day["t_zone"] - (day["setpoint"] + 1.0)).clip(lower=0)
        + ((day["setpoint"] - 1.0) - day["t_zone"]).clip(lower=0)
    ).sum() * STEP_HOURS
    on_peak = elec_kw[(hours >= 17.0) & (hours < 20.0)]
    return {
        "cooling_kwh": float((cooling_kw * STEP_HOURS).sum()),
        "elec_kwh": float((elec_kw * STEP_HOURS).sum()),
        "cost_usd": float(
            sum(elec_kw.iloc[i] * STEP_HOURS * _price(float(hours.iloc[i])) for i in range(len(day)))
        ),
        "peak_load_kw": float(cooling_kw.max()),
        "peak_hour_avg_kw": float(on_peak.mean()),
        "discomfort_dh": float(discomfort),
    }


def run(controller_path: Path, baseline_path: Path, output_dir: Path) -> dict[str, Any]:
    module = _load_controller(controller_path)
    baseline = _canonical_columns(pd.read_csv(baseline_path))
    model = module.fit_model(_training_rows(baseline))
    json.dumps(model)
    validation = _model_validation(module, model, baseline)
    energy, energy_actions = _simulate_policy(module, model, baseline, "energy_saving")
    demand, demand_actions = _simulate_policy(module, model, baseline, "demand_response")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(output_dir / "baseline_data.csv", index=False)
    energy.to_csv(output_dir / "mpc_energy_saving_data.csv", index=False)
    demand.to_csv(output_dir / "mpc_demand_response_data.csv", index=False)
    energy_actions.to_csv(output_dir / "mpc_actions_energy_saving.csv", index=False)
    demand_actions.to_csv(output_dir / "mpc_actions_demand_response.csv", index=False)
    rc_log = {"model": model, "metrics": validation, "driver_version": DRIVER_VERSION}
    (output_dir / "rc_log_energy_saving.json").write_text(json.dumps(rc_log, indent=2))
    (output_dir / "rc_log_demand_response.json").write_text(json.dumps(rc_log, indent=2))

    metrics = {
        "baseline": _metrics(baseline),
        "mpc_energy_saving": _metrics(energy),
        "mpc_demand_response": _metrics(demand),
    }
    pd.DataFrame([{"label": label, **values} for label, values in metrics.items()]).to_csv(
        output_dir / "metrics_comparison.csv", index=False
    )
    controller_sha = hashlib.sha256(controller_path.read_bytes()).hexdigest()
    summary = {
        "driver_version": DRIVER_VERSION,
        "driver_sha256": DRIVER_SOURCE_SHA256,
        "controller_sha256": controller_sha,
        "period": "July 1 through July 28 at 15-minute intervals",
        "plant": "deterministic EnergyPlus-generated closed-loop benchmark trace",
        "model_validation": validation,
        "metrics": metrics,
    }
    (output_dir / "results_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run(Path(args.controller), Path(args.baseline), Path(args.output_dir))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
