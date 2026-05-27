#!/usr/bin/env python3
"""Run the active-perception benchmark against a submitted solver."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not parse to a JSON object")
    return payload


def _load_benchmark_module(input_dir: Path):
    benchmark_path = input_dir / "benchmark.py"
    if str(input_dir) not in sys.path:
        sys.path.insert(0, str(input_dir))
    spec = importlib.util.spec_from_file_location("active_perception_benchmark", benchmark_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load benchmark module from {benchmark_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["active_perception_benchmark"] = module
    spec.loader.exec_module(module)
    return module


def _component_result(metric_name: str, passed: bool, value: float, threshold: float) -> dict[str, Any]:
    return {
        "metric": metric_name,
        "passed": bool(passed),
        "score": 1.0 if passed else 0.0,
        "value": float(value),
        "threshold": float(threshold),
    }


def _emit(payload: dict[str, Any], json_out: Path | None) -> int:
    text = json.dumps(payload)
    if json_out is not None:
        json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--json-out", required=False)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    reference_file = Path(args.reference_file).resolve()
    solver_path = output_dir / "active_perception_ik.py"
    json_out = Path(args.json_out).resolve() if args.json_out else None

    if not solver_path.exists():
        return _emit({"score": 0.0, "reason": f"missing solver file at {solver_path}"}, json_out)

    verification = _load_json(reference_file)
    thresholds = verification.get("pass_thresholds", {})
    env_name = str(verification.get("benchmark_environment", "mavis_base"))
    num_trials = int(verification.get("num_trials", 100))
    seed = int(verification.get("benchmark_seed", 42))

    benchmark = None
    try:
        os.chdir(input_dir)
        benchmark_module = _load_benchmark_module(input_dir)
        benchmark = benchmark_module.ActivePerceptionIKBenchmark(
            environment=env_name,
            seed=seed,
        )
        lower_bounds, upper_bounds = benchmark._get_ik_bounds()
        solver = benchmark_module.load_solver_from_file(
            str(solver_path),
            lower_bounds,
            upper_bounds,
        )
        results = benchmark.run_evaluation(solver, num_trials=num_trials)
    except Exception as exc:  # noqa: BLE001 - benchmark failures should score 0
        return _emit(
            {
                "score": 0.0,
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=20),
            },
            json_out,
        )
    finally:
        if benchmark is not None:
            try:
                benchmark.close()
            except Exception:
                pass

    avg_time = float(results["avg_computation_time_ms"])
    avg_pixel = float(results["avg_pixel_distance_after"])
    success_rate = float(results["success_rate_after"])

    time_threshold = float(thresholds["avg_computation_time_ms_max"])
    pixel_threshold = float(thresholds["avg_pixel_distance_after_max"])
    success_threshold = float(thresholds["success_rate_after_min"])
    success_pixels = float(thresholds["success_threshold_pixels"])

    breakdown = {
        "avg_computation_time_ms": _component_result(
            "avg_computation_time_ms",
            math.isfinite(avg_time) and avg_time < time_threshold,
            avg_time,
            time_threshold,
        ),
        "avg_pixel_distance_after": _component_result(
            "avg_pixel_distance_after",
            math.isfinite(avg_pixel) and avg_pixel < pixel_threshold,
            avg_pixel,
            pixel_threshold,
        ),
        "success_rate_after": _component_result(
            "success_rate_after",
            math.isfinite(success_rate) and success_rate >= success_threshold,
            success_rate,
            success_threshold,
        ),
    }

    score = sum(component["score"] for component in breakdown.values()) / len(breakdown)

    return _emit(
        {
            "score": float(score),
            "benchmark_environment": env_name,
            "num_trials": num_trials,
            "seed": seed,
            "success_threshold_pixels": success_pixels,
            "metrics": {
                "avg_computation_time_ms": avg_time,
                "avg_pixel_distance_after": avg_pixel,
                "success_rate_after": success_rate,
            },
            "breakdown": breakdown,
        },
        json_out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
