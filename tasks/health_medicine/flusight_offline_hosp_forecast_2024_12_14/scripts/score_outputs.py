"""Task-local scoring helpers for flusight_offline_hosp_forecast_2024_12_14."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load evaluator module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_submission_bundle(
    *,
    submission_path: Path,
    evaluator_script_path: Path,
    truth_path: Path,
    scoring_baseline_path: Path,
) -> dict[str, Any]:
    evaluator_module = _load_module(evaluator_script_path)
    raw_result = evaluator_module.evaluate(submission_path, truth_path)
    baseline = json.loads(scoring_baseline_path.read_text(encoding="utf-8"))
    baseline_wis = float(
        baseline["naive_baseline_metrics"]["primary_metric_mean_wis_95_excluding_us"]
    )
    if baseline_wis <= 0:
        raise RuntimeError(f"Invalid baseline WIS {baseline_wis}")

    candidate_wis = float(raw_result["primary_metric_mean_wis_95_excluding_us"])
    score = 1.0 - (candidate_wis / baseline_wis)
    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "baseline_wis": baseline_wis,
        "candidate_metrics": raw_result,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-path", type=Path, required=True)
    parser.add_argument("--evaluator-script-path", type=Path, required=True)
    parser.add_argument("--truth-path", type=Path, required=True)
    parser.add_argument("--scoring-baseline-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = score_submission_bundle(
        submission_path=args.submission_path,
        evaluator_script_path=args.evaluator_script_path,
        truth_path=args.truth_path,
        scoring_baseline_path=args.scoring_baseline_path,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
