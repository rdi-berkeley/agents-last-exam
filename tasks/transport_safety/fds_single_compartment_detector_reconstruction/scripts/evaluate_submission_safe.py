#!/usr/bin/env python
"""Evaluate a submitted FDS reconstruction CLI without passing gold-adjacent input paths."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


def load_raw_evaluator(reference_dir: Path):
    evaluator_path = reference_dir / "evaluate.py"
    spec = importlib.util.spec_from_file_location("fds_reference_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_scenario_to_scratch(source: Path, scratch_root: Path) -> Path:
    destination = scratch_root / "scenario_inputs" / source.name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--work-dir", default=None)
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    scenarios_root = reference_dir / "evaluator_only"
    references_root = reference_dir / "reference_outputs"
    evaluator = load_raw_evaluator(reference_dir)

    with tempfile.TemporaryDirectory(dir=args.work_dir, prefix="fds_eval_") as tmp:
        tmp_path = Path(tmp)
        scenario_dirs = sorted(path for path in scenarios_root.iterdir() if path.is_dir())
        results = []
        all_messages = []
        for scenario in scenario_dirs:
            safe_input = copy_scenario_to_scratch(scenario, tmp_path)
            output_root = tmp_path / "outputs"
            reference = references_root / scenario.name
            score, messages = evaluator.score_scenario(submission_dir, safe_input, reference, output_root)
            weight = 0.55 if scenario.name == "visible" else 0.45 / max(len(scenario_dirs) - 1, 1)
            results.append(
                {
                    "scenario": scenario.name,
                    "raw_score": score,
                    "weight": weight,
                    "weighted": score * weight,
                }
            )
            all_messages.extend([f"{scenario.name}: {message}" for message in messages])

    total = sum(item["weighted"] for item in results)
    report = {
        "score": round(total, 3),
        "pass": total >= 82.0,
        "scenario_scores": results,
        "messages": all_messages[:80],
    }
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
