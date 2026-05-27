"""Verify HumanStudy-Bench agent-design task outputs."""

from __future__ import annotations

import argparse
import csv
import contextlib
import importlib.util
import difflib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PHENOMENA = [
    "false consensus",
    "anchoring",
    "framing",
    "representativeness",
    "intentional action",
    "knobe",
    "asch",
    "minimal group",
    "pluralistic ignorance",
    "guessing game",
    "disjunction",
    "dictator",
    "ultimatum",
    "trust",
    "reciprocity",
]
FORBIDDEN = ["human_ground_truth", "ground_truth.json", "ground_truth_raw_response"]
BASELINE_SIM_THRESHOLD = 0.80


def _fail(reason: str) -> dict:
    return {"score": 0.0, "passed": False, "reason": reason}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))



def _log_span_seconds(text: str) -> float:
    stamps = re.findall(r"\b20\d{2}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", text)
    if len(stamps) < 2:
        return 0.0
    parsed = [datetime.fromisoformat(s.replace("T", " ")) for s in stamps]
    return (max(parsed) - min(parsed)).total_seconds()


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, re.findall(r"\w+", a), re.findall(r"\w+", b)).ratio()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _recompute_from_raw_runs(output_dir: Path, reference_dir: Path) -> list[dict[str, object]]:
    repo = reference_dir / "hsbench_repo"
    raw_root = output_dir / "results" / "raw_runs"
    if not repo.exists():
        raise FileNotFoundError("reference/hsbench_repo is missing")
    if not raw_root.exists():
        raise FileNotFoundError("results/raw_runs is missing")

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    rows: list[dict[str, object]] = []
    try:
        sys.path.insert(0, str(repo))
        # Study evaluators read data/studies/<study_id>/ground_truth.json relative to cwd.
        import os

        os.chdir(repo)
        try:
            from scipy import stats

            if not hasattr(stats, "binomtest") and hasattr(stats, "binom_test"):
                class _CompatBinomResult:
                    def __init__(self, pvalue: float) -> None:
                        self.pvalue = pvalue

                def _compat_binomtest(k, n, p=0.5, alternative="two-sided"):
                    return _CompatBinomResult(float(stats.binom_test(k, n, p, alternative=alternative)))

                stats.binomtest = _compat_binomtest
        except Exception:
            pass
        runner = _load_module(repo / "src" / "evaluation" / "evaluator_runner.py", "_hsbench_eval_runner")
        for study_dir in sorted(raw_root.glob("study_*")):
            full_benchmark = study_dir / "full_benchmark.json"
            if not full_benchmark.exists():
                continue
            study_id = study_dir.name
            benchmark_data = json.loads(full_benchmark.read_text(encoding="utf-8"))
            with contextlib.redirect_stdout(sys.stderr):
                result = runner.run_evaluator(study_id, benchmark_data)
            if result.get("error"):
                raise RuntimeError(f"{study_id} evaluator failed: {result['error']}")
            for idx, fr in enumerate(result.get("finding_results", [])):
                rows.append(
                    {
                        "study_id": study_id,
                        "finding_idx": idx,
                        "sub_study_id": str(fr.get("sub_study_id", "")),
                        "finding_id": str(fr.get("finding_id", "")),
                        "finding_score": float(fr.get("finding_score", 0.0)),
                        "n_tests": fr.get("n_tests", 0),
                    }
                )
    finally:
        import os

        os.chdir(old_cwd)
        sys.path[:] = old_path
    if not rows:
        raise RuntimeError("no recomputed finding rows produced from raw_runs")
    return rows


def verify(output_dir: Path, input_dir: Path, reference_dir: Path) -> dict:
    required = [
        output_dir / "design_rationale.md",
        output_dir / "results" / "trial_responses.csv",
        output_dir / "results" / "per_finding_metrics.csv",
        output_dir / "results" / "per_study_metrics.csv",
        output_dir / "results" / "summary.json",
        output_dir / "logs" / "run.log",
    ]
    for path in required:
        if not path.exists():
            return _fail(f"missing required file: {path.relative_to(output_dir)}")

    spec_dir = output_dir / "src" / "agents" / "custom_methods"
    specs = sorted(p for p in spec_dir.glob("*.py") if p.name != "__init__.py")
    if len(specs) != 1:
        return _fail(f"expected exactly one custom spec file, found {len(specs)}")
    spec_text = specs[0].read_text(encoding="utf-8", errors="ignore")
    for needle in FORBIDDEN:
        if needle in spec_text:
            return _fail(f"custom spec contains forbidden reference: {needle}")

    baseline_dir = input_dir / "baseline_specs"
    for baseline in baseline_dir.glob("v*.py"):
        sim = _similarity(spec_text, baseline.read_text(encoding="utf-8", errors="ignore"))
        if sim >= BASELINE_SIM_THRESHOLD:
            return _fail(f"custom spec is {sim:.2%} similar to {baseline.name}")

    rationale = (output_dir / "design_rationale.md").read_text(encoding="utf-8", errors="ignore")
    if len(rationale.split()) < 100:
        return _fail("design_rationale.md has fewer than 100 words")
    if not any(term in rationale.lower() for term in PHENOMENA):
        return _fail("design_rationale.md does not name a target social-science phenomenon")

    log_text = (output_dir / "logs" / "run.log").read_text(encoding="utf-8", errors="ignore")
    if re.search(r"(sk-ant-|ANTHROPIC_API_KEY=\\w|OPENROUTER_API_KEY=\\w|OPENAI_API_KEY=\\w)", log_text):
        return _fail("logs/run.log appears to contain an API secret")
    if _log_span_seconds(log_text) < 60.0:
        return _fail("logs/run.log wall-clock span is under 60 seconds")

    try:
        summary = json.loads((output_dir / "results" / "summary.json").read_text(encoding="utf-8"))
        trial_rows = _read_csv(output_dir / "results" / "trial_responses.csv")
        recomputed_rows = _recompute_from_raw_runs(output_dir, reference_dir)
        values = [float(r["finding_score"]) for r in recomputed_rows]
    except Exception as exc:
        return _fail(f"failed to parse output metrics: {exc}")

    if len(values) < 60:
        return _fail(f"recomputed finding metrics has too few rows: {len(values)}")
    if not trial_rows:
        return _fail("trial_responses.csv is empty")

    mean_pas = sum(values) / len(values)
    replicated = sum(1 for v in values if v >= 0.90)
    if replicated >= 56:
        pass_level = "outstanding"
    elif replicated >= 43:
        pass_level = "strong"
    elif replicated >= 25:
        pass_level = "minimum"
    else:
        pass_level = "fail"

    return {
        "score": round(mean_pas, 6),
        "passed": True,
        "pass_level": pass_level,
        "findings_replicated": replicated,
        "findings_total": len(values),
        "mean_pas": round(mean_pas, 6),
        "n_trial_rows": len(trial_rows),
        "spec_file": specs[0].name,
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
