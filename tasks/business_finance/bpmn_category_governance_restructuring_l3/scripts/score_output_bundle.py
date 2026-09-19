"""Local wrapper for the copied L3 BPMN evaluator chain."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import os
import sys
import traceback
import shutil
from contextlib import contextmanager
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_module(module_name: str, file_name: str):
    module_path = SCRIPTS_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_evaluator():
    return _load_module("evaluate_L3", "evaluate_L3.py")


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_scenarios(scenario_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios") if isinstance(payload, dict) else payload
    if not isinstance(scenarios, list):
        raise ValueError("scenario manifest must contain a scenarios list")
    if len(scenarios) != 60:
        raise ValueError(f"scenario manifest must contain 60 scenarios, found {len(scenarios)}")
    return scenarios


def _validate_test_results(results_path: Path, scenario_path: Path) -> dict[str, Any]:
    expected_scenarios = _load_scenarios(scenario_path)
    expected = {
        str(scenario.get("id")): str(scenario.get("category", ""))
        for scenario in expected_scenarios
    }
    if len(expected) != 60 or any(not scenario_id for scenario_id in expected):
        raise ValueError("scenario manifest has missing or duplicate scenario ids")

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("test_results.json must contain a scenarios list")
    if len(scenarios) != len(expected):
        raise ValueError(
            f"test_results.json must contain {len(expected)} scenario results, found {len(scenarios)}"
        )

    seen: set[str] = set()
    passed = 0
    summary: dict[str, dict[str, int]] = {}
    anti_total = 0
    anti_passed = 0
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("each scenario result must be an object")
        scenario_id = str(scenario.get("scenario_id") or scenario.get("id") or "")
        if scenario_id not in expected:
            raise ValueError(f"unexpected scenario result id: {scenario_id!r}")
        if scenario_id in seen:
            raise ValueError(f"duplicate scenario result id: {scenario_id}")
        seen.add(scenario_id)
        category = str(scenario.get("category", ""))
        expected_category = expected[scenario_id]
        if category != expected_category:
            raise ValueError(
                f"scenario {scenario_id} category mismatch: expected {expected_category!r}, got {category!r}"
            )
        bucket = summary.setdefault(category, {"total": 0, "passed": 0})
        bucket["total"] += 1
        scenario_passed = bool(scenario.get("pass"))
        if scenario_passed:
            passed += 1
            bucket["passed"] += 1
        if category in {"anti_gaming", "anti-gaming"}:
            anti_total += 1
            if scenario_passed:
                anti_passed += 1

    missing = sorted(set(expected) - seen)
    if missing:
        raise ValueError(f"missing scenario result ids: {missing[:5]}")
    if int(payload.get("total_scenarios", -1)) != len(expected):
        raise ValueError("total_scenarios must match the scenario manifest")
    if int(payload.get("passed_scenarios", -1)) != passed:
        raise ValueError("passed_scenarios must match the scenario list")
    reported_rate = float(payload.get("pass_rate", -1.0))
    computed_rate = passed / len(expected)
    if abs(reported_rate - computed_rate) > 1e-6:
        raise ValueError("pass_rate must match passed_scenarios / total_scenarios")
    if anti_total != 5:
        raise ValueError(f"expected 5 anti_gaming scenario results, found {anti_total}")

    reported_summary = payload.get("scenarios_summary")
    if isinstance(reported_summary, dict):
        normalized = {
            category: {
                "total": int(values.get("total", -1)),
                "passed": int(values.get("passed", -1)),
            }
            for category, values in reported_summary.items()
            if isinstance(values, dict)
        }
        if normalized != summary:
            raise ValueError("scenarios_summary must match the scenario list")

    return {
        "scenario_count": len(expected),
        "passed_scenarios": passed,
        "pass_rate": computed_rate,
        "anti_gaming_passed": anti_passed,
        "anti_gaming_total": anti_total,
    }


def score_output_bundle(
    *,
    bpmn_path: Path,
    structural_path: Path,
    rules_path: Path,
    results_path: Path,
    scenario_path: Path | None = None,
    trusted_runtime: dict[str, Any] | None = None,
    static_only: bool = False,
) -> dict[str, Any]:
    evaluator = _load_evaluator()
    bpmn_path = Path(bpmn_path).resolve()
    structural_path = Path(structural_path).resolve()
    rules_path = Path(rules_path).resolve()
    results_path = Path(results_path).resolve()
    if scenario_path is None:
        scenario_path = bpmn_path.parent.parent / "input" / "starter_project" / "test_scenarios_L3.json"
    scenario_path = Path(scenario_path).resolve()
    if trusted_runtime is None and not static_only:
        raise RuntimeError("Evaluator-owned runtime results required; static_only is diagnostic, not production scoring")
    if trusted_runtime is not None:
        if trusted_runtime.get("authority") != "evaluator_owned_flowable":
            raise RuntimeError("Untrusted runtime result authority")
        manifest = json.loads(scenario_path.read_text())
        expected_manifest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        if trusted_runtime.get("bpmn_sha256") != hashlib.sha256(bpmn_path.read_bytes()).hexdigest() or trusted_runtime.get("manifest_sha256") != expected_manifest:
            raise RuntimeError("Runtime evidence does not bind the candidate and scenario manifest")
        if trusted_runtime.get("candidate_failure"):
            return {"score": 0.0, "overall_pass": False, "runtime": trusted_runtime}
        from .runtime_validation import summarize_results, validate_manifest

        summary = summarize_results(trusted_runtime.get("scenarios"), validate_manifest(manifest))
        if any(trusted_runtime.get(key) != value for key, value in summary.items()):
            raise RuntimeError("Runtime aggregate counts differ from trusted scenario outcomes")
    try:
        validation = _validate_test_results(results_path, scenario_path)
    except Exception as exc:
        return {
            "score": 0.0,
            "overall_pass": False,
            "error": str(exc),
            "validation": "test_results_contract",
        }

    scenario_destination = results_path.parent / "starter_project" / "test_scenarios_L3.json"
    scenario_destination.parent.mkdir(parents=True, exist_ok=True)
    if scenario_path.resolve() != scenario_destination.resolve():
        shutil.copy2(scenario_path, scenario_destination)

    args = SimpleNamespace(
        bpmn=str(bpmn_path),
        structural=str(structural_path),
        rules=str(rules_path),
        results=str(results_path),
        output=str(results_path.parent / "evaluation_report_L3.json"),
        trusted_results=trusted_runtime,
    )

    stdout_buffer = io.StringIO()
    try:
        with _pushd(results_path.parent), redirect_stdout(stdout_buffer):
            report = evaluator.evaluate(args)
    except Exception as exc:
        return {
            "score": 0.0,
            "overall_pass": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "stdout": stdout_buffer.getvalue(),
        }

    return {
        "score": report.get("weighted_score", 1.0 if report.get("overall_pass") else 0.0),
        "overall_pass": bool(report.get("overall_pass")),
        "validation": validation,
        "report": report,
        "stdout": stdout_buffer.getvalue(),
        "diagnostic_only": static_only,
        "runtime": trusted_runtime,
    }
