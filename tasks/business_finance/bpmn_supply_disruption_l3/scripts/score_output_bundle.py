"""Local wrapper for the copied L3 BPMN evaluator chain."""

from __future__ import annotations

import importlib.util
import io
import sys
import traceback
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
    _load_module("evaluate", "evaluate.py")
    _load_module("evaluate_L2", "evaluate_L2.py")
    return _load_module("evaluate_L3", "evaluate_L3.py")


def score_output_bundle(
    *,
    bpmn_path: Path,
    structural_path: Path,
    rules_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    evaluator = _load_evaluator()
    args = SimpleNamespace(
        bpmn=str(bpmn_path),
        structural=str(structural_path),
        rules=str(rules_path),
        results=str(results_path),
        output=str(results_path.parent / "evaluation_report_L3.json"),
    )

    stdout_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer):
            report = evaluator.evaluate_L3(args)
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
        "report": report,
        "stdout": stdout_buffer.getvalue(),
    }
