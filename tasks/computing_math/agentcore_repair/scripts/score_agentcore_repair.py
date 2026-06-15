"""Deterministic scorer for computing_math/agentcore_repair.

Reconstructs the agent's edited agentcore package + the hidden conformance suite in a temp dir,
runs the suite in an isolated, timed subprocess, and gate-and-scores the result. Mirrors
task-intake/agentcore_repair/grade.py + reference/tests_conformance/conformance_runner.py.
No LLM judge. Anti-gaming: the suite is our own (never the workspace's tests), runs many scenarios
per behavior, and a timeout defeats an unbounded-retry "fix".
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TIER_CHECKS = [
    "step_budget",
    "arg_coercion",
    "error_recovery",
    "retry_budget",
    "unknown_tool",
    "final_answer",
    "loop_detection",
]


def score_submission(modules: dict, conformance_runner_bytes: bytes, *, timeout: int = 120):
    """modules: {relative_path: bytes} for the agent's edited agentcore package (paths are
    relative to the package root, e.g. "agent.py" or "sub/helper.py"). Returns (score in [0,1], report)."""
    ws = Path(tempfile.mkdtemp(prefix="ale_agentcore_"))
    try:
        pkg = ws / "agentcore"
        pkg.mkdir(parents=True)
        for rel, data in modules.items():
            dest = pkg / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8"))
        runner = ws / "conformance_runner.py"
        runner.write_bytes(conformance_runner_bytes)

        try:
            proc = subprocess.run(
                [sys.executable, str(runner), "--src", str(ws)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return 0.0, {"reason": "conformance run timed out (e.g. unbounded-retry fix)"}

        data = None
        for line in reversed(proc.stdout.strip().splitlines()):
            if line.strip().startswith("{"):
                data = json.loads(line)
                break
        if not data or not data.get("import_ok"):
            return 0.0, {"reason": "agentcore failed to import"}

        checks = data.get("checks", {})
        if not checks.get("happy_path"):
            return 0.0, {"reason": "happy-path regression (gate failed)", "checks": checks}

        passed = sum(1 for n in TIER_CHECKS if checks.get(n))
        score = passed / len(TIER_CHECKS)
        return score, {"score": round(score, 3), "tiers_passed": passed, "tiers_total": len(TIER_CHECKS), "checks": checks}
    finally:
        shutil.rmtree(ws, ignore_errors=True)
