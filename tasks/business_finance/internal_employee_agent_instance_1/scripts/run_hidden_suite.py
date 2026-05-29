#!/usr/bin/env python
"""Run the benchmark-owned shell suite against a results.json artifact."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    return parser.parse_args()


def parse_metrics(stdout: str) -> dict[str, float | int]:
    cleaned = strip_ansi(stdout)
    total_match = re.search(r"Total:\s+(\d+)", cleaned)
    passed_match = re.search(r"Passed:\s+(\d+)", cleaned)
    failed_match = re.search(r"Failed:\s+(\d+)", cleaned)
    pass_rate_match = re.search(r"Pass rate:\s+(\d+)%", cleaned)
    return {
        "total": int(total_match.group(1)) if total_match else 0,
        "passed": int(passed_match.group(1)) if passed_match else 0,
        "failed": int(failed_match.group(1)) if failed_match else 0,
        "pass_rate": (int(pass_rate_match.group(1)) / 100.0) if pass_rate_match else 0.0,
    }


def score_results(results_path: Path, suite_path: Path, threshold: float = 0.90) -> dict[str, float | int | str]:
    if not results_path.exists():
        return {"score": 0.0, "error": f"missing_results: {results_path}"}
    if not suite_path.exists():
        return {"score": 0.0, "error": f"missing_suite: {suite_path}"}
    missing_runtime = [name for name in ("bash", "jq") if shutil.which(name) is None]
    if missing_runtime:
        return {"score": 0.0, "error": f"missing_runtime_deps: {', '.join(missing_runtime)}"}

    try:
        json.loads(results_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return {"score": 0.0, "error": f"invalid_results_json: {exc}"}

    with tempfile.TemporaryDirectory(prefix="employee_suite_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmp_results = tmpdir_path / "results.json"
        tmp_suite = tmpdir_path / "test_suite.sh"
        shutil.copy2(results_path, tmp_results)
        shutil.copy2(suite_path, tmp_suite)
        tmp_suite.chmod(0o755)

        env = os.environ.copy()
        env["TERM"] = "dumb"
        proc = subprocess.run(
            ["bash", str(tmp_suite)],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        metrics = parse_metrics(proc.stdout)
        score = 1.0 if float(metrics["pass_rate"]) >= threshold else 0.0
        return {
            **metrics,
            "threshold": threshold,
            "score": score,
            "suite_exit_code": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-1000:],
        }


def main() -> None:
    args = parse_args()
    payload = score_results(Path(args.results), Path(args.suite), args.threshold)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
