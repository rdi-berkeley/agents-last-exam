#!/usr/bin/env python
"""Task-local wrapper around the hidden SUMO calibration evaluator."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--evaluator-python", type=Path, required=True)
    parser.add_argument("--evaluator-script", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--hidden-seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    return parser.parse_args()


def _load_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"hidden evaluator returned non-JSON stdout: {exc}") from exc


def _validate_hidden_report(report: dict) -> None:
    required_fields = {
        "all_passed": bool,
        "passed_count": int,
        "total_gates": int,
        "gates": list,
    }
    missing = [field for field in required_fields if field not in report]
    if missing:
        raise RuntimeError(f"hidden evaluator report missing required fields: {', '.join(missing)}")
    for field, expected_type in required_fields.items():
        if not isinstance(report[field], expected_type):
            raise RuntimeError(
                f"hidden evaluator report field {field!r} has wrong type: "
                f"expected {expected_type.__name__}, got {type(report[field]).__name__}"
            )
    for index, gate in enumerate(report["gates"]):
        if not isinstance(gate, dict):
            raise RuntimeError(f"hidden evaluator gate #{index} is not an object")
        if "name" not in gate or "passed" not in gate:
            raise RuntimeError(f"hidden evaluator gate #{index} is missing name/passed")
        if not isinstance(gate["name"], str) or not isinstance(gate["passed"], bool):
            raise RuntimeError(f"hidden evaluator gate #{index} has malformed name/passed fields")


def _write_sumo_shim(shim_dir: Path, evaluator_python: Path) -> None:
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_path = shim_dir / "sumo"
    shim_path.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                f'exec "{evaluator_python}" -c \'import sys; from sumo import sumo; sys.exit(sumo())\' \"$@\"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    shim_path.chmod(shim_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    args = _parse_args()
    temp_root_cm = None
    env = os.environ.copy()
    evaluator_bin = str(args.evaluator_python.resolve().parent)
    if args.tmp_dir is not None:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
        scratch_root = args.tmp_dir.resolve()
    else:
        temp_root_cm = tempfile.TemporaryDirectory(prefix="sumo_urban_am_peak_calibration_")
        scratch_root = Path(temp_root_cm.name).resolve()
    env["TMPDIR"] = str(scratch_root)
    shim_dir = scratch_root / "bin"
    _write_sumo_shim(shim_dir, args.evaluator_python.resolve())
    env["PATH"] = str(shim_dir.resolve()) + os.pathsep + evaluator_bin + os.pathsep + env.get("PATH", "")
    cmd = [
        str(args.evaluator_python),
        str(args.evaluator_script),
        "--submission",
        str(args.submission_dir),
        "--ground-truth",
        str(args.ground_truth_dir),
        "--hidden-seeds",
        *[str(seed) for seed in args.hidden_seeds],
    ]
    try:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        except OSError as exc:
            payload = {
                "score": 0.0,
                "all_passed": False,
                "error": f"failed to launch hidden evaluator: {exc}",
                "return_code": None,
                "stdout": "",
                "stderr": "",
            }
            print(json.dumps(payload, indent=2))
            return 2

        try:
            report = _load_json(proc.stdout)
            _validate_hidden_report(report)
            gates = report.get("gates", [])
            failed_gates = [gate.get("name", "<unknown>") for gate in gates if not gate.get("passed")]
            payload = {
                "score": 1.0 if bool(report.get("all_passed")) else 0.0,
                "all_passed": bool(report.get("all_passed")),
                "passed_count": int(report.get("passed_count", 0)),
                "total_gates": int(report.get("total_gates", 0)),
                "failed_gates": failed_gates,
                "return_code": proc.returncode,
                "stderr": proc.stderr.strip(),
                "raw_report": report,
            }
            print(json.dumps(payload, indent=2))
            return 0 if payload["all_passed"] else 1
        except Exception as exc:
            payload = {
                "score": 0.0,
                "all_passed": False,
                "error": str(exc),
                "return_code": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
            print(json.dumps(payload, indent=2))
            return 2
    finally:
        if temp_root_cm is not None:
            temp_root_cm.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
