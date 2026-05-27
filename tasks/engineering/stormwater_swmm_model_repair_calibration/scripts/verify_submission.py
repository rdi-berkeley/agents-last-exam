#!/usr/bin/env python
"""Task-local wrapper around the hidden stormwater SWMM evaluator."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import uuid
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--evaluator-python", type=Path, required=True)
    parser.add_argument("--evaluator-script", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - integration-only failure path
        raise RuntimeError(f"hidden evaluator report parse failed: {exc}") from exc


def _validate_report(report: dict) -> None:
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


def main() -> int:
    args = _parse_args()
    report_name = f"evaluation_report_{uuid.uuid4().hex}.json"
    if args.tmp_dir is not None:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.tmp_dir / report_name
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="stormwater_swmm_hidden_eval_")
        report_path = Path(temp_dir.name) / report_name

    cmd = [
        str(args.evaluator_python),
        str(args.evaluator_script),
        "--submission",
        str(args.submission_dir),
        "--ground-truth",
        str(args.ground_truth_dir),
        "--output",
        str(report_path),
    ]
    try:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError as exc:
            print(
                json.dumps(
                    {
                        "score": 0.0,
                        "all_passed": False,
                        "error": f"failed to launch hidden evaluator: {exc}",
                        "return_code": None,
                        "stdout": "",
                        "stderr": "",
                    },
                    indent=2,
                )
            )
            return 2

        if not report_path.exists():
            print(
                json.dumps(
                    {
                        "score": 0.0,
                        "all_passed": False,
                        "error": "hidden evaluator did not write the expected report file",
                        "return_code": proc.returncode,
                        "stdout": proc.stdout.strip(),
                        "stderr": proc.stderr.strip(),
                    },
                    indent=2,
                )
            )
            return 2

        try:
            report = _load_json(report_path)
            _validate_report(report)
            failed_gates = [gate.get("name", "<unknown>") for gate in report["gates"] if not gate.get("passed")]
            payload = {
                "score": 1.0 if bool(report.get("all_passed")) else 0.0,
                "all_passed": bool(report.get("all_passed")),
                "passed_count": int(report.get("passed_count", 0)),
                "total_gates": int(report.get("total_gates", 0)),
                "failed_gates": failed_gates,
                "return_code": proc.returncode,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "raw_report": report,
            }
            print(json.dumps(payload, indent=2))
            return 0 if payload["all_passed"] else 1
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "score": 0.0,
                        "all_passed": False,
                        "error": str(exc),
                        "return_code": proc.returncode,
                        "stdout": proc.stdout.strip(),
                        "stderr": proc.stderr.strip(),
                    },
                    indent=2,
                )
            )
            return 2
    finally:
        if "temp_dir" in locals():
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
