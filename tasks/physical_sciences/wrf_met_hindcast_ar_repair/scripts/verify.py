#!/usr/bin/env python3
"""VM-side verifier for physical_sciences/wrf_met_hindcast_ar_repair.

Runs the submitter-authored 3-layer scorer (`reference/evaluate.py`) against
the submission directory and emits a JSON record with a single binary
pass/fail score that the AgentHLE harness can consume.

The task's Layer B requires the pinned WRF/UPP/MET container (Stage 4 work),
so this verifier scores Layer A + Layer C only and returns a score of 1.0 iff
both discriminator gates pass:
  - A_cfl_consistency
  - C_repair_allowlist

All other gates are reported in the JSON payload for diagnostics.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


DISCRIMINATOR_GATES = ("A_cfl_consistency", "C_repair_allowlist")


def _run_evaluator(
    *,
    reference_dir: Path,
    submission_dir: Path,
    starter_dir: Path,
    hidden_truth_dir: Path,
    json_out: Path,
    layer: str,
) -> subprocess.CompletedProcess:
    cmd = [
        "python3",
        str(reference_dir / "evaluate.py"),
        "--submission",
        str(submission_dir),
        "--starter",
        str(starter_dir),
        "--hidden-truth-dir",
        str(hidden_truth_dir),
        "--layer",
        layer,
        "--skip-runtime",
        "--json-out",
        str(json_out),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _gate(report: dict, key: str) -> dict:
    for g in report.get("gates", []):
        if g.get("key") == key:
            return g
    return {}


def _emit(payload: dict) -> int:
    print(json.dumps(payload))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, help="output / output_test_pos / output_test_neg")
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--submission-dir", required=True, type=Path)
    args = ap.parse_args()

    mode = args.mode
    input_dir: Path = args.input_dir
    reference_dir: Path = args.reference_dir
    submission_dir: Path = args.submission_dir
    hidden_truth_dir = reference_dir / "evaluator_only"

    if not submission_dir.is_dir():
        return _emit(
            {
                "score": 0.0,
                "mode": mode,
                "reason": f"submission dir missing: {submission_dir}",
            }
        )

    if not (submission_dir / "namelist.input").exists():
        return _emit(
            {
                "score": 0.0,
                "mode": mode,
                "reason": "submission missing namelist.input",
            }
        )

    with tempfile.TemporaryDirectory(prefix="wrf_met_eval_") as tmpdir:
        reports: dict[str, dict] = {}
        for layer in ("a", "c"):
            json_out = Path(tmpdir) / f"layer_{layer}.json"
            result = _run_evaluator(
                reference_dir=reference_dir,
                submission_dir=submission_dir,
                starter_dir=input_dir,
                hidden_truth_dir=hidden_truth_dir,
                json_out=json_out,
                layer=layer,
            )
            if not json_out.exists():
                return _emit(
                    {
                        "score": 0.0,
                        "mode": mode,
                        "reason": f"evaluate.py --layer {layer} produced no JSON report",
                        "stdout_tail": (result.stdout or "")[-500:],
                        "stderr_tail": (result.stderr or "")[-500:],
                        "return_code": result.returncode,
                    }
                )
            reports[layer] = json.loads(json_out.read_text())

        gate_states = {}
        for layer, report in reports.items():
            for g in report.get("gates", []):
                gate_states[g["key"]] = {
                    "passed": bool(g.get("passed", False)),
                    "score": float(g.get("score", 0.0)),
                    "max": float(g.get("max", 0.0)),
                    "detail": g.get("detail", ""),
                }

        discriminator = {
            k: gate_states.get(k, {"passed": False, "score": 0.0, "detail": "gate not scored"})
            for k in DISCRIMINATOR_GATES
        }
        all_pass = all(v["passed"] for v in discriminator.values())
        score = 1.0 if all_pass else 0.0

        layer_a_score = float(reports.get("a", {}).get("layer_scores", {}).get("A", 0.0))
        layer_c_score = float(reports.get("c", {}).get("layer_scores", {}).get("C", 0.0))
        layer_a_max = float(reports.get("a", {}).get("layer_max", {}).get("A", 25.0))
        layer_c_max = float(reports.get("c", {}).get("layer_max", {}).get("C", 35.0))

        return _emit(
            {
                "score": score,
                "mode": mode,
                "all_discriminator_gates_pass": all_pass,
                "discriminator": discriminator,
                "layer_a": {"score": layer_a_score, "max": layer_a_max},
                "layer_c": {"score": layer_c_score, "max": layer_c_max},
                "a_plus_c_fraction": (
                    (layer_a_score + layer_c_score) / (layer_a_max + layer_c_max)
                    if (layer_a_max + layer_c_max)
                    else 0.0
                ),
                "gates": gate_states,
            }
        )


if __name__ == "__main__":
    sys.exit(main())
