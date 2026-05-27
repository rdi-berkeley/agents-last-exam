#!/usr/bin/env python
"""VM-side verifier for physical_sciences/wrf_met_hindcast_multi_defect_v2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


DISCRIMINATOR_GATES = (
    "A_cfl_consistency",
    "A_metgrid_levels",
    "A_pb2nc_message",
    "C_diag_D1_cfl",
    "C_diag_D2_metgrid",
    "C_diag_D3_pb2nc",
    "C_diag_evidence",
    "C_repair_allowlist",
)


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
        "python",
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
    input_dir = args.input_dir
    reference_dir = args.reference_dir
    submission_dir = args.submission_dir
    hidden_truth_dir = reference_dir / "evaluator_only"

    if not submission_dir.is_dir():
        return _emit({"score": 0.0, "mode": mode, "reason": f"submission dir missing: {submission_dir}"})

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

        gate_states: dict[str, dict] = {}
        for report in reports.values():
            for gate in report.get("gates", []):
                gate_states[gate["key"]] = {
                    "passed": bool(gate.get("passed", False)),
                    "score": float(gate.get("score", 0.0)),
                    "max": float(gate.get("max", 0.0)),
                    "detail": gate.get("detail", ""),
                }

        discriminator = {
            key: gate_states.get(key, {"passed": False, "score": 0.0, "detail": "gate not scored"})
            for key in DISCRIMINATOR_GATES
        }
        all_pass = all(gate["passed"] for gate in discriminator.values())
        score = 1.0 if all_pass else 0.0

        return _emit(
            {
                "score": score,
                "mode": mode,
                "all_discriminator_gates_pass": all_pass,
                "discriminator": discriminator,
                "layer_a": reports.get("a", {}).get("layer_scores", {}).get("A", 0.0),
                "layer_c": reports.get("c", {}).get("layer_scores", {}).get("C", 0.0),
                "gates": gate_states,
            }
        )


if __name__ == "__main__":
    sys.exit(main())
