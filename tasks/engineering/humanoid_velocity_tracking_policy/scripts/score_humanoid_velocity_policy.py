"""Score humanoid velocity tracking policy submissions.

The real performance signal comes from Isaac Lab rollout metrics. This helper
also provides structure/API gates and a hidden baseline hash fallback used only
for Stage 2 fixture replay before Isaac Lab is provisioned on the VM.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


ACTION_DIM = 23
PROPRIO_DIM = 78
COMMAND_DIM = 3
MAX_ABS_ACTION = 100.0
INIT_TIMEOUT_SECONDS = 60.0
PASS_SCORE_MAX = 0.5691
PASS_FALL_RATE_MAX = 0.05

DISALLOWED_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+(rsl_rl|stable_baselines3|sb3|ray|hydra)\b",
    re.MULTILINE,
)
DISALLOWED_RUNTIME_RE = re.compile(
    r"\b(?:subprocess|socket)\b|os\.system\s*\(|Popen\s*\(",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("agenthle_user_policy", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def _output_file_set(output_dir: Path) -> set[str]:
    return {path.name for path in output_dir.iterdir() if path.is_file()}


def _structure_gate(output_dir: Path) -> tuple[bool, str]:
    expected = {"policy.py", "checkpoint.pt"}
    observed = _output_file_set(output_dir)
    if observed != expected:
        return False, f"expected exactly {sorted(expected)}, found {sorted(observed)}"

    policy_text = (output_dir / "policy.py").read_text(encoding="utf-8", errors="replace")
    if DISALLOWED_IMPORT_RE.search(policy_text):
        return False, "policy.py imports a disallowed training/runtime package"
    if DISALLOWED_RUNTIME_RE.search(policy_text):
        return False, "policy.py appears to use subprocess/socket execution"
    return True, "structure_ok"


def _api_gate(output_dir: Path) -> tuple[bool, str, dict[str, Any]]:
    policy_path = output_dir / "policy.py"
    checkpoint_path = output_dir / "checkpoint.pt"
    module = _load_module(policy_path)
    if not hasattr(module, "Policy"):
        return False, "missing top-level Policy class", {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start = time.monotonic()
    policy = module.Policy(str(checkpoint_path), device=str(device))
    elapsed = time.monotonic() - start
    if elapsed > INIT_TIMEOUT_SECONDS:
        return False, f"Policy.__init__ took {elapsed:.2f}s", {"init_seconds": elapsed}

    command = torch.tensor(
        [
            [-1.0, -0.5, -1.0],
            [-0.25, 0.0, 0.25],
            [0.25, 0.25, -0.5],
            [1.0, 0.5, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    state = torch.linspace(
        -1.0,
        1.0,
        steps=4 * PROPRIO_DIM,
        dtype=torch.float32,
        device=device,
    ).reshape(4, PROPRIO_DIM)
    obs = {"command": command, "state": state}

    with torch.no_grad():
        first = policy.inference(obs)
        second = policy.inference(obs)

    if not isinstance(first, dict) or set(first.keys()) != {"action"}:
        return False, "Policy.inference must return exactly {'action': tensor}", {}
    action = first["action"]
    action_2 = second.get("action") if isinstance(second, dict) else None
    if action.shape != (4, ACTION_DIM):
        return False, f"action shape mismatch: {tuple(action.shape)}", {}
    if action.dtype != torch.float32:
        return False, f"action dtype mismatch: {action.dtype}", {}
    if action.device != command.device:
        return False, f"action device mismatch: {action.device} != {command.device}", {}
    if not torch.isfinite(action).all():
        return False, "action contains NaN or Inf", {}
    max_abs = float(action.abs().max().detach().cpu())
    if max_abs > MAX_ABS_ACTION:
        return False, f"action magnitude {max_abs:.4f} exceeds {MAX_ABS_ACTION}", {}
    if action_2 is None or not torch.allclose(action, action_2, atol=1e-6):
        return False, "inference is not deterministic for identical input", {}

    return True, "api_ok", {"device": str(device), "init_seconds": elapsed, "max_abs_action": max_abs}


def _check_repeat_determinism(first: dict[str, Any], second: dict[str, Any]) -> tuple[bool, str]:
    first_per_seed = first.get("per_seed")
    second_per_seed = second.get("per_seed")
    if not isinstance(first_per_seed, dict) or not isinstance(second_per_seed, dict):
        return False, "missing per_seed metrics in one or both rollout results"
    if first_per_seed.get("seeds") != second_per_seed.get("seeds"):
        return False, "per_seed seed lists differ between repeated rollout results"
    for metric in ["lin_vel_rmse_xy", "ang_vel_mae_z", "fall_rate"]:
        first_values = first_per_seed.get(metric)
        second_values = second_per_seed.get(metric)
        if not isinstance(first_values, list) or not isinstance(second_values, list):
            return False, f"missing per_seed {metric} arrays"
        if len(first_values) != len(second_values):
            return False, f"per_seed {metric} lengths differ"
        for idx, (first_value, second_value) in enumerate(zip(first_values, second_values)):
            if abs(float(first_value) - float(second_value)) > 1e-4:
                return (
                    False,
                    f"per_seed {metric}[{idx}] differs by more than 1e-4: "
                    f"{first_value} vs {second_value}",
                )
    return True, "repeat_determinism_ok"


def _score_results(results_path: Path, repeat_results_path: Path | None = None) -> dict[str, Any]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if repeat_results_path is not None:
        repeat_data = json.loads(repeat_results_path.read_text(encoding="utf-8"))
        deterministic, reason = _check_repeat_determinism(data, repeat_data)
        if not deterministic:
            return {"score": 0.0, "reason": "repeat_determinism_failed", "details": reason}
    lin = float(data["lin_vel_rmse_xy"])
    ang = float(data["ang_vel_mae_z"])
    fall = float(data["fall_rate"])
    rounded_lin = round(lin, 4)
    rounded_ang = round(ang, 4)
    rounded_fall = round(fall, 4)
    primary = round(rounded_lin + rounded_ang, 4)
    passed = rounded_fall <= PASS_FALL_RATE_MAX and primary <= PASS_SCORE_MAX
    return {
        "score": 1.0 if passed else 0.0,
        "reason": "sim_metrics_pass" if passed else "sim_metrics_fail",
        "metrics": {
            "lin_vel_rmse_xy": rounded_lin,
            "ang_vel_mae_z": rounded_ang,
            "fall_rate": rounded_fall,
            "primary_score": primary,
        },
        "determinism": "checked" if repeat_results_path is not None else "not_checked",
    }


def _matches_hidden_baseline(output_dir: Path, reference_dir: Path) -> bool:
    baseline_dir = reference_dir / "baseline"
    return (
        sha256(output_dir / "policy.py") == sha256(baseline_dir / "policy.py")
        and sha256(output_dir / "checkpoint.pt") == sha256(baseline_dir / "checkpoint.pt")
    )


def evaluate_submission(
    output_dir: Path,
    reference_dir: Path,
    *,
    results_path: Path | None = None,
    repeat_results_path: Path | None = None,
    allow_fixture_hash_fallback: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    reference_dir = reference_dir.resolve()

    structure_ok, structure_reason = _structure_gate(output_dir)
    if not structure_ok:
        return {"score": 0.0, "reason": "structure_gate_failed", "details": structure_reason}

    try:
        api_ok, api_reason, api_details = _api_gate(output_dir)
    except Exception as exc:
        return {
            "score": 0.0,
            "reason": "api_gate_exception",
            "details": f"{type(exc).__name__}: {exc}",
        }
    if not api_ok:
        return {"score": 0.0, "reason": "api_gate_failed", "details": api_reason}

    if results_path is not None and results_path.exists():
        sim_result = _score_results(results_path, repeat_results_path)
        sim_result["api_details"] = api_details
        return sim_result

    if allow_fixture_hash_fallback and _matches_hidden_baseline(output_dir, reference_dir):
        return {
            "score": 1.0,
            "reason": "stage2_hidden_baseline_fixture_match",
            "api_details": api_details,
        }

    return {
        "score": 0.0,
        "reason": "no_sim_results_available",
        "details": (
            "submission passed structure/API gates, but no Isaac Lab rollout "
            "results were available and it is not the hidden Stage 2 baseline fixture"
        ),
        "api_details": api_details,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--repeat-results-path", type=Path)
    parser.add_argument("--allow-fixture-hash-fallback", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_submission(
        args.output_dir,
        args.reference_dir,
        results_path=args.results_path,
        repeat_results_path=args.repeat_results_path,
        allow_fixture_hash_fallback=args.allow_fixture_hash_fallback,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if float(result.get("score", 0.0)) >= 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
