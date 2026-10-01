"""Score SIM1 T-shirt folding policy submissions.

The performance signal comes from the SIM1/Newton rollouts written by the
pristine ``reference/eval/run_eval.py`` on the hidden test set: 5 unseen
fabric materials (``reference/materials/test``) x 5 cloth-pose seeds each
(``reference/eval_config.json``) = 25 episodes with a 10 s policy phase. A
score is the mean per-episode ``fold_credit`` (``eval/metrics.py``), averaged
within each material first so every fabric weighs the same. Credit is
continuous in [0, 1] and is 0 for an unfolded shirt, so submissions that all
fall far short of a complete fold still rank against each other. The strict
per-material all-pass verdict is reported alongside it as ``materials_solved``.
This module also provides the structure/API gates, the input-integrity check,
and a hidden-baseline hash fallback used only for fixture replay on hosts
without the simulator.
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

import numpy as np
import torch

ACTION_DIM = 16
INIT_TIMEOUT_SECONDS = 60.0
EPISODE_SECONDS = 10.0

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


def _is_manifest_tracked(path: Path, root: Path) -> bool:
    """Interpreter- and OS-generated files must stay out of the manifest.

    ``input/eval`` is imported on the grading VM, so CPython rewrites
    ``__pycache__`` under a different magic tag than the authoring host used.
    Hashing those files makes ``verify_manifest`` report a missing input and
    zeroes every submission before any gate runs.
    """
    rel = path.relative_to(root)
    if "__pycache__" in rel.parts:
        return False
    return rel.name != ".DS_Store" and rel.suffix not in (".pyc", ".pyo")


def build_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and _is_manifest_tracked(path, root)
    }


def verify_manifest(root: Path, manifest: dict[str, str]) -> tuple[bool, str]:
    for rel, digest in manifest.items():
        path = root / rel
        if not path.is_file():
            return False, f"missing input file {rel}"
        if sha256(path) != digest:
            return False, f"input file modified: {rel}"
    return True, "inputs_intact"


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("agenthle_user_policy", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def _structure_gate(output_dir: Path) -> tuple[bool, str]:
    expected = {"policy.py", "checkpoint.pt"}
    observed = {p.name for p in output_dir.iterdir() if p.is_file()}
    if observed != expected:
        return False, f"expected exactly {sorted(expected)}, found {sorted(observed)}"
    text = (output_dir / "policy.py").read_text(encoding="utf-8", errors="replace")
    if DISALLOWED_IMPORT_RE.search(text):
        return False, "policy.py imports a disallowed training/runtime package"
    if DISALLOWED_RUNTIME_RE.search(text):
        return False, "policy.py appears to use subprocess/socket execution"
    return True, "structure_ok"


def smoke_observation(smoke_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    data = np.load(smoke_path)
    obs = {k: torch.as_tensor(data[k][None], dtype=torch.float32, device=device) for k in data.files}
    obs["time"] = torch.zeros(1, dtype=torch.float32, device=device)
    return obs


def _api_gate(output_dir: Path, smoke_path: Path) -> tuple[bool, str, dict[str, Any]]:
    module = _load_module(output_dir / "policy.py")
    if not hasattr(module, "Policy"):
        return False, "missing top-level Policy class", {}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start = time.monotonic()
    policy = module.Policy(str(output_dir / "checkpoint.pt"), device=str(device))
    elapsed = time.monotonic() - start
    if elapsed > INIT_TIMEOUT_SECONDS:
        return False, f"Policy.__init__ took {elapsed:.2f}s", {"init_seconds": elapsed}
    if not callable(getattr(policy, "reset", None)):
        return False, "Policy has no reset() method", {}

    obs = smoke_observation(smoke_path, device)
    with torch.no_grad():
        policy.reset()
        first = policy.inference(obs)
        policy.reset()
        second = policy.inference(obs)

    if not isinstance(first, dict) or set(first.keys()) != {"action"}:
        return False, "Policy.inference must return exactly {'action': tensor}", {}
    action = first["action"]
    if not isinstance(action, torch.Tensor):
        return False, "action is not a torch.Tensor", {}
    if tuple(action.shape) != (1, ACTION_DIM):
        return False, f"action shape mismatch: {tuple(action.shape)}", {}
    if action.dtype != torch.float32:
        return False, f"action dtype mismatch: {action.dtype}", {}
    if action.device != obs["cloth_pos"].device:
        return False, f"action device mismatch: {action.device}", {}
    if not torch.isfinite(action).all():
        return False, "action contains NaN or Inf", {}
    action_2 = second.get("action") if isinstance(second, dict) else None
    if action_2 is None or not torch.allclose(action, action_2, atol=1e-6):
        return False, "inference is not deterministic after reset()", {}
    return True, "api_ok", {"device": str(device), "init_seconds": elapsed}


def expected_episodes(eval_config: dict) -> list[tuple[str, int]]:
    """(material, seed) pairs of the hidden test set."""
    return sorted((m, int(s)) for m, seeds in eval_config["episodes"].items() for s in seeds)


def merge_results(paths: list[Path]) -> dict[str, Any]:
    """Merge the results.json files of several run_eval.py shards into one."""
    per_episode: list[dict] = []
    seconds = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        per_episode.extend(data.get("per_episode") or [])
        seconds.add(float(data.get("episode_seconds", -1.0)))
    return {"episode_seconds": seconds.pop() if len(seconds) == 1 else -1.0, "per_episode": per_episode}


def _score_results(results_path: Path, expected: list[tuple[str, int]]) -> dict[str, Any]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    per_episode = data.get("per_episode")
    if not isinstance(per_episode, list):
        return {"score": 0.0, "reason": "results_missing_per_episode"}
    observed = sorted((str(m.get("material")), int(m.get("seed", -1))) for m in per_episode)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        return {
            "score": 0.0,
            "reason": "results_episode_mismatch",
            "details": f"{len(observed)} episodes, {len(missing)} missing, e.g. {missing[:5]}",
        }
    if abs(float(data.get("episode_seconds", -1.0)) - EPISODE_SECONDS) > 1e-9:
        return {"score": 0.0, "reason": "results_episode_length_mismatch"}

    successes = [bool(m.get("success", False)) for m in per_episode]
    per_material: dict[str, dict[str, Any]] = {}
    for m in per_episode:
        d = per_material.setdefault(
            str(m["material"]), {"episodes": 0, "successes": 0, "credit_sum": 0.0}
        )
        d["episodes"] += 1
        d["successes"] += int(bool(m.get("success", False)))
        d["credit_sum"] += float(m.get("fold_credit", 0.0))
    for d in per_material.values():
        d["success"] = d["successes"] == d["episodes"]
        d["fold_credit"] = d.pop("credit_sum") / d["episodes"]

    # Score is the mean per-episode credit, averaged per material first so every
    # fabric weighs the same. It is continuous: a task this hard needs to rank
    # submissions that all fall far short of a complete fold.
    score = float(np.mean([d["fold_credit"] for d in per_material.values()]))
    materials_passed = sum(d["success"] for d in per_material.values())
    return {
        "score": score,
        "reason": "sim_metrics",
        "passed": materials_passed == len(per_material),
        "metrics": {
            "fold_credit": score,
            "materials_solved": materials_passed,
            "materials": len(per_material),
            "materials_solved_fraction": round(materials_passed / len(per_material), 4),
            "successes": int(sum(successes)),
            "episodes": len(successes),
            "episode_success_rate": round(float(np.mean(successes)), 4),
            "per_material": per_material,
            "per_episode": per_episode,
        },
    }


def _matches_hidden_baseline(output_dir: Path, reference_dir: Path) -> bool:
    baseline = reference_dir / "baseline"
    return sha256(output_dir / "policy.py") == sha256(baseline / "policy.py") and sha256(
        output_dir / "checkpoint.pt"
    ) == sha256(baseline / "checkpoint.pt")


def evaluate_submission(
    output_dir: Path,
    reference_dir: Path,
    smoke_obs_path: Path,
    *,
    input_integrity: tuple[bool, str],
    results_path: Path | None = None,
    allow_fixture_hash_fallback: bool = False,
) -> dict[str, Any]:
    """Score one submission.

    ``input_integrity`` is the caller's check of the staged ``input/`` tree
    against ``reference/input_manifest.json`` (``verify_manifest`` locally,
    or ``sha256sum`` on the VM).
    """
    output_dir = output_dir.resolve()
    reference_dir = reference_dir.resolve()

    intact, reason = input_integrity
    if not intact:
        return {"score": 0.0, "reason": "input_integrity_failed", "details": reason}

    ok, reason = _structure_gate(output_dir)
    if not ok:
        return {"score": 0.0, "reason": "structure_gate_failed", "details": reason}

    try:
        ok, reason, api_details = _api_gate(output_dir, smoke_obs_path)
    except Exception as exc:
        return {"score": 0.0, "reason": "api_gate_exception", "details": f"{type(exc).__name__}: {exc}"}
    if not ok:
        return {"score": 0.0, "reason": "api_gate_failed", "details": reason}

    if results_path is not None and results_path.exists():
        cfg = json.loads((reference_dir / "eval_config.json").read_text(encoding="utf-8"))
        result = _score_results(results_path, expected_episodes(cfg))
        result["api_details"] = api_details
        return result

    if allow_fixture_hash_fallback and _matches_hidden_baseline(output_dir, reference_dir):
        return {"score": 1.0, "reason": "fixture_hidden_baseline_match", "api_details": api_details}

    return {
        "score": 0.0,
        "reason": "no_sim_results_available",
        "details": "submission passed structure/API gates, but no rollout results were available",
        "api_details": api_details,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument(
        "--results-path", type=Path, nargs="+", help="run_eval.py results.json file(s); shards are merged"
    )
    parser.add_argument("--allow-fixture-hash-fallback", action="store_true")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Author tool: write reference/input_manifest.json from --input-dir and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.write_manifest:
        manifest = build_manifest(args.input_dir.resolve())
        (args.reference_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"wrote {len(manifest)} entries")
        return 0
    input_dir = args.input_dir.resolve()
    results_path = None
    if args.results_path:
        results_path = args.results_path[0]
        if len(args.results_path) > 1:
            results_path = args.output_dir.parent / "merged_results.json"
            results_path.write_text(json.dumps(merge_results(args.results_path)))
    manifest = json.loads((args.reference_dir / "input_manifest.json").read_text(encoding="utf-8"))
    result = evaluate_submission(
        args.output_dir,
        args.reference_dir,
        input_dir / "assets" / "smoke_obs.npz",
        input_integrity=verify_manifest(input_dir, manifest),
        results_path=results_path,
        allow_fixture_hash_fallback=args.allow_fixture_hash_fallback,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
