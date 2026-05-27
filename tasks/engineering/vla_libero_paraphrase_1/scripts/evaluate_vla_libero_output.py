#!/usr/bin/env python
"""Evaluate a VLA-Adapter/LIBERO-Para output directory.

The checks are adapted from the submitter-provided hidden grader staged under
reference/grader/2_eval_script.py.  This copy is task-local evaluator code so
`main.py` can upload a small verifier to the eval temp directory at runtime.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ANCHOR_SR = 46.3
ANCHOR_PRIDE_A050 = 36.1
ANCHOR_OVERESTIMATION = 22.0
SR_BAND = 1.0
PRIDE_BAND = 1.0
OVEREST_BAND = 1.0

SK_ST_AGREE_RATE_MIN = 0.70
N_EPISODES_MIN = 4000
WALL_CLOCK_HOURS_MAX = 24.0
DISK_GB_MAX = 5.0
MAX_FILE_BYTES = 500 * 2**20

REQUIRED_FILES = [
    "method.md",
    "setup.sh",
    "run_rollouts.sh",
    "compute_pride.py",
    "compute_pride.sh",
    "ablation.sh",
    "results.json",
    "results_ablation.json",
    "ci_smoke.json",
    "build_provenance.txt",
    "sk_st_validation.json",
]
REQUIRED_LOG_DIR = "logs"


def _emit(report: dict, code: int) -> int:
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


def _malformed(message: str) -> int:
    return _emit({"overall": "MALFORMED", "score": 0.0, "errors": [message]}, 2)


def validate_schema(results: dict) -> list[str]:
    required = (
        "sr",
        "pride_a0.50",
        "pride_a0.00",
        "pride_a1.00",
        "overestimation_pct",
        "n_episodes_total",
        "n_episodes_matched",
        "n_success",
        "alpha",
        "wall_clock_hours",
        "sentence_bert_model",
        "spacy_model",
        "metadata_csv_sha256",
        "vla_adapter_commit",
        "torch_version",
        "transformers_version",
    )
    errors = [f"results.{key} missing" for key in required if key not in results]
    if results.get("alpha") not in (0.5, 0.50):
        errors.append(f"results.alpha must be 0.5, got {results.get('alpha')}")
    return errors


def grep(pattern: str, path: Path, flags: int = re.MULTILINE) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(re.search(pattern, text, flags))


def grep_any(pattern: str, paths: list[Path], flags: int = re.MULTILINE) -> bool:
    return any(grep(pattern, path, flags) for path in paths)


def invalidator_checks(repo: Path) -> dict[str, bool]:
    setup = repo / "setup.sh"
    run_rollouts = repo / "run_rollouts.sh"
    rollout_py = repo / "run_rollouts.py"
    compute = repo / "compute_pride.py"
    provenance = repo / "build_provenance.txt"
    rollout_paths = [run_rollouts, rollout_py]

    c1 = (
        grep("LIBERO-Goal-Pro", setup)
        and not grep(r"LIBERO-Goal[^-_/A-Za-z]", setup)
        and not grep(r"LIBERO-Spatial-Pro|LIBERO-Object-Pro", setup)
    )
    c2 = grep(r"MUJOCO_GL\s*=\s*egl", setup) or grep_any(r"MUJOCO_GL\s*=\s*egl", rollout_paths)
    c3 = grep_any(
        r"NUM_OPEN_LOOP_STEPS\s*=\s*8|num_open_loop_steps\s*=\s*8|action_chunk_size\s*=\s*8",
        rollout_paths,
    )
    c4 = grep_any(
        r"camera_heights\s*=\s*256|LIBERO_ENV_RESOLUTION\s*=\s*256|"
        r"\bRES(OLUTION)?\s*=\s*256\b|256\s*[,)]\s*camera",
        rollout_paths,
    )
    c5 = grep(r"--seed\s+7\b|seed\s*[:=]\s*7\b", run_rollouts)
    c6 = grep_any(
        r"max_steps\s*[=]\s*300|MAX_STEPS\s*=\s*300|--max[_-]?steps\s+300\b",
        rollout_paths,
    )
    c7 = not any(
        grep(r"\.backward\(|optim\.step\(|model\.train\(\)|Trainer\(", path)
        for path in [run_rollouts, rollout_py, compute]
        if path.exists()
    )
    c8 = (
        grep(r"all-MiniLM-L6-v2", compute)
        and grep(r"SentenceTransformer\s*\(", compute)
        and not grep(r"all-mpnet-base-v2|paraphrase-MiniLM|distilbert", compute)
    )
    c9 = grep("en_core_web_sm", setup) and grep("en_core_web_sm", compute)
    c10 = not grep(r"\bmax\s*\(\s*\w*size\w*\s*,\s*\w*size\w*\s*\)", compute) and grep(
        r"size1\s*\+\s*size2|tree_size_1\s*\+\s*tree_size_2|"
        r"len\s*\([^)]*\)\s*\+\s*len\s*\([^)]*\)",
        compute,
    )
    c11 = grep(r"alpha\s*=\s*0\.5\b|α\s*=\s*0\.5\b", compute)
    c12 = all(
        grep(pattern, provenance)
        for pattern in [
            r"vla_adapter_commit\s+[0-9a-f]{40}",
            r"libero_para_commit\s+[0-9a-f]{40}",
            r"hf_checkpoint_revision\s+[0-9a-f]{40}",
            r"torch_version\s+\S+",
            r"transformers_version\s+\S+",
            r"sentence_transformers\s+\S+",
            r"spacy\s+\S+",
            r"en_core_web_sm\s+\S+",
            r"^[0-9a-f]{64}\s+.*libero_para_metadata\.csv",
        ]
    )

    return {
        "C1_checkpoint_libero_goal_pro": c1,
        "C2_mujoco_gl_egl": c2,
        "C3_action_chunk_8": c3,
        "C4_image_resolution_256": c4,
        "C5_seed_7": c5,
        "C6_max_steps_300": c6,
        "C7_no_training": c7,
        "C8_sentence_bert_model": c8,
        "C9_spacy_model": c9,
        "C10_st_norm_size1_plus_size2": c10,
        "C11_alpha_default_0.5": c11,
        "C12_provenance_complete": c12,
    }


def determinism_check(repo: Path) -> tuple[bool, str]:
    path = repo / "ci_smoke.json"
    if not path.exists():
        return False, "ci_smoke.json missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"ci_smoke.json invalid JSON: {exc}"
    h7 = data.get("seed7_action_hash")
    h8 = data.get("seed8_action_hash")
    main = data.get("seed7_main_log_hash")
    if not (h7 and h8 and main):
        return False, "ci_smoke.json missing required hash fields"
    if h7 != main:
        return False, "seed=7 reproduction differs from main rollout"
    if h7 == h8:
        return False, "seed=7 and seed=8 produce identical paths"
    return True, "ok"


def sk_st_validation(repo: Path) -> tuple[bool, float, str]:
    path = repo / "sk_st_validation.json"
    if not path.exists():
        return False, 0.0, "sk_st_validation.json missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, 0.0, f"invalid JSON: {exc}"
    rate = data.get("agree_rate")
    if not isinstance(rate, (int, float)):
        return False, 0.0, "agree_rate missing"
    return rate >= SK_ST_AGREE_RATE_MIN, float(rate), ""


def disk_and_walltime(repo: Path, results: dict) -> tuple[list[str], dict]:
    failures = []
    total = 0
    biggest = ("", 0)
    skip_dirs = {".git", "__pycache__", "logs"}
    for path in repo.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if size > biggest[1]:
            biggest = (str(path.relative_to(repo)), size)
    if total > DISK_GB_MAX * 10**9:
        failures.append(f"disk total (excl. logs/) {total / 1e9:.2f} GB > {DISK_GB_MAX} GB")
    if biggest[1] > MAX_FILE_BYTES:
        failures.append(
            f"largest file {biggest[0]} = {biggest[1] / 2**20:.0f} MiB > "
            f"{MAX_FILE_BYTES // 2**20} MiB"
        )
    wall_clock = results.get("wall_clock_hours")
    if not isinstance(wall_clock, (int, float)) or wall_clock > WALL_CLOCK_HOURS_MAX:
        failures.append(f"wall_clock_hours = {wall_clock} > {WALL_CLOCK_HOURS_MAX}")
    return failures, {"total_bytes": total, "largest": biggest, "wall_clock_h": wall_clock}


def anchor_check(results: dict) -> dict:
    sr_ok = abs(results["sr"] - ANCHOR_SR) <= SR_BAND
    pride_ok = abs(results["pride_a0.50"] - ANCHOR_PRIDE_A050) <= PRIDE_BAND
    over_ok = abs(results["overestimation_pct"] - ANCHOR_OVERESTIMATION) <= OVEREST_BAND
    return {
        "sr": {"value": results["sr"], "anchor": ANCHOR_SR, "band": SR_BAND, "ok": sr_ok},
        "pride_a0.50": {
            "value": results["pride_a0.50"],
            "anchor": ANCHOR_PRIDE_A050,
            "band": PRIDE_BAND,
            "ok": pride_ok,
        },
        "overest": {
            "value": results["overestimation_pct"],
            "anchor": ANCHOR_OVERESTIMATION,
            "band": OVEREST_BAND,
            "ok": over_ok,
        },
        "all_ok": sr_ok and pride_ok and over_ok,
    }


def coverage_check(results: dict) -> tuple[bool, int]:
    n_matched = int(results.get("n_episodes_matched", 0))
    return n_matched >= N_EPISODES_MIN, n_matched


def evaluate(repo: Path) -> tuple[dict, int]:
    if not repo.is_dir():
        return {"overall": "MALFORMED", "score": 0.0, "errors": [f"{repo} is not a directory"]}, 2

    missing = [name for name in REQUIRED_FILES if not (repo / name).exists()]
    if missing:
        return {"overall": "MALFORMED", "score": 0.0, "errors": [f"missing files: {missing}"]}, 2
    if not (repo / REQUIRED_LOG_DIR).is_dir():
        return {"overall": "MALFORMED", "score": 0.0, "errors": ["missing logs/ directory"]}, 2

    try:
        results = json.loads((repo / "results.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"overall": "MALFORMED", "score": 0.0, "errors": [f"results.json invalid: {exc}"]}, 2

    schema_errors = validate_schema(results)
    if schema_errors:
        return {"overall": "MALFORMED", "score": 0.0, "errors": schema_errors}, 2

    invalidators = invalidator_checks(repo)
    invalidators_ok = all(invalidators.values())
    det_ok, det_msg = determinism_check(repo)
    sk_st_ok, sk_st_rate, sk_st_msg = sk_st_validation(repo)
    cov_ok, cov_n = coverage_check(results)
    anchors = anchor_check(results)
    disk_fails, disk_meta = disk_and_walltime(repo, results)

    passed = (
        anchors["all_ok"] and invalidators_ok and det_ok and sk_st_ok and cov_ok and not disk_fails
    )
    report = {
        "overall": "PASS" if passed else "FAIL",
        "score": 1.0 if passed else 0.0,
        "anchors": anchors,
        "invalidators": invalidators,
        "invalidators_ok": invalidators_ok,
        "determinism": {"ok": det_ok, "msg": det_msg},
        "sk_st_validation": {"ok": sk_st_ok, "agree_rate": sk_st_rate, "msg": sk_st_msg},
        "coverage": {"ok": cov_ok, "n_episodes_matched": cov_n, "min_required": N_EPISODES_MIN},
        "disk_walltime": {"ok": not disk_fails, "fails": disk_fails, "meta": disk_meta},
    }
    return report, 0 if passed else 1


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        return _malformed("usage: evaluate_vla_libero_output.py <output_dir>")
    report, code = evaluate(Path(argv[1]))
    return _emit(report, code)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
