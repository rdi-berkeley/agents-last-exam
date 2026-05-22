"""Verifier for amber_three_stage_mmgbsa_workflow_instance_1."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

SYSTEM_BASENAME = "GLN_phb2_parl_pgam5_model_0"
REQUIRED_FILES = (
    "submit_min.sh",
    "submit_prod.sh",
    "submit_mmgbsa.sh",
    "FINAL_RESULTS_MMGBSA.dat",
)
IGNORED_FILENAMES = {".gitkeep"}
ACCEPTED_DELTA_RANGE = (-130.0, -100.0)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _parse_delta_total(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(
        r"DELTA\s+TOTAL\s+(-?\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _check_common_script_shape(name: str, text: str) -> list[str]:
    errors: list[str] = []
    if not (text.startswith("#!/bin/bash") or text.startswith("#!/usr/bin/env bash")):
        errors.append(f"{name}:missing bash shebang")
    if "#SBATCH" not in text:
        errors.append(f"{name}:missing sbatch directives")
    if SYSTEM_BASENAME not in text:
        errors.append(f"{name}:missing system basename")
    return errors


def _check_submit_min(text: str) -> list[str]:
    errors = _check_common_script_shape("submit_min.sh", text)
    if "pmemd.cuda" not in text:
        errors.append("submit_min.sh:missing pmemd.cuda")
    if "params" not in text.lower():
        errors.append("submit_min.sh:missing params directory handling")
    if not re.search(r"min.*\.out|\.out.*min", text):
        errors.append("submit_min.sh:missing minimization outputs")
    if not re.search(r"min.*\.rst|\.rst.*min", text):
        errors.append("submit_min.sh:missing minimization restart")
    if not re.search(r"(heat|equil|prod-ready|production-ready)", text, flags=re.IGNORECASE):
        errors.append("submit_min.sh:missing equilibration / pre-production stage")
    return errors


def _check_submit_prod(text: str) -> list[str]:
    errors = _check_common_script_shape("submit_prod.sh", text)
    if "pmemd.cuda" not in text:
        errors.append("submit_prod.sh:missing pmemd.cuda")
    if "prod.out" not in text:
        errors.append("submit_prod.sh:missing prod.out")
    if "prod.rst" not in text:
        errors.append("submit_prod.sh:missing prod.rst")
    if "prod.mdcrd" not in text:
        errors.append("submit_prod.sh:missing prod.mdcrd")
    has_restart_literal = bool(re.search(r"-c\s+.*(?:equil|min|heat).*(?:rst|ncrst)", text))
    has_restart_var = bool(re.search(r"(?:equil|min|heat).*\.(?:rst|ncrst)", text) and re.search(r"-c\s", text))
    if not has_restart_literal and not has_restart_var:
        errors.append("submit_prod.sh:missing previous-stage restart input")
    if "step2_implicit" not in text and "prod" not in text.lower():
        errors.append("submit_prod.sh:missing production mdin wiring")
    return errors


def _check_submit_mmgbsa(text: str) -> list[str]:
    errors = _check_common_script_shape("submit_mmgbsa.sh", text)
    if "MMPBSA.py" not in text:
        errors.append("submit_mmgbsa.sh:missing MMPBSA.py")
    if "FINAL_RESULTS_MMGBSA.dat" not in text:
        errors.append("submit_mmgbsa.sh:missing FINAL_RESULTS_MMGBSA.dat output")
    if "prod.mdcrd" not in text:
        errors.append("submit_mmgbsa.sh:missing prod.mdcrd trajectory input")
    has_cp = bool(re.search(r"-cp\s", text))
    has_prmtop = ".prmtop" in text
    if not (has_cp and has_prmtop):
        errors.append("submit_mmgbsa.sh:missing complex topology wiring")
    has_rp = bool(re.search(r"-rp\s", text))
    has_receptor_topo = bool(re.search(r"(_A|_rec|_receptor).*\.prmtop|\.prmtop.*(_A|_rec|_receptor)", text))
    if not (has_rp and has_receptor_topo):
        errors.append("submit_mmgbsa.sh:missing receptor topology wiring")
    has_lp = bool(re.search(r"-lp\s", text))
    has_ligand_topo = bool(re.search(r"(_BC|_lig|_ligand).*\.prmtop|\.prmtop.*(_BC|_lig|_ligand)", text))
    if not (has_lp and has_ligand_topo):
        errors.append("submit_mmgbsa.sh:missing ligand topology wiring")
    has_chain_mask = bool(re.search(r":%[A-C]", text))
    has_residue_mask = bool(re.search(r"-m\s+[\"']?:\d+-\d+", text))
    has_receptor_ligand_split = bool(re.search(r"ante-MMPBSA", text))
    if not (has_chain_mask or has_residue_mask or has_receptor_ligand_split):
        errors.append("submit_mmgbsa.sh:missing receptor/ligand split logic")
    if "igb=8" not in text.replace(" ", ""):
        errors.append("submit_mmgbsa.sh:missing igb=8")
    return errors


def _check_results(text: str, hidden_reference_text: str | None) -> list[str]:
    errors: list[str] = []
    delta_total = _parse_delta_total(text)
    if delta_total is None:
        errors.append("FINAL_RESULTS_MMGBSA.dat:missing DELTA TOTAL")
        return errors
    low, high = ACCEPTED_DELTA_RANGE
    if not (low <= delta_total <= high):
        errors.append("FINAL_RESULTS_MMGBSA.dat:delta total out of accepted range")
    if "Calculations performed using" not in text:
        errors.append("FINAL_RESULTS_MMGBSA.dat:missing frame-count summary")
    if "Receptor mask" not in text or "Ligand mask" not in text:
        errors.append("FINAL_RESULTS_MMGBSA.dat:missing receptor/ligand masks")
    if hidden_reference_text:
        hidden_delta = _parse_delta_total(hidden_reference_text)
        if hidden_delta is not None and abs(delta_total - hidden_delta) > 50.0:
            errors.append("FINAL_RESULTS_MMGBSA.dat:delta total too far from hidden reference")
    return errors


def evaluate_output_bundle(
    files: Mapping[str, str],
    *,
    present_files: list[str] | None = None,
    hidden_reference_text: str | None = None,
) -> dict:
    reasons: list[str] = []
    visible_files = sorted(name for name in (present_files or files.keys()) if name not in IGNORED_FILENAMES)
    expected = sorted(REQUIRED_FILES)

    missing = sorted(set(expected) - set(visible_files))
    if missing:
        reasons.append(f"missing_required_files:{missing}")

    submit_min = files.get("submit_min.sh")
    submit_prod = files.get("submit_prod.sh")
    submit_mmgbsa = files.get("submit_mmgbsa.sh")
    final_results = files.get("FINAL_RESULTS_MMGBSA.dat")

    if submit_min is None:
        reasons.append("missing submit_min.sh")
    if submit_prod is None:
        reasons.append("missing submit_prod.sh")
    if submit_mmgbsa is None:
        reasons.append("missing submit_mmgbsa.sh")
    if final_results is None:
        reasons.append("missing FINAL_RESULTS_MMGBSA.dat")

    if submit_min is not None:
        reasons.extend(_check_submit_min(submit_min))
    if submit_prod is not None:
        reasons.extend(_check_submit_prod(submit_prod))
    if submit_mmgbsa is not None:
        reasons.extend(_check_submit_mmgbsa(submit_mmgbsa))
    if final_results is not None:
        reasons.extend(_check_results(final_results, hidden_reference_text))

    deduped: list[str] = []
    seen = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)

    passed = not deduped
    delta_total = _parse_delta_total(final_results)
    hidden_delta = _parse_delta_total(hidden_reference_text)
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "reasons": deduped,
        "delta_total": delta_total,
        "hidden_delta_total": hidden_delta,
    }


def _load_directory(path: Path) -> tuple[dict[str, str], list[str]]:
    files: dict[str, str] = {}
    names: list[str] = []
    for child in sorted(path.iterdir()):
        if not child.is_file():
            continue
        names.append(child.name)
        files[child.name] = child.read_text(encoding="utf-8", errors="replace")
    return files, names


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify amber_three_stage_mmgbsa_workflow_instance_1 outputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-reference", help="Optional hidden reference FINAL_RESULTS_MMGBSA.dat")
    args = parser.parse_args()

    files, names = _load_directory(Path(args.output_dir))
    hidden_reference_text = None
    if args.hidden_reference:
        hidden_reference_text = Path(args.hidden_reference).read_text(encoding="utf-8", errors="replace")
    payload = evaluate_output_bundle(
        files,
        present_files=names,
        hidden_reference_text=hidden_reference_text,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
