"""Parser-based verifier for the Amber minimization workflow task."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

SYSTEM_BASENAME = "GLN_phb2_lc3_aurka_model_0"
REQUIRED_FILES = ("leap.in", "step2_implicit.mini.mdin", "submit_min.sh")
IGNORED_FILENAMES = {".gitkeep"}


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _extract_namelist_params(text: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,\n/]+)", text):
        key = match.group(1).lower()
        value = match.group(2).strip().strip('"').strip("'")
        params[key] = value
    return params


def _as_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_mem_gb(script: str) -> float | None:
    match = re.search(r"#SBATCH\s+--mem(?:=|\s+)([0-9.]+)\s*([A-Za-z]+)", script, flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"g", "gb", "gib"}:
        return value
    if unit in {"m", "mb", "mib"}:
        return value / 1024.0
    return None


def _parse_time_hours(script: str) -> float | None:
    match = re.search(r"#SBATCH\s+--time(?:=|\s+)(\d+):(\d{2}):(\d{2})", script, flags=re.IGNORECASE)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    return hours + minutes / 60.0 + seconds / 3600.0


def _check_leap(text: str) -> list[str]:
    errors: list[str] = []
    normalized = _collapse_ws(text)
    if "source leaprc.protein.ff14SB" not in text:
        errors.append("missing force-field source")
    if "set default PBRadii mbondi3" not in text:
        errors.append("missing mbondi3 radii")
    if not re.search(r'loadpdb\s+"?complex_structure\.pdb"?', text):
        errors.append("missing complex_structure.pdb load")
    if not re.search(
        rf"saveamberparm\s+\S+\s+{re.escape(SYSTEM_BASENAME)}\.prmtop\s+{re.escape(SYSTEM_BASENAME)}\.inpcrd",
        normalized,
    ):
        errors.append("missing saveamberparm outputs")
    if not re.search(
        rf"savepdb\s+\S+\s+{re.escape(SYSTEM_BASENAME)}_fixed\.pdb",
        normalized,
    ):
        errors.append("missing fixed pdb output")
    if "quit" not in text:
        errors.append("missing quit")
    return errors


def _check_mdin(text: str) -> list[str]:
    errors: list[str] = []
    params = _extract_namelist_params(text)
    if params.get("imin") != "1":
        errors.append("imin must equal 1")
    if params.get("ntb") != "0":
        errors.append("ntb must equal 0")
    if params.get("cut") != "999.0":
        errors.append("cut must equal 999.0")
    if "ntpr" not in params:
        errors.append("missing ntpr")
    if "ntxo" not in params:
        errors.append("missing ntxo")
    if params.get("igb") not in {"7", "8"}:
        errors.append("igb must be 7 or 8")

    maxcyc = _as_float(params.get("maxcyc"))
    ncyc = _as_float(params.get("ncyc"))
    if maxcyc is None or maxcyc < 2000:
        errors.append("maxcyc must be >= 2000")
    if ncyc is None or maxcyc is None or not (0 < ncyc < maxcyc):
        errors.append("ncyc must be between 0 and maxcyc")

    saltcon = _as_float(params.get("saltcon"))
    if saltcon is not None and not (0.0 <= saltcon <= 0.2):
        errors.append("saltcon out of range")
    intdiel = _as_float(params.get("intdiel"))
    if intdiel is not None and not (1.0 <= intdiel <= 4.0):
        errors.append("intdiel out of range")
    extdiel = _as_float(params.get("extdiel"))
    if extdiel is not None and not (60.0 <= extdiel <= 90.0):
        errors.append("extdiel out of range")
    return errors


def _check_submit(script: str) -> list[str]:
    errors: list[str] = []
    if not (script.startswith("#!/bin/bash") or script.startswith("#!/usr/bin/env bash")):
        errors.append("missing bash shebang")
    if "#SBATCH" not in script:
        errors.append("missing sbatch directives")
    if not re.search(r"#SBATCH\s+--nodes(?:=|\s+)1\b", script):
        errors.append("nodes must equal 1")
    if not (
        re.search(r"#SBATCH\s+--gres(?:=|\s+)gpu:1\b", script)
        or re.search(r"#SBATCH\s+--gpus(?:=|\s+)1\b", script)
        or re.search(r"#SBATCH\s+--gpus-per-node(?:=|\s+)1\b", script)
    ):
        errors.append("must request exactly one gpu")
    if not re.search(r"#SBATCH\s+--ntasks-per-node(?:=|\s+)1\b", script):
        errors.append("ntasks-per-node must equal 1")
    cpus_match = re.search(r"#SBATCH\s+--cpus-per-task(?:=|\s+)(\d+)\b", script)
    if not cpus_match or not (1 <= int(cpus_match.group(1)) <= 4):
        errors.append("cpus-per-task out of range")
    mem_gb = _parse_mem_gb(script)
    if mem_gb is None or not (8.0 <= mem_gb <= 32.0):
        errors.append("mem out of range")
    time_hours = _parse_time_hours(script)
    if time_hours is None or not (1.0 <= time_hours <= 12.0):
        errors.append("time out of range")
    if SYSTEM_BASENAME not in script:
        errors.append("missing system basename")
    if "params" not in script:
        errors.append("missing params directory handling")
    if "amber/22" not in script:
        errors.append("missing amber/22 module load")
    if "cuda/11.6.2" not in script:
        errors.append("missing cuda/11.6.2 module load")
    if "tleap" not in script or "if " not in script:
        errors.append("missing conditional tleap build")
    if not re.search(r"pmemd\.cuda\s+-O", script):
        errors.append("missing pmemd.cuda -O")
    if not re.search(r"-i\s+.*step2_implicit\.mini\.mdin", script):
        errors.append("missing -i wiring")
    if not re.search(r"-p\s+.*(?:GLN_phb2_lc3_aurka_model_0|\$?\{?BASE\}?).*\.prmtop", script):
        errors.append("missing -p wiring")
    if not re.search(r"-c\s+.*(?:GLN_phb2_lc3_aurka_model_0|\$?\{?BASE\}?).*\.inpcrd", script):
        errors.append("missing -c wiring")
    if not re.search(r"-o\s+.*min\.out", script):
        errors.append("missing -o wiring")
    if not re.search(r"-r\s+.*min\.rst", script):
        errors.append("missing -r wiring")
    if not re.search(r"-ref\s+.*(?:GLN_phb2_lc3_aurka_model_0|\$?\{?BASE\}?).*\.inpcrd", script):
        errors.append("missing -ref wiring")
    return errors


def evaluate_output_bundle(files: Mapping[str, str], *, present_files: list[str] | None = None) -> dict:
    reasons: list[str] = []
    visible_files = sorted(name for name in (present_files or files.keys()) if name not in IGNORED_FILENAMES)
    expected = sorted(REQUIRED_FILES)

    if visible_files != expected:
        reasons.append(f"file_set_mismatch:{visible_files}")

    leap_text = files.get("leap.in")
    mdin_text = files.get("step2_implicit.mini.mdin")
    submit_text = files.get("submit_min.sh")
    if leap_text is None:
        reasons.append("missing leap.in")
    if mdin_text is None:
        reasons.append("missing step2_implicit.mini.mdin")
    if submit_text is None:
        reasons.append("missing submit_min.sh")

    if leap_text is not None:
        reasons.extend(_check_leap(leap_text))
    if mdin_text is not None:
        reasons.extend(_check_mdin(mdin_text))
    if submit_text is not None:
        reasons.extend(_check_submit(submit_text))

    deduped = []
    seen = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)

    passed = not deduped
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "reasons": deduped,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an Amber minimization workflow output directory.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    files, names = _load_directory(output_dir)
    payload = evaluate_output_bundle(files, present_files=names)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
