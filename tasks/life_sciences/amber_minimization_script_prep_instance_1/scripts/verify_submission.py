"""Parser-based verifier for the Amber minimization workflow task."""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import shlex
from pathlib import Path
from typing import Mapping

if __package__:
    from .workflow_parser import (
        ContractError,
        Shell,
        leap_commands,
        namelists,
        number,
        options,
        path,
    )
else:
    from workflow_parser import (
        ContractError,
        Shell,
        leap_commands,
        namelists,
        number,
        options,
        path,
    )

SYSTEM_BASENAME = "GLN_phb2_lc3_aurka_model_0"
REQUIRED_FILES = ("leap.in", "step2_implicit.mini.mdin", "submit_min.sh")
IGNORED_FILENAMES = {".gitkeep"}


def _parse_mem_gb(script: str) -> float | None:
    match = re.search(
        r"#SBATCH\s+--mem(?:=|\s+)([0-9.]+)\s*([A-Za-z]+)", script, flags=re.IGNORECASE
    )
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
    match = re.search(
        r"#SBATCH\s+--time(?:=|\s+)(\d+):(\d{2}):(\d{2})", script, flags=re.IGNORECASE
    )
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    return hours + minutes / 60.0 + seconds / 3600.0


def _check_leap(text: str) -> list[str]:
    units, outputs = {}, set()
    force_field = radii = quit_seen = False
    try:
        for words in leap_commands(text):
            if quit_seen:
                break
            if len(words) >= 3 and words[1] == "=":
                unit, command, *args = words[0], words[2].lower(), *words[3:]
                units[unit] = (
                    command == "loadpdb"
                    and len(args) == 1
                    and posixpath.basename(args[0]) == "complex_structure.pdb"
                    and force_field
                    and radii
                )
                continue
            command, *args = words
            command = command.lower()
            if command == "source":
                force_field = (
                    len(args) == 1 and posixpath.basename(args[0]) == "leaprc.protein.ff14SB"
                )
            elif command == "set":
                if len(args) >= 2 and args[0] == "default" and args[1] in {"PBRadii", "PBradii"}:
                    radii = args[2:] == ["mbondi3"]
            elif command in {"saveamberparm", "savepdb"}:
                if not args or not units.get(args[0]):
                    raise ContractError("LEaP output uses an unbuilt or incorrect unit")
                expected = (
                    [f"{SYSTEM_BASENAME}.prmtop", f"{SYSTEM_BASENAME}.inpcrd"]
                    if command == "saveamberparm"
                    else [f"{SYSTEM_BASENAME}_fixed.pdb"]
                )
                if [posixpath.basename(value) for value in args[1:]] != expected:
                    raise ContractError("incorrect LEaP output wiring")
                outputs.add(command)
            elif command == "quit":
                quit_seen = not args
            elif command not in {"check", "charge", "desc"}:
                raise ContractError(f"unsupported LEaP command: {command}")
        if not quit_seen or outputs != {"saveamberparm", "savepdb"}:
            raise ContractError("missing LEaP build, outputs, or quit")
    except ContractError as error:
        return [str(error)]
    return []


def _check_mdin(text: str) -> list[str]:
    errors: list[str] = []
    try:
        params = namelists(text)["cntrl"]
    except (ContractError, KeyError) as error:
        return [f"invalid &cntrl: {error}"]
    if number(params.get("imin")) != 1:
        errors.append("imin must equal 1")
    if number(params.get("ntb")) != 0:
        errors.append("ntb must equal 0")
    if number(params.get("cut")) != 999:
        errors.append("cut must equal 999.0")
    if number(params.get("ntpr")) is None or params["ntpr"] < 1 or not params["ntpr"].is_integer():
        errors.append("ntpr must be a positive integer")
    if number(params.get("ntxo")) not in {1, 2}:
        errors.append("ntxo must be 1 or 2")
    if number(params.get("igb")) not in {7, 8}:
        errors.append("igb must be 7 or 8")

    maxcyc = number(params.get("maxcyc"))
    ncyc = number(params.get("ncyc"))
    if maxcyc is None or maxcyc < 2000 or not maxcyc.is_integer():
        errors.append("maxcyc must be >= 2000")
    if ncyc is None or maxcyc is None or not (0 < ncyc < maxcyc) or not ncyc.is_integer():
        errors.append("ncyc must be between 0 and maxcyc")

    for key, bounds in {"saltcon": (0, 0.2), "intdiel": (1, 4), "extdiel": (60, 90)}.items():
        if key in params:
            value = number(params[key])
            if value is None or not bounds[0] <= value <= bounds[1]:
                errors.append(f"{key} out of range or malformed")
    return errors


def _check_submit(script: str, leap_text: str = "", mdin_text: str = "") -> list[str]:
    errors: list[str] = []
    try:
        staged = {"leap.in": leap_text, "step2_implicit.mini.mdin": mdin_text}
        shell = Shell(script, staged)
        directives = []
        for line in script.splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                break
            if line.startswith("#SBATCH "):
                directives.extend(shlex.split(line[len("#SBATCH ") :]))
        resources = {}
        index = 0
        while index < len(directives):
            token = directives[index]
            index += 1
            if not token.startswith("--"):
                raise ContractError("use long SLURM options")
            if "=" in token:
                key, value = token.split("=", 1)
            else:
                key, value = token, directives[index]
                index += 1
            resources[key] = value
        directive_text = "\n".join(f"#SBATCH {key}={value}" for key, value in resources.items())
        commands = shell.commands
        modules = set()
        for command in commands:
            if posixpath.basename(command.argv[0]) == "pmemd.cuda":
                break
            if command.argv[:2] in (["module", "load"], ["module", "add"]):
                modules.update(command.argv[2:])
        if not {"amber/22", "cuda/11.6.2"} <= modules:
            errors.append("missing active Amber/CUDA module loads")
        builds = [command for command in commands if posixpath.basename(command.argv[0]) == "tleap"]
        if len(builds) != 1 or not builds[0].guards:
            errors.append("missing conditional tleap build")
        elif posixpath.basename(options(builds[0].argv).get("-f", "")) != "leap.in":
            errors.append("incorrect tleap input")
        runs = [
            command for command in commands if posixpath.basename(command.argv[0]) == "pmemd.cuda"
        ]
        if len(runs) != 1:
            errors.append("must invoke exactly one active pmemd.cuda command")
        else:
            run = runs[0]
            flags = options(run.argv)
            expected = {
                "-i": "step2_implicit.mini.mdin",
                "-p": f"{SYSTEM_BASENAME}.prmtop",
                "-c": f"{SYSTEM_BASENAME}.inpcrd",
                "-ref": f"{SYSTEM_BASENAME}.inpcrd",
                "-o": "min.out",
                "-r": "min.rst",
            }
            if "-O" not in flags:
                errors.append("missing pmemd.cuda -O")
            for key, basename in expected.items():
                if posixpath.basename(path(flags.get(key, ""), run.cwd)) != basename:
                    errors.append(f"incorrect {key} wiring")
            if all(key in flags for key in ("-p", "-c", "-ref")):
                topology, coordinates, reference = (
                    path(flags[key], run.cwd) for key in ("-p", "-c", "-ref")
                )
                if (
                    coordinates != reference
                    or posixpath.dirname(topology) != posixpath.dirname(coordinates)
                    or posixpath.basename(posixpath.dirname(topology)) != "params"
                ):
                    errors.append("inconsistent params topology/coordinate/reference paths")
                for existing in (
                    {},
                    {topology: "topology", coordinates: "coordinates"},
                    {topology: "topology"},
                    {coordinates: "coordinates"},
                ):
                    branch = Shell(script, staged | existing)
                    branch_builds = [
                        command
                        for command in branch.commands
                        if posixpath.basename(command.argv[0]) == "tleap"
                    ]
                    if len(branch_builds) != (0 if len(existing) == 2 else 1):
                        errors.append(
                            "tleap must build if either product is missing, and skip if both exist"
                        )
                        break
                    branch_runs = [
                        command
                        for command in branch.commands
                        if posixpath.basename(command.argv[0]) == "pmemd.cuda"
                    ]
                    if len(branch_runs) != 1:
                        errors.append(
                            "each topology/coordinate guard path must reach one pmemd.cuda"
                        )
                        break
                    branch_run = branch_runs[0]
                    branch_flags = options(branch_run.argv)
                    if "-O" not in branch_flags or any(
                        path(branch_flags.get(key, ""), branch_run.cwd)
                        != path(flags.get(key, ""), run.cwd)
                        for key in expected
                    ):
                        errors.append("inconsistent pmemd wiring across guard paths")
                    for build in branch_builds:
                        control = build.files.get(
                            path(options(build.argv).get("-f", ""), build.cwd)
                        )
                        if control is None or _check_leap(control):
                            errors.append("tleap must consume the valid submitted LEaP control")
                    for filename, kind in (
                        (topology, "tleap:topology"),
                        (coordinates, "tleap:coordinates"),
                    ):
                        if branch_run.products.get(filename) != kind and not (
                            filename in existing
                            and branch_run.files.get(filename) == existing[filename]
                        ):
                            errors.append("LEaP products do not reach pmemd inputs")
        if not any(
            command.argv[:1] == ["mkdir"]
            and any(posixpath.basename(value) == "params" for value in command.argv[1:])
            for command in commands
        ):
            errors.append("missing params directory creation")
    except (ContractError, ValueError, IndexError) as error:
        return [f"invalid submission workflow: {error}"]
    if not script.startswith(("#!/bin/bash\n", "#!/usr/bin/env bash\n")):
        errors.append("missing bash shebang")
    script = directive_text
    if "#SBATCH" not in script:
        errors.append("missing sbatch directives")
    if resources.get("--nodes") != "1":
        errors.append("nodes must equal 1")
    gpu_options = {
        key: resources[key] for key in ("--gres", "--gpus", "--gpus-per-node") if key in resources
    }
    if len(gpu_options) != 1 or any(
        value != ("gpu:1" if key == "--gres" else "1") for key, value in gpu_options.items()
    ):
        errors.append("must request exactly one gpu")
    if resources.get("--ntasks-per-node") != "1" or resources.get("--ntasks", "1") != "1":
        errors.append("ntasks-per-node must equal 1")
    if resources.get("--cpus-per-task") not in {"1", "2", "3", "4"}:
        errors.append("cpus-per-task out of range")
    mem_gb = _parse_mem_gb(script)
    if mem_gb is None or not (8.0 <= mem_gb <= 32.0):
        errors.append("mem out of range")
    time_hours = _parse_time_hours(script)
    if time_hours is None or not (1.0 <= time_hours <= 12.0):
        errors.append("time out of range")
    return errors


def evaluate_output_bundle(
    files: Mapping[str, str], *, present_files: list[str] | None = None
) -> dict:
    reasons: list[str] = []
    visible_files = sorted(
        name
        for name in (present_files if present_files is not None else files.keys())
        if name not in IGNORED_FILENAMES
    )
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
        reasons.extend(_check_submit(submit_text, leap_text or "", mdin_text or ""))

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
    parser = argparse.ArgumentParser(
        description="Verify an Amber minimization workflow output directory."
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    files, names = _load_directory(output_dir)
    payload = evaluate_output_bundle(files, present_files=names)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
