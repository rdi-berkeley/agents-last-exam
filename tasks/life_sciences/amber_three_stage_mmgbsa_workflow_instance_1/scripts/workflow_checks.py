"""Semantic checks for authored Amber stages and the deterministic MMGBSA report."""

from __future__ import annotations

import math
import posixpath
import re
import shlex
from functools import lru_cache

from tasks.life_sciences.amber_minimization_script_prep_instance_1.scripts.workflow_parser import (
    ContractError,
    Shell,
    leap_commands,
    namelists,
    number,
    options,
    path,
)

SYSTEM_BASENAME = "GLN_phb2_parl_pgam5_model_0"
RECEPTOR = frozenset(range(1, 300))
LIGAND = frozenset(range(300, 965))
COMPLEX = RECEPTOR | LIGAND
STAGED_TRAJECTORY = "staged:trajectory"


class _StageShell(Shell):
    def __init__(self, script: str, files: dict, products: dict, directories: set):
        self.initial_state = (products.copy(), directories.copy())
        super().__init__(script, files)

    def _command(self, raw: list[str]) -> bool:
        if self.initial_state is not None:
            self.products, directories = self.initial_state
            self.directories.update(directories)
            self.initial_state = None
        return super()._command(raw)


@lru_cache(maxsize=16)
def _resolve_stages(scripts: tuple[str | None, ...]) -> tuple[dict, list[str], set[str]]:
    shells, errors, text_files, products = {}, [], {}, {}
    directories = set()
    for input_dir in (
        "/input",
        "/script/input",
        "/media/user/data/agenthle/life_sciences/amber_three_stage_mmgbsa_workflow_instance_1/base/input",
    ):
        for basename, kind in {
            "prod.mdcrd": STAGED_TRAJECTORY,
            "complex_structure.pdb": "staged:pdb",
            "input_environment_spec.md": "staged:spec",
            "task_sop.md": "staged:spec",
        }.items():
            products[posixpath.join(input_dir, basename)] = kind
        directory = input_dir
        while directory != "/":
            directories.add(directory)
            directory = posixpath.dirname(directory)
    known_products = set(products)
    for name, script in zip(("submit_min.sh", "submit_prod.sh", "submit_mmgbsa.sh"), scripts):
        if script is None:
            continue
        if name == "submit_mmgbsa.sh":
            products["/script/prod.mdcrd"] = STAGED_TRAJECTORY
            text_files.pop("/script/prod.mdcrd", None)
        try:
            shell = _StageShell(script, text_files, products, directories)
            shells[name] = shell
            text_files, products, directories = shell.files, shell.products, shell.directories
            known_products.update(products)
            for command in shell.commands:
                known_products.update(command.products)
        except (ContractError, KeyError, ValueError, IndexError) as error:
            errors.append(f"{name}:{error}")
    return shells, errors, known_products


def mask_residues(mask: str) -> frozenset[int]:
    mask = mask.strip().strip("\"'").replace(" ", "")
    if mask.startswith("!"):
        return COMPLEX - mask_residues(mask[1:].strip("()"))
    if "|" in mask:
        return frozenset().union(*(mask_residues(part) for part in mask.split("|")))
    if mask.startswith("::"):
        chains = {"A": RECEPTOR, "B": frozenset(range(300, 677)), "C": frozenset(range(677, 965))}
        try:
            return frozenset().union(*(chains[chain] for chain in mask[2:].split(",")))
        except KeyError as error:
            raise ContractError("invalid chain mask") from error
    if not mask.startswith(":"):
        raise ContractError("expected a residue or chain mask")
    residues = set()
    for span in mask[1:].split(","):
        parts = span.split("-")
        if not all(part.isdigit() for part in parts) or len(parts) > 2:
            raise ContractError("unsupported residue mask")
        start, end = int(parts[0]), int(parts[-1])
        if not 1 <= start <= end <= 964:
            raise ContractError("residue mask outside supplied complex")
        residues.update(range(start, end + 1))
    return frozenset(residues)


def mmgbsa_parameters(text: str) -> dict[str, dict]:
    groups = namelists(text, mmpbsa=True)
    if "gb" not in groups:
        raise ContractError("missing active &gb namelist")
    general, gb = groups.get("general", {}), groups["gb"]
    for params, names in (
        (gb, {"igb", "saltcon", "molsurf", "surften", "surfoff", "ifqnt", "probe", "msoffset"}),
        (
            general,
            {
                "startframe",
                "endframe",
                "interval",
                "entropy",
                "receptor_mask",
                "ligand_mask",
                "keep_files",
                "verbose",
                "netcdf",
                "use_sander",
                "search_path",
                "debug_printlevel",
                "full_traj",
                "strip_mask",
            },
        ),
    ):
        for key in list(params):
            candidates = [
                name for name in names if name == key or len(key) >= 4 and name.startswith(key)
            ]
            if len(candidates) != 1:
                raise ContractError(f"unknown or ambiguous MMPBSA parameter: {key}")
            name = candidates[0]
            if name != key:
                if name in params:
                    raise ContractError(f"duplicate MMPBSA parameter: {name}")
                params[name] = params.pop(key)
    for key, expected in {"igb": 8, "saltcon": 0.150}.items():
        if number(gb.get(key)) != expected:
            raise ContractError(f"MMGBSA {key} must equal {expected}")
    for key, expected in {"molsurf": 0, "surften": 0.0072, "surfoff": 0, "ifqnt": 0}.items():
        if number(gb.get(key, expected)) != expected:
            raise ContractError(f"MMGBSA {key} must equal {expected}")
    for key, expected in {"startframe": 1, "interval": 1, "entropy": 0}.items():
        if number(general.get(key, expected)) != expected:
            raise ContractError(f"MMGBSA {key} must equal {expected}")
    end = number(general.get("endframe", 9999999))
    if end is None or end < 250 or not end.is_integer():
        raise ContractError("MMGBSA must analyze all 250 staged frames")
    for key, expected in {"receptor_mask": RECEPTOR, "ligand_mask": LIGAND}.items():
        if key in general and mask_residues(str(general[key])) != expected:
            raise ContractError(f"incorrect MMGBSA {key}")
    if any(name not in {"general", "gb", "decomp"} for name in groups):
        raise ContractError("MMGBSA must use the unmutated classical single-trajectory calculation")
    return groups


def parse_delta_total(text: str | None) -> float | None:
    rows = [
        line.split()
        for line in (text or "").splitlines()
        if line.upper().split()[:2] == ["DELTA", "TOTAL"]
    ]
    if len(rows) != 1 or len(rows[0]) != 5:
        return None
    try:
        values = [float(value.lower().replace("d", "e")) for value in rows[0][2:]]
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in values) or any(value < 0 for value in values[1:]):
        return None
    return values[0]


def _topology_role(filename: str, roles: dict[str, frozenset]) -> frozenset:
    if filename in roles:
        return roles[filename]
    basename = posixpath.basename(filename)
    if basename in {SYSTEM_BASENAME + ".prmtop", SYSTEM_BASENAME + "_hmr.prmtop"}:
        return COMPLEX
    for suffix, role in {
        "A": RECEPTOR,
        "rec": RECEPTOR,
        "receptor": RECEPTOR,
        "receptor_A": RECEPTOR,
        "BC": LIGAND,
        "lig": LIGAND,
        "ligand": LIGAND,
        "ligand_BC": LIGAND,
    }.items():
        if basename == f"{SYSTEM_BASENAME}_{suffix}.prmtop":
            return role
    raise ContractError(f"unresolved topology role: {filename}")


def _topology_commands(commands: list, roles: dict[str, frozenset]) -> None:
    previous_files = {}
    for command in commands:
        executable = posixpath.basename(command.argv[0])
        for filename, content in command.files.items():
            if filename.endswith((".prmtop", ".parm7")) and previous_files.get(filename) != content:
                roles[filename] = frozenset()
        previous_files = command.files
        if executable == "MMPBSA.py":
            break
        if executable in {"cp", "mv"}:
            args = [value for value in command.argv[1:] if not value.startswith("-")]
            if len(args) == 2:
                source, target = (path(value, command.cwd) for value in args)
                if target.endswith((".prmtop", ".parm7")):
                    roles[target] = _topology_role(source, roles)
        elif executable == "rm":
            for value in command.argv[1:]:
                if not value.startswith("-"):
                    roles[path(value, command.cwd)] = frozenset()
        elif executable == "ante-MMPBSA.py":
            flags = options(command.argv)
            aliases = {
                "--prmtop": "-p",
                "--complex-prmtop": "-c",
                "--receptor-prmtop": "-r",
                "--ligand-prmtop": "-l",
                "--receptor-mask": "-m",
                "--ligand-mask": "-n",
            }
            flags = {aliases.get(key, key): value for key, value in flags.items()}
            source = path(flags["-p"], command.cwd)
            if _topology_role(source, roles) != COMPLEX:
                raise ContractError("ante-MMPBSA input is not the full complex")
            if "--radii" in flags and flags["--radii"] != "mbondi3":
                raise ContractError("MMGBSA topology radii must be mbondi3")
            if ("-m" in flags) == ("-n" in flags):
                raise ContractError("provide exactly one receptor or ligand split mask")
            selected = mask_residues(flags.get("-m", flags.get("-n", "")))
            receptor = selected if "-m" in flags else COMPLEX - selected
            ligand = COMPLEX - receptor
            if "-c" in flags:
                roles[path(flags["-c"], command.cwd)] = COMPLEX
            roles[path(flags["-r"], command.cwd)] = receptor
            roles[path(flags["-l"], command.cwd)] = ligand
        elif executable == "cpptraj":
            flags = options(command.argv)
            content = command.stdin or command.files.get(path(flags.get("-i", ""), command.cwd))
            if content is None:
                continue
            current = None
            for line in content.splitlines():
                words = shlex.split(line, comments=True)
                if not words:
                    continue
                if words[0] == "parm" and len(words) == 2:
                    current = _topology_role(path(words[1], command.cwd), roles)
                elif words[0] == "parmstrip" and len(words) == 2 and current is not None:
                    current -= mask_residues(words[1])
                elif words[:2] == ["parmwrite", "out"] and len(words) == 3 and current is not None:
                    roles[path(words[2], command.cwd)] = current
        elif executable == "tleap":
            flags = options(command.argv)
            content = command.files.get(path(flags.get("-f", ""), command.cwd))
            if content is not None:
                units = {}
                force_field = radii = False
                for words in leap_commands(content):
                    if words[0].lower() == "quit":
                        break
                    if words[0].lower() == "source":
                        force_field = (
                            len(words) == 2
                            and posixpath.basename(words[1]) == "leaprc.protein.ff14SB"
                        )
                    if (
                        len(words) >= 3
                        and words[0].lower() == "set"
                        and words[1] == "default"
                        and words[2] in {"PBRadii", "PBradii"}
                    ):
                        radii = words[3:] == ["mbondi3"]
                    if len(words) == 4 and words[1] == "=" and words[2].lower() == "loadpdb":
                        if not force_field or not radii:
                            raise ContractError("MMGBSA build must use ff14SB and mbondi3")
                        units[words[0]] = (
                            COMPLEX
                            if posixpath.basename(words[3]) == "complex_structure.pdb"
                            else frozenset()
                        )
                    if words[0].lower() == "saveamberparm" and len(words) == 4:
                        roles[path(words[2], command.cwd)] = units.get(words[1], frozenset())


def check_workflow(files: dict[str, str]) -> list[str]:
    shells, resolution_errors, known_products = _resolve_stages(
        tuple(files.get(name) for name in ("submit_min.sh", "submit_prod.sh", "submit_mmgbsa.sh"))
    )
    errors, roles = list(resolution_errors), {}
    for name in ("submit_min.sh", "submit_prod.sh", "submit_mmgbsa.sh"):
        if name not in files:
            continue
        text = files[name]
        if not text.startswith(("#!/bin/bash\n", "#!/usr/bin/env bash\n")):
            errors.append(f"{name}:missing bash shebang")
        prefix = []
        for line in text.splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                break
            prefix.append(line)
        if not any(line.startswith("#SBATCH ") for line in prefix):
            errors.append(f"{name}:missing active sbatch directives")
        try:
            if name in shells:
                _topology_commands(shells[name].commands, roles)
        except (ContractError, KeyError, ValueError, IndexError) as error:
            errors.append(f"{name}:{error}")
    previous_restart = None
    for name in ("submit_min.sh", "submit_prod.sh"):
        if name not in shells:
            continue
        try:
            runs = [
                command
                for command in shells[name].commands
                if posixpath.basename(command.argv[0]) == "pmemd.cuda"
            ]
            if len(runs) < 2 if name == "submit_min.sh" else len(runs) != 1:
                raise ContractError(
                    "missing minimization/equilibration or single production command"
                )
            seen_minimization = seen_md = False
            for index, command in enumerate(runs):
                flags = options(command.argv)
                if "-O" not in flags or not {"-i", "-p", "-c", "-o", "-r"} <= flags.keys():
                    raise ContractError("missing MD options")
                if _topology_role(path(flags["-p"], command.cwd), roles) != COMPLEX:
                    raise ContractError("MD topology is not the full complex")
                topology = path(flags["-p"], command.cwd)
                coordinates = path(flags["-c"], command.cwd)
                if topology in command.files or (
                    topology in known_products
                    and not command.products.get(topology, "").endswith(":topology")
                ):
                    raise ContractError("MD topology product is missing or overwritten")
                if previous_restart is not None and coordinates != previous_restart:
                    raise ContractError("broken restart handoff between MD stages")
                if (
                    previous_restart is None
                    and posixpath.basename(flags["-c"]) != SYSTEM_BASENAME + ".inpcrd"
                ):
                    raise ContractError("incorrect starting coordinates")
                coordinate_kind = "pmemd.cuda:restart" if previous_restart else "tleap:coordinates"
                if coordinates in command.files or (
                    (coordinates in known_products or previous_restart is not None)
                    and command.products.get(coordinates) != coordinate_kind
                ):
                    raise ContractError("MD coordinate/restart product is missing or overwritten")
                previous_restart = path(flags["-r"], command.cwd)
                content = command.files.get(path(flags["-i"], command.cwd))
                mode = 1 if name == "submit_min.sh" and index == 0 else 0
                if content is not None:
                    params = namelists(content).get("cntrl", {})
                    mode = number(params.get("imin", 0))
                    if (
                        mode not in {0, 1}
                        or number(params.get("ntb", 1)) != 0
                        or number(params.get("igb", 0)) not in {1, 2, 5, 7, 8}
                    ):
                        raise ContractError("incorrect implicit-solvent MD stage")
                    length = number(params.get("maxcyc" if mode else "nstlim", 1))
                    if length is None or length <= 0 or not length.is_integer():
                        raise ContractError("MD stage must have a positive integer step count")
                elif not flags["-i"].endswith((".mdin", ".in")):
                    raise ContractError("incorrect MD input artifact")
                if name == "submit_min.sh":
                    if mode == 1:
                        if seen_md:
                            raise ContractError("minimization must precede preparation MD")
                        seen_minimization = True
                    else:
                        if not seen_minimization:
                            raise ContractError("preparation MD must follow minimization")
                        seen_md = True
                elif mode != 0:
                    raise ContractError("production must be molecular dynamics")
                if name == "submit_prod.sh" and any(
                    posixpath.basename(flags.get(key, "")) != value
                    for key, value in {
                        "-o": "prod.out",
                        "-r": "prod.rst",
                        "-x": "prod.mdcrd",
                    }.items()
                ):
                    raise ContractError("incorrect production output wiring")
            if name == "submit_min.sh" and not (seen_minimization and seen_md):
                raise ContractError("missing minimization or preparation MD phase")
        except (ContractError, KeyError, ValueError) as error:
            errors.append(f"{name}:{error}")
    shell = shells.get("submit_mmgbsa.sh")
    if shell is not None:
        try:
            runs = [
                command
                for command in shell.commands
                if posixpath.basename(command.argv[0]) == "MMPBSA.py"
            ]
            if len(runs) != 1:
                raise ContractError("must invoke exactly one active MMPBSA.py calculation")
            run = runs[0]
            flags = options(run.argv)
            for key, role in {"-cp": COMPLEX, "-rp": RECEPTOR, "-lp": LIGAND}.items():
                topology = path(flags[key], run.cwd)
                if _topology_role(topology, roles) != role:
                    raise ContractError(f"incorrect {key} topology role")
                if topology in run.files or (
                    topology in known_products
                    and not run.products.get(topology, "").endswith(":topology")
                ):
                    raise ContractError(f"missing or overwritten {key} topology product")
            if posixpath.basename(flags["-o"]) != "FINAL_RESULTS_MMGBSA.dat":
                raise ContractError("incorrect MMGBSA result/trajectory artifact")
            if any(
                option in flags
                for option in {"-yr", "-yl", "-use-mdins", "-make-mdins", "-rewrite-output"}
            ):
                raise ContractError("must calculate from the staged single trajectory")
            trajectory = path(flags["-y"], run.cwd)
            if run.products.get(trajectory) != STAGED_TRAJECTORY or trajectory in run.files:
                raise ContractError("MMGBSA trajectory is not the unmodified staged input")
            content = run.files.get(path(flags["-i"], run.cwd))
            if content is None:
                raise ContractError("MMGBSA -i is not wired to generated namelist content")
            mmgbsa_parameters(content)
        except (ContractError, KeyError, ValueError) as error:
            errors.append(f"submit_mmgbsa.sh:{error}")
    return errors


def check_results(
    text: str, hidden_reference_text: str | None, files: dict[str, str] | None = None
) -> list[str]:
    errors = []
    delta = parse_delta_total(text)
    if delta is None:
        return ["FINAL_RESULTS_MMGBSA.dat:malformed or missing DELTA TOTAL"]
    if not -130 <= delta <= -100:
        errors.append("FINAL_RESULTS_MMGBSA.dat:delta total out of accepted range")
    hidden = parse_delta_total(hidden_reference_text)
    if hidden is not None and abs(delta - hidden) > 50:
        errors.append("FINAL_RESULTS_MMGBSA.dat:delta total too far from hidden reference")
    try:
        echoed = "\n".join(
            line.lstrip()[1:] for line in text.splitlines() if line.lstrip().startswith("|")
        )
        mmgbsa_parameters(echoed)
        frames = re.findall(
            r"Calculations\s+performed\s+using\s+(\S+)\s+complex\s+frames\.", text, re.I
        )
        if len(frames) != 1 or float(frames[0]) != 250:
            raise ContractError("result must report 250 complex frames")
        for label, expected in {"Receptor": RECEPTOR, "Ligand": LIGAND}.items():
            masks = re.findall(rf"^\|?\s*{label}\s+mask:\s*(.+)$", text, re.I | re.M)
            if len(masks) != 1 or mask_residues(masks[0]) != expected:
                raise ContractError(f"incorrect result {label.lower()} mask")
        if "GENERALIZED BORN:" not in text.upper():
            raise ContractError("missing Generalized Born result section")
        subtotals = []
        for label in ("gas", "solv"):
            rows = [
                line.split()
                for line in text.splitlines()
                if line.upper().split()[:3] == ["DELTA", "G", label.upper()]
            ]
            if len(rows) != 1 or len(rows[0]) != 6:
                raise ContractError("missing or malformed GB component totals")
            values = [float(value) for value in rows[0][3:]]
            if not all(math.isfinite(value) for value in values):
                raise ContractError("nonfinite GB component total")
            subtotals.append(values[0])
        if not math.isclose(sum(subtotals), delta, rel_tol=0, abs_tol=0.0002):
            raise ContractError("DELTA TOTAL inconsistent with gas and solvation components")
        if files and "submit_mmgbsa.sh" in files:
            shells, _, _ = _resolve_stages(
                tuple(
                    files.get(name)
                    for name in ("submit_min.sh", "submit_prod.sh", "submit_mmgbsa.sh")
                )
            )
            shell = shells.get("submit_mmgbsa.sh")
            if shell is None:
                raise ContractError("MMGBSA workflow could not be resolved")
            calls = [
                command
                for command in shell.commands
                if posixpath.basename(command.argv[0]) == "MMPBSA.py"
            ]
            if len(calls) != 1:
                raise ContractError("must resolve one active MMPBSA.py calculation")
            flags = options(calls[0].argv)
            for label, option in {"Complex": "-cp", "Receptor": "-rp", "Ligand": "-lp"}.items():
                headers = re.findall(
                    rf"^\|?\s*{label}\s+topology\s+file:\s*(.+)$", text, re.I | re.M
                )
                if len(headers) != 1 or posixpath.basename(
                    headers[0].strip().strip("\"'")
                ) != posixpath.basename(flags.get(option, "")):
                    raise ContractError(
                        f"result {label.lower()} topology differs from command input"
                    )
    except (ContractError, ValueError, KeyError, IndexError) as error:
        errors.append(f"FINAL_RESULTS_MMGBSA.dat:{error}")
    return errors
