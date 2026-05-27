"""Score the KiCad navswitch repair output.

The script is intentionally dependency-free so it can run on the Windows VM
with the standard Python install used by the benchmark harness.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_BASENAME = "SparkFun_Qwiic_Navigation"
TARGET_REF = "U4"
APPROVED_U4_FOOTPRINT = "nav_local_footprints:PCA9554_PW_local"
PROJECT_FILES = [
    f"{PROJECT_BASENAME}.kicad_pro",
    f"{PROJECT_BASENAME}.kicad_sch",
    f"{PROJECT_BASENAME}.kicad_pcb",
    "fp-lib-table",
    "sym-lib-table",
    "nav_local_symbols.kicad_sym",
]
REQUIRED_OUTPUT_FILES = [
    "erc.json",
    "drc.json",
    "project.net",
    "board.xml",
    "board.d356",
    "placements.csv",
]
REQUIRED_OUTPUT_DIRS = ["gerbers", "drill"]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _find_file(root: Path, filename: str) -> Path | None:
    direct = root / filename
    if direct.is_file():
        return direct
    project = root / "project" / filename
    if project.is_file():
        return project
    matches = [p for p in root.rglob(filename) if p.is_file()]
    if not matches:
        return None
    return min(matches, key=lambda p: (len(p.parts), str(p).lower()))


def _find_dir(root: Path, dirname: str) -> Path | None:
    direct = root / dirname
    if direct.is_dir():
        return direct
    project = root / "project" / dirname
    if project.is_dir():
        return project
    matches = [p for p in root.rglob(dirname) if p.is_dir()]
    if not matches:
        return None
    return min(matches, key=lambda p: (len(p.parts), str(p).lower()))


def _balanced_blocks(text: str, head: str) -> list[str]:
    blocks: list[str] = []
    token = f"({head}"
    start = 0
    while True:
        idx = text.find(token, start)
        if idx < 0:
            break
        depth = 0
        in_string = False
        escaped = False
        for pos in range(idx, len(text)):
            char = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    blocks.append(text[idx : pos + 1])
                    start = pos + 1
                    break
        else:
            break
    return blocks


def _property_value(block: str, name: str) -> str | None:
    match = re.search(rf'\(property\s+"{re.escape(name)}"\s+"([^"]*)"', block)
    return match.group(1) if match else None


def _schematic_u4_block(text: str) -> str | None:
    for block in _balanced_blocks(text, "symbol"):
        if "(lib_id " not in block:
            continue
        if _property_value(block, "Reference") == TARGET_REF:
            return block
    return None


def _pcb_u4_block(text: str) -> str | None:
    for block in _balanced_blocks(text, "footprint"):
        if _property_value(block, "Reference") == TARGET_REF:
            return block
    return None


def _footprint_name_from_pcb_block(block: str) -> str | None:
    match = re.match(r'\(footprint\s+"([^"]+)"', block.lstrip())
    return match.group(1) if match else None


def _component_footprints_from_schematic(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for block in _balanced_blocks(text, "symbol"):
        if "(lib_id " not in block:
            continue
        ref = _property_value(block, "Reference")
        footprint = _property_value(block, "Footprint")
        if ref and footprint is not None and not ref.startswith("#"):
            result[ref] = footprint
    return result


def _pcb_footprints_by_ref(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for block in _balanced_blocks(text, "footprint"):
        ref = _property_value(block, "Reference")
        name = _footprint_name_from_pcb_block(block)
        if ref and name:
            result[ref] = name
    return result


def _netlist_u4_pin_nets(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for block in _balanced_blocks(text, "net"):
        name_match = re.search(r'\(name\s+"([^"]*)"\)', block)
        if not name_match:
            continue
        net_name = name_match.group(1)
        for pin in re.findall(r'\(node\s+\(ref\s+"U4"\)\s+\(pin\s+"([^"]+)"\)', block):
            pins[pin] = net_name
    return pins


def _json_report_signature(path: Path, kind: str) -> dict[str, Any]:
    data = json.loads(_read_text(path))
    if kind == "erc":
        violations = []
        for sheet in data.get("sheets", []):
            violations.extend(sheet.get("violations", []))
        return {
            "violations": Counter(
                (item.get("severity", ""), item.get("type", "")) for item in violations
            )
        }
    return {
        "violations": Counter(
            (item.get("severity", ""), item.get("type", "")) for item in data.get("violations", [])
        ),
        "unconnected": len(data.get("unconnected_items", [])),
        "schematic_parity": len(data.get("schematic_parity", [])),
    }


def _report_compatible(agent: Path, reference: Path, kind: str) -> bool:
    try:
        return _json_report_signature(agent, kind) == _json_report_signature(reference, kind)
    except Exception:
        return False


def _non_empty_dir(path: Path) -> bool:
    return path.is_dir() and any(p.is_file() and p.stat().st_size > 0 for p in path.rglob("*"))


def _gerber_names(root: Path) -> set[str]:
    gerbers = _find_dir(root, "gerbers")
    if not gerbers:
        return set()
    return {p.name for p in gerbers.iterdir() if p.is_file() and p.stat().st_size > 0}


def _drill_names(root: Path) -> set[str]:
    drill = _find_dir(root, "drill")
    if not drill:
        return set()
    return {p.name for p in drill.iterdir() if p.is_file() and p.stat().st_size > 0}


def _placements_contains_u4(path: Path) -> bool:
    try:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            for row in csv.DictReader(handle):
                if (row.get("Ref") or "").strip().strip('"') == TARGET_REF:
                    return True
    except Exception:
        return False
    return False


def score(agent_dir: Path, reference_project_dir: Path, reference_outputs_dir: Path) -> dict[str, Any]:
    details: dict[str, Any] = {"checks": {}, "missing": []}

    if not agent_dir.is_dir():
        return {"score": 0.0, "details": {"error": f"agent dir not found: {agent_dir}"}}

    project_paths = {name: _find_file(agent_dir, name) for name in PROJECT_FILES}
    for name, path in project_paths.items():
        ok = path is not None and path.is_file() and path.stat().st_size > 0
        details["checks"][f"project_file:{name}"] = ok
        if not ok:
            details["missing"].append(name)

    footprint_dir = _find_dir(agent_dir, "nav_local_footprints.pretty")
    footprint_mod = (
        footprint_dir / "PCA9554_PW_local.kicad_mod"
        if footprint_dir is not None
        else None
    )
    details["checks"]["project_local_footprint_file"] = bool(
        footprint_mod and footprint_mod.is_file() and footprint_mod.stat().st_size > 0
    )

    required_paths = {name: _find_file(agent_dir, name) for name in REQUIRED_OUTPUT_FILES}
    for name, path in required_paths.items():
        ok = path is not None and path.is_file() and path.stat().st_size > 0
        details["checks"][f"deliverable:{name}"] = ok
        if not ok:
            details["missing"].append(name)
    for dirname in REQUIRED_OUTPUT_DIRS:
        path = _find_dir(agent_dir, dirname)
        ok = path is not None and _non_empty_dir(path)
        details["checks"][f"deliverable_dir:{dirname}"] = ok
        if not ok:
            details["missing"].append(dirname)

    sch_path = project_paths[f"{PROJECT_BASENAME}.kicad_sch"]
    pcb_path = project_paths[f"{PROJECT_BASENAME}.kicad_pcb"]
    ref_sch_path = reference_project_dir / f"{PROJECT_BASENAME}.kicad_sch"
    ref_pcb_path = reference_project_dir / f"{PROJECT_BASENAME}.kicad_pcb"
    if not sch_path or not pcb_path or not ref_sch_path.is_file() or not ref_pcb_path.is_file():
        details["hard_fail"] = "missing project or reference source"
        return {"score": 0.0, "details": details}

    sch_text = _read_text(sch_path)
    pcb_text = _read_text(pcb_path)
    ref_sch_text = _read_text(ref_sch_path)
    ref_pcb_text = _read_text(ref_pcb_path)

    u4_sch = _schematic_u4_block(sch_text)
    u4_pcb = _pcb_u4_block(pcb_text)
    u4_footprint = _property_value(u4_sch, "Footprint") if u4_sch else None
    u4_pcb_footprint = _footprint_name_from_pcb_block(u4_pcb) if u4_pcb else None
    details["checks"]["u4_in_schematic"] = u4_sch is not None
    details["checks"]["u4_in_pcb"] = u4_pcb is not None
    details["checks"]["u4_footprint_correct"] = u4_footprint == APPROVED_U4_FOOTPRINT
    details["checks"]["u4_pcb_footprint_correct"] = u4_pcb_footprint == APPROVED_U4_FOOTPRINT

    hard_gates = [
        details["checks"]["u4_in_schematic"],
        details["checks"]["u4_in_pcb"],
        details["checks"]["u4_footprint_correct"],
        details["checks"]["u4_pcb_footprint_correct"],
        all(details["checks"][f"deliverable:{name}"] for name in REQUIRED_OUTPUT_FILES),
        all(details["checks"][f"deliverable_dir:{name}"] for name in REQUIRED_OUTPUT_DIRS),
        all(details["checks"][f"project_file:{name}"] for name in PROJECT_FILES[:3]),
    ]
    if not all(hard_gates):
        details["hard_fail"] = "one or more required repair or deliverable gates failed"
        return {"score": 0.0, "details": details}

    ref_component_fps = _component_footprints_from_schematic(ref_sch_text)
    component_fps = _component_footprints_from_schematic(sch_text)
    comparable_refs = {ref for ref in ref_component_fps if ref != TARGET_REF}
    changed_refs = sorted(
        ref for ref in comparable_refs if component_fps.get(ref) != ref_component_fps.get(ref)
    )
    details["checks"]["non_target_schematic_footprints_preserved"] = not changed_refs
    if changed_refs:
        details["changed_non_target_schematic_refs"] = changed_refs[:20]

    ref_pcb_refs = set(_pcb_footprints_by_ref(ref_pcb_text))
    pcb_refs = set(_pcb_footprints_by_ref(pcb_text))
    missing_pcb_refs = sorted(ref for ref in ref_pcb_refs if ref not in pcb_refs)
    extra_pcb_refs = sorted(ref for ref in pcb_refs if ref not in ref_pcb_refs)
    details["checks"]["pcb_reference_set_preserved"] = not missing_pcb_refs and not extra_pcb_refs
    if missing_pcb_refs:
        details["missing_pcb_refs"] = missing_pcb_refs[:20]
    if extra_pcb_refs:
        details["extra_pcb_refs"] = extra_pcb_refs[:20]

    netlist_path = required_paths["project.net"]
    ref_netlist_path = reference_outputs_dir / "project.net"
    agent_u4_nets = _netlist_u4_pin_nets(_read_text(netlist_path))
    ref_u4_nets = _netlist_u4_pin_nets(_read_text(ref_netlist_path))
    details["checks"]["u4_netlist_matches_reference"] = agent_u4_nets == ref_u4_nets

    erc_ok = _report_compatible(required_paths["erc.json"], reference_outputs_dir / "erc.json", "erc")
    drc_ok = _report_compatible(required_paths["drc.json"], reference_outputs_dir / "drc.json", "drc")
    details["checks"]["erc_baseline_signature_matches"] = erc_ok
    details["checks"]["drc_baseline_signature_matches"] = drc_ok

    details["checks"]["gerber_file_set_matches"] = _gerber_names(agent_dir) == _gerber_names(
        reference_outputs_dir
    )
    details["checks"]["drill_file_set_matches"] = _drill_names(agent_dir) == _drill_names(
        reference_outputs_dir
    )
    details["checks"]["placements_include_u4"] = _placements_contains_u4(
        required_paths["placements.csv"]
    )

    weighted_checks = {
        "project_local_footprint_file": 0.05,
        "non_target_schematic_footprints_preserved": 0.10,
        "pcb_reference_set_preserved": 0.10,
        "u4_netlist_matches_reference": 0.20,
        "erc_baseline_signature_matches": 0.10,
        "drc_baseline_signature_matches": 0.10,
        "gerber_file_set_matches": 0.10,
        "drill_file_set_matches": 0.05,
        "placements_include_u4": 0.05,
    }
    base = 0.15
    score_value = base + sum(
        weight for key, weight in weighted_checks.items() if details["checks"].get(key)
    )
    score_value = round(min(score_value, 1.0), 6)
    return {"score": score_value, "details": details}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--reference-project-dir", required=True)
    parser.add_argument("--reference-outputs-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = score(
        Path(args.agent_dir),
        Path(args.reference_project_dir),
        Path(args.reference_outputs_dir),
    )
    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
