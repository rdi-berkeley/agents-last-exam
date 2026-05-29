"""VM-side verifier for engineering/openroad_sky130_ibex_pnr_signoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

CLOCK_NAME = "clk_i"
DIE_AREA_TARGET_UM2 = 290_000.0
AREA_TOL_FULL = 0.08
AREA_TOL_PARTIAL_6 = 0.12
AREA_TOL_PARTIAL_3 = 0.20
POWER_TARGET_MW = 23.0
POWER_TOL_FULL = 0.15
POWER_TOL_PARTIAL_6 = 0.25
POWER_TOL_PARTIAL_3 = 0.40
SETUP_WNS_FULL = 0.0
SETUP_WNS_PARTIAL_10 = -0.25
SETUP_WNS_PARTIAL_5 = -0.75
SETUP_TNS_FULL = 0.0
SETUP_TNS_PARTIAL_7_NS = 0.5
SETUP_TNS_PARTIAL_3_NS = 5.0
HOLD_WNS_FULL = 0.0
HOLD_WNS_PARTIAL_6 = -0.05
DRC_FULL_MAX = 0
DRC_PARTIAL_7_MAX = 3
PASS_FULL_MAX = 8
PASS_PARTIAL_3_MAX = 12

FINAL_GDS_REL = "flow/results/sky130hd/ibex/base/6_final.gds"
FINAL_DEF_REL = "flow/results/sky130hd/ibex/base/6_final.def"
FINISH_RPT_REL = "flow/reports/sky130hd/ibex/base/6_finish.rpt"
DRC_COUNT_REL = "flow/reports/sky130hd/ibex/base/6_drc_count.rpt"
LVS_LOG_REL = "flow/logs/sky130hd/ibex/base/6_lvs.log"
SYNTH_STAT_REL = "flow/reports/sky130hd/ibex/base/synth_stat.txt"
PASS_LOG_DIR_REL = "flow/logs/sky130hd/ibex/base"
CTS_LOG_REL = "flow/logs/sky130hd/ibex/base/4_1_cts.log"
CTS_ILLEGAL_INSTRUCTION_MARKER = "child killed: illegal instruction"

BANNED_SDC_PATTERNS = [
    re.compile(r"\bset_false_path\b.*\bclk_i\b"),
    re.compile(r"\bset_multicycle_path\b.*\bclk_i\b"),
    re.compile(r"\bset_disable_timing\b.*\bclk_i\b"),
]

CONFIG_MK_IMMUTABLE_KEYS = {
    "DESIGN_NICKNAME": "ibex",
    "DESIGN_NAME": "ibex_core",
    "PLATFORM": "sky130hd",
    "SYNTH_HDL_FRONTEND": "slang",
    "SDC_FILE": "$(DESIGN_HOME)/constraint.sdc",
    "VERILOG_INCLUDE_DIRS": "$(DESIGN_HOME)/src/ibex_sv/vendor/lowrisc_ip/prim/rtl/",
    "VERILOG_FILES": (
        "$(sort $(wildcard $(DESIGN_HOME)/src/ibex_sv/*.sv))"
        " $(DESIGN_HOME)/src/ibex_sv/syn/rtl/prim_clock_gating.v"
    ),
}
CONFIG_MK_BANNED_KEY_PREFIXES = ("PRE_", "POST_", "EXTRA_", "CUSTOM_")
CONFIG_MK_BANNED_KEY_SUFFIXES = (
    "_HOOK",
    "_HOOKS",
    "_SCRIPT",
    "_SCRIPTS",
    "_TCL",
    "_INIT",
    "_PROLOGUE",
    "_EPILOGUE",
)

RUBRIC = {
    "G1": {"weight": 5, "name": "Artifacts present"},
    "G2": {"weight": 10, "name": "KLayout DRC = 0 violations"},
    "G3": {"weight": 5, "name": "KLayout LVS invoked + verdict reached"},
    "G4": {"weight": 10, "name": "OpenSTA setup WNS >= 0 ns"},
    "G5": {"weight": 10, "name": "OpenSTA setup TNS ~ 0 ns"},
    "G6": {"weight": 5, "name": "OpenSTA hold WNS >= 0 ns"},
    "G7": {"weight": 5, "name": "Die area within tolerance of target"},
    "G8": {"weight": 20, "name": "Total power within tolerance of target"},
    "G9": {"weight": 10, "name": "Reproducibility (reseed)"},
    "G10": {"weight": 5, "name": "Flow pass budget <= 8"},
    "G11": {"weight": 10, "name": "Journal fidelity"},
    "G12": {"weight": 5, "name": "Anti-gaming: frozen files + SDC hygiene"},
}

_UNSET_SYNONYMS = ("none", "(unset)", "unset", "(empty)", "empty", "-", "")


@dataclass
class GateResult:
    id: str
    name: str
    weight: int
    score: float
    passed: bool
    detail: str = ""


@dataclass
class Report:
    gates: list[GateResult] = field(default_factory=list)
    total: float = 0.0
    max_total: int = 100
    passed_overall: bool = False

    def add(self, gate: GateResult) -> None:
        self.gates.append(gate)
        self.total += gate.score


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def _extract_starter(reference_dir: Path, destination: Path) -> Path:
    starter_zip = reference_dir / "starter_project.zip"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(starter_zip) as zf:
        zf.extractall(destination)
    starter_dir = destination / "starter_project"
    if not starter_dir.exists():
        raise FileNotFoundError(f"starter_project missing after extract: {starter_zip}")
    return starter_dir


def _finish_section(rpt: str, header_marker: str) -> str:
    pat = re.compile(
        r"=+\s*\n\s*" + re.escape(header_marker) + r"\s*\n-+\s*\n",
        re.IGNORECASE,
    )
    match = pat.search(rpt)
    if not match:
        return ""
    tail = rpt[match.end():]
    next_sec = re.search(r"\n=+\s*\n", tail)
    return tail[: next_sec.start()] if next_sec else tail


def parse_setup_wns(rpt: str) -> float | None:
    section = _finish_section(rpt, "finish report_wns")
    if section:
        match = re.search(r"\bwns\s+max\s+(-?\d+(?:\.\d+)?)", section, re.IGNORECASE)
        if match:
            return float(match.group(1))
    match = re.search(r"^\s*wns\s+max\s+(-?\d+(?:\.\d+)?)", rpt, re.IGNORECASE | re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_setup_tns(rpt: str) -> float | None:
    section = _finish_section(rpt, "finish report_tns")
    if section:
        match = re.search(r"\btns\s+max\s+(-?\d+(?:\.\d+)?)", section, re.IGNORECASE)
        if match:
            return float(match.group(1))
    match = re.search(r"^\s*tns\s+max\s+(-?\d+(?:\.\d+)?)", rpt, re.IGNORECASE | re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_hold_wns(rpt: str) -> float | None:
    section = _finish_section(rpt, "finish report_checks -path_delay min")
    if not section:
        return None
    match = re.search(r"(-?\d+\.\d+)\s+slack\s*\(\s*(MET|VIOLATED)\s*\)", section, re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse_drc_count(text: str) -> int | None:
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else None


def parse_lvs_verdict(log_text: str) -> str | None:
    if not log_text.strip():
        return None
    if re.search(r"Congratulations! Netlists match", log_text, re.IGNORECASE):
        return "match"
    if re.search(r"Netlists don'?t match", log_text, re.IGNORECASE):
        return "mismatch"
    if re.search(r"ERROR|RuntimeError|Traceback", log_text):
        return "error"
    return None


def parse_power_mw(rpt: str) -> float | None:
    section = _finish_section(rpt, "finish report_power") or rpt
    match = re.search(
        r"^\s*Total\s+[\d.eE+\-]+\s+[\d.eE+\-]+\s+[\d.eE+\-]+\s+([\d.eE+\-]+)\s+[\d.]+%\s*$",
        section,
        re.MULTILINE,
    )
    if match:
        return float(match.group(1)) * 1000.0
    match = re.search(r"Total Power\s*[:=]\s*([\d.eE+\-]+)\s*W", section, re.IGNORECASE)
    return float(match.group(1)) * 1000.0 if match else None


def parse_die_area_um2(def_text: str) -> float | None:
    match = re.search(
        r"DIEAREA\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)",
        def_text,
    )
    if not match:
        return None
    x0, y0, x1, y1 = (int(value) for value in match.groups())
    units_match = re.search(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)", def_text)
    if not units_match:
        return None
    units = int(units_match.group(1))
    width_um = abs(x1 - x0) / units
    height_um = abs(y1 - y0) / units
    return width_um * height_um


def parse_config_mk_exports(text: str) -> dict[str, str]:
    joined_lines: list[str] = []
    buffer: list[str] = []
    for line in text.splitlines():
        if line.endswith("\\"):
            buffer.append(line[:-1].rstrip())
            continue
        if buffer:
            buffer.append(line.strip())
            joined_lines.append(" ".join(buffer).strip())
            buffer = []
        else:
            joined_lines.append(line)
    if buffer:
        joined_lines.append(" ".join(buffer).strip())

    parsed: dict[str, str] = {}
    for line in joined_lines:
        hash_idx = line.find("#")
        bare = line if hash_idx < 0 else line[:hash_idx]
        match = re.match(r"\s*export\s+([A-Z_][A-Z0-9_]*)\s*(?:\?=|=)\s*(.*)$", bare)
        if not match:
            continue
        parsed[match.group(1)] = _normalize_make_value(match.group(2))
    return parsed


def _normalize_make_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\\\n", " ")).strip()


def _pass_index(name: str, prefix: str, suffix: str = "") -> int | None:
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    core = name[len(prefix):len(name) - len(suffix)] if suffix else name[len(prefix):]
    try:
        return int(core)
    except ValueError:
        return None


def _journal_records_diff(section: str, var: str, before, after) -> bool:
    def _candidates(value) -> list[str]:
        if value is None or value == "":
            return list(_UNSET_SYNONYMS)
        return [str(value)]

    arrow = r"(?:->|→|=>)"
    for before_value in _candidates(before):
        for after_value in _candidates(after):
            before_pattern = (
                re.escape(before_value)
                if before_value
                else r"(?:\(unset\)|unset|None|\(empty\)|empty|-|\s|$)"
            )
            after_pattern = (
                re.escape(after_value)
                if after_value
                else r"(?:\(unset\)|unset|None|\(empty\)|empty|-|\s|$)"
            )
            pattern = re.compile(
                rf"{re.escape(var)}\s*:\s*{before_pattern}\s*{arrow}\s*{after_pattern}",
                re.IGNORECASE,
            )
            if pattern.search(section):
                return True
    return False


def gate_g1_artifacts(submission_dir: Path) -> GateResult:
    required = [
        ("flow/designs/sky130hd/ibex/config.mk", "config.mk"),
        ("JOURNAL.md",),
        (FINAL_GDS_REL,),
        (FINAL_DEF_REL,),
        (FINISH_RPT_REL,),
        (DRC_COUNT_REL,),
        (LVS_LOG_REL,),
        (SYNTH_STAT_REL,),
    ]
    missing = []
    for options in required:
        if not any((submission_dir / rel).exists() for rel in options):
            missing.append(options[0])
    passed = not missing
    return GateResult(
        id="G1",
        name=RUBRIC["G1"]["name"],
        weight=RUBRIC["G1"]["weight"],
        score=RUBRIC["G1"]["weight"] if passed else 0.0,
        passed=passed,
        detail=f"all {len(required)} required deliverables present" if passed else f"missing: {missing}",
    )


def gate_g2_drc(submission_dir: Path) -> GateResult:
    drc = parse_drc_count(_read(submission_dir / DRC_COUNT_REL))
    weight = RUBRIC["G2"]["weight"]
    if drc is None:
        return GateResult("G2", RUBRIC["G2"]["name"], weight, 0.0, False, "could not parse DRC count")
    if drc <= DRC_FULL_MAX:
        return GateResult("G2", RUBRIC["G2"]["name"], weight, weight, True, f"{drc} violations")
    if drc <= DRC_PARTIAL_7_MAX:
        return GateResult("G2", RUBRIC["G2"]["name"], weight, 7.0, False, f"{drc} violations (partial)")
    return GateResult("G2", RUBRIC["G2"]["name"], weight, 0.0, False, f"{drc} violations (fail)")


def gate_g3_lvs(submission_dir: Path) -> GateResult:
    weight = RUBRIC["G3"]["weight"]
    verdict = parse_lvs_verdict(_read(submission_dir / LVS_LOG_REL))
    if verdict == "match":
        return GateResult("G3", RUBRIC["G3"]["name"], weight, weight, True, "netlists match")
    if verdict == "mismatch":
        partial = weight / 2.0
        return GateResult(
            "G3",
            RUBRIC["G3"]["name"],
            weight,
            partial,
            False,
            f"verdict: mismatch (partial {partial}/{weight})",
        )
    if verdict == "error":
        return GateResult("G3", RUBRIC["G3"]["name"], weight, 0.0, False, "LVS crashed before verdict")
    return GateResult("G3", RUBRIC["G3"]["name"], weight, 0.0, False, "no LVS log")


def gate_g4_setup_wns(submission_dir: Path) -> GateResult:
    weight = RUBRIC["G4"]["weight"]
    wns = parse_setup_wns(_read(submission_dir / FINISH_RPT_REL))
    if wns is None:
        return GateResult("G4", RUBRIC["G4"]["name"], weight, 0.0, False, "could not parse setup WNS")
    if wns >= SETUP_WNS_FULL:
        return GateResult("G4", RUBRIC["G4"]["name"], weight, weight, True, f"WNS = {wns:+.3f} ns")
    if wns >= SETUP_WNS_PARTIAL_10:
        score = round(weight * 0.7, 1)
        return GateResult("G4", RUBRIC["G4"]["name"], weight, score, False, f"WNS = {wns:+.3f} ns (partial {score}/{weight})")
    if wns >= SETUP_WNS_PARTIAL_5:
        score = round(weight * 0.35, 1)
        return GateResult("G4", RUBRIC["G4"]["name"], weight, score, False, f"WNS = {wns:+.3f} ns (partial {score}/{weight})")
    return GateResult("G4", RUBRIC["G4"]["name"], weight, 0.0, False, f"WNS = {wns:+.3f} ns (fail)")


def gate_g5_setup_tns(submission_dir: Path) -> GateResult:
    weight = RUBRIC["G5"]["weight"]
    tns_ns = parse_setup_tns(_read(submission_dir / FINISH_RPT_REL))
    if tns_ns is None:
        return GateResult("G5", RUBRIC["G5"]["name"], weight, 0.0, False, "could not parse setup TNS")
    tns_abs = abs(tns_ns)
    if tns_ns >= SETUP_TNS_FULL:
        return GateResult("G5", RUBRIC["G5"]["name"], weight, weight, True, f"TNS = {tns_ns:+.3f} ns")
    if tns_abs <= SETUP_TNS_PARTIAL_7_NS:
        return GateResult("G5", RUBRIC["G5"]["name"], weight, 7.0, False, f"|TNS| = {tns_abs:.2f} ns (partial 7)")
    if tns_abs <= SETUP_TNS_PARTIAL_3_NS:
        return GateResult("G5", RUBRIC["G5"]["name"], weight, 3.0, False, f"|TNS| = {tns_abs:.2f} ns (partial 3)")
    return GateResult("G5", RUBRIC["G5"]["name"], weight, 0.0, False, f"|TNS| = {tns_abs:.2f} ns (fail)")


def gate_g6_hold_wns(submission_dir: Path) -> GateResult:
    weight = RUBRIC["G6"]["weight"]
    wns = parse_hold_wns(_read(submission_dir / FINISH_RPT_REL))
    if wns is None:
        return GateResult("G6", RUBRIC["G6"]["name"], weight, 0.0, False, "could not parse hold WNS")
    if wns >= HOLD_WNS_FULL:
        return GateResult("G6", RUBRIC["G6"]["name"], weight, weight, True, f"WNS = {wns:+.3f} ns")
    if wns >= HOLD_WNS_PARTIAL_6:
        score = round(weight * 0.6, 1)
        return GateResult("G6", RUBRIC["G6"]["name"], weight, score, False, f"WNS = {wns:+.3f} ns (partial {score}/{weight})")
    return GateResult("G6", RUBRIC["G6"]["name"], weight, 0.0, False, f"WNS = {wns:+.3f} ns (fail)")


def gate_g7_area(submission_dir: Path) -> GateResult:
    weight = RUBRIC["G7"]["weight"]
    area = parse_die_area_um2(_read(submission_dir / FINAL_DEF_REL))
    if area is None:
        return GateResult("G7", RUBRIC["G7"]["name"], weight, 0.0, False, "could not parse DIEAREA")
    frac = abs(area - DIE_AREA_TARGET_UM2) / DIE_AREA_TARGET_UM2
    if frac <= AREA_TOL_FULL:
        return GateResult("G7", RUBRIC["G7"]["name"], weight, weight, True, f"area = {area:.0f} um^2 ({frac * 100:+.1f}%)")
    if frac <= AREA_TOL_PARTIAL_6:
        score = round(weight * 0.6, 1)
        return GateResult("G7", RUBRIC["G7"]["name"], weight, score, False, f"area = {area:.0f} um^2 ({frac * 100:+.1f}%) partial {score}/{weight}")
    if frac <= AREA_TOL_PARTIAL_3:
        score = round(weight * 0.3, 1)
        return GateResult("G7", RUBRIC["G7"]["name"], weight, score, False, f"area = {area:.0f} um^2 ({frac * 100:+.1f}%) partial {score}/{weight}")
    return GateResult("G7", RUBRIC["G7"]["name"], weight, 0.0, False, f"area = {area:.0f} um^2 ({frac * 100:+.1f}%) fail")


def gate_g8_power(submission_dir: Path) -> GateResult:
    weight = RUBRIC["G8"]["weight"]
    power = parse_power_mw(_read(submission_dir / FINISH_RPT_REL))
    if power is None:
        return GateResult("G8", RUBRIC["G8"]["name"], weight, 0.0, False, "could not parse total power")
    frac = abs(power - POWER_TARGET_MW) / POWER_TARGET_MW
    if frac <= POWER_TOL_FULL:
        return GateResult("G8", RUBRIC["G8"]["name"], weight, weight, True, f"power = {power:.2f} mW ({frac * 100:+.1f}%)")
    if frac <= POWER_TOL_PARTIAL_6:
        score = round(weight * 0.6, 1)
        return GateResult("G8", RUBRIC["G8"]["name"], weight, score, False, f"power = {power:.2f} mW ({frac * 100:+.1f}%) partial {score}/{weight}")
    if frac <= POWER_TOL_PARTIAL_3:
        score = round(weight * 0.3, 1)
        return GateResult("G8", RUBRIC["G8"]["name"], weight, score, False, f"power = {power:.2f} mW ({frac * 100:+.1f}%) partial {score}/{weight}")
    return GateResult("G8", RUBRIC["G8"]["name"], weight, 0.0, False, f"power = {power:.2f} mW ({frac * 100:+.1f}%) fail")


def gate_g10_pass_budget(submission_dir: Path) -> GateResult:
    weight = RUBRIC["G10"]["weight"]
    pass_dir = submission_dir / PASS_LOG_DIR_REL
    stamps = sorted(pass_dir.glob("pass*.stamp")) if pass_dir.exists() else []
    snap_nums = {
        _pass_index(path.name, "config.mk.pass")
        for path in (pass_dir.glob("config.mk.pass*") if pass_dir.exists() else [])
    }
    stamp_nums = {_pass_index(path.name, "pass", ".stamp") for path in stamps}
    snap_nums.discard(None)
    stamp_nums.discard(None)
    paired = snap_nums & stamp_nums
    n_passes = len(paired)
    orphan_snaps = snap_nums - stamp_nums
    orphan_stamps = stamp_nums - snap_nums
    detail_tail = ""
    if orphan_snaps:
        detail_tail += f"; snapshots without stamps: {sorted(orphan_snaps)}"
    if orphan_stamps:
        detail_tail += f"; stamps without snapshots: {sorted(orphan_stamps)}"
    if n_passes == 0:
        return GateResult("G10", RUBRIC["G10"]["name"], weight, 0.0, False, f"0 completed flow passes{detail_tail}")
    if n_passes <= PASS_FULL_MAX:
        return GateResult("G10", RUBRIC["G10"]["name"], weight, weight, True, f"{n_passes} completed passes (<= {PASS_FULL_MAX}){detail_tail}")
    if n_passes <= PASS_PARTIAL_3_MAX:
        return GateResult("G10", RUBRIC["G10"]["name"], weight, 3.0, False, f"{n_passes} completed passes (partial 3){detail_tail}")
    return GateResult("G10", RUBRIC["G10"]["name"], weight, 0.0, False, f"{n_passes} completed passes (fail){detail_tail}")


def gate_g11_journal(submission_dir: Path, starter_dir: Path) -> GateResult:
    weight = RUBRIC["G11"]["weight"]
    journal = submission_dir / "JOURNAL.md"
    pass_dir = submission_dir / PASS_LOG_DIR_REL
    if not journal.exists():
        return GateResult("G11", RUBRIC["G11"]["name"], weight, 0.0, False, "JOURNAL.md missing")
    snapshots = sorted(
        (path for path in (pass_dir.glob("config.mk.pass*") if pass_dir.exists() else [])),
        key=lambda path: _pass_index(path.name, "config.mk.pass") or -1,
    )
    if not snapshots:
        return GateResult("G11", RUBRIC["G11"]["name"], weight, 0.0, False, "no config.mk.pass* snapshots")

    journal_text = journal.read_text()
    sections = re.split(r"^## Pass (\d+)", journal_text, flags=re.MULTILINE)
    sections_by_num: dict[int, str] = {}
    for idx in range(1, len(sections), 2):
        try:
            num = int(sections[idx])
        except ValueError:
            continue
        sections_by_num[num] = sections[idx + 1] if idx + 1 < len(sections) else ""

    def _is_stub(section: str) -> bool:
        if not section or not section.strip():
            return True
        return bool(re.search(r"_\(fill in", section))

    starter_cfg = starter_dir / "flow" / "designs" / "sky130hd" / "ibex" / "config.mk"
    starter_exports = parse_config_mk_exports(starter_cfg.read_text()) if starter_cfg.exists() else {}
    pass0_exports = parse_config_mk_exports(snapshots[0].read_text())
    starter_to_pass0 = {
        key: (starter_exports.get(key), pass0_exports.get(key))
        for key in set(starter_exports) | set(pass0_exports)
        if starter_exports.get(key) != pass0_exports.get(key)
    }

    total_expected = 0
    total_matched = 0
    details: list[str] = []

    if starter_to_pass0:
        section = sections_by_num.get(1, "")
        if _is_stub(section):
            details.append("pass 1 (starter→pass0) journal section empty or stub")
            total_expected += len(starter_to_pass0)
        else:
            if not re.search(r"(?im)^-\s*\*\*Observation", section):
                details.append("pass 1: missing Observation bullet")
            for var, (before, after) in starter_to_pass0.items():
                total_expected += 1
                if _journal_records_diff(section, var, before, after):
                    total_matched += 1
                else:
                    details.append(f"pass 1: journal missing {var}: {before} -> {after}")

    prev_exports = pass0_exports
    for idx, snapshot in enumerate(snapshots[1:], start=1):
        cur = parse_config_mk_exports(snapshot.read_text())
        diff = {
            key: (prev_exports.get(key), cur.get(key))
            for key in set(prev_exports) | set(cur)
            if prev_exports.get(key) != cur.get(key)
        }
        prev_exports = cur
        section_num = idx + 1 if starter_to_pass0 else idx
        section = sections_by_num.get(section_num, "")
        if _is_stub(section):
            details.append(f"pass {section_num}: journal section empty or stub")
            total_expected += len(diff)
            continue
        if not re.search(r"(?im)^-\s*\*\*Observation", section):
            details.append(f"pass {section_num}: missing Observation bullet")
        for var, (before, after) in diff.items():
            total_expected += 1
            if _journal_records_diff(section, var, before, after):
                total_matched += 1
            else:
                details.append(f"pass {section_num}: journal missing {var}: {before} -> {after}")

    if total_expected == 0:
        pass1 = sections_by_num.get(1, "")
        if _is_stub(pass1):
            return GateResult(
                "G11",
                RUBRIC["G11"]["name"],
                weight,
                0.0,
                False,
                "no parameter diffs across passes and JOURNAL.md Pass 1 is empty/stub",
            )
        return GateResult(
            "G11",
            RUBRIC["G11"]["name"],
            weight,
            round(weight * 0.3, 1),
            False,
            "no parameter diffs across passes; journal non-empty but unverifiable",
        )

    ratio = total_matched / total_expected
    if ratio >= 0.90:
        score = weight
        passed = True
    elif ratio >= 0.70:
        score = round(weight * 0.6, 1)
        passed = False
    else:
        score = 0.0
        passed = False
    detail = f"fidelity = {total_matched}/{total_expected} ({ratio * 100:.0f}%)"
    if details:
        detail += "; " + "; ".join(details[:5])
        if len(details) > 5:
            detail += f"; (+{len(details) - 5} more)"
    return GateResult("G11", RUBRIC["G11"]["name"], weight, score, passed, detail)


def gate_g12_anti_gaming(submission_dir: Path, starter_dir: Path, frozen_hashes: dict[str, str]) -> GateResult:
    weight = RUBRIC["G12"]["weight"]
    violations: list[str] = []

    for rel_path, expected_hash in frozen_hashes.items():
        target = submission_dir / rel_path
        if not target.exists():
            violations.append(f"frozen file missing: {rel_path}")
            continue
        actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            violations.append(f"frozen file hash changed: {rel_path}")

    for sdc in submission_dir.rglob("*.sdc"):
        text = sdc.read_text(errors="replace")
        stripped_lines = []
        for line in text.splitlines():
            hash_idx = line.find("#")
            stripped_lines.append(line if hash_idx < 0 else line[:hash_idx])
        stripped = "\n".join(stripped_lines)
        for pattern in BANNED_SDC_PATTERNS:
            if pattern.search(stripped):
                violations.append(f"banned SDC construct in {sdc.relative_to(submission_dir)}: {pattern.pattern}")

    starter_design_sdcs = {
        str(path.relative_to(starter_dir))
        for path in starter_dir.rglob("*.sdc")
        if "flow/designs" in str(path.relative_to(starter_dir))
    }
    for sdc in submission_dir.rglob("*.sdc"):
        rel_path = str(sdc.relative_to(submission_dir))
        if "flow/designs" not in rel_path:
            continue
        if rel_path not in starter_design_sdcs:
            violations.append(f"new SDC file introduced under flow/designs: {rel_path}")

    agent_cfg = submission_dir / "flow" / "designs" / "sky130hd" / "ibex" / "config.mk"
    if not agent_cfg.exists():
        agent_cfg = submission_dir / "config.mk"
    if agent_cfg.exists():
        exports = parse_config_mk_exports(agent_cfg.read_text())
        for key, required_value in CONFIG_MK_IMMUTABLE_KEYS.items():
            actual = exports.get(key)
            if actual is None:
                violations.append(f"config.mk missing required immutable key: {key}")
                continue
            if _normalize_make_value(actual) != _normalize_make_value(required_value):
                violations.append(
                    f"config.mk immutable key {key} changed: '{actual}' (expected '{required_value}')"
                )
        for key in exports:
            if any(key.startswith(prefix) for prefix in CONFIG_MK_BANNED_KEY_PREFIXES) or any(
                key.endswith(suffix) for suffix in CONFIG_MK_BANNED_KEY_SUFFIXES
            ):
                violations.append(f"config.mk banned hook/script key: {key}")
    else:
        violations.append("config.mk missing — cannot enforce edit-allowlist")

    if not violations:
        return GateResult(
            "G12",
            RUBRIC["G12"]["name"],
            weight,
            weight,
            True,
            "all frozen files intact; no banned SDC constructs; config.mk allowlist clean",
        )
    return GateResult("G12", RUBRIC["G12"]["name"], weight, 0.0, False, "; ".join(violations[:8]))


def _run_docker_flow_once(work_dir: Path, env: dict[str, str] | None = None) -> int:
    dockerfile = work_dir / "Dockerfile"
    match = re.search(r"FROM\s+openroad/orfs@(sha256:[0-9a-f]+)", dockerfile.read_text())
    if not match:
        print("[docker] no pinned digest in Dockerfile; aborting", file=sys.stderr)
        return 2

    image = f"openroad/orfs@{match.group(1)}"
    uid = os.getuid()
    gid = os.getgid()
    tmp_dir = work_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command: list[str] = []
    if os.geteuid() != 0:
        command.extend(["sudo", "-n"])
    command.extend(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--user",
            f"{uid}:{gid}",
            "-v",
            f"{work_dir}:/workspace",
            "-w",
            "/workspace",
            "-e",
            "HOME=/workspace",
            "-e",
            "TMPDIR=/workspace/.tmp",
            "-e",
            "TMP=/workspace/.tmp",
            "-e",
            "TEMP=/workspace/.tmp",
        ]
    )
    for key, value in (env or {}).items():
        command.extend(["-e", f"{key}={value}"])
    command.extend([image, "bash", "-lc", "make run"])
    print(f"[docker] {' '.join(command)}", file=sys.stderr)
    return subprocess.run(command, check=False).returncode


def _cts_illegal_instruction_detected(work_dir: Path) -> bool:
    cts_log = work_dir / CTS_LOG_REL
    if not cts_log.exists():
        return False
    return CTS_ILLEGAL_INSTRUCTION_MARKER in cts_log.read_text(errors="ignore")


def run_docker_flow(work_dir: Path, env: dict[str, str] | None = None) -> int:
    rc = _run_docker_flow_once(work_dir, env=env)
    if rc == 0:
        return 0

    if (env or {}).get("SKIP_CTS_REPAIR_TIMING"):
        return rc

    if not _cts_illegal_instruction_detected(work_dir):
        return rc

    retry_env = dict(env or {})
    retry_env["SKIP_CTS_REPAIR_TIMING"] = "1"
    print(
        "[docker] detected CTS illegal-instruction failure; retrying with SKIP_CTS_REPAIR_TIMING=1",
        file=sys.stderr,
    )
    return _run_docker_flow_once(work_dir, env=retry_env)


def reseed_environment(seed: int) -> dict[str, str]:
    return {
        "GPL_RANDOM_SEED": str(seed),
        "YOSYS_SEED": str(seed),
        "OPENROAD_SEED": str(seed),
    }


def score_submission(submission_dir: Path, starter_dir: Path, frozen_hashes: dict[str, str]) -> Report:
    report = Report()
    for gate_fn in (
        gate_g1_artifacts,
        gate_g2_drc,
        gate_g3_lvs,
        gate_g4_setup_wns,
        gate_g5_setup_tns,
        gate_g6_hold_wns,
        gate_g7_area,
        gate_g8_power,
        gate_g10_pass_budget,
    ):
        report.add(gate_fn(submission_dir))
    report.add(gate_g11_journal(submission_dir, starter_dir))
    report.add(gate_g12_anti_gaming(submission_dir, starter_dir, frozen_hashes))
    return report


def gate_g9_reseed(
    starter_dir: Path,
    submission_config: Path,
    submission_journal: Path,
    tmp_root: Path,
    frozen_hashes: dict[str, str],
) -> GateResult:
    weight = RUBRIC["G9"]["weight"]
    reseed_dir = tmp_root / "reseed"
    if reseed_dir.exists():
        shutil.rmtree(reseed_dir)
    shutil.copytree(starter_dir, reseed_dir)
    shutil.copy(submission_config, reseed_dir / "flow" / "designs" / "sky130hd" / "ibex" / "config.mk")
    shutil.copy(submission_journal, reseed_dir / "JOURNAL.md")

    rc = run_docker_flow(reseed_dir, env=reseed_environment(seed=20260417))
    if rc != 0:
        return GateResult("G9", RUBRIC["G9"]["name"], weight, 0.0, False, f"reseed flow exited {rc}")

    report = score_submission(reseed_dir, starter_dir, frozen_hashes)
    core_gate_ids = {"G2", "G3", "G4", "G5", "G6", "G7", "G8"}
    hard_gates = [gate for gate in report.gates if gate.id in core_gate_ids]
    passed = all(gate.passed for gate in hard_gates)
    score = weight if passed else sum(gate.score for gate in hard_gates) / sum(gate.weight for gate in hard_gates) * weight
    detail = "reseed passes G2-G8" if passed else "reseed: " + ", ".join(f"{gate.id}={gate.score}/{gate.weight}" for gate in hard_gates)
    return GateResult("G9", RUBRIC["G9"]["name"], weight, round(score, 2), passed, detail)


def finalize(report: Report) -> Report:
    report.total = round(sum(gate.score for gate in report.gates), 2)
    by_id = {gate.id: gate for gate in report.gates}
    g2 = by_id["G2"]
    g3 = by_id["G3"]
    g4 = by_id["G4"]
    g5 = by_id["G5"]
    g8 = by_id["G8"]
    g12 = by_id["G12"]
    closure_ok = (
        g4.score >= g4.weight / 2.0
        and g5.score >= g5.weight / 2.0
        and g8.score >= g8.weight / 2.0
    )
    report.passed_overall = (
        report.total >= 80.0
        and g2.passed
        and g3.score >= g3.weight / 2.0
        and closure_ok
        and g12.passed
    )
    return report


def _copy_audit_trail(submission_dir: Path, graded_dir: Path) -> None:
    audit_src = submission_dir / PASS_LOG_DIR_REL
    audit_dst = graded_dir / PASS_LOG_DIR_REL
    if not audit_src.exists():
        return
    audit_dst.mkdir(parents=True, exist_ok=True)
    for pattern in ("config.mk.pass*", "pass*.stamp"):
        for path in audit_src.glob(pattern):
            shutil.copy(path, audit_dst / path.name)


def _load_reference_data(reference_dir: Path) -> dict[str, object]:
    frozen_hashes = json.loads((reference_dir / "frozen_hashes.json").read_text())
    reference_metrics = json.loads((reference_dir / "reference_metrics.json").read_text())
    return {
        "frozen_hashes": frozen_hashes,
        "reference_metrics": reference_metrics,
    }


def _copy_debug_flow_outputs(submission_dir: Path, graded_dir: Path) -> None:
    agent_flow = submission_dir / "flow"
    if not agent_flow.exists():
        return
    for subdir in ("results", "reports", "logs", "objects"):
        src = agent_flow / subdir
        if not src.exists():
            continue
        for file_path in src.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(agent_flow)
            destination = graded_dir / "flow" / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, destination)


def _serialize_report(report: Report, reference_metrics: dict[str, object], work_dir: Path) -> dict[str, object]:
    return {
        "score": report.total,
        "total_score": report.total,
        "normalized_score": round(report.total / report.max_total, 6),
        "passed": report.passed_overall,
        "max_total": report.max_total,
        "gates": [asdict(gate) for gate in sorted(report.gates, key=lambda gate: int(gate.id[1:]))],
        "reference_metrics": reference_metrics,
        "work_dir": str(work_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--skip-docker-unsafe", action="store_true")
    parser.add_argument("--skip-reseed", action="store_true")
    args = parser.parse_args()

    submission_dir = args.submission_dir.resolve()
    reference_dir = args.reference_dir.resolve()
    if not submission_dir.exists():
        payload = {"score": 0.0, "normalized_score": 0.0, "passed": False, "error": f"submission dir missing: {submission_dir}"}
        print(json.dumps(payload))
        return 1
    if not reference_dir.exists():
        payload = {"score": 0.0, "normalized_score": 0.0, "passed": False, "error": f"reference dir missing: {reference_dir}"}
        print(json.dumps(payload))
        return 1

    work_root = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="orfs_ibex_eval_"))
    starter_unpack_root = work_root / "starter_unpack"
    starter_dir = _extract_starter(reference_dir, starter_unpack_root)
    reference_data = _load_reference_data(reference_dir)
    frozen_hashes = reference_data["frozen_hashes"]
    reference_metrics = reference_data["reference_metrics"]

    graded_dir = work_root / "graded"
    if graded_dir.exists():
        shutil.rmtree(graded_dir)
    shutil.copytree(starter_dir, graded_dir)

    submission_config = submission_dir / "config.mk"
    submission_journal = submission_dir / "JOURNAL.md"
    if not submission_config.exists() or not submission_journal.exists():
        payload = {
            "score": 0.0,
            "normalized_score": 0.0,
            "passed": False,
            "error": "submission must contain config.mk and JOURNAL.md",
        }
        print(json.dumps(payload))
        return 1

    shutil.copy(submission_config, graded_dir / "flow" / "designs" / "sky130hd" / "ibex" / "config.mk")
    shutil.copy(submission_journal, graded_dir / "JOURNAL.md")
    _copy_audit_trail(submission_dir, graded_dir)

    if args.skip_docker_unsafe:
        print("[warning] running in skip-docker debug mode", file=sys.stderr)
        _copy_debug_flow_outputs(submission_dir, graded_dir)
    else:
        rc = run_docker_flow(graded_dir)
        if rc != 0:
            print(f"[warn] graded flow exited {rc}; scoring what exists in the tree", file=sys.stderr)

    report = score_submission(graded_dir, starter_dir, frozen_hashes)

    if not args.skip_docker_unsafe and not args.skip_reseed:
        g9 = gate_g9_reseed(
            starter_dir,
            graded_dir / "flow" / "designs" / "sky130hd" / "ibex" / "config.mk",
            graded_dir / "JOURNAL.md",
            work_root,
            frozen_hashes,
        )
    else:
        reason = "--skip-docker-unsafe" if args.skip_docker_unsafe else "--skip-reseed"
        g9 = GateResult("G9", RUBRIC["G9"]["name"], RUBRIC["G9"]["weight"], 0.0, False, f"reseed skipped ({reason})")
    report.add(g9)

    finalize(report)
    print(json.dumps(_serialize_report(report, reference_metrics, work_root), ensure_ascii=True))
    return 0 if report.passed_overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
