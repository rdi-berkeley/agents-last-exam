"""Local evaluator for ECG analog front-end outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

REQUIRED_FILES = (
    "filter.cir",
    "amplifier.cir",
    "filter_sweep.csv",
    "amp_sweep.csv",
)

MIN_FILTER_SAMPLES = 8
MIN_AMP_SAMPLES = 4
MAX_FILTER_GAIN_RANGE = (7.2, 8.2)
FILTER_CUTOFF_RANGE = (79.0, 89.0)
AMP_GAIN_RANGE = (22.7, 23.7)
SWEEP_MIN_START_HZ = 1.05
SWEEP_MIN_END_HZ = 9500.0
MIN_NETLIST_BYTES = 32
AC_SOURCE_RE = re.compile(r"\bac\b", re.IGNORECASE)
SPICE_NUMBER_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*([a-z]+)?\s*$",
    re.IGNORECASE,
)
SPICE_SUFFIXES = {
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,
    "k": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}
FILTER_MIN_RESISTORS = 6
FILTER_MIN_CAPACITORS = 4
FILTER_MIN_ACTIVE_ELEMENTS = 2
AMP_MIN_RESISTORS = 2
AMP_MIN_ACTIVE_ELEMENTS = 1


@dataclass
class ScoreReport:
    score: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "details": self.details}


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("output", data, 0, len(data), "unable to decode text")


def _parse_csv_rows(data: bytes) -> list[tuple[float, float]]:
    text = _decode_text(data).strip()
    if not text:
        raise ValueError("csv is empty")

    rows: list[tuple[float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            parts = [cell.strip() for cell in line.split(",")]
        else:
            parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"expected exactly two columns, got {parts!r}")
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise ValueError(f"non-numeric csv row: {parts!r}") from exc
    if not rows:
        raise ValueError("csv contains no data rows")
    return rows


def _check_frequency_axis(rows: list[tuple[float, float]]) -> None:
    freqs = [freq for freq, _ in rows]
    if any(freq <= 0 for freq in freqs):
        raise ValueError("frequency axis must stay positive")
    if any(curr <= prev for prev, curr in zip(freqs, freqs[1:])):
        raise ValueError("frequency axis must be strictly increasing")
    if freqs[0] > SWEEP_MIN_START_HZ:
        raise ValueError(f"sweep starts too high: {freqs[0]}")
    if freqs[-1] < SWEEP_MIN_END_HZ:
        raise ValueError(f"sweep ends too early: {freqs[-1]}")


def _max_gain(rows: list[tuple[float, float]]) -> float:
    return max(mag for _, mag in rows)


def _nearest_cutoff_freq(rows: list[tuple[float, float]], target_mag: float) -> float:
    return min(rows, key=lambda row: abs(row[1] - target_mag))[0]


def _active_netlist_lines(data: bytes) -> list[str]:
    text = _decode_text(data)
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.split(";", 1)[0].strip()
        if not stripped or stripped.startswith("*"):
            continue
        lines.append(stripped)
    if not lines:
        raise ValueError("netlist has no active statements")
    return lines


def _parse_spice_number(token: str) -> float:
    match = SPICE_NUMBER_RE.match(token)
    if match is None:
        raise ValueError(f"invalid SPICE number: {token!r}")
    magnitude = float(match.group(1))
    suffix = (match.group(2) or "").lower()
    if suffix not in SPICE_SUFFIXES and suffix:
        raise ValueError(f"unsupported SPICE suffix: {suffix!r}")
    return magnitude * SPICE_SUFFIXES.get(suffix, 1.0)


def _count_prefixed_elements(lines: list[str], prefixes: tuple[str, ...]) -> int:
    normalized = tuple(prefix.upper() for prefix in prefixes)
    return sum(
        1
        for line in lines
        if line and not line.startswith(".") and line[0].upper() in normalized
    )


def _validate_common_netlist_requirements(lines: list[str]) -> None:
    if not any(line.lower() == ".end" for line in lines):
        raise ValueError("netlist is missing .end")
    if not any(
        line
        and not line.startswith(".")
        and line[0].upper() == "V"
        and AC_SOURCE_RE.search(line)
        for line in lines
    ):
        raise ValueError("netlist is missing an AC voltage source")

    ac_line = next((line for line in lines if line.lower().startswith(".ac ")), None)
    if ac_line is None:
        raise ValueError("netlist is missing an .ac sweep command")
    tokens = ac_line.split()
    if len(tokens) < 5:
        raise ValueError("netlist .ac command is incomplete")
    sweep_start_hz = _parse_spice_number(tokens[-2])
    sweep_end_hz = _parse_spice_number(tokens[-1])
    if sweep_start_hz > SWEEP_MIN_START_HZ:
        raise ValueError(f"netlist sweep starts too high: {sweep_start_hz}")
    if sweep_end_hz < SWEEP_MIN_END_HZ:
        raise ValueError(f"netlist sweep ends too early: {sweep_end_hz}")


def _validate_filter_netlist(data: bytes) -> None:
    lines = _active_netlist_lines(data)
    _validate_common_netlist_requirements(lines)
    resistor_count = _count_prefixed_elements(lines, ("R",))
    capacitor_count = _count_prefixed_elements(lines, ("C",))
    active_count = _count_prefixed_elements(lines, ("E", "X", "O", "U"))
    if resistor_count < FILTER_MIN_RESISTORS:
        raise ValueError(f"filter netlist has too few resistors: {resistor_count}")
    if capacitor_count < FILTER_MIN_CAPACITORS:
        raise ValueError(f"filter netlist has too few capacitors: {capacitor_count}")
    if active_count < FILTER_MIN_ACTIVE_ELEMENTS:
        raise ValueError(f"filter netlist has too few active stages: {active_count}")


def _validate_amplifier_netlist(data: bytes) -> None:
    lines = _active_netlist_lines(data)
    _validate_common_netlist_requirements(lines)
    resistor_count = _count_prefixed_elements(lines, ("R",))
    active_count = _count_prefixed_elements(lines, ("E", "X", "O", "U"))
    if resistor_count < AMP_MIN_RESISTORS:
        raise ValueError(f"amplifier netlist has too few resistors: {resistor_count}")
    if active_count < AMP_MIN_ACTIVE_ELEMENTS:
        raise ValueError(f"amplifier netlist has no active amplifier stage: {active_count}")


def score_submission(output_files: dict[str, bytes]) -> ScoreReport:
    details: dict[str, Any] = {
        "required_files_present": True,
    }

    for name in REQUIRED_FILES:
        payload = output_files.get(name)
        if payload is None:
            details["required_files_present"] = False
            details.setdefault("missing", []).append(name)
            return ScoreReport(0.0, details)

    for name in ("filter.cir", "amplifier.cir"):
        payload = output_files[name]
        if len(payload.strip()) < MIN_NETLIST_BYTES:
            details["required_files_present"] = False
            details.setdefault("empty_netlists", []).append(name)
            return ScoreReport(0.0, details)

    try:
        _validate_filter_netlist(output_files["filter.cir"])
        _validate_amplifier_netlist(output_files["amplifier.cir"])
    except ValueError as exc:
        details["netlist_validation_error"] = str(exc)
        return ScoreReport(0.0, details)

    try:
        filter_rows = _parse_csv_rows(output_files["filter_sweep.csv"])
        amp_rows = _parse_csv_rows(output_files["amp_sweep.csv"])
    except ValueError as exc:
        details["csv_parse_error"] = str(exc)
        return ScoreReport(0.0, details)

    details["filter_samples"] = len(filter_rows)
    details["amp_samples"] = len(amp_rows)

    if len(filter_rows) < MIN_FILTER_SAMPLES or len(amp_rows) < MIN_AMP_SAMPLES:
        details["sample_floor_failed"] = True
        return ScoreReport(0.0, details)

    try:
        _check_frequency_axis(filter_rows)
        _check_frequency_axis(amp_rows)
    except ValueError as exc:
        details["frequency_axis_error"] = str(exc)
        return ScoreReport(0.0, details)

    filter_max_gain = _max_gain(filter_rows)
    filter_cutoff_freq = _nearest_cutoff_freq(filter_rows, filter_max_gain - 3.0)
    amp_max_gain = _max_gain(amp_rows)

    details.update(
        {
            "filter_max_gain_db": filter_max_gain,
            "filter_cutoff_hz": filter_cutoff_freq,
            "amp_max_gain_db": amp_max_gain,
        }
    )

    if not (MAX_FILTER_GAIN_RANGE[0] <= filter_max_gain <= MAX_FILTER_GAIN_RANGE[1]):
        details["filter_gain_ok"] = False
        return ScoreReport(0.0, details)
    details["filter_gain_ok"] = True

    if not (FILTER_CUTOFF_RANGE[0] <= filter_cutoff_freq <= FILTER_CUTOFF_RANGE[1]):
        details["filter_cutoff_ok"] = False
        return ScoreReport(0.0, details)
    details["filter_cutoff_ok"] = True

    if not (AMP_GAIN_RANGE[0] <= amp_max_gain <= AMP_GAIN_RANGE[1]):
        details["amp_gain_ok"] = False
        return ScoreReport(0.0, details)
    details["amp_gain_ok"] = True

    return ScoreReport(1.0, details)
