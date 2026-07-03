"""Shared helpers for Linux-based materials-science benchmark tasks."""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

MATERIALS_DOMAIN = "physical_sciences"
MATERIALS_DATA_ROOT = f"{LinuxTaskConfig.REMOTE_ROOT_DIR}/{MATERIALS_DOMAIN}"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class MaterialsEvalSpec:
    """Required files and physical scorer for a materials-science task."""

    task_name: str
    numeric_files: tuple[str, ...]
    png_files: tuple[str, ...]

    @property
    def required_files(self) -> tuple[str, ...]:
        return self.numeric_files + self.png_files


SILICON_GW_BANDGAP_SPEC = MaterialsEvalSpec(
    task_name="silicon_gw_bandgap",
    numeric_files=("bandstructure.dat", "eqp.dat"),
    png_files=("bandstructure_inteqp.png",),
)

SILICON_BSE_ABSORPTION_SPEC = MaterialsEvalSpec(
    task_name="silicon_bse_absorption",
    numeric_files=(
        "absorption_eh.dat",
        "absorption_noeh.dat",
        "bandstructure.dat",
        "eigenvalues.dat",
        "eigenvalues_noeh.dat",
        "eqp.dat",
        "eqp_q.dat",
    ),
    png_files=("absorption.png", "bandstructure_inteqp.png"),
)

MOSE2_BSE_ABSORPTION_SOC_SPEC = MaterialsEvalSpec(
    task_name="mose2_bse_absorption_soc",
    numeric_files=("absorption_eh.dat", "MoSe2_bands.dat.gnu"),
    png_files=("MoSe2_bands.png", "exciton_absorption_spectra_avg.png"),
)


async def run_command(
    session: Any,
    command: str,
    *,
    timeout: float | None = None,
    check: bool = False,
) -> dict[str, Any]:
    """Run a command against a DesktopSession, tolerating differing signatures."""

    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


def _decode_text(data: bytes, filename: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{filename}: failed to decode as UTF-8 text") from exc


def _parse_numeric_table(data: bytes, filename: str) -> list[list[float]]:
    text = _decode_text(data, filename)
    rows: list[list[float]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            row = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"{filename}: line {line_no} is not purely numeric") from exc
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"{filename}: line {line_no} contains a non-finite value")
        rows.append(row)
    if not rows:
        raise ValueError(f"{filename}: no numeric data rows found")
    return rows


def _parse_band_blocks(data: bytes, filename: str) -> list[list[tuple[float, float]]]:
    text = _decode_text(data, filename)
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"{filename}: line {line_no} needs at least two columns")
        try:
            point = (float(parts[0]), float(parts[1]))
        except ValueError as exc:
            raise ValueError(f"{filename}: line {line_no} is not numeric") from exc
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"{filename}: line {line_no} contains a non-finite value")
        current.append(point)
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"{filename}: no band blocks found")
    return blocks


def _interval_score(value: float, low: float, high: float, shoulder: float) -> float:
    if low <= value <= high:
        return 1.0
    distance = low - value if value < low else value - high
    return max(0.0, 1.0 - distance / shoulder)


def _weighted_score(parts: dict[str, tuple[float, float]]) -> tuple[float, dict[str, Any]]:
    total = sum(weight * max(0.0, min(1.0, value)) for value, weight in parts.values())
    details = {
        name: {"value": round(value, 6), "weight": weight}
        for name, (value, weight) in parts.items()
    }
    return round(max(0.0, min(1.0, total)), 6), details


def _silicon_band_metrics(rows: list[list[float]], filename: str) -> dict[str, Any]:
    parsed: list[tuple[int, tuple[float, float, float], float, float]] = []
    for row_idx, row in enumerate(rows, start=1):
        if len(row) < 7:
            raise ValueError(f"{filename}: row {row_idx} needs at least seven columns")
        band = int(round(row[1]))
        parsed.append((band, (row[2], row[3], row[4]), row[5], row[6]))

    valence = [row for row in parsed if row[0] <= 4]
    conduction = [row for row in parsed if row[0] >= 5]
    if not valence or not conduction:
        raise ValueError(f"{filename}: could not identify silicon valence/conduction bands")

    mf_vbm = max(valence, key=lambda row: row[2])
    mf_cbm = min(conduction, key=lambda row: row[2])
    qp_vbm = max(valence, key=lambda row: row[3])
    qp_cbm = min(conduction, key=lambda row: row[3])

    def gamma_distance(kpoint: tuple[float, float, float]) -> float:
        return math.sqrt(sum(value * value for value in kpoint))

    def delta_line_distance(kpoint: tuple[float, float, float]) -> float:
        components = sorted(abs(value) for value in kpoint)
        return math.hypot(components[0], components[1])

    return {
        "dft_gap_ev": mf_cbm[2] - mf_vbm[2],
        "gw_gap_ev": qp_cbm[3] - qp_vbm[3],
        "dft_vbm_gamma_distance": gamma_distance(mf_vbm[1]),
        "gw_vbm_gamma_distance": gamma_distance(qp_vbm[1]),
        "dft_cbm_delta_distance": delta_line_distance(mf_cbm[1]),
        "gw_cbm_delta_distance": delta_line_distance(qp_cbm[1]),
    }


def _silicon_topology_score(metrics: dict[str, Any]) -> float:
    distances = (
        metrics["dft_vbm_gamma_distance"],
        metrics["gw_vbm_gamma_distance"],
        metrics["dft_cbm_delta_distance"],
        metrics["gw_cbm_delta_distance"],
    )
    return sum(_interval_score(value, 0.0, 0.08, 0.16) for value in distances) / 4.0


def _parse_eqp_corrections(rows: list[list[float]], filename: str) -> list[float]:
    corrections: list[float] = []
    index = 0
    while index < len(rows):
        header = rows[index]
        if len(header) != 4:
            raise ValueError(f"{filename}: malformed block header at numeric row {index + 1}")
        band_count = int(round(header[3]))
        if band_count <= 0 or index + band_count >= len(rows):
            raise ValueError(f"{filename}: invalid band count at numeric row {index + 1}")
        for row in rows[index + 1 : index + 1 + band_count]:
            if len(row) != 4:
                raise ValueError(f"{filename}: malformed quasiparticle row")
            corrections.append(row[3] - row[2])
        index += band_count + 1
    if not corrections:
        raise ValueError(f"{filename}: no quasiparticle corrections found")
    return corrections


def _eqp_plausibility(corrections: list[float]) -> float:
    median_abs = sorted(abs(value) for value in corrections)[len(corrections) // 2]
    bounded_fraction = sum(abs(value) <= 5.0 for value in corrections) / len(corrections)
    magnitude_score = _interval_score(median_abs, 0.05, 2.5, 1.0)
    return 0.5 * magnitude_score + 0.5 * bounded_fraction


def _spectrum(rows: list[list[float]], filename: str) -> list[tuple[float, float]]:
    values: dict[float, float] = {}
    for row_idx, row in enumerate(rows, start=1):
        if len(row) < 2:
            raise ValueError(f"{filename}: row {row_idx} needs at least two columns")
        values[row[0]] = row[1]
    spectrum = sorted(values.items())
    if len(spectrum) < 3:
        raise ValueError(f"{filename}: too few spectral samples")
    return spectrum


def _local_peaks(
    spectrum: list[tuple[float, float]], low: float, high: float
) -> list[tuple[float, float]]:
    window = [(energy, value) for energy, value in spectrum if low <= energy <= high]
    peaks = [
        window[index]
        for index in range(1, len(window) - 1)
        if window[index][1] >= window[index - 1][1]
        and window[index][1] > window[index + 1][1]
    ]
    if peaks:
        return peaks
    return [max(window, key=lambda point: point[1])] if window else []


def _strongest_peak(
    spectrum: list[tuple[float, float]], low: float, high: float
) -> tuple[float, float]:
    peaks = _local_peaks(spectrum, low, high)
    if not peaks:
        raise ValueError(f"spectrum: no samples in {low:.2f}-{high:.2f} eV")
    return max(peaks, key=lambda point: point[1])


def _first_prominent_peak(
    spectrum: list[tuple[float, float]], low: float, high: float
) -> tuple[float, float]:
    peaks = _local_peaks(spectrum, low, high)
    if not peaks:
        raise ValueError(f"spectrum: no samples in {low:.2f}-{high:.2f} eV")
    threshold = 0.20 * max(value for _, value in peaks)
    return next(peak for peak in peaks if peak[1] >= threshold)


def _interpolate(spectrum: list[tuple[float, float]], energy: float) -> float:
    energies = [point[0] for point in spectrum]
    index = bisect.bisect_left(energies, energy)
    if index <= 0:
        return spectrum[0][1]
    if index >= len(spectrum):
        return spectrum[-1][1]
    x0, y0 = spectrum[index - 1]
    x1, y1 = spectrum[index]
    if x1 == x0:
        return y1
    fraction = (energy - x0) / (x1 - x0)
    return y0 + fraction * (y1 - y0)


def _spectral_similarity(
    agent: list[tuple[float, float]],
    reference: list[tuple[float, float]],
    *,
    low: float,
    high: float,
) -> float:
    step = 0.02
    count = int(round((high - low) / step)) + 1
    energies = [low + index * step for index in range(count)]
    agent_values = [max(0.0, _interpolate(agent, energy)) for energy in energies]
    ref_values = [max(0.0, _interpolate(reference, energy)) for energy in energies]
    agent_norm = math.sqrt(sum(value * value for value in agent_values))
    ref_norm = math.sqrt(sum(value * value for value in ref_values))
    if agent_norm == 0.0 or ref_norm == 0.0:
        return 0.0
    cosine = sum(a * b for a, b in zip(agent_values, ref_values)) / (
        agent_norm * ref_norm
    )
    return max(0.0, min(1.0, (cosine - 0.40) / 0.60))


def _first_bright_exciton(rows: list[list[float]], filename: str) -> float:
    candidates = [row for row in rows if len(row) >= 2 and 2.0 <= row[0] <= 5.0]
    if not candidates:
        raise ValueError(f"{filename}: no exciton rows in the expected energy window")
    max_strength = max(row[1] for row in candidates)
    threshold = max(1e-10, 0.005 * max_strength)
    bright = [row[0] for row in candidates if row[1] >= threshold]
    if not bright:
        raise ValueError(f"{filename}: no optically bright exciton found")
    return min(bright)


def _score_silicon_gw(agent_tables: dict[str, list[list[float]]]) -> dict[str, Any]:
    metrics = _silicon_band_metrics(agent_tables["bandstructure.dat"], "bandstructure.dat")
    corrections = _parse_eqp_corrections(agent_tables["eqp.dat"], "eqp.dat")
    parts = {
        "dft_gap": (_interval_score(metrics["dft_gap_ev"], 0.50, 0.70, 0.20), 0.25),
        "gw_gap": (_interval_score(metrics["gw_gap_ev"], 1.00, 1.30, 0.25), 0.35),
        "band_edge_topology": (_silicon_topology_score(metrics), 0.20),
        "gap_opening": (
            _interval_score(metrics["gw_gap_ev"] - metrics["dft_gap_ev"], 0.35, 1.00, 0.35),
            0.10,
        ),
        "eqp_plausibility": (_eqp_plausibility(corrections), 0.10),
    }
    score, component_scores = _weighted_score(parts)
    return {"score": score, "metrics": metrics, "components": component_scores}


def _score_silicon_bse(
    agent_tables: dict[str, list[list[float]]],
    reference_tables: dict[str, list[list[float]]],
) -> dict[str, Any]:
    metrics = _silicon_band_metrics(agent_tables["bandstructure.dat"], "bandstructure.dat")
    bright = _first_bright_exciton(agent_tables["eigenvalues.dat"], "eigenvalues.dat")
    eh = _spectrum(agent_tables["absorption_eh.dat"], "absorption_eh.dat")
    noeh = _spectrum(agent_tables["absorption_noeh.dat"], "absorption_noeh.dat")
    ref_eh = _spectrum(reference_tables["absorption_eh.dat"], "reference/absorption_eh.dat")
    e1 = _strongest_peak(eh, 3.00, 3.85)
    e2 = _strongest_peak(eh, 3.85, 4.70)
    noeh_e1 = _strongest_peak(noeh, 3.00, 3.95)
    redshift = noeh_e1[0] - e1[0]
    enhancement = e1[1] / max(noeh_e1[1], 1e-12)
    eh_effect = 0.5 * _interval_score(redshift, 0.03, 0.60, 0.20) + 0.5 * min(
        1.0, max(0.0, (enhancement - 0.80) / 0.40)
    )
    metrics.update(
        {
            "first_bright_exciton_ev": bright,
            "e1_peak_ev": e1[0],
            "e2_peak_ev": e2[0],
            "e1_redshift_ev": redshift,
            "e1_enhancement_ratio": enhancement,
        }
    )
    parts = {
        "dft_gap": (_interval_score(metrics["dft_gap_ev"], 0.50, 0.70, 0.20), 0.10),
        "gw_gap": (_interval_score(metrics["gw_gap_ev"], 1.00, 1.30, 0.25), 0.15),
        "band_edge_topology": (_silicon_topology_score(metrics), 0.10),
        "gap_opening": (
            _interval_score(metrics["gw_gap_ev"] - metrics["dft_gap_ev"], 0.35, 1.00, 0.35),
            0.05,
        ),
        "first_bright_exciton": (_interval_score(bright, 3.20, 3.35, 0.25), 0.15),
        "e1_peak": (_interval_score(e1[0], 3.30, 3.60, 0.30), 0.15),
        "e2_peak": (_interval_score(e2[0], 4.00, 4.40, 0.40), 0.05),
        "electron_hole_effect": (eh_effect, 0.10),
        "spectrum_shape": (
            _spectral_similarity(eh, ref_eh, low=2.50, high=6.00),
            0.15,
        ),
    }
    score, component_scores = _weighted_score(parts)
    return {"score": score, "metrics": metrics, "components": component_scores}


def _mose2_band_metrics(blocks: list[list[tuple[float, float]]]) -> dict[str, Any]:
    # Fully relativistic pseudopotentials provide 46 electrons. In the spinor
    # bands.x output this makes band 46 (zero-based block 45) the VBM.
    if len(blocks) <= 47:
        raise ValueError("MoSe2_bands.dat.gnu: fewer than 48 spinor band blocks")
    lengths = {len(blocks[index]) for index in (44, 45, 46, 47)}
    if len(lengths) != 1:
        raise ValueError("MoSe2_bands.dat.gnu: inconsistent band-path lengths")

    valence_lower, valence = blocks[44], blocks[45]
    conduction, conduction_upper = blocks[46], blocks[47]
    direct_gaps = [
        (conduction[index][1] - valence[index][1], index)
        for index in range(len(valence))
    ]
    direct_gap, gap_index = min(direct_gaps)
    vbm_index = max(range(len(valence)), key=lambda index: valence[index][1])
    cbm_index = min(range(len(conduction)), key=lambda index: conduction[index][1])
    valence_split = valence[gap_index][1] - valence_lower[gap_index][1]
    conduction_split = conduction_upper[gap_index][1] - conduction[gap_index][1]
    path_steps = max(1, len(valence) - 1)
    return {
        "dft_direct_gap_ev": direct_gap,
        "band_edge_path_separation": abs(vbm_index - cbm_index) / path_steps,
        "valence_soc_split_ev": abs(valence_split),
        "conduction_soc_split_ev": abs(conduction_split),
        "gap_path_coordinate": valence[gap_index][0],
    }


def _score_mose2(
    agent_payloads: dict[str, bytes],
    agent_tables: dict[str, list[list[float]]],
    reference_tables: dict[str, list[list[float]]],
) -> dict[str, Any]:
    blocks = _parse_band_blocks(agent_payloads["MoSe2_bands.dat.gnu"], "MoSe2_bands.dat.gnu")
    metrics = _mose2_band_metrics(blocks)
    absorption = _spectrum(agent_tables["absorption_eh.dat"], "absorption_eh.dat")
    ref_absorption = _spectrum(reference_tables["absorption_eh.dat"], "reference/absorption_eh.dat")
    a_peak = _first_prominent_peak(absorption, 1.35, 1.90)
    later_peaks = [
        peak
        for peak in _local_peaks(absorption, a_peak[0] + 0.10, 2.25)
        if peak[1] >= 0.10 * a_peak[1]
    ]
    b_peak = later_peaks[0] if later_peaks else _strongest_peak(absorption, 1.75, 2.25)
    splitting = b_peak[0] - a_peak[0]
    direct_score = _interval_score(metrics["band_edge_path_separation"], 0.0, 0.02, 0.08)
    soc_score = 0.75 * _interval_score(
        metrics["valence_soc_split_ev"], 0.14, 0.24, 0.12
    ) + 0.25 * _interval_score(metrics["conduction_soc_split_ev"], 0.0, 0.08, 0.10)
    metrics.update(
        {
            "a_exciton_peak_ev": a_peak[0],
            "b_exciton_peak_ev": b_peak[0],
            "b_minus_a_ev": splitting,
        }
    )
    parts = {
        "dft_direct_gap": (
            _interval_score(metrics["dft_direct_gap_ev"], 1.28, 1.45, 0.25),
            0.25,
        ),
        "direct_k_band_edges": (direct_score, 0.15),
        "soc_splitting": (soc_score, 0.15),
        "a_exciton_peak": (_interval_score(a_peak[0], 1.50, 1.70, 0.30), 0.20),
        "b_minus_a_splitting": (_interval_score(splitting, 0.15, 0.28, 0.15), 0.10),
        "spectrum_shape": (
            _spectral_similarity(absorption, ref_absorption, low=1.20, high=4.50),
            0.15,
        ),
    }
    score, component_scores = _weighted_score(parts)
    return {"score": score, "metrics": metrics, "components": component_scores}


def score_file_payloads(
    spec: MaterialsEvalSpec,
    *,
    agent_payloads: dict[str, bytes],
    reference_payloads: dict[str, bytes],
) -> dict[str, Any]:
    """Score complete outputs by physical observables rather than exact tables."""

    failures: list[str] = []
    for name in spec.required_files:
        agent_bytes = agent_payloads.get(name)
        ref_bytes = reference_payloads.get(name)
        if agent_bytes is None:
            failures.append(f"{name}: missing from agent output")
        elif not agent_bytes:
            failures.append(f"{name}: agent file is empty")
        if ref_bytes is None:
            raise RuntimeError(f"{name}: missing from evaluator reference payload")
        if not ref_bytes:
            raise RuntimeError(f"{name}: evaluator reference file is empty")
    if failures:
        return {"score": 0.0, "passed": False, "failures": failures}

    for name in spec.png_files:
        if not agent_payloads[name].startswith(PNG_SIGNATURE):
            failures.append(f"{name}: agent file is not a PNG")

    agent_tables: dict[str, list[list[float]]] = {}
    reference_tables: dict[str, list[list[float]]] = {}
    for name in spec.numeric_files:
        try:
            agent_tables[name] = _parse_numeric_table(agent_payloads[name], name)
        except ValueError as exc:
            failures.append(str(exc))
        try:
            reference_tables[name] = _parse_numeric_table(
                reference_payloads[name], f"reference/{name}"
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    if failures:
        return {"score": 0.0, "passed": False, "failures": failures}

    try:
        if spec.task_name == "silicon_gw_bandgap":
            result = _score_silicon_gw(agent_tables)
        elif spec.task_name == "silicon_bse_absorption":
            result = _score_silicon_bse(agent_tables, reference_tables)
        elif spec.task_name == "mose2_bse_absorption_soc":
            result = _score_mose2(agent_payloads, agent_tables, reference_tables)
        else:
            raise RuntimeError(f"unsupported materials scorer: {spec.task_name}")
    except ValueError as exc:
        return {"score": 0.0, "passed": False, "failures": [str(exc)]}

    score = float(result["score"])
    return {
        "score": score,
        "passed": score >= 0.70,
        "failures": [],
        "metrics": result["metrics"],
        "components": result["components"],
    }


async def evaluate_remote_output_dir(
    session: Any,
    *,
    output_dir: str,
    reference_dir: str,
    spec: MaterialsEvalSpec,
) -> dict[str, Any]:
    """Fetch remote files, then score locally."""

    agent_payloads: dict[str, bytes] = {}
    reference_payloads: dict[str, bytes] = {}
    for name in spec.required_files:
        try:
            agent_payloads[name] = await session.read_bytes(f"{output_dir}/{name}")
        except Exception:
            continue
        try:
            reference_payloads[name] = await session.read_bytes(f"{reference_dir}/{name}")
        except Exception as exc:
            raise RuntimeError(
                f"{name}: failed to read evaluator-controlled reference ({exc})"
            ) from exc

    # Fetch references for missing agent files too, so a missing output is
    # reported as an agent failure rather than mistaken for evaluator staging.
    for name in spec.required_files:
        if name in reference_payloads:
            continue
        try:
            reference_payloads[name] = await session.read_bytes(f"{reference_dir}/{name}")
        except Exception as exc:
            raise RuntimeError(
                f"{name}: failed to read evaluator-controlled reference ({exc})"
            ) from exc

    return score_file_payloads(
        spec,
        agent_payloads=agent_payloads,
        reference_payloads=reference_payloads,
    )


def _load_dir_payloads(root: Path, filenames: tuple[str, ...]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name in filenames:
        path = root / name
        if path.is_file():
            payloads[name] = path.read_bytes()
    return payloads


def verify_local_directories(
    spec: MaterialsEvalSpec,
    *,
    agent_dir: Path,
    reference_dir: Path,
) -> dict[str, Any]:
    """Convenience entry point for local fixture tests."""

    return score_file_payloads(
        spec,
        agent_payloads=_load_dir_payloads(agent_dir, spec.required_files),
        reference_payloads=_load_dir_payloads(reference_dir, spec.required_files),
    )


def cli_verify(spec: MaterialsEvalSpec) -> int:
    """CLI wrapper for local verification scripts."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    args = parser.parse_args()

    result = verify_local_directories(
        spec,
        agent_dir=Path(args.agent_dir),
        reference_dir=Path(args.reference_dir),
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1
