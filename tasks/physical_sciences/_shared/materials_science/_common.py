"""Shared helpers for Linux-based materials-science benchmark tasks."""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class MaterialsEvalSpec:
    """Binary evaluator spec for a materials-science task."""

    task_name: str
    numeric_files: tuple[str, ...]
    png_files: tuple[str, ...]
    rtol: float = 1e-4
    atol: float = 5e-3

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
        rows.append(row)
    if not rows:
        raise ValueError(f"{filename}: no numeric data rows found")
    return rows


def _compare_numeric_tables(
    agent_rows: list[list[float]],
    reference_rows: list[list[float]],
    *,
    filename: str,
    rtol: float,
    atol: float,
) -> list[str]:
    failures: list[str] = []
    if len(agent_rows) != len(reference_rows):
        failures.append(
            f"{filename}: row count mismatch ({len(agent_rows)} != {len(reference_rows)})"
        )
        return failures

    for row_idx, (agent_row, ref_row) in enumerate(zip(agent_rows, reference_rows), start=1):
        if len(agent_row) != len(ref_row):
            failures.append(
                f"{filename}: column count mismatch on row {row_idx} "
                f"({len(agent_row)} != {len(ref_row)})"
            )
            continue
        for col_idx, (agent_value, ref_value) in enumerate(zip(agent_row, ref_row), start=1):
            if not math.isclose(agent_value, ref_value, rel_tol=rtol, abs_tol=atol):
                failures.append(
                    f"{filename}: value mismatch at row {row_idx}, col {col_idx} "
                    f"({agent_value} != {ref_value})"
                )
                return failures
    return failures


def score_file_payloads(
    spec: MaterialsEvalSpec,
    *,
    agent_payloads: dict[str, bytes],
    reference_payloads: dict[str, bytes],
) -> dict[str, Any]:
    """Binary score: every required file must pass."""

    failures: list[str] = []

    for name in spec.required_files:
        agent_bytes = agent_payloads.get(name)
        ref_bytes = reference_payloads.get(name)
        if agent_bytes is None:
            failures.append(f"{name}: missing from agent output")
            continue
        if ref_bytes is None:
            failures.append(f"{name}: missing from reference payload")
            continue
        if not agent_bytes:
            failures.append(f"{name}: agent file is empty")
            continue
        if not ref_bytes:
            failures.append(f"{name}: reference file is empty")
            continue

    if failures:
        return {"score": 0.0, "passed": False, "failures": failures}

    for name in spec.png_files:
        agent_bytes = agent_payloads[name]
        if not agent_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            failures.append(f"{name}: agent file is not a PNG")

    for name in spec.numeric_files:
        try:
            agent_rows = _parse_numeric_table(agent_payloads[name], name)
            ref_rows = _parse_numeric_table(reference_payloads[name], name)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        failures.extend(
            _compare_numeric_tables(
                agent_rows,
                ref_rows,
                filename=name,
                rtol=spec.rtol,
                atol=spec.atol,
            )
        )

    return {
        "score": 1.0 if not failures else 0.0,
        "passed": not failures,
        "failures": failures,
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
        agent_path = f"{output_dir}/{name}"
        ref_path = f"{reference_dir}/{name}"
        try:
            agent_payloads[name] = await session.read_bytes(agent_path)
        except Exception as exc:
            return {
                "score": 0.0,
                "passed": False,
                "failures": [f"{name}: failed to read agent output ({exc})"],
            }
        try:
            reference_payloads[name] = await session.read_bytes(ref_path)
        except Exception as exc:
            return {
                "score": 0.0,
                "passed": False,
                "failures": [f"{name}: failed to read reference output ({exc})"],
            }

    return score_file_payloads(
        spec,
        agent_payloads=agent_payloads,
        reference_payloads=reference_payloads,
    )


def _load_dir_payloads(root: Path, filenames: tuple[str, ...]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name in filenames:
        payloads[name] = (root / name).read_bytes()
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
