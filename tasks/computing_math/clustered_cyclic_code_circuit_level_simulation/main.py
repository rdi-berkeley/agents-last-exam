"""Clustered-cyclic code circuit-level simulation task."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from score_logical_error_rates import (  # noqa: E402
    LogicalErrorRateScoreResult,
    score_logical_error_rates_bytes,
)

logger = logging.getLogger(__name__)

OUTPUT_CSV_NAME = "logical_error_rates_3codes.csv"

VARIANTS = [
    (
        "base",
        "Three clustered-cyclic CSS codes with QUITS/Stim circuit-level memory simulation.",
    )
]


@dataclass
class ClusteredCyclicCodeCircuitLevelSimulationConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "computing_math"
    TASK_NAME: str = "clustered_cyclic_code_circuit_level_simulation"
    VARIANT_NAME: str = "base"
    VARIANT_LABEL: str = VARIANTS[0][1]

    @property
    def tex_file(self) -> str:
        return f"{self.input_dir}/cc_codes_quits_extraction.tex"

    @property
    def simulation_grid_file(self) -> str:
        return f"{self.input_dir}/simulation_grid.csv"

    @property
    def output_requirements_file(self) -> str:
        return f"{self.input_dir}/output_requirements.txt"

    @property
    def runtime_manifest_file(self) -> str:
        return f"{self.input_dir}/runtime_env/pyproject.toml"

    @property
    def output_csv(self) -> str:
        return f"{self.output_dir}/{OUTPUT_CSV_NAME}"

    @property
    def reference_csv(self) -> str:
        return f"{self.reference_dir}/{OUTPUT_CSV_NAME}"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Input Files
- Task notes: `{self.tex_file}`
- Simulation grid: `{self.simulation_grid_file}`
- Output requirements: `{self.output_requirements_file}`
- Python runtime manifest: `{self.runtime_manifest_file}`

## Your Task
Reproduce the circuit-level memory-simulation logical failure-rate table for the three clustered-cyclic CSS codes described in the task notes.

Use the staged notes to extract the code construction, QUITS repeated-syndrome-extraction workflow, direction-aware edge-coloring schedule, Stim detector-error-model workflow, BP+OSD decoder settings, and logical failure-rate formulas.

Run every row in the simulation grid. For each row, report the logical failure count, total logical failure probability, per-round logical failure rate, and per-logical-qubit per-round logical failure rate.

You may install the task-specific open-source Python dependencies from `input/runtime_env/pyproject.toml` or `input/requirements.txt`.

## Output
Write the final CSV here:
- `{self.output_csv}`
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "tex_file": self.tex_file,
                "simulation_grid_file": self.simulation_grid_file,
                "output_requirements_file": self.output_requirements_file,
                "runtime_manifest_file": self.runtime_manifest_file,
                "output_csv": self.output_csv,
                "output_csv_name": OUTPUT_CSV_NAME,
                "reference_csv": self.reference_csv,
            }
        )
        return metadata


def _cfg_for_variant(
    spec: tuple[str, str],
) -> ClusteredCyclicCodeCircuitLevelSimulationConfig:
    return ClusteredCyclicCodeCircuitLevelSimulationConfig(
        VARIANT_NAME=spec[0], VARIANT_LABEL=spec[1]
    )


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_cfg_for_variant(spec).task_description,
            metadata=_cfg_for_variant(spec).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
        for spec in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


def _log_result(tag: str, result: LogicalErrorRateScoreResult) -> None:
    logger.info(
        "[%s] score=%.1f passed=%s rows_checked=%s reasons=%s",
        tag,
        result.score,
        result.passed,
        result.rows_checked,
        result.reasons,
    )


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_csv = meta["output_csv"]
    reference_csv = meta["reference_csv"]

    if not (await session.file_exists(output_csv) or await session.directory_exists(output_csv)):
        logger.error("Missing output CSV: %s", output_csv)
        return [0.0]
    if not (await session.file_exists(reference_csv) or await session.directory_exists(reference_csv)):
        logger.error("Missing reference CSV: %s", reference_csv)
        return [0.0]

    try:
        agent_bytes = await session.read_bytes(output_csv)
        reference_bytes = await session.read_bytes(reference_csv)
    except Exception as exc:
        logger.error("Failed to read output or reference CSV: %s", exc)
        return [0.0]

    try:
        result = score_logical_error_rates_bytes(
            agent_bytes=agent_bytes,
            reference_bytes=reference_bytes,
        )
    except Exception as exc:
        logger.error("Logical error-rate scoring failed: %s", exc)
        return [0.0]

    _log_result(meta["variant_name"], result)
    return [result.score]
