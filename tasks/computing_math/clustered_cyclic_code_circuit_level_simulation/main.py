"""Clustered-cyclic code circuit-level simulation task."""

from __future__ import annotations

import hashlib
import json
import logging
import posixpath
import shlex
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
from circuit_contract import circuit_signature  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

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
    def experiment_file(self) -> str:
        return f"{self.input_dir}/experiment.md"

    @property
    def output_requirements_file(self) -> str:
        return f"{self.input_dir}/output_requirements.txt"

    @property
    def runtime_manifest_file(self) -> str:
        return f"{self.input_dir}/runtime_env/pyproject.toml"

    @property
    def output_csv(self) -> str:
        return f"{self.remote_output_dir}/{OUTPUT_CSV_NAME}"

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
- Experiment conditions: `{self.experiment_file}`
- Output requirements: `{self.output_requirements_file}`
- Python runtime manifest: `{self.runtime_manifest_file}`

## Your Task
Reproduce the circuit-level memory-simulation logical failure-rate table for the three clustered-cyclic CSS codes described in the task notes.

Implement the code construction, circuit scheduling, simulation and decoding
workflow from the notes under the conditions in `experiment.md`. You may use
the supplied low-level BP+OSD decoder. Choose independent sampling seeds and
record the circuits and implementation used to produce your results.

Run every row in the simulation grid using exactly the listed 5,000 independent
shots. For each row, report the logical failure count, total logical failure
probability, per-round logical failure rate, and per-logical-qubit per-round
logical failure rate.

Create the validated task-local runtime with `uv sync --project input/runtime_env --locked`.

## Output
Write the final CSV here:
- `{self.output_csv}`
Also deliver `simulation_manifest.json`, `circuits/` and `source/` under
`{self.remote_output_dir}`, following `output_requirements.txt`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "tex_file": self.tex_file,
                "simulation_grid_file": self.simulation_grid_file,
                "experiment_file": self.experiment_file,
                "output_requirements_file": self.output_requirements_file,
                "runtime_manifest_file": self.runtime_manifest_file,
                "output_csv": self.output_csv,
                "output_csv_name": OUTPUT_CSV_NAME,
                "reference_csv": self.reference_csv,
            }
        )
        return metadata


def _cfg_for_variant(
    spec: tuple[str, str], remote_output_dir: str | None = None
) -> ClusteredCyclicCodeCircuitLevelSimulationConfig:
    kwargs: dict[str, Any] = {"VARIANT_NAME": spec[0], "VARIANT_LABEL": spec[1]}
    if remote_output_dir is not None:
        kwargs["REMOTE_OUTPUT_DIR"] = remote_output_dir
    return ClusteredCyclicCodeCircuitLevelSimulationConfig(**kwargs)


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
    meta = task_cfg.metadata
    input_dir = meta["input_dir"]
    await session.run_command(
        f"rm -f -- {shlex.quote(input_dir + '/run_simulation.py')} "
        f"{shlex.quote(input_dir + '/__pycache__')}/run_simulation.*.pyc"
    )
    for name in (
        "experiment.md",
        "output_requirements.txt",
        "cc_codes_quits_extraction.tex",
        "quits_source.txt",
    ):
        await session.write_file(f"{input_dir}/{name}", (ASSETS_DIR / name).read_text())
    await session.write_file(f"{input_dir}/task_description.txt", task_cfg.description)


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
    if not (
        await session.file_exists(reference_csv) or await session.directory_exists(reference_csv)
    ):
        raise RuntimeError(f"Missing reference CSV: {reference_csv}")

    try:
        agent_bytes = await session.read_bytes(output_csv)
        reference_bytes = await session.read_bytes(reference_csv)
    except Exception as exc:
        raise RuntimeError("Failed to read QEC evaluation files") from exc

    reference_manifest_path = posixpath.join(posixpath.dirname(reference_csv), "stage1_manifest.json")
    if not await session.file_exists(reference_manifest_path):
        raise RuntimeError("Missing QEC decoder reference manifest")
    try:
        reference_manifest = json.loads(await session.read_bytes(reference_manifest_path))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError("Invalid QEC decoder reference manifest") from exc
    if (
        not isinstance(reference_manifest, dict)
        or reference_manifest.get("decoder_representation") != "stim-flattened-error-columns-v1"
        or reference_manifest.get("reference_csv_sha256") != hashlib.sha256(reference_bytes).hexdigest()
    ):
        raise RuntimeError("QEC reference does not match the full joint DEM decoder contract")

    output_dir = meta["remote_output_dir"]
    manifest_path = f"{output_dir}/simulation_manifest.json"
    if not await session.file_exists(manifest_path):
        return [0.0]
    try:
        manifest = json.loads(await session.read_bytes(manifest_path))
        seeds = manifest["sampling_seeds"]
        if not isinstance(seeds, list) or len(seeds) != 60:
            return [0.0]
        if any(type(seed) is not int or not 0 <= seed < 2**64 for seed in seeds):
            return [0.0]
        if len(set(seeds)) != 60:
            return [0.0]
    except (ValueError, KeyError, TypeError):
        return [0.0]
    signatures = json.loads((SCRIPTS_DIR / "circuit_signatures.json").read_text())
    dimensions = {"qubits", "measurements", "detectors", "observables"}
    digest_key = "detector_error_distribution_sha256"
    if not isinstance(signatures, list) or len(signatures) != 60:
        raise RuntimeError("Invalid QEC reference circuit signatures: expected 60 entries")
    for signature in signatures:
        if (
            not isinstance(signature, dict)
            or set(signature) != dimensions | {digest_key}
            or any(
                type(signature[key]) is not int or not 0 < signature[key] <= 200_000
                for key in dimensions
            )
            or not isinstance(signature[digest_key], str)
            or len(signature[digest_key]) != 64
            or any(character not in "0123456789abcdef" for character in signature[digest_key])
        ):
            raise RuntimeError("Invalid QEC reference circuit signature entry")
    for row_index, expected in enumerate(signatures):
        path = f"{output_dir}/circuits/row_{row_index:03}.stim"
        if not await session.file_exists(path):
            logger.info("Missing circuit: %s", path)
            return [0.0]
        circuit_bytes = await session.read_bytes(path)
        if len(circuit_bytes) > 2_000_000:
            return [0.0]
        try:
            actual = circuit_signature(circuit_bytes.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            logger.info("Invalid circuit at grid row %s", row_index)
            return [0.0]
        if actual != expected:
            logger.info("Experiment mismatch at grid row %s", row_index)
            return [0.0]
    if not await session.file_exists(f"{output_dir}/source/simulate.py"):
        return [0.0]
    if not (await session.read_bytes(f"{output_dir}/source/simulate.py")).strip():
        return [0.0]

    try:
        result = score_logical_error_rates_bytes(
            agent_bytes=agent_bytes,
            reference_bytes=reference_bytes,
        )
    except Exception as exc:
        raise RuntimeError("Logical error-rate scoring failed") from exc

    _log_result(meta["variant_name"], result)
    return [result.score]
