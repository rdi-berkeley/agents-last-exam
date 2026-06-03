"""AgentHLE task: exact_diag_heisenberg_j1j2."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
VERIFY_SCRIPT_PATH = SCRIPTS_DIR / "verify_outputs.py"


def _load_verify_module():
    spec = importlib.util.spec_from_file_location(
        "exact_diag_heisenberg_j1j2_verify_outputs",
        VERIFY_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load verifier module from {VERIFY_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load_verify_module()
ScoreResult = VERIFY_MODULE.ScoreResult
score_submission = VERIFY_MODULE.score_submission

logger = logging.getLogger(__name__)

TASK_NAME = "exact_diag_heisenberg_j1j2"
VARIANT_NAME = "base"
CANONICAL_GCS_ROOT = "gs://ale-data-all/physical_sciences/exact_diag_heisenberg_j1j2/base/"


class ExactDiagHeisenbergConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "physical_sciences"
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def problem_spec(self) -> str:
        return f"{self.input_dir}/problem_spec.md"

    @property
    def output_ground_state(self) -> str:
        return f"{self.remote_output_dir}/ground_state.npz"

    @property
    def output_correlations(self) -> str:
        return f"{self.remote_output_dir}/correlations.npz"

    @property
    def output_dynamical_sf(self) -> str:
        return f"{self.remote_output_dir}/dynamical_sf.npz"

    @property
    def output_results(self) -> str:
        return f"{self.remote_output_dir}/results.json"

    @property
    def reference_outputs_dir(self) -> str:
        return f"{self.reference_dir}/reference_outputs"

    @property
    def verification_metadata_file(self) -> str:
        return f"{self.reference_dir}/metadata/verification_targets.json"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Your Task
Implement an exact diagonalization workflow for the spin-1/2 `J1-J2` Heisenberg antiferromagnet on a `4x4` periodic square lattice in the conserved `S_z = 0` sector.

## Visible Input
- Problem specification: `{self.problem_spec}`

## Runtime
- Use the benchmark-provided `software/python` entry point at `{self.task_dir}/software/python`.
- That entry point resolves to the VM's preinstalled `/usr/bin/python3.10` with NumPy and SciPy available.
- Do not install extra packages or use external quantum frameworks.

## What You Must Produce
Write the required files under `{self.remote_output_dir}`:
- `ground_state.npz`
- `correlations.npz`
- `dynamical_sf.npz`
- `results.json`

## Important Constraints
- Follow the file schema and scientific checks in `{self.problem_spec}`.
- Do not modify files under `{self.input_dir}`.
- Do not write outside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "problem_spec": self.problem_spec,
                "output_ground_state": self.output_ground_state,
                "output_correlations": self.output_correlations,
                "output_dynamical_sf": self.output_dynamical_sf,
                "output_results": self.output_results,
                "reference_outputs_dir": self.reference_outputs_dir,
                "verification_metadata_file": self.verification_metadata_file,
                "canonical_gcs_root": CANONICAL_GCS_ROOT,
            }
        )
        return metadata


config = ExactDiagHeisenbergConfig(
    DOMAIN_NAME="physical_sciences",
    TASK_NAME=TASK_NAME,
    VARIANT_NAME=VARIANT_NAME,
)


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _write_remote_file_to_local(
    session: cb.DesktopSession, remote_path: str, local_path: Path
) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(await session.read_bytes(remote_path))


def _log_score(result: ScoreResult) -> None:
    logger.info(
        "score=%.6f tier1=%s tier2=%s tier3=%s",
        result.score,
        result.tier1_passed,
        result.tier2_passed,
        result.tier3_passed,
    )
    logger.info("details=%s", json.dumps(result.to_dict(), ensure_ascii=True))


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_files = {
        "ground_state.npz": meta["output_ground_state"],
        "correlations.npz": meta["output_correlations"],
        "dynamical_sf.npz": meta["output_dynamical_sf"],
        "results.json": meta["output_results"],
    }

    missing = [name for name, path in output_files.items() if not (await session.file_exists(path) or await session.directory_exists(path))]
    if missing:
        logger.error("Missing output files: %s", ", ".join(missing))
        return [0.0]

    reference_files = {
        "ground_state.npz": f"{meta['reference_outputs_dir']}/ground_state.npz",
        "correlations.npz": f"{meta['reference_outputs_dir']}/correlations.npz",
        "dynamical_sf.npz": f"{meta['reference_outputs_dir']}/dynamical_sf.npz",
        "results.json": f"{meta['reference_outputs_dir']}/results.json",
    }
    reference_missing = [
        name for name, path in reference_files.items() if not (await session.file_exists(path) or await session.directory_exists(path))
    ]
    if reference_missing:
        logger.error("Missing reference files during evaluation: %s", ", ".join(reference_missing))
        return [0.0]

    with tempfile.TemporaryDirectory(prefix=f"{TASK_NAME}_eval_") as tmpdir:
        tmp_root = Path(tmpdir)
        local_output_dir = tmp_root / "output"
        local_reference_dir = tmp_root / "reference"
        local_metadata = tmp_root / "verification_targets.json"

        try:
            for name, remote_path in output_files.items():
                await _write_remote_file_to_local(session, remote_path, local_output_dir / name)
            for name, remote_path in reference_files.items():
                await _write_remote_file_to_local(session, remote_path, local_reference_dir / name)
            if (await session.file_exists(meta["verification_metadata_file"]) or await session.directory_exists(meta["verification_metadata_file"])):
                await _write_remote_file_to_local(
                    session,
                    meta["verification_metadata_file"],
                    local_metadata,
                )
        except Exception as exc:
            logger.error("Failed to pull evaluation artifacts from VM: %s", exc)
            return [0.0]

        try:
            result = score_submission(
                output_dir=local_output_dir,
                reference_dir=local_reference_dir,
                metadata_path=local_metadata if local_metadata.exists() else None,
            )
        except Exception as exc:
            logger.error("Local verifier crashed: %s", exc)
            return [0.0]

    _log_score(result)
    return [float(result.score)]
