"""Stage 2 task implementation for amber_three_stage_mmgbsa_workflow_instance_1."""

from __future__ import annotations

import logging
import os
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.life_sciences.amber_three_stage_mmgbsa_workflow_instance_1.scripts.verify_submission import (
    evaluate_output_bundle,
)
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "life_sciences"
TASK_NAME = "amber_three_stage_mmgbsa_workflow_instance_1"
VARIANT_NAME = "base"
SYSTEM_BASENAME = "GLN_phb2_parl_pgam5_model_0"
REQUIRED_OUTPUT_FILES = (
    "submit_min.sh",
    "submit_prod.sh",
    "submit_mmgbsa.sh",
    "FINAL_RESULTS_MMGBSA.dat",
)
SUPPORTED_AMBERTOOLS_TOOLS = (
    "tleap",
    "cpptraj",
    "MMPBSA.py",
    "ante-MMPBSA.py",
    "antechamber",
    "parmchk2",
)


class AmberThreeStageMmgbsaTaskConfig(LinuxTaskConfig):
    """Linux task config for the Amber three-stage MMGBSA workflow task."""

    def __init__(self, *, REMOTE_OUTPUT_DIR: str | None = None) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_OUTPUT_DIR=REMOTE_OUTPUT_DIR or os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def output_submit_min(self) -> str:
        return f"{self.remote_output_dir}/submit_min.sh"

    @property
    def output_submit_prod(self) -> str:
        return f"{self.remote_output_dir}/submit_prod.sh"

    @property
    def output_submit_mmgbsa(self) -> str:
        return f"{self.remote_output_dir}/submit_mmgbsa.sh"

    @property
    def output_results(self) -> str:
        return f"{self.remote_output_dir}/FINAL_RESULTS_MMGBSA.dat"

    @property
    def hidden_reference_results(self) -> str:
        return f"{self.reference_dir}/FINAL_RESULTS_MMGBSA.dat"

    @property
    def ambertools_wrapper(self) -> str:
        return f"{self.software_dir}/run_ambertools.sh"

    @property
    def task_description(self) -> str:
        return f"""\
You are preparing a Linux Amber workflow definition for a protein-only three-chain complex.

Task directory:
- `{self.task_dir}`

Input directory:
- `{self.input_dir}`

Software:
- Benchmark-owned AmberTools 23 CLI wrapper: `{self.ambertools_wrapper}`
- Invoke staged AmberTools commands as: `{self.ambertools_wrapper} <tool> ...`
- Supported tool names: `{", ".join(SUPPORTED_AMBERTOOLS_TOOLS)}`
- `pmemd.cuda`, CUDA, and SLURM remain reference-environment details for the authored scripts; they are not benchmark-provided on this VM.

Your task:
1. Inspect `complex_structure.pdb` and the two markdown specs under `input/`.
2. Treat chain `A` as the receptor and chains `B` plus `C` together as the ligand.
3. Create exactly these four files under `{self.remote_output_dir}`:
   - `submit_min.sh`
   - `submit_prod.sh`
   - `submit_mmgbsa.sh`
   - `FINAL_RESULTS_MMGBSA.dat`

Requirements:
- Use basename `{SYSTEM_BASENAME}` consistently in the workflow.
- `submit_min.sh` should define topology/build plus minimization and short equilibration.
- `submit_prod.sh` should define one implicit-solvent production MD stage with `pmemd.cuda`.
- `submit_mmgbsa.sh` should run `MMPBSA.py` with complex / receptor-A / ligand-BC roles wired correctly.
- The graded MMGBSA result must use the provided `input/prod.mdcrd` as trajectory input.
- Do not write extra deliverable files.

"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "system_basename": SYSTEM_BASENAME,
                "required_output_files": list(REQUIRED_OUTPUT_FILES),
                "output_submit_min": self.output_submit_min,
                "output_submit_prod": self.output_submit_prod,
                "output_submit_mmgbsa": self.output_submit_mmgbsa,
                "output_results": self.output_results,
                "hidden_reference_results": self.hidden_reference_results,
                "ambertools_wrapper": self.ambertools_wrapper,
            }
        )
        return metadata


config = AmberThreeStageMmgbsaTaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


async def _run_command(
    session: cb.DesktopSession, command: str, *, check: bool = False
) -> dict[str, Any]:
    try:
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command)


async def _read_text_if_exists(session: cb.DesktopSession, path: str) -> str | None:
    try:
        payload = await session.read_bytes(path)
    except Exception:
        return None
    if not payload:
        return None
    return payload.decode("utf-8", errors="replace")


async def _list_output_files(session: cb.DesktopSession, output_dir: str) -> list[str]:
    command = f'find "{output_dir}" -maxdepth 1 -type f -printf "%f\\n" | sort'
    result = await _run_command(session, command, check=False)
    if result.get("return_code", 0) != 0:
        return []
    return [line.strip() for line in result.get("stdout", "").splitlines() if line.strip()]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_dir = meta["remote_output_dir"]
    files = await _list_output_files(session, output_dir)

    bundle: dict[str, str] = {}
    for name in REQUIRED_OUTPUT_FILES:
        text = await _read_text_if_exists(session, f"{output_dir}/{name}")
        if text is not None:
            bundle[name] = text

    hidden_reference_text = await _read_text_if_exists(session, meta["hidden_reference_results"])
    result = evaluate_output_bundle(
        bundle,
        present_files=files,
        hidden_reference_text=hidden_reference_text,
    )
    logger.info(
        "amber_three_stage_mmgbsa_workflow_instance_1 score=%s passed=%s reasons=%s",
        result["score"],
        result["passed"],
        result["reasons"],
    )
    return [float(result["score"])]
