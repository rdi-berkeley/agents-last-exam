"""Parser-based Amber minimization workflow benchmark."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.life_sciences.amber_minimization_script_prep_instance_1.scripts.verify_submission import (
    evaluate_output_bundle,
)
from tasks.linux_runtime import DATA_ROOT, LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "life_sciences"
TASK_NAME = "amber_minimization_script_prep_instance_1"
VARIANT_NAME = "base"
SYSTEM_BASENAME = "GLN_phb2_lc3_aurka_model_0"
EVAL_FILES = ("leap.in", "step2_implicit.mini.mdin", "submit_min.sh")


@dataclass
class AmberMinimizationTaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "life_sciences"
    TASK_NAME: str = "amber_minimization_script_prep_instance_1"
    VARIANT_NAME: str = "base"
    OS_TYPE: str = "linux"

    @property
    def task_description(self) -> str:
        return f"""You are preparing a Linux Amber workflow definition from a cleaned protein-complex structure.

Task directory:
- `{self.task_dir}`

Input directory:
- `{self.input_dir}`

Available environment:
- Linux terminal access
- Python 3 on the VM
- `uv` is available for evaluator-side tooling, but you should not rely on hidden benchmark scripts

Your task:
1. Inspect `complex_structure.pdb`
2. Create exactly these three files under `{self.remote_output_dir}`:
   - `leap.in`
   - `step2_implicit.mini.mdin`
   - `submit_min.sh`

Requirements:
- Build the system from `complex_structure.pdb`
- Use basename `{SYSTEM_BASENAME}` consistently
- Define a realistic implicit-solvent Amber minimization workflow
- Write a SLURM submission script for one GPU using Amber 22 and CUDA 11.6.2 module loads
- Do not write extra files

"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "data_task_dir": self.data_task_dir,
                "input_dir": self.input_dir,
                "reference_dir": self.reference_dir,
                "software_dir": self.software_dir,
                "remote_output_dir": self.remote_output_dir,
                "system_basename": SYSTEM_BASENAME,
            }
        )
        return metadata


config = AmberMinimizationTaskConfig(
    DOMAIN_NAME=DOMAIN_NAME,
    TASK_NAME=TASK_NAME,
    VARIANT_NAME=VARIANT_NAME,
    OS_TYPE="linux",
)


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


async def _run_command(session: cb.DesktopSession, command: str, *, check: bool = False) -> dict:
    try:
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command)


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


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


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    output_dir = meta["remote_output_dir"]
    files = await _list_output_files(session, output_dir)

    bundle: dict[str, str] = {}
    for name in EVAL_FILES:
        text = await _read_text_if_exists(session, f"{output_dir}/{name}")
        if text is not None:
            bundle[name] = text

    result = evaluate_output_bundle(bundle, present_files=files)
    logger.info(
        "amber_minimization_script_prep_instance_1 score=%s passed=%s reasons=%s",
        result["score"],
        result["passed"],
        result["reasons"],
    )
    return [float(result["score"])]
