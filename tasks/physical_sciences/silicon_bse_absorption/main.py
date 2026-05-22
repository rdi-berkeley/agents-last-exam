"""Stage 2 implementation for the silicon GW-BSE absorption task."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.physical_sciences._shared.materials_science._common import (
    SILICON_BSE_ABSORPTION_SPEC,
    evaluate_remote_output_dir,
)

logger = logging.getLogger(__name__)

TASK_NAME = "silicon_bse_absorption"
VARIANT_NAME = "base"


@dataclass
class SiliconBSEAbsorptionConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "physical_sciences"

    TASK_NAME: str = "silicon_bse_absorption"
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    @property
    def software_bin_dir(self) -> str:
        return f"{self.software_dir}/bin"

    @property
    def task_description(self) -> str:
        return f"""\
You are a computational materials scientist working on a Linux VM.

## Your Task
Compute the GW-BSE absorption spectrum of bulk silicon from the staged structure and pseudopotential.

## Input Files
- Silicon structure: `{self.input_dir}/silicon.vasp`
- Silicon pseudopotential: `{self.input_dir}/Si.UPF`

## Software
Use the task-local command-line tools staged here:
- `{self.software_bin_dir}/pw.x`
- `{self.software_bin_dir}/mpirun`
- `{self.software_bin_dir}/epsilon.cplx.x`
- `{self.software_bin_dir}/sigma.cplx.x`
- `{self.software_bin_dir}/kernel.cplx.x`
- `{self.software_bin_dir}/absorption.cplx.x`
- `{self.software_bin_dir}/inteqp.cplx.x`

## What You Must Do
1. Create the missing Quantum ESPRESSO and BerkeleyGW input decks from scratch.
2. Run the silicon QE mean-field workflow.
3. Run the BerkeleyGW GW+BSE workflow needed for both quasiparticle and absorption outputs.
4. Save the required outputs exactly under `{self.remote_output_dir}`.

## Required Output Files
- `{self.remote_output_dir}/bandstructure.dat`
- `{self.remote_output_dir}/eqp.dat`
- `{self.remote_output_dir}/eqp_q.dat`
- `{self.remote_output_dir}/absorption_eh.dat`
- `{self.remote_output_dir}/absorption_noeh.dat`
- `{self.remote_output_dir}/eigenvalues.dat`
- `{self.remote_output_dir}/eigenvalues_noeh.dat`
- `{self.remote_output_dir}/bandstructure_inteqp.png`
- `{self.remote_output_dir}/absorption.png`

Do not write outputs outside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "reference_dir": self.reference_dir,
                "software_dir": self.software_dir,
                "software_bin_dir": self.software_bin_dir,
                "remote_output_dir": self.remote_output_dir,
            }
        )
        return metadata


CONFIG = SiliconBSEAbsorptionConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=CONFIG.task_description,
            metadata=CONFIG.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": CONFIG.OS_TYPE}},
        )
    ]


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    result = await evaluate_remote_output_dir(
        session,
        output_dir=task_cfg.metadata["remote_output_dir"],
        reference_dir=task_cfg.metadata["reference_dir"],
        spec=SILICON_BSE_ABSORPTION_SPEC,
    )
    if result["failures"]:
        logger.warning("silicon_bse_absorption evaluation failures: %s", result["failures"])
    return [float(result["score"])]
