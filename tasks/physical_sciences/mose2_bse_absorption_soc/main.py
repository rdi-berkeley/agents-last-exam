"""Stage 2 implementation for the MoSe2 SOC BSE absorption task."""

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
    MOSE2_BSE_ABSORPTION_SOC_SPEC,
    evaluate_remote_output_dir,
)

logger = logging.getLogger(__name__)

TASK_NAME = "mose2_bse_absorption_soc"
VARIANT_NAME = "base"


@dataclass
class MoSe2BSEAbsorptionSOCConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "physical_sciences"

    TASK_NAME: str = "mose2_bse_absorption_soc"
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
Compute the SOC-enabled GW-BSE optical response of monolayer MoSe2 from the staged structure and pseudopotentials.

## Input Files
- MoSe2 structure: `{self.input_dir}/MoSe2.vasp`
- Mo pseudopotential: `{self.input_dir}/Mo.soc.upf`
- Se pseudopotential: `{self.input_dir}/Se.soc.upf`

## Software
Use the task-local command-line tools staged here:
- `{self.software_bin_dir}/pw.x`
- `{self.software_bin_dir}/mpirun`
- `{self.software_bin_dir}/epsilon.cplx.x`
- `{self.software_bin_dir}/sigma.cplx.x`
- `{self.software_bin_dir}/kernel.cplx.x`
- `{self.software_bin_dir}/absorption.cplx.x`

## What You Must Do
1. Create the missing Quantum ESPRESSO and BerkeleyGW input decks from scratch.
2. Run the monolayer MoSe2 QE workflow with explicit SOC / noncollinear settings.
3. Run the BerkeleyGW workflow needed for the required BSE absorption output, including the 2D truncation treatment.
4. Save the required outputs exactly under `{self.remote_output_dir}`.

## Required Output Files
- `{self.remote_output_dir}/MoSe2_bands.dat.gnu`
- `{self.remote_output_dir}/MoSe2_bands.png`
- `{self.remote_output_dir}/absorption_eh.dat`
- `{self.remote_output_dir}/exciton_absorption_spectra_avg.png`

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


CONFIG = MoSe2BSEAbsorptionSOCConfig()


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
        spec=MOSE2_BSE_ABSORPTION_SOC_SPEC,
    )
    if result["failures"]:
        logger.warning("mose2_bse_absorption_soc evaluation failures: %s", result["failures"])
    return [float(result["score"])]
