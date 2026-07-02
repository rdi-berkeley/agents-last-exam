"""Stage 2 implementation for the silicon GW band-gap task."""

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
    SILICON_GW_BANDGAP_SPEC,
    evaluate_remote_output_dir,
)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

TASK_NAME = "computational_materials_science"
VARIANT_NAME = "base"


@dataclass
class ComputationalMaterialsScienceConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "physical_sciences"

    TASK_NAME: str = "computational_materials_science"
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    @property
    def software_launcher(self) -> str:
        return f"{self.software_dir}/launch_qe_bgw.sh"

    @property
    def task_description(self) -> str:
        return f"""\
You are a computational materials scientist working on a Linux VM.

## Your Task
Compute the indirect GW quasiparticle band gap of bulk silicon from the staged structure and pseudopotential.

## Input Files
- Silicon structure: `{self.input_dir}/silicon/silicon.vasp`
- Silicon pseudopotential: `{self.input_dir}/silicon/Si.UPF`

## Software
Use the task-local launcher shortcut staged here:
- `{self.software_launcher}`

Run Quantum ESPRESSO and BerkeleyGW commands through the launcher, for example:
- `bash {self.software_launcher} pw.x ...`
- `bash {self.software_launcher} epsilon.cplx.x ...`
- `bash {self.software_launcher} bash` to open a shell with the QE/BerkeleyGW PATH set

You may also use Python, shell utilities, and editors available on the VM.

## Benchmark Workflow Settings
- Run QE SCF and NSCF/bands calculations, then `pw2bgw`, BerkeleyGW `epsilon`, `sigma`, and `inteqp`.
- Use approximately `5x5x5` wavefunction k-point sampling, a `10 Ry` dielectric cutoff, and about `39` GW summation bands.
- Set `OMP_NUM_THREADS=1` and use no more than four MPI ranks, for example `mpirun -np 4`.
- Scientific sanity targets: DFT indirect gap near `0.6 eV`, GW indirect gap near `1.1 eV`, VBM at Gamma, and CBM near X along Gamma-X.

## What You Must Do
1. Create the missing Quantum ESPRESSO and BerkeleyGW input decks from scratch.
2. Run the complete silicon GW workflow using the staged structure and pseudopotential.
3. Save the required outputs exactly under `{self.remote_output_dir}/silicon`.

## Required Output Files
- `{self.remote_output_dir}/silicon/bandstructure.dat`
- `{self.remote_output_dir}/silicon/eqp.dat`
- `{self.remote_output_dir}/silicon/bandstructure_inteqp.png`

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
                "software_launcher": self.software_launcher,
                "remote_output_dir": self.remote_output_dir,
                "subcases": ["silicon"],
            }
        )
        return metadata


CONFIG = ComputationalMaterialsScienceConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=CONFIG.task_description,
            metadata=CONFIG.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": CONFIG.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    result = await evaluate_remote_output_dir(
        session,
        output_dir=f"{task_cfg.metadata['remote_output_dir']}/silicon",
        reference_dir=f"{task_cfg.metadata['reference_dir']}/silicon",
        spec=SILICON_GW_BANDGAP_SPEC,
    )
    if result["failures"]:
        logger.warning(
            "computational_materials_science evaluation failures: %s",
            result["failures"],
        )
    return [float(result["score"])]
