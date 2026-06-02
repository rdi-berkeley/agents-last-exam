"""Stage 2 implementation for the composite computational_materials_science task."""

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
    SILICON_BSE_ABSORPTION_SPEC,
    SILICON_GW_BANDGAP_SPEC,
    evaluate_remote_output_dir,
)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

TASK_NAME = "computational_materials_science"
VARIANT_NAME = "base"
SUBCASE_SPECS = {
    "silicon": SILICON_GW_BANDGAP_SPEC,
    "silicon-BSE": SILICON_BSE_ABSORPTION_SPEC,
    "MoSe2-BSE": MOSE2_BSE_ABSORPTION_SOC_SPEC,
}


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
Produce benchmark outputs for three staged materials-science subcases from the visible structures and pseudopotentials.

## Input Subcases
- silicon GW band gap: `{self.input_dir}/silicon`
- silicon GW-BSE absorption: `{self.input_dir}/silicon-BSE`
- MoSe2 SOC GW-BSE absorption: `{self.input_dir}/MoSe2-BSE`

## Software
Use the task-local launcher shortcut staged here:
- `{self.software_launcher}`

Run Quantum ESPRESSO and BerkeleyGW commands through the launcher, for example:
- `{self.software_launcher} pw.x ...`
- `{self.software_launcher} epsilon.cplx.x ...`
- `{self.software_launcher} bash` to open a shell with the QE/BerkeleyGW PATH set

You may also use Python, shell utilities, and editors available on the VM.

## Benchmark Workflow Settings
- For `silicon`, run QE SCF/NSCF followed by BerkeleyGW `epsilon -> sigma -> inteqp`. Use approximately `5x5x5` wavefunction k-point sampling, `10 Ry` dielectric cutoff, and about `39` GW summation bands. Scientific sanity targets: DFT gap near `0.6 eV`, GW gap near `1.1 eV`, indirect VBM at Gamma and CBM near X.
- For `silicon-BSE`, run QE SCF/NSCF followed by BerkeleyGW `epsilon -> sigma -> kernel -> absorption`, plus `inteqp` for quasiparticle band outputs. Use the same benchmark-scale silicon settings: approximately `5x5x5` wavefunction k-point sampling, `10 Ry` dielectric cutoff, and about `39` GW summation bands. Produce absorption with and without electron-hole interaction; a prominent electron-hole peak should be near `3.4 eV`.
- For `MoSe2-BSE`, run QE SCF/NSCF with explicit SOC and noncollinear settings followed by BerkeleyGW `epsilon -> sigma -> kernel -> absorption`. Include 2D truncation, approximately `16x16x1` k-point sampling, about `2500` GW summation bands, `32x32x1` fine-grid interpolation, and screened Coulomb cutoff near `25 Ry`. Scientific sanity targets: direct K-point gap near `1.33 eV` and sharp excitonic absorption peak near `1.61 eV`.

## What You Must Do
1. Create the missing Quantum ESPRESSO and BerkeleyGW input decks from scratch for each subcase.
2. Run the silicon GW workflow and save its required outputs under `{self.remote_output_dir}/silicon`.
3. Run the silicon GW+BSE workflow and save its required outputs under `{self.remote_output_dir}/silicon-BSE`.
4. Run the SOC-enabled monolayer MoSe2 GW+BSE workflow and save its required outputs under `{self.remote_output_dir}/MoSe2-BSE`.

## Required Output Files

### silicon
- `{self.remote_output_dir}/silicon/bandstructure.dat`
- `{self.remote_output_dir}/silicon/eqp.dat`
- `{self.remote_output_dir}/silicon/bandstructure_inteqp.png`

### silicon-BSE
- `{self.remote_output_dir}/silicon-BSE/bandstructure.dat`
- `{self.remote_output_dir}/silicon-BSE/eqp.dat`
- `{self.remote_output_dir}/silicon-BSE/eqp_q.dat`
- `{self.remote_output_dir}/silicon-BSE/absorption_eh.dat`
- `{self.remote_output_dir}/silicon-BSE/absorption_noeh.dat`
- `{self.remote_output_dir}/silicon-BSE/eigenvalues.dat`
- `{self.remote_output_dir}/silicon-BSE/eigenvalues_noeh.dat`
- `{self.remote_output_dir}/silicon-BSE/bandstructure_inteqp.png`
- `{self.remote_output_dir}/silicon-BSE/absorption.png`

### MoSe2-BSE
- `{self.remote_output_dir}/MoSe2-BSE/MoSe2_bands.dat.gnu`
- `{self.remote_output_dir}/MoSe2-BSE/MoSe2_bands.png`
- `{self.remote_output_dir}/MoSe2-BSE/absorption_eh.dat`
- `{self.remote_output_dir}/MoSe2-BSE/exciton_absorption_spectra_avg.png`

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
                "subcases": list(SUBCASE_SPECS),
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
    failures: list[str] = []

    for subcase, spec in SUBCASE_SPECS.items():
        result = await evaluate_remote_output_dir(
            session,
            output_dir=f'{task_cfg.metadata["remote_output_dir"]}/{subcase}',
            reference_dir=f'{task_cfg.metadata["reference_dir"]}/{subcase}',
            spec=spec,
        )
        if result["failures"]:
            failures.extend([f"{subcase}: {failure}" for failure in result["failures"]])

    if failures:
        logger.warning("computational_materials_science evaluation failures: %s", failures)
    return [1.0 if not failures else 0.0]
