"""AgentHLE task: physical_sciences/phonon_dispersion_thermodynamics."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.physical_sciences.phonon_dispersion_thermodynamics.scripts.verify_outputs import (
    REQUIRED_FILES,
    evaluate_output_tree,
)

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

VARIANTS = [("base", "Phonon dispersion and thermodynamics for a 2D hexagonal lattice")]


async def _missing(session: cb.DesktopSession, path: str, *, label: str, tag: str) -> bool:
    if await session.exists(path):
        return False
    logger.error("[%s] Missing %s: %s", tag, label, path)
    return True


@dataclass
class PhononDispersionThermodynamicsConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "physical_sciences"
    TASK_NAME: str = "phonon_dispersion_thermodynamics"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def input_problem_spec(self) -> str:
        return f"{self.input_dir}/problem_spec.md"

    @property
    def input_runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def input_runtime_pyproject(self) -> str:
        return f"{self.input_runtime_env_dir}/pyproject.toml"

    @property
    def software_readme(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Your Task
Use the staged problem statement to compute the phonon dispersion relation, phonon density of states, and thermodynamic properties for the specified 2D hexagonal lattice.

## Input Files
- Problem specification: `{self.input_problem_spec}`
- Optional runtime manifest: `{self.input_runtime_pyproject}`

## Optional Python Setup
If you want a clean scientific Python environment, run:

```bash
cd "{self.input_runtime_env_dir}"
uv sync
```

## What You Must Do
1. Read `{self.input_problem_spec}` carefully and follow its lattice geometry, path discretization, and output contract exactly.
2. Construct the dynamical matrix yourself; do not use phonon or materials-science frameworks such as phonopy or ASE.
3. Write the five required deliverables under `{self.remote_output_dir}`:
   - `diatomic_1d.npz`
   - `dispersion_2d.npz`
   - `dos.npz`
   - `thermodynamics.npz`
   - `results.json`
4. Keep all final deliverables inside `{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_problem_spec": self.input_problem_spec,
                "input_runtime_env_dir": self.input_runtime_env_dir,
                "input_runtime_pyproject": self.input_runtime_pyproject,
                "software_readme": self.software_readme,
                "required_output_files": list(REQUIRED_FILES),
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=PhononDispersionThermodynamicsConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).task_description,
            metadata=PhononDispersionThermodynamicsConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
        for variant_name, variant_label in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]

    with tempfile.TemporaryDirectory(prefix="phonon_dispersion_eval_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        local_output_dir = tmp_root / "output"
        local_reference_dir = tmp_root / "reference"
        local_output_dir.mkdir()
        local_reference_dir.mkdir()

        for name in REQUIRED_FILES:
            output_path = f'{meta["remote_output_dir"]}/{name}'
            reference_path = f'{meta["reference_dir"]}/{name}'
            if not await session.exists(output_path):
                logger.error("[%s] Missing output file at %s", tag, output_path)
                return [0.0]
            if not await session.exists(reference_path):
                logger.error("[%s] Missing reference file at %s", tag, reference_path)
                return [0.0]
            try:
                (local_output_dir / name).write_bytes(await session.read_bytes(output_path))
                (local_reference_dir / name).write_bytes(await session.read_bytes(reference_path))
            except Exception as exc:
                logger.exception("[%s] Failed to fetch %s: %s", tag, name, exc)
                return [0.0]

        try:
            result = evaluate_output_tree(local_output_dir, local_reference_dir)
        except Exception as exc:
            logger.exception("[%s] Evaluation failed: %s", tag, exc)
            return [0.0]

    logger.info("[%s] evaluation=%s", tag, json.dumps(result, sort_keys=True))
    return [float(result.get("score", 0.0))]
