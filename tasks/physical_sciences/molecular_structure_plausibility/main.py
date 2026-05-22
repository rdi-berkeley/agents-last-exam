"""AgentHLE task: physical_sciences/molecular_structure_plausibility."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

# cua_bench loads task modules via exec_module without always pre-registering
# them in sys.modules; dataclass needs this for string annotation handling.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.physical_sciences.molecular_structure_plausibility.scripts.score_molecular_structure_plausibility import (
    evaluate_files,
)

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

VARIANTS = [("base", "Molecular structure plausibility filtering")]


async def _missing(session: cb.DesktopSession, path: str, *, label: str, tag: str) -> bool:
    if await session.exists(path):
        return False
    logger.error("[%s] Missing %s: %s", tag, label, path)
    return True


@dataclass
class MolecularStructurePlausibilityConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "physical_sciences"
    TASK_NAME: str = "molecular_structure_plausibility"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def input_xyz_dir(self) -> str:
        return f"{self.input_dir}/xyz_files"

    @property
    def input_task_brief(self) -> str:
        return f"{self.input_dir}/task_brief.md"

    @property
    def input_runtime_env_dir(self) -> str:
        return f"{self.input_dir}/runtime_env"

    @property
    def input_runtime_pyproject(self) -> str:
        return f"{self.input_runtime_env_dir}/pyproject.toml"

    @property
    def input_runtime_lock(self) -> str:
        return f"{self.input_runtime_env_dir}/uv.lock"

    @property
    def software_readme(self) -> str:
        return f"{self.software_dir}/README.txt"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/problematic_structures.txt"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/answer_key.txt"

    @property
    def task_description(self) -> str:
        return f"""\
You are filtering molecular structure files based on physical plausibility.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Input Files
- Structure directory: `{self.input_xyz_dir}`
- Task brief: `{self.input_task_brief}`
- Optional Python dependency manifest: `{self.input_runtime_env_dir}`

## Optional Python Setup
If you want RDKit and scientific Python tools, use the staged runtime manifest:

```bash
cd "{self.input_runtime_env_dir}"
uv sync --frozen
```

## What You Must Do
1. Read `{self.input_task_brief}`.
2. Inspect the `.xyz` molecular structures in `{self.input_xyz_dir}`.
3. Identify every structure that is physically implausible under the task brief.
4. Save one text file at `{self.output_file}`.

## Output Requirements
- Write one `.xyz` filename per line.
- Use filenames exactly as they appear in `{self.input_xyz_dir}`.
- Do not include paths.
- Save the final answer only at `{self.output_file}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_xyz_dir": self.input_xyz_dir,
                "input_task_brief": self.input_task_brief,
                "input_runtime_env_dir": self.input_runtime_env_dir,
                "input_runtime_pyproject": self.input_runtime_pyproject,
                "input_runtime_lock": self.input_runtime_lock,
                "software_readme": self.software_readme,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
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
            description=MolecularStructurePlausibilityConfig(
                VARIANT_NAME=variant_name,
                VARIANT_LABEL=variant_label,
            ).task_description,
            metadata=MolecularStructurePlausibilityConfig(
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

    for key, label in [
        ("output_file", "output problematic_structures.txt"),
        ("reference_file", "hidden reference answer_key.txt"),
        ("input_xyz_dir", "input xyz directory"),
    ]:
        if not await session.exists(meta[key]):
            logger.error("[%s] Missing %s at %s", tag, label, meta[key])
            return [0.0]

    with tempfile.TemporaryDirectory(prefix="molecular_structure_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_output = tmp / "problematic_structures.txt"
        local_reference = tmp / "answer_key.txt"
        local_xyz_dir = tmp / "xyz_files"
        local_xyz_dir.mkdir()
        try:
            local_output.write_bytes(await session.read_bytes(meta["output_file"]))
            local_reference.write_bytes(await session.read_bytes(meta["reference_file"]))
            for filename in await session.list_dir(meta["input_xyz_dir"]):
                if not filename.endswith(".xyz"):
                    continue
                local_path = local_xyz_dir / filename
                remote_path = f'{meta["input_xyz_dir"]}/{filename}'
                local_path.write_bytes(await session.read_bytes(remote_path))
            result = evaluate_files(
                output_file=local_output,
                reference_file=local_reference,
                input_xyz_dir=local_xyz_dir,
            )
        except Exception as exc:
            logger.exception("[%s] Evaluation failed: %s", tag, exc)
            return [0.0]

    logger.info("[%s] evaluation=%s", tag, json.dumps(result, sort_keys=True))
    return [float(result.get("score", 0.0))]
