"""AgentHLE task: life_sciences/zdock_hiv_dimer_interface_scoring_v1."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

# cua_bench loads task modules via exec_module without always pre-registering
# them in sys.modules; dataclass needs this for string annotation handling.
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

from tasks.common_setup import BaseTaskSetup  # noqa: E402
from tasks.life_sciences.zdock_hiv_dimer_interface_scoring_v1.scripts.score_zdock_interface import (  # noqa: E402
    evaluate_files,
)
from tasks.linux_runtime import LinuxTaskConfig  # noqa: E402

_setup = BaseTaskSetup()

DOMAIN_NAME = "life_sciences"
TASK_NAME = "zdock_hiv_dimer_interface_scoring_v1"
VARIANT_NAME = "base"
OUTPUT_FILENAME = "zdock_interface_scores.csv"

logger = logging.getLogger(__name__)


def _normalize_output_dir_name(raw: str | None) -> str:
    value = (raw or "output").strip().strip("/")
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"OUTPUT_SUBDIR must be a single directory name, got {raw!r}")
    return value


@dataclass
class ZDockHivDimerConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    def __post_init__(self) -> None:
        # validate the configured output subdir name (allowed-set check)
        _normalize_output_dir_name(self.OUTPUT_SUBDIR)

    @property
    def native_complex_file(self) -> str:
        return f"{self.input_dir}/1HVR.pdb"

    @property
    def chain_a_file(self) -> str:
        return f"{self.input_dir}/1HVR_chainA.pdb"

    @property
    def chain_b_file(self) -> str:
        return f"{self.input_dir}/1HVR_chainB.pdb"

    @property
    def pose_archive_file(self) -> str:
        return f"{self.input_dir}/top_preds.tar.gz"

    @property
    def output_file(self) -> str:
        return f"{self.output_dir}/{OUTPUT_FILENAME}"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/zdock_v4_reference_output.csv"

    @property
    def task_description(self) -> str:
        return f"""\
You are evaluating precomputed HIV protease dimer docking predictions on a Linux VM.

## Input Files
- Native complex: `{self.native_complex_file}`
- Chain A structure: `{self.chain_a_file}`
- Chain B structure: `{self.chain_b_file}`
- Docking-pose archive: `{self.pose_archive_file}`

## Your Task
1. Use `{self.native_complex_file}` to derive native Chain A / Chain B interface residues.
   An interface residue is any residue with at least one atom within 5 Angstrom of any atom
   on the opposite chain.
2. Unpack `{self.pose_archive_file}` to obtain `complex.1.pdb` through `complex.10.pdb`.
3. For each docking pose, identify predicted interface residues using the same 5 Angstrom
   atom-distance cutoff.
4. Compute these metrics for every pose:
   - `Overlap Score` = |Predicted interface residues intersect Native interface residues| /
     |Native interface residues|.
   - `Fnat` = fraction of native heavy-atom contacts recovered, where a contact is any pair
     of heavy atoms within 5 Angstrom.
   - `IRMSD` = RMSD in Angstrom after superposing the predicted structure onto the native
     structure using backbone C-alpha atoms of the native interface residues, then measuring
     RMSD over those same C-alpha atoms.
   - `Final Score` = 0.5 * Fnat + 0.3 * Overlap Score - 0.2 * (IRMSD / 10).
5. Rank all 10 poses by `Final Score`, higher is better.

## Required Output
Save exactly one CSV file at:

```text
{self.output_file}
```

The CSV must contain exactly 10 rows, one for each ZDOCK pose rank 1 through 10, and these
columns with exact spelling:

- `Pose Rank (ZDOCK)`
- `Overlap Score`
- `Fnat`
- `IRMSD`
- `Final Score`
- `Final Rank`

Do not modify files under `{self.input_dir}`. Keep all generated files inside
`{self.output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "native_complex_file": self.native_complex_file,
                "chain_a_file": self.chain_a_file,
                "chain_b_file": self.chain_b_file,
                "pose_archive_file": self.pose_archive_file,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "output_filename": OUTPUT_FILENAME,
            }
        )
        return metadata


config = ZDockHivDimerConfig()


@cb.tasks_config(split="train")
def load():
    cfg = ZDockHivDimerConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    if not (await session.file_exists(meta["output_file"]) or await session.directory_exists(meta["output_file"])):
        logger.error("Missing output CSV: %s", meta["output_file"])
        return [0.0]
    if not (await session.file_exists(meta["reference_file"]) or await session.directory_exists(meta["reference_file"])):
        logger.error("Missing hidden reference CSV: %s", meta["reference_file"])
        return [0.0]

    with tempfile.TemporaryDirectory(prefix="zdock_hiv_dimer_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        local_output = tmp / OUTPUT_FILENAME
        local_reference = tmp / "zdock_v4_reference_output.csv"
        try:
            local_output.write_bytes(await session.read_bytes(meta["output_file"]))
            local_reference.write_bytes(await session.read_bytes(meta["reference_file"]))
            result = evaluate_files(local_output, local_reference)
        except Exception as exc:
            logger.exception("Evaluation failed: %s", exc)
            return [0.0]

    logger.info("ZDOCK interface score payload: %s", json.dumps(result, sort_keys=True)[:4000])
    return [float(result.get("score", 0.0))]
