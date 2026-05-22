"""AgentHLE task: physical_sciences/qm9_mmff94_forcefield_survey_1."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

VARIANTS = [
    (
        "base",
        "Five-phase MMFF94 vs QM9 force-field failure survey on the full QM9 dataset.",
    )
]

SCRIPTS_DIR = Path(__file__).parent / "scripts"
EVAL_TMP_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_eval\qm9_mmff94_forcefield_survey_1"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def _extract_json_payload(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON payload found in verifier stdout")


@dataclass
class Qm9Mmff94ForceFieldSurveyConfig(GeneralTaskConfig):
    REMOTE_ROOT_DIR: str = r"E:\agenthle"
    DOMAIN_NAME: str = "physical_sciences"
    TASK_NAME: str = "qm9_mmff94_forcefield_survey_1"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def input_archive(self) -> str:
        return rf"{self.input_dir}\dsgdb9nsd.xyz.tar.bz2"

    @property
    def task_description(self) -> str:
        return f"""\
You are surveying systematic failures of the MMFF94 molecular-mechanics force field
relative to B3LYP/6-31G(2df,p) reference geometries across the QM9 dataset.

## Variant
`{self.VARIANT_NAME}`: {self.VARIANT_LABEL}

## Input
- QM9 archive (bz2-compressed tar of 130,831 XYZ files): `{self.input_archive}`
- Stream the archive with `tarfile.open(..., "r:bz2")`. Do not extract it on disk.

## Software
- Python 3.12 is on PATH (`python`).
- RDKit, NumPy, pandas, and matplotlib are pre-installed in the system Python.
- No internet access is required or permitted.

## Output Directory
Write every result file directly inside:

```
{self.remote_output_dir}
```

## What You Must Produce

The following 10 files under `{self.remote_output_dir}`:

1. `force_field_failures.csv` — Phase 1. For every QM9 molecule with exactly
   2 heteroatoms (O/N/F): embed one ETKDG conformer with `randomSeed=42`,
   MMFF94-optimize, measure the heteroatom–heteroatom distance, and compare
   to the QM9 B3LYP distance. Emit rows with `|Discrepancy_A| >= 1.0` using
   columns `Molecule_ID, SMILES, QM9_Dist_A, RDKit_Dist_A, Discrepancy_A`.
   Flush continuously so a crash late in the scan does not lose earlier rows.
2. `phase2_classified.csv` — Phase 2. For every Phase 1 row, run a 200-conformer
   ETKDGv3 search (`randomSeed=42`), MMFF94-optimize each conformer, pick the
   lowest-energy conformer, and measure its heteroatom distance. Classify
   `genuine_ff_failure` when the residual discrepancy (|global-min dist − QM9 dist|)
   >= 0.5 Å, otherwise `sampling_artifact`. Columns: `Molecule_ID, SMILES,
   QM9_Dist_A, Naive_MMFF94_Dist_A, Naive_Discrepancy_A, Global_Min_Dist_A,
   Global_Min_Energy_kcal, Residual_Discrepancy_A, Classification`.
3. `phase3_scaffold_analysis.json` — Phase 3. Murcko-scaffold decomposition and
   SMARTS-based functional-group tagging of the genuine failures. Mean/max/min
   residual discrepancy per scaffold and per functional group, plus the top-5
   worst molecules. Top-level keys: `survey_statistics`, `scaffold_analysis`,
   `top5_worst_molecules`, `all_genuine_failures`. **This JSON must not include
   QM9 molecule IDs** — identify molecules by SMILES and rank only.
4. `phase4_pes_results.json` — Phase 4. For each of the top-5 worst genuine
   failures ranked by residual discrepancy, run a constrained MMFF94 PES scan
   compressing the Phase 1 heteroatom pair from 8.0 Å down to 2.5 Å in 0.1 Å
   steps using `MMFFAddDistanceConstraint(relative=False, forceConstant=10000.0)`.
   Re-initialise the force field fresh at each step. Use the Phase 2 global-minimum
   geometry as the starting structure and RDKit indices after `AddHs`. Record
   the global minimum energy/distance, the quantum-forced energy at the exact
   QM9 distance, the conformational snap distance (arg-max |d²E/dd²|), the snap
   energy drop, and ΔE (quantum-forced − global-min). Top-level keys `rank_1`…
   `rank_5`, each with a `data` object containing `rank, molecule_id,
   canonical_isomeric_smiles, heteroatom_pair, qm9_heteroatom_distance,
   mmff94_global_min_distance, mmff94_global_min_energy, quantum_forced_energy,
   conformational_snap_distance, snap_energy_drop, delta_e`.
5. `pes_scan_rank1.png`, `pes_scan_rank2.png`, `pes_scan_rank3.png`,
   `pes_scan_rank4.png`, `pes_scan_rank5.png` — one PES plot per top-5 molecule.
6. `phase5_final_report.json` — Phase 5. Top-level keys `survey_statistics`,
   `scaffold_analysis`, `top5_characterization`, `top5_pes_summary`. The
   `top5_characterization` object must split the top-5 into
   `short_qm9_distance_molecules` (QM9 heteroatom distance < 3.5 Å) and
   `long_qm9_distance_molecules` (≥ 3.5 Å), plus a `failure_mode_distinction`
   text field explaining what physically distinguishes the two groups.

## Runtime Expectations
- Wall-clock budget: roughly 12–16 hours end to end. Phase 2 dominates
  (200-conformer search over ~635 candidates).
- Peak memory is observed in Phase 2 (~50–100 MB per molecule during
  conformer generation).
- Do not attempt to parallelise with OS-level process spawning inside the
  same Python process — sequential per-molecule processing is sufficient.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_dir": self.input_dir,
                "input_archive": self.input_archive,
                "eval_tmp_dir": EVAL_TMP_DIR,
                "canonical_gcs_root": (
                    f"gs://ale-data-all/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}/"
                ),
            }
        )
        return metadata


def _cfg(variant_name: str, variant_label: str) -> Qm9Mmff94ForceFieldSurveyConfig:
    return Qm9Mmff94ForceFieldSurveyConfig(VARIANT_NAME=variant_name, VARIANT_LABEL=variant_label)


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=_cfg(name, label).task_description,
            metadata=_cfg(name, label).to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "windows"}},
        )
        for name, label in VARIANTS
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]
    agent_output_dir = meta["remote_output_dir"]
    reference_dir = meta["reference_dir"]
    eval_tmp_dir = meta["eval_tmp_dir"]

    if not await session.exists(reference_dir):
        logger.error("[%s] Missing reference dir: %s", tag, reference_dir)
        return [0.0]
    if not await session.exists(agent_output_dir):
        logger.error("[%s] Missing agent output dir: %s", tag, agent_output_dir)
        return [0.0]

    await session.makedirs(eval_tmp_dir)
    verifier_path = rf"{eval_tmp_dir}\verify.py"
    await session.write_file(verifier_path, _read_script("verify.py"))

    command = (
        f'python "{verifier_path}" '
        f'--agent "{agent_output_dir}" '
        f'--reference "{reference_dir}"'
    )

    result = await session.run_command(command, check=False)

    stdout = result.get("stdout", "") if isinstance(result, dict) else ""
    stderr = result.get("stderr", "") if isinstance(result, dict) else ""
    rc = result.get("return_code", 1) if isinstance(result, dict) else 1

    if stderr:
        logger.info("[%s] verifier stderr: %s", tag, stderr.strip()[:2000])

    if rc != 0:
        logger.error(
            "[%s] verifier command failed rc=%s stdout=%s stderr=%s",
            tag,
            rc,
            stdout[:1000],
            stderr[:1000],
        )
        return [0.0]

    try:
        payload = _extract_json_payload(stdout)
    except Exception as exc:
        logger.error(
            "[%s] failed to parse verifier JSON: %s stdout=%s",
            tag,
            exc,
            stdout[:1500],
        )
        return [0.0]

    score = float(payload.get("score", 0.0))
    logger.info(
        "[%s] score=%.4f hard_gate=%s phase_fracs=%s counts=%s",
        tag,
        score,
        payload.get("hard_gate"),
        payload.get("phase_fracs"),
        payload.get("phase_check_counts"),
    )
    return [score]
