"""idp_ensemble_scoring -- AgentHLE computational structural biology task.

The agent must rank 5 IDP ensemble generation models by how well their
ensembles match experimental NMR data (chemical shifts, J-couplings,
NOE/PRE) using the provided CSpred/UCBShift and X-EISD tools.
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

TASK_NAME = "idp_ensemble_scoring"
SCRIPTS_DIR = Path(__file__).parent / "scripts"
EVAL_TMP_DIR = f"/tmp/agenthle_eval/{TASK_NAME}"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: Optional[float] = None,
    check: bool = False,
) -> dict:
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


async def _log_missing_path(
    session: cb.DesktopSession,
    path: str,
    *,
    tag: str,
    label: str,
) -> bool:
    if await session.file_exists(path) or await session.directory_exists(path):
        return False
    logger.error("[%s] Missing staged %s at %s", tag, label, path)
    return True


@dataclass
class IDPEnsembleScoringConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "life_sciences"
    TASK_NAME: str = "idp_ensemble_scoring"
    VARIANT_NAME: str = ""

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/Final_Output.csv"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/Expected_Final_Output.csv"

    @property
    def task_description(self) -> str:
        return f"""\
You are a computational structural biologist. Your task is to rank 5 IDP \
(intrinsically disordered protein) ensemble generation models by how well \
their ensembles match experimental NMR data.

## Your Task

Use the tools provided locally to back-calculate experimental observables \
from protein ensemble conformers, score each ensemble against experimental \
data, then normalize and rank the models.

### Step 1: Set up your environment
Use an isolated Python 3.7.12 environment for the supplied legacy models. \
Install numpy==1.19.0, pandas==1.1.0, biopython==1.74, scikit-learn==0.22, \
joblib==0.17.0, scipy==1.5.4, matplotlib==3.3.0, and tqdm==4.66.1. \
The package pins are also in `{self.input_dir}/runtime-requirements.txt`. \
Do not load these pickles under a modern incompatible sklearn/Python runtime.
Copy the supplied `{self.input_dir}/CSpred/bins/mkdssp` into a private \
executable directory, make that copy executable, provide a `dssp` alias, \
and prepend it to PATH. Keep task-local environments and scratch under output/.

### Step 2: Back-calculate observables
For each (model, protein, conformer):
- Chemical Shifts (CS): use UCBShift-X only, `CSpred.py -x --pH 5`, \
equivalently `calc_sing_pdb(pH=5, TP=False, ML=True)`. Use the supplied \
R0/R1 models and stock feature defaults; do not use UCBShift-Y or combined mode. \
Select the `atomname + '_X'` prediction for each experimental `resnum`, \
in experimental row order. Do not discard requested measurements.
- For practical runtime, `{self.input_dir}/cached_cspred.py` supplies \
`CachedUCBShiftX.predict_features(frames)` and `experimental_shifts`. \
Build each frame with the supplied `CSpred.build_input(pdb, pH=5)`; \
cache models once and batch independent feature rows. The helper does not \
contain results or perform ensemble scoring. A new CLI/model reload per \
conformer is unnecessarily expensive. Bound feature workers to 12, RF \
prediction workers to 4, and BLAS/OMP threads to 1 on the 16-core machine.
Give each feature worker its own scratch working directory. Keep models \
in one prediction process instead of copying them into feature workers.
- J-couplings (JC), NOE, PRE: use the supplied \
`xeisd.calculator.BACK_Calculators` on chain A, \
with `heavy_atom_substitute=False` for NOE and PRE. Preserve the supplied \
JC cosine/Karplus convention; NOE expands ambiguous hydrogen assignments \
with inverse-sixth-power distance averaging; PRE uses the supplied first \
hydrogen fallback. Do not substitute heavy atoms or different atom options.

### Step 3: Score ensembles
Use exactly conformers 1_validated.pdb through 200_validated.pdb for every \
declared model/protein, each with weight 1/200. Preserve all 19 proteins and \
all five models. No subsampling, reweighting, optimization, or silent omissions.
Use `xeisd.parser.read_bc_data(array, observable)` with its supplied \
uncertainty defaults, then `XEISD(..., nres=chain_A_residue_count, \
pool_size=200).calc_scores([observable], list(range(200)))`. The raw \
log-likelihood score is element 1 of the returned observable entry, not MAE. \
Use the supplied experimental uncertainties and scorer as-is. Required \
back-calculations and scores must be finite; diagnose failures rather than \
filling them with zeros or excluding them.

### Step 4: Normalize and rank
- Use the exact observable protein sets in `{self.input_dir}/info.py`: \
19 CS, 5 JC, 3 NOE, 3 PRE, and the 5-protein NOE/PRE union.
- For CS and JC separately, build a method x protein raw-score matrix on \
that observable's protein set. Min-max normalize each protein column across \
the five methods: (X - Xmin) / (Xmax - Xmin). Row-mean gives CS or JC.
- For NOE/PRE, first sum available raw NOE and PRE scores per method/protein, \
then normalize across methods per protein on the union of the NOE and PRE \
sets, then row-mean over those five proteins. A protein with both modalities \
appears once. Do not average separately normalized NOE/PRE results.
- For each (method, protein), sum the available per-observable raw scores \
into raw_total[method, protein]. Min-max normalize each protein column of \
raw_total across methods. Row-mean gives the Total score. Rank by Total \
descending (higher = better experimental agreement, on [0, 1]). Total uses \
all 19 proteins and counts each available CS, JC, NOE, PRE raw score once. \
It is not the sum or mean of the normalized display columns.
- If a protein column has exactly zero span across the five models, assign \
0.5 to every method for that column and retain it in the row-mean. Missing \
required scores are errors, not degenerate columns. Keep full precision \
until serialization. Break exact Total ties by Method ascending.

### Step 5: Save output
Save the final results as a CSV file to:
`{self.output_file}`

The CSV must have these exact columns: Method, Total, CS, JC, NOE/PRE
Each row is one model (Model1 through Model5).
All numeric values must be finite and between 0 and 1. Column and row order \
are immaterial to grading; use one row per model and no extra columns. \
UTF-8 BOM, standard CSV quoting, and equivalent numeric representations are \
accepted. Score is the fraction of the 20 numeric cells equal to the newly \
regenerated reference after Python `round(float(value), 2)`. Ranking is \
reported as a diagnostic, not an additional score multiplier. Save sufficient \
decimal precision to preserve the calculated values.

## Input Data
- `{self.input_dir}/Ensembles/` -- 5 models (Model1-5), each with protein \
subdirectories containing 200 conformer PDB files
- `{self.input_dir}/Experimental_Data/` -- experimental NMR data per protein
- `{self.input_dir}/CSpred/` -- UCBShift tool with models and binaries
- `{self.input_dir}/xeisd/` -- X-EISD scoring module
- `{self.input_dir}/info.py` -- defines test protein sets per observable

## Output
Save your final CSV to: `{self.output_file}`

## Constraints
- Do NOT use web search. Use only the tools provided locally.
- Use only CS, JC, and NOE/PRE observables. Do NOT use Rg, FRET, SAXS, \
or any other observable type.
- Creating subprocesses and helper scripts is permitted.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "output_file": self.output_file,
                "reference_file": self.reference_file,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    cfg = IDPEnsembleScoringConfig(VARIANT_NAME="default")
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata

    # Evaluator-controlled prerequisites — staging/config bug if missing
    for ref_key in ("reference_dir", "reference_file"):
        if not (
            await session.file_exists(meta[ref_key])
            or await session.directory_exists(meta[ref_key])
        ):
            raise RuntimeError(f"evaluator-controlled {ref_key} missing: {meta[ref_key]}")

    # Upload verifier script to eval temp dir
    await session.interface.create_dir(EVAL_TMP_DIR)
    verify_script_path = f"{EVAL_TMP_DIR}/verify_output.py"
    await session.write_file(verify_script_path, _read_script("verify_output.py"))

    result = await _run_command(
        session,
        (
            f'python3 "{verify_script_path}" '
            f'--output-file "{meta["output_file"]}" '
            f'--reference-file "{meta["reference_file"]}"'
        ),
        timeout=60.0,
        check=False,
    )
    if result["return_code"] != 0:
        raise RuntimeError(
            f"IDP evaluator failed: stdout={result.get('stdout', '')[:400]!r} "
            f"stderr={result.get('stderr', '')[:400]!r}"
        )

    try:
        payload = json.loads(result["stdout"])
        if not isinstance(payload, dict) or "error" in payload:
            raise ValueError("Invalid evaluator result")
        score = float(payload["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid evaluator score")
    except (ValueError, TypeError, KeyError) as error:
        raise RuntimeError("Could not parse valid IDP evaluator result") from error
    logger.info(
        "score=%.3f passed=%s reasons=%s",
        score,
        payload.get("passed"),
        payload.get("reasons"),
    )
    return [score]
