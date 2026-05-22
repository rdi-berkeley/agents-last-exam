"""idp_ensemble_scoring -- AgentHLE computational structural biology task.

The agent must rank 5 IDP ensemble generation models by how well their
ensembles match experimental NMR data (chemical shifts, J-couplings,
NOE/PRE) using the provided CSpred/UCBShift and X-EISD tools.
"""

import json
import logging
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
    if await session.exists(path):
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
Install the required Python dependencies for CSpred/UCBShift and xeisd. \
The tools are provided under `{self.input_dir}`. Key dependencies include \
numpy, pandas, biopython==1.74, scikit-learn==0.22, joblib, matplotlib.

### Step 2: Back-calculate observables
For each (model, protein, conformer):
- Chemical Shifts (CS): use UCBShift via `{self.input_dir}/CSpred/CSpred.py`
- J-couplings (JC), NOE, PRE: use `{self.input_dir}/xeisd/calculator.py`

### Step 3: Score ensembles
Score each (model, protein) ensemble on the full 200-conformer pool using \
`xeisd.optimizer.XEISD.calc_scores()`. This produces one row per \
(model, protein) with columns: cs_score, jc_score, noe_score, pre_score.

### Step 4: Normalize and rank
- For each observable, build a method x protein matrix of raw scores \
restricted to the test protein set for that observable (defined in \
`{self.input_dir}/info.py`). Min-max normalize each protein column across \
methods: (X - Xmin) / (Xmax - Xmin). Row-mean across proteins gives one \
number per (method, observable).
- For each (method, protein), sum the available per-observable raw scores \
into raw_total[method, protein]. Min-max normalize each protein column of \
raw_total across methods. Row-mean gives the Total score. Rank by Total \
descending (higher = better experimental agreement, on [0, 1]).

### Step 5: Save output
Save the final results as a CSV file to:
`{self.output_file}`

The CSV must have these exact columns: Method, Total, CS, JC, NOE/PRE
Each row is one model (Model1 through Model5).
All numeric values must be between 0 and 1.

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

    # Agent output precheck
    if not await session.exists(meta["output_file"]):
        logger.error("agent missing output: %s", meta["output_file"])
        return [0.0]

    # Evaluator-controlled prerequisites — staging/config bug if missing
    for ref_key in ("reference_dir", "reference_file"):
        if not await session.exists(meta[ref_key]):
            raise RuntimeError(
                f"evaluator-controlled {ref_key} missing: {meta[ref_key]}"
            )

    # Upload verifier script to eval temp dir
    await session.makedirs(EVAL_TMP_DIR)
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
    if result["return_code"] != 0 and not result.get("stdout", "").strip():
        logger.error(
            "Verifier failed before JSON output: %s",
            result.get("stderr", "")[:400],
        )
        return [0.0]

    try:
        payload = json.loads(result["stdout"])
    except Exception:
        logger.error(
            "Could not parse verifier output: stdout=%r stderr=%r",
            result.get("stdout", "")[:400],
            result.get("stderr", "")[:400],
        )
        return [0.0]

    score = float(payload.get("score", 0.0))
    logger.info(
        "score=%.3f passed=%s reasons=%s",
        score,
        payload.get("passed"),
        payload.get("reasons"),
    )
    return [score]
