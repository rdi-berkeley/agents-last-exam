"""egt710_table1_smiles_extraction — AgentHLE chemistry task."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

logger = logging.getLogger(__name__)

CHEMINFO_URL = "https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html"
EVAL_TMP_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_eval\egt710_table1_smiles_extraction"
SCRIPTS_DIR = Path(__file__).parent / "scripts"

VARIANTS = [
    ("table1", "Table 1 compounds 1-9"),
]


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: float | None = None,
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
    if (await session.file_exists(path) or await session.directory_exists(path)):
        return False
    logger.error("[%s] Missing staged %s at %s", tag, label, path)
    return True


@dataclass
class EGT710TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "physical_sciences"

    TASK_NAME: str = "egt710_table1_smiles_extraction"
    VARIANT_NAME: str = ""
    VARIANT_LABEL: str = ""

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def input_pdf(self) -> str:
        return rf"{self.input_dir}\source_paper.pdf"

    @property
    def output_file(self) -> str:
        return rf"{self.remote_output_dir}\submission.csv"

    @property
    def reference_file(self) -> str:
        return rf"{self.reference_dir}\reference.csv"

    @property
    def software_shortcut(self) -> str:
        return rf"{self.software_dir}\ChemInfo.lnk"

    @property
    def evaluator_python(self) -> str:
        return rf"{self.reference_dir}\evaluator_env\.venv\Scripts\python.exe"

    @property
    def task_description(self) -> str:
        return f"""\
You are a medicinal chemistry researcher extracting a structure-activity table.

## Your Task
Use the uploaded paper PDF to reconstruct the nine compounds in **Table 1** and produce a CSV with validated SMILES strings.

## Input Files
- Paper PDF: `{self.input_pdf}`

## Software
- Launch the mandated SMILES validator from: `{self.software_shortcut}`
- The shortcut opens the ChemInfo SMILES Generator & Checker:
  `{CHEMINFO_URL}`

## What You Must Do
1. Open `{self.input_pdf}`
2. Locate Table 1 in the paper titled:
   `Discovery of EGT710, an Oral Nonpeptidomimetic Reversible Covalent SARS-CoV-2 Main Protease Inhibitor`
3. Extract compounds `1` through `9`
4. Reconstruct and validate each compound's SMILES string using ChemInfo
5. Save a CSV exactly to:
   `{self.output_file}`

## Output Requirements
- The file must be CSV
- The columns must be exactly:
  `Compound_ID,SMILES,IC50_uM,Solubility_mM`
- Include exactly one row each for compound IDs `1,2,3,4,5,6,7,8,9`
- Do not save extra files

## Evaluation
- Rows are matched by `Compound_ID`
- `SMILES` is checked by chemical identity, not raw string equality
- `IC50_uM` and `Solubility_mM` must match the reference values
- Final score = fraction of correctly matched rows out of 9
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_label": self.VARIANT_LABEL,
                "input_dir": self.input_dir,
                "input_pdf": self.input_pdf,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "software_shortcut": self.software_shortcut,
                "evaluator_python": self.evaluator_python,
                "cheminfo_url": CHEMINFO_URL,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=EGT710TaskConfig(
                VARIANT_NAME=tag,
                VARIANT_LABEL=variant_label,
            ).task_description,
            metadata=EGT710TaskConfig(
                VARIANT_NAME=tag,
                VARIANT_LABEL=variant_label,
            ).to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": "windows"},
            },
        )
        for tag, variant_label in VARIANTS
    ]


_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    tag = meta["variant_name"]
    output_file = meta["output_file"]
    reference_file = meta["reference_file"]
    evaluator_python = meta["evaluator_python"]

    if not (await session.file_exists(evaluator_python) or await session.directory_exists(evaluator_python)):
        raise RuntimeError(
            f"evaluator-controlled python missing at {evaluator_python}; "
            "Stage 1 must stage reference/evaluator_env/.venv"
        )
    if await _log_missing_path(session, reference_file, tag=tag, label="reference file"):
        return [0.0]

    await session.interface.create_dir(EVAL_TMP_DIR)
    verify_script_path = rf"{EVAL_TMP_DIR}\verify_submission.py"
    await session.write_file(verify_script_path, _read_script("verify_submission.py"))

    if not (await session.file_exists(output_file) or await session.directory_exists(output_file)):
        logger.error("[%s] Agent output not found at %s", tag, output_file)
        return [0.0]

    result = await _run_command(
        session,
        f'"{evaluator_python}" "{verify_script_path}" --agent "{output_file}" --ref "{reference_file}"',
        timeout=300.0,
        check=False,
    )
    if result["return_code"] != 0 and not result.get("stdout", "").strip():
        logger.error("[%s] Verification failed before JSON output: %s", tag, result.get("stderr", "")[:400])
        return [0.0]

    try:
        payload = json.loads(result["stdout"])
    except Exception:
        logger.error(
            "[%s] Could not parse verifier output: stdout=%r stderr=%r",
            tag,
            result.get("stdout", "")[:400],
            result.get("stderr", "")[:400],
        )
        return [0.0]

    score = float(payload.get("score", 0.0))
    logger.info(
        "[%s] score=%.3f matched_rows=%s total_rows=%s reason=%s",
        tag,
        score,
        payload.get("matched_rows"),
        payload.get("total_rows"),
        payload.get("reason"),
    )
    return [score]
