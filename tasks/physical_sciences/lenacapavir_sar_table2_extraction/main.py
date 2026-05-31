"""lenacapavir_sar_table2_extraction — AgentHLE chemistry task."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

logger = logging.getLogger(__name__)

EVAL_TMP_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_eval\lenacapavir_sar_table2_extraction"
SCRIPTS_DIR = Path(__file__).parent / "scripts"

VARIANTS = [
    ("table2", "Lenacapavir Table 2 R1 SAR extraction"),
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


@dataclass
class LenacapavirTaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "physical_sciences"

    TASK_NAME: str = "lenacapavir_sar_table2_extraction"
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
        return rf"{self.output_dir}\submission.csv"

    @property
    def reference_file(self) -> str:
        return rf"{self.reference_dir}\reference.csv"

    @property
    def edge_shortcut(self) -> str:
        return rf"{self.software_dir}\Microsoft Edge.lnk"

    @property
    def vscode_shortcut(self) -> str:
        return rf"{self.software_dir}\Visual Studio Code.lnk"

    @property
    def python_entry(self) -> str:
        return rf"{self.software_dir}\python.bat"

    @property
    def evaluator_python(self) -> str:
        return rf"{self.reference_dir}\evaluator_env\.venv\Scripts\python.exe"

    @property
    def task_description(self) -> str:
        return f"""\
You are a medicinal chemistry researcher extracting one SAR table from a staged paper PDF.

## Your Task
Use the provided manuscript PDF to extract every compound in **Table 2** and reconstruct a full-molecule SMILES string for each row.

## Input Files
- Manuscript PDF: `{self.input_pdf}`

## Software
- PDF/browser shortcut: `{self.edge_shortcut}`
- Editor shortcut: `{self.vscode_shortcut}`
- Python helper: `{self.python_entry}`

## What You Must Do
1. Open `{self.input_pdf}`
2. Go to Table 2, titled `SAR for R1 Analogs`
3. Use the scaffold shown above the table together with each row's R1 drawing
4. Reconstruct the full molecule for every compound row in Table 2
5. Save one CSV exactly to:
   `{self.output_file}`

## Output Requirements
- The file must be CSV
- The columns must be exactly:
  `Ligand_ID,SMILES,EC50_MT4`
- Include one row for every compound listed in Table 2
- `SMILES` must describe the full molecule, not only the R1 fragment
- `EC50_MT4` should be the nM value reported in the table
- Do not save extra files

## Important Constraints
- Use the staged paper PDF as your source of truth
- Do not rely on web search or external answer sources
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
                "edge_shortcut": self.edge_shortcut,
                "vscode_shortcut": self.vscode_shortcut,
                "python_entry": self.python_entry,
                "evaluator_python": self.evaluator_python,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=LenacapavirTaskConfig(
                VARIANT_NAME=tag,
                VARIANT_LABEL=variant_label,
            ).task_description,
            metadata=LenacapavirTaskConfig(
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

    await session.interface.create_dir(EVAL_TMP_DIR)
    verify_script_path = rf"{EVAL_TMP_DIR}\verify_submission.py"
    await session.write_file(verify_script_path, _read_script("verify_submission.py"))

    if not (await session.file_exists(output_file) or await session.directory_exists(output_file)):
        logger.error("[%s] Agent output not found at %s", tag, output_file)
        return [0.0]
    if not (await session.file_exists(reference_file) or await session.directory_exists(reference_file)):
        logger.error("[%s] Reference file not found at %s", tag, reference_file)
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
