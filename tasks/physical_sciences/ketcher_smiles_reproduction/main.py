"""ketcher_smiles_reproduction — AgentHLE chemistry task."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.95
EVAL_TMP_DIR = r"C:\Users\User\AppData\Local\Temp\agenthle_eval\ketcher_smiles_reproduction"
SCRIPTS_DIR = Path(__file__).parent / "scripts"

VARIANTS = [
    (
        "7yy",
        "7YY",
        "https://lifescience.opensource.epam.com/ketcher/",
    ),
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
class KetcherSmilesTaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "physical_sciences"

    TASK_NAME: str = "ketcher_smiles_reproduction"
    VARIANT_NAME: str = ""
    MOLECULE_ID: str = ""
    KETCHER_URL: str = ""

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def structure_svg_file(self) -> str:
        return rf"{self.input_dir}\structure.svg"

    @property
    def output_file(self) -> str:
        return rf"{self.output_dir}\submission.smi"

    @property
    def reference_file(self) -> str:
        return rf"{self.reference_dir}\reference.smi"

    @property
    def software_shortcut(self) -> str:
        return rf"{self.software_dir}\Ketcher.lnk"

    @property
    def evaluator_python(self) -> str:
        return rf"{self.reference_dir}\evaluator_env\.venv\Scripts\python.exe"

    @property
    def task_description(self) -> str:
        return f"""\
You are a chemistry researcher reproducing a molecular structure in Ketcher.

## Your Task
Recreate molecule **{self.MOLECULE_ID}** from the provided SVG image and export it as a SMILES file.

## Input Files
- Molecule structure image: `{self.structure_svg_file}`

## Software
- Launch Ketcher from: `{self.software_shortcut}`
- The shortcut opens the verified Ketcher web app URL:
  `{self.KETCHER_URL}`

## What You Must Do
1. Open Ketcher from the shortcut in `software/`
2. Reproduce the molecular structure shown in `{self.structure_svg_file}`
3. Preserve the intended bonding and ring connectivity
4. Export the result as a SMILES file
5. Save the exported file exactly to:
   `{self.output_file}`

## Output Requirements
- Output format must be `.smi`
- The file must contain exactly one non-comment SMILES entry
- Do not save screenshots, extra files, or alternate molecule versions
- The exported molecule must preserve the intended chemical identity of the SVG structure
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "molecule_id": self.MOLECULE_ID,
                "ketcher_url": self.KETCHER_URL,
                "input_dir": self.input_dir,
                "structure_svg_file": self.structure_svg_file,
                "output_file": self.output_file,
                "reference_file": self.reference_file,
                "software_shortcut": self.software_shortcut,
                "evaluator_python": self.evaluator_python,
                "similarity_threshold": SIMILARITY_THRESHOLD,
            }
        )
        return metadata


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=KetcherSmilesTaskConfig(
                VARIANT_NAME=tag,
                MOLECULE_ID=molecule_id,
                KETCHER_URL=ketcher_url,
            ).task_description,
            metadata=KetcherSmilesTaskConfig(
                VARIANT_NAME=tag,
                MOLECULE_ID=molecule_id,
                KETCHER_URL=ketcher_url,
            ).to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": "windows"},
            },
        )
        for tag, molecule_id, ketcher_url in VARIANTS
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
    threshold = float(meta["similarity_threshold"])
    evaluator_python = meta["evaluator_python"]

    if not (await session.file_exists(evaluator_python) or await session.directory_exists(evaluator_python)):
        raise RuntimeError(
            f"evaluator-controlled python missing at {evaluator_python}; "
            "Stage 1 must stage reference/evaluator_env/.venv"
        )
    if await _log_missing_path(session, reference_file, tag=tag, label="reference file"):
        return [0.0]

    await session.interface.create_dir(EVAL_TMP_DIR)
    verify_script_path = rf"{EVAL_TMP_DIR}\verify_smiles.py"
    await session.write_file(verify_script_path, _read_script("verify_smiles.py"))

    if not (await session.file_exists(output_file) or await session.directory_exists(output_file)):
        logger.error("[%s] Agent output not found at %s", tag, output_file)
        return [0.0]

    result = await _run_command(
        session,
        f'"{evaluator_python}" "{verify_script_path}" --agent "{output_file}" --ref "{reference_file}" --threshold {threshold}',
        timeout=300.0,
        check=False,
    )
    if result["return_code"] != 0 and not result.get("stdout", "").strip():
        logger.error("[%s] Verification failed before JSON output: %s", tag, result.get("stderr", "")[:400])
        return [0.0]

    try:
        payload = json.loads(result["stdout"])
    except Exception:
        logger.error("[%s] Could not parse verifier output: stdout=%r stderr=%r", tag, result.get("stdout", "")[:400], result.get("stderr", "")[:400])
        return [0.0]

    score = float(payload.get("score", 0.0))
    logger.info(
        "[%s] score=%.3f similarity=%s passed=%s reason=%s",
        tag,
        score,
        payload.get("similarity"),
        payload.get("passed"),
        payload.get("reason"),
    )
    return [score]
