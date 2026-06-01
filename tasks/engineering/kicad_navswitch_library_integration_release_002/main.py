"""KiCad navswitch library integration release task."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:
    import cua_bench as cb
except ModuleNotFoundError:  # pragma: no cover - local import fallback only
    class _FallbackTask:
        def __init__(self, description, metadata, computer):
            self.description = description
            self.metadata = metadata
            self.computer = computer

    def _identity_decorator(*args, **kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    cb = SimpleNamespace(
        Task=_FallbackTask,
        DesktopSession=object,
        tasks_config=_identity_decorator,
        setup_task=_identity_decorator,
        evaluate_task=_identity_decorator,
    )

from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup


_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "kicad_navswitch_library_integration_release_002"
VARIANT_NAME = "base"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
PROJECT_BASENAME = "SparkFun_Qwiic_Navigation"
EVAL_TMP_DIR = rf"C:\Users\User\AppData\Local\Temp\agenthle_eval\{TASK_NAME}"
SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def _cmd_stdout(result) -> str:
    if isinstance(result, dict):
        return result.get("stdout", "") or ""
    return getattr(result, "stdout", "") or ""


def _cmd_stderr(result) -> str:
    if isinstance(result, dict):
        return result.get("stderr", "") or ""
    return getattr(result, "stderr", "") or ""


def _cmd_returncode(result) -> int | None:
    if isinstance(result, dict):
        return result.get("return_code", result.get("returncode"))
    return getattr(result, "returncode", None)


@dataclass
class TaskConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "windows"

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def project_input_dir(self) -> str:
        return rf"{self.input_dir}\project"

    @property
    def input_project_file(self) -> str:
        return rf"{self.project_input_dir}\{PROJECT_BASENAME}.kicad_pro"

    @property
    def input_schematic_file(self) -> str:
        return rf"{self.project_input_dir}\{PROJECT_BASENAME}.kicad_sch"

    @property
    def input_pcb_file(self) -> str:
        return rf"{self.project_input_dir}\{PROJECT_BASENAME}.kicad_pcb"

    @property
    def open_kicad_launcher(self) -> str:
        return rf"{self.software_dir}\OpenKiCad.bat"

    @property
    def reference_project_dir(self) -> str:
        return rf"{self.reference_dir}\gold_project"

    @property
    def reference_outputs_dir(self) -> str:
        return rf"{self.reference_dir}\expected_outputs"

    @property
    def task_description(self) -> str:
        return f"""\
You are repairing a KiCad project for the SparkFun Qwiic Navigation Switch board.

## Input
- Broken KiCad project: `{self.input_project_file}`
- Project files and local libraries: `{self.project_input_dir}`
- Datasheets: `{self.input_dir}\\datasheets`
- KiCad launcher: `{self.open_kicad_launcher}`

## Your Task
1. Open the project in the KiCad desktop UI.
2. Inspect U4 (`PCA9554`) and the project-local symbol/footprint libraries.
3. Restore U4's schematic Footprint field to the correct project-local footprint.
4. Update the PCB from the schematic so U4 is restored to the board. The repaired PCB must reference U4's footprint using the full library-qualified name (`library:footprint`) consistent with the project's `fp-lib-table`.
5. Save the repaired KiCad project and leave it release-ready.
6. Do not change circuit intent, non-target component values, non-target footprint assignments, unrelated nets, board outline, connectors, or mounting holes.
7. Do not solve this by directly editing raw KiCad source text files.

## Required Deliverables
Save all deliverables under:
`{self.remote_output_dir}`

Required files and directories:
- `{PROJECT_BASENAME}.kicad_pro`
- `{PROJECT_BASENAME}.kicad_sch`
- `{PROJECT_BASENAME}.kicad_pcb`
- `fp-lib-table`
- `sym-lib-table`
- `nav_local_symbols.kicad_sym`
- `nav_local_footprints.pretty\\`
- `erc.json`
- `drc.json`
- `project.net`
- `gerbers\\`
- `drill\\`
- `board.xml`
- `board.d356`
- `placements.csv`
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "task_tag": self.VARIANT_NAME,
                "input_dir": self.input_dir,
                "project_input_dir": self.project_input_dir,
                "input_project_file": self.input_project_file,
                "input_schematic_file": self.input_schematic_file,
                "input_pcb_file": self.input_pcb_file,
                "open_kicad_launcher": self.open_kicad_launcher,
                "reference_project_dir": self.reference_project_dir,
                "reference_outputs_dir": self.reference_outputs_dir,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{self.VARIANT_NAME}/",
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig(REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"))
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
    await session.interface.create_dir(EVAL_TMP_DIR)
    verifier_path = rf"{EVAL_TMP_DIR}\score_outputs.py"
    await session.write_file(verifier_path, _read_script("score_outputs.py"))

    command = (
        f'python "{verifier_path}" '
        f'--agent-dir "{meta["remote_output_dir"]}" '
        f'--reference-project-dir "{meta["reference_project_dir"]}" '
        f'--reference-outputs-dir "{meta["reference_outputs_dir"]}"'
    )
    result = await session.run_command(command)
    stdout = _cmd_stdout(result).strip()
    stderr = _cmd_stderr(result).strip()
    returncode = _cmd_returncode(result)
    if stderr:
        logger.info("navswitch scorer stderr: %s", stderr)
    if returncode not in (0, None):
        logger.warning("navswitch scorer exited %s: %s", returncode, stderr)
        return [0.0]
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning("navswitch scorer returned non-JSON stdout: %r", stdout[:500])
        return [0.0]
    logger.info("navswitch score result: %s", json.dumps(data, sort_keys=True))
    return [float(data.get("score", 0.0))]
