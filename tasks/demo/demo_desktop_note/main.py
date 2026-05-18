"""Demo Task: Desktop Note created through the Windows UI.

This task validates desktop-control agents, task-data staging, eval-only
reference visibility, and output pullback. The agent must use Notepad to
create the requested note from visible input data.
"""

import logging
from dataclasses import dataclass

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

EXPECTED_LINES = [
    "AgentHLE Desktop Plugin Demo",
    "This note was created through the desktop UI.",
]


@dataclass
class TaskConfig(GeneralTaskConfig):
    """Configuration for the desktop note demo task."""

    DOMAIN_NAME: str = "demo"
    TASK_NAME: str = "demo_desktop_note"
    VARIANT_NAME: str = "base"
    OS_TYPE: str = "windows"
    REQUIRES_TASK_DATA: bool = True

    @property
    def input_dir(self) -> str:
        return rf"{self.task_dir}\input"

    @property
    def input_request_file(self) -> str:
        return rf"{self.input_dir}\note_request.txt"

    @property
    def notepad_shortcut(self) -> str:
        return rf"{self.software_dir}\Notepad.lnk"

    @property
    def output_file(self) -> str:
        return rf"{self.remote_output_dir}\agenthle_desktop_plugin_demo.txt"

    @property
    def reference_file(self) -> str:
        return rf"{self.reference_dir}\expected_note.txt"

    @property
    def expected_text(self) -> str:
        return "\r\n".join(EXPECTED_LINES)

    @property
    def task_description(self) -> str:
        lines_text = "\n".join(f"   {line}" for line in EXPECTED_LINES)
        return f"""\
Goal: Creating a note in Notepad from staged task data with any tool you have available.

Steps:
1. Open Notepad. A task-local Notepad shortcut is available at:
   {self.notepad_shortcut}
2. Read the task input file:
   {self.input_request_file}
3. Type exactly these two lines into Notepad:
{lines_text}
4. Save the file to:
   {self.output_file}

Verification:
- The file must exist exactly at {self.output_file}.
- The file content must match the two required lines exactly, ignoring Windows vs Unix newlines.
- Use the staged input, software, and output directories during solving.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "output_file": self.output_file,
                "input_dir": self.input_dir,
                "input_request_file": self.input_request_file,
                "notepad_shortcut": self.notepad_shortcut,
                "reference_file": self.reference_file,
                "expected_text": self.expected_text,
                "expected_lines": EXPECTED_LINES,
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    """Register the desktop note task."""
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {
                    "os_type": config.OS_TYPE,
                },
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score based on exact file existence and content match."""
    meta = task_cfg.metadata
    task_tag = meta.get("variant_name", "unknown")
    output_file = meta["output_file"]
    reference_file = meta["reference_file"]

    logger.info("[%s] Starting evaluation", task_tag)

    if not await session.exists(reference_file):
        logger.error(
            "[%s] Reference file not found after eval staging: %s", task_tag, reference_file
        )
        return [0.0]

    if not await session.exists(output_file):
        logger.error("[%s] Output file not found: %s", task_tag, output_file)
        return [0.0]

    try:
        content = await session.read_file(output_file)
        expected_text = await session.read_file(reference_file)
    except Exception as exc:
        logger.error("[%s] Failed to read output/reference file: %s", task_tag, exc)
        return [0.0]

    normalized_content = content.replace("\r\n", "\n").strip()
    normalized_expected = expected_text.replace("\r\n", "\n").strip()
    if normalized_content == normalized_expected:
        logger.info("[%s] Exact content match", task_tag)
        return [1.0]

    lines = meta.get("expected_lines", [])
    hits = sum(1 for line in lines if line.lower() in normalized_content.lower())
    partial = hits / len(lines) if lines else 0.0
    logger.warning(
        "[%s] Content mismatch: matched %d/%d expected lines",
        task_tag,
        hits,
        len(lines),
    )
    return [round(partial * 0.5, 3)]
