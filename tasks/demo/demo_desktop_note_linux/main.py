"""Demo Task: Desktop Note created on Linux.

Linux counterpart of demo_desktop_note. This validates task-data staging,
eval-only reference visibility, and output pullback. The agent must create the
requested note from visible input data.
"""

import logging
from dataclasses import dataclass

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

EXPECTED_LINES = [
    "AgentHLE Desktop Plugin Demo",
    "This note was created through the desktop UI.",
]

LINUX_REMOTE_ROOT = "/media/user/data/agenthle"


@dataclass
class TaskConfig(LinuxTaskConfig):
    """Configuration for the Linux desktop note demo task."""

    DOMAIN_NAME: str = "demo"
    TASK_NAME: str = "demo_desktop_note_linux"
    VARIANT_NAME: str = "base"
    OS_TYPE: str = "linux"
    REQUIRES_TASK_DATA: bool = True
    REMOTE_ROOT_DIR: str = LINUX_REMOTE_ROOT

    @property
    def input_request_file(self) -> str:
        return f"{self.input_dir}/note_request.txt"

    @property
    def editor_launcher(self) -> str:
        return f"{self.software_dir}/open_text_editor.sh"

    @property
    def output_file(self) -> str:
        return f"{self.remote_output_dir}/agenthle_desktop_plugin_demo.txt"

    @property
    def reference_file(self) -> str:
        return f"{self.reference_dir}/expected_note.txt"

    @property
    def expected_text(self) -> str:
        return "\n".join(EXPECTED_LINES)

    @property
    def task_description(self) -> str:
        lines_text = "\n".join(f"   {line}" for line in EXPECTED_LINES)
        return f"""\
Goal: Create a note in a text editor from staged task data.

Steps:
1. Open a text editor. A task-local launcher is available at:
   {self.editor_launcher}
2. Read the visible task input file:
   {self.input_request_file}
3. Type exactly these two lines:
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
                "editor_launcher": self.editor_launcher,
                "reference_file": self.reference_file,
                "expected_text": self.expected_text,
                "expected_lines": EXPECTED_LINES,
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    """Register the Linux desktop note task."""
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
