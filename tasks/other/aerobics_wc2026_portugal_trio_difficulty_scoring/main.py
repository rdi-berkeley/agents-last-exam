"""AgentHLE task: aerobics_wc2026_portugal_trio_difficulty_scoring."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import cua_bench as cb

from dataclasses import dataclass

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
VERIFY_SCRIPT_PATH = SCRIPTS_DIR / "verify_outputs.py"


def _load_verify_module():
    spec = importlib.util.spec_from_file_location(
        "aerobics_wc2026_portugal_trio_difficulty_scoring_verify_outputs",
        VERIFY_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load verifier module from {VERIFY_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load_verify_module()


async def _run_command(
    session: cb.DesktopSession,
    command: str,
    *,
    timeout: Optional[float] = None,
    check: bool = False,
) -> dict[str, Any]:
    try:
        if timeout is not None:
            return await session.run_command(command, timeout=timeout, check=check)
        return await session.run_command(command, check=check)
    except TypeError:
        return await session.run_command(command, check=check)


async def _reset_runtime_output_dir(session: cb.DesktopSession, output_dir: str) -> None:
    await _run_command(
        session,
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import shutil\n"
        f"root = Path({output_dir!r})\n"
        "if root.exists():\n"
        "    for child in root.iterdir():\n"
        "        if child.is_dir():\n"
        "            shutil.rmtree(child)\n"
        "        else:\n"
        "            child.unlink()\n"
        "else:\n"
        "    root.mkdir(parents=True, exist_ok=True)\n"
        "root.mkdir(parents=True, exist_ok=True)\n"
        "PY",
        check=True,
    )


@dataclass
class AerobicsDifficultyConfig(LinuxTaskConfig):
    """Configuration for the aerobics difficulty scoring task."""

    DOMAIN_NAME: str = "other"
    TASK_NAME: str = "aerobics_wc2026_portugal_trio_difficulty_scoring"
    VARIANT_NAME: str = "variant_1"
    OS_TYPE: str = "linux"

    @property
    def output_test_pos_dir(self) -> str:
        return f"{self.task_dir}/output_test_pos"

    @property
    def output_test_neg_dir(self) -> str:
        return f"{self.task_dir}/output_test_neg"

    @property
    def video_file(self) -> str:
        return f"{self.input_dir}/trio.mov"

    @property
    def cop_pdf(self) -> str:
        return f"{self.input_dir}/FIG Aerobic Gymnastics Code of Points (2025-2028).pdf"

    @property
    def output_file(self) -> str:
        return f"{self.output_dir}/difficulty_element_log.xlsx"

    @property
    def reference_xlsx(self) -> str:
        return f"{self.reference_dir}/difficulty_element_log.xlsx"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on Ubuntu.

## Your Task
Inspect the routine video and the FIG Aerobic Gymnastics Code of Points, then
produce a difficulty-element spreadsheet for the routine.

## Visible Inputs
- Video: `{self.video_file}`
- Rulebook PDF: `{self.cop_pdf}`

## Output
Write exactly one spreadsheet to `{self.output_file}`.

The spreadsheet must contain exactly these four columns:
- `element_name`
- `FIG_code`
- `credited_value`
- `meets_minimum_standard`

Use one row per detected difficulty element, in chronological order.
The total D-score implied by the `credited_value` column should fall between
`7.0` and `7.2`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "variant_name": self.VARIANT_NAME,
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "reference_dir": self.reference_dir,
                "output_test_pos_dir": self.output_test_pos_dir,
                "output_test_neg_dir": self.output_test_neg_dir,
                "software_dir": self.software_dir,
                "output_dir": self.output_dir,
                "video_file": self.video_file,
                "cop_pdf": self.cop_pdf,
                "output_file": self.output_file,
                "reference_gcs_prefix": (
                    "gs://ale-data-all/other/aerobics_wc2026_portugal_trio_difficulty_scoring/"
                    "variant_1/reference"
                ),
                "reference_xlsx": self.reference_xlsx,
                "canonical_gcs_root": "gs://ale-data-all/other/aerobics_wc2026_portugal_trio_difficulty_scoring/variant_1/",
            }
        )
        return metadata


config = AerobicsDifficultyConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {"os_type": config.OS_TYPE},
            },
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Score the agent output locally against the hidden reference workbook."""

    meta = task_cfg.metadata
    output_file = meta["output_file"]
    if not (await session.file_exists(output_file) or await session.directory_exists(output_file)):
        logger.warning("missing output workbook at %s", output_file)
        return [0.0]

    if not (await session.file_exists(meta["reference_xlsx"]) or await session.directory_exists(meta["reference_xlsx"])):
        raise RuntimeError(
            f"evaluator-controlled reference workbook missing: {meta['reference_xlsx']}"
        )

    agent_bytes = await session.read_bytes(output_file)
    reference_bytes = await session.read_bytes(meta["reference_xlsx"])
    report = VERIFY_MODULE.score_workbook_bytes(agent_bytes, reference_bytes)
    logger.info("aerobics verifier=%s", json.dumps(report))
    return [float(report.get("score", 0.0))]
