"""Linux healthcare SAP group-sequential NSCLC task."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cua_bench as cb

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig  # noqa: E402
from tasks.health_medicine.healthcare_sap_group_sequential_nsclc.scripts.score_outputs import (  # noqa: E501, E402
    REQUIRED_FILES,
    evaluate_output_bundle,
)



_setup = BaseTaskSetup()

logger = logging.getLogger(__name__)

DOMAIN_NAME = "health_medicine"
TASK_NAME = "healthcare_sap_group_sequential_nsclc"
VARIANT_NAME = "base"
DEFAULT_REMOTE_ROOT = "/media/user/data/agenthle"
ALLOWED_OUTPUT_DIRS = {
    "output",
    "output_test_pos",
    "output_test_neg",
    "output_admin_pos",
    "output_admin_neg",
}


def _remote_join(*parts: str) -> str:
    return str(PurePosixPath(*parts))


def _shell_quote(path: str) -> str:
    return shlex.quote(path)


def _output_dir_name(value: str) -> str:
    return value.strip().strip("/") or "output"


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    OS_TYPE: str = "linux"

    def __init__(self) -> None:
        super().__init__(
            DOMAIN_NAME=DOMAIN_NAME,
            TASK_NAME=TASK_NAME,
            VARIANT_NAME=VARIANT_NAME,
            OS_TYPE="linux",
            REMOTE_ROOT_DIR=os.environ.get("REMOTE_ROOT_DIR", DEFAULT_REMOTE_ROOT),
            REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"),
        )

    @property
    def output_dir_name(self) -> str:
        return _output_dir_name(self.REMOTE_OUTPUT_DIR)

    @property
    def remote_output_dir(self) -> str:
        return _remote_join(self.task_dir, self.output_dir_name)

    @property
    def protocol_file(self) -> str:
        return _remote_join(self.input_dir, "protocol.json")

    @property
    def instructions_file(self) -> str:
        return _remote_join(self.input_dir, "task_instructions.md")

    @property
    def reference_values_file(self) -> str:
        return _remote_join(self.reference_dir, "reference_values.json")

    @property
    def task_description(self) -> str:
        return f"""\
You are preparing a clinical-trial Statistical Analysis Plan on a Linux VM.

## Task Folder
`{self.task_dir}`

## Inputs
- Protocol JSON: `{self.protocol_file}`
- Detailed instructions: `{self.instructions_file}`

## Goal
Create the full SAP deliverable set in:
`{self.remote_output_dir}`

Required output filenames:
- `SAP.md`
- `analysis.R`
- `sample_size.json`
- `boundaries.csv`
- `multiple_testing.json`
- `power_curve.csv`
- `power_curve.png`
- `boundary_plot.png`

Read the protocol and instructions, compute the group-sequential design quantities, document the SAP, and include a reproducible R script. Do not use internet access. Hidden reference values and grading logic are not visible during solve time.
"""

    def to_metadata(self) -> dict:
        meta = super().to_metadata()
        meta.pop("software_dir", None)
        meta.update(
            {
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "protocol_file": self.protocol_file,
                "instructions_file": self.instructions_file,
                "output_dir_name": self.output_dir_name,
                "required_output_files": sorted(REQUIRED_FILES),
                "reference_values_file": self.reference_values_file,
                "canonical_gcs_root": f"gs://ale-data-all/{DOMAIN_NAME}/{TASK_NAME}/{VARIANT_NAME}/",
            }
        )
        return meta


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _read_bytes_with_retry(
    session: cb.DesktopSession, path: str, retries: int = 3, delay: float = 5.0,
) -> bytes:
    for attempt in range(retries):
        try:
            return await session.read_bytes(path)
        except Exception:
            if attempt == retries - 1:
                raise
            logger.warning("read_bytes(%s) attempt %d failed, retrying in %.0fs", path, attempt + 1, delay)
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")


async def _output_files(session: cb.DesktopSession, output_dir: str) -> dict[str, bytes]:
    result = await session.run_command(
        f"find {_shell_quote(output_dir)} -maxdepth 1 -type f -printf '%f\\n'", check=False
    )
    if result.get("return_code") not in (0, 1):
        raise RuntimeError(f"could not list output dir: {result}")
    files: dict[str, bytes] = {}
    for name in result.get("stdout", "").splitlines():
        if name.strip():
            files[name.strip()] = await _read_bytes_with_retry(
                session, _remote_join(output_dir, name.strip())
            )
    return files


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    try:
        output_files = await _output_files(session, meta["remote_output_dir"])
    except Exception as exc:
        logger.error("failed to read agent output files after retries: %s", exc)
        return [0.0]
    try:
        reference_values = await _read_bytes_with_retry(session, meta["reference_values_file"])
    except Exception as exc:
        logger.error("failed to read reference_values.json after retries: %s", exc)
        return [0.0]
    result = evaluate_output_bundle(
        output_files=output_files, reference_values_bytes=reference_values
    )
    logger.info("evaluation result: %s", json.dumps(result, sort_keys=True))
    return [float(result.get("score", 0.0))]


if __name__ == "__main__":
    for task in load():
        print(task.description)
