"""OS log permission guard task."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup = BaseTaskSetup()

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

logger = logging.getLogger(__name__)

TASK_DIR_LOCAL = Path(__file__).resolve().parent
SCORE_SCRIPT = (TASK_DIR_LOCAL / "scripts" / "score_os_log_permissions.py").read_text(
    encoding="utf-8"
)


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "computing_math"
    TASK_NAME: str = "os_log_permission_guard_v1"
    VARIANT_NAME: str = "base"

    @property
    def final_state_json(self) -> str:
        return f"{self.output_dir}/final_state.json"

    @property
    def sandbox_dir(self) -> str:
        return f"{self.output_dir}/sandbox_fs"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

## Task Directory
`{self.task_dir}`

## Visible Inputs
- Filesystem snapshot: `{self.input_dir}/fs_snapshot.tar.gz`
- Ownership metadata: `{self.input_dir}/ownership.csv`
- Initial permission metadata: `{self.input_dir}/permissions.csv`
- Active-writer exclusion list: `{self.input_dir}/active_writers.json`
- Workspace setup helper: `{self.input_dir}/setup_workspace.sh`
- Detailed instructions: `{self.input_dir}/task_instructions.md`

## Your Task
Update the sandboxed `/var/logs` file permissions while respecting the safety
constraints in the metadata files.

Use this workflow:
1. Read `{self.input_dir}/task_instructions.md`.
2. Run `{self.input_dir}/setup_workspace.sh "{self.output_dir}"` to create
   a writable workspace at `{self.sandbox_dir}`.
3. Change every regular `.log` file under `{self.sandbox_dir}/var/logs` to mode
   `444`, except files owned by `syslog` in `ownership.csv` and files listed in
   `active_writers.json`.
4. Do not modify non-`.log` files. Do not change file contents.
5. Do not change ownership/group metadata in your final state.
6. Save final metadata to `{self.final_state_json}`.

The active-writer list is a provided safety record for this benchmark. Treat
listed files as actively written and leave them unchanged; do not terminate or
interfere with any processes.

## Output Contract
`{self.final_state_json}` must be valid JSON with a top-level `files` list or
mapping. Include one entry for each file listed in `ownership.csv`, with path,
type, owner, group, and final mode.

Do not write outputs outside `{self.output_dir}`.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.pop("software_dir", None)
        metadata.update(
            {
                "final_state_json": self.final_state_json,
                "sandbox_dir": self.sandbox_dir,
                "reference_state_json": f"{self.reference_dir}/reference_state.json",
            }
        )
        return metadata


config = TaskConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    eval_dir = f"/tmp/agenthle_eval/{meta['task_name']}"
    score_script = f"{eval_dir}/score_os_log_permissions.py"
    try:
        await session.interface.create_dir(eval_dir)
        await session.write_file(score_script, SCORE_SCRIPT)
        result = await session.run_command(
            f'python "{score_script}" '
            f'--input "{meta["input_dir"]}" '
            f'--output "{meta["output_dir"]}" '
            f'--reference "{meta["reference_dir"]}"'
        )
        stdout = result.get("stdout", "")
        if result.get("return_code", 1) != 0:
            logger.warning("score script failed: %s", result.get("stderr", ""))
            return [0.0]
        report = json.loads(stdout)
        logger.info("score report: %s", report)
        return [float(report.get("score", 0.0))]
    except Exception as exc:
        logger.error("evaluation failed: %s", exc)
        return [0.0]
