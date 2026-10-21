"""AgentHLE task: frustrated_spin_triangle_yang_baxter_holonomy."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
from tasks.physical_sciences.frustrated_spin_triangle_yang_baxter_holonomy.scripts.build_fixtures import (
    CALCULUS_SPEC,
    TASK_SPEC,
    build_fixtures,
)
from tasks.physical_sciences.frustrated_spin_triangle_yang_baxter_holonomy.scripts.verify_outputs import (
    score_submission,
)

logger = logging.getLogger(__name__)

DOMAIN_NAME = "physical_sciences"
TASK_NAME = "frustrated_spin_triangle_yang_baxter_holonomy"
VARIANT_NAME = "base"
VERIFY_SCRIPT_PATH = Path(__file__).resolve().parent / "scripts" / "verify_outputs.py"

_setup = BaseTaskSetup()


@dataclass
class FrustratedSpinTriangleConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    REQUIRES_TASK_DATA: bool = False

    @property
    def task_spec_path(self) -> str:
        return f"{self.input_dir}/TASK.md"

    @property
    def calculus_path(self) -> str:
        return f"{self.input_dir}/calculus.md"

    @property
    def scenarios_path(self) -> str:
        return f"{self.input_dir}/scenarios.json"

    @property
    def check_path(self) -> str:
        return f"{self.input_dir}/check.py"

    @property
    def answers_path(self) -> str:
        return f"{self.remote_output_dir}/answers.json"

    @property
    def notes_path(self) -> str:
        return f"{self.remote_output_dir}/NOTES.md"

    @property
    def task_description(self) -> str:
        return f"""\
You are working on a Linux VM.

Construct local-frame and exchange-drive pulse schedules for 12 frustrated
three-spin quantum-device scenarios. The controller uses four calibrated frames
per spin, fixed pulse timing, and one active exchange bond per pulse. Cancel
bond-resolved leakage and all specified multichannel noise coefficients.

Read these visible files first:

- Task and output contract: `{self.task_spec_path}`
- Mathematical specification: `{self.calculus_path}`
- Numerical operators and targets: `{self.scenarios_path}`
- Standalone forward verifier: `{self.check_path}`

Write your solution to `{self.answers_path}`. You may run:

```
python3 {self.check_path} --answers {self.answers_path}
```

When finished, write `{self.notes_path}` with a short account of what you tried,
what worked, what failed, and any assumptions or ambiguities.

Do not modify files under `{self.input_dir}`. Write only under
`{self.remote_output_dir}`.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": f"{DOMAIN_NAME}/{TASK_NAME}",
                "task_spec_path": self.task_spec_path,
                "calculus_path": self.calculus_path,
                "scenarios_path": self.scenarios_path,
                "check_path": self.check_path,
                "answers_path": self.answers_path,
                "notes_path": self.notes_path,
            }
        )
        return metadata


config = FrustratedSpinTriangleConfig()


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
    meta = task_cfg.metadata

    await session.run_command(
        f"rm -rf {meta['input_dir']!r} {meta['remote_output_dir']!r} {meta['reference_dir']!r}",
        check=True,
    )
    await session.interface.create_dir(meta["input_dir"])
    await session.interface.create_dir(meta["remote_output_dir"])

    scenarios, _, _, _ = build_fixtures()
    await session.write_file(meta["task_spec_path"], TASK_SPEC)
    await session.write_file(meta["calculus_path"], CALCULUS_SPEC)
    await session.write_file(
        meta["scenarios_path"], json.dumps(scenarios, indent=2, sort_keys=True)
    )
    await session.write_file(meta["check_path"], VERIFY_SCRIPT_PATH.read_text(encoding="utf-8"))
    await session.run_command(f"chmod 755 {meta['check_path']!r}", check=False)

    if await session.file_exists(meta["reference_dir"]) or await session.directory_exists(
        meta["reference_dir"]
    ):
        raise RuntimeError("reference data must not exist during agent execution")
    logger.info("staged spin-triangle inputs at %s", meta["input_dir"])


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    answers_path = task_cfg.metadata["answers_path"]
    if not await session.file_exists(answers_path):
        logger.info("missing answers file: %s", answers_path)
        return [0.0]

    raw_submission = await session.read_file(answers_path)
    try:
        submission = json.loads(raw_submission)
    except (ValueError, RecursionError) as exc:
        logger.info("invalid answers JSON: %s", exc)
        return [0.0]

    scenarios, _, _, _ = build_fixtures()
    result = score_submission(scenarios, submission)

    logger.info(
        "score=%.6f passed=%d/%d details=%s",
        result.score,
        result.correct,
        result.total,
        json.dumps(result.results, sort_keys=True),
    )
    return [float(result.score)]
