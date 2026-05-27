from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.psychology_neuro._shared.cognitive_science.neuro_runtime import (
    ensure_neuro_runtime,
    evaluate_single_task,
    load_single_task,
)

TASK_NAME = "scene3_skullstrip_qc"
TASK_TITLE = "Skull-stripping QC selection in FSLeyes"
TASK_DIR = Path(__file__).resolve().parent


@cb.tasks_config(split="train")
def load():
    return load_single_task(TASK_NAME, TASK_TITLE, domain_name="health_medicine")


class _NeuroSetup(BaseTaskSetup):
    async def setup(self, task_cfg, session):
        await ensure_neuro_runtime(session, task_cfg.metadata["task_name"])


_setup = _NeuroSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    return await evaluate_single_task(task_cfg, session, TASK_DIR)
