"""Mota exploration task."""

import logging
import os
from dataclasses import dataclass

import cua_bench as cb
from tasks.common_config import GeneralTaskConfig
from tasks.common_setup import BaseTaskSetup

logger = logging.getLogger(__name__)

@dataclass
class TaskConfig(GeneralTaskConfig):
    REMOTE_ROOT_DIR: str = r"E:\agenthle"
    DOMAIN_NAME: str = "other"

    TASK_NAME: str = "mota_exploration"
    VARIANT_NAME: str = "mota_24_ez"
    GAME_TAG: str = "mota-24"

    @property
    def game_url(self) -> str:
        return fr"{self.task_dir}\input\{self.GAME_TAG}.swf"

    @property
    def task_description(self) -> str:
        return f"""
Goal: Launch Magic Tower and navigate to the 3rd floor.
1. Open `{self.game_url}` in Ruffle yourself — e.g. double-click the file in Explorer, or run `Start-Process '{self.game_url}'` in PowerShell.
2. Wait for the game to load and enter the game.
3. Navigate to the 3rd floor.

Verification:
1. When you reach each new floor, take a screenshot and save it to "{self.remote_output_dir}\\$FLOOR_NUMBER$.png", where $FLOOR_NUMBER$ is the floor number you reached.
2. The task is successful if the screenshot files exist and demonstrate the floor you reached.
"""

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update({
            "game_tag": self.GAME_TAG,
            "game_url": self.game_url,
        })
        return metadata

config = TaskConfig()

MODE = os.environ.get("TASK_MODE", "DEBUG") 

@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={
                "provider": "computer",
                "setup_config": {
                    "os_type": config.OS_TYPE,
                }
            }
        )
    ]

_setup = BaseTaskSetup()


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)

async def query_milestone(
    target_image_bytes: bytes, 
    reference_image_bytes: bytes, 
    floor_number: str
) -> dict:

    from utils.evaluation import compare_screenshots_game

    comparison_criteria = "- Is the player on the same floor number?"

    return await compare_screenshots_game(
        target_image_bytes=target_image_bytes,
        reference_image_bytes=reference_image_bytes,
        context_description=f"floor {floor_number}",
        comparison_criteria=comparison_criteria
    )


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    from utils.evaluation import evaluate_milestone_mode

    remote_output_path = task_cfg.metadata["remote_output_dir"]
    reference_path = task_cfg.metadata["reference_dir"]
    task_tag = task_cfg.metadata.get("variant_name", "unknown")

    try:
        final_score, _ = await evaluate_milestone_mode(
            session=session,
            target_path=remote_output_path,
            reference_path=reference_path,
            task_tag=task_tag,
            comparison_fn=query_milestone,
            output_dir=os.environ.get("EVALUATION_OUTPUT_DIR", "./trycua/cua-bench/")
        )
        
        return [final_score]

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return [0.0]
