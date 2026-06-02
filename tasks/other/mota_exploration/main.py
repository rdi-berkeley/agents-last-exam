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
Context: A small game studio plans to re-implement the legacy Flash dungeon-crawler Magic Tower (mota-24) on a modern stack. Before any port code is written, the porting workflow needs a clean set of per-floor reference screenshots from the original build — these will be the visual ground truth downstream engineers diff against while re-creating tile maps, HUD, and per-floor enemy/item placement.

You are doing the reference-capture pass for floors 1–3.

Steps:
1. Open `{self.game_url}` in Ruffle yourself — e.g. double-click the file in Explorer, or run `Start-Process '{self.game_url}'` in PowerShell. Wait for the title / loading screens and enter the actual game world.
2. Play forward through floors 1 → 3 using normal movement and interaction. No cheats and no save-state hacks — this is what a porting engineer would do to surface each floor honestly.
3. On arrival at each new floor, capture a full-window screenshot of the game and save it to "{self.remote_output_dir}\\$FLOOR_NUMBER$.png", where $FLOOR_NUMBER$ is the integer floor index (1.png, 2.png, 3.png).

Acceptance: one PNG per floor reached, named by floor number, depicting the correct floor — each capture will be diffed against a held-out reference frame of the same floor from the original build.
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

    from tasks.utils.evaluation import compare_screenshots_game

    comparison_criteria = "- Is the player on the same floor number?"

    return await compare_screenshots_game(
        target_image_bytes=target_image_bytes,
        reference_image_bytes=reference_image_bytes,
        context_description=f"floor {floor_number}",
        comparison_criteria=comparison_criteria
    )


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    from tasks.utils.evaluation import evaluate_milestone_mode

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
