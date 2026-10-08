"""Native FreeCAD CAM task with LinuxCNC replay and verified stock simulation."""

import os
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.engineering.gcode.native_task import evaluate_native_task, stage_native_task
from tasks.linux_runtime import LinuxTaskConfig


VARIANTS = [
    # Existing 15 variants
    ("125162_319", "125162-319-NCFM-T"),
    ("A125117_301", "A125117-301-NCSM-B"),
    ("A125138_301", "A125138-301-NCFM-T"),
    ("A125138_302", "A125138-302-NCSM-B"),
    ("MDBZDHZJ25_SKC_1_NCSM_T", "MDBZDHZJ25_SKC-1_NCSM_T"),
    ("MR250692C00_M2", "MR250692C00-M2-NCFM-T"),
    ("MR250696C00_F1", "MR250696C00-F1-NCSM-B"),
    ("MR250696C00_S5", "MR250696C00-S5-NCSM-T"),
    ("MR250697C00_M1", "MR250697C00-M1-NCFM-B"),
    ("MR250697C00_S1", "MR250697C00-S1-NCSM-B"),
    ("MR250697C00_S2", "MR250697C00-S2-NCFM-T"),
    ("MR250698C00_F3", "MR250698C00-F3-NCSM-B"),
    ("MR250698C00_P6", "MR250698C00-P6-NCSM-B"),
    ("MR250698C00_U005", "MR250698C00-U005-NCFM-L"),
    ("T29153_050", "T29153-050-NCRM-F"),
    # New 4 variants (added 2026-03-25)
    ("MDB240386_S2", "MDB240386-S2-NCSM-T"),
    ("MM250645B00_S2", "MM250645B00-S2-NCFM-T"),
    ("MM250645B00_S3", "MM250645B00-S3-NCSM-F"),
    # NOTE: MM250689C00_M1 excluded — raw data has no .pmlprj (not a valid PM project)
]


ASSETS = Path(__file__).with_name("assets")


@dataclass
class GCodeTaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "engineering"
    TASK_NAME: str = "gcode"
    VARIANT_NAME: str = "125162_319"
    PM_PROJECT_NAME: str = ""

    @property
    def output_project(self):
        return f"{self.remote_output_dir}/machining.FCStd"

    @property
    def task_description(self):
        return (
            (ASSETS / "instruction.md")
            .read_text()
            .format(
                variant_name=self.VARIANT_NAME,
                input_dir=self.input_dir,
                output_dir=self.remote_output_dir,
                software_dir=self.software_dir,
            )
        )

    def to_metadata(self):
        return {
            **super().to_metadata(),
            "output_project": self.output_project,
            "cam_runtime_config": os.environ.get(
                "ALE_CAM_RUNTIME_CONFIG", "/opt/ale-cam/runtime.json"
            ),
            "release_status": (
                "native_entrypoint_passed_awaiting_agent_review"
                if self.VARIANT_NAME == "125162_319"
                else "awaiting_variant_data_validation"
            ),
        }


@cb.tasks_config(split="train")
def load():
    tasks = []
    for tag, project in VARIANTS:
        config = GCodeTaskConfig(VARIANT_NAME=tag, PM_PROJECT_NAME=project)
        tasks.append(
            cb.Task(
                description=config.task_description,
                metadata=config.to_metadata(),
                computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
            )
        )
    return tasks


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await BaseTaskSetup()(task_cfg, session)
    await stage_native_task(task_cfg.metadata, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    return await evaluate_native_task(task_cfg.metadata, session)
