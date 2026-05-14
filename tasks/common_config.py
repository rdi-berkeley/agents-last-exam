"""Common configuration for AgentHLE tasks.

Migrated verbatim from ``agenthle/tasks/common_config.py``. Existing tasks
that ``from tasks.common_config import GeneralTaskConfig`` work unchanged.
"""

import os
from dataclasses import dataclass


@dataclass
class GeneralTaskConfig:
    """Base configuration for tasks.

    Primary fields follow the canonical domain/task/variant hierarchy:
      - DOMAIN_NAME: top-level task family (e.g. "manufacturing")
      - TASK_NAME:   task implementation id within the domain (e.g. "2dto3d")
      - VARIANT_NAME: one concrete runnable case (e.g. "32300A_000001" or "base")
    """

    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")
    REMOTE_ROOT_DIR: str = os.environ.get("REMOTE_ROOT_DIR", r"E:\agenthle")
    DOMAIN_NAME: str = ""
    TASK_NAME: str = ""
    VARIANT_NAME: str = ""
    OS_TYPE: str = os.environ.get("OS_TYPE", "windows")
    REQUIRES_TASK_DATA: bool = True

    @property
    def task_description(self) -> str:
        return ""

    @property
    def task_dir(self) -> str:
        return rf"{self.REMOTE_ROOT_DIR}\{self.DOMAIN_NAME}\{self.TASK_NAME}\{self.VARIANT_NAME}"

    @property
    def software_dir(self) -> str:
        return rf"{self.task_dir}\software"

    @property
    def remote_output_dir(self) -> str:
        return rf"{self.task_dir}\{self.REMOTE_OUTPUT_DIR}"

    @property
    def reference_dir(self) -> str:
        return rf"{self.task_dir}\reference"

    def to_metadata(self) -> dict:
        return {
            "domain_name": self.DOMAIN_NAME,
            "task_name": self.TASK_NAME,
            "variant_name": self.VARIANT_NAME,
            "requires_task_data": self.REQUIRES_TASK_DATA,
            "task_dir": self.task_dir,
            "software_dir": self.software_dir,
            "remote_output_dir": self.remote_output_dir,
            "reference_dir": self.reference_dir,
        }
