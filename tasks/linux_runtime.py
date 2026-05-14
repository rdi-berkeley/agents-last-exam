"""Shared Linux runtime helpers for Ubuntu-native tasks.

Migrated verbatim from ``agenthle/tasks/linux_runtime.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

from tasks.common_config import GeneralTaskConfig

DATA_ROOT = "/media/user/data/agenthle"


@dataclass
class LinuxTaskConfig(GeneralTaskConfig):
    """Base config for Ubuntu-native tasks."""

    DOMAIN_NAME: str = ""
    OS_TYPE: str = "linux"
    VARIANT_NAME: str = "base"

    @property
    def task_dir(self) -> str:
        return f"{DATA_ROOT}/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}"

    @property
    def data_task_dir(self) -> str:
        return self.task_dir

    @property
    def input_dir(self) -> str:
        return f"{self.task_dir}/input"

    @property
    def reference_dir(self) -> str:
        return f"{self.task_dir}/reference"

    @property
    def eval_dir(self) -> str:
        return f"{self.task_dir}/eval_data"

    @property
    def software_dir(self) -> str:
        return f"{self.task_dir}/software"

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.REMOTE_OUTPUT_DIR}"

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_dir": self.task_dir,
                "data_task_dir": self.data_task_dir,
                "input_dir": self.input_dir,
                "reference_dir": self.reference_dir,
                "eval_dir": self.eval_dir,
                "software_dir": self.software_dir,
                "remote_output_dir": self.remote_output_dir,
            }
        )
        return metadata
