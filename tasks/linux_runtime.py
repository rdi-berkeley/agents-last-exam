"""Shared Linux runtime helpers for Ubuntu-native tasks."""

import os
from dataclasses import dataclass

from tasks.common_config import GeneralTaskConfig, _UNSET_DATA_ROOT

# Back-compat module constant: several task main.py files still
# `from tasks.linux_runtime import DATA_ROOT`. LinuxTaskConfig itself uses the
# _UNSET_DATA_ROOT sentinel (resolved at runtime from REMOTE_ROOT_DIR).
DATA_ROOT = os.environ.get("REMOTE_ROOT_DIR", "/media/user/data/agenthle")


@dataclass
class LinuxTaskConfig(GeneralTaskConfig):
    """Base config for Ubuntu-native tasks.
    """

    DOMAIN_NAME: str = ""
    OS_TYPE: str = "linux"
    VARIANT_NAME: str = "base"
    REMOTE_ROOT_DIR: str = _UNSET_DATA_ROOT

    @property
    def task_dir(self) -> str:
        return f"{self.REMOTE_ROOT_DIR}/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}"

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
    def software_dir(self) -> str:
        return f"{self.task_dir}/software"

    @property
    def remote_output_dir(self) -> str:
        return f"{self.task_dir}/{self.REMOTE_OUTPUT_DIR}"

    def to_metadata(self) -> dict:
        # Parent to_metadata() already pushes task_dir / input_dir /
        # software_dir / reference_dir / remote_output_dir via self.<prop>,
        # which resolves to the POSIX overrides above. Only add the keys
        # that are LinuxTaskConfig-specific.
        metadata = super().to_metadata()
        metadata.update(
            {
                "data_task_dir": self.data_task_dir,
            }
        )
        return metadata
