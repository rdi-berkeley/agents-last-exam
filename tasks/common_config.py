"""Common configuration for AgentHLE tasks."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

_TASKS_ROOT = Path(__file__).resolve().parent


@dataclass
class GeneralTaskConfig:
    """Base configuration for Windows tasks (default OS).

    Primary fields follow the canonical domain/task/variant hierarchy:
      - DOMAIN_NAME: top-level task family. One of:
            engineering, physical_sciences, life_sciences, health_medicine,
            psychology_neuro, business_finance, legal, visual_media,
            computing_math, transport_safety, education_info, agriculture_env,
            social_sciences, other
      - TASK_NAME:   task implementation id within the domain (e.g. "taxform_6_1")
      - VARIANT_NAME: one concrete runnable case (e.g. "variant_1" or "base")

    Paths use Windows backslash convention rooted at ``REMOTE_ROOT_DIR``
    (default ``E:\\agenthle``, overridable via the ``REMOTE_ROOT_DIR`` env var).
    """

    # Global settings
    REMOTE_OUTPUT_DIR: str = os.environ.get("REMOTE_OUTPUT_DIR", "output")
    REMOTE_ROOT_DIR: str = os.environ.get("REMOTE_ROOT_DIR", r"E:\agenthle")
    DOMAIN_NAME: str = ""
    TASK_NAME: str = ""
    VARIANT_NAME: str = ""
    OS_TYPE: str = os.environ.get("OS_TYPE", "windows")
    REQUIRES_TASK_DATA: bool = True

    @property
    def task_description(self) -> str:
        """Task description for the agent."""
        return ""

    @property
    def task_dir(self) -> str:
        """Generate task directory based on domain/task/variant."""
        return rf"{self.REMOTE_ROOT_DIR}\{self.DOMAIN_NAME}\{self.TASK_NAME}\{self.VARIANT_NAME}"

    @property
    def input_dir(self) -> str:
        """Agent-visible input directory."""
        return rf"{self.task_dir}\input"

    @property
    def software_dir(self) -> str:
        """Generate software directory."""
        return rf"{self.task_dir}\software"

    @property
    def remote_output_dir(self) -> str:
        """Output directory."""
        return rf"{self.task_dir}\{self.REMOTE_OUTPUT_DIR}"

    @property
    def reference_dir(self) -> str:
        """Reference directory."""
        return rf"{self.task_dir}\reference"

    @property
    def task_card_path(self) -> Path:
        """Operator-side path to this task's task_card.json."""
        return _TASKS_ROOT / self.DOMAIN_NAME / self.TASK_NAME / "task_card.json"

    @property
    def required_credentials(self) -> list[dict]:
        """Agent-side credentials this task needs at run time.

        Auto-derived from ``task_card.json``'s ``requiredCredentials`` field.
        Empty list when the task declares none. See
        ``docs/task_impl_guides/admin/stage1/07_CREDENTIALS_AND_LICENSES.md``
        for schema.
        """
        card = self.task_card_path
        if not card.exists():
            return []
        try:
            data = json.loads(card.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return data.get("requiredCredentials") or []

    def to_metadata(self) -> dict:
        """Convert config to metadata dict for cua_bench Task."""
        meta = {
            "domain_name": self.DOMAIN_NAME,
            "task_name": self.TASK_NAME,
            "variant_name": self.VARIANT_NAME,
            "requires_task_data": self.REQUIRES_TASK_DATA,
            "task_dir": self.task_dir,
            "input_dir": self.input_dir,
            "software_dir": self.software_dir,
            "remote_output_dir": self.remote_output_dir,
            "reference_dir": self.reference_dir,
        }
        creds = self.required_credentials
        if creds:
            meta["required_credentials"] = creds
        return meta
