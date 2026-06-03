"""Shared helpers for canonical finance annual-report task families."""

from __future__ import annotations

import os
from dataclasses import dataclass

import cua_bench as cb

from tasks.common_config import GeneralTaskConfig
from tasks.business_finance._shared.finance.finance_evaluation import win_join

DOMAIN_NAME = "business_finance"
# Re-exported for operator-side staging scripts; single source of truth is the
# base config so the data root is changed in exactly one place.
CANONICAL_REMOTE_ROOT = GeneralTaskConfig.REMOTE_ROOT_DIR
CANONICAL_GCS_ROOT = f"gs://ale-data-all/{DOMAIN_NAME}"
TARGET_VM_PROJECT = "sunblaze-4"
TARGET_VM_ZONE = "us-central1-a"
TARGET_VM_NAME = "agenthle-dev-cpu-free"

TASK_FAMILIES = (
    "ar_full",
    "ar_metric_company",
    "ar_challenge_filter",
)
TASK_SCALES = (60, 300, 1500)


@dataclass(frozen=True)
class VariantSpec:
    tag: str
    scale: int
    legacy_source_task: str


def variant_tag(scale: int) -> str:
    return f"scale_{scale}"


def variant_specs_for_task(task_name: str) -> tuple[VariantSpec, ...]:
    return tuple(
        VariantSpec(
            tag=variant_tag(scale),
            scale=scale,
            legacy_source_task=f"{task_name}_{scale}",
        )
        for scale in TASK_SCALES
    )


@dataclass
class FinanceFamilyConfig(GeneralTaskConfig):
    """Canonical config for one finance task family variant."""

    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = ""
    VARIANT_NAME: str = ""
    SCALE: int = 0
    LEGACY_SOURCE_TASK: str = ""

    @property
    def task_dir(self) -> str:
        return rf"{self.REMOTE_ROOT_DIR}\{self.DOMAIN_NAME}\{self.TASK_NAME}\{self.VARIANT_NAME}"

    @property
    def input_dir(self) -> str:
        return win_join(self.task_dir, "input")

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update(
            {
                "scale": self.SCALE,
                "legacy_source_task": self.LEGACY_SOURCE_TASK,
            }
        )
        return metadata


def build_task(config: FinanceFamilyConfig) -> cb.Task:
    return cb.Task(
        description=config.task_description,
        metadata=config.to_metadata(),
        computer={"provider": "computer", "setup_config": {"os_type": config.OS_TYPE}},
    )
