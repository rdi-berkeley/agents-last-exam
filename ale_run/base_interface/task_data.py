"""Task-side data shapes.

Currently just :class:`TaskDataSpec` — the contract describing what
input/reference data a task needs staged into the VM. Read by
:mod:`ale_run.environments.data_staging` to drive ``stage_input`` /
``stage_reference``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class TaskDataSpec:
    """Where a task's data lives and what to stage where.

    Populated by ``TaskLoader`` from ``task_card.json`` + task module
    metadata. Consumed by ``data_staging`` (pushes into the VM) and
    indirectly by the orchestrator (decides whether to fetch a GCS
    prefix, etc.).

    ``sandbox_task_name`` changes only the sandbox directory name;
    ``task_name`` and ``source_relpath`` retain the canonical source identity.
    """

    requires_task_data: bool = False
    domain_name: str | None = None
    task_name: str | None = None
    variant_name: str | None = None
    source_relpath: str | None = None
    input_dir: str | None = None
    software_dir: str | None = None
    reference_dir: str | None = None
    reference_gcs_prefix: str | None = None
    remote_output_dir: str | None = None
    sandbox_task_name: str | None = None

    def __post_init__(self) -> None:
        if self.sandbox_task_name is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", self.sandbox_task_name
        ):
            raise ValueError("sandbox_task_name must be a plain task directory name")
