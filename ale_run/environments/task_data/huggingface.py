"""``task_data_source: hf://<dataset>`` — pull task data from HuggingFace Hub.

STUB. Implement alongside the first task family that uses HF as source.
"""
from __future__ import annotations

from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec


_STUB = (
    "huggingface task_data source not yet implemented. "
    "Add the download/extract logic in this module when a task wants it."
)


async def stage_input(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    raise NotImplementedError(_STUB)


async def stage_reference(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    raise NotImplementedError(_STUB)
