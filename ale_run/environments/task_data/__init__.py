"""Task data staging — pull input/software onto the sandbox, ready
reference for evaluation.

Three sources today, dispatched on yaml ``artifacts_path.task_data_source``:

  ``"baked_in_sandbox"``     image already has input/ + reference.7z baked in
  ``"gs://<bucket>"``        rsync from a GCS bucket
  ``"hf://<dataset>"``       HuggingFace Hub (STUB)

Each backend module exposes two coroutines:

  ``async def stage_input(sandbox, task_data, *, source) -> dict``
  ``async def stage_reference(sandbox, task_data, *, source) -> dict``

The lifecycle calls :func:`select` to pick the module by scheme, then
invokes its two functions in the right phases.
"""
from __future__ import annotations

import shlex
from typing import Any, Protocol

from ...base_interface import SandboxHandle, TaskDataSpec


class _Backend(Protocol):
    async def stage_input(
        self, sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
    ) -> dict[str, Any]: ...

    async def stage_reference(
        self, sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
    ) -> dict[str, Any]: ...


def select(task_data_source: str):
    """Pick a backend by scheme of ``task_data_source``."""
    if task_data_source == "baked_in_sandbox":
        from . import baked_in_sandbox
        return baked_in_sandbox
    if task_data_source.startswith("gs://"):
        from . import gsbucket
        return gsbucket
    if task_data_source.startswith("hf://"):
        from . import huggingface
        return huggingface
    raise ValueError(
        f"unknown task_data_source {task_data_source!r}: expected "
        f"'baked_in_sandbox', 'gs://<bucket>', or 'hf://<dataset>'"
    )


# ============================================================================
# Shared helpers
# ============================================================================

def task_subdir(sandbox: SandboxHandle, task_data: TaskDataSpec) -> str:
    """``<task_data_root>/<domain>/<task>/<variant>`` with OS-native separator."""
    sep = "/" if sandbox.is_linux else "\\"
    return sep.join([
        sandbox.task_data_root.rstrip("/\\"),
        task_data.domain_name,
        task_data.task_name,
        task_data.variant_name,
    ])


def join(sandbox: SandboxHandle, *parts: str) -> str:
    """OS-native path join. ``parts[0]`` is the root."""
    sep = "/" if sandbox.is_linux else "\\"
    head = parts[0].rstrip("/\\")
    tail = sep.join(p.strip("/\\") for p in parts[1:])
    return f"{head}{sep}{tail}" if tail else head


def shell_q(sandbox: SandboxHandle, path: str) -> str:
    """Path quoter — POSIX shlex on linux, single-quote literal on windows
    (PowerShell single-quoted strings have no interpolation)."""
    return shlex.quote(path) if sandbox.is_linux else f"'{path}'"


__all__ = ["select", "task_subdir", "join", "shell_q"]
