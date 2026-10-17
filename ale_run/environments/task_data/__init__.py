"""Task data staging — pull input/software onto the sandbox, ready
reference for evaluation.

Sources dispatched on yaml ``task_data_source``:

  ``"baked_in_sandbox"``     image already has input/ + reference.7z baked in
  ``"gs://<bucket>"``        rsync from a GCS bucket
  ``"s3://<bucket>"``        sync from an S3 bucket
  ``"oss://<bucket>"``       sync from an Alibaba Cloud OSS bucket
  ``"hf://<dataset>"``       HuggingFace Hub (STUB)
  ``"local:<host_dir>"``     docker cp from a HOST dir — for a data-less Docker
                             image: input before the agent, reference before eval

Each backend module exposes two coroutines:

  ``async def stage_input(sandbox, task_data, *, source) -> dict``
  ``async def stage_reference(sandbox, task_data, *, source) -> dict``

The lifecycle calls :func:`select` to pick the module by scheme, then
invokes its two functions in the right phases.
"""
from __future__ import annotations

import shlex
from dataclasses import replace
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
    if task_data_source.startswith("s3://"):
        from . import s3bucket
        return s3bucket
    if task_data_source.startswith("oss://"):
        from . import ossbucket
        return ossbucket
    if task_data_source.startswith("hf://"):
        from . import huggingface
        return huggingface
    if task_data_source.startswith("local:"):
        from . import local_host
        return local_host
    raise ValueError(
        f"unknown task_data_source {task_data_source!r}: expected "
        f"'baked_in_sandbox', 'gs://<bucket>', 's3://<bucket>', 'oss://<bucket>', "
        f"'hf://<dataset>', "
        f"'local:<dir>'"
    )


# ============================================================================
# Shared helpers
# ============================================================================

def task_subdir(sandbox: SandboxHandle, task_data: TaskDataSpec) -> str:
    """Sandbox destination, independent of the canonical source identity."""
    sep = "/" if sandbox.is_linux else "\\"
    return sep.join([
        sandbox.task_data_root.rstrip("/\\"),
        task_data.domain_name,
        task_data.sandbox_task_name or task_data.task_name,
        task_data.variant_name,
    ])


async def prepare_task_subdir(sandbox: SandboxHandle, task_data: TaskDataSpec) -> str:
    """Apply an opt-in Linux workspace alias before staging or solver access."""
    base = task_subdir(sandbox, task_data)
    if not task_data.sandbox_task_name or task_data.sandbox_task_name == task_data.task_name:
        return base
    if not sandbox.is_linux:
        raise NotImplementedError("Sandbox task aliases currently require Linux")
    canonical = task_subdir(sandbox, replace(task_data, sandbox_task_name=None))
    script = """
import os
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:])
for candidate in (source, destination):
    if any(parent.is_symlink() for parent in (candidate, *candidate.parents)):
        raise RuntimeError('Task alias paths must not traverse symlinks')
    if candidate.exists() and not candidate.is_dir():
        raise RuntimeError('Task alias path is not a directory')
if source.exists():
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError('Neutral task directory is not empty')
    if os.path.ismount(source):
        raise RuntimeError('Cannot alias a mounted task directory')
    os.rename(source, destination)
"""
    result = await sandbox.run_command(
        shlex.join([
            sandbox.python, "-c", script,
            canonical.rsplit("/", 1)[0], base.rsplit("/", 1)[0],
        ]),
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(f"Could not prepare neutral task workspace: {result.stderr}")
    return base


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


__all__ = ["select", "task_subdir", "prepare_task_subdir", "join", "shell_q"]
