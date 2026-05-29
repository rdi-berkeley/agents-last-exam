"""``task_data_source: gs://<bucket>`` — pull task data from a GCS bucket.

Behavior:

* ``stage_input``: pull ``<gcs_prefix>/input`` and ``<gcs_prefix>/software``
  to the sandbox. **Skip if already on the sandbox** (image-baked data
  intact) — saves cold-start rsync cost on large datasets.
* ``stage_reference``: always wipe + fresh rsync. Reference is the eval
  truth; we don't trust any baked / partial state on the sandbox.

GCS auth is assumed pre-configured on the sandbox (ambient SA via
metadata server, or ``gcloud auth`` done at image-bake time). If you
need to push a SA key at run time, do it out-of-band.
"""
from __future__ import annotations

import logging
import shlex
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import join, shell_q, task_subdir

logger = logging.getLogger(__name__)


async def stage_input(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    gcs_prefix = _gcs_prefix(source, task_data)
    base = task_subdir(sandbox, task_data)
    await sandbox.mkdir(base)

    staged: list[str] = []
    for subdir in ("input", "software"):
        dst = join(sandbox, base, subdir)
        if await _has_baked_files(sandbox, dst):
            logger.info("gsbucket: %s already present on sandbox, skipping rsync", dst)
            staged.append(f"{subdir}(baked)")
            continue
        src = f"{gcs_prefix}/{subdir}"
        if not await _gcs_exists(sandbox, src):
            await sandbox.mkdir(dst)
            continue
        r = await sandbox.run_command(_rsync_cmd(sandbox, src, dst), timeout=600)
        if r.returncode != 0:
            raise RuntimeError(
                f"gsutil rsync {subdir} failed (rc={r.returncode}): "
                f"{(r.stderr or '')[:300]}"
            )
        if subdir == "software" and sandbox.is_linux:
            await sandbox.run_command(
                f"find {shlex.quote(dst)} -type f -exec chmod +x {{}} +",
                timeout=60,
            )
        staged.append(subdir)

    await sandbox.mkdir(join(sandbox, base, "output"))
    return {"staged": staged, "source": source}


async def stage_reference(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    gcs_prefix = _gcs_prefix(source, task_data)
    base = task_subdir(sandbox, task_data)
    src = f"{gcs_prefix}/reference"
    dst = join(sandbox, base, "reference")

    if not await _gcs_exists(sandbox, src):
        return {"skipped": True, "reason": "no_reference_on_gcs"}

    await sandbox.rm([dst])
    r = await sandbox.run_command(_rsync_cmd(sandbox, src, dst), timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"gsutil rsync reference failed (rc={r.returncode}): "
            f"{(r.stderr or '')[:300]}"
        )
    return {"staged": ["reference"], "source": source}


# ---- helpers ----

def _gcs_prefix(source: str, task_data: TaskDataSpec) -> str:
    return (
        f"{source.rstrip('/')}/{task_data.domain_name}/"
        f"{task_data.task_name}/{task_data.variant_name}"
    )


async def _has_baked_files(sandbox: SandboxHandle, path: str) -> bool:
    """True if path exists and contains at least one regular file."""
    if not await sandbox.exists(path):
        return False
    entries = await sandbox.list_dir(path)
    return any(not e["is_dir"] for e in entries)


async def _gcs_exists(sandbox: SandboxHandle, gs_url: str) -> bool:
    cmd = (
        f"gsutil ls '{gs_url}' >/dev/null 2>&1" if sandbox.is_linux
        else f"powershell -NoProfile -Command \"gsutil ls '{gs_url}' *> $null; exit $LASTEXITCODE\""
    )
    r = await sandbox.run_command(cmd, timeout=30)
    return r.returncode == 0


def _rsync_cmd(sandbox: SandboxHandle, src: str, dst: str) -> str:
    if sandbox.is_linux:
        return (
            f"mkdir -p {shlex.quote(dst)} && "
            f"gsutil -m rsync -r {shlex.quote(src)} {shlex.quote(dst)}"
        )
    return (
        'powershell -NoProfile -Command "'
        f"New-Item -ItemType Directory -Force -Path '{dst}' | Out-Null; "
        f"gsutil -m rsync -r '{src}' '{dst}'"
        '"'
    )
