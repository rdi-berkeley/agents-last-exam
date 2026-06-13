"""``task_data_source: s3://<bucket>`` — pull task data from an S3 bucket.

Mirror of :mod:`ale_run.environments.task_data.gsbucket`, using the in-box
``aws s3`` CLI instead of ``gsutil``.

Behavior (identical to gsbucket):

* ``stage_input``: pull ``<s3_prefix>/input`` and ``<s3_prefix>/software`` to
  the sandbox. **Skip if already on the sandbox** (image-baked data intact).
* ``stage_reference``: always wipe + fresh sync. Reference is eval truth.

S3 auth: EC2 instances launched by :class:`AwsProvider` carry an **IAM instance
profile**, so the in-box ``aws`` CLI is already authenticated — there is no
credential to inject (contrast gcloud, which pushes an SA key into each VM).
The image must therefore have the ``aws`` CLI on PATH; the AwsProvider AMIs
bake it in. If a snapshot uses requester-pays-style buckets, add
``--request-payer requester`` in ``_sync_cmd`` (off by default; S3 has no
GCS-style mandatory user-project).
"""
from __future__ import annotations

import logging
import shlex
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import join, task_subdir

logger = logging.getLogger(__name__)


async def stage_input(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    s3_prefix = _s3_prefix(source, task_data)
    base = task_subdir(sandbox, task_data)
    await sandbox.mkdir(base)

    staged: list[str] = []
    for subdir in ("input", "software"):
        dst = join(sandbox, base, subdir)
        if await _has_baked_files(sandbox, dst):
            logger.info("s3bucket: %s already present on sandbox, skipping sync", dst)
            staged.append(f"{subdir}(baked)")
            continue
        src = f"{s3_prefix}/{subdir}"
        if not await _s3_exists(sandbox, src):
            await sandbox.mkdir(dst)
            continue
        r = await sandbox.run_command(_sync_cmd(sandbox, src, dst), timeout=600)
        if r.returncode != 0:
            raise RuntimeError(
                f"aws s3 sync {subdir} failed (rc={r.returncode}): "
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
    s3_prefix = _s3_prefix(source, task_data)
    base = task_subdir(sandbox, task_data)
    src = f"{s3_prefix}/reference"
    dst = join(sandbox, base, "reference")

    if not await _s3_exists(sandbox, src):
        return {"skipped": True, "reason": "no_reference_on_s3"}

    await sandbox.rm([dst])
    r = await sandbox.run_command(_sync_cmd(sandbox, src, dst), timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"aws s3 sync reference failed (rc={r.returncode}): "
            f"{(r.stderr or '')[:300]}"
        )
    # aws s3 sync does not preserve POSIX mode bits; normalize like gsbucket so
    # grading sees predictable perms.
    if sandbox.is_linux:
        await sandbox.run_command(f"chmod -R 777 {shlex.quote(dst)}", timeout=60)
    return {"staged": ["reference"], "source": source}


# ---- helpers ----


def _s3_prefix(source: str, task_data: TaskDataSpec) -> str:
    return (
        f"{source.rstrip('/')}/{task_data.domain_name}/"
        f"{task_data.task_name}/{task_data.variant_name}"
    )


async def _has_baked_files(sandbox: SandboxHandle, path: str) -> bool:
    if not await sandbox.exists(path):
        return False
    entries = await sandbox.list_dir(path)
    return any(not e["is_dir"] for e in entries)


async def _s3_exists(sandbox: SandboxHandle, s3_url: str) -> bool:
    # `aws s3 ls <prefix>/` exits 0 with output only if the prefix has objects;
    # exits 1 (or 0/empty) otherwise. Trailing slash forces prefix semantics.
    url = s3_url.rstrip("/") + "/"
    if sandbox.is_linux:
        cmd = f"aws s3 ls {shlex.quote(url)} >/dev/null 2>&1"
    else:
        cmd = (
            "powershell -NoProfile -Command \""
            f"aws s3 ls '{url}' *> $null; exit $LASTEXITCODE\""
        )
    r = await sandbox.run_command(cmd, timeout=30)
    return r.returncode == 0


def _sync_cmd(sandbox: SandboxHandle, src: str, dst: str) -> str:
    if sandbox.is_linux:
        return (
            f"mkdir -p {shlex.quote(dst)} && "
            f"aws s3 sync {shlex.quote(src)} {shlex.quote(dst)}"
        )
    return (
        'powershell -NoProfile -Command "'
        f"New-Item -ItemType Directory -Force -Path '{dst}' | Out-Null; "
        f"aws s3 sync '{src}' '{dst}'"
        '"'
    )
