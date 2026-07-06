"""``task_data_source: oss://<bucket>`` — pull task data from an Alibaba Cloud
OSS bucket.

Mirror of :mod:`ale_run.environments.task_data.s3bucket` / ``gsbucket``, using
the in-box ``ossutil`` CLI instead of ``aws s3`` / ``gsutil``.

Behavior (identical to the others):

* ``stage_input``: pull ``<oss_prefix>/input`` and ``<oss_prefix>/software`` to
  the sandbox. **Skip if already on the sandbox** (image-baked data intact).
* ``stage_reference``: always wipe + fresh sync. Reference is eval truth.

OSS auth: ECS instances launched by :class:`~ale_run.environments.providers.aliyun.AliyunProvider`
carry an **instance RAM role** (``ram_role_name``), so the in-box ``ossutil``
authenticates via STS credentials pulled from instance metadata — there is no
key to inject (contrast gcloud, which pushes an SA key into each VM; like aws,
which relies on the instance profile). The image must therefore have ``ossutil``
on PATH AND a baked ``~/.ossutilconfig`` that sets ``mode=EcsRamRole`` +
``ramRoleName=<role>`` + the region OSS ``endpoint`` (use the
``oss-<region>-internal.aliyuncs.com`` endpoint from inside ECS to avoid public
egress); the AliyunProvider images bake both.
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
    oss_prefix = _oss_prefix(source, task_data)
    base = task_subdir(sandbox, task_data)
    await sandbox.mkdir(base)

    staged: list[str] = []
    for subdir in ("input", "software"):
        dst = join(sandbox, base, subdir)
        if await _has_baked_files(sandbox, dst):
            logger.info("ossbucket: %s already present on sandbox, skipping sync", dst)
            staged.append(f"{subdir}(baked)")
            continue
        src = f"{oss_prefix}/{subdir}"
        if not await _oss_exists(sandbox, src):
            await sandbox.mkdir(dst)
            continue
        r = await sandbox.run_command(_sync_cmd(sandbox, src, dst), timeout=600)
        if r.returncode != 0:
            raise RuntimeError(
                f"ossutil sync {subdir} failed (rc={r.returncode}): "
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
    oss_prefix = _oss_prefix(source, task_data)
    base = task_subdir(sandbox, task_data)
    src = f"{oss_prefix}/reference"
    dst = join(sandbox, base, "reference")

    if not await _oss_exists(sandbox, src):
        return {"skipped": True, "reason": "no_reference_on_oss"}

    await sandbox.rm([dst])
    r = await sandbox.run_command(_sync_cmd(sandbox, src, dst), timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"ossutil sync reference failed (rc={r.returncode}): "
            f"{(r.stderr or '')[:300]}"
        )
    # ossutil sync does not preserve POSIX mode bits; normalize like gsbucket so
    # grading sees predictable perms.
    if sandbox.is_linux:
        await sandbox.run_command(f"chmod -R 777 {shlex.quote(dst)}", timeout=60)
    return {"staged": ["reference"], "source": source}


# ---- helpers ----


def _oss_prefix(source: str, task_data: TaskDataSpec) -> str:
    return (
        f"{source.rstrip('/')}/{task_data.domain_name}/"
        f"{task_data.task_name}/{task_data.variant_name}"
    )


async def _has_baked_files(sandbox: SandboxHandle, path: str) -> bool:
    if not await sandbox.exists(path):
        return False
    entries = await sandbox.list_dir(path)
    return any(not e["is_dir"] for e in entries)


# The task-data bucket is requester-pays (the puller's account is billed, not
# the owner's — mirrors gcloud's ale-data-public / aws's requester-pays bucket).
# Every read must carry --payer requester or OSS returns AccessDenied.
_RP = "--payer requester"


async def _oss_exists(sandbox: SandboxHandle, oss_url: str) -> bool:
    """True if the prefix has at least one object.

    ``ossutil ls`` exits 0 even for an empty prefix (unlike ``aws s3 ls``), so —
    rather than rely on the exit code — we ask for at most one object and parse
    the ``Object Number is: N`` summary line ossutil prints."""
    url = oss_url.rstrip("/") + "/"
    if sandbox.is_linux:
        cmd = f"ossutil ls {_RP} {shlex.quote(url)} --limited-num 1"
    else:
        cmd = (
            "powershell -NoProfile -Command \""
            f"ossutil ls {_RP} '{url}' --limited-num 1\""
        )
    r = await sandbox.run_command(cmd, timeout=30)
    if r.returncode != 0:
        return False
    out = (r.stdout or "")
    # "Object Number is: 0" → empty; any object line starts with the oss:// url.
    if "Object Number is: 0" in out:
        return False
    return "oss://" in out or "Object Number is:" in out and "Object Number is: 0" not in out


def _sync_cmd(sandbox: SandboxHandle, src: str, dst: str) -> str:
    # ossutil sync mirrors `aws s3 sync` / `gsutil rsync`: it copies the prefix
    # tree under src into dst. Trailing slashes force directory semantics.
    src = src.rstrip("/") + "/"
    if sandbox.is_linux:
        return (
            f"mkdir -p {shlex.quote(dst)} && "
            f"ossutil sync {_RP} {shlex.quote(src)} {shlex.quote(dst)}"
        )
    return (
        'powershell -NoProfile -Command "'
        f"New-Item -ItemType Directory -Force -Path '{dst}' | Out-Null; "
        f"ossutil sync {_RP} '{src}' '{dst}'"
        '"'
    )
