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
    # gsutil rsync does not preserve POSIX mode bits, so reference lands with
    # umask-derived perms (644 files / 755 dirs) — no exec bit. Grading is the
    # only consumer of reference and may execute scripts or expect specific
    # perms, so normalize the whole tree to 777 to remove any surprise.
    if sandbox.is_linux:
        await sandbox.run_command(
            f"chmod -R 777 {shlex.quote(dst)}",
            timeout=60,
        )
    return {"staged": ["reference"], "source": source}


# ---- helpers ----

def _gsutil(sandbox: SandboxHandle) -> str:
    """``gsutil`` invocation, with ``-u <project>`` for requester-pays buckets
    and ``-o gs_service_key_file`` when the SA key was pushed into the VM.

    Requester-pays buckets (e.g. ``gs://ale-data-public``) reject every request
    that doesn't name a billing/user project, and no boto-config knob supplies
    it — only the command-line ``-u`` flag works. The provider surfaces a usable
    project (derived from the injected SA key's ``project_id``) via
    ``sandbox.metadata['gcs_user_project']``.

    On the GCE provider the VM's baked gsutil is unauthenticated, so the
    provider also pushes the SA key into the VM and surfaces its path via
    ``sandbox.metadata['gcs_key_path']``; we add ``-o
    Credentials:gs_service_key_file=<path>`` so gsutil authenticates as the SA.
    (The docker provider instead writes /etc/boto.cfg and sets no key_path, so
    its behaviour is unchanged.) The key path is benchmark-controlled and has
    no spaces, so it is appended unquoted to compose inside the linux/windows
    command shapes below.
    """
    meta = sandbox.metadata or {}
    proj = meta.get("gcs_user_project")
    cmd = f"gsutil -u {shlex.quote(str(proj))}" if proj else "gsutil"
    key_path = meta.get("gcs_key_path")
    if key_path:
        cmd += f" -o Credentials:gs_service_key_file={key_path}"
    return cmd


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
    gsutil = _gsutil(sandbox)
    cmd = (
        f"{gsutil} ls '{gs_url}' >/dev/null 2>&1" if sandbox.is_linux
        else f"powershell -NoProfile -Command \"{gsutil} ls '{gs_url}' *> $null; exit $LASTEXITCODE\""
    )
    r = await sandbox.run_command(cmd, timeout=30)
    return r.returncode == 0


def _rsync_cmd(sandbox: SandboxHandle, src: str, dst: str) -> str:
    gsutil = _gsutil(sandbox)
    if sandbox.is_linux:
        return (
            f"mkdir -p {shlex.quote(dst)} && "
            f"{gsutil} -m rsync -r {shlex.quote(src)} {shlex.quote(dst)}"
        )
    return (
        'powershell -NoProfile -Command "'
        f"New-Item -ItemType Directory -Force -Path '{dst}' | Out-Null; "
        f"{gsutil} -m rsync -r '{src}' '{dst}'"
        '"'
    )
