"""``task_data_source: gs://<bucket>`` — pull task data from a GCS bucket.

Behavior:

* ``stage_input``: pull ``<gcs_prefix>/input`` and ``<gcs_prefix>/software``
  to the sandbox. **Skip if already on the sandbox** (image-baked data
  intact) — saves cold-start rsync cost on large datasets.
* ``stage_reference``: always wipe + fresh rsync. Reference is the eval
  truth; we don't trust any baked / partial state on the sandbox.
* Fresh Linux downloads restore GCS native symlink placeholders after mode
  normalization, using gsutil metadata and the sandbox's existing Python.

GCS auth: the images carry NO baked credential and the GCE metadata SA is
not provisioned on them, so all in-VM gsutil/gcloud calls authenticate via
the SA key the provider injects at run time (``gcs_sa_key`` → pushed into the
VM, surfaced as ``sandbox.metadata['gcs_key_path']`` and added by ``_gsutil``
as ``-o Credentials:gs_service_key_file=...``). Reads (staging/reference) and
writes (output push) share this one path.
"""
from __future__ import annotations

import logging
import re
import shlex
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import baked_in_sandbox, join, shell_q, task_subdir

logger = logging.getLogger(__name__)


async def stage_input(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    gcs_prefix = _gcs_prefix(source, task_data)
    base = task_subdir(sandbox, task_data)
    await sandbox.mkdir(base)
    versioned = re.search(r"/v\d+\.\d+(?:\.\d+)?$", source.rstrip("/")) is not None

    staged: list[str] = []
    for subdir in ("input", "software"):
        dst = join(sandbox, base, subdir)
        if not versioned and await _has_baked_files(sandbox, dst):
            logger.info("gsbucket: %s already present on sandbox, skipping rsync", dst)
            staged.append(f"{subdir}(baked)")
            continue
        src = f"{gcs_prefix}/{subdir}"
        if versioned:
            await sandbox.rm([dst])
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
        await _restore_symlinks(sandbox, src, dst)
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

    archive_source = f"{gcs_prefix}/reference.7z"
    if await _gcs_exists(sandbox, archive_source):
        archive = join(sandbox, base, "reference.7z")
        await sandbox.mkdir(base)
        await sandbox.rm([archive])
        command = (
            f"{_gsutil(sandbox)} cp {shell_q(sandbox, archive_source)} "
            f"{shell_q(sandbox, archive)}"
        )
        if not sandbox.is_linux:
            command = (
                'powershell -NoProfile -Command "'
                f"{command}; exit $LASTEXITCODE"
                '"'
            )
        result = await sandbox.run_command(command, timeout=600)
        if result.returncode:
            raise RuntimeError(
                f"gsutil encrypted reference download failed (rc={result.returncode})"
            )
        report = await baked_in_sandbox.stage_reference(sandbox, task_data, source=source)
        return {**report, "source": source}

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
    await _restore_symlinks(sandbox, src, dst)
    return {"staged": ["reference"], "source": source}


# ---- helpers ----

_RESTORE_SYMLINKS = r'''
import base64
from contextlib import ExitStack
import hashlib
import os
from pathlib import PurePosixPath
import stat
import subprocess
import sys
import uuid

source, destination, *gsutil = sys.argv[1:]
prefix = source.rstrip('/') + '/'
links = []
current = None
no_matches = False
unexpected_output = False
with subprocess.Popen(gsutil + ['stat', prefix + '**'],
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, encoding='utf-8') as listing:
    for line in listing.stdout:
        if line.startswith('gs://') and line.endswith(':\n'):
            current = {'url': line[:-2]}
        elif current is not None:
            if line.startswith('    Content-Length:'):
                current['size'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('    Hash (md5):'):
                current['md5'] = line.split(':', 1)[1].strip()
            elif line.startswith('        goog-reserved-file-is-symlink:'):
                if line.split(':', 1)[1].strip().lower() == 'true':
                    links.append(current)
        elif line.strip() == 'No URLs matched: ' + prefix + '**':
            no_matches = True
        elif line.strip():
            unexpected_output = True
    returncode = listing.wait()
    if returncode and not (returncode == 1 and current is None
                           and no_matches and not unexpected_output):
        raise RuntimeError('gsutil symlink metadata listing failed')

with ExitStack() as resources:
    pending = []
    seen = set()
    for link in links:
        if not link['url'].startswith(prefix):
            raise ValueError('Symlink object outside requested prefix')
        relative = link['url'][len(prefix):]
        parts = relative.split('/')
        if any(part in ('', '.', '..') for part in parts) or relative in seen:
            raise ValueError('Unsafe or duplicate symlink object path')
        seen.add(relative)
        size = link.get('size', 0)
        if not 0 < size <= 4096 or not link.get('md5'):
            raise ValueError('Invalid symlink placeholder size or missing hash')
        root = PurePosixPath(destination)
        if not root.is_absolute() or '..' in root.parts:
            raise ValueError('Symlink destination must be an absolute directory')
        directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        try:
            for component in list(root.parts[1:]) + parts[:-1]:
                child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory)
                os.close(directory)
                directory = child
        except BaseException:
            os.close(directory)
            raise
        resources.callback(os.close, directory)
        name = parts[-1]
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            payload = os.readlink(name, dir_fd=directory).encode('utf-8')
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory)
            with os.fdopen(descriptor, 'rb') as placeholder:
                opened = os.fstat(placeholder.fileno())
                if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    raise ValueError('Symlink placeholder changed during restore')
                payload = placeholder.read(4097)
        else:
            raise ValueError('Symlink placeholder is not a regular file or symlink')
        digest = base64.b64encode(hashlib.md5(payload).digest()).decode('ascii')
        if len(payload) != size or digest != link['md5']:
            raise ValueError('Symlink placeholder does not match GCS metadata')
        target = payload.decode('utf-8')
        if not target or '\0' in target:
            raise ValueError('Invalid symlink target')
        if not stat.S_ISLNK(info.st_mode):
            pending.append((directory, name, target, info))
    for directory, name, target, original in pending:
        info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        fields = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(info, field) != getattr(original, field) for field in fields):
            raise ValueError('Symlink placeholder changed during restore')
        temporary = '.ale-symlink-' + uuid.uuid4().hex
        os.symlink(target, temporary, dir_fd=directory)
        try:
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        except BaseException:
            os.unlink(temporary, dir_fd=directory)
            raise
'''


async def _restore_symlinks(sandbox: SandboxHandle, src: str, dst: str) -> None:
    """Restore marked, hash-checked placeholders without reading link targets."""
    if not sandbox.is_linux:
        return
    command = shlex.join([
        sandbox.python, "-c", _RESTORE_SYMLINKS, src, dst, *shlex.split(_gsutil(sandbox)),
    ])
    result = await sandbox.run_command(command, timeout=600)
    if result.returncode:
        raise RuntimeError(
            f"GCS symlink restore failed (rc={result.returncode}): "
            f"{(result.stderr or '')[-300:]}"
        )


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
