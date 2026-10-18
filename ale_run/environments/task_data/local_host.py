"""Stage host task data into local Docker or Daytona containers.

For the local Docker provider the host holds the task data (fetched once, e.g.
from Hugging Face) in the canonical layout:

    <host_dir>/<domain>/<task>/<variant>/{input, software, reference}

``stage_input`` copies ``input/`` (+ ``software/``) into the container before the
agent runs. ``stage_reference`` copies ``reference/`` in AFTER the agent, just
before ``evaluate`` — so the reference (the answers) is never inside the
container while the agent runs, exactly like the gs:// path. No encryption: the
answer is hidden by *timing*, not a password, so the host data can be plain
(no reference.7z).

Docker uses ``docker cp``. Daytona streams a tar archive through its SDK, with
ownership assigned to the sandbox user while retaining modes and symlinks.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import tarfile
import tempfile
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import join, prepare_task_subdir, shell_q, task_subdir

logger = logging.getLogger(__name__)

_PREFIX = "local:"


def _host_task_dir(source: str, task_data: TaskDataSpec) -> str:
    """Host-side ``<root>/<domain>/<task>/<variant>`` for ``local:<root>``."""
    root = source[len(_PREFIX):]
    return os.path.join(
        root,
        task_data.domain_name or "",
        task_data.task_name or "",
        task_data.variant_name or "",
    )


async def _docker_cp(src: str, sandbox: SandboxHandle, dst: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "docker", "cp", src, f"{sandbox.id}:{dst}",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker cp {src} -> {sandbox.id}:{dst} failed "
            f"(rc={proc.returncode}): {err.decode(errors='replace')[:300]}"
        )
    owner = []
    for flag in ("-u", "-g"):
        result = await sandbox.run_command(f"id {flag}")
        value = (result.stdout or "").strip()
        if result.returncode or not value.isascii() or not value.isdecimal():
            raise RuntimeError("Could not determine task-data user/group in Docker sandbox")
        owner.append(value)
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", "--user", "0", sandbox.id,
        "chown", "-R", "-h", "--", ":".join(owner), dst,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker task-data ownership repair failed for {sandbox.id}:{dst} "
            f"(rc={proc.returncode}): {err.decode(errors='replace')[:300]}"
        )


async def _daytona_copy(src: str, sandbox: SandboxHandle, dst: str) -> None:
    """Transfer a directory with POSIX modes and symlinks intact."""
    with tempfile.TemporaryDirectory(prefix="ale-task-data-") as tmp:
        archive = os.path.join(tmp, "data.tar")

        def pack() -> None:
            with tarfile.open(archive, "w", dereference=False) as tar:
                tar.add(src, arcname=".")

        await asyncio.to_thread(pack)
        remote = join(sandbox, sandbox.work_dir_base, f"task-data-{os.path.basename(tmp)}.tar")
        try:
            await sandbox.mkdir(sandbox.work_dir_base)
            await sandbox.upload_local_file(archive, remote)
            result = await sandbox.run_command(
                f"mkdir -p {shlex.quote(dst)} && tar -xf {shlex.quote(remote)} "
                f"--no-same-owner -C {shlex.quote(dst)}", timeout=3600,
            )
            if result.returncode:
                raise RuntimeError(f"Daytona task-data extraction failed: {result.stderr}")
        finally:
            await sandbox.rm([remote])


async def stage_input(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    """Copy input/ (+ software/) from the host into the container; make output/."""
    provider = sandbox.metadata.get("provider")
    if provider not in ("docker", "daytona"):
        raise NotImplementedError(
            "task_data_source=local requires the Docker or Daytona provider"
        )
    host = _host_task_dir(source, task_data)
    if not os.path.isdir(os.path.join(host, "input")):
        raise RuntimeError(
            f"task_data_source=local: expected input/ at {host!r} on the host, "
            f"not found. Did you fetch the task data? (see the local-docker README)"
        )
    base = await prepare_task_subdir(sandbox, task_data)
    # Make the target dirs first, then ``docker cp <src>/. <dst>`` copies the
    # CONTENTS into them — correct whether or not the dir pre-exists (a baked
    # image may already have <base>/input; plain `cp src dst` would nest it).
    in_dst = join(sandbox, base, "input")
    await sandbox.mkdir(in_dst)
    await sandbox.mkdir(join(sandbox, base, "output"))
    copy = _docker_cp if provider == "docker" else _daytona_copy
    await copy(os.path.join(host, "input") + "/.", sandbox, in_dst)
    staged = ["input"]
    sw = os.path.join(host, "software")
    if os.path.isdir(sw):
        sw_dst = join(sandbox, base, "software")
        await sandbox.mkdir(sw_dst)
        await copy(sw + "/.", sandbox, sw_dst)
        # mirror baked_in_sandbox: make software wrappers/binaries executable.
        result = await sandbox.run_command(
            f"find {shell_q(sandbox, join(sandbox, base, 'software'))} "
            f"-type f -exec chmod +x {{}} +",
            timeout=60,
        )
        if result.returncode:
            raise RuntimeError(f"Could not make staged software executable: {result.stderr}")
        staged.append("software")
    logger.info("local: staged %s for %s from %s", staged, task_data.task_name, host)
    return {"staged": staged, "source": "local"}


async def stage_reference(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    """Copy reference/ from the host into the container — AFTER the agent, just
    before evaluate, so the agent never sees the answers during its run."""
    provider = sandbox.metadata.get("provider")
    if provider not in ("docker", "daytona"):
        raise NotImplementedError(
            "task_data_source=local requires the Docker or Daytona provider"
        )
    host = _host_task_dir(source, task_data)
    ref = os.path.join(host, "reference")
    base = task_subdir(sandbox, task_data)
    archive = os.path.join(host, "reference.7z")
    if not os.path.isdir(ref) and provider == "daytona" and os.path.isfile(archive):
        from . import baked_in_sandbox

        target_archive = join(sandbox, base, "reference.7z")
        await sandbox.upload_local_file(archive, target_archive)
        report = await baked_in_sandbox.stage_reference(sandbox, task_data, source=source)
        return {**report, "source": "local"}
    if not os.path.isdir(ref):
        return {"skipped": True, "reason": "no_reference"}
    target = join(sandbox, base, "reference")
    await sandbox.rm([target])  # defend against stale reference from a prior run
    copy = _docker_cp if provider == "docker" else _daytona_copy
    await copy(ref, sandbox, target)
    logger.info("local: staged reference %s -> %s:%s", ref, sandbox.id, target)
    return {"staged": ["reference"], "source": "local"}
