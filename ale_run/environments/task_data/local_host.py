"""``task_data_source: local:<host_dir>`` — stage task data from a HOST directory
into the Docker container via ``docker cp``.

For the local Docker provider the host holds the task data (fetched once, e.g.
from Hugging Face) in the canonical layout:

    <host_dir>/<domain>/<task>/<variant>/{input, software, reference}

``stage_input`` copies ``input/`` (+ ``software/``) into the container before the
agent runs. ``stage_reference`` copies ``reference/`` in AFTER the agent, just
before ``evaluate`` — so the reference (the answers) is never inside the
container while the agent runs, exactly like the gs:// path. No encryption: the
answer is hidden by *timing*, not a password, so the host data can be plain
(no reference.7z).

Docker-only: it shells out to ``docker cp`` against the running container, whose
name is ``sandbox.id``. ``docker cp`` moves a whole directory tree in one fast
local-disk operation (no per-file HTTP, unlike ``upload_local_file``).
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import join, shell_q, task_subdir

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


async def _docker_cp(src: str, container: str, dst: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "docker", "cp", src, f"{container}:{dst}",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker cp {src} -> {container}:{dst} failed "
            f"(rc={proc.returncode}): {err.decode(errors='replace')[:300]}"
        )


async def stage_input(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    """Copy input/ (+ software/) from the host into the container; make output/."""
    if sandbox.metadata.get("provider") != "docker":
        raise NotImplementedError(
            "task_data_source=local is currently supported only by the direct "
            "Docker provider. QEMU guests need a host-to-container share plus "
            "guest-side CIFS/Samba mounting, which is not implemented yet."
        )
    host = _host_task_dir(source, task_data)
    if not os.path.isdir(os.path.join(host, "input")):
        raise RuntimeError(
            f"task_data_source=local: expected input/ at {host!r} on the host, "
            f"not found. Did you fetch the task data? (see the local-docker README)"
        )
    base = task_subdir(sandbox, task_data)
    # Make the target dirs first, then ``docker cp <src>/. <dst>`` copies the
    # CONTENTS into them — correct whether or not the dir pre-exists (a baked
    # image may already have <base>/input; plain `cp src dst` would nest it).
    in_dst = join(sandbox, base, "input")
    await sandbox.mkdir(in_dst)
    await sandbox.mkdir(join(sandbox, base, "output"))
    await _docker_cp(os.path.join(host, "input") + "/.", sandbox.id, in_dst)
    staged = ["input"]
    sw = os.path.join(host, "software")
    if os.path.isdir(sw):
        sw_dst = join(sandbox, base, "software")
        await sandbox.mkdir(sw_dst)
        await _docker_cp(sw + "/.", sandbox.id, sw_dst)
        # mirror baked_in_sandbox: make software wrappers/binaries executable.
        await sandbox.run_command(
            f"find {shell_q(sandbox, join(sandbox, base, 'software'))} "
            f"-type f -exec chmod +x {{}} +",
            timeout=60,
        )
        staged.append("software")
    logger.info("local: staged %s for %s from %s", staged, task_data.task_name, host)
    return {"staged": staged, "source": "local"}


async def stage_reference(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    """Copy reference/ from the host into the container — AFTER the agent, just
    before evaluate, so the agent never sees the answers during its run."""
    if sandbox.metadata.get("provider") != "docker":
        raise NotImplementedError(
            "task_data_source=local is currently supported only by the direct "
            "Docker provider. QEMU guest staging is not implemented yet."
        )
    host = _host_task_dir(source, task_data)
    ref = os.path.join(host, "reference")
    if not os.path.isdir(ref):
        return {"skipped": True, "reason": "no_reference"}
    base = task_subdir(sandbox, task_data)
    target = join(sandbox, base, "reference")
    await sandbox.rm([target])  # defend against stale reference from a prior run
    await _docker_cp(ref, sandbox.id, target)
    logger.info("local: staged reference %s -> %s:%s", ref, sandbox.id, target)
    return {"staged": ["reference"], "source": "local"}
