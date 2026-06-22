"""Pull the agent's output off the sandbox after the run.

Dispatched by the lifecycle on ``artifacts_path.output_path``:

  None         → skip; output stays on the sandbox and is lost on teardown
  ``"local"``  → :func:`pull_to_host` (cua HTTP, files pulled concurrently; a
                 local Docker sandbox takes a one-shot ``docker cp`` fast path)
  ``"gs://X"`` → :func:`push_to_gcs` (VM-side gsutil; nothing on host)
"""
from __future__ import annotations

import asyncio
import logging
import shlex
from pathlib import Path
from typing import Any

from ..base_interface import SandboxHandle, TaskDataSpec

logger = logging.getLogger(__name__)

# Max concurrent per-file downloads in pull_to_host. Each download is itself
# chunked (see download_to_local), so this bounds the number of in-flight cua
# RPCs, not the payload size.
_PULL_CONCURRENCY = 8


def _output_dir(sandbox: SandboxHandle, task_data: TaskDataSpec) -> str:
    sep = "/" if sandbox.is_linux else "\\"
    return sep.join([
        sandbox.task_data_root.rstrip("/\\"),
        task_data.domain_name,
        task_data.task_name,
        task_data.variant_name,
        "output",
    ])


def _docker_container(sandbox: SandboxHandle) -> str | None:
    """Container name iff this is a local Docker sandbox, else None.

    The direct Docker provider stamps both ``provider=docker`` and
    ``container_name`` into metadata. The QEMU provider also has an outer
    container, but that container does not expose the guest filesystem, so it
    must use the normal cua download path."""
    if sandbox.metadata.get("provider") != "docker":
        return None
    return sandbox.metadata.get("container_name")


async def _pull_via_docker_cp(
    container: str, src: str, dest_dir: Path,
) -> dict[str, Any]:
    """``output_path == 'local'`` fast path for the docker provider: copy the
    whole output tree off the container in one host-side ``docker cp`` — no
    per-file base64 round-trip over cua-server :5000. Symmetric with the
    ``local:`` task-data source's docker-cp input staging.

    ``docker cp <c>:<src>/. <dest>`` copies the *contents* of ``src`` into
    ``dest`` (so the run dir holds the files directly, not a nested ``output/``).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "docker", "cp", f"{container}:{src.rstrip('/')}/.", str(dest_dir),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker cp {container}:{src} -> {dest_dir} failed "
            f"(rc={proc.returncode}): {err.decode(errors='replace')[:300]}"
        )

    files = 0
    total_bytes = 0
    for p in dest_dir.rglob("*"):
        if p.is_file():
            files += 1
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass
    logger.info(
        "pull_to_host(docker cp): %s → %s (files=%d bytes=%d)",
        src, dest_dir, files, total_bytes,
    )
    return {
        "transport": "docker-cp",
        "vm_path": src,
        "files": files,
        "bytes": total_bytes,
        "errors": [],
    }


async def pull_to_host(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, dest_dir: Path,
) -> dict[str, Any]:
    """``output_path == 'local'`` — walk + per-file download to host run dir."""
    src = _output_dir(sandbox, task_data)
    entries = await sandbox.list_dir(src)
    if not entries:
        return {"skipped": True, "reason": "empty_or_missing", "vm_path": src}

    # Local Docker sandbox: one host-side `docker cp` for the whole tree instead
    # of the per-file cua transport below. Best-effort — fall back to cua on any
    # failure so a docker quirk never loses output.
    container = _docker_container(sandbox)
    if container:
        try:
            return await _pull_via_docker_cp(container, src, dest_dir)
        except Exception as e:
            logger.warning("docker cp output pull failed, falling back to cua: %s", e)

    dest_dir.mkdir(parents=True, exist_ok=True)
    sep = "/" if sandbox.is_linux else "\\"

    # Materialise the directory tree first (cheap, ordering-sensitive), then
    # download the files concurrently — one slow/large file no longer blocks the
    # rest, and many small files no longer serialise into a long tail.
    jobs: list[tuple[str, Path]] = []
    for entry in entries:
        rel = entry["relpath"]
        if entry.get("is_dir"):
            (dest_dir / rel.replace("\\", "/")).mkdir(parents=True, exist_ok=True)
            continue
        remote_path = f"{src.rstrip(sep)}{sep}{rel}"
        local_path = dest_dir / rel.replace("\\", "/")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((remote_path, local_path))

    sem = asyncio.Semaphore(_PULL_CONCURRENCY)

    async def _fetch(remote_path: str, local_path: Path) -> tuple[str, Path, bool]:
        async with sem:
            ok = await sandbox.download_to_local(
                remote_path, str(local_path), timeout=120,
            )
        return remote_path, local_path, ok

    results = await asyncio.gather(*(_fetch(rp, lp) for rp, lp in jobs))

    files = 0
    total_bytes = 0
    errors: list[dict[str, str]] = []
    for remote_path, local_path, ok in results:
        if ok:
            files += 1
            try:
                total_bytes += local_path.stat().st_size
            except OSError:
                pass
        else:
            errors.append({"vm_path": remote_path, "error": "download_failed"})
            marker = local_path.with_suffix(local_path.suffix + ".unreadable")
            marker.write_text(f"vm_path={remote_path}\nreason=download_failed\n")

    logger.info(
        "pull_to_host: %s → %s (files=%d bytes=%d errors=%d)",
        src, dest_dir, files, total_bytes, len(errors),
    )
    return {
        "transport": "cua",
        "vm_path": src,
        "files": files,
        "bytes": total_bytes,
        "errors": errors,
    }


async def push_to_gcs(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *,
    run_id: str, bucket: str,
) -> dict[str, Any]:
    """``output_path == 'gs://...'`` — VM-side gsutil push.

    cp -r preserves the trailing src dir name (``output``) under dst, so
    dst is the run prefix; final landing is ``<bucket>/<run_id>/output/``.

    Uses ``gsutil`` with the injected SA key (via ``_gsutil`` — same path the
    read/staging side uses) rather than ``gcloud storage cp``. The VMs carry NO
    baked credential: ``gcloud``'s ambient auth falls back to the GCE metadata
    SA, which isn't provisioned on these images, so ``gcloud storage cp`` fails
    with a metadata-server token error. The injected key authenticates writes
    consistently with reads.
    """
    from .task_data.gsbucket import _gsutil

    src = _output_dir(sandbox, task_data)
    run_prefix = f"{bucket.rstrip('/')}/{run_id}/"
    gcs_dst = f"{run_prefix}output/"
    gsutil = _gsutil(sandbox)

    if sandbox.is_linux:
        cmd = f"{gsutil} -m cp -r {shlex.quote(src)} {shlex.quote(run_prefix)}"
    else:
        cmd = (
            'powershell -NoProfile -Command "'
            f"{gsutil} -m cp -r '{src}' '{run_prefix}'"
            '"'
        )
    logger.info("push_to_gcs: %s → %s", src, gcs_dst)
    r = await sandbox.run_command(cmd, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"gsutil cp failed (rc={r.returncode}): "
            f"{(r.stderr or '')[:300]}"
        )
    return {"transport": "gcs", "gcs_path": gcs_dst}
