"""Pull the agent's output off the sandbox after the run.

Dispatched by the lifecycle on ``artifacts_path.output_path``:

  None         → skip; output stays on the sandbox and is lost on teardown
  ``"local"``  → :func:`pull_to_host` (cua HTTP, files pulled concurrently)
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


async def pull_to_host(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, dest_dir: Path,
) -> dict[str, Any]:
    """``output_path == 'local'`` — walk + per-file download to host run dir."""
    src = _output_dir(sandbox, task_data)
    entries = await sandbox.list_dir(src)
    if not entries:
        return {"skipped": True, "reason": "empty_or_missing", "vm_path": src}

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
