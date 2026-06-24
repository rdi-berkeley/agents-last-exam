"""Pull the agent's output off the sandbox after the run.

Dispatched by the lifecycle on ``artifacts_path.output_path``:

  None         → skip; output stays on the sandbox and is lost on teardown
  ``"local"``  → :func:`pull_to_host` (provider-local fast path with cua HTTP
                 fallback)
  ``"gs://X"`` → :func:`push_to_gcs` (VM-side gsutil; nothing on host)
"""

from __future__ import annotations

import asyncio
import errno
import logging
import shlex
import shutil
from pathlib import Path
from typing import Any

from ..base_interface import SandboxHandle, TaskDataSpec

logger = logging.getLogger(__name__)

# Max concurrent per-file downloads in pull_to_host. Each download is itself
# chunked (see download_to_local), so this bounds the number of in-flight cua
# RPCs, not the payload size.
_PULL_CONCURRENCY = 8
_QEMU_SHARE_COPY_TIMEOUT_S = 3600
_GCS_PUSH_TIMEOUT_S = 3600


def _output_dir(sandbox: SandboxHandle, task_data: TaskDataSpec) -> str:
    sep = "/" if sandbox.is_linux else "\\"
    return sep.join(
        [
            sandbox.task_data_root.rstrip("/\\"),
            task_data.domain_name,
            task_data.task_name,
            task_data.variant_name,
            "output",
        ]
    )


def _docker_container(sandbox: SandboxHandle) -> str | None:
    """Container name iff this is a local Docker sandbox, else None.

    The direct Docker provider stamps both ``provider=docker`` and
    ``container_name`` into metadata. The QEMU provider also has an outer
    container, but that container does not expose the guest filesystem to
    ``docker cp``."""
    if sandbox.metadata.get("provider") != "docker":
        return None
    return sandbox.metadata.get("container_name")


def _qemu_exchange(sandbox: SandboxHandle) -> tuple[Path, str] | None:
    if sandbox.metadata.get("provider") != "qemu":
        return None
    host_dir = sandbox.metadata.get("exchange_host_dir")
    guest_share = sandbox.metadata.get("exchange_guest_share")
    if not host_dir or not guest_share:
        return None
    return Path(host_dir), str(guest_share)


def _count_output_files(dest_dir: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    for path in dest_dir.rglob("*"):
        if not path.is_file():
            continue
        files += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
    return files, total_bytes


async def _pull_via_docker_cp(
    container: str,
    src: str,
    dest_dir: Path,
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
        "docker",
        "cp",
        f"{container}:{src.rstrip('/')}/.",
        str(dest_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker cp {container}:{src} -> {dest_dir} failed "
            f"(rc={proc.returncode}): {err.decode(errors='replace')[:300]}"
        )

    files, total_bytes = _count_output_files(dest_dir)
    logger.info(
        "pull_to_host(docker cp): %s → %s (files=%d bytes=%d)",
        src,
        dest_dir,
        files,
        total_bytes,
    )
    return {
        "transport": "docker-cp",
        "vm_path": src,
        "files": files,
        "bytes": total_bytes,
        "errors": [],
    }


def _promote_qemu_output(staged_dir: Path, dest_dir: Path) -> None:
    if not staged_dir.is_dir():
        raise RuntimeError(f"QEMU shared output staging directory is missing: {staged_dir}")
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if not dest_dir.exists():
        try:
            staged_dir.replace(dest_dir)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
    shutil.copytree(staged_dir, dest_dir, dirs_exist_ok=True)
    shutil.rmtree(staged_dir)


async def _pull_via_qemu_share(
    sandbox: SandboxHandle,
    src: str,
    exchange_dir: Path,
    guest_share: str,
    dest_dir: Path,
) -> dict[str, Any]:
    staged_dir = exchange_dir / "output"
    if staged_dir.exists():
        await asyncio.to_thread(shutil.rmtree, staged_dir)

    if sandbox.is_linux:
        source = shlex.quote(src)
        share = shlex.quote(guest_share)
        command = (
            "set -e; "
            f"test -d {source} || exit 44; "
            "mount_dir=$(mktemp -d); "
            'cleanup() { sudo -n umount "$mount_dir" 2>/dev/null || true; '
            'rmdir "$mount_dir" 2>/dev/null || true; }; '
            "trap cleanup EXIT; "
            f'sudo -n mount -t cifs {share} "$mount_dir" -o guest,vers=3.0; '
            'sudo -n mkdir -p "$mount_dir/output"; '
            f'sudo -n cp -rL {source}/. "$mount_dir/output/"'
        )
    else:
        source = src.replace("'", "''")
        destination = f"{guest_share.rstrip('/\\')}\\output".replace("'", "''")
        command = (
            'powershell -NoProfile -Command "'
            f"$src='{source}'; $dst='{destination}'; "
            "if (-not (Test-Path -LiteralPath $src -PathType Container)) { exit 44 }; "
            "New-Item -ItemType Directory -Force -Path $dst | Out-Null; "
            "& robocopy $src $dst /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 "
            "/NFL /NDL /NJH /NJS /NP /XJ; "
            "if ($LASTEXITCODE -ge 8) { exit $LASTEXITCODE } else { exit 0 }"
            '"'
        )

    result = await sandbox.run_command(command, timeout=_QEMU_SHARE_COPY_TIMEOUT_S)
    if result.returncode == 44:
        return {"skipped": True, "reason": "empty_or_missing", "vm_path": src}
    if result.returncode != 0:
        raise RuntimeError(
            f"QEMU shared output copy failed (rc={result.returncode}): "
            f"{(result.stderr or result.stdout or '')[:300]}"
        )

    await asyncio.to_thread(_promote_qemu_output, staged_dir, dest_dir)
    files, total_bytes = await asyncio.to_thread(_count_output_files, dest_dir)
    logger.info(
        "pull_to_host(qemu share): %s → %s (files=%d bytes=%d)",
        src,
        dest_dir,
        files,
        total_bytes,
    )
    return {
        "transport": "qemu-share",
        "vm_path": src,
        "files": files,
        "bytes": total_bytes,
        "errors": [],
    }


async def pull_to_host(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    dest_dir: Path,
) -> dict[str, Any]:
    """Copy the sandbox output tree into the host run directory."""
    src = _output_dir(sandbox, task_data)

    exchange = _qemu_exchange(sandbox)
    if exchange:
        try:
            return await _pull_via_qemu_share(
                sandbox,
                src,
                exchange[0],
                exchange[1],
                dest_dir,
            )
        except Exception as exc:
            logger.warning(
                "QEMU shared output pull failed, falling back to cua: %s",
                exc,
            )

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
                remote_path,
                str(local_path),
                timeout=120,
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
        src,
        dest_dir,
        files,
        total_bytes,
        len(errors),
    )
    return {
        "transport": "cua",
        "vm_path": src,
        "files": files,
        "bytes": total_bytes,
        "errors": errors,
    }


async def push_to_gcs(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    run_id: str,
    bucket: str,
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
        cmd = f"powershell -NoProfile -Command \"{gsutil} -m cp -r '{src}' '{run_prefix}'\""
    logger.info("push_to_gcs: %s → %s", src, gcs_dst)
    r = await sandbox.run_command(cmd, timeout=_GCS_PUSH_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(f"gsutil cp failed (rc={r.returncode}): {(r.stderr or '')[:300]}")
    return {"transport": "gcs", "gcs_path": gcs_dst}
