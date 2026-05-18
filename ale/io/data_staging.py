"""Data staging — copy task input/eval/reference from GCS onto the VM.

Async port of ``agenthle/scripts/web_console/lib/simprun/data.py``. All
gsutil invocations happen via ``session.run_command`` (cua-bench's
async RPC), wrapped in a 3-retry with 10s linear backoff. Unrecoverable
errors (matched-no-objects, bucket-not-found) skip retry.

Lifecycle (called from :class:`ale.core.env.AgenthleEnv`):

  reset_async():
    ensure_data_disk        (mount + chown /media/user/data/agenthle)
    ensure_gcs_auth         (upload SA key + gcloud activate-service-account)
    stage_input             (input/, software/, output/ dirs — visible to agent)
    stage_eval              (optional — eval scripts, visible if task wants)
    task.setup_fn           (the task's own @setup_task code runs)

  step_async(Submit):
    stage_reference         (reference/ dir — invisible during agent solve,
                             materialized on VM only here, just before evaluate)
    task.evaluate_fn

  close_async():
    upload_output           (optional — push agent's output/ to results bucket)

Visibility rule (formal benchmark): :func:`stage_reference` is gated to
the eval phase so the agent never sees ground-truth data. The task author
need not (and should not) stage reference themselves.

SA key path: read from env var ``ALE_GCS_SA_KEY_PATH``. If unset and a
task requires staging, raises clearly. Operators set this in shell once.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ale.core.cmd_result import cmd_ok, cmd_stderr, cmd_stdout
from ale.core.task_data import (
    DEFAULT_RESULTS_BUCKET,
    DEFAULT_TASK_DATA_BUCKET,
    TaskDataSpec,
    gcs_task_prefix,
    vm_subdir,
)

if TYPE_CHECKING:
    import cua_bench as cb

logger = logging.getLogger(__name__)


# =============================================================================
# Retry / detection (simprun parity)
# =============================================================================

_MAX_RETRIES = 3
_RETRY_DELAY_S = 10.0
_NO_RETRY_PATTERNS = (
    "matched no objects or files",
    "no urls matched",
    "one or more urls matched no objects",
    "bucketnotfoundexception",
    "commandexception: no such object",
)


def _is_unrecoverable(error_text: str) -> bool:
    lower = (error_text or "").lower()
    return any(p in lower for p in _NO_RETRY_PATTERNS)


async def _run_on_vm(
    session: "cb.DesktopSession", command: str, *, timeout: float = 300.0,
) -> str:
    """Run ``command`` on the VM with 3-retry + 10s linear backoff.

    Returns combined stdout on success. Raises ``RuntimeError`` on final
    failure with truncated stderr/stdout. ``_NO_RETRY_PATTERNS`` short-
    circuit the retry loop.
    """
    last_err = ""
    for attempt in range(1, _MAX_RETRIES + 1):
        cr = await session.run_command(command, timeout=timeout)
        if cmd_ok(cr):
            return cmd_stdout(cr)
        last_err = (cmd_stderr(cr) or cmd_stdout(cr) or "").strip()
        logger.warning(
            "vm cmd failed (attempt %d/%d): %s",
            attempt, _MAX_RETRIES, last_err[:200],
        )
        if _is_unrecoverable(last_err):
            break
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(_RETRY_DELAY_S)
    raise RuntimeError(
        f"vm command failed after {_MAX_RETRIES} attempts: {last_err[:500]}"
    )


# =============================================================================
# Shell builders (linux + windows)
# =============================================================================

def _gcs_cp_cmd(src: str, dst: str, os_type: str) -> str:
    if os_type == "linux":
        return f"gcloud storage cp -r '{src}' '{dst}'"
    return f'gcloud storage cp -r "{src}" "{dst}"'


def _gcs_rsync_cmd(src: str, dst: str, os_type: str) -> str:
    if os_type == "linux":
        return f"mkdir -p '{dst}' && gsutil -m rsync -r '{src}' '{dst}'"
    return (
        'powershell -NoProfile -Command "'
        f"New-Item -ItemType Directory -Force -Path '{dst}' | Out-Null; "
        f"gsutil -m rsync -r '{src}' '{dst}'"
        '"'
    )


def _gcs_ls_cmd(src: str, os_type: str) -> str:
    if os_type == "linux":
        return f"gsutil ls '{src}' >/dev/null 2>&1"
    return (
        'powershell -NoProfile -Command "'
        f"gsutil ls '{src}' *> $null; "
        "if ($LASTEXITCODE -eq 0) {{ exit 0 }} else {{ exit 1 }}"
        '"'
    )


def _mkdir_cmd(path: str, os_type: str) -> str:
    if os_type == "linux":
        return f"mkdir -p '{path}'"
    return f"powershell -Command \"New-Item -ItemType Directory -Force -Path '{path}'\""


def _verify_nonempty_dir_cmd(path: str, label: str, os_type: str) -> str:
    if os_type == "linux":
        return (
            f"test -d '{path}' && "
            f"find '{path}' -mindepth 1 -print -quit | grep -q . || "
            f"(echo 'missing or empty {label}: {path}' >&2; exit 1)"
        )
    return (
        'powershell -NoProfile -Command "'
        f"$path = '{path}'; $label = '{label}'; "
        "if ((Test-Path -LiteralPath $path -PathType Container) -and "
        "(Get-ChildItem -LiteralPath $path -Force | Select-Object -First 1)) "
        "{ exit 0 } "
        "Write-Error ('missing or empty ' + $label + ': ' + $path); exit 1"
        '"'
    )


def _repair_linux_data_root_cmd() -> str:
    return (
        "sudo mkdir -p /media/user/data/agenthle && "
        "sudo chown -R user:user /media/user/data/agenthle && "
        "test -w /media/user/data/agenthle"
    )


# =============================================================================
# GCS auth
# =============================================================================

def _sa_key_path() -> Path | None:
    """Resolve the local SA key path from env var. Returns None if unset/missing."""
    p = os.environ.get("ALE_GCS_SA_KEY_PATH")
    if not p:
        return None
    path = Path(p).expanduser()
    if not path.is_file():
        logger.warning(
            "ALE_GCS_SA_KEY_PATH=%r but file does not exist — skipping GCS auth setup",
            p,
        )
        return None
    return path


async def ensure_gcs_auth(session: "cb.DesktopSession", os_type: str) -> None:
    """Upload local SA key to VM + activate via ``gcloud auth activate-service-account``.

    No-op (with warning) if ``ALE_GCS_SA_KEY_PATH`` is unset. The VM is
    assumed to have ``gcloud`` baked in. simprun parity: ``_ensure_gcs_auth``.
    """
    key_path = _sa_key_path()
    if key_path is None:
        logger.info(
            "ensure_gcs_auth: no ALE_GCS_SA_KEY_PATH set — relying on VM's "
            "existing gcloud credentials (default service account etc)"
        )
        return

    remote_key = "/tmp/.gcp_key.json" if os_type == "linux" else r"C:\tmp\.gcp_key.json"
    data = key_path.read_bytes()
    await session.write_bytes(remote_key, data)

    if os_type == "linux":
        activate = (
            f"gcloud auth activate-service-account --key-file='{remote_key}' && echo ok"
        )
    else:
        activate = (
            f'gcloud auth activate-service-account --key-file="{remote_key}" && echo ok'
        )
    cr = await session.run_command(activate, timeout=30)
    if "ok" in (cmd_stdout(cr) or ""):
        logger.info("gcs auth activated on VM via SA key")
    else:
        logger.warning(
            "gcs auth activation returned non-ok: %s / %s",
            (cmd_stdout(cr) or "")[:200], (cmd_stderr(cr) or "")[:200],
        )


# =============================================================================
# Disk prep (Linux + Windows)
# =============================================================================

_LINUX_DATA_DISK_CANDIDATES = (
    "/dev/sdb", "/dev/nvme1n1", "/dev/nvme0n2", "/dev/vdb",
)


async def ensure_data_disk(session: "cb.DesktopSession", os_type: str) -> None:
    """Discover + format + mount the data disk at /media/user/data (linux)
    or E: (windows). Idempotent: skips formatting if disk is already mounted
    and writable.

    simprun parity: ``data.ensure_data_disk`` (+ helpers).
    """
    if os_type == "windows":
        await _ensure_windows_data_disk(session)
    else:
        await _ensure_linux_data_disk(session)


async def _ensure_linux_data_disk(session: "cb.DesktopSession") -> None:
    """Mount the attached empty data disk at ``/media/user/data``.

    Strategy: if /media/user/data is already a writable mountpoint, repair
    permissions and return. Otherwise discover the data device, wipe stale
    signatures, format ext4, mount, chown.
    """
    # Fast path: if already mounted + writable, just repair perms.
    check = await session.run_command(
        "mountpoint -q /media/user/data && "
        "sudo mkdir -p /media/user/data/agenthle && "
        "sudo chown -R user:user /media/user/data/agenthle && "
        "test -w /media/user/data/agenthle && echo ready",
        timeout=30,
    )
    if "ready" in (cmd_stdout(check) or ""):
        logger.info("ensure_data_disk: /media/user/data already mounted + writable")
        return

    # Discover device.
    candidates_csv = " ".join(shlex.quote(d) for d in _LINUX_DATA_DISK_CANDIDATES)
    find_script = f"""set -u
root_src="$(findmnt -nro SOURCE / 2>/dev/null | head -n1 || true)"
root_src="$(readlink -f "$root_src" 2>/dev/null || printf '%s' "$root_src")"
root_parent="$(lsblk -ndo PKNAME "$root_src" 2>/dev/null || true)"
if [ -n "$root_parent" ]; then root_disk="/dev/$root_parent"; else root_disk="$root_src"; fi
{{
  for d in {candidates_csv}; do printf '%s\\n' "$d"; done
  lsblk -dnpo NAME,TYPE 2>/dev/null | awk '$2 == "disk" {{ print $1 }}'
}} | while IFS= read -r d; do
  [ -n "$d" ] || continue
  real_d="$(readlink -f "$d" 2>/dev/null || printf '%s' "$d")"
  [ -b "$real_d" ] || continue
  [ "$real_d" = "$root_disk" ] && continue
  lsblk -nrpo MOUNTPOINT "$real_d" 2>/dev/null | grep -qx "/" && continue
  echo "$real_d"; break
done
"""
    disk = ""
    for _ in range(5):
        cr = await session.run_command(
            f"bash -lc {shlex.quote(find_script)}", timeout=15,
        )
        disk = (cmd_stdout(cr) or "").strip()
        if disk:
            break
        await asyncio.sleep(3)
    if not disk:
        diag = await session.run_command(
            "lsblk -o NAME,TYPE,SIZE,MOUNTPOINT,FSTYPE 2>/dev/null || true",
            timeout=15,
        )
        raise RuntimeError(
            "no data disk device found on Linux VM (lsblk: "
            f"{(cmd_stdout(diag) or cmd_stderr(diag) or '').strip()[:600]})"
        )

    # Wipe + format + mount.
    prep_script = f"""set -u
disk={shlex.quote(disk)}
sudo umount /media/user/data 2>/dev/null || true
for dev in $(lsblk -nrpo NAME "$disk" 2>/dev/null | sort -r); do
  sudo umount -fl "$dev" 2>/dev/null || true
  sudo wipefs -a "$dev" 2>/dev/null || true
done
sudo udevadm settle --timeout=5 2>/dev/null || true
sudo mkfs.ext4 -F -q "$disk"
sudo mkdir -p /media/user/data
sudo mount "$disk" /media/user/data
sudo chown -R user:user /media/user/data
sudo mkdir -p /media/user/data/agenthle
sudo chown -R user:user /media/user/data/agenthle
echo prepped
"""
    cr = await session.run_command(
        f"bash -lc {shlex.quote(prep_script)}", timeout=180,
    )
    if "prepped" not in (cmd_stdout(cr) or ""):
        raise RuntimeError(
            f"failed to prep+mount {disk}: "
            f"{(cmd_stderr(cr) or cmd_stdout(cr) or '').strip()[:600]}"
        )
    logger.info("ensure_data_disk: linux data disk %s mounted at /media/user/data", disk)


async def _ensure_windows_data_disk(session: "cb.DesktopSession") -> None:
    """Bring up E: drive — online + initialize if needed."""
    cr = await session.run_command(
        "powershell -Command \"if (Test-Path 'E:\\') { echo ok } else { echo missing }\"",
        timeout=30,
    )
    if "ok" in (cmd_stdout(cr) or ""):
        return

    # Try bringing offline disk online first.
    online = (
        'powershell -Command "'
        "$disk = Get-Disk | Where-Object { $_.OperationalStatus -eq 'Offline' } | Select-Object -First 1; "
        "if ($disk) { Set-Disk -Number $disk.Number -IsOffline $false; "
        "Set-Disk -Number $disk.Number -IsReadOnly $false; echo online } "
        "else { echo no_offline_disk }"
        '"'
    )
    await session.run_command(online, timeout=60)
    cr = await session.run_command(
        "powershell -Command \"if (Test-Path 'E:\\') { echo ok } else { echo missing }\"",
        timeout=30,
    )
    if "ok" in (cmd_stdout(cr) or ""):
        logger.info("ensure_data_disk: E: brought online")
        return

    # Initialize RAW disk.
    init = (
        'powershell -Command "'
        "$disk = Get-Disk | Where-Object { $_.PartitionStyle -eq 'RAW' } | Select-Object -First 1; "
        "if ($disk) { "
        "Initialize-Disk -Number $disk.Number -PartitionStyle GPT -PassThru | "
        "New-Partition -UseMaximumSize -DriveLetter E | "
        "Format-Volume -FileSystem NTFS -NewFileSystemLabel Data -Confirm:$false; "
        "echo initialized "
        "} else { echo no_raw_disk }"
        '"'
    )
    await session.run_command(init, timeout=120)
    cr = await session.run_command(
        "powershell -Command \"if (Test-Path 'E:\\') { echo ok } else { echo missing }\"",
        timeout=30,
    )
    if "ok" not in (cmd_stdout(cr) or ""):
        raise RuntimeError("E: drive still not available after initialization")
    logger.info("ensure_data_disk: E: initialized")


# =============================================================================
# Stage operations
# =============================================================================

async def _gcs_prefix_exists(
    session: "cb.DesktopSession", src: str, os_type: str,
) -> bool:
    """gsutil ls $src → True if exists. Doesn't raise on absent path."""
    cr = await session.run_command(_gcs_ls_cmd(src, os_type), timeout=60)
    return cmd_ok(cr)


async def _rsync_staged_dir(
    session: "cb.DesktopSession",
    *,
    src: str, dst: str, label: str, os_type: str, timeout: float = 1800.0,
) -> None:
    logger.info("staging %s: %s → %s", label, src, dst)
    await _run_on_vm(session, _gcs_rsync_cmd(src, dst, os_type), timeout=timeout)
    await _run_on_vm(
        session, _verify_nonempty_dir_cmd(dst, label, os_type), timeout=60,
    )


async def stage_input(
    session: "cb.DesktopSession",
    task_data: TaskDataSpec,
    os_type: str,
    *,
    bucket: str = DEFAULT_TASK_DATA_BUCKET,
) -> dict[str, Any]:
    """Stage agent-visible data: input/ + software/ + output/ dirs.

    No-op when ``task_data.requires_task_data is False``.

    For each subdir, behaviour:
      - input/    — if GCS prefix exists, rsync; else create empty target dir
      - software/ — if GCS prefix exists, rsync + chmod +x (linux only); else skip
      - output/   — always ensure target dir exists (agent writes here)

    Matches simprun.data.stage_input.
    """
    if not task_data.requires_task_data:
        return {"staged_dirs": [], "skipped": True}

    domain = task_data.domain_name or ""
    task = task_data.task_name or ""
    variant = task_data.variant_name or ""
    gcs_prefix = gcs_task_prefix(domain, task, variant, bucket=bucket)
    staged: list[str] = []

    # 1. Repair root perms on linux (idempotent).
    if os_type == "linux":
        try:
            await _run_on_vm(session, _repair_linux_data_root_cmd(), timeout=60)
        except RuntimeError as exc:
            logger.warning("repair data root failed (continuing): %s", exc)

    # 2. Ensure task root dir exists.
    base = vm_subdir(os_type, domain, task, variant, "")
    await _run_on_vm(session, _mkdir_cmd(base.rstrip("/\\"), os_type))

    # 3. input/
    input_src = f"{gcs_prefix}/input"
    input_dst = task_data.input_dir or vm_subdir(os_type, domain, task, variant, "input")
    if await _gcs_prefix_exists(session, input_src, os_type):
        await _rsync_staged_dir(
            session, src=input_src, dst=input_dst, label="input", os_type=os_type,
        )
        staged.append("input")
    else:
        logger.info("stage_input: no input/ at %s; creating empty target", input_src)
        try:
            await _run_on_vm(session, _mkdir_cmd(input_dst, os_type))
        except RuntimeError:
            pass

    # 4. software/
    software_src = f"{gcs_prefix}/software"
    if await _gcs_prefix_exists(session, software_src, os_type):
        software_dst = task_data.software_dir or vm_subdir(
            os_type, domain, task, variant, "software",
        )
        await _rsync_staged_dir(
            session, src=software_src, dst=software_dst,
            label="software", os_type=os_type,
        )
        if os_type == "linux":
            await _run_on_vm(
                session,
                f"find '{software_dst}' -type f -exec chmod +x {{}} +",
                timeout=60,
            )
        staged.append("software")
    else:
        logger.info("stage_input: no software/ at %s", software_src)

    # 5. output/ (just create dir — agent populates).
    output_dst = task_data.remote_output_dir or vm_subdir(
        os_type, domain, task, variant, "output",
    )
    try:
        await _run_on_vm(session, _mkdir_cmd(output_dst, os_type))
    except RuntimeError:
        pass

    return {"staged_dirs": staged, "skipped": False, "gcs_prefix": gcs_prefix}


async def stage_eval(
    session: "cb.DesktopSession",
    task_data: TaskDataSpec,
    os_type: str,
) -> dict[str, Any]:
    """Stage eval scripts (visible to agent if needed during solve).

    Only fires when both ``eval_gcs_prefix`` AND ``eval_dir`` are set in
    task metadata. Most tasks don't use this — eval logic lives in
    task's evaluate_fn directly. Some tasks ship eval scripts that the
    agent invokes during solve; those set these fields.
    """
    if not task_data.eval_gcs_prefix or not task_data.eval_dir:
        return {"staged_dirs": [], "skipped": True}

    await _run_on_vm(session, _mkdir_cmd(task_data.eval_dir, os_type))
    await _rsync_staged_dir(
        session,
        src=task_data.eval_gcs_prefix,
        dst=task_data.eval_dir,
        label="eval", os_type=os_type,
    )
    if os_type == "linux":
        await _run_on_vm(
            session,
            f"find '{task_data.eval_dir}' -type f -exec chmod +x {{}} +",
            timeout=60,
        )
    return {"staged_dirs": ["eval"], "skipped": False}


async def stage_reference(
    session: "cb.DesktopSession",
    task_data: TaskDataSpec,
    os_type: str,
    *,
    bucket: str = DEFAULT_TASK_DATA_BUCKET,
) -> dict[str, Any]:
    """Stage reference/ground-truth — ONLY called from eval phase.

    The framework guarantees this fires after the agent has stopped and
    just before task.evaluate_fn runs. The agent never sees reference
    data during solve (formal benchmark visibility rule).
    """
    if not task_data.requires_task_data:
        return {"staged_dirs": [], "skipped": True}

    domain = task_data.domain_name or ""
    task = task_data.task_name or ""
    variant = task_data.variant_name or ""
    gcs_prefix = gcs_task_prefix(domain, task, variant, bucket=bucket)

    ref_src = task_data.reference_gcs_prefix or f"{gcs_prefix}/reference"
    ref_dst = task_data.reference_dir or vm_subdir(
        os_type, domain, task, variant, "reference",
    )
    base = vm_subdir(os_type, domain, task, variant, "")
    await _run_on_vm(session, _mkdir_cmd(base.rstrip("/\\"), os_type))
    await _rsync_staged_dir(
        session, src=ref_src, dst=ref_dst,
        label="reference", os_type=os_type,
    )
    return {"staged_dirs": ["reference"], "skipped": False}


async def upload_output(
    session: "cb.DesktopSession",
    task_data: TaskDataSpec,
    os_type: str,
    *,
    run_id: str,
    bucket: str = DEFAULT_RESULTS_BUCKET,
) -> dict[str, Any]:
    """Push agent's output/ dir back to results bucket. Best-effort.

    Called from cleanup phase. Failures are logged + returned, not raised
    (this is post-eval housekeeping; the score is already determined).
    """
    if not task_data.requires_task_data:
        return {"uploaded": False, "skipped": True}

    domain = task_data.domain_name or ""
    task = task_data.task_name or ""
    variant = task_data.variant_name or ""

    output_src = task_data.remote_output_dir or vm_subdir(
        os_type, domain, task, variant, "output",
    )
    gcs_dst = f"{bucket}/{run_id}/output/"

    cmd = _gcs_cp_cmd(output_src, gcs_dst, os_type)
    logger.info("upload_output: %s → %s", output_src, gcs_dst)
    try:
        await _run_on_vm(session, cmd, timeout=600)
        return {"uploaded": True, "gcs_path": gcs_dst}
    except RuntimeError as exc:
        logger.warning("upload_output failed (best-effort): %s", exc)
        return {"uploaded": False, "error": str(exc)}
