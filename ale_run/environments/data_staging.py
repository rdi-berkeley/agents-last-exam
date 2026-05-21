"""GCS-staged task data: env disk prep, input/reference staging, output upload.

Ported from simprun/data.py. Drops the ``_ensure_gcs_auth`` helper that loaded
a local SA key from a hardcoded REPO_ROOT — agenthle-public expects gcloud
auth to be done out-of-band (or via an SA key passed through the provider
config). Calls into ``_ensure_gcs_auth`` are kept but become no-ops when no
key path is configured; the caller can opt in by passing ``local_key_path``.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

from .images import gcs_task_prefix, env_subdir
from .remote import run_remote, upload_file
from ..base_interface import EnvHandle, TaskDataSpec

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY_S = 10
_LINUX_DATA_DISK_CANDIDATES = (
    "/dev/sdb",
    "/dev/nvme1n1",
    "/dev/nvme0n2",
    "/dev/vdb",
)


class MountFailureError(RuntimeError):
    """Data disk on this env's capacity profile failed to surface / mount.

    Raised by ``ensure_data_disk`` (Linux and Windows paths) when the
    attached data volume can't be brought online. The lifecycle catches
    this specifically to swap capacity profiles and re-provision —
    *not* RuntimeError, so unrelated runtime failures aren't retried.
    """


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
        "if ($LASTEXITCODE -eq 0) { exit 0 } else { exit 1 }"
        '"'
    )


def _mkdir_cmd(path: str, os_type: str) -> str:
    if os_type == "linux":
        return f"mkdir -p '{path}'"
    return f"powershell -Command \"New-Item -ItemType Directory -Force -Path '{path}'\""


def _repair_linux_data_root_cmd() -> str:
    return (
        "sudo mkdir -p /media/user/data/agenthle && "
        "sudo chown -R user:user /media/user/data/agenthle && "
        "test -w /media/user/data/agenthle"
    )


def _verify_nonempty_dir_cmd(path: str, label: str, os_type: str) -> str:
    if os_type == "linux":
        return (
            f"test -d '{path}' && "
            f"find '{path}' -mindepth 1 -print -quit | grep -q . || "
            f"(echo 'missing or empty {label}: {path}' >&2; exit 1)"
        )
    return (
        'powershell -NoProfile -Command "'
        f"$path = '{path}'; "
        f"$label = '{label}'; "
        "if ((Test-Path -LiteralPath $path -PathType Container) -and "
        "(Get-ChildItem -LiteralPath $path -Force | Select-Object -First 1)) "
        "{ exit 0 } "
        "Write-Error ('missing or empty ' + $label + ': ' + $path); exit 1"
        '"'
    )


_NO_RETRY_PATTERNS = (
    "matched no objects or files",
    "No URLs matched",
    "One or more URLs matched no objects",
    "BucketNotFoundException",
    "CommandException: No such object",
)


def _is_unrecoverable(error_text: str) -> bool:
    return any(p in error_text for p in _NO_RETRY_PATTERNS)


def _run_on_env(env_handle: EnvHandle, command: str, timeout: float = 300):
    import time

    last_err = None
    for attempt in range(1, _MAX_RETRIES + 1):
        result = run_remote(env_handle, command, timeout=timeout)
        if result.returncode == 0:
            return result
        last_err = result.stderr or result.stdout
        logger.warning(
            "Remote command failed (attempt %d/%d, rc=%d): %s",
            attempt,
            _MAX_RETRIES,
            result.returncode,
            last_err[:200],
        )
        if _is_unrecoverable(last_err):
            break
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY_S)
    raise RuntimeError(f"Remote command failed after {_MAX_RETRIES} attempts: {last_err}")


def ensure_gcs_auth(env_handle: EnvHandle, os_type: str, local_key_path: Path | None) -> None:
    """Activate gcloud's service account on the env using ``local_key_path``.

    No-op when ``local_key_path`` is None or missing — the env is assumed to
    have ambient credentials (default service account) in that case.
    """
    if local_key_path is None or not local_key_path.exists():
        logger.debug("No GCS key path configured; skipping env-side gcloud auth.")
        return

    if os_type == "linux":
        remote_key = "/tmp/.gcp_key.json"
    else:
        remote_key = r"C:\tmp\.gcp_key.json"

    key_content = local_key_path.read_text(encoding="utf-8")
    upload_file(env_handle, remote_key, key_content)

    if os_type == "linux":
        activate_cmd = f"gcloud auth activate-service-account --key-file='{remote_key}' && echo ok"
    else:
        activate_cmd = f'gcloud auth activate-service-account --key-file="{remote_key}" && echo ok'

    result = run_remote(env_handle, activate_cmd, timeout=30)
    if "ok" in (result.stdout or ""):
        logger.info("GCS auth activated via service account key")
    else:
        logger.warning("GCS auth activation returned: %s", result.stdout or result.stderr)


def _gcs_prefix_exists(env_handle: EnvHandle, src: str, os_type: str) -> bool:
    result = run_remote(env_handle, _gcs_ls_cmd(src, os_type), timeout=60)
    return result.returncode == 0


def _rsync_staged_dir(
    env_handle: EnvHandle,
    *,
    src: str,
    dst: str,
    label: str,
    os_type: str,
    timeout: float = 1800,
) -> None:
    logger.info("Staging %s with rsync: %s -> %s", label, src, dst)
    _run_on_env(env_handle, _gcs_rsync_cmd(src, dst, os_type), timeout=timeout)
    _run_on_env(env_handle, _verify_nonempty_dir_cmd(dst, label, os_type), timeout=60)


def stage_input(
    env_handle: EnvHandle,
    task_data: TaskDataSpec,
    os_type: str,
    *,
    gcs_bucket: str,
    gcs_local_key_path: Path | None = None,
) -> dict:
    if not task_data.requires_task_data:
        return {"staged_dirs": [], "skipped": True}

    domain = task_data.domain_name
    task = task_data.task_name
    variant = task_data.variant_name
    gcs_prefix = gcs_task_prefix(gcs_bucket, domain, task, variant)
    staged = []

    ensure_gcs_auth(env_handle, os_type, gcs_local_key_path)

    if os_type == "linux":
        _run_on_env(env_handle, _repair_linux_data_root_cmd(), timeout=60)

    base_dir = env_subdir(os_type, domain, task, variant, "")
    _run_on_env(env_handle, _mkdir_cmd(base_dir.rstrip("/\\"), os_type))

    input_src = f"{gcs_prefix}/input"
    input_dst = task_data.input_dir or env_subdir(os_type, domain, task, variant, "input")
    if _gcs_prefix_exists(env_handle, input_src, os_type):
        _rsync_staged_dir(
            env_handle,
            src=input_src,
            dst=input_dst,
            label="input",
            os_type=os_type,
        )
        staged.append("input")
    else:
        logger.info("No input/ to stage at %s — creating empty target dir", input_src)
        try:
            _run_on_env(env_handle, _mkdir_cmd(input_dst, os_type))
        except RuntimeError:
            pass

    software_src = f"{gcs_prefix}/software"
    if _gcs_prefix_exists(env_handle, software_src, os_type):
        software_dst = task_data.software_dir or env_subdir(
            os_type, domain, task, variant, "software"
        )
        _rsync_staged_dir(
            env_handle,
            src=software_src,
            dst=software_dst,
            label="software",
            os_type=os_type,
        )
        if os_type == "linux":
            _run_on_env(
                env_handle,
                f"find '{software_dst}' -type f -exec chmod +x {{}} +",
                timeout=60,
            )
        staged.append("software")
    else:
        logger.info("No software/ to stage at %s", software_src)

    output_dir = env_subdir(os_type, domain, task, variant, "output")
    try:
        _run_on_env(env_handle, _mkdir_cmd(output_dir, os_type))
    except RuntimeError:
        pass

    return {"staged_dirs": staged, "skipped": False}


def stage_reference(
    env_handle: EnvHandle,
    task_data: TaskDataSpec,
    os_type: str,
    *,
    gcs_bucket: str,
) -> dict:
    if not task_data.requires_task_data:
        return {"staged_dirs": [], "skipped": True}

    domain = task_data.domain_name
    task = task_data.task_name
    variant = task_data.variant_name
    gcs_prefix = gcs_task_prefix(gcs_bucket, domain, task, variant)

    base_dir = env_subdir(os_type, domain, task, variant, "")
    ref_src = task_data.reference_gcs_prefix or f"{gcs_prefix}/reference"
    ref_dst = task_data.reference_dir or env_subdir(os_type, domain, task, variant, "reference")
    _run_on_env(env_handle, _mkdir_cmd(base_dir.rstrip("/\\"), os_type))
    _rsync_staged_dir(
        env_handle,
        src=ref_src,
        dst=ref_dst,
        label="reference",
        os_type=os_type,
    )
    return {"staged_dirs": ["reference"], "skipped": False}


def upload_output(
    env_handle: EnvHandle,
    task_data: TaskDataSpec,
    os_type: str,
    run_id: str,
    *,
    gcs_results_bucket: str,
) -> dict:
    if not task_data.requires_task_data:
        return {"uploaded": False, "skipped": True}

    domain = task_data.domain_name
    task = task_data.task_name
    variant = task_data.variant_name

    output_src = env_subdir(os_type, domain, task, variant, "output")
    gcs_dst = f"{gcs_results_bucket.rstrip('/')}/{run_id}/output/"

    cmd = _gcs_cp_cmd(output_src, gcs_dst, os_type)
    logger.info("Uploading output: %s → %s", output_src, gcs_dst)
    try:
        _run_on_env(env_handle, cmd, timeout=600)
        return {"uploaded": True, "gcs_path": gcs_dst}
    except RuntimeError as e:
        logger.warning("Output upload failed (best-effort): %s", e)
        return {"uploaded": False, "error": str(e)}


def ensure_data_disk(env_handle: EnvHandle, os_type: str) -> None:
    if os_type == "windows":
        _ensure_windows_data_disk(env_handle)
    else:
        _ensure_linux_data_disk(env_handle)


# ======================================================================
# Windows display resolution (verbatim from simprun/runner.py:30-76)
# ======================================================================
#
# Tasks that drive the GUI assume a known framebuffer size — GPU envs get
# 1920×1080, CPU envs 1024×768. Linux envs are skipped (X server picks its
# own size). Called from lifecycle Phase 1 once the CUA server is ready.

_EXPECTED_RESOLUTION = {
    True: (1920, 1080),   # GPU envs
    False: (1024, 768),   # CPU envs
}

_SET_RES_PY = """\
import ctypes, ctypes.wintypes as wt, sys
u = ctypes.windll.user32
cur_w, cur_h = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
tw, th = int(sys.argv[1]), int(sys.argv[2])
if (cur_w, cur_h) == (tw, th):
    print("already_ok"); sys.exit(0)
fields = [
    ("a",ctypes.c_wchar*32),("b",wt.WORD),("c",wt.WORD),
    ("d",wt.WORD),("e",wt.WORD),("f",wt.DWORD),
    ("g",ctypes.c_long),("h",ctypes.c_long),
    ("i",wt.DWORD),("j",wt.DWORD),
    ("k",ctypes.c_short),("l",ctypes.c_short),
    ("m",ctypes.c_short),("n",ctypes.c_short),("o",ctypes.c_short),
    ("p",ctypes.c_wchar*32),("q",wt.WORD),("r",wt.DWORD),
    ("w",wt.DWORD),("ht",wt.DWORD),("fl",wt.DWORD),("fr",wt.DWORD),
]
DM = type("DM", (ctypes.Structure,), {"_fields_": fields})
dm = DM(); dm.d = ctypes.sizeof(dm)
u.EnumDisplaySettingsW(None, -1, ctypes.byref(dm))
dm.w = tw; dm.ht = th; dm.f = 0x80000 | 0x100000
r = u.ChangeDisplaySettingsW(ctypes.byref(dm), 0)
print("set_ok" if r == 0 else f"failed:{r}")
"""


def set_windows_resolution(env_handle: EnvHandle, has_gpu: bool) -> None:
    """Force the Windows env's framebuffer to (1920,1080)/(1024,768).

    No-op on Linux envs (the caller guards on os_type). Best-effort: a
    failure here is logged but doesn't fail the run — the tested agent
    may still solve the task at the default resolution.
    """
    target_w, target_h = _EXPECTED_RESOLUTION[has_gpu]
    remote_path = r"C:\Users\User\_set_resolution.py"
    try:
        upload_file(env_handle, remote_path, _SET_RES_PY)
        result = run_remote(
            env_handle,
            f'python "{remote_path}" {target_w} {target_h}',
            timeout=20,
        )
        out = result.stdout.strip()
        if "set_ok" in out:
            logger.info("Display resolution set to %dx%d", target_w, target_h)
        elif "already_ok" in out:
            logger.info("Display resolution already %dx%d", target_w, target_h)
        else:
            logger.warning("Display resolution change result: %s", out)
    except Exception as e:
        logger.warning("Failed to set display resolution: %s", e)


def _linux_data_disk_find_cmd() -> str:
    candidates = " ".join(shlex.quote(d) for d in _LINUX_DATA_DISK_CANDIDATES)
    return f"""bash -lc {shlex.quote(f'''
set -u
root_src="$(findmnt -nro SOURCE / 2>/dev/null | head -n1 || true)"
root_src="$(readlink -f "$root_src" 2>/dev/null || printf "%s" "$root_src")"
root_parent="$(lsblk -ndo PKNAME "$root_src" 2>/dev/null || true)"
if [ -n "$root_parent" ]; then
  root_disk="/dev/$root_parent"
else
  root_disk="$root_src"
fi

{{
  for d in {candidates}; do
    printf '%s\\n' "$d"
  done
  lsblk -dnpo NAME,TYPE 2>/dev/null | awk '$2 == "disk" {{ print $1 }}'
  find /dev/disk/by-id -maxdepth 1 -type l -name 'google-*' -print 2>/dev/null
}} | while IFS= read -r d; do
  [ -n "$d" ] || continue
  real_d="$(readlink -f "$d" 2>/dev/null || printf "%s" "$d")"
  [ -b "$real_d" ] || continue
  if [ "$real_d" = "$root_disk" ]; then
    continue
  fi
  if lsblk -nrpo MOUNTPOINT "$real_d" 2>/dev/null | grep -qx "/"; then
    continue
  fi
  echo "$real_d"
  break
done
''')}"""


def _linux_data_disk_prep_cmd(disk_device: str) -> str:
    script = f"""
set -u
disk={shlex.quote(disk_device)}

if [ ! -b "$disk" ]; then
  echo "data disk device missing: $disk" >&2
  exit 1
fi

sudo umount /media/user/data 2>/dev/null || true

if [ -f /etc/fstab ]; then
  sudo cp -n /etc/fstab /etc/fstab.ale.bak 2>/dev/null || true
  sudo sed -i -E '\\|[[:space:]]/media/user/data[[:space:]]| s|^[[:space:]]*([^#])|# ale disabled data disk automount: \\1|' /etc/fstab
  sudo systemctl daemon-reload 2>/dev/null || true
fi

for pass in 1 2 3; do
  lsblk -nrpo NAME "$disk" 2>/dev/null | sort -r | while IFS= read -r dev; do
    [ -n "$dev" ] || continue
    sudo swapoff "$dev" 2>/dev/null || true
  done

  lsblk -nrpo MOUNTPOINT "$disk" 2>/dev/null | awk 'NF' | sort -r | while IFS= read -r mountpoint; do
    sudo umount -fl "$mountpoint" 2>/dev/null || true
  done

  lsblk -nrpo NAME "$disk" 2>/dev/null | sort -r | while IFS= read -r dev; do
    [ -n "$dev" ] || continue
    sudo umount -fl "$dev" 2>/dev/null || true
  done

  sudo udevadm settle --timeout=5 2>/dev/null || true
  if ! lsblk -nrpo MOUNTPOINT "$disk" 2>/dev/null | awk 'NF {{ found=1 }} END {{ exit found ? 0 : 1 }}'; then
    break
  fi
  sleep 1
done

if lsblk -nrpo MOUNTPOINT "$disk" 2>/dev/null | awk 'NF {{ found=1 }} END {{ exit found ? 0 : 1 }}'; then
  echo "data disk still mounted before format:" >&2
  lsblk -nrpo NAME,MOUNTPOINT "$disk" >&2 || true
  exit 1
fi

lsblk -nrpo NAME "$disk" 2>/dev/null | sort -r | while IFS= read -r dev; do
  [ -n "$dev" ] || continue
  sudo wipefs -a "$dev" 2>/dev/null || true
done
sudo blockdev --rereadpt "$disk" 2>/dev/null || true
sudo udevadm settle --timeout=5 2>/dev/null || true
echo prepped
"""
    return f"bash -lc {shlex.quote(script)}"


def _ensure_linux_data_disk(env_handle: EnvHandle) -> None:
    """Format the attached empty data disk and mount it at /media/user/data."""
    import time as _time

    run_remote(
        env_handle,
        "sudo udevadm settle --timeout=15 2>/dev/null || true",
        timeout=30,
    )

    find_script = _linux_data_disk_find_cmd()

    disk_device = None
    for attempt in range(5):
        result = run_remote(env_handle, find_script, timeout=15)
        disk_device = (result.stdout or "").strip()
        if disk_device:
            break
        logger.info(
            "Data disk device not yet visible (attempt %d/5), waiting...",
            attempt + 1,
        )
        _time.sleep(3)

    if not disk_device:
        diag = run_remote(
            env_handle,
            "lsblk -o NAME,TYPE,SIZE,MOUNTPOINT,FSTYPE,PKNAME 2>/dev/null || true",
            timeout=15,
        )
        raise RuntimeError(
            "No data disk device found on Linux env after retries — "
            "expected an attached non-root block disk. "
            f"lsblk: {(diag.stdout or diag.stderr or '').strip()[:1000]}"
        )

    prep = run_remote(env_handle, _linux_data_disk_prep_cmd(disk_device), timeout=60)
    if "prepped" not in (prep.stdout or ""):
        raise RuntimeError(
            f"Failed to prepare {disk_device} for formatting: {prep.stderr or prep.stdout}"
        )

    last_err = ""
    for attempt in range(3):
        fmt = run_remote(
            env_handle,
            f"sudo mkfs.ext4 -F -q {disk_device} && echo formatted",
            timeout=120,
        )
        if "formatted" in (fmt.stdout or ""):
            break
        last_err = fmt.stderr or fmt.stdout or ""
        logger.warning(
            "mkfs.ext4 failed (attempt %d/3): %s",
            attempt + 1,
            last_err[:200],
        )
        _time.sleep(5)
    else:
        raise RuntimeError(f"Failed to format {disk_device} after 3 attempts: {last_err}")

    mount_cmd = (
        f"sudo umount /media/user/data 2>/dev/null; "
        f"sudo mkdir -p /media/user/data && "
        f"sudo mount {disk_device} /media/user/data && "
        f"sudo chown user:user /media/user/data && "
        f"echo mounted"
    )
    mnt = run_remote(env_handle, mount_cmd, timeout=30)
    if "mounted" not in (mnt.stdout or ""):
        raise MountFailureError(f"Failed to mount {disk_device}: {mnt.stderr}")

    run_remote(
        env_handle,
        "mkdir -p /media/user/data/agenthle",
        timeout=15,
    )

    logger.info(
        "Linux data disk %s formatted and mounted at /media/user/data",
        disk_device,
    )


def _dismiss_format_dialog(env_handle: EnvHandle) -> None:
    try:
        run_remote(
            env_handle,
            'powershell -Command "Get-Process -Name explorer -ErrorAction SilentlyContinue | ForEach-Object {'
            "  $wshell = New-Object -ComObject WScript.Shell;"
            "  $null = $wshell.AppActivate('Format');"
            "  Start-Sleep -Milliseconds 200;"
            "  $wshell.SendKeys('{ESC}')"
            '}"',
            timeout=10,
        )
    except Exception:
        pass


def _ensure_windows_data_disk(env_handle: EnvHandle) -> None:
    _dismiss_format_dialog(env_handle)

    check = run_remote(
        env_handle,
        "powershell -Command \"if (Test-Path 'E:\\') { echo ok } else { echo missing }\"",
        timeout=30,
    )
    if "ok" in (check.stdout or ""):
        return

    logger.info("E: drive not found, attempting to bring data disk online")
    online_script = (
        'powershell -Command "'
        "$disk = Get-Disk | Where-Object { $_.OperationalStatus -eq 'Offline' } | Select-Object -First 1; "
        "if ($disk) { Set-Disk -Number $disk.Number -IsOffline $false; "
        "Set-Disk -Number $disk.Number -IsReadOnly $false; echo online } "
        "else { echo no_offline_disk }"
        '"'
    )
    result = run_remote(env_handle, online_script, timeout=60)
    if result.returncode != 0:
        raise MountFailureError(f"Failed to bring data disk online: {result.stderr}")

    check2 = run_remote(
        env_handle,
        "powershell -Command \"if (Test-Path 'E:\\') { echo ok } else { echo missing }\"",
        timeout=30,
    )
    if "ok" in (check2.stdout or ""):
        logger.info("E: drive is now available")
        return

    logger.info("E: drive still missing after online — initializing raw disk")
    init_script = (
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
    result = run_remote(env_handle, init_script, timeout=120)
    if result.returncode != 0:
        raise MountFailureError(f"Failed to initialize data disk: {result.stderr}")

    check3 = run_remote(
        env_handle,
        "powershell -Command \"if (Test-Path 'E:\\') { echo ok } else { echo missing }\"",
        timeout=30,
    )
    if "ok" not in (check3.stdout or ""):
        raise MountFailureError("E: drive still not available after initialization")
    logger.info("E: drive initialized and available")
    _dismiss_format_dialog(env_handle)
