"""Force-timeout marker: writes a sentinel file the VM can poll to self-cancel.

Ported from simprun/force_timeout.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .remote import LINUX_USER_HOME, RemoteVMConfig, run_remote, upload_file


FORCE_TIMEOUT_FILENAME = "ale_force_timeout.json"
WINDOWS_FORCE_TIMEOUT_PATH = rf"C:\Users\User\{FORCE_TIMEOUT_FILENAME}"
LINUX_FORCE_TIMEOUT_PATH = f"{LINUX_USER_HOME}/{FORCE_TIMEOUT_FILENAME}"
LOCAL_FORCE_TIMEOUT_DIR = Path(".force_timeouts")


def force_timeout_path(os_type: str) -> str:
    return LINUX_FORCE_TIMEOUT_PATH if os_type == "linux" else WINDOWS_FORCE_TIMEOUT_PATH


def local_force_timeout_path(run_id: str) -> Path:
    return LOCAL_FORCE_TIMEOUT_DIR / f"{run_id}.json"


def _local_path_for_config(vm_config: RemoteVMConfig) -> Path | None:
    if not vm_config.run_id:
        return None
    return local_force_timeout_path(vm_config.run_id)


def _write_local_force_timeout_request(run_id: str, payload: dict[str, Any]) -> Path:
    LOCAL_FORCE_TIMEOUT_DIR.mkdir(parents=True, exist_ok=True)
    path = local_force_timeout_path(run_id)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def clear_force_timeout_request(vm_config: RemoteVMConfig) -> None:
    local_path = _local_path_for_config(vm_config)
    if local_path is not None:
        local_path.unlink(missing_ok=True)

    path = force_timeout_path(vm_config.os_type)
    if vm_config.is_linux:
        run_remote(vm_config, f"rm -f '{path}'", timeout=5)
    else:
        run_remote(
            vm_config,
            f"powershell -NoProfile -Command \""
            f"Remove-Item -Path '{path}' -Force -ErrorAction SilentlyContinue\"",
            timeout=5,
        )


def write_force_timeout_request(
    vm_config: RemoteVMConfig,
    *,
    task_id: str,
    run_id: str | None = None,
    reason: str = "manual_force_timeout",
    requested_by: str = "manager",
    extra: dict[str, Any] | None = None,
) -> str:
    path = force_timeout_path(vm_config.os_type)
    payload = {
        "task_id": task_id,
        "reason": reason,
        "requested_by": requested_by,
        "requested_at": time.time(),
    }
    if extra:
        payload["extra"] = extra

    local_path: Path | None = None
    effective_run_id = run_id or vm_config.run_id or (extra or {}).get("run_id")
    if effective_run_id:
        local_path = _write_local_force_timeout_request(str(effective_run_id), payload)

    try:
        upload_file(vm_config, path, json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception:
        if local_path is None:
            raise
        return str(local_path)
    return path


def force_timeout_requested(vm_config: RemoteVMConfig) -> bool:
    local_path = _local_path_for_config(vm_config)
    if local_path is not None and local_path.exists():
        return True

    path = force_timeout_path(vm_config.os_type)
    if vm_config.is_linux:
        result = run_remote(vm_config, f"test -f '{path}' && echo yes || true", timeout=5)
    else:
        result = run_remote(
            vm_config,
            f"powershell -NoProfile -Command \""
            f"if (Test-Path '{path}') {{ 'yes' }}\"",
            timeout=5,
        )
    return result.returncode == 0 and "yes" in (result.stdout or "").strip().lower()
