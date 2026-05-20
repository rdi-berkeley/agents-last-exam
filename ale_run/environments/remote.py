"""CUA HTTP API primitives for remote VM command execution and file transfer.

Ported from simprun/remote.py. Trimmed to the host-side primitives used by
data_staging, the gcloud provider, and orchastration/gather. Drop the
Node/MCP install helpers — image-baking is now the deployer's responsibility.

All remote operations go through the CUA server on port 5000 — no SSH.
"""

from __future__ import annotations

import base64
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Linux conventions used by force_timeout and a few other helpers.
LINUX_USER_HOME = "/home/user"


@dataclass
class RemoteVMConfig:
    server_url: str
    os_type: str = "windows"
    run_id: str | None = None
    task_id: str | None = None

    @property
    def is_linux(self) -> bool:
        return self.os_type == "linux"


# ======================================================================
# BOM stripping
# ======================================================================

_DOUBLE_BOM = b"\xc3\xaf\xc2\xbb\xc2\xbf"
_UTF8_BOM = b"\xef\xbb\xbf"


def strip_bom(raw: bytes) -> bytes:
    if raw.startswith(_DOUBLE_BOM):
        return raw[len(_DOUBLE_BOM):]
    if raw.startswith(_UTF8_BOM):
        return raw[len(_UTF8_BOM):]
    return raw


# ======================================================================
# Core HTTP helpers
# ======================================================================


def _cua_url(vm_config: RemoteVMConfig) -> str:
    return vm_config.server_url.rstrip("/")


def _read_first_sse_event(resp: requests.Response, read_timeout: float = 30) -> dict[str, Any] | None:
    # iter_lines is O(N^2) on a single multi-MB `data:` event; resp.content
    # collects via b"".join in linear time.
    try:
        sock = resp.raw._fp.fp.raw._sock
    except AttributeError:
        sock = None
    if sock is not None:
        try:
            sock.settimeout(read_timeout)
        except Exception:
            pass
    try:
        body = resp.content
    except (OSError, requests.exceptions.RequestException):
        return None
    for line in body.splitlines():
        if line.startswith(b"data: "):
            try:
                return json.loads(line[6:])
            except json.JSONDecodeError:
                return None
    return None


class VMUnreachableError(RuntimeError):
    pass


def require_vm_reachable(vm_config: RemoteVMConfig, *, agent_label: str = "VM") -> None:
    probe_cmd = "true" if vm_config.is_linux else "cmd /c echo ok"
    result = run_remote(vm_config, probe_cmd, timeout=60)
    if result.returncode == -1:
        raise VMUnreachableError(
            f"{agent_label} VM unreachable at {vm_config.server_url}: "
            f"{result.stderr or 'no response'}"
        )


def run_remote(vm_config: RemoteVMConfig, command: str, timeout: float = 60) -> subprocess.CompletedProcess:
    payload = {"command": "run_command", "params": {"command": command}}

    try:
        with requests.post(
            f"{_cua_url(vm_config)}/cmd",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            stream=True,
        ) as resp:
            data = _read_first_sse_event(resp, read_timeout=timeout)
        if data is not None:
            return subprocess.CompletedProcess(
                args=command,
                returncode=data.get("return_code", data.get("returncode", 0)),
                stdout=data.get("stdout", data.get("output", "")),
                stderr=data.get("stderr", ""),
            )
    except requests.Timeout:
        return subprocess.CompletedProcess(args=command, returncode=-1, stdout="", stderr="timeout")
    except Exception as e:
        return subprocess.CompletedProcess(args=command, returncode=-1, stdout="", stderr=str(e))

    return subprocess.CompletedProcess(args=command, returncode=-1, stdout="", stderr="no response")


# ======================================================================
# File transfer
# ======================================================================


def upload_file(vm_config: RemoteVMConfig, remote_path: str, content: str, timeout: float = 60) -> None:
    url = f"{_cua_url(vm_config)}/cmd"

    if vm_config.is_linux:
        try:
            with requests.post(
                url,
                json={"command": "write_text", "params": {"path": remote_path, "content": content}},
                headers={"Content-Type": "application/json"},
                timeout=timeout,
                stream=True,
            ) as resp:
                data = _read_first_sse_event(resp, read_timeout=timeout)
            if data is not None and data.get("success"):
                return
            detail = (data or {}).get("error", "unknown")
        except Exception as e:
            detail = str(e)
        raise RuntimeError(f"upload_file failed for {remote_path}: {detail}")

    # Windows: binary-safe base64 upload
    try:
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        b64_path = f"{remote_path}.b64"
        with requests.post(
            url,
            json={"command": "write_text", "params": {"path": b64_path, "content": encoded}},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            stream=True,
        ) as resp:
            data = _read_first_sse_event(resp, read_timeout=timeout)
        if not (data and data.get("success")):
            detail = (data or {}).get("error", "unknown")
            raise RuntimeError(
                f"upload_file failed for {remote_path} (base64 write): {detail}"
            )

        ps_cmd = (
            f"$b=[IO.File]::ReadAllText('{b64_path}').Trim();"
            f"[IO.File]::WriteAllBytes('{remote_path}',[Convert]::FromBase64String($b));"
            f"Remove-Item -Path '{b64_path}' -Force -ErrorAction SilentlyContinue"
        )
        with requests.post(
            url,
            json={
                "command": "run_command",
                "params": {"command": f'powershell -NoProfile -Command "{ps_cmd}"'},
            },
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            stream=True,
        ) as resp:
            data = _read_first_sse_event(resp, read_timeout=timeout)
        rc = data.get("return_code", data.get("returncode", 1)) if data else 1
        if rc == 0:
            return
        stderr = (data or {}).get("stderr", "")[:200]
        raise RuntimeError(
            f"upload_file failed for {remote_path} (base64 decode): rc={rc} {stderr}"
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"upload_file failed for {remote_path}: {e}") from e


def upload_binary_file(
    vm_config: RemoteVMConfig,
    local_path: str | Path,
    remote_path: str,
    *,
    chunk_size: int = 768 * 1024,
    timeout: float = 60,
) -> None:
    """Upload a binary file through text-only CUA APIs using base64 chunks."""
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    chunk_size -= chunk_size % 3
    chunk_size = max(chunk_size, 3)

    sep = "/" if vm_config.is_linux else "\\"
    part_dir = f"{remote_path}.parts"
    if vm_config.is_linux:
        run_remote(vm_config, f"rm -rf '{part_dir}' && mkdir -p '{part_dir}'", timeout=30)
    else:
        run_remote(
            vm_config,
            f'powershell -NoProfile -Command "'
            f"Remove-Item -Recurse -Force '{part_dir}' -ErrorAction SilentlyContinue; "
            f"New-Item -ItemType Directory -Force -Path '{part_dir}' | Out-Null\"",
            timeout=30,
        )

    part_paths: list[str] = []
    with local_path.open("rb") as fh:
        idx = 0
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            part_path = f"{part_dir}{sep}{idx:06d}.chunk"
            upload_file(vm_config, part_path, base64.b64encode(chunk).decode("ascii"), timeout=timeout)
            part_paths.append(part_path)
            idx += 1

    if not part_paths:
        upload_file(vm_config, f"{part_dir}{sep}000000.chunk", "", timeout=timeout)
        part_paths.append(f"{part_dir}{sep}000000.chunk")

    if vm_config.is_linux:
        decode_cmd = (
            f"cat '{part_dir}'/*.chunk | base64 -d > '{remote_path}' && "
            f"rm -rf '{part_dir}' && test -s '{remote_path}'"
        )
    else:
        decode_cmd = (
            f'powershell -NoProfile -Command "'
            f"$b = (Get-ChildItem -Path '{part_dir}' -Filter '*.chunk' | Sort-Object Name | "
            f"ForEach-Object {{ [IO.File]::ReadAllText($_.FullName).Trim() }}) -join ''; "
            f"[IO.File]::WriteAllBytes('{remote_path}', [Convert]::FromBase64String($b)); "
            f"Remove-Item -Recurse -Force '{part_dir}'; "
            f"if ((Get-Item '{remote_path}').Length -le 0) {{ exit 1 }}\""
        )
    result = run_remote(vm_config, decode_cmd, timeout=max(timeout, 300))
    if result.returncode != 0:
        raise RuntimeError(
            f"binary upload decode failed for {remote_path}: rc={result.returncode} "
            f"stdout={result.stdout[-1000:]} stderr={result.stderr[-1000:]}"
        )


def download_file(vm_config: RemoteVMConfig, remote_path: str, local_path: str, timeout: float = 60) -> bool:
    url = f"{_cua_url(vm_config)}/cmd"

    if vm_config.is_linux:
        try:
            with requests.post(
                url,
                json={"command": "read_text", "params": {"path": remote_path}},
                headers={"Content-Type": "application/json"},
                timeout=timeout,
                stream=True,
            ) as resp:
                data = _read_first_sse_event(resp, read_timeout=timeout)
            if data and data.get("success"):
                content = data.get("content", "")
                if content:
                    Path(local_path).write_text(content, encoding="utf-8")
                    return True
        except Exception as e:
            logger.debug("read_text failed for %s: %s", remote_path, e)
        return False

    # Windows: binary-safe base64 download
    try:
        escaped = remote_path.replace("'", "''")
        ps_cmd = (
            f"$fs=[IO.File]::Open('{escaped}','Open','Read','ReadWrite');"
            "try{$buf=New-Object byte[] $fs.Length;"
            "$null=$fs.Read($buf,0,$fs.Length);"
            "[Convert]::ToBase64String($buf)}"
            "finally{$fs.Close()}"
        )
        with requests.post(
            url,
            json={
                "command": "run_command",
                "params": {"command": f'powershell -NoProfile -Command "{ps_cmd}"'},
            },
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            stream=True,
        ) as resp:
            data = _read_first_sse_event(resp, read_timeout=timeout)
        if data and data.get("return_code", data.get("returncode", 1)) == 0:
            b64_text = (data.get("stdout", "") or "").strip()
            if b64_text:
                raw = base64.b64decode(b64_text)
                Path(local_path).write_bytes(raw)
                return True
    except Exception as e:
        logger.debug("base64 download failed for %s: %s", remote_path, e)

    return False


# ======================================================================
# Range (incremental) download — kept for future incremental puller use.
# ======================================================================


@dataclass
class RangeResult:
    success: bool
    remote_size: int = 0
    delta: bytes = b""
    error: str | None = None


def _build_range_cmd_linux(remote_path: str, start: int, max_chunk_bytes: int) -> str:
    safe_path = remote_path.replace("'", "'\\''")
    return (
        f"S=$(stat -c%s '{safe_path}' 2>/dev/null || echo -1); "
        f"if [ \"$S\" -lt 0 ]; then printf 'SIZE=-1\\nB64=\\n'; exit 0; fi; "
        f"B=\"\"; "
        f"if [ \"$S\" -gt {start} ]; then "
        f"  B=$(tail -c +$(( {start} + 1 )) '{safe_path}' "
        f"     | head -c {max_chunk_bytes} | base64 -w0); "
        f"fi; "
        f"printf 'SIZE=%s\\nB64=%s\\n' \"$S\" \"$B\""
    )


def _build_range_cmd_windows(remote_path: str, start: int, max_chunk_bytes: int) -> str:
    safe_path = remote_path.replace("'", "''")
    ps = (
        "try{"
        f"$fi=[IO.FileInfo]::new('{safe_path}');"
        "if(-not $fi.Exists){\"SIZE=-1\";\"B64=\";exit};"
        "$len=$fi.Length;$b64='';"
        f"if($len -gt {start}){{"
        f"$fs=[IO.File]::Open('{safe_path}','Open','Read','ReadWrite');"
        "try{"
        f"$null=$fs.Seek({start},'Begin');"
        f"$rem=[Math]::Min($len-{start},{max_chunk_bytes});"
        "$buf=New-Object byte[] $rem;"
        "$null=$fs.Read($buf,0,$rem);"
        "$b64=[Convert]::ToBase64String($buf)"
        "}finally{$fs.Close()}"
        "};"
        "\"SIZE=$len\";\"B64=$b64\""
        "}catch{\"SIZE=-2\";\"B64=\";\"ERR=$($_.Exception.Message)\"}"
    )
    return f'powershell -NoProfile -Command "{ps}"'


_DEFAULT_MAX_CHUNK_BYTES = 16 * 1024 * 1024


def _parse_range_stdout(stdout: str, *, expected_start: int) -> RangeResult:
    if not stdout:
        return RangeResult(success=False, error="empty stdout")

    size: int | None = None
    b64_text: str | None = None
    err_text: str | None = None
    for line in stdout.splitlines():
        if line.startswith("SIZE="):
            try:
                size = int(line[5:].strip())
            except ValueError:
                return RangeResult(success=False, error=f"bad SIZE line: {line!r}")
        elif line.startswith("B64="):
            b64_text = line[4:]
        elif line.startswith("ERR="):
            err_text = line[4:]
    if size is None or b64_text is None:
        return RangeResult(success=False, error=f"missing SIZE/B64 (err={err_text!r})")

    if size < 0:
        if b64_text:
            return RangeResult(success=False, error=f"size={size} but B64 non-empty")
        return RangeResult(success=True, remote_size=size, delta=b"", error=err_text)

    expected_delta_len = max(0, size - expected_start)
    if expected_delta_len == 0:
        if b64_text:
            return RangeResult(success=False, error="expected empty delta but B64 non-empty")
        return RangeResult(success=True, remote_size=size, delta=b"")

    try:
        delta = base64.b64decode(b64_text, validate=True)
    except (ValueError, base64.binascii.Error) as e:
        return RangeResult(success=False, error=f"base64 decode: {e}")

    if len(delta) > expected_delta_len:
        return RangeResult(
            success=False,
            error=f"delta {len(delta)} exceeds expected {expected_delta_len}",
        )

    return RangeResult(success=True, remote_size=size, delta=delta)


def download_file_range(
    vm_config: RemoteVMConfig,
    remote_path: str,
    *,
    start: int,
    timeout: float = 60,
    max_chunk_bytes: int = _DEFAULT_MAX_CHUNK_BYTES,
) -> RangeResult:
    url = f"{_cua_url(vm_config)}/cmd"
    if vm_config.is_linux:
        cmd = _build_range_cmd_linux(remote_path, start, max_chunk_bytes)
    else:
        cmd = _build_range_cmd_windows(remote_path, start, max_chunk_bytes)

    payload = {"command": "run_command", "params": {"command": cmd}}
    try:
        with requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            stream=True,
        ) as resp:
            data = _read_first_sse_event(resp, read_timeout=timeout)
    except requests.Timeout:
        return RangeResult(success=False, error="timeout")
    except Exception as e:
        return RangeResult(success=False, error=f"transport: {e}")

    if data is None:
        return RangeResult(success=False, error="no SSE event")

    rc = data.get("return_code", data.get("returncode", 0))
    stdout = data.get("stdout", data.get("output", "")) or ""
    if rc != 0:
        return RangeResult(success=False, error=f"rc={rc} stderr={data.get('stderr', '')[:200]}")

    return _parse_range_stdout(stdout, expected_start=start)


# ======================================================================
# Directory listing (used by gather.pull_dir)
# ======================================================================


def list_remote_dir(vm_config: RemoteVMConfig, remote_dir: str, timeout: float = 30) -> list[dict[str, Any]]:
    """Return a flat list of entries under ``remote_dir`` (recursive).

    Each entry: ``{"relpath": "<posix-style>", "is_dir": <bool>, "size": <int>}``.
    Empty list if the directory is missing.
    """
    if vm_config.is_linux:
        safe = remote_dir.replace("'", "'\\''")
        cmd = (
            f"if [ ! -d '{safe}' ]; then echo '[]'; exit 0; fi; "
            f"cd '{safe}' && find . -mindepth 1 -printf '%P\\t%y\\t%s\\n'"
        )
    else:
        safe = remote_dir.replace("'", "''")
        cmd = (
            'powershell -NoProfile -Command "'
            f"if (-not (Test-Path -LiteralPath '{safe}' -PathType Container)) {{ '[]'; exit 0 }}; "
            f"Get-ChildItem -LiteralPath '{safe}' -Recurse | ForEach-Object {{ "
            f"  $rel = $_.FullName.Substring(('{safe}').Length).TrimStart('\\\\','/'); "
            f"  $type = if ($_.PSIsContainer) {{ 'd' }} else {{ 'f' }}; "
            f"  $size = if ($_.PSIsContainer) {{ 0 }} else {{ $_.Length }}; "
            f"  Write-Output ($rel + [char]9 + $type + [char]9 + $size) "
            f"}}"
            '"'
        )

    result = run_remote(vm_config, cmd, timeout=timeout)
    if result.returncode != 0:
        return []
    out = (result.stdout or "").strip()
    if not out or out == "[]":
        return []

    entries: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rel, kind, size = parts[0], parts[1], parts[2]
        is_dir = kind == "d"
        try:
            size_i = int(size)
        except ValueError:
            size_i = 0
        rel_posix = rel.replace("\\", "/")
        entries.append({"relpath": rel_posix, "is_dir": is_dir, "size": size_i})
    return entries
