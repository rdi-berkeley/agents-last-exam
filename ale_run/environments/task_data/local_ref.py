"""``task_data_source: localref:<host_dir>`` — inputs baked in the image, the hidden
reference staged from a HOST directory only after the agent has finished.

Layout on the host (ALE's canonical one, e.g. a checkout of the data mirror):

    <host_dir>/<domain>/<task>/<variant>/reference/...

``stage_input`` is ``baked_in_sandbox``'s (input/ + software/ are in the image);
``stage_reference`` zips ``reference/`` on the host, uploads the archive (Daytona
SDK when the sandbox is a Daytona one, the cua base64 path otherwise), expands
it into ``<task dir>\\reference`` and repairs a vendored evaluator venv like the
baked backend does. The answers never exist in the sandbox while the agent runs.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import zipfile
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import baked_in_sandbox, join, shell_q, task_subdir

logger = logging.getLogger(__name__)

_PREFIX = "localref:"


def host_reference_dir(source: str, task_data: TaskDataSpec) -> str:
    root = os.path.expanduser(source[len(_PREFIX):])
    return os.path.join(root, task_data.domain_name or "", task_data.task_name or "",
                        task_data.variant_name or "", "reference")


def zip_dir_to_file(src_dir: str, zip_path: str, *, store: bool = False) -> int:
    """Zip a directory's contents (paths relative to it) to ``zip_path``; returns the file count.
    UTF-8 names (zipfile sets the flag); extracted with 7-Zip on Windows, which honours it."""
    n = 0
    method = zipfile.ZIP_STORED if store else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(zip_path, "w", method, compresslevel=1, allowZip64=True) as zf:
        for dirpath, dirnames, filenames in os.walk(src_dir):
            dirnames.sort()
            rel_dir = os.path.relpath(dirpath, src_dir)
            if rel_dir != "." and not filenames and not dirnames:
                zf.writestr(rel_dir.replace(os.sep, "/") + "/", b"")
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                zf.write(full, os.path.relpath(full, src_dir).replace(os.sep, "/"))
                n += 1
    return n


def zip_dir(src_dir: str) -> bytes:
    """In-memory variant for small trees (references)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        path = tmp.name
    try:
        zip_dir_to_file(src_dir, path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


async def upload_path(sandbox: SandboxHandle, local_path: str, remote_path: str) -> None:
    """Stream a host file into the sandbox (Daytona SDK streams from a path)."""
    if sandbox.metadata.get("provider") == "daytona":
        from daytona import Daytona, DaytonaConfig

        kwargs: dict[str, Any] = {"api_key": os.environ["DAYTONA_API_KEY"]}
        if sandbox.metadata.get("daytona_target"):
            kwargs["target"] = sandbox.metadata["daytona_target"]
        client = Daytona(DaytonaConfig(**kwargs))
        sb = await asyncio.to_thread(client.get, sandbox.metadata.get("sandbox_id") or sandbox.id)
        await asyncio.to_thread(sb.fs.upload_file, local_path, remote_path, 7200)
        return
    await sandbox.upload_local_file(local_path, remote_path)


async def upload_bytes(sandbox: SandboxHandle, data: bytes, remote_path: str) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        await upload_path(sandbox, path, remote_path)
    finally:
        os.unlink(path)


async def expand_zip(sandbox: SandboxHandle, zip_path: str, target: str, *, clear: bool = True) -> int:
    """Expand the archive into ``target`` (replacing it when ``clear``); returns the file count."""
    if sandbox.is_linux:
        cmd = ((f"rm -rf {shell_q(sandbox, target)} && " if clear else "") + f"mkdir -p {shell_q(sandbox, target)} && "
               f"unzip -q -o {shell_q(sandbox, zip_path)} -d {shell_q(sandbox, target)} && "
               f"rm -f {shell_q(sandbox, zip_path)} && find {shell_q(sandbox, target)} -type f | wc -l")
    else:
        # 7-Zip, not Expand-Archive: .NET's extractor decodes UTF-8 member names with
        # the system code page and then fails on them (Chinese PDF names in ar_full).
        cmd = ('powershell -NoProfile -Command "'
               f"$ErrorActionPreference='Stop'; $t={shell_q(sandbox, target)}; "
               + ("if (Test-Path -LiteralPath $t) { Remove-Item -LiteralPath $t -Recurse -Force }; " if clear else "") +
               "New-Item -ItemType Directory -Force $t | Out-Null; "
               "$sz = @('C:\\Program Files\\7-Zip\\7z.exe', 'C:\\ProgramData\\chocolatey\\bin\\7z.exe') | Where-Object { Test-Path $_ } | Select-Object -First 1; "
               f"if ($sz) {{ & $sz x -y -bso0 -bsp0 ('-o' + $t) {shell_q(sandbox, zip_path)} | Out-Null; if ($LASTEXITCODE -ne 0) {{ throw ('7z exit ' + $LASTEXITCODE) }} }} "
               f"else {{ Expand-Archive -LiteralPath {shell_q(sandbox, zip_path)} -DestinationPath $t -Force }}; "
               f"Remove-Item -LiteralPath {shell_q(sandbox, zip_path)} -Force; "
               "(Get-ChildItem -LiteralPath $t -Recurse -File | Measure-Object).Count\"")
    r = await sandbox.run_command(cmd, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"expanding {zip_path} into {target} failed (rc={r.returncode}): "
                           f"{(r.stderr or r.stdout or '')[:300]}")
    try:
        return int((r.stdout or "0").strip().splitlines()[-1])
    except ValueError:
        return -1


async def stage_input(sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str) -> dict[str, Any]:
    _ = source
    report = await baked_in_sandbox.stage_input(sandbox, task_data, source="baked_in_sandbox")
    report["source"] = "localref"
    return report


async def stage_reference(sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str) -> dict[str, Any]:
    ref = host_reference_dir(source, task_data)
    if not os.path.isdir(ref):
        return {"skipped": True, "reason": "no_reference", "host_dir": ref}
    base = task_subdir(sandbox, task_data)
    target = join(sandbox, base, "reference")
    data = await asyncio.to_thread(zip_dir, ref)
    staging = join(sandbox, sandbox.task_data_root, "_stage")
    await sandbox.mkdir(staging)
    zip_path = join(sandbox, staging, f"{task_data.task_name}_reference.zip")
    await upload_bytes(sandbox, data, zip_path)
    n = await expand_zip(sandbox, zip_path, target)
    await baked_in_sandbox._repoint_evaluator_venv_home(sandbox, target)
    logger.info("localref: staged reference %s -> %s:%s (%d files, %.1f MB zipped)",
                ref, sandbox.id, target, n, len(data) / 1e6)
    return {"staged": ["reference"], "source": "localref", "files": n}
