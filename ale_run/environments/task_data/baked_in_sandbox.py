"""``task_data_source: baked_in_sandbox`` — task data ships with the image.

Convention on the sandbox image:

  <sandbox.task_data_root>/<domain>/<task>/<variant>/
  ├── input/                ← visible from stage_inputs onward
  └── reference.7z          ← password-encrypted; opaque to the agent;
                              decrypted into reference/ at stage_reference

The agent's task code reads from input/. reference.7z stays encrypted
during the run so the agent can't peek at the answer; evaluation
decrypts it just-in-time.

Canonical archive layout is *flat*: the reference files sit at the
archive root (no wrapping ``reference/`` dir), so they extract straight
into ``<base>/reference/``. A legacy layout that wraps everything in a
single top-level ``reference/`` dir is also tolerated — stage_reference
detects and flattens it, so graders always find ``<base>/reference/<file>``.

The password is read by the host orchestrator from
``ALE_REFERENCE_ARCHIVE_PASSWORD``. It is deliberately absent from the
framework source copied into the sandbox. Both Linux and Windows images
have ``7z`` on PATH.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import join, shell_q, task_subdir

logger = logging.getLogger(__name__)


async def _repoint_evaluator_venv_home(
    sandbox: SandboxHandle,
    target: str,
) -> None:
    """Windows-only: fix a baked ``reference/evaluator_env/.venv`` whose
    ``pyvenv.cfg`` ``home =`` points at the *image it was built on* rather than
    this one.

    Some graders shell out to a vendored evaluator venv
    (``reference\\evaluator_env\\.venv\\Scripts\\python.exe``). The venv was
    created on one image (e.g. ale-win10, where base Python sits under the
    per-user ``...\\AppData\\Local\\Programs\\Python\\Python312``) and baked into
    ``reference.7z``; on a different image (ale-win-server, base Python at
    ``C:\\Python312``) that ``home`` path is dangling, so the venv launcher
    aborts and the grader scores 0 despite correct agent output. A single
    archive can't carry a home that's valid on both images, so we repair it at
    stage time: if ``home`` is dangling, repoint it to a real python3.12 on this
    image. No-op when the baked ``home`` already exists (the image that built
    the venv is left untouched) and for tasks without an evaluator venv.
    """
    if sandbox.is_linux:
        return
    cfg = join(sandbox, target, "evaluator_env", ".venv", "pyvenv.cfg")
    if not await sandbox.exists(cfg):
        return
    script = (
        "\n".join(
            [
                f"$cfg = '{cfg}'",
                "$lines = Get-Content -LiteralPath $cfg",
                # current baked home; if it already resolves, leave it (keeps the image
                # that built the venv — e.g. ale-win10 — untouched).
                "$vhome = ($lines | Where-Object {$_ -match '^home\\s*='}) -replace '^home\\s*=\\s*',''",
                "if ($vhome -and (Test-Path (Join-Path $vhome 'python.exe'))) { exit 0 }",
                # candidate base interpreters: the image's system Python first, then any
                # python on PATH. Accept the first that exists and is a base install
                # (a venv dir carries a pyvenv.cfg; pointing home at a venv lacks the
                # base DLLs/layout). Version match (3.12) is guaranteed on this image
                # and verified below by actually starting the venv.
                "$cands = @('C:\\Python312\\python.exe')",
                "$wp = (Get-Command python -All -EA SilentlyContinue).Source; if ($wp) { $cands += $wp }",
                "$found = $null",
                "foreach ($c in $cands) {",
                "  if ((Test-Path $c) -and -not (Test-Path (Join-Path (Split-Path $c -Parent) 'pyvenv.cfg'))) { $found = $c; break }",
                "}",
                "if (-not $found) { Write-Error 'no base python to repoint evaluator venv'; exit 1 }",
                "$newhome = Split-Path $found -Parent",
                "($lines -replace '^home\\s*=.*$', \"home = $newhome\") | Set-Content -LiteralPath $cfg",
                # self-validate by exit code: the venv's own launcher must start now.
                "$venvpy = Join-Path (Split-Path $cfg -Parent) 'Scripts\\python.exe'",
                "& $venvpy -c 'import sys' 2>$null | Out-Null",
                "if ($LASTEXITCODE -ne 0) { Write-Error 'venv still broken after repoint'; exit 2 }",
                'Write-Output "repointed:$newhome"',
            ]
        )
        + "\n"
    )
    enc = base64.b64encode(script.encode("utf-16-le")).decode()
    r = await sandbox.run_command(
        f"powershell -NoProfile -EncodedCommand {enc}",
        timeout=60,
    )
    if r.returncode == 0 and "repointed:" in (r.stdout or ""):
        logger.info(
            "baked_in_sandbox: repointed dangling evaluator venv home (%s)",
            (r.stdout or "").strip().split("repointed:")[-1].strip(),
        )
    elif r.returncode != 0:
        logger.warning(
            "baked_in_sandbox: evaluator venv repoint failed (rc=%d): %s",
            r.returncode,
            (r.stderr or r.stdout or "")[:200],
        )


async def stage_input(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    source: str,
) -> dict[str, Any]:
    """Assert input/ is on the sandbox; make output/. Reference stays
    locked (.7z) until ``stage_reference``."""
    _ = source
    base = task_subdir(sandbox, task_data)
    input_dir = join(sandbox, base, "input")
    if not await sandbox.exists(input_dir):
        raise RuntimeError(
            f"task_data_source=baked_in_sandbox: expected baked input at "
            f"{input_dir!r}, not found on sandbox. Re-bake the image."
        )
    await sandbox.mkdir(join(sandbox, base, "output"))
    # Make baked software/ scripts executable (mirrors the gs:// staging path).
    # The image preserves original file modes, often 0644, so tasks that exec a
    # software wrapper (e.g. .../software/run.sh) would otherwise hit
    # "Permission denied" under baked_in_sandbox.
    software_dir = join(sandbox, base, "software")
    if sandbox.is_linux and await sandbox.exists(software_dir):
        await sandbox.run_command(
            f"find {shell_q(sandbox, software_dir)} -type f -exec chmod +x {{}} +",
            timeout=60,
        )
    return {"staged": ["input"], "source": "baked_in_sandbox"}


async def stage_reference(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    source: str,
) -> dict[str, Any]:
    """Decrypt ``reference.7z`` → ``<base>/reference/`` on the sandbox.

    Tasks without reference data have no reference.7z; we skip cleanly.
    Always wipes any existing reference/ first (defends against stale
    state from a prior run on the same sandbox).

    Extracts into a temp dir, then normalises: a flat archive (canonical)
    is moved into ``reference/`` as-is; a legacy archive whose only
    top-level entry is a ``reference/`` dir is flattened by promoting that
    dir. Either way the result is ``<base>/reference/<file>``.
    """
    _ = source
    base = task_subdir(sandbox, task_data)
    archive = join(sandbox, base, "reference.7z")
    target = join(sandbox, base, "reference")
    tmp = join(sandbox, base, ".reference_extract")

    if not await sandbox.exists(archive):
        return {"skipped": True, "reason": "no_reference_7z"}

    await sandbox.rm([target, tmp])
    await sandbox.mkdir(tmp)

    pwd = os.environ.get("ALE_REFERENCE_ARCHIVE_PASSWORD", "").strip()
    if not pwd:
        await sandbox.rm([tmp])
        raise RuntimeError(
            "ALE_REFERENCE_ARCHIVE_PASSWORD is required to decrypt baked reference data"
        )
    a, t = shell_q(sandbox, archive), shell_q(sandbox, tmp)
    if sandbox.is_linux:
        cmd = f"7z x -p{shell_q(sandbox, pwd)} {a} -o{t} -y"
    else:
        # PowerShell with single-quoted strings (no interpolation).
        cmd = f"powershell -NoProfile -Command \"7z x -p'{pwd}' {a} -o{t} -y\""
    r = await sandbox.run_command(cmd, timeout=300)
    if r.returncode != 0:
        await sandbox.rm([tmp])
        raise RuntimeError(
            f"7z decrypt {archive} failed (rc={r.returncode}): {(r.stderr or r.stdout or '')[:300]}"
        )

    # Promote the payload to ``target``. ``list_dir`` is recursive, so the
    # archive is the legacy ``reference/``-wrapped layout iff every entry's
    # first path segment is ``reference``; otherwise it's the flat layout.
    entries = await sandbox.list_dir(tmp)
    tops = {e["relpath"].replace("\\", "/").split("/", 1)[0] for e in entries}
    src = join(sandbox, tmp, "reference") if tops == {"reference"} else tmp
    src_q, tgt_q = shell_q(sandbox, src), shell_q(sandbox, target)
    if sandbox.is_linux:
        mv_cmd = f"mv {src_q} {tgt_q}"
    else:
        mv_cmd = (
            f'powershell -NoProfile -Command "Move-Item -LiteralPath {src_q} -Destination {tgt_q}"'
        )
    mr = await sandbox.run_command(mv_cmd, timeout=120)
    await sandbox.rm([tmp])
    if mr.returncode != 0:
        raise RuntimeError(
            f"reference normalise move failed (rc={mr.returncode}): "
            f"{(mr.stderr or mr.stdout or '')[:300]}"
        )
    logger.info("baked_in_sandbox: decrypted %s → %s", archive, target)
    # Repair a vendored evaluator venv whose baked home points at a different
    # image's base Python (Windows; no-op otherwise / when home is valid).
    await _repoint_evaluator_venv_home(sandbox, target)
    return {"staged": ["reference"], "source": "baked_in_sandbox", "decrypted_from": "reference.7z"}
