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

The password is a project-wide constant (this is throwaway benchmark
infrastructure — the encryption stops the agent from reading the
answer, not external attackers). Both linux and windows images have
``7z`` 26.01 on PATH (installer adds it to system PATH).
"""
from __future__ import annotations

import logging
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import join, shell_q, task_subdir

logger = logging.getLogger(__name__)


# Project-wide reference-archive password. Plain string — not a secret
# in the security sense (anyone with image access could read it anyway),
# but stops the agent from passively reading the answer.
_REFERENCE_PASSWORD = "rdi-ucberkeley-Gov8EV7wGHYAc7XQBzhd"


async def stage_input(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
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
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
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

    pwd = _REFERENCE_PASSWORD
    a, t = shell_q(sandbox, archive), shell_q(sandbox, tmp)
    if sandbox.is_linux:
        cmd = f"7z x -p{shell_q(sandbox, pwd)} {a} -o{t} -y"
    else:
        # PowerShell with single-quoted strings (no interpolation).
        cmd = (
            'powershell -NoProfile -Command "'
            f"7z x -p'{pwd}' {a} -o{t} -y"
            '"'
        )
    r = await sandbox.run_command(cmd, timeout=300)
    if r.returncode != 0:
        await sandbox.rm([tmp])
        raise RuntimeError(
            f"7z decrypt {archive} failed (rc={r.returncode}): "
            f"{(r.stderr or r.stdout or '')[:300]}"
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
            'powershell -NoProfile -Command "'
            f"Move-Item -LiteralPath {src_q} -Destination {tgt_q}"
            '"'
        )
    mr = await sandbox.run_command(mv_cmd, timeout=120)
    await sandbox.rm([tmp])
    if mr.returncode != 0:
        raise RuntimeError(
            f"reference normalise move failed (rc={mr.returncode}): "
            f"{(mr.stderr or mr.stdout or '')[:300]}"
        )
    logger.info("baked_in_sandbox: decrypted %s → %s", archive, target)
    return {"staged": ["reference"], "source": "baked_in_sandbox",
            "decrypted_from": "reference.7z"}
