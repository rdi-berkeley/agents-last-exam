"""``task_data_source: baked_in_sandbox`` — task data ships with the image.

Convention on the sandbox image:

  <sandbox.task_data_root>/<domain>/<task>/<variant>/
  ├── input/                ← visible from stage_inputs onward
  └── reference.7z          ← password-encrypted; opaque to the agent;
                              decrypted into reference/ at stage_reference

The agent's task code reads from input/. reference.7z stays encrypted
during the run so the agent can't peek at the answer; evaluation
decrypts it just-in-time.

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
    return {"staged": ["input"], "source": "baked_in_sandbox"}


async def stage_reference(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    """Decrypt ``reference.7z`` → ``reference/`` on the sandbox.

    Tasks without reference data have no reference.7z; we skip cleanly.
    Always wipes any existing reference/ first (defends against stale
    state from a prior run on the same sandbox)."""
    _ = source
    base = task_subdir(sandbox, task_data)
    archive = join(sandbox, base, "reference.7z")
    target = join(sandbox, base, "reference")

    if not await sandbox.exists(archive):
        return {"skipped": True, "reason": "no_reference_7z"}

    await sandbox.rm([target])

    pwd = _REFERENCE_PASSWORD
    a, t = shell_q(sandbox, archive), shell_q(sandbox, target)
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
        raise RuntimeError(
            f"7z decrypt {archive} failed (rc={r.returncode}): "
            f"{(r.stderr or r.stdout or '')[:300]}"
        )
    logger.info("baked_in_sandbox: decrypted %s → %s", archive, target)
    return {"staged": ["reference"], "source": "baked_in_sandbox",
            "decrypted_from": "reference.7z"}
