"""Normalize the various command-result shapes ``session.run_command`` returns.

cua-bench's ``RemoteDesktopSession.run_command`` returns a ``dict``:
    {"success": bool, "stdout": str, "stderr": str, "return_code": int}

Our ``StubDesktopSession`` returns a dataclass with
``.stdout / .stderr / .exit_code`` attributes.

Some agenthle code paths used ``.returncode`` (no underscore).

These helpers extract the canonical fields regardless of shape so every
caller can write::

    cr = await session.run_command(...)
    if cmd_ok(cr):
        text = cmd_stdout(cr)
"""
from __future__ import annotations

from typing import Any


def cmd_rc(cr: Any) -> int:
    """Return the command's exit code. ``-1`` if not determinable."""
    if isinstance(cr, dict):
        for key in ("return_code", "returncode", "exit_code"):
            if key in cr and cr[key] is not None:
                return int(cr[key])
        return -1
    for attr in ("exit_code", "returncode", "return_code"):
        val = getattr(cr, attr, None)
        if val is not None:
            return int(val)
    return -1


def cmd_ok(cr: Any) -> bool:
    return cmd_rc(cr) == 0


def cmd_stdout(cr: Any) -> str:
    if isinstance(cr, dict):
        return cr.get("stdout") or ""
    return getattr(cr, "stdout", "") or ""


def cmd_stderr(cr: Any) -> str:
    if isinstance(cr, dict):
        return cr.get("stderr") or ""
    return getattr(cr, "stderr", "") or ""
