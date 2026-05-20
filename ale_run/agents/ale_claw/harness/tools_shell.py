"""Remote-VM shell execution tool: exec (US-OC-057).

One BaseTool subclass — :class:`ExecTool` — that runs a single shell command
inside the remote VM via ``session.interface.run_command`` (computer-server
RPC → cmd.exe on Windows, ``/bin/sh`` on POSIX) and returns structured
``stdout``/``stderr``/``exit_code``. Non-GUI only.

Kept from OpenClaw (``openclaw/src/agents/bash-tools.exec.ts``,
``bash-tools.exec-runtime.ts``):
  - Foreground-only contract: ``{status, exitCode, durationMs, aggregated}``
    return shape mirrors ``ExecProcessOutcome``.
  - ``DEFAULT_MAX_OUTPUT = 200_000`` chars per stream (``PI_BASH_MAX_OUTPUT_CHARS``
    default at :106-111). Middle truncation preserves head + tail so the final
    exit/error lines stay visible — matches ``truncateMiddle``.
  - Workspace-only ``cwd`` policy via :func:`_assert_within_workspace`
    (shared with ``tools_fs.py``).

Dropped:
  - Background / ``yieldMs`` paths (need a per-task process registry —
    deferred to US-OC-061).
  - PTY mode (no computer-server PTY RPC).
  - Elevated / sudo, approvals/allowlists, exec-host routing, docker sandbox.
  - Script-preflight shell-bleed validation (``run_command`` RPC takes a
    single string — no argv injection surface).
  - ``env`` dict injection (RPC doesn't forward env; users can inline
    ``set KEY=VAL && cmd`` if needed).

Key difference from OpenClaw:
  - ``timeout`` is **client-side only**. ``asyncio.wait_for`` cancels our
    await; the VM-side subprocess may keep running until process management
    lands (US-OC-061). The tool surfaces ``timed_out=True`` so the agent
    doesn't silently assume the process is dead.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional, Union

from agent.tools.base import BaseTool, register_tool

from .tools_fs import (
    _assert_within_workspace,
    _get_required_str,
    _is_windows_path,
    _run_async,
)

if TYPE_CHECKING:
    from computer.interface import BaseComputerInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (match OpenClaw bash-tools.exec-runtime.ts:106-117)
# ---------------------------------------------------------------------------

_DEFAULT_MAX_OUTPUT_CHARS = 200_000
_DEFAULT_TIMEOUT_SECONDS = 60
_MIN_TIMEOUT_SECONDS = 1
_MAX_TIMEOUT_SECONDS = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_middle(s: str, cap: int) -> tuple[str, bool]:
    """Keep head + tail halves of ``s`` if it exceeds ``cap`` chars.

    Returns ``(truncated_string, was_truncated)``. When truncated, inserts
    a marker in the middle showing how many chars were dropped. Mirrors
    OpenClaw's ``truncateMiddle`` behavior.
    """
    if cap <= 0 or len(s) <= cap:
        return s, False
    half = cap // 2
    head = s[:half]
    tail = s[-(cap - half):] if (cap - half) > 0 else ""
    omitted = len(s) - len(head) - len(tail)
    marker = f"\n\n[... output truncated: {omitted} chars omitted ...]\n\n"
    return head + marker + tail, True


def _wrap_with_cwd(command: str, cwd: Optional[str]) -> str:
    """Prefix a ``cd`` to bind the command to a working directory.

    Windows-style ``cwd`` → ``cd /d "<cwd>" && <cmd>`` (cmd.exe supports
    ``/d`` for cross-drive changes). POSIX-style ``cwd`` → ``cd "<cwd>" && <cmd>``.
    Returns ``command`` unchanged when ``cwd`` is ``None`` or empty.
    """
    if not cwd:
        return command
    # Escape embedded double-quotes so cmd.exe/sh don't mis-parse the prefix.
    safe_cwd = cwd.replace('"', '\\"')
    if _is_windows_path(cwd):
        return f'cd /d "{safe_cwd}" && {command}'
    return f'cd "{safe_cwd}" && {command}'


def _resolve_timeout(raw: object, default: int) -> int:
    """Clamp a user-supplied timeout to ``[_MIN_TIMEOUT_SECONDS, _MAX_TIMEOUT_SECONDS]``.

    Falls back to ``default`` when ``raw`` is missing or not a positive number.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return default
    if raw <= 0:
        return default
    clamped = int(raw)
    if clamped < _MIN_TIMEOUT_SECONDS:
        return _MIN_TIMEOUT_SECONDS
    if clamped > _MAX_TIMEOUT_SECONDS:
        return _MAX_TIMEOUT_SECONDS
    return clamped


# ---------------------------------------------------------------------------
# ExecTool
# ---------------------------------------------------------------------------


@register_tool("exec")
class ExecTool(BaseTool):
    """Run a single non-GUI shell command inside the remote VM.

    Uses ``interface.run_command`` (cmd.exe on Windows, ``/bin/sh`` on POSIX).
    Adapted from OpenClaw ``createExecTool`` (bash-tools.exec.ts) — foreground
    happy path only.
    """

    def __init__(
        self,
        interface: "BaseComputerInterface",
        workspace_root: Optional[str] = None,
        max_output_chars: Optional[int] = None,
        default_timeout: Optional[int] = None,
        cfg: Optional[dict] = None,
    ):
        self.interface = interface
        self.workspace_root = workspace_root
        self.max_output_chars = (
            int(max_output_chars)
            if isinstance(max_output_chars, (int, float)) and max_output_chars > 0
            else _DEFAULT_MAX_OUTPUT_CHARS
        )
        self.default_timeout = _resolve_timeout(default_timeout, _DEFAULT_TIMEOUT_SECONDS)
        if workspace_root is None:
            logger.info("ExecTool: workspace_root is None — permissive cwd policy")
        super().__init__(cfg)

    @property
    def description(self) -> str:
        return (
            "Run a single non-GUI shell command inside the remote VM and return "
            "stdout/stderr/exit_code. cmd.exe on Windows, /bin/sh on POSIX. "
            "GUI apps block until they exit — use the computer tool for GUI work."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Shell command to run. On Windows prefer direct executables "
                        "(dir, type, where, python3); use `powershell -NoProfile "
                        "-Command \"...\"` explicitly for PowerShell."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Working directory on the VM. Must resolve inside the task "
                        "workspace. Emulated via a `cd` prefix — no RPC field."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        f"Client-side timeout in seconds (default "
                        f"{_DEFAULT_TIMEOUT_SECONDS}, max {_MAX_TIMEOUT_SECONDS}). "
                        "On expiry the VM-side process may keep running."
                    ),
                },
            },
            "required": ["command"],
        }

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        try:
            parsed = self._verify_json_format_args(params)
            command = _get_required_str(parsed, "command", "exec")
            cwd_raw = parsed.get("cwd")
            cwd: Optional[str] = None
            if cwd_raw is not None:
                if not isinstance(cwd_raw, str) or not cwd_raw.strip():
                    raise ValueError('exec: "cwd" must be a non-empty string when provided')
                cwd = cwd_raw
                _assert_within_workspace(cwd, self.workspace_root)
            timeout_seconds = _resolve_timeout(parsed.get("timeout"), self.default_timeout)
        except ValueError as e:
            return {"success": False, "error": f"Error: {e}"}

        try:
            return _run_async(self._execute(command, cwd, timeout_seconds))
        except Exception as e:  # noqa: BLE001 — surface RPC errors as tool errors
            logger.error("exec tool failure for %r: %s", command, e)
            return {"success": False, "error": f"Error: {e}"}

    async def _execute(
        self,
        command: str,
        cwd: Optional[str],
        timeout_seconds: int,
    ) -> dict:
        wrapped = _wrap_with_cwd(command, cwd)
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.interface.run_command(wrapped),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            duration_ms = int((time.monotonic() - t0) * 1000)
            return {
                "success": False,
                "status": "failed",
                "error": (
                    f"Error: exec timed out after {timeout_seconds}s "
                    "(VM-side process may still be running)"
                ),
                "timed_out": True,
                "duration_ms": duration_ms,
                "cwd": cwd,
            }

        duration_ms = int((time.monotonic() - t0) * 1000)
        stdout_raw = getattr(result, "stdout", "") or ""
        stderr_raw = getattr(result, "stderr", "") or ""
        exit_code = getattr(result, "returncode", 0) or 0

        stdout, out_truncated = _truncate_middle(stdout_raw, self.max_output_chars)
        stderr, err_truncated = _truncate_middle(stderr_raw, self.max_output_chars)
        truncated = out_truncated or err_truncated

        status = "completed" if exit_code == 0 else "failed"
        return {
            "success": True,
            "status": status,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": truncated,
            "cwd": cwd,
        }
