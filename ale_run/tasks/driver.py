"""TaskDriver — drives a task's setup/evaluate hooks on an open session.

Ported from simprun/task_env.py. Decoupled from VM provisioning: the
constructor takes an already-open ``cua_bench.computers.remote.RemoteDesktopSession``
from the environment layer plus an optional ``session_rebuilder`` async
callback. The provider is responsible for ``wait_cua_ready`` and
initializing the underlying ``Computer`` object; TaskDriver owns the task
setup + evaluate retry loop and — when the session goes transient-bad —
calls ``session_rebuilder()`` to get a fresh one, exactly as
simprun's ``TaskEnv.close(force=True) + connect()`` does.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict

from cua_bench.computers.remote import RemoteDesktopSession

from .loader import TaskLoader

logger = logging.getLogger(__name__)

_CUA_COMMAND_ATTEMPTS = 8
_CUA_COMMAND_BASE_DELAY_S = 5
_CUA_SESSION_ATTEMPTS = 4
_CUA_SESSION_CLOSE_TIMEOUT_S = 10

# Detached run_command (detach + poll). A task's evaluate() runs its scorer
# via one ``session.run_command`` that holds a SINGLE cua HTTP connection open
# for the whole command — for a long eval (minutes→tens of minutes) that
# connection eventually gets dropped by the network and the result (which lives
# only in that stream, the server doesn't persist it) is lost, hanging the unit.
# Instead we run the command in the background ON the VM, redirect
# stdout/stderr/exit-code to files + a done-marker, then poll the marker with
# SHORT calls and read the result files. Each call is brief (drop-resistant) and
# the result survives connection blips because it lives on disk.
# Poll backs off from quick early checks (so fast scorers stay fast) up to a
# 30s ceiling for long-running ones.
_DETACHED_POLL_BACKOFF_S = (1, 1, 2, 3, 5, 10, 15)
_DETACHED_POLL_MAX_S = 30
# Overall ceiling for the detached command. Kept just under the eval phase
# wall-clock (lifecycle._EVAL_TIMEOUT_S = 3600) so the poll loop surfaces a
# clean error before the phase-level wait_for fires.
_DETACHED_TIMEOUT_S = 3300


class _DetachedSetupError(Exception):
    """Raised when the detached-run scaffolding (mkdir/write/launch) fails —
    signals the caller it is safe to fall back to a direct run_command (the
    command has NOT started yet). Failures AFTER launch raise their own
    exceptions and must NOT trigger a re-run."""

_TRANSIENT_ERROR_SNIPPETS = (
    "Request failed",
    "Server returned malformed response",
    "Failed to establish WebSocket",
    "connection_lost",
    "Connection reset",
    "Connection refused",
    "Cannot connect",
    "timed out",
    "Timeout",
    "no response",
    "did not become ready",
    "generator didn't stop after throw",
)


def _is_transient_cua_result(result: dict | None) -> bool:
    if not result or result.get("success", True):
        return False
    text = " ".join(str(result.get(key, "")) for key in ("error", "message", "detail", "stderr"))
    return _is_transient_cua_error(text)


def _is_transient_cua_error(error: object) -> bool:
    text = str(error)
    return any(snippet in text for snippet in _TRANSIENT_ERROR_SNIPPETS)


async def _sleep_before_retry(attempt: int) -> None:
    delay = min(_CUA_COMMAND_BASE_DELAY_S * (2**attempt), 30)
    await asyncio.sleep(delay)


async def _detached_run_command(
    interface: Any, raw_run_command: Any, command: str, *, os_type: str,
) -> Any:
    """Run ``command`` on the VM in the background, polling a done-marker, and
    return a ``CommandResult`` read from on-disk stdout/stderr/exit-code files.

    All VM I/O here uses short, drop-resistant calls (``raw_run_command`` is the
    UN-wrapped interface.run_command captured before wrapping, so no recursion).
    The command is written to a script file (not interpolated into a shell line)
    so arbitrary quoting in ``command`` is preserved exactly.
    """
    from computer.interface.models import CommandResult
    import uuid

    is_win = os_type != "linux"
    jid = "ale_job_" + uuid.uuid4().hex[:12]
    if is_win:
        jdir = rf"C:\Windows\Temp\{jid}"
        cmd_f, wrap_f = rf"{jdir}\cmd.bat", rf"{jdir}\wrap.bat"
        out_f, err_f, rc_f, done_f = (rf"{jdir}\out", rf"{jdir}\err", rf"{jdir}\rc", rf"{jdir}\done")
        mkdir = f'cmd /c if not exist "{jdir}" mkdir "{jdir}"'
        wrap = (
            "@echo off\r\n"
            f'call "{cmd_f}" > "{out_f}" 2> "{err_f}"\r\n'
            f'echo %ERRORLEVEL%> "{rc_f}"\r\n'
            f'echo done> "{done_f}"\r\n'
        )
        launch = f'cmd /c start "" /b cmd /c "{wrap_f}"'
        done_check = f'cmd /c if exist "{done_f}" (echo __DONE__) else (echo __WAIT__)'
        cleanup = f'cmd /c rmdir /s /q "{jdir}"'
        cmd_body = command + "\r\n"
    else:
        jdir = f"/tmp/{jid}"
        cmd_f, wrap_f = f"{jdir}/cmd.sh", f"{jdir}/wrap.sh"
        out_f, err_f, rc_f, done_f = (f"{jdir}/out", f"{jdir}/err", f"{jdir}/rc", f"{jdir}/done")
        mkdir = f"mkdir -p '{jdir}'"
        wrap = (
            "#!/bin/bash\n"
            f"bash '{cmd_f}' > '{out_f}' 2> '{err_f}'\n"
            f"echo $? > '{rc_f}'\n"
            f"touch '{done_f}'\n"
        )
        launch = f"nohup bash '{wrap_f}' >/dev/null 2>&1 &"
        done_check = f"[ -f '{done_f}' ] && echo __DONE__ || echo __WAIT__"
        cleanup = f"rm -rf '{jdir}'"
        cmd_body = command + "\n"

    # --- setup (mkdir + stage scripts + launch). Failure here is safe to fall
    #     back from: the command has NOT started. ---
    try:
        await raw_run_command(mkdir)
        await interface.write_text(cmd_f, cmd_body)
        await interface.write_text(wrap_f, wrap)
        await raw_run_command(launch)
    except Exception as e:  # noqa: BLE001
        raise _DetachedSetupError(str(e)) from e

    # --- collect: poll the done-marker (short, drop-resistant calls), backing
    #     off to a 30s ceiling, then read the on-disk result. Failures here
    #     propagate (the command already ran — never re-run it). ---
    deadline = asyncio.get_event_loop().time() + _DETACHED_TIMEOUT_S
    poll = 0
    while True:
        r = await raw_run_command(done_check)
        if "__DONE__" in (getattr(r, "stdout", "") or ""):
            break
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(f"detached command exceeded {_DETACHED_TIMEOUT_S}s")
        wait = _DETACHED_POLL_BACKOFF_S[poll] if poll < len(_DETACHED_POLL_BACKOFF_S) else _DETACHED_POLL_MAX_S
        poll += 1
        await asyncio.sleep(wait)

    out = (await interface.read_bytes(out_f)).decode("utf-8", errors="replace")
    err = (await interface.read_bytes(err_f)).decode("utf-8", errors="replace")
    rc_txt = (await interface.read_bytes(rc_f)).decode("utf-8", errors="replace").strip()
    try:
        rc = int(rc_txt)
    except (ValueError, TypeError):
        logger.warning("detached run_command: unparseable exit code %r; defaulting 0", rc_txt)
        rc = 0
    try:
        await raw_run_command(cleanup)
    except Exception:  # noqa: BLE001 -- cleanup is best-effort
        pass
    return CommandResult(stdout=out, stderr=err, returncode=rc)


def install_resilient_cua_commands(interface: Any, *, os_type: str = "linux") -> None:
    """Wrap the interface's _send_command with extra retries for transient errors,
    and make ``run_command`` robust for long evals via detach + poll.

    Idempotent — sets ``interface._ale_resilient_commands = True`` after the
    first install.
    """
    if getattr(interface, "_ale_resilient_commands", False):
        return

    original_send_command = interface._send_command

    async def resilient_send_command(command: str, params: dict | None = None):
        last_error: object = None
        for attempt in range(_CUA_COMMAND_ATTEMPTS):
            try:
                result = await original_send_command(command, params)
            except Exception as e:
                last_error = e
                if not _is_transient_cua_error(e) or attempt == _CUA_COMMAND_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "CUA command %r transport failed (attempt %d/%d): %s",
                    command,
                    attempt + 1,
                    _CUA_COMMAND_ATTEMPTS,
                    e,
                )
                await _sleep_before_retry(attempt)
                continue

            if not _is_transient_cua_result(result):
                return result

            last_error = result.get("error") or result.get("message") or result
            if attempt == _CUA_COMMAND_ATTEMPTS - 1:
                break
            logger.warning(
                "CUA command %r returned transient failure (attempt %d/%d): %s",
                command,
                attempt + 1,
                _CUA_COMMAND_ATTEMPTS,
                last_error,
            )
            await _sleep_before_retry(attempt)

        raise RuntimeError(
            f"CUA command {command!r} failed after {_CUA_COMMAND_ATTEMPTS} "
            f"attempts: {last_error}"
        )

    interface._send_command = resilient_send_command

    # Robust run_command: detach + poll so a long eval's result survives a
    # dropped long-lived connection. On ANY setup failure we fall back to the
    # original run_command — never worse than today's behaviour.
    raw_run_command = interface.run_command

    async def resilient_run_command(command: str):
        try:
            return await _detached_run_command(
                interface, raw_run_command, command, os_type=os_type,
            )
        except _DetachedSetupError as e:
            # Scaffolding failed before the command started → safe to run it
            # directly (never worse than the old single-connection path).
            logger.warning(
                "detached run_command setup failed (%s); falling back to direct run", e,
            )
            return await raw_run_command(command)
        # Any other error (poll timeout, result-read failure, cancellation)
        # propagates: the command has already run on the VM, so re-running it
        # would be wrong (and could re-run a multi-minute eval).

    interface.run_command = resilient_run_command
    interface._ale_resilient_commands = True


class TaskDriver:
    """Task setup + evaluate against a pre-opened DesktopSession.

    The session is owned by the environment layer (the Provider produced it).
    TaskDriver does NOT close the session — its ``close()`` method only resets
    its own reference; the env releases the VM (and hence the session) in
    its own ``close_async``.
    """

    def __init__(
        self,
        task_path: str,
        session: RemoteDesktopSession,
        variant: int = 0,
        skip_setup: bool = False,
        os_type: str | None = None,
        session_rebuilder: Callable[[], Awaitable[RemoteDesktopSession]] | None = None,
    ):
        self._variant = variant
        self._skip_setup = skip_setup

        self._task_loader = TaskLoader(task_path)
        self._task_info = self._task_loader.load(variant_index=variant)
        self._os_type = os_type or self._task_info.get("os_type", "windows")

        self._session: RemoteDesktopSession | None = session
        # Async callable returning a fresh DesktopSession bound to the
        # same VM (env.reset_session). None means "no reconnect available"
        # → transient errors fall through after the per-call retry loop.
        self._rebuilder = session_rebuilder
        self._install_resilient(session)

    def _install_resilient(self, session: RemoteDesktopSession) -> None:
        try:
            os_type = getattr(session, "_os_type", None) or "linux"
            install_resilient_cua_commands(session.computer.interface, os_type=os_type)
        except Exception as e:
            logger.debug("install_resilient_cua_commands skipped: %s", e)

    async def _reconnect(self) -> bool:
        """Force-close current session + ask rebuilder for a new one.

        Returns True on success, False when no rebuilder is configured
        (caller falls back to plain retry on the same session).
        """
        if self._rebuilder is None:
            return False
        # The env's reset_session() already force-closes the old session,
        # so we just drop the local reference + adopt the new one.
        self._session = None
        new_session = await self._rebuilder()
        self._session = new_session
        self._install_resilient(new_session)
        return True

    @property
    def task_description(self) -> str:
        return self._task_info["description"]

    @property
    def task_metadata(self) -> dict:
        return self._task_info["metadata"]

    @property
    def task_info(self) -> dict:
        return self._task_info

    @property
    def os_type(self) -> str:
        return self._os_type

    @property
    def session(self) -> RemoteDesktopSession:
        if self._session is None:
            raise RuntimeError("TaskDriver session was already closed.")
        return self._session

    async def setup(self) -> None:
        if self._skip_setup:
            logger.info("Skipping task setup (skip_setup=True)")
            return

        setup_fn = self._task_loader.get_setup_fn()
        if setup_fn is None:
            logger.info("No setup function found — skipping")
            return

        task_cfg = self._make_task_cfg()
        last_error: Exception | None = None
        for attempt in range(_CUA_SESSION_ATTEMPTS):
            try:
                logger.info("Running task setup (attempt %d/%d)...",
                            attempt + 1, _CUA_SESSION_ATTEMPTS)
                result = setup_fn(task_cfg, self._session)
                if asyncio.iscoroutine(result):
                    await result
                logger.info("Task setup completed")
                return
            except Exception as e:
                last_error = e
                if not _is_transient_cua_error(e):
                    raise
                logger.warning("Task setup failed with transient CUA error: %s", e)
                if attempt < _CUA_SESSION_ATTEMPTS - 1:
                    # Force-close and rebuild via env.reset_session if available
                    # (simprun parity: close(force=True) + connect()).
                    try:
                        await self._reconnect()
                    except Exception as reconnect_err:
                        logger.warning("Reconnect for task setup failed: %s", reconnect_err)
                    await _sleep_before_retry(attempt)

        raise RuntimeError(f"Task setup failed after CUA reconnect attempts: {last_error}")

    async def evaluate(self) -> Dict[str, Any]:
        evaluate_fn = self._task_loader.get_evaluate_fn()
        if evaluate_fn is None:
            logger.info("No evaluate function found — skipping")
            return {"score": None, "success": None}

        task_cfg = self._make_task_cfg()
        last_error: Exception | None = None
        for attempt in range(_CUA_SESSION_ATTEMPTS):
            try:
                result = evaluate_fn(task_cfg, self._session)
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception as e:
                last_error = e
                if not _is_transient_cua_error(e):
                    logger.error("Evaluate failed: %s", e)
                    return {"error": str(e), "score": None, "success": None}
                logger.warning("Evaluate failed (transient, attempt %d): %s", attempt + 1, e)
                if attempt < _CUA_SESSION_ATTEMPTS - 1:
                    try:
                        await self._reconnect()
                    except Exception as reconnect_err:
                        logger.warning("Reconnect for evaluate failed: %s", reconnect_err)
                    await _sleep_before_retry(attempt)
                continue
            if isinstance(result, list):
                return {"score": result[0] if result else None, "raw_scores": result}
            if isinstance(result, dict):
                return result
            return {"score": result}

        logger.error("Evaluate failed after retries: %s", last_error)
        return {"error": str(last_error), "score": None, "success": None}

    async def close(self) -> None:
        """Drop the session reference. The env owns session lifecycle."""
        self._session = None

    def _make_task_cfg(self):
        task_cfg = self._task_loader.build_task_cfg(variant_index=self._variant)
        if not hasattr(task_cfg, "metadata") or getattr(task_cfg, "metadata") is None:
            task_cfg.metadata = self._task_info.get("metadata", {})
        return task_cfg
