"""Task environment for setup/evaluate lifecycle against an open session.

Ported from simprun/task_env.py. Decoupled from VM provisioning: the
constructor takes an already-open ``cua_bench.computers.remote.RemoteDesktopSession``
from the environment layer plus an optional ``session_rebuilder`` async
callback. The provider is responsible for ``wait_cua_ready`` and
initializing the underlying ``Computer`` object; TaskEnv owns the task
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


def install_resilient_cua_commands(interface: Any) -> None:
    """Wrap the interface's _send_command with extra retries for transient errors.

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
    interface._ale_resilient_commands = True


class TaskEnv:
    """Task setup + evaluate against a pre-opened DesktopSession.

    The session is owned by the environment layer (the Provider produced it).
    TaskEnv does NOT close the session — its ``close()`` method only resets
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
            install_resilient_cua_commands(session.computer.interface)
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
            raise RuntimeError("TaskEnv session was already closed.")
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
