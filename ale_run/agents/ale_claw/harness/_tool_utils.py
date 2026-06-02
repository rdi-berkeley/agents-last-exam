"""Shared validation/async helpers for the fs/shell/web tools.

These were duplicated (``_get_required_str``) or cross-imported from
``tools_fs`` (``_run_async``). Centralizing them here gives the tool modules a
single source of truth that doesn't tie them to one another.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging


def _run_async(coro):
    """Drive an async coroutine from a sync ``BaseTool.call``.

    Mirrors ``AnalyzeImageTool.call`` (analyze_image.py:149-170): spawn a
    fresh loop in a worker thread when one is already running, otherwise
    ``asyncio.run`` directly.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


def _get_required_str(params: dict, key: str, tool_name: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{tool_name}: required parameter "{key}" is missing or empty')
    return value


def _run_tool_execute(coro, logger: logging.Logger, error_context: str) -> dict:
    """Run a tool's async ``_execute()`` coroutine, surfacing any exception as a
    ``{"success": False, "error": ...}`` tool-error dict.

    ``error_context`` is the log-line prefix (e.g. ``"read tool failure on /x"``);
    ``logger`` is the calling module's logger so records keep their origin.
    """
    try:
        return _run_async(coro)
    except Exception as e:  # noqa: BLE001 — surface RPC errors as tool errors
        logger.error("%s: %s", error_context, e)
        return {"success": False, "error": f"Error: {e}"}
