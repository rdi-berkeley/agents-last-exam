"""MCPRuntime — MCP stdio bridges on a dedicated background event loop.

The OpenClaw runtime ale_claw embeds has no MCP client, and it executes each
tool call inside a worker thread running a *fresh* event loop: a tool's sync
``call()`` drives its async work via ``asyncio.run`` (see
``_tool_utils._run_async`` + ``agent_loop`` dispatching tools through
``asyncio.to_thread``). An ``mcp`` ``ClientSession`` and its stdio transport are
bound to the loop that created them, so a session created in the deployer's main
loop *cannot* be driven from those per-call worker loops — the call would hang.

To stay loop-agnostic, this runtime runs ALL sessions on one private event loop
in a background thread. ``call()`` hops onto that loop via
``run_coroutine_threadsafe`` and awaits the result through ``wrap_future``, so it
is safe to ``await`` from the main loop or from any worker-thread loop. The
harness consumes MCP purely as an I/O *backend*; it does not expose MCP tools to
the model. One instance per episode.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import AsyncExitStack
from typing import Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult

logger = logging.getLogger(__name__)


class MCPToolError(RuntimeError):
    """A bridge tool returned ``isError`` (the VM-side op failed)."""


def result_text(res: CallToolResult) -> str:
    """Concatenate the text content blocks of a tool result (e.g. read_bytes b64)."""
    return "".join(b.text for b in res.content if getattr(b, "type", None) == "text")


class MCPRuntime:
    """Owns one ``ClientSession`` per named MCP stdio server on a private loop."""

    def __init__(self, servers: dict[str, StdioServerParameters]) -> None:
        self._servers = dict(servers)
        self._sessions: dict[str, ClientSession] = {}
        # Per-server lock (lives on the background loop) — a single ClientSession
        # is not safe under concurrent call_tool; subagents may drive tools
        # alongside the main loop (cf. SESSION_API.md §14).
        self._locks: dict[str, asyncio.Lock] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._start_exc: Optional[BaseException] = None
        self._stop: Optional[asyncio.Event] = None

    # ------------------------------------------------------------------
    # Background loop thread
    # ------------------------------------------------------------------
    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._stop = asyncio.Event()
        try:
            loop.run_until_complete(self._serve())
        finally:
            loop.close()

    async def _serve(self) -> None:
        """Connect, hold the sessions open, then tear down — all in ONE task.

        stdio_client / ClientSession use anyio cancel scopes that must be entered
        and exited in the *same* task, so connect + the wait + teardown live here
        together. ``close()`` signals ``_stop`` from another thread to unwind.
        """
        try:
            async with AsyncExitStack() as stack:
                for name, params in self._servers.items():
                    read, write = await stack.enter_async_context(stdio_client(params))
                    sess = await stack.enter_async_context(ClientSession(read, write))
                    await sess.initialize()
                    self._sessions[name] = sess
                    self._locks[name] = asyncio.Lock()
                    logger.info("MCPRuntime: connected MCP server %r (%s)", name, params.command)
                self._ready.set()
                await self._stop.wait()  # hold contexts open until close()
            # stack unwinds here, same task → clean teardown (node children killed)
        except BaseException as exc:  # connection failed before ready
            if not self._ready.is_set():
                self._start_exc = exc
                self._ready.set()
            else:
                logger.exception("MCPRuntime: serve loop error")

    # ------------------------------------------------------------------
    # Lifecycle (sync core + async wrappers for the deployer's `async with`)
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="mcp-runtime", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._start_exc is not None:
            raise self._start_exc

    def close(self) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed() and self._stop is not None:
            loop.call_soon_threadsafe(self._stop.set)  # unwind _serve in its own task
        if self._thread is not None:
            self._thread.join(timeout=15)
        self._sessions.clear()
        self._locks.clear()

    async def __aenter__(self) -> "MCPRuntime":
        await asyncio.to_thread(self.start)  # connect off the main loop
        return self

    async def __aexit__(self, *exc) -> None:
        await asyncio.to_thread(self.close)

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    # ------------------------------------------------------------------
    # Calls — safe from any thread/loop
    # ------------------------------------------------------------------
    async def call(self, server: str, tool: str, arguments: dict) -> CallToolResult:
        """Call ``tool`` on ``server`` from any loop; raise on tool error."""
        if server not in self._sessions:
            raise RuntimeError(
                f"MCPRuntime: no connected server {server!r} "
                f"(have: {sorted(self._sessions)})"
            )
        if self._loop is None:
            raise RuntimeError("MCPRuntime: not started")
        # Hop onto the background loop where the session lives, then await the
        # concurrent.futures.Future from the *caller's* loop via wrap_future.
        fut = asyncio.run_coroutine_threadsafe(
            self._call(server, tool, arguments), self._loop
        )
        return await asyncio.wrap_future(fut)

    async def _call(self, server: str, tool: str, arguments: dict) -> CallToolResult:
        # Runs ON the background loop (session + lock belong to it).
        async with self._locks[server]:
            res = await self._sessions[server].call_tool(tool, arguments)
        if res.isError:
            raise MCPToolError(f"{server}.{tool} failed: {result_text(res) or '<no detail>'}")
        return res
