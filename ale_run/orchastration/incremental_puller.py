"""Background tail of agent log files on the VM (vm-runtime only).

Mirrors simprun's per-deployer ``_sync_incremental`` loop, but at the
framework layer (since agenthle-public's deployers run inside the VM via
``cua.python_exec`` and don't have host-side polling of their own).

Per LOG_SPEC §7:

  - ticks every 15 s
  - for each ``hot_artifacts`` file, calls a single
    ``stat | tail | head | base64`` round-trip via ``download_file_range``
  - applies the boundary-safe slice (``apply_jsonl_boundary``) and appends
    deltas to the host-side mirror
  - on stop, runs **one** ``reconcile_final()`` pass with up to 3
    size-equality retries to catch the trailing flush, bounded at 60 s

Failures within one tick are isolated per-file: a transport error on
``transcript.jsonl`` doesn't stop the loop or block ``stderr.log``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..environments.remote import RemoteVMConfig, download_file_range
from .sync_helpers import RangeStates, apply_range_step

logger = logging.getLogger(__name__)


_TICK_INTERVAL_S = 15.0
_RECONCILE_TIMEOUT_S = 60.0
_RECONCILE_SETTLE_RETRIES = 3
_RECONCILE_SETTLE_DELAY_S = 1.0


@dataclass(frozen=True)
class PullTarget:
    """One (vm_path, host_path) pair the puller mirrors."""

    vm_path: str
    host_path: Path


class IncrementalPuller:
    """Background poller: tail VM logs into host files, byte-offset state.

    Construction binds a fixed list of targets; ``start()`` schedules the
    loop on the current event loop, ``stop()`` cancels + reconciles.
    """

    def __init__(
        self,
        *,
        vm_config: RemoteVMConfig,
        targets: list[PullTarget],
        interval_s: float = _TICK_INTERVAL_S,
    ):
        self._vm_config = vm_config
        self._targets = list(targets)
        self._interval_s = interval_s
        self._states = RangeStates()
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    @property
    def targets(self) -> list[PullTarget]:
        return list(self._targets)

    @property
    def interval_s(self) -> float:
        return self._interval_s

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        """Schedule the loop. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop(), name="IncrementalPuller")

    async def stop(self) -> str | None:
        """Cancel the loop and run a single final reconcile.

        Returns ``None`` on success, or the reconcile error message
        (still logged + emitted by the caller as ``incremental_pull_final_failed``).
        """
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self._interval_s + 5)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, BaseException):
                    pass
            self._task = None
        return await self._reconcile_final()

    # ------------------------------------------------------------------ loop

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._interval_s,
                )
                # stop signalled; let stop() drive the reconcile.
                return
            except asyncio.TimeoutError:
                pass
            await self._tick_all()

    async def _tick_all(self) -> None:
        # Per-target isolation: errors on one file don't break siblings.
        for tgt in self._targets:
            try:
                await self._tick_one(tgt)
            except Exception as e:
                logger.debug("IncrementalPuller tick failed for %s: %s", tgt.vm_path, e)

    async def _tick_one(self, tgt: PullTarget) -> None:
        state = self._states.get(tgt.vm_path)
        range_result = await asyncio.to_thread(
            download_file_range,
            self._vm_config,
            tgt.vm_path,
            start=state.offset,
        )
        outcome = apply_range_step(state, tgt.host_path, range_result)
        if outcome.advanced:
            logger.debug(
                "IncrementalPuller: +%d bytes for %s (offset=%d)",
                outcome.advanced, tgt.vm_path, state.offset,
            )

    # ------------------------------------------------------------------ reconcile

    async def _reconcile_final(self) -> str | None:
        """One final pass after agent stop, bounded at ``_RECONCILE_TIMEOUT_S``.

        Settle loop: pull up to ``_RECONCILE_SETTLE_RETRIES`` more times if
        the remote size is still growing, so we don't return mid-flush.
        Returns the first non-recoverable error, or ``None`` on success.
        """
        deadline = time.monotonic() + _RECONCILE_TIMEOUT_S
        last_err: str | None = None
        for tgt in self._targets:
            state = self._states.get(tgt.vm_path)
            prev_size = -1
            for _ in range(_RECONCILE_SETTLE_RETRIES + 1):
                if time.monotonic() > deadline:
                    last_err = f"reconcile timeout after {_RECONCILE_TIMEOUT_S}s"
                    break
                try:
                    range_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            download_file_range,
                            self._vm_config,
                            tgt.vm_path,
                            start=state.offset,
                        ),
                        timeout=max(1.0, deadline - time.monotonic()),
                    )
                except asyncio.TimeoutError:
                    last_err = "reconcile per-call timeout"
                    break
                outcome = apply_range_step(state, tgt.host_path, range_result)
                if outcome.transport_error:
                    last_err = outcome.error or "transport error"
                    await asyncio.sleep(_RECONCILE_SETTLE_DELAY_S)
                    continue
                if outcome.file_missing:
                    break
                if state.last_remote_size == prev_size:
                    # Size stable across two ticks → file flushed; stop.
                    break
                prev_size = state.last_remote_size
                await asyncio.sleep(_RECONCILE_SETTLE_DELAY_S)
        return last_err
