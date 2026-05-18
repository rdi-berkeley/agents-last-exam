"""Incremental file pull — VM-runtime only.

Async port of ``simprun/sync_helpers.py`` + ``simprun/remote.py:download_file_range``.
Keeps a per-file ``RangeState`` (offset, last_remote_size, error counters);
each tick issues one ``stat + tail | head | base64`` round-trip via
``session.run_command``, advances local offset only on confirmed fsync,
and handles file-rotation / missing-file / partial-record-at-end.

Used by :mod:`ale.runner.lifecycle` to keep host-side copies of hot agent
files (transcript.jsonl, stderr.log) fresh during long agent runs so
Ctrl-C / VM-revert / network-blip don't lose diagnostic data.

Scope: ONLY ``runtime: vm``. For ``local`` / ``docker`` the agent writes
directly to a host-visible work_dir (no pull needed).

Behaviour matches simprun:
  - boundary="newline" → cut delta at last ``\\n`` so JSONL never ships
    half a record. Residue is re-fetched next tick.
  - rotation (remote size < local offset) → unlink local + reset offset.
  - missing-file (remote -1) → unlink local + reset.
  - all retries / backoffs are HARDCODED (operator should not have to
    tune transcript-sync). 3 attempts per range call with 1s/3s/9s.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ale.core.cmd_result import cmd_ok, cmd_stderr, cmd_stdout

if TYPE_CHECKING:
    import cua_bench as cb

logger = logging.getLogger(__name__)


# =============================================================================
# Tunables (hardcoded — operator should not tune these)
# =============================================================================

DEFAULT_INTERVAL_S = 15.0          # how often to tick during agent run
DEFAULT_MAX_CHUNK_BYTES = 16 * 1024 * 1024
_RANGE_TIMEOUT_S = 60.0
_RANGE_RETRIES = 3
_RANGE_BACKOFF_S = (1.0, 3.0, 9.0)


# =============================================================================
# Boundary handling (simprun parity)
# =============================================================================

def apply_jsonl_boundary(delta: bytes) -> tuple[bytes, int]:
    """Split ``delta`` at the last ``\\n``. JSONL invariant.

    Returns ``(safe_to_commit, residue_bytes)``. Residue is dropped client-
    side; caller does NOT advance offset by ``residue_bytes`` so the same
    bytes get re-fetched next tick once the writer flushes a ``\\n``.
    """
    if not delta:
        return b"", 0
    last_nl = delta.rfind(b"\n")
    if last_nl == -1:
        return b"", len(delta)
    return delta[: last_nl + 1], len(delta) - (last_nl + 1)


def recover_offset_from_local(local_path: Path) -> int:
    """Resume position = right after the last ``\\n`` in ``local_path``.

    Used on fresh ``IncrementalPuller`` instances when a partial local file
    exists from a prior process (e.g. resumed batch). Returns 0 if file
    missing / empty / no ``\\n`` (re-pull from start).
    """
    if not local_path.exists():
        return 0
    try:
        data = local_path.read_bytes()
    except OSError:
        return 0
    if not data:
        return 0
    last_nl = data.rfind(b"\n")
    return 0 if last_nl == -1 else last_nl + 1


# =============================================================================
# State
# =============================================================================

@dataclass
class RangeState:
    """Per-remote-path state. ``offset`` advances ONLY after fsync."""

    offset: int = 0
    last_remote_size: int = 0
    consecutive_errors: int = 0
    rotation_count: int = 0

    def reset(self) -> None:
        self.offset = 0
        self.last_remote_size = 0
        self.consecutive_errors = 0


@dataclass
class RangeResult:
    success: bool
    remote_size: int = 0     # -1 = file missing, -2 = remote error
    delta: bytes = b""
    error: str | None = None


@dataclass
class StepOutcome:
    advanced: int = 0
    rotated: bool = False
    file_missing: bool = False
    transport_error: bool = False
    no_progress: bool = False
    error: str | None = None


# =============================================================================
# Range commands (Linux + Windows). Same wire format as simprun.
# =============================================================================

def _build_range_cmd_linux(remote: str, start: int, max_chunk_bytes: int) -> str:
    safe = remote.replace("'", "'\\''")
    return (
        f"S=$(stat -c%s '{safe}' 2>/dev/null || echo -1); "
        f"if [ \"$S\" -lt 0 ]; then printf 'SIZE=-1\\nB64=\\n'; exit 0; fi; "
        f"B=\"\"; "
        f"if [ \"$S\" -gt {start} ]; then "
        f"  B=$(tail -c +$(( {start} + 1 )) '{safe}' "
        f"     | head -c {max_chunk_bytes} | base64 -w0); "
        f"fi; "
        f"printf 'SIZE=%s\\nB64=%s\\n' \"$S\" \"$B\""
    )


def _build_range_cmd_windows(remote: str, start: int, max_chunk_bytes: int) -> str:
    safe = remote.replace("'", "''")
    ps = (
        "try{"
        f"$fi=[IO.FileInfo]::new('{safe}');"
        "if(-not $fi.Exists){\"SIZE=-1\";\"B64=\";exit};"
        "$len=$fi.Length;$b64='';"
        f"if($len -gt {start}){{"
        f"$fs=[IO.File]::Open('{safe}','Open','Read','ReadWrite');"
        "try{"
        f"$null=$fs.Seek({start},'Begin');"
        f"$rem=[Math]::Min($len-{start},{max_chunk_bytes});"
        "$buf=New-Object byte[] $rem;"
        "$null=$fs.Read($buf,0,$rem);"
        "$b64=[Convert]::ToBase64String($buf)"
        "}finally{$fs.Close()}"
        "};"
        "\"SIZE=$len\";\"B64=$b64\""
        "}catch{\"SIZE=-2\";\"B64=\";\"ERR=$($_.Exception.Message)\"}"
    )
    return f'powershell -NoProfile -Command "{ps}"'


def _parse_range_stdout(stdout: str, *, expected_start: int) -> RangeResult:
    """Parse the ``SIZE=<n>\\nB64=<...>`` envelope. simprun parity."""
    if not stdout:
        return RangeResult(success=False, error="empty stdout")

    size: int | None = None
    b64_text: str | None = None
    err_text: str | None = None
    for line in stdout.splitlines():
        if line.startswith("SIZE="):
            try:
                size = int(line[5:].strip())
            except ValueError:
                return RangeResult(success=False, error=f"bad SIZE: {line!r}")
        elif line.startswith("B64="):
            b64_text = line[4:]
        elif line.startswith("ERR="):
            err_text = line[4:]
    if size is None or b64_text is None:
        return RangeResult(success=False, error=f"missing SIZE/B64 (err={err_text!r})")

    if size < 0:
        if b64_text:
            return RangeResult(success=False, error=f"size={size} but B64 non-empty")
        return RangeResult(success=True, remote_size=size, delta=b"", error=err_text)

    expected_delta = max(0, size - expected_start)
    if expected_delta == 0:
        if b64_text:
            return RangeResult(success=False, error="expected empty delta but B64 set")
        return RangeResult(success=True, remote_size=size, delta=b"")

    try:
        delta = base64.b64decode(b64_text, validate=True)
    except Exception as exc:                            # noqa: BLE001
        return RangeResult(success=False, error=f"base64 decode: {exc}")

    if len(delta) > expected_delta:
        return RangeResult(
            success=False,
            error=f"delta {len(delta)} exceeds expected {expected_delta}",
        )
    return RangeResult(success=True, remote_size=size, delta=delta)


async def _pull_range(
    session: "cb.DesktopSession",
    remote: str,
    os_type: str,
    *,
    start: int,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
) -> RangeResult:
    """One range fetch — 3-retry with 1/3/9s backoff. simprun parity."""
    if os_type == "linux":
        cmd = _build_range_cmd_linux(remote, start, max_chunk_bytes)
    else:
        cmd = _build_range_cmd_windows(remote, start, max_chunk_bytes)
    last_err = ""
    for attempt in range(_RANGE_RETRIES):
        try:
            cr = await session.run_command(cmd, timeout=_RANGE_TIMEOUT_S)
        except Exception as exc:                        # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        else:
            if cmd_ok(cr):
                return _parse_range_stdout(cmd_stdout(cr) or "", expected_start=start)
            last_err = (cmd_stderr(cr) or cmd_stdout(cr) or "").strip()[:300]
        if attempt < _RANGE_RETRIES - 1:
            await asyncio.sleep(_RANGE_BACKOFF_S[attempt])
    return RangeResult(success=False, error=last_err)


# =============================================================================
# Step: take one range_result and apply to local file + state
# =============================================================================

def apply_range_step(
    state: RangeState,
    local_path: Path,
    range_result: RangeResult,
    *,
    boundary: Literal["newline", "none"] = "newline",
) -> StepOutcome:
    """Mirror of simprun.sync_helpers.apply_range_step (kept verbatim)."""
    if not range_result.success:
        state.consecutive_errors += 1
        return StepOutcome(transport_error=True, error=range_result.error)

    rs = range_result.remote_size

    if rs == -2:
        state.consecutive_errors += 1
        return StepOutcome(transport_error=True, error=range_result.error or "remote -2")

    if rs == -1:
        state.reset()
        if local_path.exists():
            try:
                local_path.unlink()
            except OSError:
                pass
        return StepOutcome(file_missing=True)

    if rs < state.offset:
        state.offset = 0
        state.last_remote_size = rs
        state.consecutive_errors = 0
        state.rotation_count += 1
        if local_path.exists():
            try:
                local_path.unlink()
            except OSError:
                pass
        return StepOutcome(rotated=True)

    state.last_remote_size = rs
    state.consecutive_errors = 0

    if not range_result.delta:
        return StepOutcome(no_progress=True)

    if boundary == "newline":
        safe, _residue = apply_jsonl_boundary(range_result.delta)
    else:
        safe = range_result.delta

    if not safe:
        return StepOutcome(no_progress=True)

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "ab") as f:
            f.write(safe)
            f.flush()
            os.fsync(f.fileno())
    except OSError as exc:
        state.consecutive_errors += 1
        return StepOutcome(transport_error=True, error=f"local write: {exc}")

    state.offset += len(safe)
    return StepOutcome(advanced=len(safe))


# =============================================================================
# Per-target spec + Puller class
# =============================================================================

@dataclass(frozen=True)
class PullTarget:
    """One hot file to keep in sync.

    ``remote_path`` is the absolute VM path. ``local_path`` is where it
    lands on host disk. ``boundary`` defaults to ``"newline"`` (JSONL);
    set to ``"none"`` for binary or non-line-oriented files.
    """

    remote_path: str
    local_path: Path
    boundary: Literal["newline", "none"] = "newline"


class IncrementalPuller:
    """Manages per-target ``RangeState`` and exposes ``tick()`` + ``reconcile_final()``."""

    def __init__(
        self,
        session_factory,            # zero-arg awaitable that returns a fresh session
        targets: list[PullTarget],
        os_type: str,
    ):
        self._session_factory = session_factory
        self._targets = targets
        self._os_type = os_type
        self._states: dict[str, RangeState] = {}
        # Recover offsets from any partial local files (resumed batch).
        for t in targets:
            st = RangeState()
            if t.local_path.exists():
                st.offset = recover_offset_from_local(t.local_path)
            self._states[t.remote_path] = st

    async def tick(self) -> dict[str, StepOutcome]:
        """One round of range-pulls. Returns per-target outcome dict.

        Acquires its own session so it doesn't race with other framework
        tasks (cua.DesktopSession is not task-safe).
        """
        session = await self._session_factory()
        outcomes: dict[str, StepOutcome] = {}
        for t in self._targets:
            st = self._states[t.remote_path]
            rr = await _pull_range(
                session, t.remote_path, self._os_type, start=st.offset,
            )
            outcomes[t.remote_path] = apply_range_step(st, t.local_path, rr, boundary=t.boundary)
        return outcomes

    async def reconcile_final(self) -> dict[str, StepOutcome]:
        """One final tick — does NOT loop. Caller can call this on cancel
        or after agent finishes to top up the tail bytes."""
        return await self.tick()


# =============================================================================
# Loop driver (the asyncio.task target)
# =============================================================================

async def incremental_pull_loop(
    puller: IncrementalPuller,
    *,
    interval_s: float = DEFAULT_INTERVAL_S,
    on_tick=None,
) -> None:
    """Keep ticking until cancelled. Sleeps ``interval_s`` between ticks.

    Designed to be started via ``asyncio.create_task`` and cancelled
    when the agent finishes. On cancel the caller should call
    ``puller.reconcile_final()`` separately to grab any last bytes.

    ``on_tick`` (optional) is called with the per-tick outcome dict for
    logging / event-emit hooks.
    """
    try:
        while True:
            try:
                outcomes = await puller.tick()
                if on_tick is not None:
                    try:
                        on_tick(outcomes)
                    except Exception:                   # noqa: BLE001
                        pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:                    # noqa: BLE001
                logger.warning("incremental_pull_loop tick failed: %s", exc)
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        logger.debug("incremental_pull_loop cancelled (final reconcile is caller's job)")
        raise
