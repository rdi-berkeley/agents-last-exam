"""Helpers for incremental log sync.

Ported verbatim from simprun/sync_helpers.py. Pure functions + small
dataclasses used by :mod:`incremental_puller`: byte-offset state machine,
jsonl record-boundary slicing, and the per-tick step driver. Lives in
orchastration so the boundary/state logic is independently testable
without any network or VM.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..environments.remote import RangeResult

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Boundary handling
# ----------------------------------------------------------------------


def apply_jsonl_boundary(delta: bytes) -> tuple[bytes, int]:
    """Split ``delta`` at the last newline.

    Returns ``(safe_to_commit, residue_bytes)`` where:

    - ``safe_to_commit`` is the prefix of ``delta`` ending at the last
      ``\\n`` (inclusive). May be empty if ``delta`` contains no ``\\n``.
    - ``residue_bytes`` is the count of bytes after the last ``\\n`` —
      these are dropped client-side; the caller must NOT advance its
      offset by this much, so the same bytes are re-fetched on the next
      pull.

    The contract assumes JSONL: each record ends in a literal ``\\n``,
    and any ``\\n`` inside a JSON string is escaped as ``\\\\n`` (two
    bytes 0x5C 0x6E), so a raw 0x0A is unambiguously a record boundary.
    """
    if not delta:
        return b"", 0
    last_nl = delta.rfind(b"\n")
    if last_nl == -1:
        return b"", len(delta)
    return delta[: last_nl + 1], len(delta) - (last_nl + 1)


def recover_offset_from_local(local_path: Path) -> int:
    """Return the byte position right after the last ``\\n`` in ``local_path``.

    Used when in-memory state is gone (process restart) but a partial
    local transcript exists from a previous tick. Returns 0 if the file
    is missing, empty, or contains no ``\\n`` (treat the whole file as a
    partial record and re-fetch from start).
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


# ----------------------------------------------------------------------
# Per-file range state
# ----------------------------------------------------------------------


@dataclass
class RangeState:
    """Per-remote-path state for incremental pulls.

    ``offset`` advances only after a successful fsync to the local file.
    Any transport / parse / write failure leaves it untouched so the next
    tick re-pulls from the same position.
    """

    offset: int = 0
    last_remote_size: int = 0
    consecutive_errors: int = 0
    rotation_count: int = 0

    def reset(self) -> None:
        self.offset = 0
        self.last_remote_size = 0
        self.consecutive_errors = 0


@dataclass
class RangeStates:
    """Container holding RangeState per remote path."""

    by_path: dict[str, RangeState] = field(default_factory=dict)

    def get(self, remote_path: str) -> RangeState:
        st = self.by_path.get(remote_path)
        if st is None:
            st = RangeState()
            self.by_path[remote_path] = st
        return st

    def clear(self) -> None:
        self.by_path.clear()


# ----------------------------------------------------------------------
# Per-tick step driver (pure-ish: only does file I/O + state mutation)
# ----------------------------------------------------------------------


@dataclass
class StepOutcome:
    advanced: int = 0          # bytes appended to local file (and to state.offset)
    rotated: bool = False      # remote file shrank since last tick (rotation/truncation)
    file_missing: bool = False  # remote file does not exist (size=-1)
    transport_error: bool = False
    no_progress: bool = False  # success but nothing to commit (e.g. delta < one record)
    error: str | None = None


def apply_range_step(
    state: RangeState,
    local_path: Path,
    range_result: "RangeResult",
    *,
    boundary: Literal["newline", "none"] = "newline",
) -> StepOutcome:
    """Apply one range-pull result to local file + state.

    Invariants (the whole reason this is centralized):

    - ``state.offset`` advances **only** after a successful fsync.
    - On any transport / parse / write failure, state is untouched so the
      next tick re-pulls from the same offset.
    - On rotation or missing-file, the local file is deleted and offset
      is reset to 0; the next tick will repopulate from scratch.
    - When the boundary rule has no newline to cut at, nothing is
      committed and offset is unchanged — the same bytes will be
      re-fetched next tick once the writer has flushed a ``\\n``.
    """
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
        # Rotation / truncation: server-side file shrank. Nuke local and reset.
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

    # Normal path: rs >= state.offset
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
    except OSError as e:
        # Local write failed; do NOT advance offset. Treat as transport
        # error for retry semantics.
        state.consecutive_errors += 1
        return StepOutcome(transport_error=True, error=f"local write: {e}")

    state.offset += len(safe)
    return StepOutcome(advanced=len(safe))
