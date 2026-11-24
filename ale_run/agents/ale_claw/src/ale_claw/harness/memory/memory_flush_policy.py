"""Memory-flush policy: constants + the "should we flush now?" guards.

Split out of ``session.py`` (one of its four concerns). This is the POLICY
(when a pre-compaction flush should run); the execution lives in
``memory_flush.py``.

Based on OpenClaw's memory-flush.ts (openclaw/src/auto-reply/reply/memory-flush.ts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..session import SessionState

SILENT_REPLY_TOKEN = "[!silent]"

MEMORY_FLUSH_PROMPT = (
    "Pre-compaction memory flush. "
    "Store durable memories now (use memory_write with target='session' for observations, "
    "or target='task_memory' for strategies and insights worth keeping across sessions). "
    "IMPORTANT: Only store information that would be valuable for future runs — "
    "key decisions, progress milestones, discovered patterns, or important state. "
    f"If nothing to store, reply with {SILENT_REPLY_TOKEN}."
)

MEMORY_FLUSH_SYSTEM_PROMPT = (
    "Pre-compaction memory flush turn. "
    "The session is near auto-compaction; capture durable memories to disk. "
    f"You may reply, but usually {SILENT_REPLY_TOKEN} is correct."
)

DEFAULT_MEMORY_FLUSH_SOFT_THRESHOLD_TOKENS = 4000
DEFAULT_MEMORY_FLUSH_RESERVE_TOKENS_FLOOR = 20_000
# Mirrors OpenClaw's DEFAULT_MEMORY_FLUSH_FORCE_TRANSCRIPT_BYTES
# (extensions/memory-core/src/flush-plan.ts:11). Set to 0 to disable.
DEFAULT_MEMORY_FLUSH_FORCE_TRANSCRIPT_BYTES = 2 * 1024 * 1024  # 2 MB


DEFAULT_COMPACTION_RATIO = 0.80


def should_run_memory_flush(
    state: SessionState,
    *,
    current_tokens: int,
    context_window: int,
    transcript_bytes: int = 0,
    compaction_ratio: float = DEFAULT_COMPACTION_RATIO,
    soft_threshold_tokens: int = DEFAULT_MEMORY_FLUSH_SOFT_THRESHOLD_TOKENS,
    reserve_tokens: int = DEFAULT_MEMORY_FLUSH_RESERVE_TOKENS_FLOOR,
    force_transcript_bytes: int = DEFAULT_MEMORY_FLUSH_FORCE_TRANSCRIPT_BYTES,
) -> bool:
    """Determine whether a pre-compaction memory flush should run.

    Two independent triggers (matching OpenClaw):

    1. Token-count: flush when ``current_tokens`` reaches the threshold below
       the compaction trigger. agenthle's compaction is proactive at
       ``compaction_ratio * context_window`` (default 80%), so the threshold is
       anchored to that trigger — not the raw context window — with
       ``reserve + soft_threshold`` of headroom below it. For a 200K window
       @ 0.80 this is 136K (24K cushion); for 1M @ 0.80 it is 776K.
       Pass ``compaction_ratio=1.0`` to recover OpenClaw's window-edge
       semantics (correct only when compaction also fires at the literal limit).

    2. Transcript-size: flush when the on-disk transcript file reaches
       ``force_transcript_bytes`` (default 2 MB). Mirrors OpenClaw's
       ``forceFlushTranscriptBytes`` — useful when token estimation drifts
       but the transcript still grows. Pass 0 to disable.

    Either trigger fires independently. The "already flushed in this
    compaction cycle" dedup guard applies to both.

    Based on OpenClaw's shouldRunMemoryFlush()
    (openclaw/src/auto-reply/reply/memory-flush.ts:124-169) and
    buildMemoryFlushPlan() (extensions/memory-core/src/flush-plan.ts:95-140).
    """
    by_tokens = False
    if current_tokens > 0 and 0 < compaction_ratio <= 1:
        compaction_trigger = int(context_window * compaction_ratio)
        threshold = max(0, compaction_trigger - reserve_tokens - soft_threshold_tokens)
        by_tokens = threshold > 0 and current_tokens >= threshold

    by_transcript = (
        force_transcript_bytes > 0 and transcript_bytes >= force_transcript_bytes
    )

    if not (by_tokens or by_transcript):
        return False
    return not has_already_flushed_for_current_compaction(state)


def has_already_flushed_for_current_compaction(state: SessionState) -> bool:
    """Check whether a memory flush has already occurred in the current compaction cycle.

    Returns True when the flush's compaction count matches the session's current
    compaction count, preventing redundant flushes within the same cycle.

    Based on OpenClaw's hasAlreadyFlushedForCurrentCompaction()
    (openclaw/src/auto-reply/reply/memory-flush.ts:176-182).
    """
    if state.memory_flush_compaction_count is None:
        return False
    return state.memory_flush_compaction_count == state.compaction_count
