"""Context overflow detection, tool result truncation, and compaction pipeline.

Proactive detection: ContextOverflowCallback runs in CUA's on_llm_start callback chain,
estimating token usage and truncating oversized tool results before the API call.

Reactive detection: is_context_overflow_error() catches API rejections when proactive
detection underestimates.

Compaction pipeline: When overflow is detected, compact_messages() summarizes
older conversation history while preserving key identifiers, producing a CompactionResult
that the agent loop uses to restart with a compacted context.

Budget-aware compaction: compact_messages() calculates a token budget for
kept messages based on context_window * max_history_share, iteratively pruning the kept
portion until it fits. Recent turns are split out and preserved unconditionally.
Tool pairing repair ensures orphaned tool_use/tool_result pairs are fixed after splits.

Reference implementation:
  - openclaw/src/agents/compaction.ts — chunk splitting, summarization, identifier preservation
  - openclaw/src/agents/pi-embedded-runner/compact.ts — compaction orchestration
  - openclaw/src/agents/pi-embedded-runner/tool-result-truncation.ts — truncation logic
  - openclaw/src/agents/pi-embedded-helpers/errors.ts — error classification
  - openclaw/src/agents/session-transcript-repair.ts — tool pairing repair, duplicate detection
  - openclaw/src/agents/compaction-safeguard.ts — budget settings, history share
"""

from __future__ import annotations

import asyncio
import copy
import re
from typing import Any

from ..inference.model_config import ResolvedModel, resolve_model
from agent.callbacks.base import AsyncCallbackHandler

from .token_estimation import (  # re-exported for back-compat
    FIXED_IMAGE_TOKENS,
    SAFETY_MARGIN,
    estimate_message_tokens,
    estimate_messages_tokens,
)
from .compaction import (  # re-exported for back-compat
    BASE_CHUNK_RATIO,
    CompactionResult,
    DEFAULT_SUMMARY_FALLBACK,
    IDENTIFIER_PRESERVATION_INSTRUCTIONS,
    MAX_RECENT_TURNS_PRESERVE,
    MIN_CHUNK_RATIO,
    SUMMARIZATION_OVERHEAD_TOKENS,
    SUMMARIZATION_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    SUMMARIZATION_TIMEOUT,
    SYNTHETIC_TOOL_RESULT_CONTENT,
    ToolPairingRepairReport,
    UPDATE_SUMMARIZATION_PROMPT,
    chunk_messages_by_max_tokens,
    chunk_messages_by_token_share,
    compact_messages,
    compute_adaptive_chunk_ratio,
    repair_tool_use_result_pairing,
    serialize_messages_for_summary,
    split_preserved_recent_turns,
    summarize_chunk,
    summarize_chunks_iterative,
    summarize_with_fallback,
)

# ---------------------------------------------------------------------------
# Constants (from OpenClaw reference)
# ---------------------------------------------------------------------------

# SAFETY_MARGIN and FIXED_IMAGE_TOKENS moved to token_estimation.py (re-exported above).

DEFAULT_CONTEXT_TOKENS = 200_000
"""Fallback when the model's context window can't be resolved."""

MAX_TOOL_RESULT_SHARE = 0.25
"""Maximum share of context window a single tool result may occupy (PRD: 25%)."""

HARD_MAX_TOOL_RESULT_CHARS = 16_000
"""Absolute character cap for a single tool result.

Mirrors OpenClaw's DEFAULT_MAX_LIVE_TOOL_RESULT_CHARS (tool-result-truncation.ts:28).
Was 400_000 before US-OC-context-fix-1 — that allowed a single tool output to consume
~50% of a 200K window, which inflated context usage relative to OpenClaw and
triggered compaction on noisy turns.
"""

MIN_KEEP_CHARS = 2_000
"""Minimum characters to preserve when truncating."""

TRUNCATION_SUFFIX = (
    "\n\n\u26a0\ufe0f [Content truncated \u2014 original was too large for the model's "
    "context window. The content above is a partial view. If you need more, "
    "request specific sections or use offset/limit parameters to read smaller chunks.]"
)

MIDDLE_OMISSION_MARKER = (
    "\n\n\u26a0\ufe0f [... middle content omitted \u2014 showing head and tail ...]\n\n"
)

# --- Head+tail truncation tuning (truncate_tool_result_text) ---

TAIL_BUDGET_SHARE = 0.3
"""Fraction of the truncation budget reserved for the tail (the rest goes to the head)."""

MAX_TAIL_BUDGET_CHARS = 4_000
"""Absolute cap on tail-budget chars, regardless of TAIL_BUDGET_SHARE."""

NEWLINE_SNAP_THRESHOLD = 0.8
"""Snap a head/tail cut to a nearby newline only if it lands within this fraction of
the budget \u2014 avoids discarding most of the budget to reach a clean boundary."""

TAIL_NEWLINE_SNAP_SHARE = 0.2
"""Advance the tail start to a newline only if that newline is within this fraction of
the tail budget \u2014 keeps the tail close to its intended size."""


# ---------------------------------------------------------------------------
# Context window resolution
# ---------------------------------------------------------------------------

def resolve_context_window(model: str) -> int:
    """Resolve context window tokens for a model via litellm's model registry.

    Falls back to DEFAULT_CONTEXT_TOKENS for unknown models.
    """
    resolved = resolve_model(model)
    return resolved.context_window or DEFAULT_CONTEXT_TOKENS


# Token estimation (estimate_message_tokens / estimate_messages_tokens + the
# base64 regex and FIXED_IMAGE_TOKENS) moved to token_estimation.py; re-exported above.


# ---------------------------------------------------------------------------
# Tool result truncation
# ---------------------------------------------------------------------------

_IMPORTANT_TAIL_RE = re.compile(
    r"\b(error|exception|failed|fatal|traceback|panic|stack trace|errno|exit code)\b",
    re.IGNORECASE,
)
_SUMMARY_TAIL_RE = re.compile(
    r"\b(total|summary|result|complete|finished|done)\b",
    re.IGNORECASE,
)


def has_important_tail(text: str) -> bool:
    """Detect error/summary patterns in the last 2000 chars of text."""
    tail = text[-2000:]
    if _IMPORTANT_TAIL_RE.search(tail):
        return True
    # JSON closing structure
    if re.search(r"\}\s*$", tail.strip()):
        return True
    if _SUMMARY_TAIL_RE.search(tail):
        return True
    return False


def truncate_tool_result_text(text: str, max_chars: int) -> str:
    """Truncate a single text string to fit within max_chars.

    Uses head+tail strategy when the tail contains important content (errors,
    results, JSON), otherwise preserves the beginning.

    Adapted from openclaw/src/agents/pi-embedded-runner/tool-result-truncation.ts
    """
    if len(text) <= max_chars:
        return text

    budget = max(MIN_KEEP_CHARS, max_chars - len(TRUNCATION_SUFFIX))

    # Head+tail when tail looks important
    if has_important_tail(text) and budget > MIN_KEEP_CHARS * 2:
        tail_budget = min(int(budget * TAIL_BUDGET_SHARE), MAX_TAIL_BUDGET_CHARS)
        head_budget = budget - tail_budget - len(MIDDLE_OMISSION_MARKER)

        if head_budget > MIN_KEEP_CHARS:
            # Find clean cut points at newline boundaries
            head_cut = head_budget
            head_newline = text.rfind("\n", 0, head_budget)
            if head_newline > head_budget * NEWLINE_SNAP_THRESHOLD:
                head_cut = head_newline

            tail_start = len(text) - tail_budget
            tail_newline = text.find("\n", tail_start)
            if tail_newline != -1 and tail_newline < tail_start + int(tail_budget * TAIL_NEWLINE_SNAP_SHARE):
                tail_start = tail_newline + 1

            return text[:head_cut] + MIDDLE_OMISSION_MARKER + text[tail_start:] + TRUNCATION_SUFFIX

    # Default: keep the beginning
    cut_point = budget
    last_newline = text.rfind("\n", 0, budget)
    if last_newline > budget * NEWLINE_SNAP_THRESHOLD:
        cut_point = last_newline
    return text[:cut_point] + TRUNCATION_SUFFIX


def _calculate_max_tool_result_chars(context_window: int) -> int:
    """Max allowed chars for a single tool result given the context window."""
    max_tokens = int(context_window * MAX_TOOL_RESULT_SHARE)
    max_chars = max_tokens * 4  # chars/4 heuristic inverse
    return min(max_chars, HARD_MAX_TOOL_RESULT_CHARS)


def truncate_tool_results(
    msgs: list[dict[str, Any]], context_window: int
) -> list[dict[str, Any]]:
    """Truncate oversized function_call_output items in a message list (in-memory).

    CUA format: items with type=function_call_output have an "output" string field.
    Returns a new list — does not mutate the original.
    """
    max_chars = _calculate_max_tool_result_chars(context_window)
    result: list[dict[str, Any]] = []
    for msg in msgs:
        if msg.get("type") == "function_call_output":
            output = msg.get("output", "")
            if isinstance(output, str) and len(output) > max_chars:
                msg = copy.copy(msg)
                msg["output"] = truncate_tool_result_text(output, max_chars)
        result.append(msg)
    return result


# ---------------------------------------------------------------------------
# Reactive error detection
# ---------------------------------------------------------------------------

_CONTEXT_OVERFLOW_PATTERNS = [
    "request_too_large",
    "context length exceeded",
    "prompt is too long",
    "exceeds model context window",
    "request size exceeds",
    "maximum context length",
    "context overflow",
    "too many tokens",
    "content_too_large",
]

_RATE_LIMIT_EXCLUDE = re.compile(r"rate.?limit|tpm|tpd|rpm|rpd", re.IGNORECASE)


def is_context_overflow_error(error_message: str) -> bool:
    """Detect if an API error was caused by context window overflow.

    Adapted from OpenClaw's isLikelyContextOverflowError (errors.ts).
    Excludes rate limit false positives.
    """
    if not error_message:
        return False
    if _RATE_LIMIT_EXCLUDE.search(error_message):
        return False
    lower = error_message.lower()
    return any(p in lower for p in _CONTEXT_OVERFLOW_PATTERNS)


# ---------------------------------------------------------------------------
# ContextOverflowCallback
# ---------------------------------------------------------------------------

class ContextOverflowCallback(AsyncCallbackHandler):
    """Pre-LLM callback that estimates token usage and truncates oversized tool results.

    Wired into the CUA agent via ``callbacks=[overflow_cb]``. Runs before
    PromptInstructionsCallback and ImageRetentionCallback in the chain, so it sees
    messages before image stripping (conservative — overestimates, which is safer).

    After each ``on_llm_start``, check ``needs_compaction`` to decide whether to
    trigger the compaction pipeline.
    """

    def __init__(
        self,
        context_window: int | None = None,
        threshold: float = 0.80,
        model: str = "",
        instructions_tokens: int = 0,
        resolved_model: ResolvedModel | None = None,
        tag: str | None = None,
    ):
        if context_window is not None:
            self._context_window = context_window
        elif resolved_model is not None:
            self._context_window = resolved_model.context_window or DEFAULT_CONTEXT_TOKENS
        else:
            self._context_window = resolve_context_window(model)
        self._threshold = threshold
        self._instructions_tokens = instructions_tokens
        self._current_tokens = 0
        self._turn_count = 0
        self._needs_compaction = False
        self._tag = tag
        # API-reported actual prompt size from the most recent turn. Source-of-
        # truth for context pressure when set; the chars/4 estimator can drift
        # 10-20%. Mirrors OpenClaw's lastPromptTokens (pi-embedded-runner/run.ts).
        self._last_api_prompt_tokens = 0

    # -- Public read-only properties --

    @property
    def current_tokens(self) -> int:
        """Estimated token count after the last on_llm_start call."""
        return self._current_tokens

    @property
    def context_window(self) -> int:
        """Resolved context window size in tokens."""
        return self._context_window

    @property
    def compaction_threshold_ratio(self) -> float:
        """Fraction of ``context_window`` at which compaction is triggered.

        Exposed so the memory-flush threshold can be anchored to the same
        boundary (flush must fire before compaction).
        """
        return self._threshold

    @property
    def needs_compaction(self) -> bool:
        """Whether estimated usage exceeds the threshold."""
        return self._needs_compaction

    @property
    def overflow_ratio(self) -> float:
        """Current tokens as a fraction of the context window."""
        if self._context_window <= 0:
            return 0.0
        return self._current_tokens / self._context_window

    @property
    def turn_count(self) -> int:
        """Number of on_llm_start calls so far."""
        return self._turn_count

    @property
    def last_api_prompt_tokens(self) -> int:
        """Actual prompt size reported by the most recent API response.

        Zero when no API call has completed since the last reset_after_compaction
        (e.g., before the first turn or immediately after compaction).
        """
        return self._last_api_prompt_tokens

    # -- Mutation --

    def force_compaction(self) -> None:
        """Force needs_compaction=True (called by agent loop on reactive overflow detection)."""
        self._needs_compaction = True

    # -- Callback --

    async def on_llm_start(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Estimate tokens, truncate oversized tool results, set needs_compaction flag.

        When the previous turn's API-reported prompt size is known, use it as a
        floor on the estimate — the next turn's prompt is at least as big as
        the last one (we only add messages between turns). This catches cases
        where chars/4 estimation underreports relative to the real tokenizer.
        """
        self._turn_count += 1
        messages = truncate_tool_results(messages, self._context_window)
        raw = estimate_messages_tokens(messages)
        estimated = int(raw * SAFETY_MARGIN) + self._instructions_tokens
        self._current_tokens = max(estimated, self._last_api_prompt_tokens)
        self._needs_compaction = (
            self._current_tokens > self._context_window * self._threshold
        )
        prefix = f"[ContextOverflow:{self._tag}]" if self._tag else "[ContextOverflow]"
        source = "api+est" if self._last_api_prompt_tokens else "est"
        print(
            f"{prefix} turn {self._turn_count}: "
            f"~{self._current_tokens // 1000}K/{self._context_window // 1000}K tokens "
            f"({self.overflow_ratio:.0%}, source={source}), "
            f"needs_compaction={self._needs_compaction}"
        )
        return messages

    async def on_usage(self, usage: dict[str, Any]) -> None:
        """Capture API-reported prompt size to refine the next turn's trigger.

        After litellm's chat-completion-to-responses transform, ``input_tokens``
        is the total prompt size including any cached portions (Anthropic
        cache_read + cache_creation roll up into prompt_tokens upstream).
        Falls back to ``prompt_tokens`` for older/native shapes.

        If the actual prompt already exceeds the threshold, force compaction
        so the next iteration's check fires without waiting for an overflow
        error from the API.
        """
        actual = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        if not isinstance(actual, (int, float)) or actual <= 0:
            return
        self._last_api_prompt_tokens = int(actual)
        if actual > self._context_window * self._threshold:
            self._needs_compaction = True
            prefix = f"[ContextOverflow:{self._tag}]" if self._tag else "[ContextOverflow]"
            print(
                f"{prefix} API actual prompt="
                f"{int(actual) // 1000}K/{self._context_window // 1000}K "
                f"({actual / max(1, self._context_window):.0%}) — forcing compaction"
            )

    def reset_after_compaction(self) -> None:
        """Reset state after a compaction cycle so the next on_llm_start re-evaluates."""
        self._needs_compaction = False
        self._current_tokens = 0
        # Drop the stale lastPromptTokens — it reflects the pre-compaction
        # prompt and would mask the post-compaction shrink on the next turn.
        self._last_api_prompt_tokens = 0


# ===========================================================================
# Compaction Pipeline
#
# Adapted from openclaw/src/agents/compaction.ts.
# Key differences from OpenClaw:
#   - CUA stop-compact-resume pattern (can't inject mid-run)
#   - litellm.acompletion for summarization
#   - Transcript-based message extraction
# ===========================================================================
