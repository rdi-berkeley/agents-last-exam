"""Context overflow detection, tool result truncation, and compaction pipeline.

Proactive detection: ContextOverflowCallback runs in CUA's on_llm_start callback chain,
estimating token usage and truncating oversized tool results before the API call.

Reactive detection: is_context_overflow_error() catches API rejections when proactive
detection underestimates.

Compaction pipeline (US-OC-006): When overflow is detected, compact_messages() summarizes
older conversation history while preserving key identifiers, producing a CompactionResult
that the agent loop uses to restart with a compacted context.

Budget-aware compaction (US-OC-013): compact_messages() calculates a token budget for
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
import json
import re
from dataclasses import dataclass
from typing import Any

from .model_config import ResolvedModel, resolve_model
from agent.callbacks.base import AsyncCallbackHandler

from .helper_runtime import call_helper_model

# ---------------------------------------------------------------------------
# Constants (from OpenClaw reference)
# ---------------------------------------------------------------------------

SAFETY_MARGIN = 1.2
"""Multiply raw token estimate by this factor to absorb tokenizer variance."""

DEFAULT_CONTEXT_TOKENS = 200_000
"""Fallback when the model's context window can't be resolved."""

FIXED_IMAGE_TOKENS = 1200
"""Standard API cost for a 1024x768 screenshot (Anthropic billing)."""

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


# ---------------------------------------------------------------------------
# Context window resolution
# ---------------------------------------------------------------------------

def resolve_context_window(model: str) -> int:
    """Resolve context window tokens for a model via litellm's model registry.

    Falls back to DEFAULT_CONTEXT_TOKENS for unknown models.
    """
    resolved = resolve_model(model)
    return resolved.context_window or DEFAULT_CONTEXT_TOKENS


def _model_candidates(model: str) -> list[str]:
    """Yield model name variants to try (full name, then without provider prefix)."""
    candidates = [model]
    if "/" in model:
        candidates.append(model.split("/", 1)[1])
    return candidates


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# Matches base64 image data URLs in computer_call_output content
_BASE64_IMAGE_RE = re.compile(r'"data:image/[^;]+;base64,[A-Za-z0-9+/=]+"')


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimate token count for a single message using chars/4 heuristic.

    Special handling for images: subtracts the base64 string length and adds
    FIXED_IMAGE_TOKENS per image (matching actual API billing). Thinking
    blocks are counted naturally because they remain part of the serialized
    message payload.
    """
    raw = json.dumps(msg, separators=(",", ":"))
    # Count and subtract base64 image data, replace with fixed token cost
    image_count = 0
    base64_chars = 0
    for match in _BASE64_IMAGE_RE.finditer(raw):
        image_count += 1
        base64_chars += len(match.group())
    char_tokens = (len(raw) - base64_chars) // 4
    return char_tokens + (image_count * FIXED_IMAGE_TOKENS)


def estimate_messages_tokens(msgs: list[dict[str, Any]]) -> int:
    """Estimate total token count for a list of messages."""
    return sum(estimate_message_tokens(m) for m in msgs)


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
        tail_budget = min(int(budget * 0.3), 4_000)
        head_budget = budget - tail_budget - len(MIDDLE_OMISSION_MARKER)

        if head_budget > MIN_KEEP_CHARS:
            # Find clean cut points at newline boundaries
            head_cut = head_budget
            head_newline = text.rfind("\n", 0, head_budget)
            if head_newline > head_budget * 0.8:
                head_cut = head_newline

            tail_start = len(text) - tail_budget
            tail_newline = text.find("\n", tail_start)
            if tail_newline != -1 and tail_newline < tail_start + int(tail_budget * 0.2):
                tail_start = tail_newline + 1

            return text[:head_cut] + MIDDLE_OMISSION_MARKER + text[tail_start:] + TRUNCATION_SUFFIX

    # Default: keep the beginning
    cut_point = budget
    last_newline = text.rfind("\n", 0, budget)
    if last_newline > budget * 0.8:
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
    trigger the compaction pipeline (US-OC-006).
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
# Compaction Pipeline (US-OC-006)
#
# Adapted from openclaw/src/agents/compaction.ts.
# Key differences from OpenClaw:
#   - CUA stop-compact-resume pattern (can't inject mid-run)
#   - litellm.acompletion for summarization
#   - Transcript-based message extraction
# ===========================================================================

# ---------------------------------------------------------------------------
# Compaction constants
# ---------------------------------------------------------------------------

BASE_CHUNK_RATIO = 0.4
"""Default chunk size as a share of context window."""

MIN_CHUNK_RATIO = 0.15
"""Minimum chunk ratio when large messages reduce it."""

SUMMARIZATION_OVERHEAD_TOKENS = 4096
"""Reserved tokens for summarization prompt, system message, serialization."""

DEFAULT_SUMMARY_FALLBACK = "No prior history."
"""Fallback when summarization fails completely."""

MAX_SUMMARIZATION_RETRIES = 3
"""Number of retry attempts for LLM summarization calls."""

SUMMARIZATION_TIMEOUT = 120
"""Timeout in seconds for each litellm.acompletion summarization call."""

# ---------------------------------------------------------------------------
# Compaction prompts (from OpenClaw compaction.ts)
# ---------------------------------------------------------------------------

IDENTIFIER_PRESERVATION_INSTRUCTIONS = (
    "Preserve all opaque identifiers exactly as written (no shortening or "
    "reconstruction), including UUIDs, hashes, IDs, tokens, API keys, "
    "hostnames, IPs, ports, URLs, and file names."
)

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI coding assistant, then produce a structured summary "
    "following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary.\n\n"
    + IDENTIFIER_PRESERVATION_INSTRUCTIONS
)

SUMMARIZATION_PROMPT = """\
The messages above are a conversation to summarize. Create a structured context \
checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, identifiers, \
and error messages.\
"""

UPDATE_SUMMARIZATION_PROMPT = """\
The messages above are NEW conversation messages to incorporate into the existing \
summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, identifiers, and error messages
- If something is no longer relevant, you may remove it

Use the same structured format as the original summary.\
"""


# ---------------------------------------------------------------------------
# CompactionResult
# ---------------------------------------------------------------------------

@dataclass
class CompactionResult:
    """Result of a compaction operation."""

    summary: str
    """The compaction summary text."""

    tokens_before: int
    """Estimated tokens before compaction."""

    tokens_after: int
    """Estimated tokens after compaction (summary + kept messages)."""

    first_kept_message_index: int
    """Index in the original message list where kept messages start."""

    chunks_processed: int
    """Number of chunks that were summarized."""


# ---------------------------------------------------------------------------
# Tool pairing repair (US-OC-013)
#
# Adapted from openclaw/src/agents/session-transcript-repair.ts
# (repairToolUseResultPairing). Fixes orphaned tool_use/tool_result pairs
# that arise when messages are split at arbitrary boundaries during compaction.
# ---------------------------------------------------------------------------

SYNTHETIC_TOOL_RESULT_CONTENT = (
    "[compaction] missing tool result — synthetic error result for transcript repair."
)

# Stop reasons that indicate partial/malformed tool calls — skip synthesis
_SKIP_SYNTHESIS_STOP_REASONS = frozenset({"error", "aborted"})


@dataclass
class ToolPairingRepairReport:
    """Report from tool pairing repair."""

    messages: list[dict[str, Any]]
    """Repaired message list."""

    dropped_orphan_count: int
    """Number of orphaned tool_results dropped (result with no matching call)."""

    dropped_duplicate_count: int
    """Number of duplicate tool_results dropped (same tool_use_id seen before)."""

    added_synthetic_count: int
    """Number of synthetic error tool_results inserted (call with no matching result)."""


def repair_tool_use_result_pairing(
    messages: list[dict[str, Any]],
) -> ToolPairingRepairReport:
    """Repair orphaned tool_use/tool_result pairs in a message list.

    Algorithm (matching OpenClaw's session-transcript-repair.ts):
    1. Collect call IDs from assistant messages (function_call/computer_call blocks)
    2. Match tool result messages to calls by ID
    3. Drop orphaned results (no matching call) and duplicates (same ID seen before)
    4. Insert synthetic error results for calls with no matching result
       (unless stop_reason is "error" or "aborted")

    This operates on role-based completion-format messages (after compaction).
    The Responses API counterpart for flat items lives in
    ``agent.loops.openai._repair_orphaned_calls``.

    Args:
        messages: List of message dicts with role, content, and optional stop_reason.

    Returns:
        ToolPairingRepairReport with repaired messages and counts.
    """
    if not messages:
        return ToolPairingRepairReport(
            messages=[], dropped_orphan_count=0,
            dropped_duplicate_count=0, added_synthetic_count=0,
        )

    dropped_orphan_count = 0
    dropped_duplicate_count = 0
    added_synthetic_count = 0
    seen_result_ids: set[str] = set()
    result_messages: list[dict[str, Any]] = []

    # First pass: collect all call IDs from assistant messages
    # and build a set of available result IDs from tool messages
    pending_call_ids: list[tuple[str, str | None]] = []  # (call_id, stop_reason)
    available_result_ids: dict[str, int] = {}  # result_id -> message index

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant" and isinstance(content, list):
            stop_reason = msg.get("stop_reason")
            for block in content:
                btype = block.get("type", "")
                if btype in ("function_call", "computer_call"):
                    call_id = block.get("id", "")
                    if call_id:
                        pending_call_ids.append((call_id, stop_reason))

        elif role == "tool" and isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    result_id = block.get("tool_use_id", "")
                    if result_id:
                        available_result_ids[result_id] = i

    # Build set of valid call IDs for orphan detection
    valid_call_ids = {cid for cid, _ in pending_call_ids}
    # Build set of call IDs that have matching results
    matched_call_ids = set()

    # Second pass: filter messages, dropping orphans and duplicates
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "tool" and isinstance(content, list):
            filtered_blocks: list[dict[str, Any]] = []
            for block in content:
                if block.get("type") == "tool_result":
                    result_id = block.get("tool_use_id", "")
                    # Drop orphans (result with no matching call)
                    if result_id and result_id not in valid_call_ids:
                        dropped_orphan_count += 1
                        continue
                    # Drop duplicates (same ID seen before)
                    if result_id in seen_result_ids:
                        dropped_duplicate_count += 1
                        continue
                    if result_id:
                        seen_result_ids.add(result_id)
                        matched_call_ids.add(result_id)
                filtered_blocks.append(block)

            if filtered_blocks:
                new_msg = dict(msg)
                new_msg["content"] = filtered_blocks
                result_messages.append(new_msg)
            # else: all blocks were dropped, skip the message entirely
        else:
            result_messages.append(msg)

    # Third pass: insert synthetic results for unmatched calls
    # Walk result_messages, after each assistant message check for unmatched calls
    final_messages: list[dict[str, Any]] = []
    for msg in result_messages:
        final_messages.append(msg)
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant" and isinstance(content, list):
            stop_reason = msg.get("stop_reason")
            unmatched_blocks: list[dict[str, Any]] = []
            for block in content:
                btype = block.get("type", "")
                if btype in ("function_call", "computer_call"):
                    call_id = block.get("id", "")
                    if call_id and call_id not in matched_call_ids:
                        # Skip synthesis for error/aborted stop reasons
                        if stop_reason in _SKIP_SYNTHESIS_STOP_REASONS:
                            continue
                        unmatched_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": SYNTHETIC_TOOL_RESULT_CONTENT,
                            "is_error": True,
                        })
                        added_synthetic_count += 1
                        matched_call_ids.add(call_id)

            if unmatched_blocks:
                final_messages.append({
                    "role": "tool",
                    "content": unmatched_blocks,
                })

    return ToolPairingRepairReport(
        messages=final_messages,
        dropped_orphan_count=dropped_orphan_count,
        dropped_duplicate_count=dropped_duplicate_count,
        added_synthetic_count=added_synthetic_count,
    )


# ---------------------------------------------------------------------------
# Recent turns preservation (US-OC-013)
#
# Adapted from openclaw/src/agents/compaction-safeguard.ts
# (splitPreservedRecentTurns). Splits out the last N turns so they
# are never pruned or summarized, guaranteeing the agent's in-flight
# working state survives compaction.
#
# Turn counting uses assistant messages (not user messages) so this works
# for both chat-style agents (alternating user/assistant) and CUA agents
# (1 user message + N assistant/tool pairs).
# ---------------------------------------------------------------------------

# Hard cap: never preserve more than this many turns regardless of request.
MAX_RECENT_TURNS_PRESERVE = 12


def split_preserved_recent_turns(
    messages: list[dict[str, Any]],
    preserve_count: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split out the last N turns (+ their tool responses).

    A "turn" is counted per assistant message, which works for both:
    - Chat-style: user/assistant pairs (N assistants ≈ N turns)
    - CUA-style: 1 user + N assistant/tool pairs (N assistants = N turns)

    Returns (pruneable_messages, preserved_messages).
    Preserved messages are never summarized or pruned.
    Tool pairing is repaired on the pruneable portion (split could orphan pairs).

    Args:
        messages: Full message list.
        preserve_count: Number of turns (assistant messages) to preserve from the end.

    Returns:
        Tuple of (pruneable, preserved) message lists.
    """
    preserve_count = min(preserve_count, MAX_RECENT_TURNS_PRESERVE)
    if preserve_count <= 0 or not messages:
        return list(messages), []

    # Walk backward, counting assistant messages as turns
    assistant_count = 0
    split_index = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            assistant_count += 1
            if assistant_count >= preserve_count:
                split_index = i
                break

    if assistant_count < preserve_count:
        # Fewer turns than preserve_count → all preserved
        return [], list(messages)

    pruneable = messages[:split_index]
    preserved = messages[split_index:]

    # Repair tool pairing on pruneable (split could orphan pairs at boundary)
    if pruneable:
        repair = repair_tool_use_result_pairing(pruneable)
        pruneable = repair.messages

    return pruneable, preserved


# ---------------------------------------------------------------------------
# Chunk splitting
# ---------------------------------------------------------------------------

def chunk_messages_by_token_share(
    messages: list[dict[str, Any]], parts: int = 2
) -> list[list[dict[str, Any]]]:
    """Split messages into ``parts`` chunks targeting equal token budgets.

    Adapted from OpenClaw's splitMessagesByTokenShare. Preserves message order.
    """
    if not messages or parts < 1:
        return []
    if parts == 1:
        return [list(messages)]

    total = estimate_messages_tokens(messages)
    if total == 0:
        return [list(messages)]

    target_per_part = total / parts
    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []
    current_tokens = 0

    for msg in messages:
        msg_tokens = estimate_message_tokens(msg)
        # Start new chunk if adding this message exceeds target and we have room
        if (
            current_chunk
            and current_tokens + msg_tokens > target_per_part
            and len(chunks) < parts - 1
        ):
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(msg)
        current_tokens += msg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def chunk_messages_by_max_tokens(
    messages: list[dict[str, Any]], max_tokens: int
) -> list[list[dict[str, Any]]]:
    """Split messages so each chunk stays under ``max_tokens``.

    Adapted from OpenClaw's chunkMessagesByMaxTokens.
    Oversized single messages get their own chunk.
    """
    if not messages or max_tokens <= 0:
        return []

    safe_max = int(max_tokens / SAFETY_MARGIN)
    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []
    current_tokens = 0

    for msg in messages:
        msg_tokens = estimate_message_tokens(msg)
        if current_chunk and current_tokens + msg_tokens > safe_max:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0
        current_chunk.append(msg)
        current_tokens += msg_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def compute_adaptive_chunk_ratio(
    messages: list[dict[str, Any]], context_window: int
) -> float:
    """Dynamically adjust chunk ratio based on average message size.

    Adapted from OpenClaw's computeAdaptiveChunkRatio.
    Reduces the ratio when messages are large relative to the context window.
    """
    if not messages or context_window <= 0:
        return BASE_CHUNK_RATIO

    total_tokens = estimate_messages_tokens(messages)
    avg_tokens = total_tokens / len(messages)
    safe_avg = avg_tokens * SAFETY_MARGIN
    ratio = safe_avg / context_window

    if ratio > 0.1:
        reduction = min(ratio * 2, BASE_CHUNK_RATIO - MIN_CHUNK_RATIO)
        return max(MIN_CHUNK_RATIO, BASE_CHUNK_RATIO - reduction)

    return BASE_CHUNK_RATIO


# ---------------------------------------------------------------------------
# Message serialization for summarization
# ---------------------------------------------------------------------------

def serialize_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    """Convert message dicts to readable text for the summarization prompt.

    Strips base64 image data and preserves role, text content, tool names,
    and action descriptions.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", msg.get("type", "unknown"))
        content = msg.get("content", "")

        if isinstance(content, str):
            lines.append(f"[{role}] {content}")
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                btype = block.get("type", "")
                if btype == "text" and block.get("text"):
                    parts.append(block["text"])
                elif btype == "function_call":
                    name = block.get("name", "unknown")
                    args = block.get("arguments", "")
                    if isinstance(args, str) and len(args) > 200:
                        args = args[:200] + "..."
                    parts.append(f"[tool_call: {name}({args})]")
                elif btype == "computer_call":
                    # Handle both "action" (computer-use-preview) and "actions" (GPT 5.4)
                    action = block.get("action") or block.get("actions", {})
                    parts.append(f"[computer: {json.dumps(action)[:200]}]")
                elif btype == "tool_result":
                    result_text = str(block.get("content", ""))[:500]
                    parts.append(f"[tool_result: {result_text}]")
                # Skip image/base64 content entirely
            if parts:
                lines.append(f"[{role}] {' | '.join(parts)}")
        else:
            text = str(content)[:500]
            lines.append(f"[{role}] {text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-based summarization
# ---------------------------------------------------------------------------

async def summarize_chunk(
    messages: list[dict[str, Any]],
    model: str,
    *,
    previous_summary: str | None = None,
    custom_instructions: str | None = None,
    timeout: int = SUMMARIZATION_TIMEOUT,
    thinking_params: dict[str, Any] | None = None,
    summary_runtime: ResolvedModel | None = None,
) -> str:
    """Summarize a chunk of messages via the shared helper runtime adapter.

    Args:
        messages: The message chunk to summarize.
        model: litellm model string (e.g. "anthropic/claude-sonnet-4-20250514").
        previous_summary: Summary of earlier chunks for continuity.
        custom_instructions: Optional additional instructions.

    Returns:
        Summary text.
    """
    system_parts = [SUMMARIZATION_SYSTEM_PROMPT]
    if custom_instructions:
        system_parts.append(custom_instructions)

    conversation_text = serialize_messages_for_summary(messages)

    user_parts: list[str] = []
    if previous_summary:
        user_parts.append(f"<previous-summary>\n{previous_summary}\n</previous-summary>\n")
    user_parts.append(f"<conversation>\n{conversation_text}\n</conversation>\n")
    if previous_summary:
        user_parts.append(UPDATE_SUMMARIZATION_PROMPT)
    else:
        user_parts.append(SUMMARIZATION_PROMPT)

    llm_messages = [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    last_error: Exception | None = None
    resolved_summary = summary_runtime or resolve_model(model)
    for attempt in range(MAX_SUMMARIZATION_RETRIES):
        try:
            response = await call_helper_model(
                resolved_summary,
                purpose="compaction",
                messages=llm_messages,
                max_tokens=2048,
                temperature=1.0,
                timeout=timeout,
                thinking_params=thinking_params,
            )
            return response.text or DEFAULT_SUMMARY_FALLBACK
        except Exception as e:
            last_error = e
            if attempt < MAX_SUMMARIZATION_RETRIES - 1:
                backoff = min(0.5 * (2 ** attempt), 5.0)
                print(f"[Compaction] Summarization attempt {attempt + 1} failed: {e}, retrying in {backoff:.1f}s")
                await asyncio.sleep(backoff)

    print(f"[Compaction] All summarization attempts failed: {last_error}")
    raise last_error  # type: ignore[misc]


async def summarize_chunks_iterative(
    chunks: list[list[dict[str, Any]]],
    model: str,
    *,
    custom_instructions: str | None = None,
    timeout: int = SUMMARIZATION_TIMEOUT,
    thinking_params: dict[str, Any] | None = None,
    summary_runtime: ResolvedModel | None = None,
) -> str:
    """Iteratively summarize chunks, feeding each summary as context to the next.

    Adapted from OpenClaw's summarizeChunks pattern.
    """
    if not chunks:
        return DEFAULT_SUMMARY_FALLBACK

    summary: str | None = None
    for i, chunk in enumerate(chunks):
        print(f"[Compaction] Summarizing chunk {i + 1}/{len(chunks)} ({len(chunk)} messages)")
        summary = await summarize_chunk(
            chunk,
            model,
            previous_summary=summary,
            custom_instructions=custom_instructions,
            timeout=timeout,
            thinking_params=thinking_params,
            summary_runtime=summary_runtime,
        )

    return summary or DEFAULT_SUMMARY_FALLBACK


def _is_oversized_for_summary(msg: dict[str, Any], context_window: int) -> bool:
    """Check if a single message exceeds 50% of the context window."""
    return estimate_message_tokens(msg) > context_window * 0.5


async def summarize_with_fallback(
    messages: list[dict[str, Any]],
    model: str,
    context_window: int,
    max_chunk_tokens: int,
    *,
    custom_instructions: str | None = None,
    timeout: int = SUMMARIZATION_TIMEOUT,
    thinking_params: dict[str, Any] | None = None,
    summary_runtime: ResolvedModel | None = None,
) -> str:
    """Three-tier summarization with progressive fallback.

    1. Full summarization of all messages
    2. Exclude oversized messages, summarize the rest
    3. Static fallback noting message count
    """
    # Tier 1: full summarization
    try:
        chunks = chunk_messages_by_max_tokens(messages, max_chunk_tokens)
        if chunks:
            return await summarize_chunks_iterative(
                chunks, model, custom_instructions=custom_instructions,
                timeout=timeout,
                thinking_params=thinking_params,
                summary_runtime=summary_runtime,
            )
    except Exception as e:
        print(f"[Compaction] Tier 1 (full) failed: {e}")

    # Tier 2: exclude oversized messages
    try:
        filtered = [m for m in messages if not _is_oversized_for_summary(m, context_window)]
        oversized_count = len(messages) - len(filtered)
        if filtered:
            chunks = chunk_messages_by_max_tokens(filtered, max_chunk_tokens)
            if chunks:
                summary = await summarize_chunks_iterative(
                    chunks, model, custom_instructions=custom_instructions,
                    timeout=timeout,
                    thinking_params=thinking_params,
                    summary_runtime=summary_runtime,
                )
                if oversized_count > 0:
                    summary += f"\n\n[Note: {oversized_count} oversized message(s) excluded from summary]"
                return summary
    except Exception as e:
        print(f"[Compaction] Tier 2 (filtered) failed: {e}")

    # Tier 3: static fallback
    return (
        f"[Compaction fallback] {len(messages)} messages could not be summarized. "
        f"The conversation contained tool calls, computer interactions, and text exchanges."
    )


# ---------------------------------------------------------------------------
# Main compaction entry point
# ---------------------------------------------------------------------------

async def compact_messages(
    messages: list[dict[str, Any]],
    model: str,
    context_window: int,
    *,
    instructions_tokens: int = 0,
    max_history_share: float = 0.5,
    recent_turns_preserve: int = 3,
    custom_instructions: str | None = None,
    timeout: int = SUMMARIZATION_TIMEOUT,
    thinking_params: dict[str, Any] | None = None,
    summary_runtime: ResolvedModel | None = None,
) -> CompactionResult:
    """Compact older conversation messages into a summary with budget-aware splitting.

    Budget-aware compaction (US-OC-013): calculates a token budget for kept messages
    based on context_window * max_history_share, iteratively pruning the kept portion
    until it fits. Recent turns are split out and preserved unconditionally.

    Adapted from OpenClaw's pruneHistoryForContextShare() and splitPreservedRecentTurns().

    Args:
        messages: Full message history to compact.
        model: litellm model string for summarization LLM calls.
        context_window: Context window size in tokens.
        instructions_tokens: Estimated token count for system instructions.
        max_history_share: Maximum share of context window for kept history (default 0.5).
        recent_turns_preserve: Number of recent user turns to preserve unconditionally.
        custom_instructions: Optional instructions for the summarization prompt.

    Returns:
        CompactionResult with summary, token counts, and split point.
    """
    if not messages:
        return CompactionResult(
            summary=DEFAULT_SUMMARY_FALLBACK,
            tokens_before=0,
            tokens_after=0,
            first_kept_message_index=0,
            chunks_processed=0,
        )

    tokens_before = estimate_messages_tokens(messages)
    print(f"[Compaction] Starting: {len(messages)} messages, ~{tokens_before} tokens")

    # 0. Split out preserved recent turns (never pruned or summarized)
    pruneable, preserved = split_preserved_recent_turns(messages, recent_turns_preserve)
    preserved_tokens = estimate_messages_tokens(preserved) if preserved else 0
    print(
        f"[Compaction] Preserved {len(preserved)} recent messages "
        f"(~{preserved_tokens} tokens), {len(pruneable)} pruneable"
    )

    # 1. Budget calculation for kept pruneable messages
    # Budget = (context_window * max_history_share) - instructions - summary estimate - preserved
    summary_estimate = SUMMARIZATION_OVERHEAD_TOKENS
    available_for_kept = (
        int(context_window * max_history_share)
        - instructions_tokens
        - summary_estimate
        - preserved_tokens
    )
    available_for_kept = max(available_for_kept, 2000)  # safety floor

    # 1b. Overflow fallback: if pruneable is empty but preserved exceeds budget,
    # move older preserved messages into pruneable so compaction can actually run.
    # This handles CUA-style conversations where turn counting still leaves
    # too many messages in the preserved set for the available budget.
    if not pruneable and preserved and preserved_tokens > available_for_kept + summary_estimate:
        # Keep the most recent half of preserved, compact the rest
        overflow_halves = chunk_messages_by_token_share(preserved, parts=2)
        if len(overflow_halves) >= 2:
            pruneable = overflow_halves[0]
            preserved = overflow_halves[1]
            preserved_tokens = estimate_messages_tokens(preserved)
            # Recalculate budget with new preserved size
            available_for_kept = (
                int(context_window * max_history_share)
                - instructions_tokens
                - summary_estimate
                - preserved_tokens
            )
            available_for_kept = max(available_for_kept, 2000)
            print(
                f"[Compaction] Overflow: moved {len(pruneable)} preserved messages "
                f"to pruneable, {len(preserved)} remain preserved "
                f"(~{preserved_tokens} tokens)"
            )

    # 2. Initial half-split on pruneable messages
    if not pruneable:
        to_compact: list[dict[str, Any]] = []
        to_keep: list[dict[str, Any]] = []
    else:
        halves = chunk_messages_by_token_share(pruneable, parts=2)
        if len(halves) < 2:
            to_compact = pruneable
            to_keep = []
        else:
            to_compact = halves[0]
            to_keep = halves[1]

    # 3. Iterative pruning: while kept exceeds budget, split kept in half,
    #    move older half to to_compact, repair pairing on remainder
    prune_iterations = 0
    while to_keep and estimate_messages_tokens(to_keep) > available_for_kept:
        sub_halves = chunk_messages_by_token_share(to_keep, parts=2)
        if len(sub_halves) < 2:
            break  # can't split further
        to_compact = to_compact + sub_halves[0]
        to_keep = sub_halves[1]
        repair = repair_tool_use_result_pairing(to_keep)
        to_keep = repair.messages
        prune_iterations += 1

    # 4. Final repair + recombine: to_keep + preserved = full kept portion
    if to_keep:
        repair = repair_tool_use_result_pairing(to_keep)
        to_keep = repair.messages
    final_kept = to_keep + preserved
    first_kept_index = len(messages) - len(final_kept)

    # Adaptive chunk ratio for the to-compact portion
    chunk_ratio = compute_adaptive_chunk_ratio(to_compact, context_window)
    max_chunk_tokens = int(context_window * chunk_ratio) - SUMMARIZATION_OVERHEAD_TOKENS
    max_chunk_tokens = max(max_chunk_tokens, 2000)  # safety floor

    print(
        f"[Compaction] Split: {len(to_compact)} to compact, {len(to_keep)} kept (pruneable), "
        f"{len(preserved)} preserved, budget={available_for_kept}, "
        f"prune_iterations={prune_iterations}, chunk_ratio={chunk_ratio:.2f}"
    )

    # Summarize the older portion
    if to_compact:
        summary = await summarize_with_fallback(
            to_compact,
            model,
            context_window,
            max_chunk_tokens,
            custom_instructions=custom_instructions,
            timeout=timeout,
            thinking_params=thinking_params,
            summary_runtime=summary_runtime,
        )
    else:
        summary = DEFAULT_SUMMARY_FALLBACK

    # Count chunks processed
    chunks = chunk_messages_by_max_tokens(to_compact, max_chunk_tokens) if to_compact else []
    chunks_processed = len(chunks)

    # Estimate tokens after: summary + kept messages
    summary_tokens = len(summary) // 4
    kept_tokens = estimate_messages_tokens(final_kept) if final_kept else 0
    tokens_after = summary_tokens + kept_tokens

    print(
        f"[Compaction] Done: ~{tokens_before} → ~{tokens_after} tokens "
        f"({chunks_processed} chunks summarized)"
    )

    return CompactionResult(
        summary=summary,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        first_kept_message_index=first_kept_index,
        chunks_processed=chunks_processed,
    )
