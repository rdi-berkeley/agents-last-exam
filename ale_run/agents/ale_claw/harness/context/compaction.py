"""Context compaction & summarization pipeline.

Split out of context.py: budget-aware history pruning, chunked summarization,
and tool-pairing repair. Depends only on the token-estimation leaf and external
leaf modules (model_config / helper_runtime / canonical_sanitize) — never on the
context-overflow core — so context.py can re-export this without a cycle.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from ..canonical.canonical_sanitize import _SKIP_SYNTHESIS_STOP_REASONS
from ..model.helper_runtime import call_helper_model
from ..model.model_config import ResolvedModel, resolve_model
from .token_estimation import (
    SAFETY_MARGIN,
    estimate_message_tokens,
    estimate_messages_tokens,
)


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
# Tool pairing repair
#
# Adapted from openclaw/src/agents/session-transcript-repair.ts
# (repairToolUseResultPairing). Fixes orphaned tool_use/tool_result pairs
# that arise when messages are split at arbitrary boundaries during compaction.
# ---------------------------------------------------------------------------

# NOTE: This synthetic string is intentionally distinct from canonical.py's
# SYNTHETIC_TOOL_RESULT_CONTENT — different audience (the compaction path).
# The skip-synthesis stop-reason set, by contrast, MUST stay in sync, so it is
# imported from canonical (single source) rather than re-declared here.
SYNTHETIC_TOOL_RESULT_CONTENT = (
    "[compaction] missing tool result — synthetic error result for transcript repair."
)


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

    Sibling of ``canonical.repair_orphaned_pairs`` — same 3-pass shape, but the
    two deliberately differ (message shape, computer_call handling, return
    type) and the differences are test-pinned, so DO NOT merge them. See that
    function's docstring for the full divergence list.

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
# Recent turns preservation
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

def _compute_kept_budget(
    context_window: int,
    max_history_share: float,
    instructions_tokens: int,
    preserved_tokens: int,
) -> int:
    """Token budget for the kept pruneable messages, with a 2000-token floor.

    Budget = context_window * max_history_share - instructions - summary estimate
    - preserved.
    """
    available = (
        int(context_window * max_history_share)
        - instructions_tokens
        - SUMMARIZATION_OVERHEAD_TOKENS
        - preserved_tokens
    )
    return max(available, 2000)  # safety floor


def _resolve_pruneable_and_preserved(
    messages: list[dict[str, Any]],
    recent_turns_preserve: int,
    context_window: int,
    max_history_share: float,
    instructions_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Split out preserved recent turns and compute the kept budget.

    Returns ``(pruneable, preserved, available_for_kept)``. Includes the overflow
    fallback that moves older preserved messages into pruneable when the
    preserved set alone already exceeds the budget (CUA-style conversations whose
    turn counting leaves too many messages preserved).
    """
    pruneable, preserved = split_preserved_recent_turns(messages, recent_turns_preserve)
    preserved_tokens = estimate_messages_tokens(preserved) if preserved else 0
    print(
        f"[Compaction] Preserved {len(preserved)} recent messages "
        f"(~{preserved_tokens} tokens), {len(pruneable)} pruneable"
    )

    available_for_kept = _compute_kept_budget(
        context_window, max_history_share, instructions_tokens, preserved_tokens
    )

    if (
        not pruneable
        and preserved
        and preserved_tokens > available_for_kept + SUMMARIZATION_OVERHEAD_TOKENS
    ):
        # Keep the most recent half of preserved, compact the rest
        overflow_halves = chunk_messages_by_token_share(preserved, parts=2)
        if len(overflow_halves) >= 2:
            pruneable = overflow_halves[0]
            preserved = overflow_halves[1]
            preserved_tokens = estimate_messages_tokens(preserved)
            available_for_kept = _compute_kept_budget(
                context_window, max_history_share, instructions_tokens, preserved_tokens
            )
            print(
                f"[Compaction] Overflow: moved {len(pruneable)} preserved messages "
                f"to pruneable, {len(preserved)} remain preserved "
                f"(~{preserved_tokens} tokens)"
            )
    return pruneable, preserved, available_for_kept


def _split_compact_and_keep(
    pruneable: list[dict[str, Any]],
    available_for_kept: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Half-split pruneable, then iteratively prune the kept side until it fits.

    Each pruning step moves the older half into to_compact and repairs tool
    pairing on the remainder. Returns ``(to_compact, to_keep, prune_iterations)``.
    """
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
    return to_compact, to_keep, prune_iterations


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

    Budget-aware compaction: calculates a token budget for kept messages
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

    # 0-1. Split out preserved recent turns and compute the kept budget
    pruneable, preserved, available_for_kept = _resolve_pruneable_and_preserved(
        messages, recent_turns_preserve, context_window, max_history_share, instructions_tokens
    )

    # 2-3. Half-split pruneable, then iteratively prune the kept side to budget
    to_compact, to_keep, prune_iterations = _split_compact_and_keep(
        pruneable, available_for_kept
    )

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
