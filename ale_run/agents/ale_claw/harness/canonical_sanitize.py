"""Canonical sanitize passes — repair, ordering, thinking sanitization.

Split out of ``canonical.py``: the individual leaf passes that operate on
``list[CanonicalMessage]``. The orchestrator (``sanitize_items``) and the format
adapters stay in canonical.py and call these. These passes depend only on the
canonical TYPES (``canonical_types``) — never the adapters — so the split is
cycle-free and canonical.py re-exports them.

Reference:
  - openclaw/src/agents/session-transcript-repair.ts — repair passes
  - openclaw/src/agents/pi-embedded-runner/thinking.ts — dropThinkingBlocks
  - openclaw/src/agents/pi-embedded-helpers/openai.ts — downgradeOpenAIReasoningBlocks
"""

from __future__ import annotations

import json
from typing import Any

from .canonical_types import (
    CanonicalMessage,
    ComputerCallBlock,
    ContentBlock,
    FunctionCallBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
)


SYNTHETIC_TOOL_RESULT_CONTENT = (
    "[compacted — result no longer available]"
)
"""Synthetic content for tool results inserted by repair_orphaned_pairs."""

_SKIP_SYNTHESIS_STOP_REASONS = frozenset({"error", "aborted"})
"""Stop reasons where we skip synthesizing tool results (the call was interrupted).

Single source of truth — context.repair_tool_use_result_pairing imports this so
the two repair paths can never drift on which stop reasons skip synthesis.
"""


def _collect_orphans(
    messages: list[CanonicalMessage],
) -> tuple[dict[str, tuple[str, str | None]], set[str], bool]:
    """Pass 1: scan messages for orphans.

    Returns ``(orphaned_calls, orphaned_results, has_duplicate_results)`` —
    orphaned_calls maps call id → (block_type, stop_reason) for calls lacking a
    matching result; orphaned_results is result ids lacking a matching call.
    """
    call_ids: dict[str, tuple[str, str | None]] = {}  # id → (block_type, stop_reason)
    result_ids: set[str] = set()
    has_duplicate_results = False
    _result_id_counts: dict[str, int] = {}

    for msg in messages:
        stop_reason = msg.get("stop_reason")
        for block in msg.get("content", []):
            btype = block.get("type", "")
            if btype in ("function_call", "computer_call"):
                bid = block.get("id", "")
                if bid:
                    call_ids[bid] = (btype, stop_reason)
            elif btype == "tool_result":
                rid = block.get("tool_use_id", "")
                if rid:
                    result_ids.add(rid)
                    _result_id_counts[rid] = _result_id_counts.get(rid, 0) + 1
                    if _result_id_counts[rid] > 1:
                        has_duplicate_results = True

    valid_call_id_set = set(call_ids.keys())
    matched_ids = valid_call_id_set & result_ids
    orphaned_calls = {
        cid: info for cid, info in call_ids.items() if cid not in matched_ids
    }
    orphaned_results = result_ids - valid_call_id_set
    return orphaned_calls, orphaned_results, has_duplicate_results


def _filter_orphaned_results(
    messages: list[CanonicalMessage],
    orphaned_calls: dict[str, tuple[str, str | None]],
    orphaned_results: set[str],
    synthesize: bool,
) -> list[CanonicalMessage]:
    """Pass 2: drop orphaned/duplicate tool_results; in replay mode
    (``synthesize=False``) also drop orphaned calls from assistant messages."""
    seen_result_ids: set[str] = set()
    filtered: list[CanonicalMessage] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", [])

        if role == "tool":
            new_blocks: list[ContentBlock] = []
            for block in content:
                if block.get("type") == "tool_result":
                    rid = block.get("tool_use_id", "")
                    # Drop orphaned results (no matching call)
                    if rid and rid in orphaned_results:
                        continue
                    # Drop duplicates
                    if rid in seen_result_ids:
                        continue
                    if rid:
                        seen_result_ids.add(rid)
                new_blocks.append(block)
            if new_blocks:
                new_msg = dict(msg)
                new_msg["content"] = new_blocks
                filtered.append(new_msg)
        elif role == "assistant" and not synthesize:
            # Drop orphaned calls when not synthesizing (replay mode)
            new_blocks_a: list[ContentBlock] = []
            for block in content:
                btype = block.get("type", "")
                if btype in ("function_call", "computer_call"):
                    bid = block.get("id", "")
                    if bid in orphaned_calls:
                        continue
                new_blocks_a.append(block)
            if new_blocks_a:
                new_msg = dict(msg)
                new_msg["content"] = new_blocks_a
                filtered.append(new_msg)
        else:
            filtered.append(msg)
    return filtered


def _synthesize_missing_results(
    filtered: list[CanonicalMessage],
    orphaned_calls: dict[str, tuple[str, str | None]],
) -> list[CanonicalMessage]:
    """Pass 3: after each assistant message, insert a synthetic error
    tool_result for each unmatched function_call.

    computer_calls are skipped (can't synthesize a screenshot) and error/aborted
    turns are skipped (``_SKIP_SYNTHESIS_STOP_REASONS``).
    """
    final: list[CanonicalMessage] = []
    for msg in filtered:
        final.append(msg)
        role = msg.get("role", "")
        if role != "assistant":
            continue

        stop_reason = msg.get("stop_reason")
        synthetic_blocks: list[ContentBlock] = []

        for block in msg.get("content", []):
            btype = block.get("type", "")
            if btype not in ("function_call", "computer_call"):
                continue
            bid = block.get("id", "")
            if bid not in orphaned_calls:
                continue

            call_type, _ = orphaned_calls[bid]

            if call_type == "computer_call":
                # Can't synthesize valid screenshots — drop by removing from
                # the assistant message content
                continue

            if stop_reason in _SKIP_SYNTHESIS_STOP_REASONS:
                continue

            synthetic_blocks.append(ToolResultBlock(
                type="tool_result",
                tool_use_id=bid,
                content=SYNTHETIC_TOOL_RESULT_CONTENT,
                is_error=True,
            ))

        if synthetic_blocks:
            final.append(CanonicalMessage(
                role="tool",
                content=synthetic_blocks,
            ))
    return final


def _drop_orphaned_computer_calls(
    final: list[CanonicalMessage],
    orphaned_calls: dict[str, tuple[str, str | None]],
) -> list[CanonicalMessage]:
    """Final cleanup: strip orphaned computer_call blocks from assistant messages."""
    cleaned: list[CanonicalMessage] = []
    for msg in final:
        role = msg.get("role", "")
        if role == "assistant":
            new_blocks_c: list[ContentBlock] = []
            for block in msg.get("content", []):
                btype = block.get("type", "")
                if btype == "computer_call":
                    bid = block.get("id", "")
                    if bid in orphaned_calls:
                        continue
                new_blocks_c.append(block)
            if new_blocks_c:
                new_msg = dict(msg)
                new_msg["content"] = new_blocks_c
                cleaned.append(new_msg)
        else:
            cleaned.append(msg)
    return cleaned


def repair_orphaned_pairs(
    messages: list[CanonicalMessage],
    *,
    synthesize: bool = True,
) -> list[CanonicalMessage]:
    """Repair orphaned tool call / result pairs in canonical messages.

    Sibling of ``context.repair_tool_use_result_pairing`` — same 3-pass shape,
    but DO NOT merge them: they diverge deliberately and the divergences are
    test-pinned.
      - This one operates on ``CanonicalMessage`` (content always a block list);
        context's takes role-based dicts whose content may be a bare string.
      - This one DROPS orphaned ``computer_call`` blocks (can't synthesize a
        screenshot); context SYNTHESIZES a text result for them
        (test_openclaw_compaction::test_repair_handles_computer_call).
      - This returns ``list[CanonicalMessage]``; context returns a
        ``ToolPairingRepairReport`` with drop/synthesize counts.
    They share ``_SKIP_SYNTHESIS_STOP_REASONS`` (imported from here) so the
    skip-synthesis policy can't drift; the synthetic-result strings differ on
    purpose (different audiences).

    Algorithm (3-pass, matching OpenClaw's session-transcript-repair.ts):
      1. Collect call IDs from assistant messages (FunctionCallBlock/ComputerCallBlock)
      2. Match ToolResultBlocks by tool_use_id. Drop orphaned results and duplicates.
      3. Insert synthetic ToolResultBlock for unmatched function_calls (skip if
         stop_reason is "error"/"aborted"). Drop unmatched computer_calls entirely
         (can't synthesize valid screenshots).

    Args:
        messages: Canonical messages to repair.
        synthesize: If True (default), insert synthetic error results for unmatched
            function_calls. If False, drop unmatched calls instead (replay mode).
    """
    if not messages:
        return []

    orphaned_calls, orphaned_results, has_duplicate_results = _collect_orphans(messages)

    # Fast path: nothing to repair
    if not orphaned_calls and not orphaned_results and not has_duplicate_results:
        return messages

    filtered = _filter_orphaned_results(
        messages, orphaned_calls, orphaned_results, synthesize
    )
    if not synthesize:
        return filtered

    final = _synthesize_missing_results(filtered, orphaned_calls)
    return _drop_orphaned_computer_calls(final, orphaned_calls)


def ensure_valid_ordering(
    messages: list[CanonicalMessage],
) -> list[CanonicalMessage]:
    """Ensure messages don't end with role=assistant.

    Non-prefill models (like Opus 4.6) reject API calls where the last message
    is from the assistant. Appends a user continuation message if needed.
    """
    if not messages:
        return messages
    if messages[-1].get("role") == "assistant":
        messages = list(messages)
        messages.append(CanonicalMessage(
            role="user",
            content=[TextBlock(type="text", text="[Continue from where you left off.]")],
        ))
    return messages


# ---------------------------------------------------------------------------
# Thinking sanitization passes (US-OC-041)
# ---------------------------------------------------------------------------


def drop_thinking_blocks(
    messages: list[CanonicalMessage],
) -> list[CanonicalMessage]:
    """Strip type="thinking" content blocks from assistant messages.

    If an assistant message becomes empty after stripping, it is replaced
    with a synthetic ``TextBlock(type="text", text="")`` to preserve turn
    structure (some providers require strict user/assistant alternation).

    Returns the original list when nothing was changed (callers can use
    reference equality to skip downstream work).

    Reference: openclaw/src/agents/pi-embedded-runner/thinking.ts:dropThinkingBlocks
    """
    touched = False
    out: list[CanonicalMessage] = []

    for msg in messages:
        if msg.get("role") != "assistant":
            out.append(msg)
            continue

        next_content: list[ContentBlock] = []
        changed = False
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "thinking":
                touched = True
                changed = True
                continue
            next_content.append(block)

        if not changed:
            out.append(msg)
            continue

        content = (
            next_content
            if next_content
            else [TextBlock(type="text", text="")]
        )
        new_msg = dict(msg)
        new_msg["content"] = content
        out.append(new_msg)  # type: ignore[arg-type]

    return out if touched else messages


def sanitize_thinking_signatures(
    messages: list[CanonicalMessage],
) -> list[CanonicalMessage]:
    """Remove thinkingSignature fields from thinking blocks.

    The signature is a tamper-proof token validated by the API on
    re-submission — if it comes from a different provider or session,
    the API rejects the request.  Stripping it allows cross-provider
    transcript replay.

    Returns the original list when nothing was changed.

    Reference: openclaw/src/agents/transcript-policy.ts (sanitizeThinkingSignatures flag)
    """
    touched = False
    out: list[CanonicalMessage] = []

    for msg in messages:
        if msg.get("role") != "assistant":
            out.append(msg)
            continue

        next_content: list[ContentBlock] = []
        changed = False
        for block in msg.get("content", []):
            if (
                isinstance(block, dict)
                and block.get("type") == "thinking"
                and "thinkingSignature" in block
            ):
                touched = True
                changed = True
                new_block = dict(block)
                del new_block["thinkingSignature"]
                next_content.append(new_block)  # type: ignore[arg-type]
            else:
                next_content.append(block)

        if not changed:
            out.append(msg)
        else:
            new_msg = dict(msg)
            new_msg["content"] = next_content
            out.append(new_msg)  # type: ignore[arg-type]

    return out if touched else messages


def _parse_openai_reasoning_signature(value: Any) -> bool:
    """Check if a thinkingSignature is a valid OpenAI reasoning signature.

    OpenAI reasoning signatures are JSON objects with ``id`` and ``type``
    fields.  Returns True if the value parses as such.

    Reference: openclaw/src/agents/pi-embedded-helpers/openai.ts:parseOpenAIReasoningSignature
    """
    if not value:
        return False
    candidate = None
    if isinstance(value, str):
        trimmed = value.strip()
        if not (trimmed.startswith("{") and trimmed.endswith("}")):
            return False
        try:
            candidate = json.loads(trimmed)
        except (json.JSONDecodeError, ValueError):
            return False
    elif isinstance(value, dict):
        candidate = value
    else:
        return False

    return (
        isinstance(candidate, dict)
        and isinstance(candidate.get("id"), str)
        and isinstance(candidate.get("type"), str)
    )


def _get_openai_reasoning_signature(value: Any) -> dict[str, str] | None:
    """Parse an OpenAI reasoning signature payload into its id/type pair."""
    if not value:
        return None
    candidate = None
    if isinstance(value, str):
        trimmed = value.strip()
        if not (trimmed.startswith("{") and trimmed.endswith("}")):
            return None
        try:
            candidate = json.loads(trimmed)
        except (json.JSONDecodeError, ValueError):
            return None
    elif isinstance(value, dict):
        candidate = value
    else:
        return None

    if not isinstance(candidate, dict):
        return None
    item_id = candidate.get("id")
    item_type = candidate.get("type")
    if not isinstance(item_id, str) or not isinstance(item_type, str):
        return None
    return {"id": item_id, "type": item_type}


def _has_following_non_thinking_block(
    content: list[ContentBlock], index: int
) -> bool:
    """Check if there is a non-thinking block after the given index.

    Reference: openclaw/src/agents/pi-embedded-helpers/openai.ts:hasFollowingNonThinkingBlock
    """
    for i in range(index + 1, len(content)):
        block = content[i]
        if not isinstance(block, dict):
            return True
        if block.get("type") != "thinking":
            return True
    return False


def downgrade_openai_reasoning(
    messages: list[CanonicalMessage],
) -> list[CanonicalMessage]:
    """Drop orphaned OpenAI reasoning blocks from assistant messages.

    A thinking block is "orphaned" if it has a valid OpenAI reasoning
    signature (JSON ``{id, type}``) but no following non-thinking block
    in the same assistant message.  Such blocks cause API rejection on
    replay because the Responses API validates reasoning items against
    server-side state.

    Thinking blocks *without* a valid OpenAI signature are left untouched
    (they may be from Anthropic or other providers).

    If all blocks in an assistant message are dropped, the entire message
    is removed (matching OpenClaw behavior).

    Returns the original list when nothing was changed.

    Reference: openclaw/src/agents/pi-embedded-helpers/openai.ts:downgradeOpenAIReasoningBlocks
    """
    touched = False
    out: list[CanonicalMessage] = []

    for msg in messages:
        if msg.get("role") != "assistant":
            out.append(msg)
            continue

        content = msg.get("content", [])
        next_content: list[ContentBlock] = []
        changed = False

        for i, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "thinking":
                next_content.append(block)
                continue

            sig = block.get("thinkingSignature")
            if not _parse_openai_reasoning_signature(sig):
                # Not an OpenAI reasoning block — keep it
                next_content.append(block)
                continue

            if _has_following_non_thinking_block(content, i):
                # Part of a valid call chain — keep it
                next_content.append(block)
                continue

            # Orphaned OpenAI reasoning — drop it
            touched = True
            changed = True

        if not changed:
            out.append(msg)
            continue

        if not next_content:
            # All blocks dropped — remove the message entirely
            continue

        new_msg = dict(msg)
        new_msg["content"] = next_content
        out.append(new_msg)  # type: ignore[arg-type]

    return out if touched else messages
