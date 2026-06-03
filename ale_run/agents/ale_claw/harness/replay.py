"""Transcript Replay — cross-run continuity (US-OC-012).

Split out of ``session.py`` (one of its four concerns). Converts transcript
entries to API messages, sanitizes stale data, repairs tool pairing, limits
history, and converts to Responses API items.

Based on OpenClaw's replay pipeline (pi-embedded-runner/sanitizeSessionHistory,
limitHistoryTurns, sanitizeToolUseResultPairing).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from .canonical import _normalize_actions

if TYPE_CHECKING:
    from .session import TranscriptEntry


_THINKING_BLOCK_RE = re.compile(r"<THINKING>.*?</THINKING>", re.DOTALL)

# Base64 data URL pattern (matches data:image/...;base64,...)
_BASE64_IMAGE_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}")


def _get_openai_reasoning_signature(value: Any) -> dict[str, str] | None:
    """Parse a persisted OpenAI thinkingSignature back into id/type metadata."""
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


def build_replay_messages(entries: list[TranscriptEntry]) -> list[dict[str, Any]]:
    """Convert transcript entries into API message dicts for replay.

    Handles three entry types:
    - message: extracts role/content from entry.data["message"], strips stale metadata
    - compaction: replaces all messages before firstKeptEntryId with a single
      assistant summary message
    - session: skipped (run boundary markers, not API messages)

    Based on OpenClaw's replaceMessages pipeline
    (openclaw/pi-embedded-runner/sanitizeSessionHistory).
    """
    messages: list[dict[str, Any]] = []
    # Map entry IDs to their index in the messages list for compaction lookups
    entry_id_to_msg_index: dict[str, int] = {}

    for entry in entries:
        if entry.type == "session":
            continue

        if entry.type == "compaction":
            summary = entry.data.get("summary", "")
            first_kept_id = entry.data.get("firstKeptEntryId")

            if first_kept_id and first_kept_id in entry_id_to_msg_index:
                # Replace all messages before firstKeptEntryId with a summary
                cut_index = entry_id_to_msg_index[first_kept_id]
                kept = messages[cut_index:]
                messages = [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"[Compaction summary] {summary}"}],
                    }
                ] + kept
                # Rebuild index since positions shifted
                entry_id_to_msg_index = {}
                # We can't easily remap, but subsequent compactions will
                # reference entries added after this point
            else:
                # No matching entry found — just prepend the summary
                messages.insert(0, {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"[Compaction summary] {summary}"}],
                })
            continue

        if entry.type == "message":
            msg_data = entry.data.get("message", {})
            role = msg_data.get("role", "user")
            content = msg_data.get("content", "")

            # Map OpenClaw roles to standard API roles
            if role == "toolResult":
                role = "user"  # tool results are user messages in the API

            # Strip stale metadata (usage, stopReason, api)
            message: dict[str, Any] = {"role": role, "content": content}
            entry_id_to_msg_index[entry.id] = len(messages)
            messages.append(message)

    return messages


def sanitize_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitize replayed messages for API compatibility.

    Five sanitization passes:
    1. Strip base64 images (huge, stale screenshots from prior runs)
    2. Strip thinking blocks (<THINKING>...</THINKING>) from assistant text
    3. Repair orphaned tool results (drop tool_result without matching call, and vice versa)
    4. Strip stale usage keys from all messages
    5. Ensure user-first ordering (Gemini compatibility)

    Based on OpenClaw's sanitizeSessionHistory and sanitizeToolUseResultPairing
    (openclaw/pi-embedded-runner/).
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        result.append(_sanitize_single_message(msg))

    # Pass 3: Repair orphaned tool results / calls
    result = _repair_orphaned_tool_pairs(result)

    # Pass 5: Ensure user-first ordering
    if result and result[0].get("role") != "user":
        result.insert(0, {"role": "user", "content": "[session history follows]"})

    return result


def _sanitize_single_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Apply per-message sanitization (passes 1, 2, 4)."""
    role = msg.get("role", "")
    content = msg.get("content", "")

    # Pass 4: Strip stale usage
    sanitized: dict[str, Any] = {"role": role}
    for k, v in msg.items():
        if k not in ("usage", "stopReason", "api", "role", "content"):
            sanitized[k] = v

    if isinstance(content, str):
        # Pass 1: Strip base64 images from text
        content = _BASE64_IMAGE_RE.sub("[image removed]", content)
        # Pass 2: Strip thinking blocks from assistant messages
        if role == "assistant":
            content = _THINKING_BLOCK_RE.sub("", content).strip()
        sanitized["content"] = content  # keep even if empty
        return sanitized

    if isinstance(content, list):
        sanitized_blocks: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type", "")

            # Pass 1: Strip image blocks entirely
            if block_type in ("image", "image_url"):
                sanitized_blocks.append({
                    "type": "text",
                    "text": "[screenshot from prior run]",
                })
                continue

            # Pass 1: Strip base64 from source blocks
            if (
                isinstance(block.get("source"), dict)
                and block["source"].get("type") == "base64"
            ):
                sanitized_blocks.append({
                    "type": "text",
                    "text": "[screenshot from prior run]",
                })
                continue

            # Pass 1: Strip base64 data URLs from text blocks
            if block_type == "text":
                text = block.get("text", "")
                text = _BASE64_IMAGE_RE.sub("[image removed]", text)
                # Pass 2: Strip thinking blocks
                if role == "assistant":
                    text = _THINKING_BLOCK_RE.sub("", text).strip()
                sanitized_blocks.append({"type": "text", "text": text})
                continue

            # Keep other blocks as-is (function_call, computer_call, tool_result, etc.)
            sanitized_blocks.append(block)

        sanitized["content"] = sanitized_blocks
        return sanitized

    sanitized["content"] = content
    return sanitized


def _repair_orphaned_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove orphaned tool results and tool calls without matching pairs.

    For each tool_result content block, verify a matching function_call or
    computer_call with the same ID exists in an earlier assistant message.
    Drop orphaned results. Also drop function_call/computer_call blocks
    with no matching result in a later message.

    Based on OpenClaw's sanitizeToolUseResultPairing.
    """
    # Collect all tool call IDs from assistant messages
    call_ids: set[str] = set()
    result_ids: set[str] = set()

    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            block_type = block.get("type", "")
            if block_type in ("function_call", "computer_call"):
                call_id = block.get("id", "")
                if call_id:
                    call_ids.add(call_id)
            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                if tool_use_id:
                    result_ids.add(tool_use_id)

    # IDs that have both a call and a result
    paired_ids = call_ids & result_ids

    # Filter messages: remove orphaned blocks
    cleaned: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            cleaned.append(msg)
            continue

        filtered_blocks: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type", "")
            if block_type in ("function_call", "computer_call"):
                call_id = block.get("id", "")
                if call_id in paired_ids:
                    filtered_blocks.append(block)
                # else: orphaned call — drop
            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id", "")
                if tool_use_id in paired_ids:
                    filtered_blocks.append(block)
                # else: orphaned result — drop
            else:
                filtered_blocks.append(block)

        if filtered_blocks:
            cleaned.append({**msg, "content": filtered_blocks})
        # If all blocks were removed, skip the message entirely

    return cleaned


def limit_history_turns(
    messages: list[dict[str, Any]], limit: int | None = None
) -> list[dict[str, Any]]:
    """Keep the last N user turns (and their associated responses).

    Iterates backwards, counting role="user" messages. When the count exceeds
    the limit, slices from that point forward.

    Based on OpenClaw's limitHistoryTurns (pi-embedded-runner/).

    Args:
        messages: List of API message dicts
        limit: Max number of user turns to keep. None or <= 0 means keep all.

    Returns:
        Sliced message list containing only the last `limit` user turns
        and their associated assistant/tool responses.
    """
    if limit is None or limit <= 0:
        return messages

    # Find the indices of all user messages
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]

    if len(user_indices) <= limit:
        return messages

    # Keep from the (limit)th-from-last user message onward
    cut_index = user_indices[-limit]
    return messages[cut_index:]


def _string_message_item(role: str, content: str) -> dict[str, Any]:
    """A Responses API message item from a plain string — assistant emits
    ``output_text``; every other role emits ``input_text``."""
    if role == "assistant":
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        }
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": content}],
    }


def convert_to_responses_api_items(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert replay messages from Chat Completions format to Responses API items.

    CUA SDK agent loops (anthropic.py, openai.py) dispatch on top-level ``type``
    fields (Responses API item format), not on ``role`` with nested content blocks
    (Chat Completions format).  Our transcript stores the latter, so without this
    conversion the loops drop function_call / computer_call blocks and ignore
    ``role: "tool"`` messages entirely.

    Unnests each Chat Completions message into one or more flat Responses API
    items that the loops can dispatch correctly:

    - User message → ``{type: "message", role: "user", content: [{type: "input_text", …}]}``
    - Assistant text → ``{type: "message", role: "assistant", content: [{type: "output_text", …}]}``
    - function_call block → ``{type: "function_call", call_id, name, arguments}``
    - computer_call block → ``{type: "computer_call", call_id, action}``
    - tool_result block → ``{type: "function_call_output", …}`` or
      ``{type: "computer_call_output", …}`` (matched by call_id to original call type)

    The ``id`` → ``call_id`` mapping matches what ``group_step_output`` stores
    (``"id"`` key in transcript) versus what the Responses API expects
    (``"call_id"`` key).

    Applied as the LAST step in the replay pipeline (after sanitize_history +
    limit_history_turns) so that orphaned-pair repair still works on the nested
    format with ``id`` / ``tool_use_id`` fields.

    US-OC-022: Replay Format Fix.
    """
    items: list[dict[str, Any]] = []
    # Track call types so tool results emit the correct output type
    # (computer_call → computer_call_output, function_call → function_call_output)
    call_type_map: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # --- String content ---
        if isinstance(content, str):
            items.append(_string_message_item(role, content))
            continue

        # --- List content (content blocks) ---
        if isinstance(content, list):
            if role == "assistant":
                unnested = _unnest_assistant_blocks(content, call_type_map)
                # Record call types for later tool result matching
                for item in unnested:
                    itype = item.get("type", "")
                    if itype in ("function_call", "computer_call"):
                        cid = item.get("call_id", "")
                        if cid:
                            call_type_map[cid] = itype
                items.extend(unnested)
            elif role in ("tool", "user"):
                # Check if this is a tool-result message (all blocks are tool_result)
                has_tool_results = any(
                    b.get("type") == "tool_result" for b in content if isinstance(b, dict)
                )
                if role == "tool" or (role == "user" and has_tool_results):
                    items.extend(_unnest_tool_blocks(content, call_type_map))
                else:
                    items.append(_wrap_user_content(content))
            else:
                # Unknown role — treat as user
                items.append(_wrap_user_content(content))
            continue

        # --- Fallback: wrap as user ---
        items.append(_string_message_item("user", str(content)))

    return _ensure_tool_adjacency(items)


def _ensure_tool_adjacency(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder items so each tool call is immediately followed by its output.

    Memory flush messages (user prompt + assistant reply) can be interleaved
    in the transcript between a tool call and its result, because the flush
    fires per-step while tool results may be logged in a later step.  The
    Anthropic API requires ``tool_result`` immediately after ``tool_use``,
    so we defer any non-output items that appear between a call and its
    matching output, then flush them after the output.
    """
    result: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    pending_call_ids: set[str] = set()

    for item in items:
        t = item.get("type", "")

        if t in ("function_call", "computer_call"):
            pending_call_ids.add(item.get("call_id", ""))
            result.append(item)
        elif t in ("function_call_output", "computer_call_output"):
            call_id = item.get("call_id", "")
            pending_call_ids.discard(call_id)
            result.append(item)
            # All pending calls resolved → flush deferred items
            if not pending_call_ids:
                result.extend(deferred)
                deferred = []
        elif pending_call_ids:
            # Non-tool item while tool calls are pending → defer
            deferred.append(item)
        else:
            result.append(item)

    # Flush any remaining deferred items
    result.extend(deferred)
    return result


def _flush_pending_text(
    items: list[dict[str, Any]],
    pending_text: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Emit accumulated text blocks as one assistant message (if any) and return
    a fresh, empty pending-text list."""
    if pending_text:
        items.append({
            "type": "message",
            "role": "assistant",
            "content": pending_text,
        })
    return []


def _unnest_assistant_blocks(
    blocks: list[dict[str, Any]],
    call_type_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Unnest an assistant message's content blocks into Responses API items.

    Text blocks are collected into a single ``message`` item; structured blocks
    (function_call, computer_call) become top-level items.  Text that precedes a
    structured block is flushed as a separate message item so ordering is
    preserved.
    """
    if call_type_map is None:
        call_type_map = {}
    items: list[dict[str, Any]] = []
    pending_text: list[dict[str, Any]] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")

        if btype == "text":
            text = block.get("text", "")
            if text:
                pending_text.append({"type": "output_text", "text": text})

        elif btype == "function_call":
            # Flush pending text before emitting the structured item
            pending_text = _flush_pending_text(items, pending_text)
            items.append({
                "type": "function_call",
                "call_id": block.get("id", block.get("call_id", "")),
                "name": block.get("name", ""),
                "arguments": block.get("arguments", ""),
            })

        elif btype == "computer_call":
            pending_text = _flush_pending_text(items, pending_text)
            call_id = block.get("id", block.get("call_id", ""))
            if call_id:
                call_type_map[call_id] = "computer_call"
            actions = _normalize_actions(block)
            action_desc = json.dumps(actions)[:200] if actions else "details unavailable"
            items.append({
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": f"[computer action: {action_desc}]",
                }],
            })

        elif btype == "thinking":
            signature = _get_openai_reasoning_signature(block.get("thinkingSignature"))
            if signature is None:
                continue
            pending_text = _flush_pending_text(items, pending_text)
            summary = []
            if block.get("thinking"):
                summary = [{
                    "type": "summary_text",
                    "text": block.get("thinking", ""),
                }]
            items.append({
                "type": signature["type"],
                "id": signature["id"],
                "summary": summary,
            })

        else:
            # Unknown block type — keep as text
            pending_text.append({"type": "output_text", "text": str(block)})

    # Flush remaining text
    _flush_pending_text(items, pending_text)

    return items


def _unnest_tool_blocks(
    blocks: list[dict[str, Any]],
    call_type_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Convert tool_result content blocks into top-level Responses API output items.

    Uses ``call_type_map`` (call_id → "function_call" | "computer_call") to emit
    the correct output type:
    - computer_call → ``computer_call_output`` with text (screenshot from prior
      turn is no longer available; OpenAI validates image data so we can't use
      a placeholder PNG)
    - function_call → ``function_call_output`` with the original text content
    """
    if call_type_map is None:
        call_type_map = {}
    items: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "tool_result":
            call_id = block.get("tool_use_id", block.get("call_id", ""))
            if call_type_map.get(call_id) == "computer_call":
                items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": f"[computer result: {block.get('content', '')[:200]}]",
                    }],
                })
            else:
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": block.get("content", ""),
                })
        elif btype == "computer_call_output":
            items.append({
                "type": "computer_call_output",
                "call_id": block.get("tool_use_id", block.get("call_id", "")),
                "output": block.get("content", ""),
            })
        # else: skip non-tool blocks in tool messages
    return items


def _wrap_user_content(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap user content blocks into a Responses API message item."""
    converted: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            converted.append({"type": "input_text", "text": block.get("text", "")})
        elif btype in ("input_text", "input_image"):
            converted.append(block)  # already correct format
        else:
            # Unknown — convert to text
            converted.append({"type": "input_text", "text": str(block)})
    return {
        "type": "message",
        "role": "user",
        "content": converted or [{"type": "input_text", "text": ""}],
    }
