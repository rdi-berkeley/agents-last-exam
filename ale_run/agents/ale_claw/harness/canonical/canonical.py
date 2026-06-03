"""Canonical internal message format and sanitize_items() pipeline.

Defines typed role-based messages and content blocks that serve as the single
internal representation for all pipeline passes (repair, sanitization,
compaction output, format conversion).

Design follows OpenClaw's AgentMessage pattern:
  - Role-based messages with typed content block arrays
  - stop_reason at the message level (needed by repair passes)
  - Content is always a list (string content normalized at ingestion)
  - actions is always a list (singular action normalized at ingestion)

Field conventions match OpenClaw / Anthropic:
  - ``id`` on FunctionCallBlock / ComputerCallBlock (not ``call_id``)
  - ``tool_use_id`` on ToolResultBlock
  - ``call_id`` is Responses API only — adapters map ``id`` → ``call_id``


Reference:
  - openclaw/src/agents/pi-embedded-runner/google.ts — sanitizeSessionHistory pipeline
  - openclaw/src/agents/session-transcript-repair.ts — repair passes on AgentMessage[]
  - openclaw/src/agents/transcript-policy.ts — TranscriptPolicy flags
  - openclaw/src/agents/pi-embedded-runner/thinking.ts — dropThinkingBlocks
  - openclaw/src/agents/pi-embedded-helpers/openai.ts — downgradeOpenAIReasoningBlocks
  - session.py:819-1080 — convert_to_responses_api_items (pattern reference)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .canonical_sanitize import (  # re-exported + used by sanitize_items
    SYNTHETIC_TOOL_RESULT_CONTENT,
    _SKIP_SYNTHESIS_STOP_REASONS,
    _get_openai_reasoning_signature,
    downgrade_openai_reasoning,
    drop_thinking_blocks,
    ensure_valid_ordering,
    repair_orphaned_pairs,
    sanitize_thinking_signatures,
)
from .canonical_types import (  # re-exported for back-compat
    CanonicalMessage,
    CompactionSummaryBlock,
    ComputerCallBlock,
    ContentBlock,
    FunctionCallBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
)

# Content-block + CanonicalMessage TypedDicts moved to canonical_types.py;
# re-exported via the import at the top of this module.

# ---------------------------------------------------------------------------
# Compaction summary preamble (used by adapters)
# ---------------------------------------------------------------------------

COMPACTION_PREAMBLE = (
    "## Prior Context (Compacted)\n"
    "The following is a summary of earlier conversation history that was "
    "compacted to save context space. Use this to maintain continuity.\n\n"
)

# ---------------------------------------------------------------------------
# TranscriptPolicy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptPolicy:
    """Per-provider flags controlling which sanitization passes run.

    Mirrors OpenClaw's TranscriptPolicy (transcript-policy.ts). Each flag
    gates a pure pass in :func:`sanitize_items`. All default to safe
    no-ops so adding a new provider only requires setting the flags it needs.

    Reference: openclaw/src/agents/transcript-policy.ts
    """

    sanitize_mode: Literal["full", "images-only"] = "images-only"
    """OpenClaw-style sanitize intent.

    ``images-only`` keeps non-image content untouched unless an explicit pass
    below is enabled. ``full`` indicates the provider needs broader transcript
    normalization. Current AgentHLE passes are still flag-driven, but runtime
    resolution should preserve this intent for parity with OpenClaw.
    """

    drop_thinking_blocks: bool = False
    """Strip type="thinking" content blocks from assistant messages.
    Needed for providers that reject replayed thinking blocks (e.g. Anthropic
    when signatures are invalid, GitHub Copilot Claude)."""

    sanitize_thinking_signatures: bool = False
    """Remove thinkingSignature fields from thinking blocks.
    Needed for cross-provider replay where signatures are provider-specific
    tamper-proof tokens that the target API will reject."""

    downgrade_openai_reasoning: bool = False
    """Drop orphaned OpenAI reasoning blocks (thinking blocks with a valid
    OpenAI reasoning signature but no following non-thinking content).
    Without this, o3/o4 model thinking blocks cause API rejection on replay."""

    repair_tool_use_result_pairing: bool = True
    """Repair orphaned tool call / result pairs. Enabled for all providers."""

    validate_anthropic_turns: bool = True
    """Ensure valid Anthropic turn structure (no trailing assistant message)."""


def get_transcript_policy(model: str | Any) -> TranscriptPolicy:
    """Resolve TranscriptPolicy from structured runtime metadata or a model string.

    Uses the same provider-detection pattern as
    :func:`thinking.resolve_thinking_params`.

    Args:
        model: litellm model identifier (e.g. "anthropic/claude-sonnet-4-20250514",
            "openai/gpt-5.4").

    Reference: openclaw/src/agents/transcript-policy.ts:resolveTranscriptPolicy
    """
    from ..inference.model_config import resolve_model

    runtime = resolve_model(model)
    model_lower = runtime.model.lower()

    if runtime.provider == "anthropic" or "claude" in model_lower:
        return TranscriptPolicy(
            sanitize_mode="full",
            drop_thinking_blocks=True,
            sanitize_thinking_signatures=False,
            downgrade_openai_reasoning=False,
            repair_tool_use_result_pairing=True,
            validate_anthropic_turns=True,
        )

    if runtime.provider == "openai":
        return TranscriptPolicy(
            sanitize_mode="images-only",
            drop_thinking_blocks=False,
            sanitize_thinking_signatures=False,
            downgrade_openai_reasoning=True,
            repair_tool_use_result_pairing=True,
            validate_anthropic_turns=False,
        )

    if runtime.provider in {"google", "vertex"}:
        return TranscriptPolicy(
            sanitize_mode="full",
            drop_thinking_blocks=False,
            sanitize_thinking_signatures=True,
            downgrade_openai_reasoning=False,
            repair_tool_use_result_pairing=True,
            validate_anthropic_turns=False,
        )

    return TranscriptPolicy()


# ---------------------------------------------------------------------------
# Ingestion: untyped dicts → canonical
# ---------------------------------------------------------------------------


def normalize_to_canonical(
    messages: list[dict[str, Any]],
) -> list[CanonicalMessage]:
    """Convert untyped dicts to typed canonical messages.

    Handles two input formats:
      1. **Role-based messages** (``{role, content}``): from compaction, session
         replay, and Anthropic completion format.
      2. **Flat Responses API items** (``{type: "function_call", call_id, ...}``):
         from the OpenAI Responses API loop's items list.

    Flat items are detected by having a ``type`` field without a ``role`` field
    (or ``type == "message"``). They are converted to canonical messages and
    grouped: consecutive assistant-role blocks merge into one message, and
    consecutive tool-role blocks merge into one message.

    Normalizes:
      - String content → ``[TextBlock]``
      - ``action: {…}`` → ``actions: [{…}]``
      - Preserves ``stop_reason`` on messages that have it
      - Strips ``acknowledged_safety_checks`` from ``computer_call_output``
    """
    result: list[CanonicalMessage] = []
    for msg in messages:
        if _is_flat_responses_item(msg):
            _ingest_flat_item(msg, result)
        else:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            blocks = _normalize_content(content, role)
            canonical: CanonicalMessage = {"role": role, "content": blocks}
            stop_reason = msg.get("stop_reason")
            if stop_reason:
                canonical["stop_reason"] = stop_reason
            result.append(canonical)
    return result


def _is_flat_responses_item(msg: dict[str, Any]) -> bool:
    """Check if a dict is a flat Responses API item (not a role-based message).

    Flat items have a ``type`` field and either no ``role`` field or
    ``type == "message"`` (Responses API wraps role-based content in a
    message item).
    """
    if "type" not in msg:
        return False
    msg_type = msg["type"]
    # {type: "message", role: ..., content: [...]} — Responses API message wrapper
    if msg_type == "message":
        return True
    # Has type but no role — flat item (function_call, computer_call, etc.)
    if "role" not in msg:
        return True
    return False


def _ingest_flat_item(
    item: dict[str, Any], result: list[CanonicalMessage]
) -> None:
    """Convert a flat Responses API item to canonical and append/merge into result.

    Groups consecutive same-role blocks into one message.
    """
    item_type = item.get("type", "")

    if item_type == "message":
        # {type: "message", role: "user"|"assistant", content: [{type: "input_text"|"output_text", ...}]}
        role = item.get("role", "user")
        content = item.get("content", [])
        blocks = _normalize_responses_content(content, role)
        if blocks:
            _append_or_merge(result, role, blocks)
        return

    if item_type == "function_call":
        block = FunctionCallBlock(
            type="function_call",
            id=item.get("call_id", item.get("id", "")),
            name=item.get("name", ""),
            arguments=item.get("arguments", ""),
        )
        _append_or_merge(result, "assistant", [block])
        return

    if item_type == "computer_call":
        block = ComputerCallBlock(
            type="computer_call",
            id=item.get("call_id", item.get("id", "")),
            actions=_normalize_actions(item),
        )
        _append_or_merge(result, "assistant", [block])
        return

    if item_type == "function_call_output":
        block = ToolResultBlock(
            type="tool_result",
            tool_use_id=item.get("call_id", ""),
            content=item.get("output", ""),
        )
        _append_or_merge(result, "tool", [block])
        return

    if item_type == "computer_call_output":
        # Strip acknowledged_safety_checks — not part of canonical format
        output = item.get("output", item.get("content", ""))
        if isinstance(output, dict):
            output = json.dumps(output)
        elif not isinstance(output, str):
            output = str(output)
        block = ToolResultBlock(
            type="tool_result",
            tool_use_id=item.get("call_id", ""),
            content=output,
        )
        _append_or_merge(result, "tool", [block])
        return

    if item_type == "reasoning":
        # OpenAI reasoning items — convert to ThinkingBlock
        summary = item.get("summary", [])
        text = ""
        if isinstance(summary, list):
            text = " ".join(
                s.get("text", "") for s in summary if isinstance(s, dict)
            )
        elif isinstance(summary, str):
            text = summary
        block = ThinkingBlock(type="thinking", thinking=text)
        _append_or_merge(result, "assistant", [block])
        return

    # Unknown flat item type — preserve as text in a user message
    text = json.dumps(item)[:500]
    block = TextBlock(type="text", text=f"[{item_type}: {text}]")
    _append_or_merge(result, "user", [block])


def _normalize_responses_content(
    content: Any, role: str
) -> list[ContentBlock]:
    """Normalize Responses API message content to canonical blocks.

    Handles Responses API content types: ``input_text``, ``output_text``,
    ``input_image``, ``computer_screenshot``, ``refusal``, ``summary_text``.
    """
    if isinstance(content, str):
        return [TextBlock(type="text", text=content)]

    if not isinstance(content, list):
        return [TextBlock(type="text", text=str(content))]

    blocks: list[ContentBlock] = []
    for block in content:
        if not isinstance(block, dict):
            blocks.append(TextBlock(type="text", text=str(block)))
            continue

        btype = block.get("type", "")
        if btype in ("input_text", "output_text", "text", "summary_text"):
            blocks.append(TextBlock(type="text", text=block.get("text", "")))
        elif btype == "refusal":
            blocks.append(TextBlock(type="text", text=f"[refusal: {block.get('refusal', '')}]"))
        elif btype in ("input_image", "computer_screenshot"):
            # Image blocks in Responses API — preserve as text reference
            # (actual images are handled by the screenshot pipeline, not replayed)
            blocks.append(TextBlock(type="text", text=f"[{btype}]"))
        else:
            # Unknown Responses API content type — preserve as text
            blocks.append(TextBlock(type="text", text=block.get("text", str(block))))
    return blocks


def _append_or_merge(
    result: list[CanonicalMessage], role: str, blocks: list[ContentBlock]
) -> None:
    """Append blocks to the last message if same role, otherwise create new message."""
    if result and result[-1]["role"] == role:
        result[-1]["content"].extend(blocks)
    else:
        result.append(CanonicalMessage(role=role, content=list(blocks)))


def _normalize_content(
    content: Any, role: str
) -> list[ContentBlock]:
    """Normalize a message's content field to a list of typed ContentBlocks."""
    if isinstance(content, str):
        return [TextBlock(type="text", text=content)]

    if not isinstance(content, list):
        return [TextBlock(type="text", text=str(content))]

    blocks: list[ContentBlock] = []
    for block in content:
        if not isinstance(block, dict):
            blocks.append(TextBlock(type="text", text=str(block)))
            continue

        btype = block.get("type", "")

        if btype == "text":
            blocks.append(TextBlock(type="text", text=block.get("text", "")))

        elif btype == "function_call":
            blocks.append(FunctionCallBlock(
                type="function_call",
                id=block.get("id", block.get("call_id", "")),
                name=block.get("name", ""),
                arguments=block.get("arguments", ""),
            ))

        elif btype == "computer_call":
            blocks.append(ComputerCallBlock(
                type="computer_call",
                id=block.get("id", block.get("call_id", "")),
                actions=_normalize_actions(block),
            ))

        elif btype == "tool_result":
            tb = ToolResultBlock(
                type="tool_result",
                tool_use_id=block.get("tool_use_id", block.get("call_id", "")),
                content=block.get("content", ""),
            )
            if block.get("is_error"):
                tb["is_error"] = True
            blocks.append(tb)

        elif btype == "computer_call_output":
            # Stored as a tool_result-like block in some transcript paths.
            # Normalize to ToolResultBlock with the original content.
            tb = ToolResultBlock(
                type="tool_result",
                tool_use_id=block.get("call_id", block.get("tool_use_id", "")),
                content=block.get("output", block.get("content", "")),
            )
            blocks.append(tb)

        elif btype == "thinking":
            tb_thinking = ThinkingBlock(
                type="thinking",
                thinking=block.get("thinking", ""),
            )
            sig = block.get("thinkingSignature")
            if sig:
                tb_thinking["thinkingSignature"] = sig
            blocks.append(tb_thinking)

        elif btype == "compaction_summary":
            blocks.append(CompactionSummaryBlock(
                type="compaction_summary",
                text=block.get("text", ""),
            ))

        elif btype in ("image_url", "image", "input_image", "computer_screenshot"):
            blocks.append(TextBlock(type="text", text=f"[{btype}]"))

        else:
            # Unknown block type — preserve as text
            text = block.get("text", block.get("content", str(block)))
            blocks.append(TextBlock(type="text", text=str(text)))

    return blocks


def _normalize_actions(block: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize action/actions to always be a list."""
    actions = block.get("actions")
    if isinstance(actions, list):
        return actions
    action = block.get("action")
    if action is not None:
        return [action]
    return []


# ---------------------------------------------------------------------------
# Adapter: canonical → OpenAI Responses API flat items
# ---------------------------------------------------------------------------


def canonical_to_responses_api(
    messages: list[CanonicalMessage],
) -> list[dict[str, Any]]:
    """Convert canonical messages to OpenAI Responses API flat items.

    Each canonical message is unnested into one or more flat items:
      - User TextBlock → ``{type: "message", role: "user", …}``
      - Assistant TextBlock → ``{type: "message", role: "assistant", …}``
      - FunctionCallBlock → ``{type: "function_call", call_id: …}``
      - ComputerCallBlock → ``{type: "computer_call", call_id: …}``
      - ToolResultBlock → ``function_call_output`` or ``computer_call_output``
      - CompactionSummaryBlock → user message with preamble
      - ThinkingBlock → skipped (not representable in Responses API items)
    """
    items: list[dict[str, Any]] = []
    # Track call types so tool results emit the correct output type
    call_type_map: dict[str, str] = {}

    for msg in messages:
        role = msg["role"]
        for block in msg["content"]:
            btype = block["type"]

            if btype == "compaction_summary":
                items.append({
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": COMPACTION_PREAMBLE + block["text"],
                    }],
                })

            elif btype == "text":
                if role == "assistant":
                    items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": block["text"]}],
                    })
                else:
                    items.append({
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": block["text"]}],
                    })

            elif btype == "function_call":
                call_id = block["id"]
                call_type_map[call_id] = "function_call"
                items.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": block["name"],
                    "arguments": block["arguments"],
                })

            elif btype == "computer_call":
                call_id = block["id"]
                call_type_map[call_id] = "computer_call"
                # Compacted/replayed computer_call blocks no longer have
                # the original screenshot. Convert to text — OpenAI validates
                # image data in computer_call_output and rejects placeholders.
                # Matches _normalize_messages_for_gpt54 behavior in openai.py.
                actions = block["actions"]
                action_desc = json.dumps(actions)[:200] if actions else "details unavailable"
                text_type = "output_text" if role == "assistant" else "input_text"
                items.append({
                    "type": "message",
                    "role": role if role != "tool" else "user",
                    "content": [{
                        "type": text_type,
                        "text": f"[computer action: {action_desc}]",
                    }],
                })

            elif btype == "tool_result":
                call_id = block["tool_use_id"]
                if call_type_map.get(call_id) == "computer_call":
                    # Computer call result after compaction — screenshot is
                    # gone. Convert to text (matching computer_call branch
                    # above). No call/result pairing issue since both are text.
                    items.append({
                        "type": "message",
                        "role": "user",
                        "content": [{
                            "type": "input_text",
                            "text": f"[computer result: {block['content'][:200]}]",
                        }],
                    })
                else:
                    items.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": block["content"],
                    })

            elif btype == "thinking":
                signature = _get_openai_reasoning_signature(
                    block.get("thinkingSignature")
                )
                if signature is None:
                    continue
                summary = []
                if block.get("thinking"):
                    summary = [{
                        "type": "summary_text",
                        "text": block["thinking"],
                    }]
                items.append({
                    "type": signature["type"],
                    "id": signature["id"],
                    "summary": summary,
                })

    return _ensure_tool_adjacency(items)


def _ensure_tool_adjacency(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder items so each tool call is immediately followed by its output.

    Defers non-output items that appear between a call and its matching output,
    then flushes them after the output. Matches the algorithm in session.py.
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
            if not pending_call_ids:
                result.extend(deferred)
                deferred = []
        elif pending_call_ids:
            deferred.append(item)
        else:
            result.append(item)

    result.extend(deferred)
    return result


# ---------------------------------------------------------------------------
# Sanitize pipeline
# ---------------------------------------------------------------------------
# The individual sanitize passes (repair_orphaned_pairs, ensure_valid_ordering,
# drop_thinking_blocks, sanitize_thinking_signatures, downgrade_openai_reasoning,
# + helpers and the SYNTHETIC/_SKIP constants) moved to canonical_sanitize.py and
# are imported back at the top of this module. sanitize_items (below) orchestrates
# them with the policy + format adapters that stay here.


def sanitize_items(
    messages: list[CanonicalMessage],
    target: Literal["openai-responses", "anthropic"] | None = None,
    *,
    model: str | Any | None = None,
    policy: TranscriptPolicy | None = None,
) -> list[dict[str, Any]]:
    """Convert canonical messages to provider-specific format.

    Modeled on OpenClaw's sanitizeSessionHistory — linear pipeline of pure passes:
      1. repair_orphaned_pairs — fix split call/result pairs
      2. drop_thinking_blocks — strip thinking content (if policy flag set)
      3. sanitize_thinking_signatures — remove signatures (if policy flag set)
      4. downgrade_openai_reasoning — drop orphaned reasoning (if policy flag set)
      5. ensure_valid_ordering — no trailing assistant message
      6. Format conversion — canonical → target format

    Args:
        messages: Canonical messages (from normalize_to_canonical).
        target: Provider format to convert to. If omitted, resolves from the
            model registry.
        model: Optional live model string. When provided and ``policy`` is not,
            policy resolves via :func:`get_transcript_policy` instead of using
            target-only defaults.
        policy: TranscriptPolicy controlling which passes run. If None, a
            model-resolved or target-default policy is used.

    Returns:
        Provider-specific messages/items ready for the API.
    """
    if target is None:
        if model is None:
            raise ValueError("sanitize_items() requires target or model")
        from ..inference.model_config import resolve_model

        target = resolve_model(model).adapter_target

    if policy is None and model is not None:
        policy = get_transcript_policy(model)

    if policy is None:
        if target == "anthropic":
            policy = TranscriptPolicy(
                sanitize_mode="full",
                drop_thinking_blocks=True,
                validate_anthropic_turns=True,
            )
        elif target == "openai-responses":
            policy = TranscriptPolicy(
                sanitize_mode="images-only",
                downgrade_openai_reasoning=True,
                validate_anthropic_turns=False,
            )
        else:
            policy = TranscriptPolicy()

    if policy.repair_tool_use_result_pairing:
        messages = repair_orphaned_pairs(messages)

    if policy.drop_thinking_blocks:
        messages = drop_thinking_blocks(messages)

    if policy.sanitize_thinking_signatures:
        messages = sanitize_thinking_signatures(messages)

    if policy.downgrade_openai_reasoning:
        messages = downgrade_openai_reasoning(messages)

    if policy.validate_anthropic_turns:
        messages = ensure_valid_ordering(messages)

    if target == "openai-responses":
        return canonical_to_responses_api(messages)
    elif target == "anthropic":
        return canonical_to_anthropic_messages(messages)
    else:
        raise ValueError(f"Unknown adapter target: {target}")


# ---------------------------------------------------------------------------
# Adapter: canonical → Anthropic completion messages
# ---------------------------------------------------------------------------


def canonical_to_anthropic_messages(
    messages: list[CanonicalMessage],
) -> list[dict[str, Any]]:
    """Convert canonical messages to Anthropic completion format.

    Groups content blocks by role into role-based messages:
      - FunctionCallBlock → ``{type: "tool_use", id, name, input}``
      - ComputerCallBlock → ``{type: "tool_use", id, name: "computer", input}``
      - ToolResultBlock → ``{role: "tool", content: [{type: "tool_result", …}]}``
      - CompactionSummaryBlock → user text with preamble
      - ThinkingBlock → ``{type: "thinking", thinking, signature}``

    Consecutive blocks within the same message are grouped. Tool messages
    break the grouping to ensure correct Anthropic turn structure.
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]

        if role == "tool":
            # Tool messages: each ToolResultBlock becomes an Anthropic tool_result
            tool_content: list[dict[str, Any]] = []
            for block in msg["content"]:
                if block["type"] == "tool_result":
                    tr: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": block["tool_use_id"],
                        "content": block["content"],
                    }
                    if block.get("is_error"):
                        tr["is_error"] = True
                    tool_content.append(tr)
            if tool_content:
                result.append({"role": "user", "content": tool_content})

        elif role == "assistant":
            # Assistant messages: map blocks to Anthropic content types
            content: list[dict[str, Any]] = []
            for block in msg["content"]:
                btype = block["type"]
                if btype == "text":
                    content.append({"type": "text", "text": block["text"]})
                elif btype == "function_call":
                    raw_args = block["arguments"]
                    if not raw_args:
                        tool_input: dict[str, Any] = {}
                    else:
                        try:
                            tool_input = json.loads(raw_args)
                        except (json.JSONDecodeError, TypeError):
                            # Defense-in-depth: a function_call may carry a
                            # truncated/malformed arguments string from a
                            # mid-stream upstream-provider drop. The write-side
                            # sanitizer in OpenClawComputerAgent should have
                            # already rewritten these, but any older transcript
                            # entries (or future code paths that bypass the
                            # sanitizer) would crash compaction's history
                            # rebuild here. Fall back to a marked placeholder
                            # so re-serialization stays self-consistent.
                            partial = raw_args[:200] if isinstance(raw_args, str) else repr(raw_args)[:200]
                            tool_input = {
                                "_truncated_by_upstream": True,
                                "_partial_args": partial,
                                "_original_length": len(raw_args) if isinstance(raw_args, str) else 0,
                            }
                    content.append({
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": tool_input,
                    })
                elif btype == "computer_call":
                    content.append({
                        "type": "tool_use",
                        "id": block["id"],
                        "name": "computer",
                        "input": {"actions": block["actions"]},
                    })
                elif btype == "thinking":
                    tb: dict[str, Any] = {
                        "type": "thinking",
                        "thinking": block["thinking"],
                    }
                    if block.get("thinkingSignature"):
                        tb["signature"] = block["thinkingSignature"]
                    content.append(tb)
                elif btype == "compaction_summary":
                    content.append({"type": "text", "text": COMPACTION_PREAMBLE + block["text"]})
            if content:
                result.append({"role": "assistant", "content": content})

        else:
            # User / system messages
            content_out: list[dict[str, Any]] = []
            for block in msg["content"]:
                btype = block["type"]
                if btype == "text":
                    content_out.append({"type": "text", "text": block["text"]})
                elif btype == "compaction_summary":
                    content_out.append({
                        "type": "text",
                        "text": COMPACTION_PREAMBLE + block["text"],
                    })
                else:
                    content_out.append({"type": "text", "text": str(block)})
            if content_out:
                result.append({"role": role, "content": content_out})

    return result
