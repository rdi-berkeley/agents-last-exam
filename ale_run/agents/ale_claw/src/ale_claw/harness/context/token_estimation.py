"""Shared token-estimation primitives (leaf module).

Split out of context.py so both the context-overflow core and the compaction
pipeline can estimate tokens without importing each other. No harness deps.
"""

from __future__ import annotations

import json
import re
from typing import Any

SAFETY_MARGIN = 1.2
"""Multiply raw token estimate by this factor to absorb tokenizer variance."""

FIXED_IMAGE_TOKENS = 1200
"""Standard API cost for a 1024x768 screenshot (Anthropic billing)."""

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
