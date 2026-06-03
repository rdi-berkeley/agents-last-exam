"""Stateless helpers for the OpenClaw agent loop.

Screenshot sanitization, image-block rewriting, DONE-signal detection, and
transcript->compaction message extraction — the free functions
``OpenClawComputerAgent`` calls that don't touch instance state. Split out of
agent_loop.py to keep the orchestrator class focused.
"""

from __future__ import annotations

import base64 as _base64
import re
from typing import TYPE_CHECKING, Any, Dict, List

from .inference._message_shapes import _image_url_block
from .utils.image_sanitization import (
    DEFAULT_LIMITS as _IMAGE_DEFAULT_LIMITS,
    sanitize_raw_image_bytes as _sanitize_raw_image_bytes,
)

if TYPE_CHECKING:
    from .session import SessionManager


def _maybe_sanitize_screenshot(b64: str) -> tuple[str, str]:
    """Resize/transcode a base64-encoded screenshot per OpenClaw image limits.

    Returns ``(out_b64, out_mime)``. On any failure to sanitize (decode error,
    exhausted resize grid), falls back to the original PNG to keep the run
    alive — the sanitizer is defensive, not strict. Mirrors the per-tool
    wrap pattern used by ReadFileTool / AnalyzeImageTool.
    """
    try:
        raw = _base64.b64decode(b64, validate=False)
    except Exception:  # noqa: BLE001
        return b64, "image/png"
    sanitized = _sanitize_raw_image_bytes(
        raw, "image/png", label="screenshot", limits=_IMAGE_DEFAULT_LIMITS
    )
    if isinstance(sanitized, str):
        # Placeholder string — sanitizer gave up. Pass the original through.
        return b64, "image/png"
    out_bytes, out_mime = sanitized
    if out_bytes is raw:
        return b64, out_mime
    return _base64.b64encode(out_bytes).decode("ascii"), out_mime


def _rewrite_input_image_to_image_url(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rewrite ``{type: input_image, image_url: <str>}`` content blocks into
    ``{type: image_url, image_url: {url: <str>}}`` in user-message content lists.

    Skips ``computer_call_output.output`` blocks: the OpenAI Responses API
    contract for the native computer_call flow requires ``input_image`` there.
    The harness doesn't currently target ``computer-use-preview`` models, but
    leaving that block untouched keeps the API contract intact if it ever runs.
    """
    for item in items:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "input_image"
                and isinstance(block.get("image_url"), str)
            ):
                new_content.append(_image_url_block(block["image_url"]))
            else:
                new_content.append(block)
        item["content"] = new_content
    return items


# Matches the convention documented in the agent's AGENTS.md:
# "When you have fully completed the task, output **DONE** on its own line."
# Naive `"DONE" in text` produced false positives on incidental mentions
# like `"JOB DONE"` in log/output discussion, ending runs mid-task.
# Mirrors `_TEXT_DONE_RE` in subagent_gui_protocol.py (the GUI subagent's
# strict matcher) but tolerates surrounding asterisks for `**DONE**`.
_DONE_LINE_RE = re.compile(
    r"^\s*\**DONE\**(?:[:\s]+(.*))?$", re.IGNORECASE
)


def has_done_signal(output: List[Dict[str, Any]]) -> bool:
    """Return True if the assistant output contains the DONE completion signal.

    Single source of truth for OpenClaw's task-completion detection. Used by
    both the inner agent loop (to break the generator) and the outer
    OpenClawAgent.perform_task consumer (to classify the run as completed).

    Matches DONE only when it appears as its own line (optionally bolded or
    followed by `: <summary>`) — incidental mentions like "JOB DONE" inside
    a sentence do NOT terminate the run.
    """
    def _line_matches(text: str) -> bool:
        return any(_DONE_LINE_RE.match(line) for line in text.splitlines())

    for item in output:
        if item.get("type") != "message":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            if _line_matches(content):
                return True
        elif isinstance(content, list):
            for block in content:
                text = block.get("text", "") if isinstance(block, dict) else str(block)
                if text and _line_matches(text):
                    return True
    return False


_IMAGE_BLOCK_TYPES = frozenset({"image_url", "image", "input_image", "computer_screenshot"})


def _strip_images_from_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace image content blocks with text placeholders.

    After compaction, images are normalized to ``[image_url]`` text by
    ``_normalize_content``.  Stripping them here ensures that the compaction
    pipeline's token budget (``estimate_message_tokens``) matches the
    post-compaction reality — otherwise each image counts as 1200 tokens in
    the budget but becomes ~3 tokens after normalization.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_blocks: list[Any] = []
        changed = False
        for block in content:
            if isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES:
                new_blocks.append({"type": "text", "text": f"[{block['type']}]"})
                changed = True
            else:
                new_blocks.append(block)
        if changed:
            replaced = dict(msg)
            replaced["content"] = new_blocks
            out.append(replaced)
        else:
            out.append(msg)
    return out


def _extract_messages_for_compaction(session_mgr: SessionManager) -> list[dict[str, Any]]:
    """Extract message entries from the transcript as dicts for compaction.

    Converts TranscriptEntry objects into the {role, content, stop_reason} format
    expected by the compaction pipeline. Propagates stop_reason from transcript
    entries so repair_tool_use_result_pairing() can skip synthesis for
    error/aborted turns.

    Image content blocks (base64 screenshots) are replaced with lightweight
    text placeholders so the compaction budget matches the post-normalization
    token count.
    """
    history = session_mgr.load_history()
    messages: list[dict[str, Any]] = []
    for entry in history:
        if entry.type != "message":
            continue
        msg_data = entry.data.get("message", {})
        msg: dict[str, Any] = {
            "role": msg_data.get("role", "unknown"),
            "content": msg_data.get("content", ""),
        }
        stop_reason = msg_data.get("stop_reason")
        if stop_reason:
            msg["stop_reason"] = stop_reason
        messages.append(msg)
    return _strip_images_from_messages(messages)
