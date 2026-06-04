"""Transcript helpers — group CUA step output into assistant/tool content blocks.

Moved from openclaw_agent.py to break a cross-package import
(agent_loop.py was importing from ..openclaw_agent).

Reference:
  - openclaw_agent.py — original location of these functions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _extract_reasoning_text(item: dict[str, Any]) -> str:
    """Extract text content from a CUA reasoning item."""
    summary = item.get("summary", [])
    if isinstance(summary, list):
        parts = [
            block.get("text", "")
            for block in summary
            if isinstance(block, dict)
            and block.get("type") == "summary_text"
            and block.get("text")
        ]
        if parts:
            return "\n".join(parts)
    reasoning = item.get("reasoning", "")
    return reasoning if isinstance(reasoning, str) else ""


def _extract_reasoning_signature(item: dict[str, Any]) -> str | None:
    """Build an OpenClaw-style thinkingSignature for a Responses reasoning item."""
    signature = item.get("thinkingSignature")
    if isinstance(signature, str) and signature.strip():
        return signature

    item_id = item.get("id")
    item_type = item.get("type")
    if not isinstance(item_id, str) or not item_id.startswith("rs_"):
        return None
    if not isinstance(item_type, str) or not item_type.startswith("reasoning"):
        return None
    return json.dumps({"id": item_id, "type": item_type}, separators=(",", ":"))


def _find_latest_screenshot(trajectory_dir: Path | None) -> str:
    """Find the most recently saved screenshot_after.png in trajectory_dir.

    TrajectorySaverCallback saves one *_screenshot_after.png per computer action
    into trajectories/<trajectory_id>/turn_NNN/. The newest file corresponds to
    the action just completed.

    Returns the absolute path string, or "image:trajectory" if not found.
    """
    if not trajectory_dir or not trajectory_dir.exists():
        return "image:trajectory"
    screenshots = list(trajectory_dir.rglob("*_screenshot_after.png"))
    if not screenshots:
        return "image:trajectory"
    return str(max(screenshots, key=lambda p: p.stat().st_mtime))


def _thinking_block(thinking: str, item_id: Any, signature: str | None) -> dict[str, Any]:
    """Assemble a thinking content block, attaching id/signature only when present."""
    block: dict[str, Any] = {"type": "thinking", "thinking": thinking}
    if item_id:
        block["id"] = item_id
    if signature is not None:
        block["thinkingSignature"] = signature
    return block


def _message_assistant_blocks(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Assistant text/thinking blocks from a ``message`` output item."""
    content = item.get("content", [])
    role = item.get("role")
    if isinstance(content, str):
        if role == "assistant" and content:
            return [{"type": "text", "text": content}]
        return []
    if not isinstance(content, list):
        return []
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("text"):
            if role != "user":
                blocks.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "thinking":
            signature = block.get("thinkingSignature")
            if not block.get("thinking") and signature is None:
                continue
            blocks.append(_thinking_block(block.get("thinking", ""), block.get("id"), signature))
    return blocks


def _reasoning_assistant_block(item: dict[str, Any]) -> dict[str, Any] | None:
    """A thinking block from a ``reasoning`` output item, or None when empty."""
    thinking_text = _extract_reasoning_text(item)
    thinking_signature = _extract_reasoning_signature(item)
    if thinking_text or thinking_signature is not None:
        return _thinking_block(thinking_text, item.get("id"), thinking_signature)
    return None


def _function_call_block(item: dict[str, Any]) -> dict[str, Any]:
    """An assistant ``function_call`` block from a function_call output item."""
    return {
        "type": "function_call",
        "id": item.get("call_id", ""),
        "name": item.get("name", ""),
        "arguments": item.get("arguments", ""),
    }


def _computer_call_block(item: dict[str, Any]) -> dict[str, Any]:
    """An assistant ``computer_call`` block.

    Handles both ``action`` (computer-use-preview) and ``actions`` (GPT 5.4).
    """
    block: dict[str, Any] = {"type": "computer_call", "id": item.get("call_id", "")}
    if "actions" in item:
        block["actions"] = item["actions"]
    else:
        block["action"] = item.get("action", {})
    return block


def _function_call_output_result(item: dict[str, Any]) -> dict[str, Any]:
    """A ``tool_result`` block from a function_call_output item."""
    return {
        "type": "tool_result",
        "tool_use_id": item.get("call_id", ""),
        "content": item.get("output", ""),
    }


def _computer_call_output_result(
    item: dict[str, Any],
    trajectory_dir: Path | None,
) -> dict[str, Any]:
    """A ``tool_result`` block from a computer_call_output item.

    Screenshot outputs resolve to the latest trajectory image; other outputs
    are stringified and capped at 500 chars.
    """
    output = item.get("output", {})
    call_id = item.get("call_id", "")
    if isinstance(output, dict) and output.get("type") in ("input_image", "computer_screenshot"):
        content_str = _find_latest_screenshot(trajectory_dir)
    else:
        content_str = str(output)[:500]
    return {"type": "tool_result", "tool_use_id": call_id, "content": content_str}


def group_step_output(
    output_items: list[dict[str, Any]],
    trajectory_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group a step's output items into assistant content blocks and tool results.

    CUA SDK yields multiple output items per step (text, function_call,
    computer_call, their outputs). This function batches them into two lists:
    - assistant_content: text + function_call + computer_call blocks (one assistant turn)
    - tool_results: function_call_output + computer_call_output blocks (one tool turn)

    Args:
        output_items: The result["output"] list from a CUA agent step.
        trajectory_dir: Path to trajectory directory for screenshot resolution.

    Returns:
        (assistant_content, tool_results) tuple of content block lists.
    """
    assistant_content: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    for item in output_items:
        item_type = item.get("type")
        if item_type == "message":
            assistant_content.extend(_message_assistant_blocks(item))
        elif item_type == "reasoning":
            block = _reasoning_assistant_block(item)
            if block is not None:
                assistant_content.append(block)
        elif item_type == "function_call":
            assistant_content.append(_function_call_block(item))
        elif item_type == "computer_call":
            assistant_content.append(_computer_call_block(item))
        elif item_type == "function_call_output":
            tool_results.append(_function_call_output_result(item))
        elif item_type == "computer_call_output":
            tool_results.append(_computer_call_output_result(item, trajectory_dir))

    return assistant_content, tool_results
