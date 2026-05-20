"""Transcript helpers — group CUA step output into assistant/tool content blocks.

Moved from openclaw_agent.py (US-OC-028) to break a cross-package import
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
            content = item.get("content", [])
            role = item.get("role")
            if isinstance(content, str):
                if role == "assistant" and content:
                    assistant_content.append({"type": "text", "text": content})
                continue
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("text"):
                    if role != "user":
                        assistant_content.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "thinking":
                    signature = block.get("thinkingSignature")
                    if not block.get("thinking") and signature is None:
                        continue
                    thinking_block = {
                        "type": "thinking",
                        "thinking": block.get("thinking", ""),
                    }
                    if block.get("id"):
                        thinking_block["id"] = block["id"]
                    if signature is not None:
                        thinking_block["thinkingSignature"] = signature
                    assistant_content.append(thinking_block)
        elif item_type == "reasoning":
            thinking_text = _extract_reasoning_text(item)
            thinking_signature = _extract_reasoning_signature(item)
            if thinking_text or thinking_signature is not None:
                thinking_block = {"type": "thinking", "thinking": thinking_text}
                if item.get("id"):
                    thinking_block["id"] = item["id"]
                if thinking_signature is not None:
                    thinking_block["thinkingSignature"] = thinking_signature
                assistant_content.append(thinking_block)
        elif item_type == "function_call":
            assistant_content.append({
                "type": "function_call",
                "id": item.get("call_id", ""),
                "name": item.get("name", ""),
                "arguments": item.get("arguments", ""),
            })
        elif item_type == "computer_call":
            # Handle both "action" (computer-use-preview) and "actions" (GPT 5.4)
            block: dict[str, Any] = {
                "type": "computer_call",
                "id": item.get("call_id", ""),
            }
            if "actions" in item:
                block["actions"] = item["actions"]
            else:
                block["action"] = item.get("action", {})
            assistant_content.append(block)
        elif item_type == "function_call_output":
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            })
        elif item_type == "computer_call_output":
            output = item.get("output", {})
            call_id = item.get("call_id", "")
            if isinstance(output, dict) and output.get("type") in ("input_image", "computer_screenshot"):
                content_str = _find_latest_screenshot(trajectory_dir)
            else:
                content_str = str(output)[:500]
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": content_str,
            })

    return assistant_content, tool_results
