"""GUI subagent — unified computer tool protocol (schema, parsing, execution).

Uses the same ``computer`` function tool schema as the main agent
(unified.py:_build_computer_tool_schema), with ``done`` added for relay loop
termination.  Dispatches to cuaComputerHandler via
``getattr(handler, action_type)(**params)`` — identical to agent.py:798-828.

US-SUB-010: Unified Computer Tool Schema.
Replaces the custom gui_action schema (US-SUB-003) with the main agent's
computer tool schema for full schema parity.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Action vocabulary — mirrors agent.py:798-810
# ---------------------------------------------------------------------------

_ACTION_PARAM_MAP: dict[str, list[str]] = {
    "click": ["x", "y", "button"],
    "double_click": ["x", "y"],
    "right_click": ["x", "y"],
    "type": ["text"],
    "keypress": ["keys"],
    "scroll": ["x", "y", "scroll_x", "scroll_y"],
    "move": ["x", "y"],
    "drag": ["start_x", "start_y", "end_x", "end_y"],
    "screenshot": [],
    "wait": ["ms"],
    "done": ["summary"],
}

_VALID_ACTIONS = frozenset(_ACTION_PARAM_MAP)

_ACTION_ALIASES: dict[str, str] = {
    "hotkey": "keypress",
    "terminate": "done",
}

_INT_FIELDS = frozenset({
    "x", "y", "scroll_x", "scroll_y",
    "start_x", "start_y", "end_x", "end_y", "ms",
})


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------


def computer_tool_schema(
    width: int = 1024,
    height: int = 768,
    environment: str = "windows",
) -> dict[str, Any]:
    """OpenAI function-calling schema for the ``computer`` tool.

    Matches unified.py:_build_computer_tool_schema with ``done`` added for
    relay loop termination (replaces ``terminate``).
    """
    return {
        "type": "function",
        "function": {
            "name": "computer",
            "description": (
                "Use a mouse and keyboard to interact with a computer, "
                "and take screenshots.\n"
                f"Screen resolution: {width}x{height} pixels.\n"
                f"Environment: {environment}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "description": "The action to perform.",
                        "type": "string",
                        "enum": [
                            "click",
                            "double_click",
                            "right_click",
                            "type",
                            "keypress",
                            "scroll",
                            "move",
                            "drag",
                            "screenshot",
                            "wait",
                            "done",
                        ],
                    },
                    "x": {
                        "description": "X coordinate for click/move/scroll actions.",
                        "type": "integer",
                    },
                    "y": {
                        "description": "Y coordinate for click/move/scroll actions.",
                        "type": "integer",
                    },
                    "text": {
                        "description": "Text to type (for action=type).",
                        "type": "string",
                    },
                    "keys": {
                        "description": (
                            "Keys to press (for action=keypress). "
                            "Example: ['ctrl', 'c']"
                        ),
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "scroll_x": {
                        "description": (
                            "Horizontal scroll amount. "
                            "Positive=right, negative=left."
                        ),
                        "type": "integer",
                    },
                    "scroll_y": {
                        "description": (
                            "Vertical scroll amount. "
                            "Positive=down, negative=up."
                        ),
                        "type": "integer",
                    },
                    "button": {
                        "description": "Mouse button for click action.",
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                    },
                    "start_x": {
                        "description": "Starting X coordinate for drag action.",
                        "type": "integer",
                    },
                    "start_y": {
                        "description": "Starting Y coordinate for drag action.",
                        "type": "integer",
                    },
                    "end_x": {
                        "description": "Ending X coordinate for drag action.",
                        "type": "integer",
                    },
                    "end_y": {
                        "description": "Ending Y coordinate for drag action.",
                        "type": "integer",
                    },
                    "ms": {
                        "description": "Wait duration in milliseconds.",
                        "type": "integer",
                    },
                    "summary": {
                        "description": (
                            "Summary of what was accomplished (for action=done)."
                        ),
                        "type": "string",
                    },
                },
                "required": ["action"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_action(action: dict[str, Any]) -> None:
    """Validate an action dict; raise ValueError if malformed."""
    action_type = action.get("action")
    if action_type not in _VALID_ACTIONS:
        raise ValueError(f"Unknown action type: {action_type!r}")

    if action_type in ("click", "double_click", "right_click", "move"):
        x, y = action.get("x", 0), action.get("y", 0)
        if x < 0 or y < 0:
            raise ValueError(
                f"{action_type} coordinates must be non-negative: ({x}, {y})"
            )

    if action_type == "type":
        if not action.get("text"):
            raise ValueError("Type action text must be non-empty")

    if action_type == "keypress":
        if not action.get("keys"):
            raise ValueError("Keypress action keys must be non-empty")

    if action_type == "drag":
        for coord in ("start_x", "start_y", "end_x", "end_y"):
            if action.get(coord, 0) < 0:
                raise ValueError(
                    f"Drag coordinate {coord} must be non-negative: "
                    f"{action.get(coord)}"
                )

    if action_type == "wait":
        ms = action.get("ms", 1000)
        if ms <= 0:
            raise ValueError(f"Wait ms must be positive: {ms}")


# ---------------------------------------------------------------------------
# Normalization — single entry point for all parse paths
# ---------------------------------------------------------------------------


def _normalize_action_dict(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a raw action dict from any parse path into canonical format.

    Accepts ``type`` (computer_call), ``action`` (function_call), or
    ``action_type`` (legacy gui_action) as the discriminator field.
    """
    action_type = raw.get("type") or raw.get("action") or raw.get("action_type")
    if action_type is None:
        return None
    action_type = str(action_type)
    action_type = _ACTION_ALIASES.get(action_type, action_type)
    if action_type not in _VALID_ACTIONS:
        return None

    result: dict[str, Any] = {"action": action_type}
    relevant_params = _ACTION_PARAM_MAP.get(action_type, [])

    for key in relevant_params:
        val = raw.get(key)
        if val is not None:
            result[key] = int(val) if key in _INT_FIELDS else val

    # Drag: convert computer_call ``path`` to start/end coordinates.
    if action_type == "drag" and "start_x" not in result:
        path = raw.get("path")
        if isinstance(path, list) and len(path) >= 2:
            start, end = path[0], path[-1]
            result["start_x"] = int(start.get("x", 0))
            result["start_y"] = int(start.get("y", 0))
            result["end_x"] = int(end.get("x", 0))
            result["end_y"] = int(end.get("y", 0))

    # Ensure ``keys`` is always a list for keypress.
    if action_type == "keypress":
        keys = result.get("keys")
        if isinstance(keys, str):
            result["keys"] = keys.replace("-", "+").split("+")

    return result


# ---------------------------------------------------------------------------
# parse_gui_response — three parse paths
# ---------------------------------------------------------------------------


def parse_gui_response(response: Any) -> list[dict[str, Any]]:
    """Parse a vision model response into one or more action dicts.

    Tries three paths in order:
      (a) OpenAI ``computer_call`` output items (native GPT-5.4 format)
      (b) function_call with ``computer`` tool (non-CU vision models)
      (c) structured text with action keywords (last resort)

    Returns ALL parsed actions from the first path that produces results.
    Raises ValueError if none of the paths can produce any action.
    """
    actions = (
        _try_parse_computer_call(response)
        or _try_parse_function_call(response)
        or _try_parse_text(response)
        or []
    )
    if not actions:
        raise ValueError(
            f"Could not parse GUI action from response: {response!r}"
        )
    for action in actions:
        validate_action(action)
    return actions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_dict(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            return None
    return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------
# Path (a): computer_call
# ---------------------------------------------------------------------------


def _try_parse_computer_call(response: Any) -> list[dict[str, Any]] | None:
    """Parse OpenAI Responses API computer_call items."""
    output = _get(response, "output")
    if not isinstance(output, list):
        return None

    actions: list[dict[str, Any]] = []
    for item in output:
        if _get(item, "type") != "computer_call":
            continue

        # Singular "action" dict
        action_raw = _get(item, "action")
        if action_raw is not None:
            if not isinstance(action_raw, dict):
                action_raw = _as_dict(action_raw)
            if action_raw is not None:
                parsed = _normalize_action_dict(action_raw)
                if parsed is not None:
                    actions.append(parsed)
            continue

        # Plural "actions" array (batch)
        actions_list = _get(item, "actions")
        if isinstance(actions_list, list):
            for entry in actions_list:
                if not isinstance(entry, dict):
                    entry = _as_dict(entry)
                if entry is not None:
                    parsed = _normalize_action_dict(entry)
                    if parsed is not None:
                        actions.append(parsed)

    return actions or None


# ---------------------------------------------------------------------------
# Path (b): function_call
# ---------------------------------------------------------------------------


def _try_parse_function_call(response: Any) -> list[dict[str, Any]] | None:
    """Parse function_call with ``computer`` or legacy ``gui_action`` tool.

    Supports two layouts:
      1. ``output`` list with ``function_call`` items.
      2. ``tool_calls`` list on the assistant message.
    """
    _TOOL_NAMES = frozenset({"computer", "gui_action"})
    actions: list[dict[str, Any]] = []

    # Layout 1: Responses API output items
    output = _get(response, "output")
    if isinstance(output, list):
        for item in output:
            if _get(item, "type") != "function_call":
                continue
            if _get(item, "name") not in _TOOL_NAMES:
                continue
            parsed = _parse_function_args(_get(item, "arguments"))
            if parsed is not None:
                actions.append(parsed)
        if actions:
            return actions

    # Layout 2: Chat Completions tool_calls
    tool_calls = _get(response, "tool_calls")
    if tool_calls is None:
        message = _get(response, "message")
        if message is not None:
            tool_calls = _get(message, "tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            fn = _get(tc, "function")
            if fn is None:
                continue
            if _get(fn, "name") not in _TOOL_NAMES:
                continue
            parsed = _parse_function_args(_get(fn, "arguments"))
            if parsed is not None:
                actions.append(parsed)

    return actions or None


def _parse_function_args(arguments: Any) -> dict[str, Any] | None:
    """Decode JSON args from a function call into a normalized action dict."""
    if arguments is None:
        return None
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    elif isinstance(arguments, dict):
        args = arguments
    else:
        return None
    return _normalize_action_dict(args)


# ---------------------------------------------------------------------------
# Path (c): structured text
# ---------------------------------------------------------------------------

_TEXT_CLICK_RE = re.compile(
    r"^\s*CLICK\s+(\d+)\s+(\d+)(?:\s+(left|right|double))?\s*$",
    re.IGNORECASE,
)
_TEXT_TYPE_RE = re.compile(
    r'^\s*TYPE\s+"(.*)"\s*$', re.IGNORECASE | re.DOTALL
)
_TEXT_KEY_RE = re.compile(
    r"^\s*(?:HOTKEY|KEYPRESS)\s+(\S+)\s*$", re.IGNORECASE
)
_TEXT_SCROLL_RE = re.compile(
    r"^\s*SCROLL\s+(up|down)(?:\s+(\d+))?\s*$", re.IGNORECASE
)
_TEXT_DRAG_RE = re.compile(
    r"^\s*DRAG\s+(\d+)\s+(\d+)\s*(?:->|to)\s*(\d+)\s+(\d+)\s*$",
    re.IGNORECASE,
)
_TEXT_WAIT_RE = re.compile(r"^\s*WAIT\s+(\d+)\s*$", re.IGNORECASE)
_TEXT_DONE_RE = re.compile(
    r"^\s*DONE(?:[:\s]+(.*))?$", re.IGNORECASE | re.DOTALL
)
_TEXT_MOVE_RE = re.compile(
    r"^\s*MOVE\s+(\d+)\s+(\d+)\s*$", re.IGNORECASE
)


def _try_parse_text(response: Any) -> list[dict[str, Any]] | None:
    """Parse structured text with action keywords."""
    text = _extract_text(response)
    if text is None:
        return None

    actions: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = _TEXT_CLICK_RE.match(line)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            btn = (m.group(3) or "left").lower()
            if btn == "double":
                actions.append({"action": "double_click", "x": x, "y": y})
            elif btn == "right":
                actions.append({"action": "right_click", "x": x, "y": y})
            else:
                actions.append({"action": "click", "x": x, "y": y})
            continue

        m = _TEXT_TYPE_RE.match(line)
        if m:
            actions.append({"action": "type", "text": m.group(1)})
            continue

        m = _TEXT_KEY_RE.match(line)
        if m:
            keys = [k.strip() for k in m.group(1).split("+") if k.strip()]
            actions.append({"action": "keypress", "keys": keys})
            continue

        m = _TEXT_SCROLL_RE.match(line)
        if m:
            direction = m.group(1).lower()
            amount = int(m.group(2)) if m.group(2) else 3
            scroll_y = amount if direction == "down" else -amount
            actions.append({"action": "scroll", "scroll_y": scroll_y})
            continue

        m = _TEXT_DRAG_RE.match(line)
        if m:
            actions.append({
                "action": "drag",
                "start_x": int(m.group(1)),
                "start_y": int(m.group(2)),
                "end_x": int(m.group(3)),
                "end_y": int(m.group(4)),
            })
            continue

        m = _TEXT_WAIT_RE.match(line)
        if m:
            actions.append({"action": "wait", "ms": int(m.group(1))})
            continue

        m = _TEXT_DONE_RE.match(line)
        if m:
            summary = (m.group(1) or "").strip()
            actions.append({"action": "done", "summary": summary})
            continue

        m = _TEXT_MOVE_RE.match(line)
        if m:
            actions.append({
                "action": "move",
                "x": int(m.group(1)),
                "y": int(m.group(2)),
            })
            continue

    return actions or None


def _extract_text(response: Any) -> str | None:
    """Pull a flat text string from assorted response shapes."""
    if isinstance(response, str):
        return response

    output = _get(response, "output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = _get(item, "content")
            if isinstance(content, list):
                for c in content:
                    text = _get(c, "text")
                    if isinstance(text, str):
                        chunks.append(text)
            elif isinstance(content, str):
                chunks.append(content)
        if chunks:
            return "\n".join(chunks)

    message = _get(response, "message")
    if message is not None:
        content = _get(message, "content")
        if isinstance(content, str):
            return content

    content = _get(response, "content")
    if isinstance(content, str):
        return content

    return None


# ---------------------------------------------------------------------------
# execute_gui_action — dispatch via cuaComputerHandler
# ---------------------------------------------------------------------------


async def execute_gui_action(
    action: dict[str, Any], handler: Any
) -> str | None:
    """Execute an action dict against a cuaComputerHandler.

    Uses the same getattr(handler, action_type)(**params) dispatch pattern
    as agent.py:824-828.

    Returns:
        The summary string for ``done`` action (terminal, no VM call).
        None for all other actions (executed on handler).
    Raises ValueError for unknown action types.
    """
    action_type = action.get("action")
    if action_type not in _VALID_ACTIONS:
        raise ValueError(f"Unknown action type: {action_type!r}")

    if action_type == "done":
        return action.get("summary", "")

    if action_type == "screenshot":
        return None

    if action_type == "wait":
        ms = action.get("ms", 1000)
        await asyncio.sleep(ms / 1000.0)
        return None

    relevant_params = _ACTION_PARAM_MAP.get(action_type, [])
    params = {
        k: v
        for k, v in action.items()
        if k != "action" and k in relevant_params and v is not None
    }

    # scroll() requires all 4 positional args; default missing ones to 0.
    if action_type == "scroll":
        for k in ("x", "y", "scroll_x", "scroll_y"):
            params.setdefault(k, 0)

    method = getattr(handler, action_type, None)
    if method is None:
        raise ValueError(f"Handler has no method for action: {action_type!r}")
    await method(**params)
    return None


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "computer_tool_schema",
    "execute_gui_action",
    "parse_gui_response",
    "validate_action",
]
