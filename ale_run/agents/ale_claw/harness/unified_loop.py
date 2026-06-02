"""
Unified agent loop for OpenRouter — single acompletion() transport for all providers.

Uses function-calling computer tools (not native hosted tools) and OpenRouter's
unified reasoning parameter. Designed to replace per-provider loops when routing
through OpenRouter.

US-OC-051: Unified Agent Loop via OpenRouter.
Design reference: docs/plan/US-OC-050-054-unified-loop.md

Vendored into the openclaw subpackage (single source of truth) so sparse-checkout
consumers that only pull ``cua_bench/agents/openclaw/`` — and would otherwise
fall through to the Responses-API ``loops/openai.py`` route — pick up the
chat-completions OpenRouter route the moment openclaw is imported. The
original ``agent/loops/unified.py`` is removed; this file is the canonical
home. Registered via the side-effect import at the bottom of
``cua_bench/agents/openclaw/__init__.py``.
"""

import base64
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import litellm
from PIL import Image

from agent.decorators import register_agent
from agent.loops.base import AsyncAgentConfig
from agent.types import AgentCapability, Messages, Tools

from ._message_shapes import _image_url_block
from .cache_policy import apply_openclaw_cache_markers


# Fallback call_id when a tool/computer call item arrives without an id/call_id.
FALLBACK_CALL_ID = "call_1"

# Default screen dimensions used when the computer handler can't report them.
DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT = 1024, 768


# ---------------------------------------------------------------------------
# Tool preparation
# ---------------------------------------------------------------------------

async def _build_computer_tool_schema(computer_handler: Any) -> Dict[str, Any]:
    """Build a function-calling computer tool schema from the computer handler.

    Returns an OpenAI Chat Completions function tool dict with the computer
    action schema.  This is the same schema upstream uses for GPT-5.4.
    """
    try:
        width, height = await computer_handler.get_dimensions()
    except Exception:
        width, height = DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT

    try:
        environment = await computer_handler.get_environment()
    except Exception:
        environment = "windows"

    return {
        "type": "function",
        "function": {
            "name": "computer",
            "description": (
                f"Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
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
                            "terminate",
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
                        "description": "Keys to press (for action=keypress). Example: ['ctrl', 'c']",
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "scroll_x": {
                        "description": "Horizontal scroll amount. Positive=right, negative=left.",
                        "type": "integer",
                    },
                    "scroll_y": {
                        "description": "Vertical scroll amount. Positive=down, negative=up.",
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
                    "status": {
                        "description": "Status for terminate action.",
                        "type": "string",
                        "enum": ["success", "failure"],
                    },
                },
                "required": ["action"],
            },
        },
    }


async def _prepare_tools(
    tool_schemas: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert CUA internal tool schemas to Chat Completions function tool format."""
    chat_tools: List[Dict[str, Any]] = []
    for schema in tool_schemas:
        if schema["type"] == "computer":
            chat_tools.append(await _build_computer_tool_schema(schema["computer"]))
        elif schema["type"] == "function":
            func = schema["function"]
            chat_tools.append({
                "type": "function",
                "function": {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                },
            })
    return chat_tools


# ---------------------------------------------------------------------------
# Input conversion: Responses API items → Chat Completions messages
# ---------------------------------------------------------------------------

def _append_tool_call(messages: List[Dict[str, Any]], tool_call: Dict[str, Any]) -> None:
    """Attach a Chat Completions tool_call to the last assistant message, or
    start a fresh assistant message when the previous one isn't an assistant."""
    if messages and messages[-1].get("role") == "assistant":
        messages[-1].setdefault("tool_calls", []).append(tool_call)
    else:
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        )


def _append_tool_role(item: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
    """Convert a ``role: tool`` item (canonical tool_result blocks or a plain
    string) into Chat Completions tool messages."""
    content = item.get("content", [])
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "tool_result":
                tool_use_id = c.get("tool_use_id", FALLBACK_CALL_ID)
                result_content = c.get("content", "")
                if isinstance(result_content, list):
                    # Extract text from content blocks, skip images
                    text_parts = []
                    for rc in result_content:
                        if isinstance(rc, dict):
                            if rc.get("type") == "text":
                                text_parts.append(rc.get("text", ""))
                            elif rc.get("type") == "image":
                                # Image in tool result — add as user message after
                                pass
                    result_content = "\n".join(text_parts) if text_parts else str(result_content)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": str(result_content),
                })
    elif isinstance(content, str):
        # Simple tool message
        messages.append({
            "role": "tool",
            "tool_call_id": item.get("tool_call_id", FALLBACK_CALL_ID),
            "content": content,
        })


def _append_user_role(item: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
    """Convert a ``role: user`` item (string or content-block list) into a
    Chat Completions user message; nested tool_result blocks become tool messages."""
    content = item.get("content", "")
    if isinstance(content, list):
        converted = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "input_image":
                image_url = c.get("image_url", "")
                if image_url and image_url != "[omitted]":
                    converted.append(_image_url_block(image_url))
            elif isinstance(c, dict) and c.get("type") == "input_text":
                converted.append({"type": "text", "text": c.get("text", "")})
            elif isinstance(c, dict) and c.get("type") == "tool_result":
                # tool_result inside user message (canonical format)
                # Convert to a tool role message instead
                tool_use_id = c.get("tool_use_id", FALLBACK_CALL_ID)
                result_content = c.get("content", "")
                if isinstance(result_content, list):
                    text_parts = []
                    for rc in result_content:
                        if isinstance(rc, dict) and rc.get("type") == "text":
                            text_parts.append(rc.get("text", ""))
                    result_content = "\n".join(text_parts) if text_parts else str(result_content)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": str(result_content),
                })
                continue
            elif isinstance(c, dict) and c.get("type") in ("text", "image_url"):
                converted.append(c)
            else:
                # Skip unknown content types rather than pass through
                pass
        if converted:
            messages.append({"role": "user", "content": converted})
    else:
        messages.append({"role": "user", "content": content})


def _append_assistant_role(
    item: Dict[str, Any],
    messages: List[Dict[str, Any]],
    call_id_to_fn_name: Dict[str, str],
) -> None:
    """Convert a ``role: assistant`` item into a Chat Completions assistant
    message, lifting tool_use blocks into ``tool_calls``."""
    content = item.get("content", [])
    if isinstance(content, str):
        messages.append({"role": "assistant", "content": content})
    elif isinstance(content, list):
        text_parts = []
        tool_calls = []
        for c in content:
            if not isinstance(c, dict):
                continue
            ctype = c.get("type", "")
            if ctype == "text":
                text_parts.append(c.get("text", ""))
            elif ctype == "output_text":
                text_parts.append(c.get("text", ""))
            elif ctype == "tool_use":
                # Canonical format: tool_use block inside assistant content
                tool_calls.append({
                    "id": c.get("id", FALLBACK_CALL_ID),
                    "type": "function",
                    "function": {
                        "name": c.get("name", ""),
                        "arguments": json.dumps(c.get("input", {})),
                    },
                })
                call_id_to_fn_name[c.get("id", FALLBACK_CALL_ID)] = c.get("name", "")
            elif ctype == "thinking":
                # Thinking block — skip (not needed for replay)
                pass
        msg: Dict[str, Any] = {"role": "assistant"}
        msg["content"] = "\n".join(text_parts) if text_parts else None
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)
    else:
        messages.append({"role": "assistant", "content": None})


def _append_reasoning(item: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
    """Convert a prior-turn ``reasoning`` item into an assistant text message."""
    summary = item.get("summary", [])
    text = ""
    if isinstance(summary, list):
        for s in summary:
            if isinstance(s, dict) and s.get("type") == "summary_text":
                text = s.get("text", "")
                break
    if not text:
        text = item.get("reasoning", "")
    if text:
        messages.append({"role": "assistant", "content": text})


def _append_function_call(
    item: Dict[str, Any],
    messages: List[Dict[str, Any]],
    call_id_to_fn_name: Dict[str, str],
) -> None:
    """Convert a ``function_call`` item into a tool_call on the assistant turn."""
    fn_name = item.get("name", "")
    fn_args = item.get("arguments", "{}")
    call_id = item.get("call_id", FALLBACK_CALL_ID)
    call_id_to_fn_name[call_id] = fn_name
    _append_tool_call(messages, {
        "id": call_id,
        "type": "function",
        "function": {"name": fn_name, "arguments": fn_args},
    })


def _append_function_call_output(
    item: Dict[str, Any],
    messages: List[Dict[str, Any]],
    call_id_to_fn_name: Dict[str, str],
) -> None:
    """Convert a ``function_call_output`` item into a Chat Completions tool message."""
    call_id = item.get("call_id", FALLBACK_CALL_ID)
    fn_name = call_id_to_fn_name.get(call_id, "computer")
    messages.append({
        "role": "tool",
        "name": fn_name,
        "tool_call_id": call_id,
        "content": str(item.get("output", "")),
    })


def _append_computer_call(
    item: Dict[str, Any],
    messages: List[Dict[str, Any]],
    call_id_to_fn_name: Dict[str, str],
) -> None:
    """Convert a legacy ``computer_call`` item into a ``computer`` function tool_call."""
    action = item.get("action", {})
    call_id = item.get("call_id", FALLBACK_CALL_ID)
    call_id_to_fn_name[call_id] = "computer"
    # Convert to function_call format
    args = dict(action)
    _append_tool_call(messages, {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "computer",
            "arguments": json.dumps(args),
        },
    })


def _append_computer_call_output(
    item: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> None:
    """Convert a legacy ``computer_call_output`` item into a tool message, plus a
    following user image message when the output carries a screenshot."""
    call_id = item.get("call_id", FALLBACK_CALL_ID)
    output = item.get("output", "")
    if isinstance(output, dict) and output.get("type") == "input_image":
        # Screenshot result — send as user message with image
        image_url = output.get("image_url", "")
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps({"success": True}),
        })
        if image_url and image_url != "[omitted]":
            messages.append({
                "role": "user",
                "content": [
                    _image_url_block(image_url),
                ],
            })
    else:
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(output) if isinstance(output, dict) else str(output),
        })


def _convert_input_to_messages(items: Messages) -> List[Dict[str, Any]]:
    """Convert Responses API input items to Chat Completions messages.

    The CUA framework internally uses Responses API item format for the
    conversation history.  OpenRouter's Chat Completions API needs standard
    role-based messages.
    """
    messages: List[Dict[str, Any]] = []
    call_id_to_fn_name: Dict[str, str] = {}

    for item in items:
        item_type = item.get("type")
        role = item.get("role")

        if role == "tool":
            _append_tool_role(item, messages)
        elif role == "user":
            _append_user_role(item, messages)
        elif role == "assistant":
            _append_assistant_role(item, messages, call_id_to_fn_name)
        elif item_type == "reasoning":
            _append_reasoning(item, messages)
        elif item_type == "function_call":
            _append_function_call(item, messages, call_id_to_fn_name)
        elif item_type == "function_call_output":
            _append_function_call_output(item, messages, call_id_to_fn_name)
        elif item_type == "computer_call":
            _append_computer_call(item, messages, call_id_to_fn_name)
        elif item_type == "computer_call_output":
            _append_computer_call_output(item, messages)

    return messages


# ---------------------------------------------------------------------------
# Output conversion: Chat Completions response → Responses API output items
# ---------------------------------------------------------------------------

def _convert_response_to_output(response: Any) -> Dict[str, Any]:
    """Convert a Chat Completions response to Responses API output format.

    Returns {"output": [...items...], "usage": {...}} matching the contract
    expected by ComputerAgent._handle_item().
    """
    output_items: List[Dict[str, Any]] = []

    if not response or not hasattr(response, "choices") or not response.choices:
        return {"output": output_items, "usage": {}}

    choice = response.choices[0]
    message = choice.message

    # --- Extract reasoning (OpenRouter returns in reasoning_content) ---
    reasoning_content = getattr(message, "reasoning_content", None)
    provider_fields = getattr(message, "provider_specific_fields", None) or {}
    reasoning_details = provider_fields.get("reasoning_details", [])

    if reasoning_content:
        # Build reasoning item compatible with Responses API format
        reasoning_item: Dict[str, Any] = {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": reasoning_content}],
        }
        # Preserve signature if available (for Anthropic replay)
        if reasoning_details:
            for rd in reasoning_details:
                if isinstance(rd, dict) and rd.get("signature"):
                    reasoning_item["signature"] = rd["signature"]
                    break
        output_items.append(reasoning_item)

    # --- Extract text content ---
    content = message.content
    if isinstance(content, str) and content:
        output_items.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        })
    elif isinstance(content, list):
        text_parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                text_parts.append(c.get("text", ""))
        if text_parts:
            output_items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "\n".join(text_parts)}],
            })

    # --- Extract tool calls → function_call items ---
    tool_calls = message.tool_calls
    if tool_calls:
        for tc in tool_calls:
            output_items.append({
                "type": "function_call",
                "call_id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })

    # --- Usage ---
    usage: Dict[str, Any] = {}
    if hasattr(response, "usage") and response.usage:
        resp_usage = response.usage
        if hasattr(resp_usage, "model_dump"):
            usage = resp_usage.model_dump()
        elif isinstance(resp_usage, dict):
            usage = resp_usage
        # Normalize field names (Chat Completions → Responses API naming)
        if "prompt_tokens" in usage and "input_tokens" not in usage:
            usage["input_tokens"] = usage["prompt_tokens"]
        if "completion_tokens" in usage and "output_tokens" not in usage:
            usage["output_tokens"] = usage["completion_tokens"]

    if hasattr(response, "_hidden_params"):
        usage["response_cost"] = response._hidden_params.get("response_cost", 0.0)

    return {"output": output_items, "usage": usage}


# ---------------------------------------------------------------------------
# Unified agent loop
# ---------------------------------------------------------------------------

@register_agent(models=r"openrouter/.*", priority=10)
class UnifiedAgentConfig(AsyncAgentConfig):
    """Unified agent loop for OpenRouter — all providers via acompletion().

    Uses function-calling computer tools and OpenRouter's unified reasoning
    parameter.  Registered at priority 10 to take precedence over the
    per-provider loops for any ``openrouter/`` model string.
    """

    async def predict_step(
        self,
        messages: Messages,
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: Optional[int] = None,
        stream: bool = False,
        computer_handler=None,
        use_prompt_caching: Optional[bool] = False,
        _on_api_start=None,
        _on_api_end=None,
        _on_usage=None,
        _on_screenshot=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Predict the next step using litellm.acompletion() via OpenRouter."""
        tools = tools or []

        # Build Chat Completions tools
        chat_tools = await _prepare_tools(tools)

        # Convert Responses API input → Chat Completions messages
        chat_messages = _convert_input_to_messages(messages)

        # Apply OpenClaw cache_control markers in-place. Must happen here
        # (not via a callback) because ``agent.py:_on_api_start`` passes
        # ``get_json(kwargs)`` — a deep copy — to callbacks, so a hook's
        # mutation never reaches the actual API call. For Anthropic-family
        # models the helper marks the system prompt + trailing message as
        # ephemeral; for OpenAI/other providers it strips markers and is a
        # no-op (those providers cache automatically server-side).
        if use_prompt_caching:
            apply_openclaw_cache_markers(chat_messages, model)

        # Build API kwargs
        api_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "tools": chat_tools if chat_tools else None,
            "stream": stream,
            "num_retries": max_retries,
        }

        # Merge generation kwargs (thinking, api_key, etc.)
        # Filter out callbacks and internal params
        for k, v in kwargs.items():
            if not k.startswith("_") and v is not None:
                api_kwargs[k] = v

        # Call API start hook
        if _on_api_start:
            await _on_api_start(api_kwargs)

        # Call OpenRouter via litellm
        response = await litellm.acompletion(**api_kwargs)

        # Call API end hook
        if _on_api_end:
            await _on_api_end(api_kwargs, response)

        # Convert response to Responses API output format
        result = _convert_response_to_output(response)

        if _on_usage:
            await _on_usage(result["usage"])

        return result

    async def predict_click(
        self,
        model: str,
        image_b64: str,
        instruction: str,
        **kwargs,
    ) -> Optional[Tuple[int, int]]:
        """Predict click coordinates using function-calling computer tool."""
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data))
            display_width, display_height = image.size
        except Exception:
            display_width, display_height = DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT

        click_tool = {
            "type": "function",
            "function": {
                "name": "computer",
                "description": (
                    f"Click on the screen. Resolution: {display_width}x{display_height}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["click"]},
                        "x": {"type": "integer", "description": "X coordinate"},
                        "y": {"type": "integer", "description": "Y coordinate"},
                    },
                    "required": ["action", "x", "y"],
                },
            },
        }

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a UI grounding expert. Complete tasks autonomously.\n"
                            f"Task: Click {instruction}. Output ONLY a click action."
                        ),
                    },
                    _image_url_block(f"data:image/png;base64,{image_b64}"),
                ],
            }
        ]

        api_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": [click_tool],
            "stream": False,
            "max_tokens": 200,
        }
        # Pass through api_key/api_base if provided
        for k in ("api_key", "api_base"):
            if k in kwargs and kwargs[k] is not None:
                api_kwargs[k] = kwargs[k]

        response = await litellm.acompletion(**api_kwargs)
        result = _convert_response_to_output(response)

        for item in result.get("output", []):
            if item.get("type") == "function_call" and item.get("name") == "computer":
                try:
                    args = json.loads(item.get("arguments", "{}"))
                    if args.get("x") is not None and args.get("y") is not None:
                        return (int(args["x"]), int(args["y"]))
                except (json.JSONDecodeError, TypeError):
                    continue
        return None

    def get_capabilities(self) -> List[AgentCapability]:
        return ["click", "step"]
