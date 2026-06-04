"""Shared helper-call adapter backed by resolved runtime metadata."""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Any, Literal

from .model_config import ResolvedModel, resolve_model


@dataclass(frozen=True)
class HelperCallResult:
    text: str
    tool_calls: list[dict[str, Any]]


async def call_helper_model(
    model: str | ResolvedModel,
    *,
    purpose: Literal["memory_flush", "compaction", "vision"],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int,
    temperature: float,
    timeout: int | None = None,
    thinking_params: dict[str, Any] | None = None,
) -> HelperCallResult:
    """Call a helper model using the resolved transport defaults for the purpose."""
    import litellm

    resolved = resolve_model(model)
    transport = resolved.helper_transport_defaults.for_purpose(purpose)

    if transport == "responses":
        kwargs: dict[str, Any] = {
            "model": resolved.model,
            "input": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **(thinking_params or {}),
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        if tools:
            kwargs["tools"] = [_to_responses_function_tool(tool) for tool in tools]
        response = await litellm.aresponses(**kwargs)
        payload = response.model_dump()
        return HelperCallResult(
            text=_extract_responses_text(payload.get("output", [])),
            tool_calls=_extract_responses_tool_calls(payload.get("output", [])),
        )

    kwargs = {
        "model": resolved.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **(thinking_params or {}),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if tools:
        kwargs["tools"] = tools
    response = await litellm.acompletion(**kwargs)
    choice = response.choices[0]
    return HelperCallResult(
        text=(choice.message.content or "").strip(),
        tool_calls=[
            {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
            for tool_call in (choice.message.tool_calls or [])
        ],
    )


def _to_responses_function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function", {})
    return {"type": "function", **function}


def _extract_responses_text(output_items: Any) -> str:
    if not isinstance(output_items, list):
        return ""

    parts: list[str] = []
    for item in output_items:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content", [])
        if isinstance(content, str):
            if content:
                parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"output_text", "text"} and block.get("text"):
                parts.append(block["text"])
    return "\n".join(parts).strip()


def _extract_responses_tool_calls(output_items: Any) -> list[dict[str, Any]]:
    if not isinstance(output_items, list):
        return []

    tool_calls: list[dict[str, Any]] = []
    for item in output_items:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        tool_calls.append(
            {
                "name": item.get("name", ""),
                "arguments": item.get("arguments", _json.dumps({})),
            }
        )
    return tool_calls
