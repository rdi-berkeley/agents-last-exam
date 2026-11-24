"""Preserve provider reasoning fields across chat-completion turns."""

from __future__ import annotations

from typing import Any


def reasoning_fields(message: Any) -> dict[str, Any]:
    """Extract the structured reasoning fields from a LiteLLM message."""
    def get(obj: Any, key: str) -> Any:
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    provider_fields = get(message, "provider_specific_fields") or {}
    result: dict[str, Any] = {}
    text = get(message, "reasoning_content")
    if text is None:
        text = get(message, "reasoning")
    if text is None:
        text = get(provider_fields, "reasoning_content")
    if text is None:
        text = get(provider_fields, "reasoning")
    if isinstance(text, str):
        result["reasoning_content"] = text

    details = get(message, "reasoning_details")
    if details is None:
        details = get(provider_fields, "reasoning_details")
    if isinstance(details, list):
        result["reasoning_details"] = [
            item.model_dump(exclude_none=False) if hasattr(item, "model_dump") else item
            for item in details
        ]
    return result
