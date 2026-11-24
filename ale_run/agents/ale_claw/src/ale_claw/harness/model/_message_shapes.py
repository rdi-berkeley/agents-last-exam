"""Tiny builders for message fragments inlined repeatedly across the loops.

``agent_loop`` and ``unified_loop`` both constructed the same literal
``image_url`` content block and ``function_call_output`` item shapes in
several places. Centralizing the literals here keeps the wire shape in one
spot without changing any behavior.
"""

from __future__ import annotations

from typing import Any


def _image_url_block(url: str) -> dict[str, Any]:
    """A Chat Completions image content block.

    ``url`` is a raw URL or a ``data:<mime>;base64,<...>`` data URI.
    """
    return {"type": "image_url", "image_url": {"url": url}}


def _function_call_output(call_id: Any, output: str) -> dict[str, Any]:
    """A Responses-API ``function_call_output`` item."""
    return {"type": "function_call_output", "call_id": call_id, "output": output}
