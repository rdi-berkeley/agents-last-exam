"""GrokCliConfig: per-episode knobs for the Grok CLI deployer.

Grok CLI is a standalone binary (no Node.js runtime required for the
CLI itself).  Node IS required for the CUA MCP Server bridge.

Auth: ``GROK_API_KEY`` for direct xAI, ``OPENROUTER_API_KEY`` for
OpenRouter routing (set ``GROK_BASE_URL`` automatically).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig

_NATIVE_TO_OPENROUTER: dict[str, str] = {
    "grok-4-1-fast-reasoning": "x-ai/grok-4.1-fast",
    "grok-4-1-fast-non-reasoning": "x-ai/grok-4.1-fast",
    "grok-4-1-fast": "x-ai/grok-4.1-fast",
    "grok-4-fast-reasoning": "x-ai/grok-4-fast",
    "grok-4-fast-non-reasoning": "x-ai/grok-4-fast",
    "grok-4-fast": "x-ai/grok-4-fast",
    "grok-4-0709": "x-ai/grok-4-0709",
    "grok-code-fast-1": "x-ai/grok-code-fast-1",
    "grok-code-fast": "x-ai/grok-code-fast-1",
    "grok-3": "x-ai/grok-3",
    "grok-3-mini": "x-ai/grok-3-mini",
    "grok-3-mini-fast": "x-ai/grok-3-mini",
}

_DISABLED_TOOLS_BASE = (
    "computer_snapshot",
    "computer_screenshot",
    "computer_click",
    "computer_mouse_move",
    "computer_type",
    "computer_press",
    "computer_scroll",
    "computer_launch",
    "computer_list_windows",
    "computer_focus_window",
    "computer_wait",
    "computer_get",
)

_DISABLED_TOOLS_OPENROUTER = _DISABLED_TOOLS_BASE + (
    "search_web",
    "search_x",
    "generate_image",
    "generate_video",
    "wallet_info",
    "wallet_history",
    "fetch_payment_info",
    "paid_request",
    "lsp",
)


def native_to_openrouter_model(model: str) -> str:
    if model.startswith(("x-ai/", "xai/")):
        return model
    return _NATIVE_TO_OPENROUTER.get(model, f"x-ai/{model}")


@dataclass
class GrokCliConfig(BaseAgentConfig):
    """Tunables for :class:`GrokCliDeployer`."""

    name: ClassVar[str] = "grok-cli"

    model: str = "grok-4-1-fast-reasoning"
    max_tool_rounds: int = 400
    disabled_tools: tuple[str, ...] = _DISABLED_TOOLS_BASE
