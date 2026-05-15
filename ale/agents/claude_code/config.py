"""ClaudeCodeConfig: per-episode knobs for the Claude Code CLI deployer.

API keys are **never auto-read** from the environment — callers pass them
in explicitly. Inherits :class:`BaseAgentConfig` for the shared
``model`` / ``max_turns`` / ``timeout_s`` / ``api_keys`` / ``install_paths``
surface; adds Claude-specific knobs below.

Typical usage::

    import os
    cfg = ClaudeCodeConfig(
        model="claude-opus-4-7",
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
        max_budget_usd=5.0,
        disabled_tools=("WebSearch",),
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale.agents.base import BaseAgentConfig


@dataclass
class ClaudeCodeConfig(BaseAgentConfig):
    """Tunables for :class:`ClaudeCodeDeployer`."""

    name: ClassVar[str] = "claude-code"

    # ---- override base default ----
    model: str = "claude-sonnet-4-6"

    # ---- routing ----
    anthropic_api_key: str = ""
    """Used when calling Anthropic directly. Mutually exclusive with
    ``openrouter_api_key``."""

    openrouter_api_key: str | None = None
    """When set, route through OpenRouter (sets ``ANTHROPIC_BASE_URL`` +
    ``ANTHROPIC_AUTH_TOKEN`` on the VM)."""

    base_url: str | None = None
    """Custom OpenAI-compatible base URL. ``None`` + ``openrouter_api_key``
    defaults to ``https://openrouter.ai/api``."""

    # ---- CLI knobs ----
    max_budget_usd: float | None = None
    disabled_tools: tuple[str, ...] = ()
    dangerously_skip_permissions: bool = True

    # ---- documentation ----
    cli_version: str = "@anthropic-ai/claude-code@2.1.85"
    """Pinned for the image-baking pipeline; the deployer only verifies the
    binary's presence, it doesn't install."""

    # ---- derived ----
    @property
    def is_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def resolved_base_url(self) -> str | None:
        if self.is_openrouter and self.base_url is None:
            return "https://openrouter.ai/api"
        return self.base_url

    def __post_init__(self) -> None:
        if not self.anthropic_api_key and not self.openrouter_api_key:
            raise ValueError(
                "ClaudeCodeConfig requires either anthropic_api_key or openrouter_api_key"
            )
