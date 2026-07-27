"""Configuration for the official xAI Grok Build CLI deployer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class GrokBuildConfig:
    """Per-episode Grok Build settings."""

    name: ClassVar[str] = "grok-build"

    model: str = "grok-4.5"

    api_key: str | None = None
    """Literal API key. ``None`` reads :attr:`api_key_env` from the executor."""

    api_key_env: str = "XAI_API_KEY"
    base_url: str | None = None
    """Optional custom model endpoint. ``None`` uses xAI's built-in model catalog."""

    api_backend: str = "responses"
    """Custom endpoint protocol: ``chat_completions``, ``responses``, or ``messages``."""

    context_window: int | None = None
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None
    max_turns: int | None = None

    disabled_tools: tuple[str, ...] = ()
    """Additional tools to remove beyond ALE's mandatory headless exclusions."""

    cli_version: str = "@xai-official/grok@0.2.112"
    """Pinned official npm package installed dynamically when missing or stale."""
