"""Configuration for the native Kimi Code CLI deployer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class KimiCodeConfig:
    """Per-episode Kimi Code settings."""

    name: ClassVar[str] = "kimi-code"

    model: str = "kimi-k3"

    api_key: str | None = None
    """Literal API key. ``None`` reads :attr:`api_key_env` from the executor."""

    api_key_env: str = "MOONSHOT_API_KEY"
    base_url: str = "https://api.moonshot.ai/v1"
    provider_type: str = "kimi"

    max_context_size: int = 262144
    capabilities: tuple[str, ...] = ("image_in", "thinking")
    thinking_effort: str | None = "high"
    max_completion_tokens: int | None = None

    disable_telemetry: bool = True
    disable_auto_update: bool = True

    cli_version: str = "@moonshot-ai/kimi-code@0.27.0"
    """Pinned npm package spec installed dynamically when missing or stale."""
