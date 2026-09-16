"""Configuration for the pi (earendil-works/pi) CLI deployer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class PiCliConfig:
    """Per-episode pi CLI settings."""

    name: ClassVar[str] = "pi_cli"

    model: str = "anthropic/claude-sonnet-4.6"
    provider: str = "openrouter"
    base_url: str | None = None
    api_key: str | None = None
    cli_version: str = "0.85.1"
    thinking_level: str | None = None
    disabled_tools: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)
