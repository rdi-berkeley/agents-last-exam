"""Configuration for the official DeepSeek Harness Python SDK deployer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class DeepSeekHarnessConfig:
    """Per-episode DeepSeek Harness settings."""

    name: ClassVar[str] = "deepseek-harness"

    model: str = "deepseek-v4-flash"
    provider: str = "deepseek-official"

    api_key: str | None = None
    """Literal key, kept out of gathered specs. ``None`` reads ``api_key_env``."""

    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str | None = None
    """Optional OpenAI-compatible API root passed as ``DEEPSEEK_BASE_URL``."""

    max_tokens: int | None = None
    """Optional positive output-token cap for the root agent and its descendants."""

    system_prompt: str | None = None
    """Deployment persona. ``None`` uses the bundled runtime's coding-agent default."""

    sdk_version: str = "0.1.0rc6"
    """Pinned ``deepseek-harness-sdk`` and bundled runtime wheel version."""

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("deepseek_harness: model must not be empty")
        if not self.provider.strip():
            raise ValueError("deepseek_harness: provider must not be empty")
        if not self.api_key_env.strip():
            raise ValueError("deepseek_harness: api_key_env must not be empty")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("deepseek_harness: max_tokens must be positive")
        if not self.sdk_version.strip():
            raise ValueError("deepseek_harness: sdk_version must not be empty")
