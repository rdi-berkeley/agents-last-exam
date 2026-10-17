"""Configuration for the pinned Qwen Code CLI and its OpenAI-compatible endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class QwenCodeConfig:
    """Tunables for :class:`QwenCodeDeployer`.

    Standalone config (no shared base). The episode wall-budget is
    orchestration-owned; ``timeout_s`` is no longer an agent knob.
    """

    name: ClassVar[str] = "qwen_code"

    model: str = "Qwen/Qwen3.5-9B"
    """Model id, sent as-is via ``OPENAI_MODEL`` -- no LiteLLM prefix
    convention here, this is a real OpenAI-compatible-chat-completions
    request, not a routed one."""

    base_url: str = "https://router.huggingface.co/v1"
    """OpenAI-compatible base URL (``OPENAI_BASE_URL``)."""

    api_key: str | None = field(default=None, repr=False)
    """Literal API key (``OPENAI_API_KEY``). Required -- there is no
    provider-name heuristic or env-var fallback name to guess here (unlike
    openhands_cli's provider-driven OPENROUTER_API_KEY/ANTHROPIC_API_KEY
    convention); set ``api_key: ${env:HF_INFERENCE_TOKEN}`` in the agent yaml."""

    cli_version: str = "0.24.4"
    """Exact version of the ``@qwen-code/qwen-code`` npm package to install."""

    extra_envs: dict[str, str] = field(default_factory=dict)
    """Free-form passthrough env vars exported to the qwen subprocess.
    Keys here override anything the deployer would otherwise compute."""
