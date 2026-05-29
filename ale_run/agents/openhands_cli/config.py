"""OpenHandsCliConfig: per-episode knobs for the OpenHands CLI deployer.

OpenHands CLI is the official ``openhands-cli`` pip package.  It uses
LiteLLM internally, so model strings follow LiteLLM conventions:

  - OpenRouter:  ``openrouter/anthropic/claude-sonnet-4.6``
  - Direct:      ``anthropic/claude-sonnet-4-6``

The deployer writes ``~/.openhands/.env`` with LLM_MODEL / LLM_API_KEY /
LLM_BASE_URL and passes ``--override-with-envs`` so the CLI honours them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig


@dataclass
class OpenHandsCliConfig(BaseAgentConfig):
    """Tunables for :class:`OpenHandsCliDeployer`."""

    name: ClassVar[str] = "openhands_cli"

    model: str = "anthropic/claude-sonnet-4-6"
    """LiteLLM model id.  For OpenRouter routing, prefix with
    ``openrouter/`` (e.g. ``openrouter/anthropic/claude-sonnet-4.6``);
    the deployer also adds this prefix automatically when
    ``provider="openrouter"`` and it is missing."""

    # ---- routing (no secrets — API keys come from shell env) ----
    provider: str = "openrouter"
    """Routing provider, drives LiteLLM model prefix + env setup explicitly
    (not a model-name heuristic):
      - ``"openrouter"`` → LiteLLM model carries the ``openrouter/`` prefix,
        LLM_BASE_URL=openrouter, LLM_API_KEY=OPENROUTER_API_KEY. Requires
        OPENROUTER_API_KEY.
      - ``"direct"`` → LiteLLM model used as-is (e.g. ``anthropic/...``),
        LLM_API_KEY=ANTHROPIC_API_KEY, no base URL. Requires
        ANTHROPIC_API_KEY.
    Missing the required key for the chosen provider is a hard error."""

    timeout_s: float = 600.0

    cli_version: str = "1.16.0"
    """Version of the ``openhands`` pip package to install."""

    disable_condenser: bool = False
    """When True, sets ``OPENHANDS_DISABLE_CONDENSER=1`` in the env file
    to suppress the LLMSummarizingCondenser.  Useful when condensation
    triggers provider-shape errors."""

    max_iterations: int = 100_000
    """OpenHands AgentBudget iteration cap.  Set very high because
    wall-clock ``timeout_s`` is the real termination cap."""

    extra_envs: dict[str, str] = field(default_factory=dict)
    """Free-form passthrough env vars exported to the runner script
    (e.g. ``LITELLM_LOG=DEBUG``).  Keys here override anything the
    deployer would otherwise compute."""
