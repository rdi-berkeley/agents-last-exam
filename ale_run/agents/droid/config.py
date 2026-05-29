"""DroidConfig: per-episode knobs for the Factory.ai ``droid`` CLI deployer.

Auth: ``FACTORY_API_KEY`` sentinel (``byok-noop``) satisfies the CLI's
auth gate.  Actual model calls route via OpenRouter using
``OPENROUTER_API_KEY`` configured in ``settings.json``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig

_DISABLED_TOOLS = (
    "squad-board",
    "slack_post_message",
    "store_agent_readiness_report",
    "GenerateDroid",
    "ProposeMission",
    "StartMissionRun",
    "EndFeatureRun",
    "DismissHandoffItems",
    "ExitSpecMode",
    "AskUser",
    "WebSearch",
)


@dataclass
class DroidConfig(BaseAgentConfig):
    """Tunables for :class:`DroidDeployer`."""

    name: ClassVar[str] = "droid"

    model: str = "anthropic/claude-sonnet-4-6"

    # ---- routing (no secrets — API keys come from shell env) ----
    provider: str = "openrouter"
    """Routing provider, matching agenthle's droid config. droid is a
    closed-source Factory binary whose model calls go through the BYOK
    ``customModels`` entry; the deployer wires that to OpenRouter
    (``baseUrl=https://openrouter.ai/api/v1`` + OPENROUTER_API_KEY), so
    ``"openrouter"`` is the only routing the deployer implements today.
    ``byok_provider`` below selects the wire protocol within that route."""

    reasoning_effort: str = "high"
    """``--reasoning-effort``: off | none | low | medium | high. Defaults to
    ``high`` to match agenthle's operational droid YAMLs (the agenthle
    dataclass default of ``medium`` was not the value actually run)."""

    skip_permissions_unsafe: bool = True
    max_output_tokens: int = 32000
    byok_provider: str = "generic-chat-completion-api"
    disabled_tools: tuple[str, ...] = _DISABLED_TOOLS
    enabled_tools: tuple[str, ...] = ()
