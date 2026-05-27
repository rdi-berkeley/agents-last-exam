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
    reasoning_effort: str = "medium"
    skip_permissions_unsafe: bool = True
    max_output_tokens: int = 32000
    byok_provider: str = "generic-chat-completion-api"
    disabled_tools: tuple[str, ...] = _DISABLED_TOOLS
    enabled_tools: tuple[str, ...] = ()
