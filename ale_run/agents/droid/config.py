"""DroidConfig: per-episode knobs for the Factory.ai ``droid`` CLI deployer.

Auth: ``FACTORY_API_KEY`` sentinel (``byok-noop``) satisfies the CLI's
auth gate.  Actual model calls route via OpenRouter using
``OPENROUTER_API_KEY`` configured in ``settings.json``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

# Deny-only tool policy: only disabled tools are listed (no allow list). WebSearch
# is intentionally NOT disabled — internet is allowed, so web access is on.
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
)


@dataclass
class DroidConfig:
    """Tunables for :class:`DroidDeployer`.

    Standalone config (no shared base). The episode wall-budget is
    orchestration-owned; ``timeout_s`` is no longer an agent knob.
    """

    name: ClassVar[str] = "droid"

    # agenthle droid_openrouter_claude_opus_4_7.yaml: anthropic/claude-opus-4.7
    # (alt droid_openrouter_gpt5_5.yaml: openai/gpt-5.5).
    model: str = "anthropic/claude-opus-4.7"

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
    # agenthle droid_openrouter_*.yaml: max_output_tokens: 128000 (the model's
    # per-request output cap on OpenRouter — the "no client-side limit" choice;
    # higher values get a hard 400 on the first call).
    max_output_tokens: int = 128000
    byok_provider: str = "generic-chat-completion-api"
    disabled_tools: tuple[str, ...] = _DISABLED_TOOLS
