"""HermesConfig: per-episode knobs for the Hermes Agent deployer.

Hermes is the open-source CLI agent from Nous Research.  ALE runs it
against the ``cua-verse/hermes-agent`` fork on branch ``agenthle``
which carries vision patches (MCP ``ImageContent`` -> multimodal
follow-up, tool-result truncation guards).

Auth: ``OPENROUTER_API_KEY`` routed through OpenRouter.  Provider is
configurable via ``provider`` field but defaults to ``openrouter``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig

# Toolset ids match hermes-agent/toolsets.py::TOOLSETS plus runtime-registered
# ``mcp-<server>`` ids.  See the old hermes_openrouter.yaml for the rationale
# behind each enable/disable decision; both lists together cover every known
# id so an audit can spot anything that was never explicitly classified.
DEFAULT_TOOLSETS: tuple[str, ...] = (
    # Always-on core
    "terminal",
    "file",
    "skills",
    "todo",
    "memory",
    # Network research (benchmark allows internet)
    "web",
    "vision",
    "browser",       # local Chromium via Playwright; no BROWSERBASE_API_KEY needed
    # Long-run optimizations
    "code_execution",
    "delegation",
    "cronjob",
    # ALE bridge
    "mcp-cua",
)

DEFAULT_DISABLED_TOOLSETS: tuple[str, ...] = (
    # Subsumed by other toolsets
    "search",
    # Require external API keys / hardware we don't ship
    "image_gen",
    "rl",
    "tts",
    "moa",
    # Don't apply to single-shot benchmark runs
    "session_search",
    "clarify",
    # Platform integrations not available on benchmark VMs
    "messaging",
    "homeassistant",
    "kanban",
    "discord",
    "discord_admin",
    "yuanbao",
    "feishu_doc",
    "feishu_drive",
    "spotify",
    # Composite presets we deliberately don't use
    "debugging",
    "safe",
    "hermes-cli",
    "hermes-acp",
    "hermes-api-server",
    "hermes-cron",
    "hermes-telegram",
    "hermes-discord",
    "hermes-whatsapp",
    "hermes-slack",
    "hermes-signal",
    "hermes-bluebubbles",
    "hermes-homeassistant",
    "hermes-email",
    "hermes-mattermost",
)


@dataclass
class HermesConfig(BaseAgentConfig):
    """Tunables for :class:`HermesDeployer`.

    Hermes is Linux-only.  The fork carries patches for MCP ImageContent
    multimodal forwarding and tool-result truncation guards.
    """

    name: ClassVar[str] = "hermes"

    model: str = "anthropic/claude-sonnet-4-6"
    timeout_s: float = 600
    max_turns: int = 100_000
    """Hermes has no ``-1 = unlimited`` sentinel for ``--max-turns``.
    ``IterationBudget.consume`` does ``if used >= max_total: stop``.
    We default to 100_000 so wall-clock ``timeout_s`` is the real cap."""

    # Sonnet 4.6 full 1M-token context window.
    context_length: int = 1_000_000

    # Compression settings (tuned for 1M context + multi-MB screenshots).
    compression_threshold: float = 0.85
    compression_target_ratio: float = 0.20
    compression_protect_last_n: int = 8

    # Provider routing
    provider: str = "openrouter"

    # Toolset configuration
    toolsets_enabled: tuple[str, ...] = DEFAULT_TOOLSETS
    toolsets_disabled: tuple[str, ...] = DEFAULT_DISABLED_TOOLSETS

    # Prepend a keepalive preamble to every prompt so the agent knows not
    # to end the chat process early when it schedules background work.
    keepalive_preamble: bool = True
