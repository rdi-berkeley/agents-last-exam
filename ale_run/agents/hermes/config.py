"""HermesConfig: per-episode knobs for the Hermes Agent deployer.

Hermes is the open-source CLI agent from Nous Research.  ALE runs it
against the ``cua-verse/hermes-agent`` fork on branch ``agenthle``
which carries vision patches (MCP ``ImageContent`` -> multimodal
follow-up, tool-result truncation guards).

Auth: ``OPENROUTER_API_KEY`` routed through OpenRouter.  Provider is
configurable via ``provider`` field but defaults to ``openrouter``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

# Toolset ids match hermes-agent/toolsets.py::TOOLSETS plus runtime-registered
# ``mcp-<server>`` ids. hermes is enable-only: ``--toolsets`` lists exactly the
# toolsets to turn on, and anything not listed is off. There is no deny list
# (the harness has no disable flag), so unlike other agents hermes carries an
# allow list rather than a deny list — by harness constraint, not preference.
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

@dataclass
class HermesConfig:
    """Tunables for :class:`HermesDeployer`.

    Hermes is Linux-only.  The fork carries patches for MCP ImageContent
    multimodal forwarding and tool-result truncation guards.

    Standalone config (no shared base). The episode wall-budget is
    orchestration-owned; ``timeout_s`` is no longer an agent knob.
    """

    name: ClassVar[str] = "hermes"

    # agenthle hermes_openrouter.yaml: anthropic/claude-sonnet-4.6.
    model: str = "anthropic/claude-sonnet-4.6"
    max_turns: int = -1
    """Turn cap. ``-1`` = unlimited, the project-wide convention. Hermes itself
    has no unlimited sentinel (``IterationBudget.consume`` does
    ``if used >= max_total: stop``), so the deployer translates ``-1`` (or any
    value < 0) to ``100_000`` before passing ``--max-turns`` — wall-clock is then
    the real cap."""

    # Sonnet 4.6 full 1M-token context window.
    context_length: int = 1_000_000

    # Compression settings (tuned for 1M context + multi-MB screenshots).
    compression_threshold: float = 0.85
    compression_target_ratio: float = 0.20
    compression_protect_last_n: int = 8

    # Provider routing
    provider: str = "openrouter"

    # Toolset configuration. hermes is enable-only (the harness exposes no deny
    # flag), so this allow list is the sole tool-control knob — by harness
    # constraint, not a departure from the project's deny-only convention.
    toolsets_enabled: tuple[str, ...] = DEFAULT_TOOLSETS

    # Prepend a keepalive preamble to every prompt so the agent knows not
    # to end the chat process early when it schedules background work.
    keepalive_preamble: bool = True
