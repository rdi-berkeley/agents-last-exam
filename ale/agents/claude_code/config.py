"""ClaudeCodeConfig: per-episode knobs for the Claude Code CLI deployer.

**API keys live in the operator's shell env**, never in this config.
The deployer's VM-side bash script reads ``ANTHROPIC_API_KEY`` /
``OPENROUTER_API_KEY`` from the inherited env (propagated host → VM by
:mod:`ale.runtime._env`). OpenRouter routing auto-detects: if
``ANTHROPIC_API_KEY`` is unset but ``OPENROUTER_API_KEY`` is set, the
script remaps to ``ANTHROPIC_AUTH_TOKEN`` + ``ANTHROPIC_BASE_URL``.

Typical usage::

    # In shell:
    #   export ANTHROPIC_API_KEY=sk-ant-...    # direct
    #   # OR
    #   export OPENROUTER_API_KEY=sk-or-...    # routed (auto-detected)

    cfg = ClaudeCodeConfig(
        model="claude-opus-4-7",
        max_budget_usd=5.0,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale.agents.base import BaseAgentConfig


@dataclass
class ClaudeCodeConfig(BaseAgentConfig):
    """Tunables for :class:`ClaudeCodeDeployer`."""

    name: ClassVar[str] = "claude-code"

    # ---- override base default ----
    model: str = "claude-sonnet-4-6"

    # ---- routing (no secrets — API keys come from shell env) ----
    base_url: str | None = None
    """Custom Anthropic-compatible base URL. Becomes ``ANTHROPIC_BASE_URL``
    when set; otherwise the bash script defaults to
    ``https://openrouter.ai/api`` if it's doing the OpenRouter remap, or
    leaves it unset (CLI uses anthropic.com default) otherwise."""

    # ---- CLI knobs ----
    max_budget_usd: float | None = None
    disabled_tools: tuple[str, ...] = ()
    dangerously_skip_permissions: bool = True

    # ---- documentation ----
    cli_version: str = "@anthropic-ai/claude-code@2.1.85"
    """Pinned for the image-baking pipeline; the deployer only verifies the
    binary's presence, it doesn't install."""
