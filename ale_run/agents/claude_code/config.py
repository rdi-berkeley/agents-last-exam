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

from ale_run.base_interface import BaseAgentConfig

# Built-in Claude Code tools that break headless (`-p`) runs. Mirrors the
# default disabled_tools list shipped in agenthle's claude_code_openrouter.yaml
# — passed to the CLI as repeated ``--disallowedTools`` flags. Each either
# blocks on human interaction or mutates persistent session state with no
# headless equivalent, so leaving them enabled risks deadlocks.
_DISABLED_TOOLS = (
    # Plan mode needs interactive user approval (ExitPlanMode waits for a
    # human accept) — deadlocks headless runs.
    "EnterPlanMode",
    "ExitPlanMode",
    # Worktree tools mutate session CWD with persistent side effects.
    "EnterWorktree",
    "ExitWorktree",
    # Pure user-interaction tool — no headless equivalent.
    "AskUserQuestion",
    # Background task lifecycle (no running task ID exists in headless).
    "TaskOutput",
    "TaskStop",
    # Requires a logged-in claude.ai account; benchmark VMs are not.
    "RemoteTrigger",
)


@dataclass
class ClaudeCodeConfig(BaseAgentConfig):
    """Tunables for :class:`ClaudeCodeDeployer`."""

    name: ClassVar[str] = "claude-code"

    # ---- override base default ----
    model: str = "claude-sonnet-4-6"

    # ---- routing (no secrets — API keys come from shell env) ----
    provider: str = "openrouter"
    """Routing provider, drives env setup explicitly (not key-presence
    heuristics):
      - ``"openrouter"`` → ANTHROPIC_BASE_URL=openrouter, AUTH_TOKEN=
        OPENROUTER_API_KEY, ANTHROPIC_API_KEY="". Requires OPENROUTER_API_KEY.
      - ``"direct"`` → uses ANTHROPIC_API_KEY against anthropic.com (or
        ``base_url`` if set). Requires ANTHROPIC_API_KEY.
    Missing the required key for the chosen provider is a hard error."""

    base_url: str | None = None
    """Custom Anthropic-compatible base URL. Overrides the provider's
    default ``ANTHROPIC_BASE_URL``."""

    # ---- CLI knobs ----
    max_budget_usd: float | None = None
    disabled_tools: tuple[str, ...] = _DISABLED_TOOLS
    dangerously_skip_permissions: bool = True

    # ---- documentation ----
    cli_version: str = "@anthropic-ai/claude-code@2.1.85"
    """Full npm spec the deployer installs when ``claude`` is not already on
    PATH (e.g. on a non-prebaked image). When the binary is baked into the
    image this is just the version of record."""
