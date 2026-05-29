"""Terminus2Config: per-episode knobs for the harbor terminus_2 deployer.

terminus_2 is harbor's tmux-driven, ReAct-style agent.  ALE runs it from
the ``cua-verse/harbor`` fork on branch ``agenthle`` which ships a thin
``harbor-terminus2`` CLI shim plus a ``LocalShellEnvironment`` so the
agent's tmux loop drives the same sandbox it runs inside.

The deployer pip-installs the fork at install() time (no submodule, no
pre-baked image requirement).

Provider routing follows the ALE convention: ``provider`` is either
``openrouter`` (default) or ``direct``.  YAML always carries the
OpenRouter-native ``vendor/model`` id; for openrouter mode the deployer
re-attaches the ``openrouter/`` prefix before handing the id to LiteLLM
inside the CLI.

Linux-only: terminus_2's TmuxSession requires tmux + asciinema and a
POSIX environment.  Windows sandboxes are explicitly unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig

# Pin matching the cua-verse/harbor agenthle fork. Bump deliberately when
# the fork's terminus_2 / LocalShellEnvironment / CLI shim changes.
HARBOR_FORK_URL = "https://github.com/cua-verse/harbor.git"
HARBOR_FORK_REF = "agenthle"


@dataclass
class Terminus2Config(BaseAgentConfig):
    """Tunables for :class:`Terminus2Deployer`."""

    name: ClassVar[str] = "terminus_2"

    # YAML carries the OpenRouter-native ``vendor/model`` id; for openrouter
    # mode the deployer prepends ``openrouter/`` before handing it to LiteLLM.
    model: str = "anthropic/claude-sonnet-4.6"
    timeout_s: float = 3600.0

    # terminus_2 has no ``unlimited`` sentinel; default high so wall-clock
    # ``timeout_s`` is the real cap.
    max_turns: int | None = 100_000

    # Provider routing: "openrouter" (default) | "direct".
    provider: str = "openrouter"

    # Agent-specific knobs (mapped onto harbor-terminus2 CLI flags).
    record_terminal_session: bool = True
    api_base: str | None = None
    temperature: float = 0.7

    @property
    def litellm_model_id(self) -> str:
        """Model id passed to LiteLLM inside ``harbor-terminus2``.

        LiteLLM needs the explicit ``openrouter/`` prefix to route through
        OpenRouter; for direct providers the bare ``vendor/model`` is what
        LiteLLM expects natively.
        """
        if self.provider == "openrouter" and not self.model.startswith("openrouter/"):
            return f"openrouter/{self.model}"
        return self.model
