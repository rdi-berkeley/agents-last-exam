"""OpenClawCliConfig: per-episode knobs for the OpenClaw CLI deployer.

OpenClaw is installed from a fork tarball (not public npm).
CUA bridge is the native OpenClaw plugin (not MCP).

Auth: API keys set via auth-profiles.json (OpenRouter or direct).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig

CUA_TOOL_NAMES = (
    "screenshot",
    "click",
    "type",
    "key",
    "key_down",
    "key_up",
    "hold_key",
    "mouse_move",
    "mouse_down",
    "mouse_up",
    "drag",
    "scroll",
    "wait",
    "cursor_position",
)

# Mirrors the 12-entry deny list shipped in every agenthle
# openclaw_*_openrouter*.yaml. The first seven are backend / late-binding
# tools whose core impl reaches for a plugin we don't load (parity with the
# gateway runner); the last five are gateway-only tools that raise
# `1006 abnormal closure` under `agent --local` because there is no live
# gateway WebSocket. Verified OS-agnostic on the 2026-05-03
# demo_tool_smoke_test smoke matrix.
_TOOLS_DENY = (
    # backend / late-binding (parity with gateway runner)
    "web_search",
    "web_fetch",
    "image_generate",
    "video_generate",
    "music_generate",
    "memory_search",
    "sessions_yield",
    # gateway-only tools — `agent --local` has no live gateway WS
    "sessions_list",
    "sessions_history",
    "sessions_spawn",
    "sessions_send",
    "cron",
)


@dataclass
class OpenClawCliConfig(BaseAgentConfig):
    """Tunables for :class:`OpenClawCliDeployer`."""

    name: ClassVar[str] = "openclaw-cli"

    model: str = "openai/gpt-5.4"

    # ---- routing (no secrets — API keys come from shell env) ----
    provider: str = "openrouter"
    """Routing provider, drives auth-profile + model-prefix setup
    explicitly (not a key-presence heuristic):
      - ``"openrouter"`` (default) → openrouter auth profile via
        OPENROUTER_API_KEY; model is prefixed ``openrouter/<model>``.
        Requires OPENROUTER_API_KEY.
      - ``"direct"`` → native-provider auth profile selected by the
        model's vendor: an OpenAI model (``gpt-*`` / ``openai/...``) uses
        the ``openai`` provider + OPENAI_API_KEY; an Anthropic model
        (``claude-*`` / ``anthropic/...``) uses the ``anthropic`` provider
        + ANTHROPIC_API_KEY. OPENROUTER_API_KEY is dropped from the launch
        env so it cannot override the chosen direct provider.
    Missing the required key for the chosen provider is a hard error."""

    thinking: str = "high"
    vision_model: str | None = None
    tools_deny: tuple[str, ...] = _TOOLS_DENY
    # Matches the agenthle openclaw_*_openrouter*.yaml plugins.allow: the
    # `memory-core` plugin provides the harmless `memory_get` file reader
    # (`memory_search` is denied above). The resolved auth provider
    # (`openrouter` / `openai` / `anthropic`) is appended at write time by
    # the deployer, so it need not be listed here.
    plugins_allow: tuple[str, ...] = ("cua", "openrouter", "memory-core")
    plugins_deny: tuple[str, ...] = ()
    heartbeat_every: str = "never"

    tarball_path: str = "/opt/ale/openclaw-fork.tgz"
    """Path to openclaw fork tarball inside the sandbox."""

    tarball_url: str = ""
    """GitHub Release URL for the fork tarball. Used as fallback when
    tarball_path does not exist on disk."""

    cua_plugin_path: str = "/opt/ale/openclaw-cua-plugin"
    """Path to CUA plugin source directory inside the sandbox."""

    cua_plugin_repo: str = "https://github.com/cua-verse/openclaw.git"
    """Git URL to clone CUA plugin source from when cua_plugin_path is missing."""

    cua_plugin_branch: str = "agenthle"
    """Branch to clone for CUA plugin source."""
