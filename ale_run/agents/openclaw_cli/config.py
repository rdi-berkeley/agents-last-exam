"""OpenClawCliConfig: per-episode knobs for the OpenClaw CLI deployer.

OpenClaw is installed from a fork tarball (not public npm).
CUA bridge is the native OpenClaw plugin (not MCP).

Auth: API keys set via auth-profiles.json (OpenRouter or direct).
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

_TOOLS_DENY = (
    "web_search",
    "web_fetch",
    "image_generate",
    "video_generate",
    "music_generate",
    "memory_search",
    "sessions_yield",
)


@dataclass
class OpenClawCliConfig(BaseAgentConfig):
    """Tunables for :class:`OpenClawCliDeployer`."""

    name: ClassVar[str] = "openclaw-cli"

    model: str = "openai/gpt-5.4"
    thinking: str = "high"
    vision_model: str | None = None
    tools_deny: tuple[str, ...] = _TOOLS_DENY
    plugins_allow: tuple[str, ...] = ("cua", "openrouter", "openai")
    plugins_deny: tuple[str, ...] = ()
    heartbeat_every: str = "never"

    tarball_path: str = "/opt/ale/openclaw-fork.tgz"
    """Path to openclaw fork tarball inside the sandbox."""

    cua_plugin_path: str = "/opt/ale/openclaw-cua-plugin"
    """Path to CUA plugin source directory inside the sandbox."""
