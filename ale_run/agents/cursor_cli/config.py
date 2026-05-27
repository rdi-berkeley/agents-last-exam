"""CursorCliConfig: per-episode knobs for the Cursor agent CLI deployer.

Auth: ``CURSOR_API_KEY`` — Cursor backend key (BYOK is blocked;
OpenRouter routing is not supported by cursor-agent).

Model IDs use Cursor's catalog names (e.g. ``claude-4.6-sonnet-medium``,
``claude-opus-4-7-thinking-high``, ``gpt-5.5-high``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig


@dataclass
class CursorCliConfig(BaseAgentConfig):
    """Tunables for :class:`CursorCliDeployer`."""

    name: ClassVar[str] = "cursor-cli"

    model: str = "claude-4.6-sonnet-medium"
    disabled_tools: tuple[str, ...] = ()
    """Permission deny patterns for ``cli-config.json``.
    Supports: ``Shell(...)``, ``Read(...)``, ``Write(...)``,
    ``WebFetch(...)``, ``Mcp(server, tool)``."""
