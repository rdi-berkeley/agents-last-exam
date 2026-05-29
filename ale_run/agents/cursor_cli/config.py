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

    model: str = ""
    """Empty string = "auto" (cursor-agent picks the model; currently
    resolves to Cursor's own Composer model). An explicit Cursor catalog
    name (e.g. ``claude-4.6-sonnet-medium``) pins the model, but those are
    subject to per-model account quotas. Default is auto for benchmark
    runs; the deployer omits ``--model`` when this is empty."""

    provider: str = "cursor"
    """Routing provider. cursor-agent is hard-wired to Cursor's own
    backend (``CURSOR_API_KEY``) — BYOK and OpenRouter routing are not
    supported, so this is fixed to ``"cursor"`` (matching agenthle's
    cursor_cli config). The deployer does not branch on it; it exists for
    parity and to make the unsupported-routing fact explicit."""
    cursor_version: str = "2026.05.28-a70ca7c"
    """Pinned cursor-agent version (Cursor's date-hash scheme). The
    deployer verifies any pre-installed binary matches this and otherwise
    installs it from
    ``https://downloads.cursor.com/lab/<version>/<os>/<arch>/agent-cli-package.tar.gz``
    so all environments converge on one version. ``cursor.com/install``
    is latest-only and cannot pin, so it is used only as a fallback."""
    disabled_tools: tuple[str, ...] = ()
    """Permission deny patterns for ``cli-config.json``.
    Supports: ``Shell(...)``, ``Read(...)``, ``Write(...)``,
    ``WebFetch(...)``, ``Mcp(server, tool)``."""
