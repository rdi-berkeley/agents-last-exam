"""CursorCliConfig: per-episode knobs for the Cursor agent CLI deployer.

Auth: ``CURSOR_API_KEY`` — Cursor backend key (BYOK is blocked;
OpenRouter routing is not supported by cursor-agent).

Model IDs use Cursor's catalog names (e.g. ``claude-4.6-sonnet-medium``,
``claude-opus-4-7-thinking-high``, ``gpt-5.5-high``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class CursorCliConfig:
    """Tunables for :class:`CursorCliDeployer`.

    Standalone config (no shared base). The episode wall-budget is
    orchestration-owned; ``timeout_s`` is no longer an agent knob.

    Note: agenthle cursor_cli.yaml also carried ``max_turns: 300``, but
    cursor-agent exposes no ``--max-turns`` flag, so that value was dead
    in agenthle and is not carried here (the deployer never read it).
    """

    name: ClassVar[str] = "cursor-cli"

    # Default = "" ("auto"): the deployer omits --model so cursor-agent picks
    # its own Composer model. agenthle's cursor_cli.yaml pinned the catalog name
    # ``claude-4.6-sonnet-medium``, but this migration deliberately runs "auto"
    # because the pinned Sonnet catalog name hits per-model Cursor ACCOUNT
    # QUOTAS (validated: pinned Sonnet failed on quota; auto passes). Keep "".
    model: str = ""
    """Cursor catalog model id (e.g. ``claude-4.6-sonnet-medium``,
    ``claude-opus-4-7-thinking-high``, ``gpt-5.5-high``, ``composer-2.5``).
    Empty string = "auto" — the deployer omits ``--model`` and cursor-agent
    picks its own Composer model."""

    composer_fast: bool = False
    """Composer tier selector (Composer models only). ``False`` = **Standard**
    (model id ``composer-2.5``); ``True`` = **Fast** (``composer-2.5-fast``).
    Both serve the SAME weights — Fast is merely inference on hotter hardware
    (lower latency / higher tok-per-sec) at ~6× the per-token price
    ($3/$15 vs $0.50/$2.50 per M) with no quality gain, so **Standard is the
    right default for batch / non-interactive runs**. When ``True`` the deployer
    appends ``-fast`` to a bare ``composer-*`` model id; non-composer models
    and ids already ending in ``-fast`` are left untouched."""

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
