"""GrokCliConfig: per-episode knobs for the Grok CLI deployer.

Grok CLI is a standalone binary (no Node.js runtime required for the
CLI itself).  Node IS required for the CUA MCP Server bridge.

Auth: ``GROK_API_KEY`` for direct xAI, ``OPENROUTER_API_KEY`` for
OpenRouter routing (set ``GROK_BASE_URL`` automatically).

Fork bundle: when ``bundle_url`` is set the deployer downloads a
pre-built JS bundle (grok-bundle.js) from a GitHub Release and
launches it via ``node <bundle>`` instead of the stock binary.  The
bundle contains OpenRouter-specific fixes: provider switching, image
injection middleware, disabledTools support, model-info resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig

# Fork ``grok.exe`` (native Windows, self-contained). Built on the win VM
# via ``bun build --compile src/index.ts`` from cua-verse/grok-cli
# ``#agenthle`` — cross-compiling to Windows fails (@opentui externalization)
# so it is built on-target and published as a release asset. Carries the
# same OpenRouter fixes as the linux ``grok-bundle.js`` on the same tag.
_DEFAULT_WIN_BINARY_URL = (
    "https://github.com/cua-verse/grok-cli/releases/download/"
    "v0.1.1-agenthle/grok-x86_64-pc-windows.exe"
)

_NATIVE_TO_OPENROUTER: dict[str, str] = {
    "grok-4-1-fast-reasoning": "x-ai/grok-4.1-fast",
    "grok-4-1-fast-non-reasoning": "x-ai/grok-4.1-fast",
    "grok-4-1-fast": "x-ai/grok-4.1-fast",
    "grok-4-fast-reasoning": "x-ai/grok-4-fast",
    "grok-4-fast-non-reasoning": "x-ai/grok-4-fast",
    "grok-4-fast": "x-ai/grok-4-fast",
    "grok-4-0709": "x-ai/grok-4-0709",
    "grok-code-fast-1": "x-ai/grok-code-fast-1",
    "grok-code-fast": "x-ai/grok-code-fast-1",
    "grok-3": "x-ai/grok-3",
    "grok-3-mini": "x-ai/grok-3-mini",
    "grok-3-mini-fast": "x-ai/grok-3-mini",
}

# 12 computer_* tools — require unpublished agent-desktop binary.
_DISABLED_TOOLS_BASE = (
    "computer_snapshot",
    "computer_screenshot",
    "computer_click",
    "computer_mouse_move",
    "computer_type",
    "computer_press",
    "computer_scroll",
    "computer_launch",
    "computer_list_windows",
    "computer_focus_window",
    "computer_wait",
    "computer_get",
)

# 21 tools total — base 12 + 9 xAI/payment/lsp tools unavailable via
# OpenRouter.  Matches the canonical list in grok_cli_openrouter.yaml.
_DISABLED_TOOLS_OPENROUTER = _DISABLED_TOOLS_BASE + (
    # xAI Responses API (web/X search, image/video generation)
    "search_web",
    "search_x",
    "generate_image",
    "generate_video",
    # xAI payment / wallet (no wallet on benchmark VMs)
    "wallet_info",
    "wallet_history",
    "fetch_payment_info",
    "paid_request",
    # LSP (no language server running on VMs)
    "lsp",
)


def native_to_openrouter_model(model: str) -> str:
    if model.startswith(("x-ai/", "xai/")):
        return model
    return _NATIVE_TO_OPENROUTER.get(model, f"x-ai/{model}")


@dataclass
class GrokCliConfig(BaseAgentConfig):
    """Tunables for :class:`GrokCliDeployer`."""

    name: ClassVar[str] = "grok-cli"

    model: str = "grok-4-1-fast-reasoning"

    # ---- routing (no secrets — API keys come from shell env) ----
    provider: str = "openrouter"
    """Routing provider, drives model mapping + env setup explicitly (not a
    key-presence heuristic):
      - ``"openrouter"`` → native model name mapped to its OpenRouter
        equivalent (:func:`native_to_openrouter_model`), GROK_BASE_URL set
        to openrouter, auth via OPENROUTER_API_KEY. Requires
        OPENROUTER_API_KEY.
      - ``"direct"`` → direct xAI routing with the native model name and
        GROK_API_KEY. Requires GROK_API_KEY.
    Missing the required key for the chosen provider is a hard error."""

    max_tool_rounds: int = 400
    disabled_tools: tuple[str, ...] = _DISABLED_TOOLS_OPENROUTER

    bundle_url: str = ""
    """URL to download the pre-built fork bundle (grok-bundle.js) from a
    GitHub Release.  When non-empty the deployer downloads the bundle and
    launches it via ``bun <bundle_path> --prompt ...`` instead of the
    stock ``grok`` binary.  The bundle uses Bun-specific APIs (bun:sqlite)
    so Bun runtime is auto-installed if missing.  Leave empty to use the
    official CLI without any bundle.

    Linux-only: the bundle is loaded by an external Bun runtime, which
    fails on Windows (the fork's ``@opentui`` deps don't externalize under
    the Windows Bun loader). Windows uses :attr:`win_binary_url` instead —
    a self-contained native ``grok.exe`` compiled from the same fork tree,
    which carries the identical OpenRouter fixes."""

    win_binary_url: str = _DEFAULT_WIN_BINARY_URL
    """Windows-only: URL of the fork ``grok.exe`` (a self-contained native
    binary produced by ``bun build --compile`` from cua-verse/grok-cli
    ``#agenthle``). It carries the same OpenRouter fixes as
    :attr:`bundle_url` — provider switching, image-injection middleware,
    disabledTools, model-info resolution — so headless ``--prompt`` over
    OpenRouter works without ZodErrors. The deployer downloads this on
    Windows and launches it directly (no Bun, no bundle). Empty string =
    skip the fork binary and use the stock ``grok.exe`` (loses the fork
    fixes — only for debugging)."""
