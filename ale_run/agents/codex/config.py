"""CodexConfig: per-episode knobs for the OpenAI Codex CLI deployer.

Codex CLI is installed from NPM (``@openai/codex@<version>``). An
optional patched native binary (from a GitHub Release URL) replaces the
vendor binary to fix the Windows ``apply_patch.bat`` corruption bug.

Auth: OpenRouter routing uses ``OPENROUTER_API_KEY`` injected via env.
Direct OpenAI routing uses ``OPENAI_API_KEY``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ale_run.base_interface import BaseAgentConfig

# Default NPM package version for the Codex CLI.
_DEFAULT_CODEX_VERSION = "0.114.0"

# GitHub Release URLs for the patched native binary, per OS. When set and
# reachable, the deployer downloads the matching binary and overwrites the
# npm-installed vendor binary. When empty the binary-replacement step is
# silently skipped. Linux ships ``codex`` (musl x86-64); Windows ships
# ``codex-x86_64-pc-windows-msvc.exe`` — separate release assets because a
# single binary can't cover both targets. Both are built from
# cua-verse/codex @ ee3e51688 (agenthle): the OpenRouter MCP adaptation,
# plus — on Windows only — the apply_patch.exe-hardlink fix.
_RELEASE_BASE = (
    "https://github.com/cua-verse/codex/releases/download/v0.114.0-agenthle"
)
_DEFAULT_PATCHED_BINARY_URL = f"{_RELEASE_BASE}/codex"
_DEFAULT_PATCHED_BINARY_URL_WINDOWS = (
    f"{_RELEASE_BASE}/codex-x86_64-pc-windows-msvc.exe"
)


@dataclass
class CodexConfig(BaseAgentConfig):
    """Tunables for :class:`CodexDeployer`."""

    name: ClassVar[str] = "codex"

    model: str = "openai/gpt-5.4"
    timeout_s: float = 600

    # ---- routing (no secrets — API keys come from shell env) ----
    provider: str = "openrouter"
    """Routing provider, drives env + config.toml setup explicitly (not a
    model-name heuristic):
      - ``"openrouter"`` → config.toml ``model_provider = "openrouter"`` +
        openrouter model_providers block, auth via OPENROUTER_API_KEY.
        Requires OPENROUTER_API_KEY.
      - ``"direct"`` → direct OpenAI routing via OPENAI_API_KEY.
        Requires OPENAI_API_KEY.
    Missing the required key for the chosen provider is a hard error."""

    # Codex sandbox policy: "danger-full-access" is the only meaningful
    # option for headless eval on an already-isolated VM.
    sandbox_mode: str = "danger-full-access"

    # Bypass all interactive approval prompts (headless exec).
    yolo: bool = True

    # Codex CLI's model_reasoning_effort. Codex 0.114 sends this through
    # the Responses-API wire as ``reasoning.effort``.
    reasoning_effort: str = "high"

    # NPM package version to install.
    codex_version: str = _DEFAULT_CODEX_VERSION

    # GitHub Release URL for the patched native binary (Linux musl x86-64).
    # Empty string = skip binary replacement (use npm's bundled binary).
    patched_binary_url: str = _DEFAULT_PATCHED_BINARY_URL

    # GitHub Release URL for the patched Windows binary (codex.exe,
    # x86_64-pc-windows-msvc). The deployer downloads this instead of
    # ``patched_binary_url`` when running on Windows. Empty string = skip
    # replacement on Windows (use npm's bundled codex.exe).
    patched_binary_url_windows: str = _DEFAULT_PATCHED_BINARY_URL_WINDOWS
