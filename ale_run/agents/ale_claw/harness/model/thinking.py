"""OpenClaw thinking level configuration and provider-specific parameter mapping.

Reproduces OpenClaw's ThinkLevel system (openclaw/src/auto-reply/thinking.ts) and
provider-specific parameter mapping (openclaw/src/agents/pi-embedded-runner/extra-params.ts).

ThinkLevel controls reasoning depth across providers. Each provider maps levels
to its own API format:
  - Anthropic: thinking={type: enabled, budget_tokens: N}
  - OpenAI: reasoning={effort: low|medium|high, summary: concise}
  - Gemini: thinking_level=MINIMAL|LOW|MEDIUM|HIGH (CUA gemini loop kwarg)
  - Fallback: reasoning_effort=<level>

Thinking blocks are kept in context by default (matches OpenClaw). dropThinkingBlocks()
is not implemented — only needed for specific providers (GitHub Copilot Claude).

Future work:
  - VerboseLevel, ReasoningLevel, ElevatedLevel — additional control dimensions
  - dropThinkingBlocks() — conditional per-provider stripping
  - Per-turn thinking level overrides

Reference:
  - openclaw/src/auto-reply/thinking.ts — ThinkLevel type
  - openclaw/src/agents/model-selection.ts — resolveThinkingDefault()
  - openclaw/src/agents/pi-embedded-runner/extra-params.ts — provider mappings
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from .model_config import ResolvedModel, resolve_model


class ThinkLevel(str, Enum):
    """Thinking level — controls reasoning depth.

    Matches OpenClaw's ThinkLevel type (openclaw/src/auto-reply/thinking.ts).
    """

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    ADAPTIVE = "adaptive"


@dataclass
class ThinkingConfig:
    """Per-run thinking configuration with per-call-site levels.

    Attributes:
        level: Thinking level for the main agent loop.
        flush_level: Thinking level for memory flush LLM calls.
        compaction_level: Thinking level for compaction summarization calls.
        vision_level: Thinking level for vision helper calls. Defaults to off unless
            explicitly configured, so VLM analysis does not opt into helper thinking
            by inheritance.
    """

    level: ThinkLevel = ThinkLevel.OFF
    flush_level: ThinkLevel = ThinkLevel.OFF
    compaction_level: ThinkLevel = ThinkLevel.OFF
    vision_level: ThinkLevel = ThinkLevel.OFF
    gui_level: ThinkLevel = ThinkLevel.OFF

    def to_api_params(self, model: str) -> dict[str, Any]:
        """Return provider-specific kwargs for the main agent loop."""
        return resolve_thinking_params(self.level, model, transport="responses")

    def flush_params(
        self, model: str, *, runtime: ResolvedModel | None = None
    ) -> dict[str, Any]:
        """Return provider-specific kwargs for memory flush helper calls."""
        resolved = runtime or resolve_model(model)
        transport = resolved.helper_transport_defaults.memory_flush
        return resolve_thinking_params(self.flush_level, model, transport=transport)

    def compaction_params(
        self, model: str, *, runtime: ResolvedModel | None = None
    ) -> dict[str, Any]:
        """Return provider-specific kwargs for compaction summarization calls."""
        resolved = runtime or resolve_model(model)
        return resolve_thinking_params(
            self.compaction_level,
            model,
            transport=resolved.helper_transport_defaults.compaction,
        )

    def vision_params(
        self, model: str, *, runtime: ResolvedModel | None = None
    ) -> dict[str, Any]:
        """Return provider-specific kwargs for vision helper calls."""
        resolved = runtime or resolve_model(model)
        return resolve_thinking_params(
            self.vision_level,
            model,
            transport=resolved.helper_transport_defaults.vision,
        )

    def gui_params(self, model: str) -> dict[str, Any]:
        """Return provider-specific kwargs for the GUI subagent's ComputerAgent."""
        return resolve_thinking_params(self.gui_level, model, transport="chat")


# ---------------------------------------------------------------------------
# Model capability detection
# ---------------------------------------------------------------------------

# Patterns for models that support thinking natively.
# Based on OpenClaw's resolveThinkingDefault() in model-selection.ts.
_CLAUDE_46_PATTERN = re.compile(
    r"claude-(opus|sonnet)-4[._-]6|claude-opus-4[._-]5", re.IGNORECASE
)
_CLAUDE_4_PATTERN = re.compile(
    r"claude-4|claude-sonnet-4|claude-opus-4|claude-haiku-4", re.IGNORECASE
)
_REASONING_MODEL_PATTERNS = [
    re.compile(r"deepseek.*r1", re.IGNORECASE),
    re.compile(r"kimi.*k2", re.IGNORECASE),
    re.compile(r"qwq", re.IGNORECASE),
    re.compile(r"o[134]-", re.IGNORECASE),  # OpenAI o1, o3, o4 series
]


def resolve_thinking_default(model: str) -> ThinkLevel:
    """Auto-detect default thinking level based on model capabilities.

    Based on OpenClaw's resolveThinkingDefault() in model-selection.ts:
      - Claude 4.6 (Opus/Sonnet 4.6, Opus 4.5) → adaptive
      - Claude 4 family (Sonnet 4, Haiku 4.5, etc.) → low
      - Reasoning models (DeepSeek R1, Kimi K2, QwQ, OpenAI o-series) → low
      - Others → off

    Args:
        model: litellm model string (e.g. "anthropic/claude-sonnet-4-20250514").
    """
    # Strip provider prefix for pattern matching
    model_name = model.split("/", 1)[-1] if "/" in model else model

    if _CLAUDE_46_PATTERN.search(model_name):
        return ThinkLevel.ADAPTIVE

    if _CLAUDE_4_PATTERN.search(model_name):
        return ThinkLevel.LOW

    for pattern in _REASONING_MODEL_PATTERNS:
        if pattern.search(model_name):
            return ThinkLevel.LOW

    return ThinkLevel.OFF


# ---------------------------------------------------------------------------
# Provider-specific parameter mapping
# ---------------------------------------------------------------------------

# Anthropic budget mapping per thinking level.
# Based on OpenClaw's extra-params.ts Anthropic section.
_ANTHROPIC_BUDGETS: dict[ThinkLevel, int] = {
    ThinkLevel.MINIMAL: 2000,
    ThinkLevel.LOW: 5000,
    ThinkLevel.MEDIUM: 10000,
    ThinkLevel.HIGH: 16000,
    ThinkLevel.XHIGH: 25000,
    ThinkLevel.ADAPTIVE: 10000,
}

# OpenAI reasoning effort mapping.
# OpenAI only supports low/medium/high — we collapse our 7 levels.
_OPENAI_EFFORT: dict[ThinkLevel, str] = {
    ThinkLevel.MINIMAL: "low",
    ThinkLevel.LOW: "low",
    ThinkLevel.MEDIUM: "medium",
    ThinkLevel.HIGH: "high",
    ThinkLevel.XHIGH: "high",
    ThinkLevel.ADAPTIVE: "medium",
}

# Gemini thinking level mapping (CUA gemini loop handles the kwarg).
_GEMINI_LEVELS: dict[ThinkLevel, str] = {
    ThinkLevel.MINIMAL: "MINIMAL",
    ThinkLevel.LOW: "LOW",
    ThinkLevel.MEDIUM: "MEDIUM",
    ThinkLevel.HIGH: "HIGH",
    ThinkLevel.XHIGH: "HIGH",
    ThinkLevel.ADAPTIVE: "MEDIUM",
}


def resolve_thinking_params(
    level: ThinkLevel,
    model: str,
    *,
    transport: Literal["responses", "chat"] = "responses",
) -> dict[str, Any]:
    """Map ThinkLevel to provider-specific API params.

    Based on OpenClaw's extra-params.ts provider mappings. Returns a dict
    that can be spread into litellm.acompletion() or ComputerAgent kwargs.

    Args:
        level: Desired thinking level.
        model: litellm model string (e.g. "anthropic/claude-sonnet-4-20250514").
        transport: Call transport. ``responses`` is the main OpenAI runtime;
            ``chat`` covers helper ``litellm.acompletion()`` paths such as
            memory flush and compaction summarization.

    Returns:
        Dict of provider-specific kwargs, or empty dict if level is OFF.
    """
    if level == ThinkLevel.OFF:
        return {}

    model_lower = model.lower()

    # OpenRouter: unified reasoning param for all providers.
    # OpenRouter translates effort levels to provider-specific formats internally.
    if model_lower.startswith("openrouter/"):
        return _openrouter_params(level)

    # Anthropic models (direct API)
    if "anthropic/" in model_lower or "claude" in model_lower:
        return _anthropic_params(level)

    # OpenAI models (direct API)
    if _is_openai_model(model_lower):
        return _openai_params(level, transport=transport)

    # Gemini / Google models
    if "gemini" in model_lower or "google" in model_lower or "vertex" in model_lower:
        return _gemini_params(level)

    # Fallback: generic reasoning_effort (some litellm providers support this)
    return {"reasoning_effort": level.value}


def _openrouter_params(level: ThinkLevel) -> dict[str, Any]:
    """OpenRouter: reasoning={effort: level} — unified for all providers.

    OpenRouter translates effort levels to provider-specific formats internally
    (e.g. budget_tokens for Anthropic, reasoning_effort for OpenAI).
    """
    effort = _OPENAI_EFFORT.get(level, "medium")
    return {"reasoning": {"effort": effort}}


def _anthropic_params(level: ThinkLevel) -> dict[str, Any]:
    """Anthropic (direct API): thinking={type: enabled, budget_tokens: N}."""
    budget = _ANTHROPIC_BUDGETS.get(level, 10000)
    return {"thinking": {"type": "enabled", "budget_tokens": budget}}


def _openai_params(
    level: ThinkLevel,
    *,
    transport: Literal["responses", "chat"] = "responses",
) -> dict[str, Any]:
    """OpenAI param mapping by transport.

    The main agent loop uses the Responses API, which accepts ``reasoning``.
    Helper paths use ``litellm.acompletion()``, which accepts LiteLLM's
    ``reasoning_effort`` kwarg instead of Responses-style ``reasoning``.
    """
    effort = _OPENAI_EFFORT.get(level, "medium")
    if transport != "responses":
        return {"reasoning_effort": effort}
    return {"reasoning": {"effort": effort, "summary": "concise"}}


def _is_openai_model(model: str) -> bool:
    """Heuristic for OpenAI-family models used by the harness."""
    model_lower = model.lower()
    return (
        "openai" in model_lower
        or "gpt" in model_lower
        or model_lower.startswith(("o1", "o3", "o4"))
    )


def _gemini_params(level: ThinkLevel) -> dict[str, Any]:
    """Gemini: thinking_level kwarg (handled by CUA's gemini loop)."""
    thinking_level = _GEMINI_LEVELS.get(level, "MEDIUM")
    return {"thinking_level": thinking_level}
