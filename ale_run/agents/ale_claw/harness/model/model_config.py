"""Model configuration registry — declarative format metadata per model variant.

Maps model identifiers to format metadata (tool schema type, screenshot format,
safety check support, action format, adapter target) so that adding a new model
variant requires one config entry and zero code changes.

Design reference:
  - OpenClaw's model.ts resolveModel pattern (provider catalog with format metadata)
  - CUA's @register_agent decorator (regex-based model matching)

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Declarative format metadata for a model variant.

    Fields:
        tool_schema_type: OpenAI tool schema type sent to the API.
            "computer" (GPT 5.4) or "computer_use_preview" (legacy).
        screenshot_output_type: Image type in computer_call_output items.
            "computer_screenshot" (GPT 5.4, with detail="original") or
            "input_image" (computer-use-preview and Anthropic).
        supports_safety_checks: Whether to include acknowledged_safety_checks
            in computer_call_output. False for GPT 5.4.
        action_format: "batched" (GPT 5.4 actions array) or "single"
            (computer-use-preview singular action).
        adapter_target: Provider format for sanitize_items() conversion.
            "openai-responses" or "anthropic".
    """

    tool_schema_type: str
    screenshot_output_type: str
    supports_safety_checks: bool
    action_format: str
    adapter_target: str
    provider: str | None = None
    model_api: str | None = None
    transcript_api_label: str | None = None
    helper_transport_defaults: "HelperTransportDefaults | None" = None
    context_window: int | None = None


@dataclass(frozen=True)
class HelperTransportDefaults:
    """Default helper transport modes for a resolved model."""

    memory_flush: Literal["responses", "chat"] = "chat"
    compaction: Literal["responses", "chat"] = "chat"
    vision: Literal["responses", "chat"] = "chat"

    def for_purpose(
        self,
        purpose: Literal["memory_flush", "compaction", "vision"],
    ) -> Literal["responses", "chat"]:
        if purpose == "memory_flush":
            return self.memory_flush
        if purpose == "vision":
            return self.vision
        return self.compaction


@dataclass(frozen=True)
class ResolvedModel:
    """Capability-aware resolved model metadata for one runtime model string."""

    model: str
    model_id: str
    provider: str
    model_api: str
    adapter_target: str
    tool_schema_type: str
    screenshot_output_type: str
    supports_safety_checks: bool
    action_format: str
    transcript_api_label: str
    helper_transport_defaults: HelperTransportDefaults
    context_window: int | None = None


# ---------------------------------------------------------------------------
# Registry: ordered list of (compiled_regex, ModelConfig) — first match wins.
# ---------------------------------------------------------------------------

_MODEL_CONFIGS: List[Tuple[re.Pattern, ModelConfig]] = [
    (
        re.compile(r"gpt-5\.4", re.IGNORECASE),
        ModelConfig(
            tool_schema_type="computer",
            screenshot_output_type="computer_screenshot",
            supports_safety_checks=False,
            action_format="batched",
            adapter_target="openai-responses",
        ),
    ),
    (
        re.compile(r"computer-use-preview", re.IGNORECASE),
        ModelConfig(
            tool_schema_type="computer_use_preview",
            screenshot_output_type="input_image",
            supports_safety_checks=True,
            action_format="single",
            adapter_target="openai-responses",
        ),
    ),
]

# Default config for models that don't match any pattern (Anthropic, etc.)
_DEFAULT_CONFIG = ModelConfig(
    tool_schema_type="computer_use_preview",
    screenshot_output_type="input_image",
    supports_safety_checks=True,
    action_format="single",
    adapter_target="anthropic",
)


def get_model_config(model: str) -> ModelConfig:
    """Look up model config by matching model string against registry patterns.

    Searches ``_MODEL_CONFIGS`` in order; returns the first match.  Falls back
    to ``_DEFAULT_CONFIG`` (Anthropic-compatible) if no pattern matches.

    Args:
        model: litellm model identifier (e.g. "openai/gpt-5.4", "anthropic/claude-sonnet-4-20250514").
    """
    for pattern, config in _MODEL_CONFIGS:
        if pattern.search(model):
            return config
    return _DEFAULT_CONFIG


def resolve_model(model: str | ResolvedModel) -> ResolvedModel:
    """Resolve a model string into structured runtime metadata."""
    if isinstance(model, ResolvedModel):
        return model

    config = get_model_config(model)
    provider = config.provider or _infer_provider(model)
    model_api = config.model_api or _infer_model_api(config, provider)
    helper_transport_defaults = (
        config.helper_transport_defaults
        or _default_helper_transports(provider, model)
    )
    transcript_api_label = (
        config.transcript_api_label
        or _default_transcript_api_label(provider, model_api)
    )

    return ResolvedModel(
        model=model,
        model_id=model.split("/", 1)[-1] if "/" in model else model,
        provider=provider,
        model_api=model_api,
        adapter_target=config.adapter_target,
        tool_schema_type=config.tool_schema_type,
        screenshot_output_type=config.screenshot_output_type,
        supports_safety_checks=config.supports_safety_checks,
        action_format=config.action_format,
        transcript_api_label=transcript_api_label,
        helper_transport_defaults=helper_transport_defaults,
        context_window=config.context_window or _lookup_context_window(model),
    )


def register_model_config(pattern: str, config: ModelConfig) -> None:
    """Register a new model config at the front of the registry.

    New entries take priority over existing ones (prepended to list).
    Useful for adding model support at runtime or in tests.

    Args:
        pattern: Regex pattern to match model strings.
        config: ModelConfig for matching models.
    """
    _MODEL_CONFIGS.insert(0, (re.compile(pattern, re.IGNORECASE), config))


def _infer_provider(model: str) -> str:
    model_lower = model.lower()
    if model_lower.startswith("anthropic/") or "claude" in model_lower:
        return "anthropic"
    if (
        "openai" in model_lower
        or "gpt" in model_lower
        or model_lower.startswith(("o1", "o3", "o4"))
    ):
        return "openai"
    if "gemini" in model_lower or "google" in model_lower:
        return "google"
    if "vertex" in model_lower:
        return "vertex"
    return "unknown"


def _infer_model_api(config: ModelConfig, provider: str) -> str:
    if config.adapter_target == "openai-responses" or provider == "openai":
        return "responses"
    return "chat"


def _default_helper_transports(
    provider: str, model: str = ""
) -> HelperTransportDefaults:
    # OpenRouter routes everything through Chat Completions — always use "chat".
    if model.lower().startswith("openrouter/"):
        return HelperTransportDefaults()
    if provider == "openai":
        return HelperTransportDefaults(memory_flush="responses")
    return HelperTransportDefaults()


def _default_transcript_api_label(provider: str, model_api: str) -> str:
    if provider == "openai" and model_api == "responses":
        return "openai-responses"
    if provider in {"anthropic", "google", "vertex"}:
        return provider
    return provider if provider != "unknown" else model_api


def _lookup_context_window(model: str) -> int | None:
    for candidate in _model_candidates(model):
        try:
            import litellm

            info = litellm.get_model_info(candidate)
            max_input = info.get("max_input_tokens")
            if max_input and max_input > 0:
                return int(max_input)
        except Exception:
            continue
    return None


def _model_candidates(model: str) -> list[str]:
    candidates = [model]
    if "/" in model:
        candidates.append(model.split("/", 1)[1])
    return candidates
