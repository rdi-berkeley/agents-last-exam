"""Tests for Model Config Registry.

Verifies:
  - get_model_config() returns correct config for known models
  - Regex matching works with provider prefixes and variants
  - register_model_config() enables zero-code-change model addition
  - Deleted functions (_is_gpt54, get_screenshot_output_type) are gone
  - Payoff: hypothetical gpt-6 config + sanitize_items produces valid output
"""

import pytest

from ale_run.agents.ale_claw.harness.model.model_config import (
    HelperTransportDefaults,
    ModelConfig,
    ResolvedModel,
    _MODEL_CONFIGS,
    _infer_provider,
    get_model_config,
    register_model_config,
    resolve_model,
)
from ale_run.agents.ale_claw.harness.model.thinking import _is_openai_model


# ---------------------------------------------------------------------------
# Registry lookup tests
# ---------------------------------------------------------------------------


class TestGetModelConfig:
    """get_model_config returns correct config for known model patterns."""

    def test_gpt54(self):
        config = get_model_config("openai/gpt-5.4")
        assert config.tool_schema_type == "computer"
        assert config.screenshot_output_type == "computer_screenshot"
        assert config.supports_safety_checks is False
        assert config.action_format == "batched"
        assert config.adapter_target == "openai-responses"

    def test_gpt54_variant(self):
        """GPT 5.4 turbo or other suffixes still match."""
        config = get_model_config("openai/gpt-5.4-turbo")
        assert config.tool_schema_type == "computer"
        assert config.screenshot_output_type == "computer_screenshot"

    def test_gpt54_case_insensitive(self):
        config = get_model_config("openai/GPT-5.4")
        assert config.tool_schema_type == "computer"

    def test_computer_use_preview(self):
        config = get_model_config("openai/computer-use-preview")
        assert config.tool_schema_type == "computer_use_preview"
        assert config.screenshot_output_type == "input_image"
        assert config.supports_safety_checks is True
        assert config.action_format == "single"
        assert config.adapter_target == "openai-responses"

    def test_anthropic_claude(self):
        """Anthropic models fall through to default config."""
        config = get_model_config("anthropic/claude-sonnet-4-20250514")
        assert config.tool_schema_type == "computer_use_preview"
        assert config.screenshot_output_type == "input_image"
        assert config.supports_safety_checks is True
        assert config.action_format == "single"
        assert config.adapter_target == "anthropic"

    def test_anthropic_opus(self):
        config = get_model_config("anthropic/claude-opus-4-6-20250616")
        assert config.adapter_target == "anthropic"

    def test_unknown_model_defaults_to_anthropic(self):
        config = get_model_config("some/unknown-model")
        assert config.adapter_target == "anthropic"
        assert config.supports_safety_checks is True


# ---------------------------------------------------------------------------
# register_model_config tests
# ---------------------------------------------------------------------------


class TestRegisterModelConfig:
    """register_model_config adds entries that take priority."""

    def setup_method(self):
        """Save original registry state."""
        self._original = list(_MODEL_CONFIGS)

    def teardown_method(self):
        """Restore original registry state."""
        _MODEL_CONFIGS.clear()
        _MODEL_CONFIGS.extend(self._original)

    def test_register_new_model(self):
        custom = ModelConfig(
            tool_schema_type="computer_v2",
            screenshot_output_type="computer_screenshot",
            supports_safety_checks=False,
            action_format="batched",
            adapter_target="openai-responses",
        )
        register_model_config(r"gpt-6", custom)
        config = get_model_config("openai/gpt-6")
        assert config is custom
        assert config.tool_schema_type == "computer_v2"

    def test_registered_config_takes_priority(self):
        """New registration overrides existing patterns."""
        override = ModelConfig(
            tool_schema_type="computer_override",
            screenshot_output_type="input_image",
            supports_safety_checks=True,
            action_format="single",
            adapter_target="openai-responses",
        )
        register_model_config(r"gpt-5\.4", override)
        config = get_model_config("openai/gpt-5.4")
        assert config.tool_schema_type == "computer_override"


class TestResolveModel:
    def setup_method(self):
        self._original = list(_MODEL_CONFIGS)

    def teardown_method(self):
        _MODEL_CONFIGS.clear()
        _MODEL_CONFIGS.extend(self._original)

    def test_openai_runtime_metadata(self):
        resolved = resolve_model("openai/gpt-5.4")
        assert isinstance(resolved, ResolvedModel)
        assert resolved.provider == "openai"
        assert resolved.model_api == "responses"
        assert resolved.transcript_api_label == "openai-responses"
        assert resolved.helper_transport_defaults.memory_flush == "responses"
        assert resolved.helper_transport_defaults.compaction == "chat"

    def test_openrouter_forces_chat_transport(self):
        """OpenRouter models always use 'chat' for all helper transports."""
        resolved = resolve_model("openrouter/openai/gpt-5.4")
        assert resolved.helper_transport_defaults.memory_flush == "chat"
        assert resolved.helper_transport_defaults.compaction == "chat"
        assert resolved.helper_transport_defaults.vision == "chat"

    def test_openrouter_anthropic_also_chat(self):
        resolved = resolve_model("openrouter/anthropic/claude-sonnet-4-20250514")
        assert resolved.helper_transport_defaults.memory_flush == "chat"
        assert resolved.helper_transport_defaults.compaction == "chat"

    def test_custom_provider_can_change_runtime_behavior_through_data_only(self):
        custom = ModelConfig(
            tool_schema_type="computer_use_preview",
            screenshot_output_type="input_image",
            supports_safety_checks=True,
            action_format="single",
            adapter_target="anthropic",
            provider="acme",
            model_api="chat",
            transcript_api_label="acme-chat",
            helper_transport_defaults=HelperTransportDefaults(
                memory_flush="responses",
                compaction="responses",
                vision="chat",
            ),
            context_window=321_000,
        )
        register_model_config(r"acme-ultra", custom)

        resolved = resolve_model("acme/acme-ultra")
        assert resolved.provider == "acme"
        assert resolved.transcript_api_label == "acme-chat"
        assert resolved.helper_transport_defaults.memory_flush == "responses"
        assert resolved.helper_transport_defaults.compaction == "responses"
        assert resolved.context_window == 321_000


# ---------------------------------------------------------------------------
# Deleted functions verification
# ---------------------------------------------------------------------------


class TestDeletedFunctions:
    """_is_gpt54 and get_screenshot_output_type must not exist in openai.py."""

    def test_is_gpt54_deleted(self):
        from agent.loops import openai

        assert not hasattr(openai, "_is_gpt54"), (
            "_is_gpt54 should be deleted from openai.py — replaced by get_model_config()"
        )

    def test_get_screenshot_output_type_deleted(self):
        from agent.loops import openai

        assert not hasattr(openai, "get_screenshot_output_type"), (
            "get_screenshot_output_type should be deleted from openai.py — "
            "replaced by ModelConfig.screenshot_output_type"
        )


# ---------------------------------------------------------------------------
# Payoff test: hypothetical gpt-6 with sanitize_items
# ---------------------------------------------------------------------------


class TestPayoff:
    """Adding a hypothetical gpt-6 config + running sanitize_items with zero code changes."""

    def setup_method(self):
        self._original = list(_MODEL_CONFIGS)

    def teardown_method(self):
        _MODEL_CONFIGS.clear()
        _MODEL_CONFIGS.extend(self._original)

    def test_gpt6_config_with_sanitize_items(self):
        """Register gpt-6, build canonical messages, sanitize — no code changes needed."""
        from ale_run.agents.ale_claw.harness.canonical.canonical import (
            CanonicalMessage,
            FunctionCallBlock,
            TextBlock,
            ToolResultBlock,
            sanitize_items,
        )

        # Register hypothetical gpt-6
        gpt6_config = ModelConfig(
            tool_schema_type="computer_v2",
            screenshot_output_type="computer_screenshot",
            supports_safety_checks=False,
            action_format="batched",
            adapter_target="openai-responses",
        )
        register_model_config(r"gpt-6", gpt6_config)

        # Verify config resolves
        config = get_model_config("openai/gpt-6")
        assert config is gpt6_config

        # Build canonical messages and sanitize to the config's target
        messages = [
            CanonicalMessage(
                role="user",
                content=[TextBlock(type="text", text="Hello")],
            ),
            CanonicalMessage(
                role="assistant",
                content=[
                    TextBlock(type="text", text="I'll help."),
                    FunctionCallBlock(
                        type="function_call",
                        id="call_123",
                        name="memory_search",
                        arguments='{"query": "test"}',
                    ),
                ],
            ),
            CanonicalMessage(
                role="tool",
                content=[
                    ToolResultBlock(
                        type="tool_result",
                        tool_use_id="call_123",
                        content="No results found.",
                    ),
                ],
            ),
        ]

        # sanitize_items uses the config's adapter_target
        result = sanitize_items(messages, target=config.adapter_target)

        # Verify output is valid Responses API items
        assert isinstance(result, list)
        assert len(result) > 0

        # Verify structure: user message, assistant message, function_call, function_call_output
        types = [item.get("type") for item in result]
        assert "message" in types
        assert "function_call" in types
        assert "function_call_output" in types


# ---------------------------------------------------------------------------
# ModelConfig is frozen (immutable)
# ---------------------------------------------------------------------------


class TestModelConfigImmutable:
    def test_frozen(self):
        config = get_model_config("openai/gpt-5.4")
        with pytest.raises(AttributeError):
            config.tool_schema_type = "something_else"


# ---------------------------------------------------------------------------
# Provider inference — OpenAI detection (regression for the over-broad
# `startswith("o")` heuristic that tagged any `o*` string as OpenAI)
# ---------------------------------------------------------------------------


class TestOpenAIProviderDetection:
    """`_infer_provider` / `_is_openai_model` must key off the real provider
    segment, not the leading letter. The OpenRouter slug for o3 is
    `openrouter/openai/o3` — its `openai` segment is mid-string, so a
    `startswith("openai/")`/`startswith("o")` pair both mis-handled it: the
    former missed it, the latter "caught" it only via the `o` in `openrouter`
    while also mis-tagging `openrouter/google/*` and `ollama/*` as OpenAI.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "openrouter/openai/o3",
            "openrouter/openai/o3-mini",
            "openrouter/openai/gpt-5.4",
            "openai/o3",
            "gpt-4o",
            "o3-mini",  # bare o-series (no provider token) — narrow prefix keeps these
            "o1",
            "o4-mini",
        ],
    )
    def test_openai_models_detected(self, model):
        assert _infer_provider(model) == "openai"
        assert _is_openai_model(model.lower()) is True

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("openrouter/google/gemini-2.5", "google"),
            ("openrouter/anthropic/claude-sonnet-4.6", "anthropic"),
            ("ollama/llama3", "unknown"),
            ("openchat/openchat-7b", "unknown"),
        ],
    )
    def test_non_openai_not_mistagged(self, model, expected):
        assert _infer_provider(model) == expected
        assert _is_openai_model(model.lower()) is False

    def test_openrouter_o3_resolves_to_openai_provider(self):
        # End-to-end through resolve_model (config.provider is unset for this
        # slug, so _infer_provider is the deciding path).
        assert resolve_model("openrouter/openai/o3").provider == "openai"
