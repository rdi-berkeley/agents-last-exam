"""Tests for context overflow detection and tool result truncation.

Covers:
  - Token estimation (chars/4, image substitution)
  - Context window resolution (litellm, fallback)
  - Tool result truncation (head-only, head+tail, under-limit passthrough)
  - Reactive error detection (true positives, rate limit exclusions, empty input)
  - ContextOverflowCallback (threshold, properties, turn counting, force_compaction)
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from ale_run.agents.ale_claw.harness.context.context import (
    DEFAULT_CONTEXT_TOKENS,
    FIXED_IMAGE_TOKENS,
    HARD_MAX_TOOL_RESULT_CHARS,
    MAX_TOOL_RESULT_SHARE,
    MIDDLE_OMISSION_MARKER,
    MIN_KEEP_CHARS,
    SAFETY_MARGIN,
    TRUNCATION_SUFFIX,
    ContextOverflowCallback,
    _calculate_max_tool_result_chars,
    estimate_message_tokens,
    estimate_messages_tokens,
    has_important_tail,
    is_context_overflow_error,
    resolve_context_window,
    truncate_tool_result_text,
    truncate_tool_results,
)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

class TestEstimateMessageTokens:
    def test_chars_div_4(self):
        msg = {"role": "user", "content": "hello world"}
        raw = json.dumps(msg, separators=(",", ":"))
        expected = len(raw) // 4
        assert estimate_message_tokens(msg) == expected

    def test_empty_message(self):
        msg = {}
        assert estimate_message_tokens(msg) == len("{}") // 4

    def test_image_substitution(self):
        """Base64 image data should be replaced with FIXED_IMAGE_TOKENS."""
        # Build a message with a large fake base64 image
        b64_data = "A" * 10_000
        img_url = f"data:image/png;base64,{b64_data}"
        msg = {
            "type": "computer_call_output",
            "output": {"type": "input_image", "image_url": img_url},
        }
        tokens = estimate_message_tokens(msg)
        raw = json.dumps(msg, separators=(",", ":"))
        naive_tokens = len(raw) // 4
        # Image substitution should significantly reduce the count
        assert tokens < naive_tokens, "Image substitution should reduce token count"
        assert tokens >= FIXED_IMAGE_TOKENS, "Should include at least FIXED_IMAGE_TOKENS"

    def test_multiple_images(self):
        """Multiple images each contribute FIXED_IMAGE_TOKENS."""
        b64_data = "B" * 500
        img1 = f"data:image/png;base64,{b64_data}"
        img2 = f"data:image/jpeg;base64,{b64_data}"
        msg = {"images": [img1, img2]}
        tokens = estimate_message_tokens(msg)
        assert tokens >= FIXED_IMAGE_TOKENS * 2

    def test_thinking_blocks_are_counted(self):
        """Thinking content should contribute to token estimation."""
        msg = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "x" * 200}],
        }
        assert estimate_message_tokens(msg) > 0


class TestEstimateMessagesTokens:
    def test_sum_of_messages(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        total = estimate_messages_tokens(msgs)
        individual = sum(estimate_message_tokens(m) for m in msgs)
        assert total == individual

    def test_empty_list(self):
        assert estimate_messages_tokens([]) == 0


# ---------------------------------------------------------------------------
# Context window resolution
# ---------------------------------------------------------------------------

class TestResolveContextWindow:
    def test_fallback_on_unknown_model(self):
        """Unknown model should return DEFAULT_CONTEXT_TOKENS."""
        with patch("litellm.get_model_info", side_effect=Exception("unknown")):
            result = resolve_context_window("unknown-model-xyz")
        assert result == DEFAULT_CONTEXT_TOKENS

    def test_strips_provider_prefix(self):
        """Should try stripping provider/ prefix as fallback."""
        call_count = 0

        def mock_get_info(model):
            nonlocal call_count
            call_count += 1
            if model == "anthropic/claude-test":
                raise Exception("not found")
            if model == "claude-test":
                return {"max_input_tokens": 500_000}
            raise Exception("not found")

        with patch("litellm.get_model_info", side_effect=mock_get_info):
            result = resolve_context_window("anthropic/claude-test")
        assert result == 500_000
        assert call_count == 2

    def test_direct_match(self):
        """Direct model name match should work."""
        with patch("litellm.get_model_info", return_value={"max_input_tokens": 1_000_000}):
            result = resolve_context_window("anthropic/claude-sonnet-4-20250514")
        assert result == 1_000_000

    def test_zero_max_input_falls_back(self):
        """Zero or negative max_input_tokens should fall back."""
        with patch("litellm.get_model_info", return_value={"max_input_tokens": 0}):
            result = resolve_context_window("some-model")
        assert result == DEFAULT_CONTEXT_TOKENS

    def test_no_litellm_installed(self):
        """Should gracefully handle litellm not being importable."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "litellm":
                raise ImportError("no litellm")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = resolve_context_window("any-model")
        assert result == DEFAULT_CONTEXT_TOKENS


# ---------------------------------------------------------------------------
# Tool result truncation
# ---------------------------------------------------------------------------

class TestHasImportantTail:
    def test_error_in_tail(self):
        text = "some output\n" * 100 + "\nTraceback (most recent call last):\n  error here"
        assert has_important_tail(text) is True

    def test_json_closing(self):
        text = '{"key": "value"}'
        assert has_important_tail(text) is True

    def test_summary_in_tail(self):
        text = "line\n" * 50 + "Total: 42 items processed"
        assert has_important_tail(text) is True

    def test_plain_text_no_important_tail(self):
        text = "just some plain text\n" * 50
        assert has_important_tail(text) is False


class TestTruncateToolResultText:
    def test_under_limit_passthrough(self):
        text = "short text"
        assert truncate_tool_result_text(text, 1000) == text

    def test_head_only_truncation(self):
        # No important tail — should keep head only
        text = "a" * 10_000
        result = truncate_tool_result_text(text, 3000)
        assert result.endswith(TRUNCATION_SUFFIX)
        assert MIDDLE_OMISSION_MARKER not in result
        assert len(result) <= 3000 + len(TRUNCATION_SUFFIX)

    def test_head_tail_truncation_with_error(self):
        # Important tail with error — should use head+tail strategy
        # Budget must be > MIN_KEEP_CHARS*2 for head+tail to activate,
        # and head_budget must be > MIN_KEEP_CHARS after subtracting tail+marker
        text = "beginning content\n" * 1000 + "\nTraceback: error at line 42\nFailed with exit code 1"
        result = truncate_tool_result_text(text, 8000)
        assert TRUNCATION_SUFFIX in result
        assert MIDDLE_OMISSION_MARKER in result
        # Tail should be preserved
        assert "exit code" in result or "Failed" in result

    def test_respects_min_keep_chars(self):
        text = "x" * 10_000
        result = truncate_tool_result_text(text, 100)  # very small limit
        # Should keep at least MIN_KEEP_CHARS
        content_len = len(result) - len(TRUNCATION_SUFFIX)
        assert content_len >= MIN_KEEP_CHARS


class TestTruncateToolResults:
    def test_truncates_oversized_function_call_output(self):
        context_window = 100_000
        max_chars = _calculate_max_tool_result_chars(context_window)
        oversized_output = "x" * (max_chars + 1000)
        msgs = [
            {"type": "function_call_output", "output": oversized_output, "call_id": "1"},
            {"type": "message", "content": [{"type": "text", "text": "hello"}]},
        ]
        result = truncate_tool_results(msgs, context_window)
        assert len(result) == 2
        assert len(result[0]["output"]) < len(oversized_output)
        assert result[1] == msgs[1]  # non-tool message unchanged

    def test_passthrough_under_limit(self):
        msgs = [
            {"type": "function_call_output", "output": "short", "call_id": "1"},
        ]
        result = truncate_tool_results(msgs, 200_000)
        assert result[0]["output"] == "short"

    def test_does_not_mutate_original(self):
        context_window = 50_000
        max_chars = _calculate_max_tool_result_chars(context_window)
        original_output = "y" * (max_chars + 500)
        msgs = [{"type": "function_call_output", "output": original_output, "call_id": "1"}]
        truncate_tool_results(msgs, context_window)
        assert msgs[0]["output"] == original_output  # original unchanged

    def test_calculate_max_respects_hard_limit(self):
        # Very large context window should still be capped by HARD_MAX
        result = _calculate_max_tool_result_chars(10_000_000)
        assert result == HARD_MAX_TOOL_RESULT_CHARS


# ---------------------------------------------------------------------------
# Reactive error detection
# ---------------------------------------------------------------------------

class TestIsContextOverflowError:
    @pytest.mark.parametrize("msg", [
        "request_too_large",
        "Error: context length exceeded for model",
        "prompt is too long for this model",
        "Request exceeds model context window",
        "maximum context length is 128000 tokens",
        "context overflow: too many tokens",
        "too many tokens in the request",
        "content_too_large error",
        "request size exceeds the limit",
    ])
    def test_true_positives(self, msg):
        assert is_context_overflow_error(msg) is True

    @pytest.mark.parametrize("msg", [
        "rate limit exceeded",
        "rate_limit error: too many tokens per minute (tpm)",
        "tpm limit reached",
        "rpm quota exceeded",
        "tpd rate limit",
    ])
    def test_rate_limit_exclusions(self, msg):
        assert is_context_overflow_error(msg) is False

    def test_empty_string(self):
        assert is_context_overflow_error("") is False

    def test_unrelated_error(self):
        assert is_context_overflow_error("connection timeout") is False

    def test_case_insensitive(self):
        assert is_context_overflow_error("REQUEST_TOO_LARGE") is True
        assert is_context_overflow_error("Context Length Exceeded") is True


# ---------------------------------------------------------------------------
# ContextOverflowCallback
# ---------------------------------------------------------------------------

class TestContextOverflowCallback:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_init_defaults(self):
        cb = ContextOverflowCallback(context_window=200_000)
        assert cb.context_window == 200_000
        assert cb.current_tokens == 0
        assert cb.needs_compaction is False
        assert cb.turn_count == 0
        assert cb.overflow_ratio == 0.0

    def test_resolves_context_from_model(self):
        with patch(
            "ale_run.agents.ale_claw.harness.context.context.resolve_context_window",
            return_value=500_000,
        ):
            cb = ContextOverflowCallback(model="test-model")
        assert cb.context_window == 500_000

    def test_explicit_context_window_overrides_model(self):
        cb = ContextOverflowCallback(context_window=300_000, model="ignored")
        assert cb.context_window == 300_000

    def test_on_llm_start_estimates_tokens(self):
        cb = ContextOverflowCallback(context_window=200_000)
        msgs = [{"role": "user", "content": "hello world"}]
        result = self._run(cb.on_llm_start(msgs))
        assert cb.current_tokens > 0
        assert cb.turn_count == 1
        assert isinstance(result, list)

    def test_threshold_triggers_compaction(self):
        # Small context window, large message — should trigger
        cb = ContextOverflowCallback(context_window=100, threshold=0.80)
        msgs = [{"role": "user", "content": "x" * 1000}]
        self._run(cb.on_llm_start(msgs))
        assert cb.needs_compaction is True

    def test_under_threshold_no_compaction(self):
        cb = ContextOverflowCallback(context_window=1_000_000, threshold=0.80)
        msgs = [{"role": "user", "content": "short"}]
        self._run(cb.on_llm_start(msgs))
        assert cb.needs_compaction is False

    def test_turn_counting(self):
        cb = ContextOverflowCallback(context_window=1_000_000)
        msgs = [{"role": "user", "content": "hi"}]
        self._run(cb.on_llm_start(msgs))
        self._run(cb.on_llm_start(msgs))
        self._run(cb.on_llm_start(msgs))
        assert cb.turn_count == 3

    def test_overflow_ratio(self):
        cb = ContextOverflowCallback(context_window=1000)
        msgs = [{"role": "user", "content": "x" * 400}]
        self._run(cb.on_llm_start(msgs))
        ratio = cb.overflow_ratio
        assert 0.0 < ratio < 10.0  # sanity bounds

    def test_force_compaction(self):
        cb = ContextOverflowCallback(context_window=1_000_000)
        assert cb.needs_compaction is False
        cb.force_compaction()
        assert cb.needs_compaction is True

    def test_instructions_tokens_offset(self):
        cb_no_offset = ContextOverflowCallback(context_window=200_000, instructions_tokens=0)
        cb_with_offset = ContextOverflowCallback(context_window=200_000, instructions_tokens=5000)
        msgs = [{"role": "user", "content": "hello"}]
        self._run(cb_no_offset.on_llm_start(msgs))
        self._run(cb_with_offset.on_llm_start(msgs))
        assert cb_with_offset.current_tokens == cb_no_offset.current_tokens + 5000

    def test_safety_margin_applied(self):
        cb = ContextOverflowCallback(context_window=1_000_000, instructions_tokens=0)
        msgs = [{"role": "user", "content": "test"}]
        self._run(cb.on_llm_start(msgs))
        raw = estimate_messages_tokens(msgs)
        expected = int(raw * SAFETY_MARGIN)
        assert cb.current_tokens == expected

    def test_truncates_tool_results_in_messages(self):
        """on_llm_start should truncate oversized function_call_output items."""
        context_window = 50_000
        max_chars = _calculate_max_tool_result_chars(context_window)
        oversized = "z" * (max_chars + 5000)
        msgs = [
            {"type": "function_call_output", "output": oversized, "call_id": "t1"},
        ]
        cb = ContextOverflowCallback(context_window=context_window)
        result = self._run(cb.on_llm_start(msgs))
        assert len(result[0]["output"]) < len(oversized)


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    def test_safety_margin(self):
        assert SAFETY_MARGIN == 1.2

    def test_default_context(self):
        assert DEFAULT_CONTEXT_TOKENS == 200_000

    def test_fixed_image_tokens(self):
        assert FIXED_IMAGE_TOKENS == 1200

    def test_max_tool_result_share(self):
        assert MAX_TOOL_RESULT_SHARE == 0.25

    def test_hard_max(self):
        assert HARD_MAX_TOOL_RESULT_CHARS == 16_000

    def test_min_keep(self):
        assert MIN_KEEP_CHARS == 2_000
