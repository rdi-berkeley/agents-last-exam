"""Tests for the compaction pipeline.

Covers:
  - Chunk splitting (by token share, by max tokens, adaptive ratio)
  - Message serialization for summarization
  - LLM-based summarization (mocked litellm)
  - Fallback tiers
  - compact_messages() entry point (budget-aware,)
  - Tool pairing repair
  - Recent turns preservation
  - Agent loop helpers (_extract_messages_for_compaction)
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ale_run.agents.ale_claw.harness.inference.model_config import HelperTransportDefaults, ModelConfig, register_model_config

from ale_run.agents.ale_claw.harness.context.context import (
    BASE_CHUNK_RATIO,
    DEFAULT_SUMMARY_FALLBACK,
    IDENTIFIER_PRESERVATION_INSTRUCTIONS,
    MAX_RECENT_TURNS_PRESERVE,
    MIN_CHUNK_RATIO,
    SAFETY_MARGIN,
    SUMMARIZATION_OVERHEAD_TOKENS,
    SUMMARIZATION_SYSTEM_PROMPT,
    SUMMARIZATION_TIMEOUT,
    SYNTHETIC_TOOL_RESULT_CONTENT,
    CompactionResult,
    chunk_messages_by_max_tokens,
    chunk_messages_by_token_share,
    compact_messages,
    compute_adaptive_chunk_ratio,
    estimate_message_tokens,
    estimate_messages_tokens,
    repair_tool_use_result_pairing,
    serialize_messages_for_summary,
    split_preserved_recent_turns,
    summarize_chunk,
    summarize_chunks_iterative,
    summarize_with_fallback,
)
from ale_run.agents.ale_claw.harness.agent_loop import _extract_messages_for_compaction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_msg(role: str, text: str) -> dict:
    """Create a simple message dict."""
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _make_messages(count: int, text_size: int = 100) -> list[dict]:
    """Create a list of messages with roughly equal token sizes."""
    filler = "x" * text_size
    msgs = []
    for i in range(count):
        role = "assistant" if i % 2 else "user"
        msgs.append(_make_msg(role, f"Message {i}: {filler}"))
    return msgs


def _mock_litellm_response(content: str) -> MagicMock:
    """Create a mock litellm completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Chunk splitting by token share
# ---------------------------------------------------------------------------

class TestChunkMessagesByTokenShare:
    def test_split_into_two_parts(self):
        msgs = _make_messages(10)
        chunks = chunk_messages_by_token_share(msgs, parts=2)
        assert len(chunks) == 2
        assert sum(len(c) for c in chunks) == 10

    def test_roughly_equal_token_budgets(self):
        msgs = _make_messages(10, text_size=200)
        chunks = chunk_messages_by_token_share(msgs, parts=2)
        tokens = [estimate_messages_tokens(c) for c in chunks]
        # Each half should be within 2x of the other (approximate)
        assert tokens[0] > 0
        assert tokens[1] > 0
        ratio = max(tokens) / min(tokens)
        assert ratio < 3.0

    def test_single_part(self):
        msgs = _make_messages(5)
        chunks = chunk_messages_by_token_share(msgs, parts=1)
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_empty_messages(self):
        assert chunk_messages_by_token_share([], parts=2) == []

    def test_single_message(self):
        msgs = _make_messages(1)
        chunks = chunk_messages_by_token_share(msgs, parts=2)
        assert len(chunks) == 1
        assert len(chunks[0]) == 1

    def test_more_parts_than_messages(self):
        msgs = _make_messages(2)
        chunks = chunk_messages_by_token_share(msgs, parts=5)
        total = sum(len(c) for c in chunks)
        assert total == 2

    def test_preserves_message_order(self):
        msgs = _make_messages(6)
        chunks = chunk_messages_by_token_share(msgs, parts=2)
        flat = [m for c in chunks for m in c]
        assert flat == msgs


# ---------------------------------------------------------------------------
# Chunk splitting by max tokens
# ---------------------------------------------------------------------------

class TestChunkMessagesByMaxTokens:
    def test_respects_max_tokens(self):
        msgs = _make_messages(10, text_size=200)
        per_msg = estimate_message_tokens(msgs[0])
        # Set max tokens to fit ~3 messages (accounting for safety margin)
        max_tokens = int(per_msg * 3 * SAFETY_MARGIN)
        chunks = chunk_messages_by_max_tokens(msgs, max_tokens)
        assert len(chunks) >= 3
        for chunk in chunks:
            chunk_tokens = estimate_messages_tokens(chunk)
            assert chunk_tokens <= max_tokens

    def test_oversized_single_message_gets_own_chunk(self):
        small = _make_msg("user", "tiny")
        big = _make_msg("assistant", "x" * 10000)
        msgs = [small, big, small]
        # Max tokens fits the small msg but not the big one
        max_tokens = int(estimate_message_tokens(small) * 2 * SAFETY_MARGIN)
        chunks = chunk_messages_by_max_tokens(msgs, max_tokens)
        assert len(chunks) >= 2
        assert sum(len(c) for c in chunks) == 3

    def test_empty_messages(self):
        assert chunk_messages_by_max_tokens([], 1000) == []

    def test_zero_max_tokens(self):
        assert chunk_messages_by_max_tokens(_make_messages(3), 0) == []


# ---------------------------------------------------------------------------
# Adaptive chunk ratio
# ---------------------------------------------------------------------------

class TestComputeAdaptiveChunkRatio:
    def test_default_for_small_messages(self):
        msgs = _make_messages(20, text_size=50)
        ratio = compute_adaptive_chunk_ratio(msgs, 200_000)
        assert ratio == BASE_CHUNK_RATIO

    def test_reduced_for_large_messages(self):
        # Each message ~5000 tokens, context 20000 → avg/ctx > 10%
        msgs = _make_messages(4, text_size=20000)
        ratio = compute_adaptive_chunk_ratio(msgs, 20_000)
        assert ratio < BASE_CHUNK_RATIO
        assert ratio >= MIN_CHUNK_RATIO

    def test_empty_messages(self):
        assert compute_adaptive_chunk_ratio([], 200_000) == BASE_CHUNK_RATIO

    def test_zero_context(self):
        msgs = _make_messages(5)
        assert compute_adaptive_chunk_ratio(msgs, 0) == BASE_CHUNK_RATIO


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------

class TestSerializeMessagesForSummary:
    def test_text_content(self):
        msgs = [_make_msg("user", "hello world")]
        result = serialize_messages_for_summary(msgs)
        assert "[user]" in result
        assert "hello world" in result

    def test_function_call_block(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "function_call", "name": "memory_search", "arguments": '{"query": "test"}'},
            ],
        }
        result = serialize_messages_for_summary([msg])
        assert "memory_search" in result
        assert "tool_call" in result

    def test_computer_call_block(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "computer_call", "action": {"type": "click", "x": 100, "y": 200}},
            ],
        }
        result = serialize_messages_for_summary([msg])
        assert "computer" in result
        assert "click" in result

    def test_tool_result_block(self):
        msg = {
            "role": "tool",
            "content": [
                {"type": "tool_result", "content": "search returned 3 results"},
            ],
        }
        result = serialize_messages_for_summary([msg])
        assert "tool_result" in result

    def test_string_content(self):
        msg = {"role": "user", "content": "plain text"}
        result = serialize_messages_for_summary([msg])
        assert "plain text" in result

    def test_long_arguments_truncated(self):
        long_args = json.dumps({"data": "x" * 500})
        msg = {
            "role": "assistant",
            "content": [{"type": "function_call", "name": "test", "arguments": long_args}],
        }
        result = serialize_messages_for_summary([msg])
        assert "..." in result


# ---------------------------------------------------------------------------
# Summarize chunk (mocked litellm)
# ---------------------------------------------------------------------------

class TestSummarizeChunk:
    def test_produces_string(self):
        mock_resp = _mock_litellm_response("This is a summary.")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                summarize_chunk(_make_messages(3), "test-model")
            )
        assert result == "This is a summary."

    def test_includes_identifier_preservation(self):
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            asyncio.run(
                summarize_chunk(_make_messages(3), "test-model")
            )
            call_args = mock_acomp.call_args
            system_msg = call_args.kwargs["messages"][0]["content"]
            assert "opaque identifiers" in system_msg

    def test_system_prompt_matches_openclaw(self):
        """System prompt must match OpenClaw's strict framing."""
        assert "context summarization assistant" in SUMMARIZATION_SYSTEM_PROMPT
        assert "Do NOT continue the conversation" in SUMMARIZATION_SYSTEM_PROMPT
        assert "ONLY output the structured summary" in SUMMARIZATION_SYSTEM_PROMPT

    def test_initial_prompt_uses_summarization_prompt(self):
        """When no previous_summary, SUMMARIZATION_PROMPT should be in user message."""
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            asyncio.run(
                summarize_chunk(_make_messages(3), "test-model")
            )
            user_msg = mock_acomp.call_args.kwargs["messages"][1]["content"]
            assert "## Goal" in user_msg
            assert "## Progress" in user_msg
            assert "## Next Steps" in user_msg
            # Should NOT contain update prompt text
            assert "NEW conversation messages" not in user_msg

    def test_update_prompt_uses_update_summarization_prompt(self):
        """When previous_summary is set, UPDATE_SUMMARIZATION_PROMPT should be used."""
        mock_resp = _mock_litellm_response("Updated summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            asyncio.run(
                summarize_chunk(
                    _make_messages(3), "test-model",
                    previous_summary="Earlier context here",
                )
            )
            user_msg = mock_acomp.call_args.kwargs["messages"][1]["content"]
            assert "NEW conversation messages" in user_msg
            assert "PRESERVE all existing information" in user_msg
            # Should NOT contain initial prompt text
            assert "Use this EXACT format:" not in user_msg

    def test_conversation_wrapped_in_xml_tags(self):
        """Conversation text must be wrapped in <conversation> tags."""
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            asyncio.run(
                summarize_chunk(_make_messages(3), "test-model")
            )
            user_msg = mock_acomp.call_args.kwargs["messages"][1]["content"]
            assert "<conversation>" in user_msg
            assert "</conversation>" in user_msg

    def test_previous_summary_wrapped_in_xml_tags(self):
        """Previous summary must be wrapped in <previous-summary> tags."""
        mock_resp = _mock_litellm_response("Updated")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            asyncio.run(
                summarize_chunk(
                    _make_messages(3), "test-model",
                    previous_summary="Prior summary content",
                )
            )
            user_msg = mock_acomp.call_args.kwargs["messages"][1]["content"]
            assert "<previous-summary>" in user_msg
            assert "</previous-summary>" in user_msg
            assert "Prior summary content" in user_msg

    def test_no_previous_summary_tags_when_initial(self):
        """Initial summarization should not contain <previous-summary> tags."""
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            asyncio.run(
                summarize_chunk(_make_messages(3), "test-model")
            )
            user_msg = mock_acomp.call_args.kwargs["messages"][1]["content"]
            assert "<previous-summary>" not in user_msg

    def test_passes_custom_instructions(self):
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            asyncio.run(
                summarize_chunk(
                    _make_messages(3), "test-model",
                    custom_instructions="Focus on game progress",
                )
            )
            call_args = mock_acomp.call_args
            system_msg = call_args.kwargs["messages"][0]["content"]
            assert "Focus on game progress" in system_msg

    def test_openai_helper_call_can_omit_reasoning_kwargs(self):
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            asyncio.run(
                summarize_chunk(
                    _make_messages(3),
                    "openai/gpt-5.4",
                    thinking_params={},
                )
            )
        assert "reasoning" not in mock_acomp.call_args.kwargs

    def test_retry_on_failure(self):
        mock_resp = _mock_litellm_response("After retry")
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[Exception("API error"), mock_resp]) as mock_acomp:
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = asyncio.run(
                    summarize_chunk(_make_messages(3), "test-model")
                )
        assert result == "After retry"
        assert mock_acomp.call_count == 2

    def test_all_retries_exhausted_raises(self):
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("persistent error")) as mock_acomp:
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(Exception, match="persistent error"):
                    asyncio.run(
                        summarize_chunk(_make_messages(3), "test-model")
                    )
        assert mock_acomp.call_count == 3

    def test_passes_timeout_to_litellm(self):
        """Default call passes timeout=120 (SUMMARIZATION_TIMEOUT) to litellm."""
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            asyncio.run(
                summarize_chunk(_make_messages(3), "test-model")
            )
            assert mock_acomp.call_args.kwargs["timeout"] == SUMMARIZATION_TIMEOUT

    def test_custom_timeout(self):
        """Custom timeout value is forwarded to litellm."""
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            asyncio.run(
                summarize_chunk(_make_messages(3), "test-model", timeout=30)
            )
            assert mock_acomp.call_args.kwargs["timeout"] == 30

    def test_passes_thinking_params(self):
        """Compaction summarization forwards provider-specific thinking params."""
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            asyncio.run(
                summarize_chunk(
                    _make_messages(3),
                    "test-model",
                    thinking_params={"reasoning_effort": "medium"},
                )
            )
            assert mock_acomp.call_args.kwargs["reasoning_effort"] == "medium"

    def test_timeout_fallback_via_summarize_with_fallback(self):
        """Timeout errors trigger Tier 3 static fallback via summarize_with_fallback."""
        timeout_error = Exception("Request timed out")
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=timeout_error):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = asyncio.run(
                    summarize_with_fallback(
                        _make_messages(5), "test-model", 200_000, 50_000,
                    )
                )
        assert "5 messages could not be summarized" in result


# ---------------------------------------------------------------------------
# Summarize chunks iterative
# ---------------------------------------------------------------------------

class TestSummarizeChunksIterative:
    def test_iterates_over_chunks(self):
        mock_resp = _mock_litellm_response("Chunk summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_acomp:
            chunks = [_make_messages(3), _make_messages(3)]
            result = asyncio.run(
                summarize_chunks_iterative(chunks, "test-model")
            )
        assert result == "Chunk summary"
        assert mock_acomp.call_count == 2

    def test_empty_chunks_returns_fallback(self):
        result = asyncio.run(
            summarize_chunks_iterative([], "test-model")
        )
        assert result == DEFAULT_SUMMARY_FALLBACK

    def test_feeds_previous_summary(self):
        """Second chunk should receive the first chunk's summary as context."""
        summaries = ["First chunk summary", "Combined summary"]
        call_count = [0]

        async def mock_acompletion(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 1:
                # Verify the second call includes the first summary
                user_msg = kwargs["messages"][1]["content"]
                assert "First chunk summary" in user_msg
            return _mock_litellm_response(summaries[idx])

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            chunks = [_make_messages(3), _make_messages(3)]
            result = asyncio.run(
                summarize_chunks_iterative(chunks, "test-model")
            )
        assert result == "Combined summary"

    def test_custom_resolved_runtime_can_switch_compaction_transport(self):
        from ale_run.agents.ale_claw.harness.inference.model_config import _MODEL_CONFIGS

        original = list(_MODEL_CONFIGS)
        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Chunk summary"}],
                }
            ]
        }
        register_model_config(
            r"acme-summarizer",
            ModelConfig(
                tool_schema_type="computer_use_preview",
                screenshot_output_type="input_image",
                supports_safety_checks=True,
                action_format="single",
                adapter_target="anthropic",
                provider="acme",
                model_api="chat",
                transcript_api_label="acme-chat",
                helper_transport_defaults=HelperTransportDefaults(compaction="responses"),
            ),
        )
        try:
            with patch("litellm.aresponses", new_callable=AsyncMock, return_value=mock_resp) as mock_aresponses, patch(
                "litellm.acompletion", new_callable=AsyncMock
            ) as mock_acomp:
                result = asyncio.run(
                    summarize_chunks_iterative(
                        [_make_messages(3)],
                        "acme/acme-summarizer",
                    )
                )
            assert result == "Chunk summary"
            assert mock_aresponses.await_count == 1
            assert mock_acomp.await_count == 0
        finally:
            _MODEL_CONFIGS.clear()
            _MODEL_CONFIGS.extend(original)


# ---------------------------------------------------------------------------
# Summarize with fallback
# ---------------------------------------------------------------------------

class TestSummarizeWithFallback:
    def test_tier1_success(self):
        mock_resp = _mock_litellm_response("Full summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                summarize_with_fallback(
                    _make_messages(5), "test-model", 200_000, 50_000,
                )
            )
        assert result == "Full summary"

    def test_tier3_fallback_on_all_failures(self):
        with patch("litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("fail")):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = asyncio.run(
                    summarize_with_fallback(
                        _make_messages(5), "test-model", 200_000, 50_000,
                    )
                )
        assert "5 messages could not be summarized" in result


# ---------------------------------------------------------------------------
# Verify removed symbols
# ---------------------------------------------------------------------------

class TestRemovedSymbols:
    def test_merge_summaries_instructions_removed(self):
        """MERGE_SUMMARIES_INSTRUCTIONS should no longer exist in context module."""
        import ale_run.agents.ale_claw.harness.context.context as ctx
        assert not hasattr(ctx, "MERGE_SUMMARIES_INSTRUCTIONS")

    def test_merge_summaries_function_removed(self):
        """_merge_summaries should no longer exist in context module."""
        import ale_run.agents.ale_claw.harness.context.context as ctx
        assert not hasattr(ctx, "_merge_summaries")


# ---------------------------------------------------------------------------
# compact_messages (main entry point)
# ---------------------------------------------------------------------------

class TestCompactMessages:
    def test_returns_compaction_result(self):
        mock_resp = _mock_litellm_response("Compacted summary of conversation.")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(_make_messages(10), "test-model", 200_000)
            )
        assert isinstance(result, CompactionResult)
        assert result.summary == "Compacted summary of conversation."
        assert result.tokens_before > 0
        assert result.first_kept_message_index > 0
        assert result.chunks_processed >= 1

    def test_tokens_after_less_than_before(self):
        mock_resp = _mock_litellm_response("Short summary.")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(_make_messages(20, text_size=500), "test-model", 200_000)
            )
        assert result.tokens_after < result.tokens_before

    def test_preserves_kept_messages(self):
        """Messages after the compaction point should not be modified."""
        msgs = _make_messages(10)
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(msgs, "test-model", 200_000)
            )
        # Verify the split point is reasonable
        assert 0 < result.first_kept_message_index < len(msgs)

    def test_empty_messages(self):
        result = asyncio.run(
            compact_messages([], "test-model", 200_000)
        )
        assert result.summary == DEFAULT_SUMMARY_FALLBACK
        assert result.tokens_before == 0
        assert result.tokens_after == 0
        assert result.first_kept_message_index == 0

    def test_single_message(self):
        """Single message: preserved by recent turns, nothing compacted."""
        mock_resp = _mock_litellm_response("Single msg summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(_make_messages(1), "test-model", 200_000)
            )
        # Single message goes to preserved (fewer user turns than preserve_count)
        assert result.summary == DEFAULT_SUMMARY_FALLBACK
        assert result.first_kept_message_index == 0  # all kept (preserved)


# ---------------------------------------------------------------------------
# Identifier preservation (prompt validation)
# ---------------------------------------------------------------------------

class TestIdentifierPreservation:
    def test_preservation_text_mentions_uuids(self):
        assert "UUIDs" in IDENTIFIER_PRESERVATION_INSTRUCTIONS

    def test_preservation_text_mentions_urls(self):
        assert "URLs" in IDENTIFIER_PRESERVATION_INSTRUCTIONS

    def test_preservation_text_mentions_file_names(self):
        assert "file names" in IDENTIFIER_PRESERVATION_INSTRUCTIONS

    def test_preservation_text_mentions_ips(self):
        assert "IPs" in IDENTIFIER_PRESERVATION_INSTRUCTIONS

    def test_injected_into_summarization_system_prompt(self):
        """Verify identifier instructions appear in the system prompt sent to LLM."""
        assert "opaque identifiers" in SUMMARIZATION_SYSTEM_PROMPT
        assert "UUIDs" in SUMMARIZATION_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Agent loop helpers
# ---------------------------------------------------------------------------

class TestExtractMessagesForCompaction:
    def test_extracts_message_entries(self):
        mock_mgr = MagicMock()
        # Simulate transcript entries
        msg_entry = MagicMock()
        msg_entry.type = "message"
        msg_entry.data = {"message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}}
        session_entry = MagicMock()
        session_entry.type = "session"
        session_entry.data = {}
        mock_mgr.load_history.return_value = [session_entry, msg_entry]

        messages = _extract_messages_for_compaction(mock_mgr)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"

    def test_skips_non_message_entries(self):
        mock_mgr = MagicMock()
        compaction_entry = MagicMock()
        compaction_entry.type = "compaction"
        compaction_entry.data = {}
        mock_mgr.load_history.return_value = [compaction_entry]

        messages = _extract_messages_for_compaction(mock_mgr)
        assert len(messages) == 0

# ===========================================================================
# Tool pairing repair
# ===========================================================================

def _make_assistant_with_call(call_id: str, call_type: str = "function_call", stop_reason: str | None = None) -> dict:
    """Create an assistant message with a tool call."""
    block = {"type": call_type, "id": call_id}
    if call_type == "function_call":
        block.update({"name": "test_tool", "arguments": "{}"})
    elif call_type == "computer_call":
        block.update({"action": {"type": "click", "x": 100, "y": 200}})
    msg = {"role": "assistant", "content": [block]}
    if stop_reason:
        msg["stop_reason"] = stop_reason
    return msg


def _make_tool_result(call_id: str, content: str = "result") -> dict:
    """Create a tool message with a result."""
    return {
        "role": "tool",
        "content": [{"type": "tool_result", "tool_use_id": call_id, "content": content}],
    }


class TestRepairToolUseResultPairing:
    def test_repair_preserves_complete_pairs(self):
        msgs = [
            _make_assistant_with_call("call-1"),
            _make_tool_result("call-1"),
            _make_assistant_with_call("call-2"),
            _make_tool_result("call-2"),
        ]
        report = repair_tool_use_result_pairing(msgs)
        assert report.dropped_orphan_count == 0
        assert report.dropped_duplicate_count == 0
        assert report.added_synthetic_count == 0
        assert len(report.messages) == 4

    def test_repair_drops_orphaned_tool_result(self):
        """Result with no matching call should be dropped."""
        msgs = [
            _make_tool_result("orphan-id"),  # no matching assistant call
            _make_assistant_with_call("call-1"),
            _make_tool_result("call-1"),
        ]
        report = repair_tool_use_result_pairing(msgs)
        assert report.dropped_orphan_count == 1
        # The orphan tool message should be gone entirely
        assert len(report.messages) == 2
        roles = [m["role"] for m in report.messages]
        assert roles == ["assistant", "tool"]

    def test_repair_synthesizes_missing_result(self):
        """Call with no matching result should get a synthetic error result."""
        msgs = [
            _make_assistant_with_call("call-1"),
            # no tool result for call-1
        ]
        report = repair_tool_use_result_pairing(msgs)
        assert report.added_synthetic_count == 1
        assert len(report.messages) == 2  # assistant + synthetic tool
        synth = report.messages[1]
        assert synth["role"] == "tool"
        assert synth["content"][0]["tool_use_id"] == "call-1"
        assert synth["content"][0]["is_error"] is True
        assert SYNTHETIC_TOOL_RESULT_CONTENT in synth["content"][0]["content"]

    def test_repair_skips_synthesis_for_error_stop_reason(self):
        """Calls with stop_reason='error' should not get synthetic results."""
        msgs = [
            _make_assistant_with_call("call-1", stop_reason="error"),
        ]
        report = repair_tool_use_result_pairing(msgs)
        assert report.added_synthetic_count == 0
        assert len(report.messages) == 1

    def test_repair_skips_synthesis_for_aborted_stop_reason(self):
        """Calls with stop_reason='aborted' should not get synthetic results."""
        msgs = [
            _make_assistant_with_call("call-1", stop_reason="aborted"),
        ]
        report = repair_tool_use_result_pairing(msgs)
        assert report.added_synthetic_count == 0

    def test_repair_handles_computer_call(self):
        """computer_call orphan should get synthetic result."""
        msgs = [
            _make_assistant_with_call("comp-1", call_type="computer_call"),
        ]
        report = repair_tool_use_result_pairing(msgs)
        assert report.added_synthetic_count == 1
        synth = report.messages[1]
        assert synth["content"][0]["tool_use_id"] == "comp-1"

    def test_repair_drops_duplicates(self):
        """Same tool_use_id appearing twice should have the second dropped."""
        msgs = [
            _make_assistant_with_call("call-1"),
            _make_tool_result("call-1"),
            _make_tool_result("call-1"),  # duplicate
        ]
        report = repair_tool_use_result_pairing(msgs)
        assert report.dropped_duplicate_count == 1

    def test_repair_empty_messages(self):
        report = repair_tool_use_result_pairing([])
        assert report.messages == []
        assert report.dropped_orphan_count == 0
        assert report.dropped_duplicate_count == 0
        assert report.added_synthetic_count == 0

    def test_repair_at_split_boundary(self):
        """Simulate a split cutting between call and result."""
        # After a split, we might have just the result from a prior call
        # and an assistant call with no result
        msgs = [
            _make_tool_result("old-call"),  # orphan — call was in compacted half
            _make_assistant_with_call("new-call"),
            # result for new-call was in the other split
        ]
        report = repair_tool_use_result_pairing(msgs)
        assert report.dropped_orphan_count == 1
        assert report.added_synthetic_count == 1


# ===========================================================================
# Recent turns preservation
# ===========================================================================

class TestSplitPreservedRecentTurns:
    def test_split_preserves_last_n_turns(self):
        """Counts assistant messages as turns (works for chat and CUA)."""
        msgs = [
            _make_msg("user", "msg 0"),
            _make_msg("assistant", "resp 0"),
            _make_msg("user", "msg 1"),
            _make_msg("assistant", "resp 1"),
            _make_msg("user", "msg 2"),
            _make_msg("assistant", "resp 2"),
            _make_msg("user", "msg 3"),
            _make_msg("assistant", "resp 3"),
        ]
        pruneable, preserved = split_preserved_recent_turns(msgs, preserve_count=3)
        # Last 3 assistant turns: resp 1, resp 2, resp 3 + surrounding messages
        assistant_msgs_in_preserved = [m for m in preserved if m.get("role") == "assistant"]
        assert len(assistant_msgs_in_preserved) == 3
        assert len(pruneable) + len(preserved) == len(msgs)

    def test_preserved_turns_count_zero(self):
        msgs = _make_messages(6)
        pruneable, preserved = split_preserved_recent_turns(msgs, preserve_count=0)
        assert len(preserved) == 0
        assert len(pruneable) == 6

    def test_preserved_turns_exceeds_messages(self):
        """Fewer assistant turns than preserve_count → all preserved."""
        msgs = [
            _make_msg("user", "only user"),
            _make_msg("assistant", "only response"),
        ]
        pruneable, preserved = split_preserved_recent_turns(msgs, preserve_count=5)
        assert len(pruneable) == 0
        assert len(preserved) == 2

    def test_cua_pattern_single_user_many_assistant_tool_pairs(self):
        """CUA-style: 1 user message + N assistant/tool pairs.

        With the old user-counting logic, everything would be preserved and
        pruneable would be empty. With assistant-counting, the function should
        split correctly.
        """
        msgs = [_make_msg("user", "Navigate to Settings and change theme")]
        for i in range(20):
            msgs.append(_make_assistant_with_call(f"comp-{i}", call_type="computer_call"))
            msgs.append(_make_tool_result(f"comp-{i}", content=f"screenshot {i}"))

        pruneable, preserved = split_preserved_recent_turns(msgs, preserve_count=3)
        # Should preserve the last 3 assistant turns + their tool results
        assert len(pruneable) > 0, "pruneable must be non-empty for CUA pattern"
        assistant_in_preserved = [m for m in preserved if m.get("role") == "assistant"]
        assert len(assistant_in_preserved) == 3
        # Total message count should be conserved
        assert len(pruneable) + len(preserved) == len(msgs)

    def test_max_recent_turns_cap(self):
        """preserve_count is capped at MAX_RECENT_TURNS_PRESERVE."""
        msgs = _make_messages(50)
        pruneable, preserved = split_preserved_recent_turns(msgs, preserve_count=100)
        # Should be capped, not preserve all 25 assistant messages
        assistant_in_preserved = [m for m in preserved if m.get("role") == "assistant"]
        assert len(assistant_in_preserved) <= MAX_RECENT_TURNS_PRESERVE
        assert len(pruneable) > 0

    def test_preserved_turns_survive_tight_budget(self):
        """Even with tiny budget, preserved turns should appear in final kept."""
        # Build messages: 10 user/assistant pairs with big content
        msgs = []
        for i in range(10):
            msgs.append(_make_msg("user", f"User message {i}: " + "x" * 500))
            msgs.append(_make_msg("assistant", f"Assistant response {i}: " + "y" * 500))

        mock_resp = _mock_litellm_response("Tight budget summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(
                    msgs, "test-model", 10_000,
                    instructions_tokens=3000,  # large instructions eat budget
                    recent_turns_preserve=2,
                )
            )
        # The last 2 assistant turns should be in kept portion
        kept = msgs[result.first_kept_message_index:]
        assistant_msgs_in_kept = [m for m in kept if m.get("role") == "assistant"]
        assert len(assistant_msgs_in_kept) >= 2


# ===========================================================================
# Budget-aware compaction
# ===========================================================================

class TestBudgetAwareCompaction:
    def test_budget_adapts_when_instructions_large(self):
        """Large instructions_tokens should result in fewer kept messages."""
        msgs = _make_messages(20, text_size=500)
        mock_resp = _mock_litellm_response("Summary")

        # Small instructions
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result_small = asyncio.run(
                compact_messages(msgs, "test-model", 50_000, instructions_tokens=1000)
            )
        # Large instructions
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result_large = asyncio.run(
                compact_messages(msgs, "test-model", 50_000, instructions_tokens=15000)
            )
        # Large instructions should compact more (higher first_kept_index)
        assert result_large.first_kept_message_index >= result_small.first_kept_message_index

    def test_budget_calculation_available_for_kept(self):
        """Verify the budget formula: context_window * share - instructions - overhead - preserved."""
        context_window = 100_000
        max_history_share = 0.5
        instructions_tokens = 10_000
        # Expected: 100000 * 0.5 - 10000 - 4096 - preserved_tokens
        expected_base = int(context_window * max_history_share) - instructions_tokens - SUMMARIZATION_OVERHEAD_TOKENS
        assert expected_base > 0  # sanity
        # We can't directly test the internal variable, but we verify the behavior:
        # with a generous budget, most messages should be kept
        msgs = _make_messages(4, text_size=100)
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(
                    msgs, "test-model", context_window,
                    instructions_tokens=instructions_tokens,
                    max_history_share=max_history_share,
                )
            )
        # With 4 small messages and a large context window, most should be kept
        assert result.first_kept_message_index < len(msgs)

    def test_kept_messages_within_budget(self):
        """After compaction, kept tokens should be within budget."""
        msgs = _make_messages(20, text_size=500)
        context_window = 30_000
        instructions_tokens = 5000
        max_history_share = 0.5

        mock_resp = _mock_litellm_response("Short")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(
                    msgs, "test-model", context_window,
                    instructions_tokens=instructions_tokens,
                    max_history_share=max_history_share,
                )
            )
        kept_msgs = msgs[result.first_kept_message_index:]
        kept_tokens = estimate_messages_tokens(kept_msgs)
        budget = int(context_window * max_history_share)
        # Kept tokens should be reasonable relative to budget
        assert kept_tokens < budget

    def test_no_cascading_post_compaction(self):
        """After compaction, kept + summary + instructions should be under 80% of context."""
        msgs = _make_messages(20, text_size=500)
        context_window = 50_000
        instructions_tokens = 5000

        mock_resp = _mock_litellm_response("Short summary.")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(
                    msgs, "test-model", context_window,
                    instructions_tokens=instructions_tokens,
                )
            )
        # tokens_after includes summary + kept; add instructions for total
        total_post = result.tokens_after + instructions_tokens
        assert total_post < context_window * 0.8

    def test_default_instructions_tokens_backward_compat(self):
        """compact_messages() works without instructions_tokens arg (defaults to 0)."""
        msgs = _make_messages(10)
        mock_resp = _mock_litellm_response("Summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(msgs, "test-model", 200_000)
            )
        assert isinstance(result, CompactionResult)
        assert result.summary == "Summary"

    def test_custom_max_history_share(self):
        """Custom max_history_share should be respected."""
        msgs = _make_messages(20, text_size=500)
        mock_resp = _mock_litellm_response("Summary")

        # Very small share → more aggressive compaction
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result_small = asyncio.run(
                compact_messages(msgs, "test-model", 50_000, max_history_share=0.2)
            )
        # Larger share → less aggressive compaction
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result_large = asyncio.run(
                compact_messages(msgs, "test-model", 50_000, max_history_share=0.8)
            )
        assert result_small.first_kept_message_index >= result_large.first_kept_message_index

    def test_iterative_pruning_multiple_rounds(self):
        """Budget so tight that kept portion needs multiple pruning rounds."""
        # Create many large messages so even half doesn't fit the budget
        msgs = _make_messages(20, text_size=2000)
        mock_resp = _mock_litellm_response("Summary")

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(
                    msgs, "test-model", 20_000,
                    instructions_tokens=5000,
                    max_history_share=0.3,
                    recent_turns_preserve=1,
                )
            )
        # With tight budget, most messages should be compacted
        assert result.first_kept_message_index > len(msgs) // 2


# ===========================================================================
# End-to-end compact + repair
# ===========================================================================

class TestCompactAppliesRepairToKept:
    def test_compact_applies_repair_to_kept(self):
        """compact_messages should repair tool pairing on the kept portion."""
        # Build messages with a function call at the boundary that will
        # lose its result when split
        msgs = []
        for i in range(8):
            msgs.append(_make_msg("user", f"User message {i}: " + "x" * 200))
            msgs.append(_make_msg("assistant", f"Response {i}: " + "y" * 200))
        # Add a function call near the middle that will lose its result
        msgs.insert(8, _make_assistant_with_call("boundary-call"))
        # Result goes right after — may end up in compacted portion
        msgs.insert(9, _make_tool_result("boundary-call"))

        mock_resp = _mock_litellm_response("Summary of compacted messages")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(msgs, "test-model", 50_000)
            )
        # Should complete without error — repair handles any orphans
        assert isinstance(result, CompactionResult)
        assert result.tokens_before > 0


# ===========================================================================
# _extract_messages_for_compaction propagates stop_reason
# ===========================================================================

class TestExtractMessagesStopReason:
    def test_propagates_stop_reason(self):
        mock_mgr = MagicMock()
        msg_entry = MagicMock()
        msg_entry.type = "message"
        msg_entry.data = {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "tool_use",
            }
        }
        mock_mgr.load_history.return_value = [msg_entry]

        messages = _extract_messages_for_compaction(mock_mgr)
        assert len(messages) == 1
        assert messages[0]["stop_reason"] == "tool_use"

    def test_omits_stop_reason_when_absent(self):
        mock_mgr = MagicMock()
        msg_entry = MagicMock()
        msg_entry.type = "message"
        msg_entry.data = {
            "message": {
                "role": "user",
                "content": "hello",
            }
        }
        mock_mgr.load_history.return_value = [msg_entry]

        messages = _extract_messages_for_compaction(mock_mgr)
        assert len(messages) == 1
        assert "stop_reason" not in messages[0]


# ===========================================================================
# CUA-pattern compaction (1 user + N assistant/tool pairs)
# ===========================================================================

class TestCuaPatternCompaction:
    def test_cua_pattern_produces_nonempty_pruneable(self):
        """CUA-style conversation must have non-empty pruneable set."""
        msgs = [_make_msg("user", "Open the settings app")]
        for i in range(20):
            msgs.append(_make_assistant_with_call(f"c-{i}", call_type="computer_call"))
            msgs.append(_make_tool_result(f"c-{i}", content=f"screenshot {i}: " + "x" * 200))

        mock_resp = _mock_litellm_response("Agent navigated to settings")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(msgs, "test-model", 50_000, recent_turns_preserve=3)
            )
        # Compaction should actually run (not return fallback)
        assert result.summary != DEFAULT_SUMMARY_FALLBACK
        assert result.chunks_processed >= 1
        assert result.first_kept_message_index > 0

    def test_cua_pattern_tight_budget_triggers_overflow(self):
        """When all messages are preserved but exceed budget, overflow kicks in."""
        msgs = [_make_msg("user", "Do the task")]
        # Use large messages to exceed the budget floor
        for i in range(12):
            msgs.append(_make_assistant_with_call(f"c-{i}", call_type="computer_call"))
            msgs.append(_make_tool_result(f"c-{i}", content=f"screenshot {i}: " + "x" * 2000))

        mock_resp = _mock_litellm_response("Overflow summary")
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(
                compact_messages(
                    msgs, "test-model", 10_000,
                    instructions_tokens=3000,
                    recent_turns_preserve=MAX_RECENT_TURNS_PRESERVE,
                )
            )
        # Overflow fallback should have moved some preserved to pruneable
        assert result.summary != DEFAULT_SUMMARY_FALLBACK
        assert result.chunks_processed >= 1
