"""Tests for Canonical internal message format + TranscriptPolicy.

Covers:
  - normalize_to_canonical: untyped dicts → CanonicalMessage
  - canonical_to_responses_api: CanonicalMessage → Responses API flat items
  - canonical_to_anthropic_messages: CanonicalMessage → Anthropic completion format
  - _build_compacted_items integration: verify canonical output from compaction
  - TranscriptPolicy: dataclass defaults, per-provider resolution
  - drop_thinking_blocks: strip thinking content, preserve turn structure
  - sanitize_thinking_signatures: remove thinkingSignature fields
  - downgrade_openai_reasoning: drop orphaned OpenAI reasoning blocks
  - sanitize_items: policy-driven pipeline integration
"""

import json

from ale_run.agents.ale_claw.harness.model.model_config import resolve_model

from ale_run.agents.ale_claw.harness.canonical.canonical import (
    COMPACTION_PREAMBLE,
    CanonicalMessage,
    CompactionSummaryBlock,
    ComputerCallBlock,
    FunctionCallBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    TranscriptPolicy,
    canonical_to_anthropic_messages,
    canonical_to_responses_api,
    downgrade_openai_reasoning,
    drop_thinking_blocks,
    get_transcript_policy,
    normalize_to_canonical,
    sanitize_items,
    sanitize_thinking_signatures,
)

# ---- Helpers ----


def _msg(role: str, content, **kwargs) -> dict:
    """Build an untyped message dict."""
    m = {"role": role, "content": content}
    m.update(kwargs)
    return m


# =========================================================================
# normalize_to_canonical
# =========================================================================


class TestNormalizeToCanonical:
    """Test ingestion from untyped dicts to typed CanonicalMessage."""

    def test_string_content_normalized_to_text_block(self):
        msgs = [_msg("user", "Hello"), _msg("assistant", "Hi")]
        result = normalize_to_canonical(msgs)

        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"type": "text", "text": "Hello"}]
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == [{"type": "text", "text": "Hi"}]

    def test_text_blocks_preserved(self):
        msgs = [_msg("user", [{"type": "text", "text": "Hello world"}])]
        result = normalize_to_canonical(msgs)

        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello world"

    def test_function_call_block(self):
        msgs = [_msg("assistant", [
            {"type": "function_call", "id": "call-1", "name": "search", "arguments": '{"q": "x"}'},
        ])]
        result = normalize_to_canonical(msgs)

        block = result[0]["content"][0]
        assert block["type"] == "function_call"
        assert block["id"] == "call-1"
        assert block["name"] == "search"
        assert block["arguments"] == '{"q": "x"}'

    def test_function_call_block_with_call_id_field(self):
        """Some transcript paths use call_id instead of id."""
        msgs = [_msg("assistant", [
            {"type": "function_call", "call_id": "call-2", "name": "get", "arguments": "{}"},
        ])]
        result = normalize_to_canonical(msgs)

        block = result[0]["content"][0]
        assert block["id"] == "call-2"

    def test_computer_call_action_normalized_to_actions(self):
        """Singular action dict normalized to actions list."""
        msgs = [_msg("assistant", [
            {"type": "computer_call", "id": "cc-1", "action": {"type": "click", "x": 100, "y": 200}},
        ])]
        result = normalize_to_canonical(msgs)

        block = result[0]["content"][0]
        assert block["type"] == "computer_call"
        assert block["id"] == "cc-1"
        assert block["actions"] == [{"type": "click", "x": 100, "y": 200}]

    def test_computer_call_actions_list_preserved(self):
        """Already-list actions preserved as-is."""
        actions = [{"type": "click", "x": 10, "y": 20}, {"type": "type", "text": "hi"}]
        msgs = [_msg("assistant", [
            {"type": "computer_call", "id": "cc-2", "actions": actions},
        ])]
        result = normalize_to_canonical(msgs)

        assert result[0]["content"][0]["actions"] == actions

    def test_tool_result_block(self):
        msgs = [_msg("tool", [
            {"type": "tool_result", "tool_use_id": "call-1", "content": "done"},
        ])]
        result = normalize_to_canonical(msgs)

        block = result[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call-1"
        assert block["content"] == "done"
        assert "is_error" not in block

    def test_tool_result_is_error_preserved(self):
        msgs = [_msg("tool", [
            {"type": "tool_result", "tool_use_id": "c-1", "content": "fail", "is_error": True},
        ])]
        result = normalize_to_canonical(msgs)

        assert result[0]["content"][0]["is_error"] is True

    def test_thinking_block(self):
        msgs = [_msg("assistant", [
            {"type": "thinking", "thinking": "Let me think..."},
        ])]
        result = normalize_to_canonical(msgs)

        block = result[0]["content"][0]
        assert block["type"] == "thinking"
        assert block["thinking"] == "Let me think..."
        assert "thinkingSignature" not in block

    def test_thinking_block_with_signature(self):
        msgs = [_msg("assistant", [
            {"type": "thinking", "thinking": "hmm", "thinkingSignature": "sig123"},
        ])]
        result = normalize_to_canonical(msgs)

        assert result[0]["content"][0]["thinkingSignature"] == "sig123"

    def test_stop_reason_preserved(self):
        msgs = [_msg("assistant", "I'm done", stop_reason="end_turn")]
        result = normalize_to_canonical(msgs)

        assert result[0]["stop_reason"] == "end_turn"

    def test_stop_reason_absent_when_not_provided(self):
        msgs = [_msg("user", "hello")]
        result = normalize_to_canonical(msgs)

        assert "stop_reason" not in result[0]

    def test_mixed_content_blocks(self):
        """Assistant message with text + function_call + computer_call."""
        msgs = [_msg("assistant", [
            {"type": "text", "text": "I'll help"},
            {"type": "function_call", "id": "f-1", "name": "search", "arguments": "{}"},
            {"type": "computer_call", "id": "c-1", "action": {"type": "click", "x": 50, "y": 50}},
        ])]
        result = normalize_to_canonical(msgs)

        blocks = result[0]["content"]
        assert len(blocks) == 3
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "function_call"
        assert blocks[2]["type"] == "computer_call"

    def test_empty_messages(self):
        assert normalize_to_canonical([]) == []

    def test_computer_call_output_normalized_to_tool_result(self):
        """computer_call_output blocks from some transcript paths become ToolResultBlock."""
        msgs = [_msg("tool", [
            {"type": "computer_call_output", "call_id": "cc-1", "output": "screenshot data"},
        ])]
        result = normalize_to_canonical(msgs)

        block = result[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "cc-1"
        assert block["content"] == "screenshot data"

    def test_image_url_block_normalized_to_placeholder(self):
        """image_url blocks (base64 screenshots from transcript) become text placeholders."""
        b64 = "iVBORw0KGgo" + "A" * 500  # fake base64
        msgs = [_msg("user", [
            {"type": "text", "text": "[VM state after GUI delegation]"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ])]
        result = normalize_to_canonical(msgs)

        blocks = result[0]["content"]
        assert len(blocks) == 2
        assert blocks[0] == {"type": "text", "text": "[VM state after GUI delegation]"}
        assert blocks[1] == {"type": "text", "text": "[image_url]"}

    def test_image_block_normalized_to_placeholder(self):
        """Anthropic image blocks become text placeholders."""
        msgs = [_msg("user", [
            {"type": "image", "source": {"type": "base64", "data": "AAAA" * 100}},
        ])]
        result = normalize_to_canonical(msgs)
        assert result[0]["content"][0] == {"type": "text", "text": "[image]"}

    def test_input_image_block_normalized_to_placeholder(self):
        """input_image blocks (Responses API) become text placeholders via _normalize_content."""
        msgs = [_msg("user", [
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ])]
        result = normalize_to_canonical(msgs)
        assert result[0]["content"][0] == {"type": "text", "text": "[input_image]"}

    def test_image_url_base64_not_in_normalized_output(self):
        """Regression: base64 data must not leak into TextBlock text after normalization."""
        b64 = "A" * 200_000  # ~150KB of base64
        msgs = [_msg("user", [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ])]
        result = normalize_to_canonical(msgs)
        serialized = json.dumps(result)
        assert "AAAA" not in serialized
        assert "[image_url]" in serialized


# =========================================================================
# canonical_to_responses_api
# =========================================================================


class TestCanonicalToResponsesApi:
    """Test conversion from canonical messages to Responses API flat items."""

    def test_user_text_to_input_text(self):
        msgs = [CanonicalMessage(
            role="user",
            content=[TextBlock(type="text", text="Hello")],
        )]
        items = canonical_to_responses_api(msgs)

        assert len(items) == 1
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "user"
        assert items[0]["content"] == [{"type": "input_text", "text": "Hello"}]

    def test_assistant_text_to_output_text(self):
        msgs = [CanonicalMessage(
            role="assistant",
            content=[TextBlock(type="text", text="Hi there")],
        )]
        items = canonical_to_responses_api(msgs)

        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        assert items[0]["content"] == [{"type": "output_text", "text": "Hi there"}]

    def test_compaction_summary_to_user_with_preamble(self):
        msgs = [CanonicalMessage(
            role="user",
            content=[CompactionSummaryBlock(type="compaction_summary", text="Prior work summary")],
        )]
        items = canonical_to_responses_api(msgs)

        assert len(items) == 1
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "user"
        text = items[0]["content"][0]["text"]
        assert text.startswith(COMPACTION_PREAMBLE)
        assert "Prior work summary" in text

    def test_function_call_id_to_call_id(self):
        msgs = [CanonicalMessage(
            role="assistant",
            content=[FunctionCallBlock(
                type="function_call", id="f-1", name="search", arguments='{"q": "test"}',
            )],
        )]
        items = canonical_to_responses_api(msgs)

        assert items[0]["type"] == "function_call"
        assert items[0]["call_id"] == "f-1"
        assert items[0]["name"] == "search"
        assert items[0]["arguments"] == '{"q": "test"}'

    def test_computer_call_converted_to_text(self):
        """Compacted computer_call blocks become text (screenshots gone, OpenAI rejects placeholders)."""
        msgs = [CanonicalMessage(
            role="assistant",
            content=[ComputerCallBlock(
                type="computer_call", id="cc-1",
                actions=[{"type": "click", "x": 100, "y": 200}],
            )],
        )]
        items = canonical_to_responses_api(msgs)

        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        assert "computer action" in items[0]["content"][0]["text"]

    def test_function_call_output(self):
        msgs = [
            CanonicalMessage(
                role="assistant",
                content=[FunctionCallBlock(
                    type="function_call", id="f-1", name="tool", arguments="{}",
                )],
            ),
            CanonicalMessage(
                role="tool",
                content=[ToolResultBlock(
                    type="tool_result", tool_use_id="f-1", content="result",
                )],
            ),
        ]
        items = canonical_to_responses_api(msgs)

        assert items[1]["type"] == "function_call_output"
        assert items[1]["call_id"] == "f-1"
        assert items[1]["output"] == "result"

    def test_computer_call_output_converted_to_text(self):
        """Compacted computer_call results become text (no screenshot data available)."""
        msgs = [
            CanonicalMessage(
                role="assistant",
                content=[ComputerCallBlock(
                    type="computer_call", id="cc-1",
                    actions=[{"type": "click", "x": 10, "y": 20}],
                )],
            ),
            CanonicalMessage(
                role="tool",
                content=[ToolResultBlock(
                    type="tool_result", tool_use_id="cc-1", content="screenshot",
                )],
            ),
        ]
        items = canonical_to_responses_api(msgs)

        # Both computer_call and its result become text messages
        assert all(i["type"] == "message" for i in items)
        assert "computer action" in items[0]["content"][0]["text"]
        assert "computer result" in items[1]["content"][0]["text"]

    def test_unsigned_thinking_blocks_skipped(self):
        msgs = [CanonicalMessage(
            role="assistant",
            content=[
                ThinkingBlock(type="thinking", thinking="Let me think..."),
                TextBlock(type="text", text="Here's my answer"),
            ],
        )]
        items = canonical_to_responses_api(msgs)

        assert len(items) == 1
        assert items[0]["type"] == "message"
        assert items[0]["content"][0]["text"] == "Here's my answer"

    def test_openai_signed_thinking_block_roundtrips_to_reasoning_item(self):
        msgs = [CanonicalMessage(
            role="assistant",
            content=[
                ThinkingBlock(
                    type="thinking",
                    thinking="Let me think...",
                    thinkingSignature='{"id":"rs_123","type":"reasoning"}',
                ),
                TextBlock(type="text", text="Here's my answer"),
            ],
        )]
        items = canonical_to_responses_api(msgs)

        assert items[0] == {
            "type": "reasoning",
            "id": "rs_123",
            "summary": [{"type": "summary_text", "text": "Let me think..."}],
        }
        assert items[1]["type"] == "message"
        assert items[1]["content"][0]["text"] == "Here's my answer"

    def test_tool_adjacency(self):
        """Tool call should be immediately followed by its output."""
        msgs = [
            CanonicalMessage(
                role="assistant",
                content=[FunctionCallBlock(
                    type="function_call", id="f-1", name="tool", arguments="{}",
                )],
            ),
            CanonicalMessage(
                role="user",
                content=[TextBlock(type="text", text="interleaved message")],
            ),
            CanonicalMessage(
                role="tool",
                content=[ToolResultBlock(
                    type="tool_result", tool_use_id="f-1", content="result",
                )],
            ),
        ]
        items = canonical_to_responses_api(msgs)

        # function_call should be immediately followed by function_call_output
        assert items[0]["type"] == "function_call"
        assert items[1]["type"] == "function_call_output"
        # interleaved message comes after
        assert items[2]["type"] == "message"

    def test_full_conversation_roundtrip(self):
        """Realistic conversation preserves all data through canonical → Responses API."""
        msgs = [
            CanonicalMessage(
                role="user",
                content=[CompactionSummaryBlock(type="compaction_summary", text="Prior context here")],
            ),
            CanonicalMessage(
                role="user",
                content=[TextBlock(type="text", text="Navigate to floor 3")],
            ),
            CanonicalMessage(
                role="assistant",
                content=[
                    TextBlock(type="text", text="I'll click the stairs"),
                    ComputerCallBlock(type="computer_call", id="cc-1", actions=[{"type": "click", "x": 50, "y": 50}]),
                ],
            ),
            CanonicalMessage(
                role="tool",
                content=[ToolResultBlock(type="tool_result", tool_use_id="cc-1", content="screenshot")],
            ),
            CanonicalMessage(
                role="assistant",
                content=[
                    TextBlock(type="text", text="I see floor 3. Let me save a milestone."),
                    FunctionCallBlock(type="function_call", id="f-1", name="milestone", arguments='{"label": "floor3"}'),
                ],
            ),
            CanonicalMessage(
                role="tool",
                content=[ToolResultBlock(type="tool_result", tool_use_id="f-1", content="saved")],
            ),
        ]
        items = canonical_to_responses_api(msgs)

        types = [i["type"] for i in items]
        assert "message" in types  # text, summary, and computer call/result all become messages
        assert "function_call" in types
        assert "function_call_output" in types

        # Computer calls/results become text (screenshots gone after compaction)
        assert "computer_call" not in types
        assert "computer_call_output" not in types

        # Verify summary preamble
        assert COMPACTION_PREAMBLE in items[0]["content"][0]["text"]

        # Verify function call has call_id (not id)
        fn_calls = [i for i in items if i["type"] == "function_call"]
        for c in fn_calls:
            assert "call_id" in c
            assert "id" not in c


# =========================================================================
# canonical_to_anthropic_messages
# =========================================================================


class TestCanonicalToAnthropicMessages:
    """Test conversion from canonical to Anthropic completion format."""

    def test_user_text(self):
        msgs = [CanonicalMessage(
            role="user",
            content=[TextBlock(type="text", text="Hello")],
        )]
        result = canonical_to_anthropic_messages(msgs)

        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"type": "text", "text": "Hello"}]

    def test_assistant_text(self):
        msgs = [CanonicalMessage(
            role="assistant",
            content=[TextBlock(type="text", text="Hi there")],
        )]
        result = canonical_to_anthropic_messages(msgs)

        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == [{"type": "text", "text": "Hi there"}]

    def test_function_call_to_tool_use(self):
        msgs = [CanonicalMessage(
            role="assistant",
            content=[FunctionCallBlock(
                type="function_call", id="f-1", name="search",
                arguments='{"query": "test"}',
            )],
        )]
        result = canonical_to_anthropic_messages(msgs)

        block = result[0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "f-1"
        assert block["name"] == "search"
        assert block["input"] == {"query": "test"}

    def test_computer_call_to_tool_use(self):
        msgs = [CanonicalMessage(
            role="assistant",
            content=[ComputerCallBlock(
                type="computer_call", id="cc-1",
                actions=[{"type": "click", "x": 100, "y": 200}],
            )],
        )]
        result = canonical_to_anthropic_messages(msgs)

        block = result[0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "cc-1"
        assert block["name"] == "computer"
        assert block["input"] == {"actions": [{"type": "click", "x": 100, "y": 200}]}

    def test_tool_result_to_user_tool_result(self):
        """Anthropic tool results are sent as role=user with type=tool_result."""
        msgs = [CanonicalMessage(
            role="tool",
            content=[ToolResultBlock(
                type="tool_result", tool_use_id="f-1", content="done",
            )],
        )]
        result = canonical_to_anthropic_messages(msgs)

        assert result[0]["role"] == "user"
        block = result[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "f-1"
        assert block["content"] == "done"

    def test_tool_result_is_error(self):
        msgs = [CanonicalMessage(
            role="tool",
            content=[ToolResultBlock(
                type="tool_result", tool_use_id="f-1", content="fail", is_error=True,
            )],
        )]
        result = canonical_to_anthropic_messages(msgs)

        assert result[0]["content"][0]["is_error"] is True

    def test_thinking_block_preserved(self):
        msgs = [CanonicalMessage(
            role="assistant",
            content=[
                ThinkingBlock(type="thinking", thinking="hmm", thinkingSignature="sig1"),
                TextBlock(type="text", text="answer"),
            ],
        )]
        result = canonical_to_anthropic_messages(msgs)

        blocks = result[0]["content"]
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["thinking"] == "hmm"
        assert blocks[0]["signature"] == "sig1"
        assert blocks[1]["type"] == "text"

    def test_compaction_summary_in_user_message(self):
        msgs = [CanonicalMessage(
            role="user",
            content=[CompactionSummaryBlock(type="compaction_summary", text="Summary text")],
        )]
        result = canonical_to_anthropic_messages(msgs)

        assert result[0]["role"] == "user"
        text = result[0]["content"][0]["text"]
        assert COMPACTION_PREAMBLE in text
        assert "Summary text" in text

    def test_full_conversation_roundtrip(self):
        """Realistic conversation → Anthropic format preserves all data."""
        msgs = [
            CanonicalMessage(
                role="user",
                content=[TextBlock(type="text", text="Do something")],
            ),
            CanonicalMessage(
                role="assistant",
                content=[
                    TextBlock(type="text", text="OK"),
                    FunctionCallBlock(type="function_call", id="f-1", name="tool", arguments='{"x": 1}'),
                ],
            ),
            CanonicalMessage(
                role="tool",
                content=[ToolResultBlock(type="tool_result", tool_use_id="f-1", content="done")],
            ),
            CanonicalMessage(
                role="assistant",
                content=[TextBlock(type="text", text="All done")],
            ),
        ]
        result = canonical_to_anthropic_messages(msgs)

        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert len(result[1]["content"]) == 2  # text + tool_use
        assert result[1]["content"][1]["type"] == "tool_use"
        assert result[1]["content"][1]["input"] == {"x": 1}
        assert result[2]["role"] == "user"  # Anthropic: tool results as user
        assert result[2]["content"][0]["type"] == "tool_result"
        assert result[3]["role"] == "assistant"


# =========================================================================
# _build_compacted_items integration (standalone)
# =========================================================================


class TestBuildCompactedItemsCanonical:
    """Verify _build_compacted_items outputs canonical messages.

    Uses the standalone version from test_message_ordering to avoid
    instantiating the full agent class.
    """

    def _build(self, summary: str, kept: list[dict]) -> list[CanonicalMessage]:
        from tests.ale_claw.test_message_ordering import _build_compacted_items_standalone
        return _build_compacted_items_standalone(summary, kept)

    def test_returns_canonical_messages(self):
        result = self._build("Summary", [_msg("user", "Hello")])

        for msg in result:
            assert "role" in msg
            assert "content" in msg
            assert isinstance(msg["content"], list)

    def test_summary_is_compaction_summary_block(self):
        result = self._build("My summary", [])

        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "compaction_summary"
        assert result[0]["content"][0]["text"] == "My summary"

    def test_kept_messages_normalized(self):
        kept = [
            _msg("assistant", [
                {"type": "text", "text": "I'll click"},
                {"type": "computer_call", "id": "cc-1", "action": {"type": "click", "x": 1, "y": 2}},
            ]),
            _msg("tool", [
                {"type": "tool_result", "tool_use_id": "cc-1", "content": "ok"},
            ]),
        ]
        result = self._build("Summary", kept)

        # summary + 2 kept messages
        assert len(result) == 3
        # assistant message has normalized blocks
        assistant = result[1]
        assert assistant["role"] == "assistant"
        assert assistant["content"][1]["type"] == "computer_call"
        assert assistant["content"][1]["actions"] == [{"type": "click", "x": 1, "y": 2}]

    def test_trailing_assistant_gets_continuation(self):
        kept = [_msg("assistant", "Last thing")]
        result = self._build("Summary", kept)

        assert result[-1]["role"] == "user"
        assert result[-1]["content"][0]["type"] == "text"
        assert result[-1]["content"][0]["text"] == "[Continue from where you left off.]"

    def test_canonical_to_responses_api_after_build(self):
        """Verify the bridge: canonical → Responses API items produces valid output."""
        kept = [
            _msg("user", "Navigate"),
            _msg("assistant", [
                {"type": "function_call", "id": "f-1", "name": "milestone", "arguments": "{}"},
            ]),
            _msg("tool", [
                {"type": "tool_result", "tool_use_id": "f-1", "content": "saved"},
            ]),
        ]
        canonical = self._build("Prior context", kept)
        items = canonical_to_responses_api(canonical)

        # Should have: summary message + user message + function_call + function_call_output
        types = [i["type"] for i in items]
        assert "message" in types
        assert "function_call" in types
        assert "function_call_output" in types

        # Summary has preamble
        assert COMPACTION_PREAMBLE in items[0]["content"][0]["text"]


# =========================================================================
# TranscriptPolicy
# =========================================================================


class TestTranscriptPolicy:
    """Test TranscriptPolicy dataclass defaults and resolution."""

    def test_defaults(self):
        policy = TranscriptPolicy()
        assert policy.sanitize_mode == "images-only"
        assert policy.drop_thinking_blocks is False
        assert policy.sanitize_thinking_signatures is False
        assert policy.downgrade_openai_reasoning is False
        assert policy.repair_tool_use_result_pairing is True
        assert policy.validate_anthropic_turns is True

    def test_frozen(self):
        """TranscriptPolicy is immutable."""
        policy = TranscriptPolicy()
        try:
            policy.drop_thinking_blocks = True  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestGetTranscriptPolicy:
    """Test per-provider policy resolution."""

    def test_anthropic_provider_prefix(self):
        policy = get_transcript_policy("anthropic/claude-sonnet-4-20250514")
        assert policy.sanitize_mode == "full"
        assert policy.drop_thinking_blocks is True
        assert policy.validate_anthropic_turns is True
        assert policy.downgrade_openai_reasoning is False

    def test_anthropic_claude_model(self):
        policy = get_transcript_policy("claude-opus-4-6")
        assert policy.sanitize_mode == "full"
        assert policy.drop_thinking_blocks is True
        assert policy.validate_anthropic_turns is True

    def test_openai_provider_prefix(self):
        policy = get_transcript_policy("openai/gpt-5.4")
        assert policy.sanitize_mode == "images-only"
        assert policy.downgrade_openai_reasoning is True
        assert policy.drop_thinking_blocks is False
        assert policy.validate_anthropic_turns is False

    def test_openai_gpt_model(self):
        policy = get_transcript_policy("gpt-5.4")
        assert policy.downgrade_openai_reasoning is True

    def test_openai_o_series(self):
        policy = get_transcript_policy("o3-mini")
        assert policy.downgrade_openai_reasoning is True

    def test_resolved_model_input(self):
        policy = get_transcript_policy(resolve_model("openai/gpt-5.4"))
        assert policy.downgrade_openai_reasoning is True
        assert policy.validate_anthropic_turns is False

    def test_gemini(self):
        policy = get_transcript_policy("gemini/gemini-2.5-pro")
        assert policy.sanitize_mode == "full"
        assert policy.sanitize_thinking_signatures is True
        assert policy.drop_thinking_blocks is False
        assert policy.downgrade_openai_reasoning is False

    def test_google_vertex(self):
        policy = get_transcript_policy("vertex/gemini-2.5-pro")
        assert policy.sanitize_thinking_signatures is True

    def test_unknown_model_defaults(self):
        policy = get_transcript_policy("some-unknown-model")
        assert policy.drop_thinking_blocks is False
        assert policy.sanitize_thinking_signatures is False
        assert policy.downgrade_openai_reasoning is False
        assert policy.repair_tool_use_result_pairing is True
        assert policy.validate_anthropic_turns is True


# =========================================================================
# drop_thinking_blocks
# =========================================================================


class TestDropThinkingBlocks:
    """Test stripping thinking blocks from assistant messages."""

    def test_strips_thinking_from_assistant(self):
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="Let me think..."),
                TextBlock(type="text", text="Hello"),
            ]},
        ]
        result = drop_thinking_blocks(msgs)
        assert len(result) == 1
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello"

    def test_preserves_turn_with_empty_text(self):
        """When all blocks are thinking, replace with empty text to preserve turn."""
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="Deep thought"),
            ]},
        ]
        result = drop_thinking_blocks(msgs)
        assert len(result) == 1
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == ""

    def test_no_op_when_no_thinking(self):
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                TextBlock(type="text", text="Hello"),
            ]},
        ]
        result = drop_thinking_blocks(msgs)
        assert result is msgs  # Reference equality — nothing changed

    def test_leaves_user_messages_untouched(self):
        msgs: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="Hi")]},
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="Hmm"),
                TextBlock(type="text", text="Response"),
            ]},
        ]
        result = drop_thinking_blocks(msgs)
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 1  # User message unchanged
        assert len(result[1]["content"]) == 1  # Thinking stripped

    def test_multiple_thinking_blocks(self):
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="First"),
                ThinkingBlock(type="thinking", thinking="Second"),
                TextBlock(type="text", text="Answer"),
            ]},
        ]
        result = drop_thinking_blocks(msgs)
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["text"] == "Answer"


# =========================================================================
# sanitize_thinking_signatures
# =========================================================================


class TestSanitizeThinkingSignatures:
    """Test removing thinkingSignature from thinking blocks."""

    def test_removes_signature(self):
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(
                    type="thinking",
                    thinking="reasoning",
                    thinkingSignature="abc123",
                ),
            ]},
        ]
        result = sanitize_thinking_signatures(msgs)
        block = result[0]["content"][0]
        assert block["type"] == "thinking"
        assert block["thinking"] == "reasoning"
        assert "thinkingSignature" not in block

    def test_no_op_without_signature(self):
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="reasoning"),
            ]},
        ]
        result = sanitize_thinking_signatures(msgs)
        assert result is msgs  # Reference equality

    def test_leaves_non_thinking_blocks(self):
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(
                    type="thinking",
                    thinking="thought",
                    thinkingSignature="sig",
                ),
                TextBlock(type="text", text="Hello"),
            ]},
        ]
        result = sanitize_thinking_signatures(msgs)
        assert len(result[0]["content"]) == 2
        assert "thinkingSignature" not in result[0]["content"][0]
        assert result[0]["content"][1]["text"] == "Hello"

    def test_leaves_user_messages_untouched(self):
        msgs: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="Hi")]},
        ]
        result = sanitize_thinking_signatures(msgs)
        assert result is msgs


# =========================================================================
# downgrade_openai_reasoning
# =========================================================================


def _openai_sig(item_id: str = "rs_abc123", sig_type: str = "reasoning") -> str:
    """Build a valid OpenAI reasoning signature JSON string."""
    return json.dumps({"id": item_id, "type": sig_type})


class TestDowngradeOpenaiReasoning:
    """Test dropping orphaned OpenAI reasoning blocks."""

    def test_drops_orphaned_reasoning(self):
        """Thinking block with valid OpenAI sig and no following content → dropped."""
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(
                    type="thinking",
                    thinking="",
                    thinkingSignature=_openai_sig(),
                ),
            ]},
        ]
        result = downgrade_openai_reasoning(msgs)
        # Entire message removed (all blocks dropped)
        assert len(result) == 0

    def test_keeps_reasoning_with_following_content(self):
        """Thinking block with valid sig but followed by text → kept."""
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(
                    type="thinking",
                    thinking="reasoning",
                    thinkingSignature=_openai_sig(),
                ),
                TextBlock(type="text", text="Answer"),
            ]},
        ]
        result = downgrade_openai_reasoning(msgs)
        assert result is msgs  # Nothing changed

    def test_keeps_non_openai_thinking(self):
        """Thinking block without OpenAI sig → kept (may be Anthropic)."""
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="deep thought"),
            ]},
        ]
        result = downgrade_openai_reasoning(msgs)
        assert result is msgs

    def test_keeps_thinking_with_anthropic_signature(self):
        """Thinking block with non-JSON signature → kept."""
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(
                    type="thinking",
                    thinking="thought",
                    thinkingSignature="base64encodedstring==",
                ),
            ]},
        ]
        result = downgrade_openai_reasoning(msgs)
        assert result is msgs

    def test_drops_only_trailing_orphaned(self):
        """Multiple thinking blocks — only trailing orphaned ones are dropped."""
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(
                    type="thinking",
                    thinking="first",
                    thinkingSignature=_openai_sig("rs_1"),
                ),
                TextBlock(type="text", text="middle"),
                ThinkingBlock(
                    type="thinking",
                    thinking="orphaned",
                    thinkingSignature=_openai_sig("rs_2"),
                ),
            ]},
        ]
        result = downgrade_openai_reasoning(msgs)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "thinking"
        assert result[0]["content"][1]["type"] == "text"

    def test_leaves_user_messages(self):
        msgs: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="Hi")]},
        ]
        result = downgrade_openai_reasoning(msgs)
        assert result is msgs

    def test_dict_signature(self):
        """Signature as dict (not JSON string) also recognized."""
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(
                    type="thinking",
                    thinking="",
                    thinkingSignature={"id": "rs_x", "type": "reasoning"},  # type: ignore[typeddict-item]
                ),
            ]},
        ]
        result = downgrade_openai_reasoning(msgs)
        assert len(result) == 0

    def test_invalid_json_signature_kept(self):
        """Malformed JSON string in signature → not recognized as OpenAI → kept."""
        msgs: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(
                    type="thinking",
                    thinking="thought",
                    thinkingSignature="{invalid json",
                ),
            ]},
        ]
        result = downgrade_openai_reasoning(msgs)
        assert result is msgs


# =========================================================================
# sanitize_items with policy
# =========================================================================


class TestSanitizeItemsPolicy:
    """Test policy-driven sanitize_items pipeline."""

    def test_default_anthropic_policy_drops_thinking(self):
        """Default Anthropic policy strips thinking blocks."""
        msgs: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="Hi")]},
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="hmm"),
                TextBlock(type="text", text="Hello"),
            ]},
        ]
        result = sanitize_items(msgs, "anthropic")
        # Thinking should be stripped
        assistant = [m for m in result if m["role"] == "assistant"]
        assert len(assistant) == 1
        assert all(
            b["type"] != "thinking" for b in assistant[0]["content"]
        )

    def test_explicit_policy_overrides_default(self):
        """Explicit policy with all passes disabled → no thinking stripped."""
        msgs: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="Hi")]},
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="hmm"),
                TextBlock(type="text", text="Hello"),
            ]},
        ]
        policy = TranscriptPolicy(
            drop_thinking_blocks=False,
            repair_tool_use_result_pairing=False,
            validate_anthropic_turns=False,
        )
        result = sanitize_items(msgs, "anthropic", policy=policy)
        # Anthropic adapter still converts thinking → {type: "thinking", ...}
        assistant = [m for m in result if m["role"] == "assistant"]
        assert len(assistant) == 1
        assert any(
            b.get("type") == "thinking" for b in assistant[0]["content"]
        )

    def test_openai_default_policy_no_thinking_drop(self):
        """Default OpenAI policy does NOT drop thinking (not needed)."""
        msgs: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="Hi")]},
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="hmm"),
                TextBlock(type="text", text="Hello"),
            ]},
        ]
        # OpenAI Responses adapter skips thinking blocks at format conversion
        # time, so drop_thinking_blocks=False is correct
        result = sanitize_items(msgs, "openai-responses")
        # Thinking blocks are skipped by the Responses adapter regardless
        assert isinstance(result, list)

    def test_all_passes_disabled_is_noop(self):
        """Policy with all passes off → only format conversion runs."""
        msgs: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="Hi")]},
            {"role": "assistant", "content": [
                TextBlock(type="text", text="Hello"),
            ]},
        ]
        policy = TranscriptPolicy(
            repair_tool_use_result_pairing=False,
            validate_anthropic_turns=False,
        )
        result = sanitize_items(msgs, "anthropic", policy=policy)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_model_resolves_anthropic_policy_without_manual_injection(self):
        msgs: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="Hi")]},
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="hmm"),
                TextBlock(type="text", text="Hello"),
            ]},
        ]
        result = sanitize_items(
            msgs,
            target="anthropic",
            model="anthropic/claude-sonnet-4-20250514",
        )
        assistant = [m for m in result if m["role"] == "assistant"]
        assert len(assistant) == 1
        assert all(b["type"] != "thinking" for b in assistant[0]["content"])

    def test_model_resolves_openai_policy_without_manual_injection(self):
        msgs: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="Hi")]},
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="hmm"),
                TextBlock(type="text", text="Hello"),
            ]},
        ]
        result = sanitize_items(msgs, model="openai/gpt-5.4")
        assert isinstance(result, list)

    def test_model_resolves_gemini_policy_without_manual_injection(self):
        msgs: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="Hi")]},
            {"role": "assistant", "content": [
                ThinkingBlock(
                    type="thinking",
                    thinking="hmm",
                    thinkingSignature="raw-signature",
                ),
                TextBlock(type="text", text="Hello"),
            ]},
        ]
        result = sanitize_items(
            msgs,
            target="anthropic",
            model="google/gemini-3-pro",
        )
        assistant = [m for m in result if m["role"] == "assistant"]
        thinking_blocks = [
            b for b in assistant[0]["content"] if b.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert "thinkingSignature" not in thinking_blocks[0]


# =========================================================================
# Compaction round-trip: images must not explode token count
# =========================================================================


class TestCompactionImageRoundTrip:
    """Regression tests for the compaction token explosion bug.

    When transcript messages contain base64 image_url blocks, the compaction
    rebuild path (normalize_to_canonical → sanitize_items) must not inflate
    the token count by stringifying raw base64 into TextBlocks.
    """

    def test_normalize_strips_base64_from_image_url(self):
        """Token estimate after normalization must not include raw base64."""
        from ale_run.agents.ale_claw.harness.context.context import estimate_messages_tokens

        b64 = "A" * 200_000  # ~150KB fake screenshot
        raw_msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "check the screen"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I see the desktop"},
            ]},
        ]

        canonical = normalize_to_canonical(raw_msgs)
        tokens = estimate_messages_tokens(canonical)

        # Without the fix, base64 as text → ~50K tokens.
        # With the fix, [image_url] placeholder → <100 tokens total.
        assert tokens < 500, f"Token estimate {tokens} suggests base64 leaked into text"

    def test_strip_images_from_messages(self):
        """_strip_images_from_messages replaces image blocks with placeholders."""
        from ale_run.agents.ale_claw.harness.agent_loop import _strip_images_from_messages

        b64 = "A" * 100_000
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "delegation done"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
            {"role": "assistant", "content": "plain text"},
            {"role": "user", "content": "also plain"},
        ]

        result = _strip_images_from_messages(messages)

        assert result[0]["content"][0] == {"type": "text", "text": "delegation done"}
        assert result[0]["content"][1] == {"type": "text", "text": "[image_url]"}
        # Non-image messages pass through unchanged
        assert result[1]["content"] == "plain text"
        assert result[2]["content"] == "also plain"
        # Original not mutated
        assert messages[0]["content"][1]["type"] == "image_url"

    def test_strip_images_preserves_non_image_blocks(self):
        """Messages without images pass through unchanged (reference equality)."""
        from ale_run.agents.ale_claw.harness.agent_loop import _strip_images_from_messages

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        result = _strip_images_from_messages(messages)
        assert result[0] is messages[0]
        assert result[1] is messages[1]
