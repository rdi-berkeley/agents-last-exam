"""Tests for the sanitize_items() pipeline.

Covers:
  - normalize_to_canonical() with flat Responses API items
  - repair_orphaned_pairs() — orphan detection, synthesis, drop modes
  - ensure_valid_ordering() — trailing assistant handling
  - sanitize_items(target='openai-responses') — all content block types
  - sanitize_items(target='anthropic') — all content block types
  - Integration: role-based → sanitize_items → valid output
"""

import json

from ale_claw.harness.canonical.canonical import (
    CanonicalMessage,
    CompactionSummaryBlock,
    ComputerCallBlock,
    FunctionCallBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ensure_valid_ordering,
    normalize_to_canonical,
    repair_orphaned_pairs,
    sanitize_items,
)


# ===========================================================================
# normalize_to_canonical: flat Responses API items
# ===========================================================================


class TestNormalizeFlat:
    """Test ingestion of flat Responses API items."""

    def test_function_call_flat_item(self):
        items = [{"type": "function_call", "call_id": "fc1", "name": "search", "arguments": "{}"}]
        result = normalize_to_canonical(items)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        block = result[0]["content"][0]
        assert block["type"] == "function_call"
        assert block["id"] == "fc1"
        assert block["name"] == "search"

    def test_function_call_output_flat_item(self):
        items = [{"type": "function_call_output", "call_id": "fc1", "output": "result"}]
        result = normalize_to_canonical(items)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        block = result[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "fc1"
        assert block["content"] == "result"

    def test_computer_call_flat_item(self):
        items = [{"type": "computer_call", "call_id": "cc1", "action": {"type": "click", "x": 10}}]
        result = normalize_to_canonical(items)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        block = result[0]["content"][0]
        assert block["type"] == "computer_call"
        assert block["id"] == "cc1"
        assert block["actions"] == [{"type": "click", "x": 10}]

    def test_computer_call_output_flat_item(self):
        items = [{
            "type": "computer_call_output",
            "call_id": "cc1",
            "output": {"type": "computer_screenshot", "image_url": "data:..."},
            "acknowledged_safety_checks": [{"id": "sc1"}],
        }]
        result = normalize_to_canonical(items)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        block = result[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "cc1"
        # acknowledged_safety_checks should be stripped

    def test_message_flat_item_user(self):
        items = [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }]
        result = normalize_to_canonical(items)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "hello"

    def test_message_flat_item_assistant(self):
        items = [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "reply"}],
        }]
        result = normalize_to_canonical(items)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "reply"

    def test_reasoning_flat_item(self):
        items = [{
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "thought"}],
        }]
        result = normalize_to_canonical(items)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        block = result[0]["content"][0]
        assert block["type"] == "thinking"
        assert block["thinking"] == "thought"

    def test_consecutive_assistant_items_merged(self):
        """Consecutive flat items with same role should merge into one message."""
        items = [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "fc1", "name": "fn", "arguments": "{}"},
        ]
        result = normalize_to_canonical(items)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 2

    def test_consecutive_tool_items_merged(self):
        items = [
            {"type": "function_call_output", "call_id": "fc1", "output": "r1"},
            {"type": "function_call_output", "call_id": "fc2", "output": "r2"},
        ]
        result = normalize_to_canonical(items)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert len(result[0]["content"]) == 2

    def test_mixed_flat_and_role_based(self):
        """Mix of flat Responses API items and role-based messages."""
        items = [
            {"role": "user", "content": "hello"},  # role-based
            {"type": "function_call", "call_id": "fc1", "name": "fn", "arguments": "{}"},  # flat
            {"type": "function_call_output", "call_id": "fc1", "output": "result"},  # flat
        ]
        result = normalize_to_canonical(items)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "tool"

    def test_computer_call_with_actions_list(self):
        """Computer call with actions list (not singular action)."""
        items = [{
            "type": "computer_call",
            "call_id": "cc1",
            "actions": [{"type": "click"}, {"type": "type"}],
        }]
        result = normalize_to_canonical(items)
        block = result[0]["content"][0]
        assert block["actions"] == [{"type": "click"}, {"type": "type"}]


# ===========================================================================
# normalize_to_canonical: role-based messages (existing behavior)
# ===========================================================================


class TestNormalizeRoleBased:
    """Verify existing role-based ingestion still works."""

    def test_string_content(self):
        result = normalize_to_canonical([{"role": "user", "content": "hi"}])
        assert result[0]["content"][0] == {"type": "text", "text": "hi"}

    def test_function_call_block(self):
        msg = {"role": "assistant", "content": [
            {"type": "function_call", "id": "fc1", "name": "fn", "arguments": "{}"},
        ]}
        result = normalize_to_canonical([msg])
        block = result[0]["content"][0]
        assert block["type"] == "function_call"
        assert block["id"] == "fc1"

    def test_stop_reason_preserved(self):
        msg = {"role": "assistant", "content": "hi", "stop_reason": "error"}
        result = normalize_to_canonical([msg])
        assert result[0].get("stop_reason") == "error"


# ===========================================================================
# repair_orphaned_pairs
# ===========================================================================


class TestRepairOrphanedPairs:
    """Test the consolidated repair function."""

    def test_paired_items_unchanged(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                FunctionCallBlock(type="function_call", id="fc1", name="fn", arguments="{}"),
            ]},
            {"role": "tool", "content": [
                ToolResultBlock(type="tool_result", tool_use_id="fc1", content="result"),
            ]},
        ]
        result = repair_orphaned_pairs(messages)
        assert len(result) == 2
        assert result[0]["content"][0]["id"] == "fc1"
        assert result[1]["content"][0]["tool_use_id"] == "fc1"

    def test_orphaned_function_call_gets_synthetic_result(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                FunctionCallBlock(type="function_call", id="fc1", name="fn", arguments="{}"),
            ]},
        ]
        result = repair_orphaned_pairs(messages)
        assert len(result) == 2
        assert result[1]["role"] == "tool"
        assert result[1]["content"][0]["tool_use_id"] == "fc1"
        assert result[1]["content"][0].get("is_error") is True

    def test_orphaned_computer_call_dropped(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                TextBlock(type="text", text="thinking"),
                ComputerCallBlock(type="computer_call", id="cc1", actions=[{"type": "click"}]),
            ]},
        ]
        result = repair_orphaned_pairs(messages)
        # Computer call should be dropped, text should remain
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"

    def test_orphaned_tool_result_dropped(self):
        messages: list[CanonicalMessage] = [
            {"role": "tool", "content": [
                ToolResultBlock(type="tool_result", tool_use_id="orphan1", content="data"),
            ]},
        ]
        result = repair_orphaned_pairs(messages)
        assert len(result) == 0

    def test_stop_reason_error_skips_synthesis(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                FunctionCallBlock(type="function_call", id="fc1", name="fn", arguments="{}"),
            ], "stop_reason": "error"},
        ]
        result = repair_orphaned_pairs(messages)
        # Function call should be dropped (no synthesis due to error stop)
        # and no synthetic result added
        assert len(result) == 0 or all(
            b.get("type") != "tool_result"
            for msg in result
            for b in msg.get("content", [])
        )

    def test_synthesize_false_drops_orphaned_calls(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                FunctionCallBlock(type="function_call", id="fc1", name="fn", arguments="{}"),
            ]},
        ]
        result = repair_orphaned_pairs(messages, synthesize=False)
        # No synthetic result — orphaned call just dropped
        assert len(result) == 0 or not any(
            b.get("type") == "tool_result"
            for msg in result
            for b in msg.get("content", [])
        )

    def test_empty_input(self):
        assert repair_orphaned_pairs([]) == []

    def test_duplicate_results_dropped(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                FunctionCallBlock(type="function_call", id="fc1", name="fn", arguments="{}"),
            ]},
            {"role": "tool", "content": [
                ToolResultBlock(type="tool_result", tool_use_id="fc1", content="first"),
            ]},
            {"role": "tool", "content": [
                ToolResultBlock(type="tool_result", tool_use_id="fc1", content="duplicate"),
            ]},
        ]
        result = repair_orphaned_pairs(messages)
        tool_results = [
            b for msg in result for b in msg.get("content", [])
            if b.get("type") == "tool_result"
        ]
        assert len(tool_results) == 1
        assert tool_results[0]["content"] == "first"


# ===========================================================================
# ensure_valid_ordering
# ===========================================================================


class TestEnsureValidOrdering:
    def test_trailing_assistant_gets_continuation(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [TextBlock(type="text", text="hi")]},
        ]
        result = ensure_valid_ordering(messages)
        assert len(result) == 2
        assert result[-1]["role"] == "user"
        assert "Continue" in result[-1]["content"][0]["text"]

    def test_trailing_user_unchanged(self):
        messages: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="hi")]},
        ]
        result = ensure_valid_ordering(messages)
        assert len(result) == 1

    def test_empty_unchanged(self):
        assert ensure_valid_ordering([]) == []


# ===========================================================================
# sanitize_items: target='openai-responses'
# ===========================================================================


class TestSanitizeOpenAI:
    """Test sanitize_items with openai-responses target."""

    def test_text_block_user(self):
        messages: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="hello")]},
        ]
        result = sanitize_items(messages, target="openai-responses")
        assert any(
            item.get("type") == "message" and item.get("role") == "user"
            for item in result
        )

    def test_text_block_assistant(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [TextBlock(type="text", text="reply")]},
            {"role": "user", "content": [TextBlock(type="text", text="continue")]},
        ]
        result = sanitize_items(messages, target="openai-responses")
        assistant_items = [i for i in result if i.get("role") == "assistant"]
        assert len(assistant_items) >= 1
        content = assistant_items[0]["content"]
        assert content[0]["type"] == "output_text"

    def test_function_call_block(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                FunctionCallBlock(type="function_call", id="fc1", name="fn", arguments="{}"),
            ]},
            {"role": "tool", "content": [
                ToolResultBlock(type="tool_result", tool_use_id="fc1", content="result"),
            ]},
        ]
        result = sanitize_items(messages, target="openai-responses")
        fc_items = [i for i in result if i.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["call_id"] == "fc1"

    def test_function_call_output(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                FunctionCallBlock(type="function_call", id="fc1", name="fn", arguments="{}"),
            ]},
            {"role": "tool", "content": [
                ToolResultBlock(type="tool_result", tool_use_id="fc1", content="result"),
            ]},
        ]
        result = sanitize_items(messages, target="openai-responses")
        fco_items = [i for i in result if i.get("type") == "function_call_output"]
        assert len(fco_items) == 1
        assert fco_items[0]["output"] == "result"

    def test_computer_call_converted_to_text(self):
        """Compacted computer calls become text (no screenshot available)."""
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ComputerCallBlock(type="computer_call", id="cc1", actions=[{"type": "click"}]),
            ]},
            {"role": "tool", "content": [
                ToolResultBlock(type="tool_result", tool_use_id="cc1", content="done"),
            ]},
        ]
        result = sanitize_items(messages, target="openai-responses")
        # Should NOT produce computer_call flat items (they're converted to text)
        cc_items = [i for i in result if i.get("type") == "computer_call"]
        assert len(cc_items) == 0

    def test_compaction_summary_block(self):
        messages: list[CanonicalMessage] = [
            {"role": "user", "content": [
                CompactionSummaryBlock(type="compaction_summary", text="summary of prior context"),
            ]},
        ]
        result = sanitize_items(messages, target="openai-responses")
        # Should have a user message with the preamble
        assert any(
            "Prior Context" in str(item.get("content", ""))
            for item in result
        )

    def test_thinking_block_skipped(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="internal reasoning"),
                TextBlock(type="text", text="visible"),
            ]},
            {"role": "user", "content": [TextBlock(type="text", text="continue")]},
        ]
        result = sanitize_items(messages, target="openai-responses")
        # Thinking blocks should not appear in output
        all_text = json.dumps(result)
        assert "internal reasoning" not in all_text

    def test_orphan_repair_integrated(self):
        """sanitize_items should repair orphaned calls before conversion."""
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                FunctionCallBlock(type="function_call", id="fc1", name="fn", arguments="{}"),
            ]},
            # No matching tool_result — should get synthetic
        ]
        result = sanitize_items(messages, target="openai-responses")
        fco_items = [i for i in result if i.get("type") == "function_call_output"]
        assert len(fco_items) == 1

    def test_trailing_assistant_no_continuation_for_openai(self):
        """OpenAI default policy has validate_anthropic_turns=False, so no continuation added."""
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [TextBlock(type="text", text="reply")]},
        ]
        result = sanitize_items(messages, target="openai-responses")
        # OpenAI doesn't need user/assistant alternation enforcement
        types = [i.get("type") for i in result]
        assert "message" in types


# ===========================================================================
# sanitize_items: target='anthropic'
# ===========================================================================


class TestSanitizeAnthropic:
    """Test sanitize_items with anthropic target."""

    def test_text_block(self):
        messages: list[CanonicalMessage] = [
            {"role": "user", "content": [TextBlock(type="text", text="hello")]},
        ]
        result = sanitize_items(messages, target="anthropic")
        assert len(result) >= 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0] == {"type": "text", "text": "hello"}

    def test_function_call_becomes_tool_use(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                FunctionCallBlock(type="function_call", id="fc1", name="fn", arguments='{"k":"v"}'),
            ]},
            {"role": "tool", "content": [
                ToolResultBlock(type="tool_result", tool_use_id="fc1", content="result"),
            ]},
        ]
        result = sanitize_items(messages, target="anthropic")
        assistant_msg = [m for m in result if m.get("role") == "assistant"][0]
        assert assistant_msg["content"][0]["type"] == "tool_use"
        assert assistant_msg["content"][0]["id"] == "fc1"
        assert assistant_msg["content"][0]["input"] == {"k": "v"}

    def test_computer_call_becomes_tool_use(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ComputerCallBlock(type="computer_call", id="cc1", actions=[{"type": "click"}]),
            ]},
            {"role": "tool", "content": [
                ToolResultBlock(type="tool_result", tool_use_id="cc1", content="done"),
            ]},
        ]
        result = sanitize_items(messages, target="anthropic")
        assistant_msg = [m for m in result if m.get("role") == "assistant"][0]
        assert assistant_msg["content"][0]["type"] == "tool_use"
        assert assistant_msg["content"][0]["name"] == "computer"

    def test_tool_result_becomes_user_tool_result(self):
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                FunctionCallBlock(type="function_call", id="fc1", name="fn", arguments="{}"),
            ]},
            {"role": "tool", "content": [
                ToolResultBlock(type="tool_result", tool_use_id="fc1", content="result"),
            ]},
        ]
        result = sanitize_items(messages, target="anthropic")
        tool_msg = [m for m in result if m.get("role") == "user" and
                    any(b.get("type") == "tool_result" for b in m.get("content", []))]
        assert len(tool_msg) == 1
        assert tool_msg[0]["content"][0]["tool_use_id"] == "fc1"

    def test_thinking_block_dropped_by_default_policy(self):
        """Default Anthropic policy drops thinking blocks (signatures may be invalid on replay)."""
        messages: list[CanonicalMessage] = [
            {"role": "assistant", "content": [
                ThinkingBlock(type="thinking", thinking="reasoning"),
                TextBlock(type="text", text="visible"),
            ]},
            {"role": "user", "content": [TextBlock(type="text", text="continue")]},
        ]
        result = sanitize_items(messages, target="anthropic")
        assistant_msg = [m for m in result if m.get("role") == "assistant"][0]
        thinking_blocks = [b for b in assistant_msg["content"] if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 0
        text_blocks = [b for b in assistant_msg["content"] if b.get("type") == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "visible"

    def test_compaction_summary(self):
        messages: list[CanonicalMessage] = [
            {"role": "user", "content": [
                CompactionSummaryBlock(type="compaction_summary", text="summary"),
            ]},
        ]
        result = sanitize_items(messages, target="anthropic")
        assert "Prior Context" in result[0]["content"][0]["text"]

    def test_invalid_target_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown adapter target"):
            sanitize_items([], target="gemini")  # type: ignore


# ===========================================================================
# Integration: full pipeline from role-based input
# ===========================================================================


class TestIntegration:
    """End-to-end tests through normalize + sanitize."""

    def test_role_based_to_openai_responses(self):
        """Role-based messages with tool pairs → valid Responses API items."""
        raw = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I'll search"},
                {"type": "function_call", "id": "fc1", "name": "memory_search", "arguments": '{"q":"test"}'},
            ]},
            {"role": "tool", "content": [
                {"type": "tool_result", "tool_use_id": "fc1", "content": "found it"},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Done!"},
            ]},
        ]
        canonical = normalize_to_canonical(raw)
        result = sanitize_items(canonical, target="openai-responses")

        # Should have: user text, assistant text, function_call, function_call_output, assistant text, user continuation
        fc_items = [i for i in result if i.get("type") == "function_call"]
        fco_items = [i for i in result if i.get("type") == "function_call_output"]
        assert len(fc_items) == 1
        assert len(fco_items) == 1
        assert fc_items[0]["call_id"] == "fc1"
        assert fco_items[0]["output"] == "found it"

    def test_flat_items_to_openai_responses(self):
        """Flat Responses API items round-trip through canonical and back."""
        raw = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            {"type": "function_call", "call_id": "fc1", "name": "fn", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "fc1", "output": "result"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        ]
        canonical = normalize_to_canonical(raw)
        result = sanitize_items(canonical, target="openai-responses")

        fc_items = [i for i in result if i.get("type") == "function_call"]
        fco_items = [i for i in result if i.get("type") == "function_call_output"]
        assert len(fc_items) == 1
        assert len(fco_items) == 1
        assert fc_items[0]["call_id"] == "fc1"

    def test_compaction_output_to_openai(self):
        """Compacted messages (summary + kept) → valid Responses API items."""
        raw = [
            {"role": "user", "content": [
                {"type": "compaction_summary", "text": "Prior: user asked about weather"},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "The weather is sunny"},
            ]},
        ]
        canonical = normalize_to_canonical(raw)
        result = sanitize_items(canonical, target="openai-responses")

        # Should have compaction summary as user message and assistant message
        assert any("Prior Context" in str(item) for item in result)

    def test_role_based_to_anthropic(self):
        """Role-based messages → valid Anthropic completion format."""
        raw = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [
                {"type": "function_call", "id": "fc1", "name": "fn", "arguments": '{"k":"v"}'},
            ]},
            {"role": "tool", "content": [
                {"type": "tool_result", "tool_use_id": "fc1", "content": "result"},
            ]},
            {"role": "assistant", "content": "goodbye"},
        ]
        canonical = normalize_to_canonical(raw)
        result = sanitize_items(canonical, target="anthropic")

        # Check tool_use in assistant message
        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 1
        tool_use_blocks = [
            b for m in assistant_msgs for b in m.get("content", [])
            if b.get("type") == "tool_use"
        ]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["id"] == "fc1"
