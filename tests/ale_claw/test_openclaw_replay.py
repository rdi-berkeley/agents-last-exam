"""Tests for transcript replay — cross-run continuity.

Tests build_replay_messages, sanitize_history, limit_history_turns,
and convert_to_responses_api_items.
"""

from ale_claw.harness.session import (
    SessionManager,
    TranscriptEntry,
    build_replay_messages,
    convert_to_responses_api_items,
    limit_history_turns,
    sanitize_history,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg_entry(
    role: str,
    content,
    entry_id: str = "msg-001",
    parent_id: str | None = None,
    usage: dict | None = None,
    stop_reason: str | None = None,
    api: str | None = None,
) -> TranscriptEntry:
    """Create a message TranscriptEntry for testing."""
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    msg_data: dict = {"role": role, "content": content}
    if usage:
        msg_data["usage"] = usage
    if stop_reason:
        msg_data["stopReason"] = stop_reason
    if api:
        msg_data["api"] = api
    return TranscriptEntry(
        type="message",
        id=entry_id,
        parent_id=parent_id,
        timestamp="2026-03-16T10:00:00Z",
        data={"message": msg_data},
    )


def _session_entry(run_number: int = 1) -> TranscriptEntry:
    return TranscriptEntry(
        type="session",
        id=f"sess-{run_number:03d}",
        parent_id=None,
        timestamp="2026-03-16T10:00:00Z",
        data={"version": 1, "task_id": "test", "run_number": run_number, "model": "claude"},
    )


def _compaction_entry(
    summary: str,
    first_kept_id: str,
    entry_id: str = "cmp-001",
) -> TranscriptEntry:
    return TranscriptEntry(
        type="compaction",
        id=entry_id,
        parent_id=None,
        timestamp="2026-03-16T10:00:00Z",
        data={"summary": summary, "firstKeptEntryId": first_kept_id, "tokensBefore": 90000},
    )


# ---------------------------------------------------------------------------
# build_replay_messages
# ---------------------------------------------------------------------------


class TestBuildReplayMessages:
    def test_converts_message_entries(self):
        entries = [
            _session_entry(),
            _msg_entry("user", "Do the task", entry_id="msg-001"),
            _msg_entry("assistant", "I'll click", entry_id="msg-002", parent_id="msg-001"),
        ]
        messages = build_replay_messages(entries)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_skips_session_entries(self):
        entries = [
            _session_entry(1),
            _msg_entry("user", "Hello", entry_id="msg-001"),
            _session_entry(2),
            _msg_entry("user", "World", entry_id="msg-002"),
        ]
        messages = build_replay_messages(entries)
        assert len(messages) == 2
        assert all(m["role"] == "user" for m in messages)

    def test_strips_stale_metadata(self):
        entries = [
            _msg_entry(
                "assistant",
                "Hello",
                entry_id="msg-001",
                usage={"input": 100, "output": 20},
                stop_reason="end_turn",
                api="openai-responses",
            ),
        ]
        messages = build_replay_messages(entries)
        msg = messages[0]
        assert "usage" not in msg
        assert "stopReason" not in msg
        assert "api" not in msg

    def test_maps_toolResult_to_user(self):
        entries = [
            _msg_entry(
                "toolResult",
                [{"type": "text", "text": "result data"}],
                entry_id="msg-001",
            ),
        ]
        messages = build_replay_messages(entries)
        assert messages[0]["role"] == "user"

    def test_compaction_replaces_prior_messages(self):
        entries = [
            _session_entry(),
            _msg_entry("user", "Old message 1", entry_id="msg-001"),
            _msg_entry("assistant", "Old reply 1", entry_id="msg-002"),
            _msg_entry("user", "Kept message", entry_id="msg-003"),
            _compaction_entry("Summary of old context", "msg-003"),
            _msg_entry("assistant", "Reply to kept", entry_id="msg-004"),
        ]
        messages = build_replay_messages(entries)
        # Should have: compaction summary + kept message + reply
        assert len(messages) == 3
        assert "[Compaction summary]" in messages[0]["content"][0]["text"]
        assert "Summary of old context" in messages[0]["content"][0]["text"]

    def test_compaction_no_matching_id(self):
        """When firstKeptEntryId doesn't match, summary is prepended."""
        entries = [
            _msg_entry("user", "Hello", entry_id="msg-001"),
            _compaction_entry("Summary", "nonexistent-id"),
        ]
        messages = build_replay_messages(entries)
        assert messages[0]["role"] == "assistant"
        assert "[Compaction summary]" in messages[0]["content"][0]["text"]

    def test_empty_entries(self):
        assert build_replay_messages([]) == []

    def test_preserves_content_arrays(self):
        content = [
            {"type": "text", "text": "I'll search"},
            {"type": "function_call", "id": "fc-1", "name": "memory_search", "arguments": "{}"},
        ]
        entries = [_msg_entry("assistant", content, entry_id="msg-001")]
        messages = build_replay_messages(entries)
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][1]["type"] == "function_call"


# ---------------------------------------------------------------------------
# sanitize_history
# ---------------------------------------------------------------------------


class TestSanitizeHistory:
    def test_strips_base64_images_from_text(self):
        b64 = "data:image/png;base64," + "A" * 200
        messages = [
            {"role": "user", "content": "Look at this"},
            {"role": "assistant", "content": f"Here is the image: {b64} done"},
        ]
        result = sanitize_history(messages)
        assert "base64" not in str(result)
        assistant_msg = [m for m in result if m["role"] == "assistant"][0]
        assert "[image removed]" in assistant_msg["content"]

    def test_strips_image_blocks(self):
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "url": "data:image/png;base64,abc"},
            ],
        }]
        result = sanitize_history(messages)
        content = result[0]["content"]
        assert all(b["type"] == "text" for b in content)
        assert any("[screenshot from prior run]" in b.get("text", "") for b in content)

    def test_strips_thinking_blocks(self):
        messages = [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": "Before <THINKING>internal reasoning\nmultiline</THINKING> After"},
        ]
        result = sanitize_history(messages)
        assistant_msg = [m for m in result if m["role"] == "assistant"][0]
        text = assistant_msg["content"]
        assert "<THINKING>" not in text
        assert "Before" in text
        assert "After" in text

    def test_strips_thinking_from_content_array(self):
        messages = [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": [{"type": "text", "text": "<THINKING>thoughts</THINKING>Visible"}]},
        ]
        result = sanitize_history(messages)
        assistant_msg = [m for m in result if m["role"] == "assistant"][0]
        assert "<THINKING>" not in assistant_msg["content"][0]["text"]
        assert "Visible" in assistant_msg["content"][0]["text"]

    def test_does_not_strip_thinking_from_user(self):
        messages = [{
            "role": "user",
            "content": "User says <THINKING>not stripped</THINKING> here",
        }]
        result = sanitize_history(messages)
        assert "<THINKING>" in result[0]["content"]

    def test_strips_usage_keys(self):
        messages = [{
            "role": "assistant",
            "content": "Hello",
            "usage": {"input": 100},
            "stopReason": "end_turn",
            "api": "openai-responses",
        }]
        result = sanitize_history(messages)
        assert "usage" not in result[0]
        assert "stopReason" not in result[0]
        assert "api" not in result[0]

    def test_ensures_user_first(self):
        messages = [
            {"role": "assistant", "content": "I start"},
            {"role": "user", "content": "Hello"},
        ]
        result = sanitize_history(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[session history follows]"
        assert len(result) == 3

    def test_no_prepend_when_user_first(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = sanitize_history(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_repairs_orphaned_tool_results(self):
        """tool_result without matching function_call is removed."""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "orphan-id", "content": "result"},
                ],
            },
        ]
        result = sanitize_history(messages)
        # The orphaned tool_result message should be removed entirely
        assert len(result) == 1

    def test_repairs_orphaned_function_calls(self):
        """function_call without matching tool_result is removed."""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search"},
                    {"type": "function_call", "id": "orphan-call", "name": "search", "arguments": "{}"},
                ],
            },
        ]
        result = sanitize_history(messages)
        # function_call block removed, text kept
        assistant_msg = [m for m in result if m["role"] == "assistant"][0]
        assert len(assistant_msg["content"]) == 1
        assert assistant_msg["content"][0]["type"] == "text"

    def test_keeps_paired_tool_calls(self):
        """Matched function_call + tool_result are preserved."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "function_call", "id": "call-1", "name": "search", "arguments": "{}"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call-1", "content": "found it"},
                ],
            },
        ]
        result = sanitize_history(messages)
        # Both should be kept (plus prepended user-first message)
        non_bootstrap = [m for m in result if m.get("content") != "[session history follows]"]
        assert len(non_bootstrap) == 2

    def test_empty_messages(self):
        assert sanitize_history([]) == []

    def test_strips_base64_source_blocks(self):
        """Image blocks with base64 source are replaced."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "data": "abc123"}},
            ],
        }]
        result = sanitize_history(messages)
        assert result[0]["content"][0]["type"] == "text"
        assert "[screenshot from prior run]" in result[0]["content"][0]["text"]


# ---------------------------------------------------------------------------
# limit_history_turns
# ---------------------------------------------------------------------------


class TestLimitHistoryTurns:
    def test_keeps_last_n_user_turns(self):
        messages = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Reply 2"},
            {"role": "user", "content": "Turn 3"},
            {"role": "assistant", "content": "Reply 3"},
        ]
        result = limit_history_turns(messages, 2)
        assert len(result) == 4
        assert result[0]["content"] == "Turn 2"
        assert result[-1]["content"] == "Reply 3"

    def test_none_returns_all(self):
        messages = [{"role": "user", "content": f"Turn {i}"} for i in range(10)]
        result = limit_history_turns(messages, None)
        assert len(result) == 10

    def test_zero_returns_all(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = limit_history_turns(messages, 0)
        assert len(result) == 1

    def test_negative_returns_all(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = limit_history_turns(messages, -1)
        assert len(result) == 1

    def test_limit_greater_than_turns(self):
        messages = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Reply 1"},
        ]
        result = limit_history_turns(messages, 100)
        assert len(result) == 2

    def test_single_turn(self):
        messages = [
            {"role": "user", "content": "Only turn"},
            {"role": "assistant", "content": "Only reply"},
        ]
        result = limit_history_turns(messages, 1)
        assert len(result) == 2

    def test_empty_messages(self):
        assert limit_history_turns([], 5) == []

    def test_keeps_assistant_responses_with_user_turns(self):
        messages = [
            {"role": "user", "content": "T1"},
            {"role": "assistant", "content": "R1"},
            {"role": "assistant", "content": "R1b"},  # multi-assistant
            {"role": "user", "content": "T2"},
            {"role": "assistant", "content": "R2"},
        ]
        result = limit_history_turns(messages, 1)
        # Should keep from T2 onward
        assert len(result) == 2
        assert result[0]["content"] == "T2"


# ---------------------------------------------------------------------------
# Re-sanitize after truncation
# ---------------------------------------------------------------------------


class TestResanitizeAfterTruncation:
    def test_truncation_orphans_are_repaired(self):
        """Truncation can orphan tool results at the cut boundary; re-sanitize fixes them."""
        messages = [
            {"role": "user", "content": "T1"},
            {
                "role": "assistant",
                "content": [
                    {"type": "function_call", "id": "call-1", "name": "search", "arguments": "{}"},
                ],
            },
            # This result pairs with call-1, but after truncation call-1 is gone
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call-1", "content": "result"},
                ],
            },
            {"role": "user", "content": "T2"},
            {"role": "assistant", "content": "Reply 2"},
        ]
        # Limit to 1 user turn — keeps T2 + Reply 2
        truncated = limit_history_turns(messages, 1)
        # Re-sanitize to fix any orphans
        result = sanitize_history(truncated)
        # Should just have T2 + Reply 2
        non_bootstrap = [m for m in result if m.get("content") != "[session history follows]"]
        assert len(non_bootstrap) == 2


# ---------------------------------------------------------------------------
# Integration: full pipeline with SessionManager
# ---------------------------------------------------------------------------


class TestReplayIntegration:
    def test_full_replay_pipeline(self, tmp_path):
        """End-to-end: create transcript, replay, sanitize, limit."""
        sm = SessionManager("test-task", base_dir=tmp_path)
        sm.init_session(model="claude")

        # Simulate a multi-step conversation
        sm.append_message("user", "Navigate to floor 2")
        sm.append_message(
            "assistant",
            [
                {"type": "text", "text": "I'll click the door"},
                {"type": "computer_call", "id": "cc-1", "action": {"type": "click", "x": 100, "y": 200}},
            ],
            stop_reason="tool_use",
        )
        sm.append_message(
            "tool",
            [{"type": "tool_result", "tool_use_id": "cc-1", "content": "image:trajectory"}],
        )
        sm.append_message("assistant", "I can see floor 2 now")

        # Load and replay
        entries = sm.load_history()
        messages = build_replay_messages(entries)
        assert len(messages) == 4

        messages = sanitize_history(messages)
        # Should have user-first bootstrap + 4 messages
        assert messages[0]["role"] == "user"

        messages = limit_history_turns(messages, 2)
        # Re-sanitize
        messages = sanitize_history(messages)
        assert len(messages) > 0

    def test_multi_run_replay(self, tmp_path):
        """Replay across two runs produces combined history."""
        sm = SessionManager("test-task", base_dir=tmp_path)

        # Run 1
        sm.init_session(model="claude")
        sm.append_message("user", "Start task")
        sm.append_message("assistant", "Working on it")

        # Run 2
        sm.init_session(model="claude")
        sm.append_message("user", "Continue task")
        sm.append_message("assistant", "Continuing")

        entries = sm.load_history()
        messages = build_replay_messages(entries)
        # 4 messages (2 per run, session entries skipped)
        assert len(messages) == 4

    def test_replay_with_compaction(self, tmp_path):
        """Compaction entry collapses prior messages."""
        sm = SessionManager("test-task", base_dir=tmp_path)
        sm.init_session(model="claude")

        sm.append_message("user", "Old message")
        sm.append_message("assistant", "Old reply")
        e3 = sm.append_message("user", "Kept message")

        sm.append_compaction("Summary of old conversation", e3.id, 90000)

        sm.append_message("assistant", "Reply to kept")

        entries = sm.load_history()
        messages = build_replay_messages(entries)
        # Old messages replaced by compaction summary, kept message + reply preserved
        assert len(messages) == 3
        assert "[Compaction summary]" in messages[0]["content"][0]["text"]

    def test_empty_history_produces_empty_replay(self, tmp_path):
        """No prior entries means empty replay messages."""
        sm = SessionManager("test-task", base_dir=tmp_path)
        sm.init_session(model="claude")

        # Only session header, no messages
        entries = sm.load_history()
        messages = build_replay_messages(entries)
        assert messages == []


# ---------------------------------------------------------------------------
# convert_to_responses_api_items
# ---------------------------------------------------------------------------


class TestConvertToResponsesApiItems:
    """Test unnesting of Chat Completions messages into Responses API items."""

    def test_user_string_content(self):
        """User message with string content → message wrapper with input_text."""
        messages = [{"role": "user", "content": "Do the task"}]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 1
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "user"
        assert items[0]["content"] == [{"type": "input_text", "text": "Do the task"}]

    def test_user_content_array(self):
        """User message with text content array → message wrapper with input_text blocks."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "text", "text": "And this"},
            ],
        }]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 1
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "user"
        assert len(items[0]["content"]) == 2
        assert all(b["type"] == "input_text" for b in items[0]["content"])

    def test_assistant_string_content(self):
        """Assistant message with string → message wrapper with output_text."""
        messages = [{"role": "assistant", "content": "I see floor 2"}]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 1
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        assert items[0]["content"] == [{"type": "output_text", "text": "I see floor 2"}]

    def test_assistant_text_plus_function_call(self):
        """Assistant text + function_call → message item + function_call item."""
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search"},
                {"type": "function_call", "id": "fc-1", "name": "memory_search", "arguments": '{"q": "test"}'},
            ],
        }]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 2
        # First: text flushed as message
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        assert items[0]["content"] == [{"type": "output_text", "text": "Let me search"}]
        # Second: function_call item with id → call_id mapping
        assert items[1]["type"] == "function_call"
        assert items[1]["call_id"] == "fc-1"
        assert items[1]["name"] == "memory_search"
        assert items[1]["arguments"] == '{"q": "test"}'

    def test_openai_reasoning_block_replayed_as_reasoning_item(self):
        messages = [{
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Check the floor label first.",
                    "thinkingSignature": '{"id":"rs_123","type":"reasoning"}',
                },
                {"type": "text", "text": "Now act."},
            ],
        }]
        items = convert_to_responses_api_items(messages)

        assert items[0] == {
            "type": "reasoning",
            "id": "rs_123",
            "summary": [{"type": "summary_text", "text": "Check the floor label first."}],
        }
        assert items[1]["type"] == "message"
        assert items[1]["role"] == "assistant"

    def test_assistant_text_plus_computer_call(self):
        """Replay computer_call blocks are downgraded to assistant text."""
        action = {"type": "click", "x": 100, "y": 200}
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll click the door"},
                {"type": "computer_call", "id": "cc-1", "action": action},
            ],
        }]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 2
        assert items[0]["type"] == "message"
        assert items[0]["content"] == [{"type": "output_text", "text": "I'll click the door"}]
        assert items[1]["type"] == "message"
        assert items[1]["role"] == "assistant"
        assert "computer action" in items[1]["content"][0]["text"]
        assert "click" in items[1]["content"][0]["text"]

    def test_assistant_batched_computer_call_downgraded_to_text(self):
        """GPT-5.4 batched computer actions replay as descriptive text, not tool calls."""
        messages = [{
            "role": "assistant",
            "content": [
                {
                    "type": "computer_call",
                    "id": "cc-batch",
                    "actions": [
                        {"type": "click", "x": 10, "y": 20},
                        {"type": "keypress", "keys": ["ARROWUP"]},
                    ],
                },
            ],
        }]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 1
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        text = items[0]["content"][0]["text"]
        assert "computer action" in text
        assert "click" in text
        assert "keypress" in text

    def test_assistant_only_function_call(self):
        """Assistant with only function_call, no text → single function_call item."""
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "function_call", "id": "fc-2", "name": "search", "arguments": "{}"},
            ],
        }]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 1
        assert items[0]["type"] == "function_call"
        assert items[0]["call_id"] == "fc-2"

    def test_tool_message_with_tool_results(self):
        """Tool message with tool_result blocks → function_call_output items."""
        messages = [{
            "role": "tool",
            "content": [
                {"type": "tool_result", "tool_use_id": "fc-1", "content": "found it"},
                {"type": "tool_result", "tool_use_id": "fc-2", "content": "also found"},
            ],
        }]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 2
        assert items[0]["type"] == "function_call_output"
        assert items[0]["call_id"] == "fc-1"
        assert items[0]["output"] == "found it"
        assert items[1]["type"] == "function_call_output"
        assert items[1]["call_id"] == "fc-2"

    def test_user_message_with_tool_results(self):
        """User message containing tool_result blocks → function_call_output items.

        After sanitize_history, tool messages have role="user" (OpenClaw convention).
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "cc-1", "content": "image:trajectory"},
            ],
        }]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 1
        assert items[0]["type"] == "function_call_output"
        assert items[0]["call_id"] == "cc-1"

    def test_computer_tool_result_replays_as_user_text(self):
        """Historical computer outputs are downgraded to user text on replay."""
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "computer_call", "id": "cc-1", "actions": [{"type": "screenshot"}]}],
            },
            {
                "role": "tool",
                "content": [{"type": "tool_result", "tool_use_id": "cc-1", "content": "image:trajectory"}],
            },
        ]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 2
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        assert items[1]["type"] == "message"
        assert items[1]["role"] == "user"
        assert "computer result" in items[1]["content"][0]["text"]

    def test_empty_messages(self):
        """Empty input → empty output."""
        assert convert_to_responses_api_items([]) == []

    def test_id_to_call_id_mapping(self):
        """Transcript uses 'id' key, Responses API expects 'call_id'."""
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "function_call", "id": "call-abc", "name": "tool", "arguments": "{}"},
            ],
        }]
        items = convert_to_responses_api_items(messages)
        assert items[0]["call_id"] == "call-abc"
        assert "id" not in items[0]

    def test_tool_use_id_to_call_id_mapping(self):
        """Transcript tool_result uses 'tool_use_id', output needs 'call_id'."""
        messages = [{
            "role": "tool",
            "content": [
                {"type": "tool_result", "tool_use_id": "call-xyz", "content": "ok"},
            ],
        }]
        items = convert_to_responses_api_items(messages)
        assert items[0]["call_id"] == "call-xyz"
        assert "tool_use_id" not in items[0]

    def test_multiple_messages_ordering(self):
        """Multiple messages produce items in correct order."""
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll search"},
                    {"type": "function_call", "id": "fc-1", "name": "search", "arguments": "{}"},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_use_id": "fc-1", "content": "result"},
                ],
            },
            {"role": "assistant", "content": "Done"},
        ]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 5
        assert items[0]["type"] == "message"  # user
        assert items[0]["role"] == "user"
        assert items[1]["type"] == "message"  # assistant text
        assert items[1]["role"] == "assistant"
        assert items[2]["type"] == "function_call"
        assert items[3]["type"] == "function_call_output"
        assert items[4]["type"] == "message"  # assistant text
        assert items[4]["role"] == "assistant"

    def test_assistant_text_after_structured_block(self):
        """Text after a function_call creates a new message item."""
        messages = [{
            "role": "assistant",
            "content": [
                {"type": "function_call", "id": "fc-1", "name": "search", "arguments": "{}"},
                {"type": "text", "text": "Trailing text"},
            ],
        }]
        items = convert_to_responses_api_items(messages)
        assert len(items) == 2
        assert items[0]["type"] == "function_call"
        assert items[1]["type"] == "message"
        assert items[1]["content"] == [{"type": "output_text", "text": "Trailing text"}]

    def test_integration_full_pipeline(self, tmp_path):
        """End-to-end: build_replay → sanitize → limit → convert."""
        sm = SessionManager("test-task", base_dir=tmp_path)
        sm.init_session(model="claude")

        # Simulate a multi-step conversation
        sm.append_message("user", "Navigate to floor 2")
        sm.append_message(
            "assistant",
            [
                {"type": "text", "text": "I'll click the door"},
                {"type": "computer_call", "id": "cc-1", "action": {"type": "click", "x": 100, "y": 200}},
            ],
            stop_reason="tool_use",
        )
        sm.append_message(
            "tool",
            [{"type": "tool_result", "tool_use_id": "cc-1", "content": "image:trajectory"}],
        )
        sm.append_message("assistant", "I can see floor 2 now")

        # Run the full pipeline
        entries = sm.load_history()
        messages = build_replay_messages(entries)
        messages = sanitize_history(messages)
        messages = limit_history_turns(messages, 10)
        messages = sanitize_history(messages)
        items = convert_to_responses_api_items(messages)

        # Verify items are all in Responses API format
        for item in items:
            assert "type" in item, f"Item missing 'type': {item}"
            if item["type"] == "message":
                assert "role" in item
                assert isinstance(item.get("content"), list)
            elif item["type"] == "function_call":
                assert "call_id" in item
                assert "name" in item
            elif item["type"] == "computer_call":
                assert "call_id" in item
                assert "action" in item
            elif item["type"] == "function_call_output":
                assert "call_id" in item
            elif item["type"] == "computer_call_output":
                assert "call_id" in item

        # Should have: bootstrap user msg + user msg + assistant text + computer_call + tool output + assistant text
        assert len(items) >= 4

    def test_tool_adjacency_with_interleaved_flush(self):
        """Memory flush messages between tool call and result are reordered.

        Reproduces the real-world pattern where memory flush (user + assistant)
        is logged between a computer_call and its tool_result in the transcript.
        """
        messages = [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll click"},
                    {"type": "computer_call", "id": "cc-1", "action": {"type": "click", "x": 100, "y": 200}},
                ],
            },
            # Memory flush interleaved BEFORE the tool result
            {"role": "user", "content": "Pre-compaction memory flush."},
            {"role": "assistant", "content": "[memory flush — tool calls executed]"},
            # Tool result comes after flush
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_use_id": "cc-1", "content": "image:trajectory"},
                ],
            },
            {"role": "assistant", "content": "I can see the result"},
        ]
        items = convert_to_responses_api_items(messages)

        action_idx = next(
            i for i, it in enumerate(items)
            if it.get("type") == "message"
            and it.get("role") == "assistant"
            and isinstance(it.get("content"), list)
            and any("computer action" in str(b.get("text", "")) for b in it["content"])
        )
        assert all(it.get("type") != "computer_call" for it in items)
        result_idx = next(
            i for i, it in enumerate(items)
            if it.get("type") == "message"
            and it.get("role") == "user"
            and isinstance(it.get("content"), list)
            and any("computer result" in str(b.get("text", "")) for b in it["content"])
        )
        assert result_idx > action_idx

        # Flush messages should come AFTER the tool output
        flush_idx = next(
            i for i, it in enumerate(items)
            if it.get("type") == "message" and it.get("role") == "user"
            and isinstance(it.get("content"), list)
            and any("flush" in str(b.get("text", "")).lower() for b in it["content"])
        )
        assert flush_idx != result_idx
