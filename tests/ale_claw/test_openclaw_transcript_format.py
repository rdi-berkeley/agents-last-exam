"""Tests for transcript JSONL format correctness.

Exercises the group_step_output() and _find_latest_screenshot() helpers from
openclaw/transcript.py without requiring a VM. All tests are pure unit tests.

additions: session reasoning retention tests — verifies thinking
blocks + thinkingSignature are retained in session logs and sanitized
differently per provider on replay.
"""

import time

from ale_run.agents.ale_claw.harness.context.transcript import _find_latest_screenshot, group_step_output
from ale_run.agents.ale_claw.harness.session import SessionManager


# ---------------------------------------------------------------------------
# group_step_output — assistant turn grouping
# ---------------------------------------------------------------------------


class TestAssistantTurnGrouping:
    def test_text_and_function_call_produce_one_assistant_entry(self):
        """text + function_call → 1 assistant content list with 2 blocks."""
        output_items = [
            {
                "type": "message",
                "content": [{"text": "Let me search memory."}],
            },
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "memory_search",
                "arguments": '{"query": "floor 2"}',
            },
        ]
        assistant_content, tool_results = group_step_output(output_items)
        assert len(assistant_content) == 2
        assert tool_results == []
        assert assistant_content[0] == {"type": "text", "text": "Let me search memory."}
        assert assistant_content[1]["type"] == "function_call"
        assert assistant_content[1]["name"] == "memory_search"

    def test_computer_call_grouped_with_text(self):
        """computer_call block is grouped into the assistant content list."""
        output_items = [
            {
                "type": "message",
                "content": [{"text": "I will click the button."}],
            },
            {
                "type": "computer_call",
                "call_id": "cc_1",
                "action": {"type": "click", "x": 100, "y": 200},
            },
        ]
        assistant_content, tool_results = group_step_output(output_items)
        assert len(assistant_content) == 2
        assert assistant_content[0]["type"] == "text"
        assert assistant_content[1]["type"] == "computer_call"
        assert assistant_content[1]["action"]["type"] == "click"

    def test_reasoning_item_preserved_as_thinking_block(self):
        """Top-level reasoning items should be persisted into assistant content."""
        output_items = [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Check the floor label first."}],
                "thinkingSignature": '{"id":"sig_1","type":"reasoning"}',
            }
        ]
        assistant_content, tool_results = group_step_output(output_items)
        assert tool_results == []
        assert assistant_content == [
            {
                "type": "thinking",
                "thinking": "Check the floor label first.",
                "id": "rs_1",
                "thinkingSignature": '{"id":"sig_1","type":"reasoning"}',
            }
        ]

    def test_reasoning_item_without_summary_keeps_signature(self):
        output_items = [
            {
                "type": "reasoning",
                "id": "rs_2",
                "summary": [],
            }
        ]
        assistant_content, tool_results = group_step_output(output_items)
        assert tool_results == []
        assert assistant_content == [
            {
                "type": "thinking",
                "thinking": "",
                "id": "rs_2",
                "thinkingSignature": '{"id":"rs_2","type":"reasoning"}',
            }
        ]


class TestToolResultGrouping:
    def test_two_function_call_outputs_batch_into_one_tool_entry(self):
        """2 function_call_outputs → tool_results list with 2 blocks."""
        output_items = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result A",
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "result B",
            },
        ]
        assistant_content, tool_results = group_step_output(output_items)
        assert assistant_content == []
        assert len(tool_results) == 2
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "call_1"
        assert tool_results[0]["content"] == "result A"
        assert tool_results[1]["tool_use_id"] == "call_2"

    def test_computer_call_output_role_is_tool(self):
        """computer_call_output → appears in tool_results (role will be "tool")."""
        output_items = [
            {
                "type": "computer_call_output",
                "call_id": "cc_out_1",
                "output": {"type": "input_image", "image_url": "..."},
            },
        ]
        assistant_content, tool_results = group_step_output(output_items)
        assert assistant_content == []
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "cc_out_1"

    def test_user_string_message_from_tool_output_is_ignored_by_grouping(self):
        """Post-tool helper user messages should not be misclassified as assistant text."""
        output_items = [
            {
                "type": "message",
                "role": "user",
                "content": "[Screenshot saved to: /tmp/example.png]",
            },
            {
                "type": "computer_call_output",
                "call_id": "cc_out_1",
                "output": {"type": "input_image", "image_url": "..."},
            },
        ]
        assistant_content, tool_results = group_step_output(output_items)
        assert assistant_content == []
        assert len(tool_results) == 1


# ---------------------------------------------------------------------------
# Usage total field
# ---------------------------------------------------------------------------


class TestUsageTotal:
    def test_usage_total_equals_input_plus_output(self, tmp_path):
        """usage["total"] == usage["input"] + usage["output"] in logged entries."""
        sm = SessionManager("t1", base_dir=tmp_path)
        sm.init_session()
        entry = sm.append_message(
            "assistant",
            "hello",
            usage={"input": 300, "output": 75, "total": 375, "cost": 0.002},
        )
        msg = entry.data["message"]
        assert msg["usage"]["total"] == msg["usage"]["input"] + msg["usage"]["output"]


# ---------------------------------------------------------------------------
# Role correctness — no "toolResult", always "tool"
# ---------------------------------------------------------------------------


class TestRoleIsToolNotToolResult:
    def test_toolresult_never_appears_in_serialized_entries(self, tmp_path):
        """role:"toolResult" must never appear; tool results use role:"tool"."""
        sm = SessionManager("t1", base_dir=tmp_path)
        sm.init_session()
        sm.append_message("tool", [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}])

        transcript = sm.transcript_path.read_text(encoding="utf-8")
        assert '"toolResult"' not in transcript
        assert '"tool"' in transcript


# ---------------------------------------------------------------------------
# Content block type — function_call not toolCall
# ---------------------------------------------------------------------------


class TestFunctionCallType:
    def test_content_block_type_is_function_call_not_toolcall(self):
        """group_step_output produces type:"function_call", not "toolCall"."""
        output_items = [
            {
                "type": "function_call",
                "call_id": "call_xyz",
                "name": "memory_get",
                "arguments": "{}",
            },
        ]
        assistant_content, _ = group_step_output(output_items)
        assert len(assistant_content) == 1
        assert assistant_content[0]["type"] == "function_call"
        # Verify the old wrong string is not present
        for block in assistant_content:
            assert block.get("type") != "toolCall"


# ---------------------------------------------------------------------------
# No consecutive assistant entries
# ---------------------------------------------------------------------------


class TestNoConsecutiveAssistantEntries:
    def test_no_two_adjacent_entries_are_both_assistant(self, tmp_path):
        """A step with text + tool_call must not produce 2 consecutive assistant entries."""
        sm = SessionManager("t1", base_dir=tmp_path)
        sm.init_session()

        # Simulate what the agent does after group_step_output
        assistant_content = [
            {"type": "text", "text": "Thinking..."},
            {"type": "function_call", "id": "c1", "name": "memory_search", "arguments": "{}"},
        ]
        tool_results = [
            {"type": "tool_result", "tool_use_id": "c1", "content": "found"},
        ]
        sm.append_message("assistant", assistant_content, usage={"input": 10, "output": 5, "total": 15, "cost": 0})
        sm.append_message("tool", tool_results)

        entries = sm.load_history()
        messages = [e for e in entries if e.type == "message"]
        for i in range(len(messages) - 1):
            r1 = messages[i].data["message"]["role"]
            r2 = messages[i + 1].data["message"]["role"]
            assert not (r1 == "assistant" and r2 == "assistant"), (
                f"Consecutive assistant entries at positions {i} and {i+1}"
            )


class TestPartialItemTranscriptLogging:
    def test_partial_tool_outputs_are_persisted(self, tmp_path):
        sm = SessionManager("t1", base_dir=tmp_path)
        sm.init_session()

        from ale_run.agents.ale_claw.harness.agent_loop import OpenClawComputerAgent

        agent = object.__new__(OpenClawComputerAgent)
        agent.session_mgr = sm
        agent.trajectory_dir = None

        partial_items = [
            {
                "type": "computer_call_output",
                "call_id": "cc_1",
                "output": {"type": "input_image", "image_url": "..."},
            },
            {
                "type": "message",
                "role": "user",
                "content": "[Screenshot saved to: /tmp/example.png]",
            },
        ]

        agent._log_partial_items_to_transcript(partial_items)

        messages = [e.data["message"] for e in sm.load_history() if e.type == "message"]
        assert any(m["role"] == "tool" for m in messages)
        assert any(m["role"] == "user" for m in messages)
        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert tool_messages[0]["content"][0]["type"] == "tool_result"
        assert tool_messages[0]["content"][0]["tool_use_id"] == "cc_1"


# ---------------------------------------------------------------------------
# _find_latest_screenshot
# ---------------------------------------------------------------------------


class TestFindLatestScreenshot:
    def test_returns_path_to_newest_screenshot(self, tmp_path):
        """_find_latest_screenshot finds the most recently modified PNG."""
        traj_dir = tmp_path / "trajectories" / "run1"
        traj_dir.mkdir(parents=True)
        older = traj_dir / "turn_001_screenshot_after.png"
        newer = traj_dir / "turn_002_screenshot_after.png"
        older.write_bytes(b"fake")
        time.sleep(0.01)  # ensure mtime differs
        newer.write_bytes(b"fake")

        result = _find_latest_screenshot(traj_dir)
        assert result == str(newer)

    def test_returns_fallback_when_dir_missing(self, tmp_path):
        """Returns "image:trajectory" when trajectory_dir does not exist."""
        missing = tmp_path / "nonexistent"
        assert _find_latest_screenshot(missing) == "image:trajectory"

    def test_returns_fallback_when_none(self):
        """Returns "image:trajectory" when trajectory_dir is None."""
        assert _find_latest_screenshot(None) == "image:trajectory"

    def test_returns_fallback_when_no_pngs(self, tmp_path):
        """Returns "image:trajectory" when directory exists but has no PNGs."""
        traj_dir = tmp_path / "empty_traj"
        traj_dir.mkdir()
        (traj_dir / "some_other_file.txt").write_text("data")
        assert _find_latest_screenshot(traj_dir) == "image:trajectory"

    def test_computer_call_output_uses_screenshot_path(self, tmp_path):
        """computer_call_output with input_image resolves to actual PNG path."""
        traj_dir = tmp_path / "trajectories"
        traj_dir.mkdir()
        png = traj_dir / "turn_001_screenshot_after.png"
        png.write_bytes(b"fake png")

        output_items = [
            {
                "type": "computer_call_output",
                "call_id": "cc_1",
                "output": {"type": "input_image", "image_url": "..."},
            },
        ]
        _, tool_results = group_step_output(output_items, trajectory_dir=traj_dir)
        assert len(tool_results) == 1
        assert tool_results[0]["content"] == str(png)
        assert tool_results[0]["content"].endswith(".png")


# ---------------------------------------------------------------------------
# Session reasoning retention (AC7)
# ---------------------------------------------------------------------------


class TestSessionReasoningRetention:
    """Verify thinking blocks + thinkingSignature are retained in session logs.

    AC7: Session persistence behavior for reasoning is explicit.
    Policy: thinking blocks are retained at write time; sanitization
    (drop/preserve/downgrade) is applied at replay time via TranscriptPolicy.
    """

    def test_thinking_block_retained_in_session_log(self, tmp_path):
        """Thinking block with thinkingSignature persists through save/load."""
        sm = SessionManager("t1", base_dir=tmp_path)
        sm.init_session()

        content = [
            {"type": "text", "text": "I will click the button."},
            {
                "type": "thinking",
                "thinking": "The user wants me to navigate to floor 3.",
                "thinkingSignature": '{"id":"rs_abc","type":"reasoning"}',
            },
        ]
        sm.append_message("assistant", content)

        # Reload from disk
        history = sm.load_history()
        msg_entries = [e for e in history if e.type == "message"]
        assert len(msg_entries) == 1

        persisted = msg_entries[0].data["message"]["content"]
        assert len(persisted) == 2
        assert persisted[1]["type"] == "thinking"
        assert persisted[1]["thinking"] == "The user wants me to navigate to floor 3."
        assert persisted[1]["thinkingSignature"] == '{"id":"rs_abc","type":"reasoning"}'

    def test_sanitize_on_replay_differs_by_provider(self):
        """Same canonical transcript sanitizes differently for Anthropic vs OpenAI.

        Anthropic: thinking blocks are dropped (drop_thinking_blocks=True).
        OpenAI: orphaned reasoning blocks are downgraded; non-orphaned kept.
        """
        from ale_run.agents.ale_claw.harness.canonical.canonical import (
            CanonicalMessage,
            TextBlock,
            ThinkingBlock,
            sanitize_items,
        )

        msgs = [
            CanonicalMessage(role="user", content=[TextBlock(type="text", text="Hello")]),
            CanonicalMessage(role="assistant", content=[
                ThinkingBlock(
                    type="thinking",
                    thinking="Let me think about this.",
                    thinkingSignature='{"id":"rs_1","type":"reasoning"}',
                ),
                TextBlock(type="text", text="Here is my answer."),
            ]),
        ]

        # Anthropic: thinking blocks dropped
        anthropic_out = sanitize_items(msgs, model="anthropic/claude-sonnet-4-20250514")
        for msg in anthropic_out:
            for block in msg.get("content", []):
                assert block.get("type") != "thinking", (
                    "Anthropic replay should drop thinking blocks"
                )

        # OpenAI: thinking with following content is kept (not orphaned)
        openai_out = sanitize_items(msgs, model="openai/gpt-5.4")
        # The output is Responses API items — reasoning should be emitted
        reasoning_items = [i for i in openai_out if i.get("type") == "reasoning"]
        assert len(reasoning_items) == 1, (
            "OpenAI replay should preserve non-orphaned reasoning"
        )

    def test_orphaned_openai_reasoning_dropped_on_replay(self):
        """Orphaned OpenAI reasoning (no following content) is dropped on replay."""
        from ale_run.agents.ale_claw.harness.canonical.canonical import (
            CanonicalMessage,
            TextBlock,
            ThinkingBlock,
            sanitize_items,
        )

        msgs = [
            CanonicalMessage(role="user", content=[TextBlock(type="text", text="Hello")]),
            CanonicalMessage(role="assistant", content=[
                ThinkingBlock(
                    type="thinking",
                    thinking="",
                    thinkingSignature='{"id":"rs_orphaned","type":"reasoning"}',
                ),
                # No following non-thinking content — this is orphaned
            ]),
        ]

        openai_out = sanitize_items(msgs, model="openai/gpt-5.4")
        reasoning_items = [i for i in openai_out if i.get("type") == "reasoning"]
        assert len(reasoning_items) == 0, (
            "Orphaned OpenAI reasoning should be dropped on replay"
        )
