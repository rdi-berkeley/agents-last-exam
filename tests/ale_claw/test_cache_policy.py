"""Tests for apply_openclaw_cache_markers — sliding cache_control breakpoint policy.

Reproduces OpenClaw's anthropic-payload-policy.ts expectations:
  - Cache markers placed on system prompt (first message) + last message.
  - CUA's broken first-4 markers stripped before re-applying.
  - Boundary marker splits system prompt into stable/dynamic blocks.
  - Non-Anthropic models get all markers stripped, none added.
  - openrouter/anthropic/* honored; openrouter/openai/* skipped.

History: originally landed as ``CachePolicyCallback``. The
orchestration team's commit ``af38574b`` discovered that
``agent.py:_on_api_start`` deep-copies kwargs via ``get_json`` before
invoking callbacks, so any cache_control mutation in ``on_api_start``
never reached ``litellm.acompletion``. The fix replaced the callback
with an inline ``apply_openclaw_cache_markers(messages, model)`` call
inside ``UnifiedAgentConfig.predict_step``. These tests were migrated
to exercise the function directly — same semantics, sync API.
"""

import pytest

from ale_run.agents.ale_claw.harness.model.cache_policy import (
    OPENCLAW_CACHE_BOUNDARY,
    apply_openclaw_cache_markers,
    supports_anthropic_cache,
)


def _apply(messages, model):
    """Helper — call markers in place and return the (mutated) messages list."""
    apply_openclaw_cache_markers(messages, model)
    return messages


class TestSupportsAnthropicCache:
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-opus-4-7",
            "openrouter/anthropic/claude-sonnet-4-6",
            "vertex_ai/claude-sonnet-4@20250514",
            "vertex_ai/anthropic-claude",
            "bedrock/anthropic.claude-3-5-sonnet",
            "claude-opus-4-7",
            # Case-insensitive
            "Anthropic/Claude-Opus-4-7",
        ],
    )
    def test_anthropic_family_models_supported(self, model):
        assert supports_anthropic_cache(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "openrouter/openai/gpt-5.4",
            "openrouter/google/gemini-2.5-pro",
            "openai/gpt-4o",
            "gpt-4o",
            "gemini/gemini-2.0-flash",
            "",
            None,
        ],
    )
    def test_non_anthropic_models_skipped(self, model):
        assert supports_anthropic_cache(model) is False


class TestSlidingBreakpoint:
    def test_marks_system_prompt_and_last_message(self):
        messages = [
            {"role": "user", "content": "Identity: agent.\nTools: foo."},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "screenshot here"},
        ]
        _apply(messages, "anthropic/claude-opus-4-7")
        # System prompt cached (no boundary present → message-level marker).
        assert messages[0]["cache_control"] == {"type": "ephemeral"}
        # Middle assistant turn NOT marked.
        assert "cache_control" not in messages[1]
        # Trailing turn marked (sliding breakpoint).
        assert messages[2]["cache_control"] == {"type": "ephemeral"}

    def test_strips_cua_broken_first_4_markers(self):
        # Simulate CUA's _add_cache_control already having marked first 4.
        messages = [
            {"role": "user", "content": "sys", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "task", "cache_control": {"type": "ephemeral"}},
            {"role": "assistant", "content": "ok", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "tool result", "cache_control": {"type": "ephemeral"}},
            {"role": "assistant", "content": "again"},
            {"role": "user", "content": "latest"},
        ]
        _apply(messages, "anthropic/claude-opus-4-7")
        markers = [m.get("cache_control") for m in messages]
        # Only first and last should have marker.
        assert markers[0] == {"type": "ephemeral"}
        assert markers[1] is None
        assert markers[2] is None
        assert markers[3] is None
        assert markers[4] is None
        assert markers[5] == {"type": "ephemeral"}

    def test_single_message_only_marks_once(self):
        messages = [{"role": "user", "content": "only message"}]
        _apply(messages, "anthropic/claude-opus-4-7")
        assert messages[0]["cache_control"] == {"type": "ephemeral"}

    def test_empty_messages_noop(self):
        messages = []
        _apply(messages, "anthropic/claude-opus-4-7")
        assert messages == []

    def test_sliding_breakpoint_advances_each_turn(self):
        """Simulate two consecutive turns; the trailing marker should move forward."""
        turn1 = [
            {"role": "user", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        _apply(turn1, "anthropic/claude-opus-4-7")
        assert turn1[1]["cache_control"] == {"type": "ephemeral"}

        turn2 = [
            {"role": "user", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "tool result"},
        ]
        _apply(turn2, "anthropic/claude-opus-4-7")
        # Old trailing marker (msg[1]) is gone; new trailing (msg[3]) is set.
        assert "cache_control" not in turn2[1]
        assert turn2[3]["cache_control"] == {"type": "ephemeral"}


class TestBoundarySplit:
    def test_string_content_with_boundary_splits_into_blocks(self):
        sys_text = (
            "## Identity\nagent.\n## Tools\nfoo, bar.\n"
            f"{OPENCLAW_CACHE_BOUNDARY}\n## Time\n2026-05-03 14:22 UTC\n"
        )
        messages = [{"role": "user", "content": sys_text}]
        _apply(messages, "anthropic/claude-opus-4-7")
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        # Stable block has cache_control.
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert "Identity" in content[0]["text"]
        assert "Tools" in content[0]["text"]
        assert OPENCLAW_CACHE_BOUNDARY not in content[0]["text"]
        # Dynamic block is uncached.
        assert "cache_control" not in content[1]
        assert "Time" in content[1]["text"]
        assert OPENCLAW_CACHE_BOUNDARY not in content[1]["text"]

    def test_list_content_with_boundary_splits_block(self):
        # CUA's _combine_completion_messages normalizes content to list.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Identity\nTools\n"
                            f"{OPENCLAW_CACHE_BOUNDARY}\n"
                            "Time\n"
                        ),
                    }
                ],
            }
        ]
        _apply(messages, "anthropic/claude-opus-4-7")
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["cache_control"] == {"type": "ephemeral"}
        assert "Identity" in content[0]["text"]
        assert "Time" not in content[0]["text"]
        assert "cache_control" not in content[1]
        assert "Time" in content[1]["text"]

    def test_no_boundary_falls_back_to_message_level(self):
        messages = [{"role": "user", "content": "no boundary here"}]
        _apply(messages, "anthropic/claude-opus-4-7")
        # Content unchanged (still a string), marker at message level.
        assert messages[0]["content"] == "no boundary here"
        assert messages[0]["cache_control"] == {"type": "ephemeral"}


class TestNonAnthropicGating:
    def test_non_anthropic_strips_all_markers(self):
        messages = [
            {"role": "user", "content": "sys", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "task", "cache_control": {"type": "ephemeral"}},
        ]
        _apply(messages, "openrouter/openai/gpt-5.4")
        for m in messages:
            assert "cache_control" not in m

    def test_non_anthropic_strips_block_level_markers(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "stable",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "dynamic"},
                ],
            }
        ]
        _apply(messages, "openrouter/openai/gpt-5.4")
        for block in messages[0]["content"]:
            assert "cache_control" not in block

    def test_openrouter_anthropic_honored(self):
        messages = [
            {"role": "user", "content": "sys"},
            {"role": "user", "content": "task"},
        ]
        _apply(messages, "openrouter/anthropic/claude-sonnet-4-6")
        assert messages[0]["cache_control"] == {"type": "ephemeral"}
        assert messages[1]["cache_control"] == {"type": "ephemeral"}


# Note: PromptBuilder does NOT auto-emit the boundary marker. The constant
# is exported from cache_policy for callers that want to insert it manually
# (e.g., to gate a per-turn dynamic prompt section). apply_openclaw_cache_markers
# falls back to message-level cache_control on the system prompt when no
# marker is present — see TestBoundarySplit::test_no_boundary_falls_back_*.
