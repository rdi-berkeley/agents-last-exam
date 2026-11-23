"""Tests for Post-compaction message ordering.

Ensures _build_compacted_items never produces a list ending with role=assistant,
which would cause API errors for models that don't support assistant prefill
(e.g., Opus 4.6).

Updated for: _build_compacted_items now returns list[CanonicalMessage]
with typed content blocks instead of untyped dicts.
"""

from ale_claw.harness.canonical.canonical import (
    CanonicalMessage,
    CompactionSummaryBlock,
    TextBlock,
    normalize_to_canonical,
)


def _build_compacted_items_standalone(
    summary: str, kept_messages: list[dict]
) -> list[CanonicalMessage]:
    """Standalone version of OpenClawComputerAgent._build_compacted_items for testing.

    Avoids instantiating the full agent class (requires async setup, VM connection, etc.).
    Must be kept in sync with the real implementation.
    """
    items: list[CanonicalMessage] = []

    if summary:
        items.append(CanonicalMessage(
            role="user",
            content=[CompactionSummaryBlock(type="compaction_summary", text=summary)],
        ))

    if kept_messages:
        items.extend(normalize_to_canonical(kept_messages))

    # Guard: ensure items don't end with role=assistant
    if items and items[-1].get("role") == "assistant":
        items.append(CanonicalMessage(
            role="user",
            content=[TextBlock(type="text", text="[Continue from where you left off.]")],
        ))

    return items


class TestPostCompactionMessageOrdering:
    """Post-compaction message list must not end with role=assistant."""

    def test_trailing_assistant_gets_user_appended(self):
        """When kept_messages ends with assistant, a user message is appended."""
        kept = [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": "I did it"},
        ]
        result = _build_compacted_items_standalone("Summary of prior work", kept)

        assert result[-1]["role"] == "user"
        assert result[-1]["content"][0]["type"] == "text"
        assert result[-1]["content"][0]["text"] == "[Continue from where you left off.]"
        assert len(result) == 4  # summary + 2 kept + 1 appended

    def test_trailing_user_no_change(self):
        """When kept_messages ends with user, no extra message is added."""
        kept = [
            {"role": "assistant", "content": "I did something"},
            {"role": "user", "content": "Thanks, now do more"},
        ]
        result = _build_compacted_items_standalone("Summary", kept)

        assert result[-1]["role"] == "user"
        assert result[-1]["content"][0]["text"] == "Thanks, now do more"
        assert len(result) == 3  # summary + 2 kept

    def test_trailing_tool_no_change(self):
        """When kept_messages ends with tool result, no extra message is added."""
        kept = [
            {"role": "assistant", "content": "Calling tool"},
            {"role": "tool", "content": "Tool result here"},
        ]
        result = _build_compacted_items_standalone("Summary", kept)

        assert result[-1]["role"] == "tool"
        assert len(result) == 3  # summary + 2 kept

    def test_empty_kept_messages(self):
        """Summary-only (no kept messages) ends with user (the summary)."""
        result = _build_compacted_items_standalone("Summary of everything", [])

        assert result[-1]["role"] == "user"
        assert result[-1]["content"][0]["type"] == "compaction_summary"
        assert len(result) == 1

    def test_only_assistant_in_kept(self):
        """Single assistant message in kept gets a user tail appended."""
        kept = [{"role": "assistant", "content": "Last thing I said"}]
        result = _build_compacted_items_standalone("Summary", kept)

        assert result[-1]["role"] == "user"
        assert len(result) == 3  # summary + 1 kept + 1 appended

    def test_output_is_canonical_messages(self):
        """All output items are CanonicalMessage with typed content blocks."""
        kept = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "I'll help"},
                {"type": "function_call", "id": "call-1", "name": "test", "arguments": "{}"},
            ]},
        ]
        result = _build_compacted_items_standalone("Summary", kept)

        for msg in result:
            assert "role" in msg
            assert "content" in msg
            assert isinstance(msg["content"], list)
            for block in msg["content"]:
                assert "type" in block
