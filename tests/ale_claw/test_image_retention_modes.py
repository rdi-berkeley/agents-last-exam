"""Tests for sticky placeholder + turn-mode.

Covers behavior added on top of test_image_retention.py:
  - Sticky placeholder replaces image-only messages (cache-thrash fix).
  - Placeholder is byte-stable across calls (idempotent).
  - mode="openclaw" counts completed turns and prunes images outside the window.
  - Mode parameter validation.

The existing test_image_retention.py file covers per-block stripping,
native+shim mixed paths, and tool-pairing invariants in count mode.
"""

import pytest
from ale_claw.harness.adapters.image_retention import (
    PRUNED_HISTORY_IMAGE_MARKER,
    OpenClawImageRetentionCallback,
)


def _user_text(text: str) -> dict:
    return {"type": "message", "role": "user", "content": text}


def _user_image_msg(url: str) -> dict:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": url}}],
    }


def _user_image_with_text(url: str, text: str) -> dict:
    return {
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_text", "text": text},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }


def _assistant(text: str = "ok") -> dict:
    return {"type": "message", "role": "assistant", "content": text}


def _function_call(call_id: str = "c1", name: str = "computer") -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}


def _function_call_output(call_id: str = "c1", text: str = "result") -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": text}


# ---------------------------------------------------------------------------
# Sticky placeholder (count mode default)
# ---------------------------------------------------------------------------

class TestStickyPlaceholder:
    def test_image_only_message_becomes_placeholder_not_dropped(self):
        cb = OpenClawImageRetentionCallback(only_n_most_recent_images=1, mode="cua")
        messages = [
            _user_image_msg("data:image/png;base64,A"),
            _user_image_msg("data:image/png;base64,B"),
        ]
        result = cb._apply_image_retention(messages)
        # Length preserved; old image replaced with placeholder text.
        assert len(result) == 2
        assert result[0]["content"] == [
            {"type": "text", "text": PRUNED_HISTORY_IMAGE_MARKER}
        ]
        assert result[1]["content"][0]["image_url"]["url"] == "data:image/png;base64,B"

    def test_message_with_text_strips_image_keeps_text_no_placeholder(self):
        """If the message had text content alongside the image, the text
        survives and no placeholder is added — the existing text already
        marks 'something was here'."""
        cb = OpenClawImageRetentionCallback(only_n_most_recent_images=1, mode="cua")
        messages = [
            _user_image_with_text("data:image/png;base64,A", "look at this"),
            _user_image_msg("data:image/png;base64,B"),
        ]
        result = cb._apply_image_retention(messages)
        assert len(result) == 2
        # Image stripped, text preserved.
        assert result[0]["content"] == [{"type": "input_text", "text": "look at this"}]

    def test_placeholder_is_idempotent_across_calls(self):
        """Apply retention twice — placeholder doesn't get re-pruned (it's
        text, not an image), so the second call is a no-op for the
        placeholder message."""
        cb = OpenClawImageRetentionCallback(only_n_most_recent_images=1, mode="cua")
        messages = [
            _user_image_msg("data:image/png;base64,A"),
            _user_image_msg("data:image/png;base64,B"),
        ]
        first = cb._apply_image_retention(messages)
        second = cb._apply_image_retention(first)
        assert second == first  # byte-stable

    def test_placeholder_byte_stable_across_independent_runs(self):
        """Two independently-constructed callbacks produce identical
        placeholder text for the same input — required for cache stability
        across re-instantiations."""
        cb1 = OpenClawImageRetentionCallback(only_n_most_recent_images=1, mode="cua")
        cb2 = OpenClawImageRetentionCallback(only_n_most_recent_images=1, mode="cua")
        messages = [
            _user_image_msg("data:image/png;base64,A"),
            _user_image_msg("data:image/png;base64,B"),
        ]
        assert cb1._apply_image_retention(messages) == cb2._apply_image_retention(messages)


# ---------------------------------------------------------------------------
# Turn mode
# ---------------------------------------------------------------------------

class TestTurnMode:
    def test_under_turn_budget_returns_unchanged(self):
        """When completed turns ≤ N, no pruning happens regardless of image count."""
        cb = OpenClawImageRetentionCallback(only_n_most_recent_images=3, mode="openclaw")
        # 1 completed turn (asst1) with 2 images in it.
        messages = [
            _user_text("task"),
            _assistant("thinking"),
            _function_call("c1"),
            _function_call_output("c1"),
            _user_image_msg("data:image/png;base64,A"),
            _user_image_msg("data:image/png;base64,B"),
        ]
        result = cb._apply_image_retention(messages)
        assert result == messages

    def test_over_turn_budget_prunes_images_in_old_turns(self):
        """5 turns, N=3 → images in turns 1 and 2 become placeholders;
        images in turns 3, 4, 5 survive."""
        cb = OpenClawImageRetentionCallback(only_n_most_recent_images=3, mode="openclaw")
        # Build 5 turns, each: assistant + tool_call + tool_output + screenshot.
        # Indices: user_text=0, then per-turn block of 4 starting at 1, 5, 9, 13, 17.
        # Turn assistants land at indices 1, 5, 9, 13, 17 — exactly the turn starts.
        # Images land at indices 4, 8, 12, 16, 20.
        messages = [_user_text("task")]
        for i in range(1, 6):
            messages.extend([
                _assistant(f"turn {i}"),
                _function_call(f"c{i}"),
                _function_call_output(f"c{i}", "ok"),
                _user_image_msg(f"data:image/png;base64,IMG_{i}"),
            ])
        result = cb._apply_image_retention(messages)

        # Length preserved (placeholder replacement, no deletion).
        assert len(result) == len(messages)

        # Window = last 3 turns = turns 3, 4, 5. Cutoff at turn 3's assistant
        # (index 9). Everything before index 9 has its images pruned.
        # Images at indices 4 (turn 1) and 8 (turn 2) → placeholders.
        # Images at indices 12, 16, 20 (turns 3, 4, 5) → preserved.
        placeholder = [{"type": "text", "text": PRUNED_HISTORY_IMAGE_MARKER}]
        assert result[4]["content"] == placeholder, "turn 1 image should be placeholder"
        assert result[8]["content"] == placeholder, "turn 2 image should be placeholder"
        assert result[12]["content"][0]["image_url"]["url"] == "data:image/png;base64,IMG_3"
        assert result[16]["content"][0]["image_url"]["url"] == "data:image/png;base64,IMG_4"
        assert result[20]["content"][0]["image_url"]["url"] == "data:image/png;base64,IMG_5"

    def test_turn_starts_skip_consecutive_assistants(self):
        """Multiple assistant-emitting messages within one turn count as
        one turn boundary, not many. E.g., reasoning + assistant + function_call
        all in one response = one turn."""
        cb = OpenClawImageRetentionCallback(only_n_most_recent_images=3, mode="openclaw")
        messages = [
            _user_text("task"),
            # Single turn with 3 assistant-emitting messages
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "x"}]},
            _assistant("response"),
            _function_call("c1"),
            _function_call_output("c1"),
            _user_image_msg("data:image/png;base64,A"),
        ]
        # Only 1 completed turn; with N=3, no pruning.
        result = cb._apply_image_retention(messages)
        assert result == messages

    def test_native_computer_call_output_in_old_turn_removed(self):
        """Native path images in pruned turns still get the triple-removal
        treatment (no placeholder for native — preserves API contract)."""
        cb = OpenClawImageRetentionCallback(only_n_most_recent_images=1, mode="openclaw")
        messages = [
            _user_text("task"),
            # Turn 1 with native screenshot
            _assistant("a1"),
            {"type": "computer_call", "call_id": "n1", "action": {"type": "screenshot"}},
            {
                "type": "computer_call_output",
                "call_id": "n1",
                "output": {"type": "input_image", "image_url": "data:image/png;base64,N1"},
            },
            # Turn 2 (kept)
            _assistant("a2"),
            _function_call("f2"),
            _function_call_output("f2"),
            _user_image_msg("data:image/png;base64,SHIM2"),
        ]
        result = cb._apply_image_retention(messages)
        # Native triple from turn 1 removed (computer_call + computer_call_output).
        types = [(m.get("type"), m.get("call_id")) for m in result]
        assert ("computer_call", "n1") not in types
        assert ("computer_call_output", "n1") not in types
        # Turn 2 image survives.
        assert any(
            isinstance(m.get("content"), list)
            and m["content"]
            and m["content"][0].get("type") == "image_url"
            and m["content"][0]["image_url"]["url"] == "data:image/png;base64,SHIM2"
            for m in result
        )


# ---------------------------------------------------------------------------
# Mode validation + None disables
# ---------------------------------------------------------------------------

class TestModeWiring:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            OpenClawImageRetentionCallback(only_n_most_recent_images=3, mode="bogus")  # type: ignore[arg-type]

    def test_none_budget_disables_both_modes(self):
        for mode in ("cua", "openclaw"):
            cb = OpenClawImageRetentionCallback(only_n_most_recent_images=None, mode=mode)
            messages = [
                _user_image_msg("data:image/png;base64,A"),
                _user_image_msg("data:image/png;base64,B"),
                _user_image_msg("data:image/png;base64,C"),
            ]
            assert cb._apply_image_retention(messages) == messages

    def test_default_mode_is_openclaw(self):
        # Harness default flipped from count-by-image to OpenClaw-parity
        # turn-based retention (renamed "count"->"cua", "turn"->"openclaw";
        # default flipped to "openclaw" after verification).
        cb = OpenClawImageRetentionCallback(only_n_most_recent_images=2)
        assert cb.mode == "openclaw"


# ---------------------------------------------------------------------------
# Cache-stability property (integration-style)
# ---------------------------------------------------------------------------

class TestCacheStabilityProperty:
    """Verify the structural property that fixes the cache thrash:
    when an image ages out, only ONE message changes (the placeholder
    replacement). Every other message keeps its index AND content.
    """

    def test_aging_out_one_image_mutates_one_index_only(self):
        cb = OpenClawImageRetentionCallback(only_n_most_recent_images=2, mode="cua")
        # Turn N state: 3 images; cb keeps 2.
        turn_n = [
            _user_text("task"),
            _user_image_msg("data:image/png;base64,A"),  # index 1 — about to age
            _assistant("a1"),
            _user_image_msg("data:image/png;base64,B"),  # index 3
            _assistant("a2"),
            _user_image_msg("data:image/png;base64,C"),  # index 5
        ]
        result_n = cb._apply_image_retention(turn_n)

        # Turn N+1 adds a 4th image; A ages out.
        turn_n1 = list(turn_n) + [
            _assistant("a3"),
            _user_image_msg("data:image/png;base64,D"),
        ]
        result_n1 = cb._apply_image_retention(turn_n1)

        # Property: result_n1[i] == result_n[i] for every i in range(len(result_n))
        # EXCEPT exactly one index where the placeholder replaced an image.
        diffs = [
            i
            for i in range(len(result_n))
            if result_n[i] != result_n1[i]
        ]
        assert len(diffs) == 1, f"Expected exactly 1 mutation, got {len(diffs)}: {diffs}"
        # The mutated message is the placeholder for image A.
        assert result_n1[diffs[0]]["content"] == [
            {"type": "text", "text": PRUNED_HISTORY_IMAGE_MARKER}
        ]

    def test_after_two_age_outs_two_placeholders_persist_in_order(self):
        """Aging out two images sequentially should leave two placeholders
        at the same positions where images A and B used to be."""
        cb = OpenClawImageRetentionCallback(only_n_most_recent_images=2, mode="cua")
        # Build a 4-image conversation, then age out twice.
        messages = [
            _user_text("task"),
            _user_image_msg("data:image/png;base64,A"),
            _assistant("a1"),
            _user_image_msg("data:image/png;base64,B"),
            _assistant("a2"),
            _user_image_msg("data:image/png;base64,C"),
            _assistant("a3"),
            _user_image_msg("data:image/png;base64,D"),
        ]
        result = cb._apply_image_retention(messages)
        # A and B aged out → both placeholders.
        assert result[1]["content"] == [
            {"type": "text", "text": PRUNED_HISTORY_IMAGE_MARKER}
        ]
        assert result[3]["content"] == [
            {"type": "text", "text": PRUNED_HISTORY_IMAGE_MARKER}
        ]
        # C and D survive.
        assert result[5]["content"][0]["image_url"]["url"] == "data:image/png;base64,C"
        assert result[7]["content"][0]["image_url"]["url"] == "data:image/png;base64,D"
