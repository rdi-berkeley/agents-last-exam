"""Assistant-text logging in ``OpenClawComputerAgent._log_assistant_text``.

Bare assistant text messages (including the DONE termination signal) used
to have no console hook — the stdout log only surfaced tool calls and
infra lines. This test locks the new ``[Agent]`` log behavior so the
operator can see text-only turns in real time (e.g. when the model emits
``DONE`` to end a run).
"""

from ale_claw.harness.agent_loop import OpenClawComputerAgent


_log = OpenClawComputerAgent._log_assistant_text


def _lines(capsys) -> list[str]:
    captured = capsys.readouterr()
    return [ln for ln in captured.out.splitlines() if ln.startswith("[Agent]")]


class TestLogAssistantText:
    def test_str_content_done_is_logged(self, capsys):
        """The exact 'DONE' turn from the 1k benchmark — string content."""
        _log([{"type": "message", "content": "DONE"}])
        assert _lines(capsys) == ["[Agent] DONE"]

    def test_list_content_text_block_is_logged(self, capsys):
        """Responses-API style: content is a list of text blocks."""
        _log(
            [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Reached floor 2."}],
                }
            ]
        )
        assert _lines(capsys) == ["[Agent] Reached floor 2."]

    def test_multiple_text_blocks_joined(self, capsys):
        _log(
            [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "First part."},
                        {"type": "output_text", "text": "Second part."},
                    ],
                }
            ]
        )
        # Embedded newlines become the visible continuation marker.
        assert _lines(capsys) == ["[Agent] First part. ⏎ Second part."]

    def test_function_call_items_skipped(self, capsys):
        """Tool invocations are logged by ToolLoggingCallback, not here."""
        _log(
            [
                {"type": "function_call", "name": "computer", "arguments": "{}"},
                {"type": "reasoning", "summary": []},
            ]
        )
        assert _lines(capsys) == []

    def test_empty_text_skipped(self, capsys):
        """Whitespace-only or empty content must not produce a log line."""
        _log(
            [
                {"type": "message", "content": ""},
                {"type": "message", "content": "   \n  "},
                {"type": "message", "content": []},
            ]
        )
        assert _lines(capsys) == []

    def test_long_text_truncated(self, capsys):
        long = "x" * 1200
        _log([{"type": "message", "content": long}])
        out = _lines(capsys)
        assert len(out) == 1
        assert out[0].startswith("[Agent] " + "x" * 500)
        assert "chars]" in out[0]

    def test_mixed_text_and_tool_call(self, capsys):
        """Turns with BOTH text and a tool call: only the text gets an
        [Agent] line (the tool call has its own ToolLoggingCallback hook)."""
        _log(
            [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Thinking about next move."}],
                },
                {"type": "function_call", "name": "computer", "arguments": "{}"},
            ]
        )
        assert _lines(capsys) == ["[Agent] Thinking about next move."]

    def test_done_with_surrounding_whitespace_logged(self, capsys):
        """strip() applied before logging — trailing/leading whitespace OK."""
        _log([{"type": "message", "content": "\n  DONE  \n"}])
        assert _lines(capsys) == ["[Agent] DONE"]
