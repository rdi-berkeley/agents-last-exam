"""Tests for OpenClaw tool registry, summaries, and logging callback.

Covers:
  - build_tools() returns 5 tools with correct types
  - get_tool_summaries() includes Computer and BaseTool instances
  - ToolLoggingCallback logs start/end with timing (function + computer calls)
  - Edge cases: truncation, missing call_id
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from ale_run.agents.ale_claw.harness.tools.tools import (
    COMPUTER_TOOL_SUMMARY,
    ToolLoggingCallback,
    _extract_result_summary,
    build_tools,
    get_tool_summaries,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures — lightweight fakes that avoid heavy CUA imports
# ---------------------------------------------------------------------------


class FakeBaseTool:
    """Minimal stand-in for BaseTool to test get_tool_summaries."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class FakeComputer:
    """Computer-like object with the idealized explicit name."""

    name = "computer"
    description = "ignored dynamic description"


class RealishComputer:
    """Mimics the real CUA Computer shape: no name attribute, class name only."""

    description = "ignored dynamic description"


# ---------------------------------------------------------------------------
# get_tool_summaries
# ---------------------------------------------------------------------------


class TestGetToolSummaries:
    def test_includes_computer_with_stable_summary(self):
        """Computer should appear even though it is not a BaseTool subclass."""
        computer = FakeComputer()
        tools = [computer]
        result = get_tool_summaries(tools)
        assert result == {"computer": COMPUTER_TOOL_SUMMARY}

    def test_includes_realish_computer_without_name_attr(self):
        """The live CUA Computer object is detected by class name, not tool.name."""
        computer = RealishComputer()
        result = get_tool_summaries([computer])
        assert result == {"computer": COMPUTER_TOOL_SUMMARY}

    def test_includes_basetool_instances(self):
        """All BaseTool subclasses should appear in summaries."""
        from ale_run.agents.ale_claw.harness.memory.memory import (
            MemoryGetTool,
            MemorySearchTool,
            MemoryStore,
            MemoryWriteTool,
        )

        store = MemoryStore(task_id="test", base_dir="/tmp/test_tools_summaries")
        tools = [
            MemorySearchTool(store),
            MemoryGetTool(store),
            MemoryWriteTool(store),
        ]
        result = get_tool_summaries(tools)
        assert len(result) == 3
        assert "memory_search" in result
        assert "memory_get" in result
        assert "memory_write" in result

    def test_empty_tool_list(self):
        assert get_tool_summaries([]) == {}

    def test_mixed_tools_correct_count(self):
        """Mixed list should include Computer plus BaseTool summaries only."""
        from ale_run.agents.ale_claw.harness.memory.memory import MemoryGetTool, MemoryStore

        store = MemoryStore(task_id="test", base_dir="/tmp/test_tools_mixed")
        computer = FakeComputer()
        tools = [computer, MemoryGetTool(store)]
        result = get_tool_summaries(tools)
        assert len(result) == 2
        assert "computer" in result
        assert "memory_get" in result


class TestBuildTools:
    def test_vision_thinking_defaults_to_none(self):
        session = MagicMock()
        session.interface = MagicMock()
        session._computer = object()

        with patch("ale_run.agents.ale_claw.harness.analyze_image.AnalyzeImageTool") as mock_analyze_image_tool, patch(
            "ale_run.agents.ale_claw.harness.milestone.MilestoneTool"
        ):
            store = MagicMock()
            build_tools(
                session,
                store,
                summary_model="anthropic/claude-sonnet-4-6-20260101",
            )

        mock_analyze_image_tool.assert_called_once_with(
            session.interface,
            model="anthropic/claude-sonnet-4-6-20260101",
            thinking_params=None,
        )

    def test_passes_vision_thinking_params_to_analyze_image(self):
        session = MagicMock()
        session.interface = MagicMock()
        session._computer = object()

        with patch("ale_run.agents.ale_claw.harness.analyze_image.AnalyzeImageTool") as mock_analyze_image_tool, patch(
            "ale_run.agents.ale_claw.harness.milestone.MilestoneTool"
        ) as mock_milestone_tool:
            store = MagicMock()
            build_tools(
                session,
                store,
                summary_model="anthropic/claude-sonnet-4-6-20260101",
                vision_thinking_params={
                    "thinking": {"type": "enabled", "budget_tokens": 5000}
                },
            )

        mock_milestone_tool.assert_called_once_with(session.interface)
        mock_analyze_image_tool.assert_called_once_with(
            session.interface,
            model="anthropic/claude-sonnet-4-6-20260101",
            thinking_params={"thinking": {"type": "enabled", "budget_tokens": 5000}},
        )

    def test_default_includes_session_computer(self):
        session = MagicMock()
        session.interface = MagicMock()
        sentinel = object()
        session._computer = sentinel

        with patch("ale_run.agents.ale_claw.harness.analyze_image.AnalyzeImageTool"), patch("ale_run.agents.ale_claw.harness.milestone.MilestoneTool"):
            tools = build_tools(
                session,
                MagicMock(),
                summary_model="anthropic/claude-sonnet-4-6-20260101",
            )

        assert sentinel in tools
        # session._computer is the first entry so it shows up at the top of
        # the prompt's tool list alongside the other direct-VM tools.
        assert tools[0] is sentinel

    def test_disable_main_computer_excludes_session_computer(self):
        session = MagicMock()
        session.interface = MagicMock()
        sentinel = object()
        session._computer = sentinel

        with patch("ale_run.agents.ale_claw.harness.analyze_image.AnalyzeImageTool"), patch("ale_run.agents.ale_claw.harness.milestone.MilestoneTool"):
            tools = build_tools(
                session,
                MagicMock(),
                summary_model="anthropic/claude-sonnet-4-6-20260101",
                disable_main_computer=True,
            )

        assert sentinel not in tools
        # Restricted computer + non-computer tools (milestone, analyze_image,
        # read, write, edit, exec, web_search, web_fetch, memory_search,
        # memory_get = 11). memory_write is no longer exposed to the main
        # agent — superseded by write(target='host').
        from ale_run.agents.ale_claw.harness.tools.tools import _RestrictedComputerHandler

        assert len(tools) == 11
        assert isinstance(tools[0], _RestrictedComputerHandler)

    def test_disable_main_computer_pairs_with_delegate_gui(self):
        """When computer is disabled, ``delegate_gui`` must still be exposed so
        the main agent has a path to the VM."""
        from ale_run.agents.ale_claw.harness.subagent.subagent_registry import SubagentRegistry
        from ale_run.agents.ale_claw.harness.subagent.subagent_tools import DelegateGUITool

        session = MagicMock()
        session.interface = MagicMock()
        session._computer = object()
        registry = SubagentRegistry()

        with patch("ale_run.agents.ale_claw.harness.analyze_image.AnalyzeImageTool"), patch("ale_run.agents.ale_claw.harness.milestone.MilestoneTool"):
            tools = build_tools(
                session,
                MagicMock(),
                summary_model="anthropic/claude-sonnet-4-6-20260101",
                registry=registry,
                parent_session_dir="/tmp/parent",
                disable_main_computer=True,
            )

        assert any(isinstance(t, DelegateGUITool) for t in tools)
        assert session._computer not in tools

    def test_disable_delegate_gui_excludes_tool(self):
        """When ``disable_delegate_gui=True``, DelegateGUITool is omitted but
        DelegateGeneralTool and SubagentsTool remain."""
        from ale_run.agents.ale_claw.harness.subagent.subagent_registry import SubagentRegistry
        from ale_run.agents.ale_claw.harness.subagent.subagent_tools import (
            DelegateGeneralTool,
            DelegateGUITool,
            SubagentsTool,
        )

        session = MagicMock()
        session.interface = MagicMock()
        session._computer = object()
        registry = SubagentRegistry()

        with patch("ale_run.agents.ale_claw.harness.analyze_image.AnalyzeImageTool"), patch("ale_run.agents.ale_claw.harness.milestone.MilestoneTool"):
            tools = build_tools(
                session,
                MagicMock(),
                summary_model="anthropic/claude-sonnet-4-6-20260101",
                registry=registry,
                parent_session_dir="/tmp/parent",
                disable_delegate_gui=True,
            )

        assert not any(isinstance(t, DelegateGUITool) for t in tools)
        assert any(isinstance(t, DelegateGeneralTool) for t in tools)
        assert any(isinstance(t, SubagentsTool) for t in tools)

    def test_disable_delegate_gui_pairs_with_main_computer(self):
        """When delegate_gui is disabled, the main ``computer`` tool must
        remain intact so the agent can drive the VM directly."""
        from ale_run.agents.ale_claw.harness.subagent.subagent_registry import SubagentRegistry

        session = MagicMock()
        session.interface = MagicMock()
        sentinel = object()
        session._computer = sentinel
        registry = SubagentRegistry()

        with patch("ale_run.agents.ale_claw.harness.analyze_image.AnalyzeImageTool"), patch("ale_run.agents.ale_claw.harness.milestone.MilestoneTool"):
            tools = build_tools(
                session,
                MagicMock(),
                summary_model="anthropic/claude-sonnet-4-6-20260101",
                registry=registry,
                parent_session_dir="/tmp/parent",
                disable_delegate_gui=True,
            )

        assert sentinel in tools
        assert tools[0] is sentinel


# ---------------------------------------------------------------------------
# ToolLoggingCallback — function calls
# ---------------------------------------------------------------------------


class TestToolLoggingCallback:
    @pytest.fixture
    def callback(self):
        return ToolLoggingCallback()

    def test_on_function_call_start_logs(self, callback, capsys):
        _run(callback.on_function_call_start({
            "call_id": "c1",
            "name": "memory_search",
            "arguments": '{"keywords": ["test"]}',
        }))
        captured = capsys.readouterr()
        assert "[Tool] memory_search(" in captured.out
        assert "c1" in callback._start_times

    def test_on_function_call_end_logs_duration(self, callback, capsys):
        _run(callback.on_function_call_start({
            "call_id": "c2",
            "name": "memory_get",
            "arguments": "{}",
        }))
        capsys.readouterr()  # clear start output

        _run(callback.on_function_call_end(
            {"call_id": "c2", "name": "memory_get"},
            [{"output": "some result data"}],
        ))
        captured = capsys.readouterr()
        assert "[Tool] memory_get -> some result data" in captured.out
        assert "ms)" in captured.out
        assert "c2" not in callback._start_times

    def test_on_function_call_end_missing_start(self, callback, capsys):
        """End without start should show ?ms."""
        _run(callback.on_function_call_end(
            {"call_id": "orphan", "name": "memory_write"},
            [{"output": "ok"}],
        ))
        captured = capsys.readouterr()
        assert "?ms" in captured.out

    def test_args_truncation(self, callback, capsys):
        """Long arguments should be truncated."""
        long_args = json.dumps({"data": "x" * 500})
        _run(callback.on_function_call_start({
            "call_id": "c3",
            "name": "memory_write",
            "arguments": long_args,
        }))
        captured = capsys.readouterr()
        assert "…" in captured.out

    def test_dict_arguments(self, callback, capsys):
        """Arguments passed as dict (not string) should be handled."""
        _run(callback.on_function_call_start({
            "call_id": "c4",
            "name": "memory_search",
            "arguments": {"keywords": ["hello"]},
        }))
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_missing_call_id_defaults(self, callback, capsys):
        """Missing call_id should use 'unknown'."""
        _run(callback.on_function_call_start({
            "name": "memory_get",
            "arguments": "{}",
        }))
        assert "unknown" in callback._start_times


# ---------------------------------------------------------------------------
# ToolLoggingCallback — computer calls
# ---------------------------------------------------------------------------


class TestToolLoggingCallbackComputer:
    @pytest.fixture
    def callback(self):
        return ToolLoggingCallback()

    def test_on_computer_call_start_logs_action_type(self, callback, capsys):
        _run(callback.on_computer_call_start({
            "call_id": "cc1",
            "action": {"type": "click", "x": 100, "y": 200},
        }))
        captured = capsys.readouterr()
        assert "[Computer] click" in captured.out
        assert "cc1" in callback._start_times

    def test_on_computer_call_end_logs_duration(self, callback, capsys):
        _run(callback.on_computer_call_start({
            "call_id": "cc2",
            "action": {"type": "screenshot"},
        }))
        capsys.readouterr()

        _run(callback.on_computer_call_end(
            {"call_id": "cc2", "action": {"type": "screenshot"}},
            [{"output": {"type": "input_image"}}],
        ))
        captured = capsys.readouterr()
        assert "[Computer] screenshot done" in captured.out
        assert "ms)" in captured.out
        assert "cc2" not in callback._start_times

    def test_on_computer_call_end_missing_start(self, callback, capsys):
        _run(callback.on_computer_call_end(
            {"call_id": "orphan", "action": {"type": "key"}},
            [],
        ))
        captured = capsys.readouterr()
        assert "?ms" in captured.out

    def test_on_computer_call_non_dict_action(self, callback, capsys):
        """Non-dict action should be stringified."""
        _run(callback.on_computer_call_start({
            "call_id": "cc3",
            "action": "some_string_action",
        }))
        captured = capsys.readouterr()
        assert "[Computer] some_string_action" in captured.out


# ---------------------------------------------------------------------------
# _extract_result_summary
# ---------------------------------------------------------------------------


class TestExtractResultSummary:
    def test_empty_result(self):
        assert _extract_result_summary([]) == "(empty)"

    def test_no_output_field(self):
        assert _extract_result_summary([{"type": "something"}]) == "(no output)"

    def test_short_output(self):
        assert _extract_result_summary([{"output": "hello"}]) == "hello"

    def test_long_output_truncated(self):
        long_output = "x" * 200
        result = _extract_result_summary([{"output": long_output}])
        assert len(result) == 101  # 100 + ellipsis
        assert result.endswith("…")

    def test_multiple_items_uses_first(self):
        result = _extract_result_summary([
            {"output": "first"},
            {"output": "second"},
        ])
        assert result == "first"
