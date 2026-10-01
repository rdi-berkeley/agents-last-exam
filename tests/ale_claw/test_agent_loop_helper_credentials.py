"""Tests for per-run credentials reaching ALE-Claw helper calls."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from ale_run.agents.ale_claw.harness import agent_loop
from ale_run.agents.ale_claw.harness.agent_loop import OpenClawComputerAgent
from ale_run.agents.ale_claw.harness.context.compaction import CompactionResult


class TestHelperCredentials:
    def test_run_setup_captures_effective_run_credentials(self):
        agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
        agent.agent_config_info = SimpleNamespace(agent_class=OpenClawComputerAgent)
        agent.get_capabilities = MagicMock(return_value={"step"})
        agent._initialize_computers = AsyncMock()
        agent._process_input = MagicMock(return_value=[])
        agent._on_run_start = AsyncMock()
        agent.kwargs = {}
        agent.model = "test-model"
        agent.api_key = "default-key"
        agent.api_base = "http://default-endpoint/v1"

        _, run_kwargs, merged_kwargs = asyncio.run(
            agent._run_setup(
                messages=[],
                stream=False,
                api_key="run-key",
                api_base="http://127.0.0.1:4010/v1",
                additional_generation_kwargs={},
            )
        )

        assert agent._helper_api_key == "run-key"
        assert agent._helper_api_base == "http://127.0.0.1:4010/v1"
        assert merged_kwargs["api_key"] == "run-key"
        assert merged_kwargs["api_base"] == "http://127.0.0.1:4010/v1"
        assert run_kwargs["api_key"] == "run-key"
        assert run_kwargs["api_base"] == "http://127.0.0.1:4010/v1"

    def test_memory_flush_receives_run_credentials(self, tmp_path):
        agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
        agent.session_mgr = SimpleNamespace(
            _state=object(),
            transcript_path=tmp_path / "transcript.jsonl",
        )
        agent.overflow_cb = SimpleNamespace(
            current_tokens=100,
            context_window=1000,
            compaction_threshold_ratio=0.8,
        )
        agent.memory_store = MagicMock()
        agent.summary_model = "summary-model"
        agent.summary_runtime = None
        agent.thinking_config = None
        agent._helper_api_key = "run-key"
        agent._helper_api_base = "http://127.0.0.1:4010/v1"

        with (
            patch.object(agent_loop, "should_run_memory_flush", return_value=True),
            patch.object(agent_loop, "run_memory_flush", new_callable=AsyncMock) as mock_flush,
        ):
            asyncio.run(agent._maybe_flush_memory())

        mock_flush.assert_awaited_once()
        assert mock_flush.await_args.kwargs["api_key"] == "run-key"
        assert mock_flush.await_args.kwargs["api_base"] == "http://127.0.0.1:4010/v1"

    def test_compaction_receives_run_credentials(self):
        agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
        agent.session_mgr = SimpleNamespace(
            _state=None,
            load_history=MagicMock(return_value=[]),
            append_compaction=MagicMock(),
        )
        agent.overflow_cb = SimpleNamespace(
            context_window=1000,
            reset_after_compaction=MagicMock(),
        )
        agent.summary_model = "summary-model"
        agent.summary_runtime = None
        agent.thinking_config = None
        agent.instructions = ""
        agent.resolved_model = None
        agent.model = "openai/gpt-5.4"
        agent._compaction_count = 0
        agent._on_compaction = None
        agent._context_files = []
        agent._helper_api_key = "run-key"
        agent._helper_api_base = "http://127.0.0.1:4010/v1"
        compaction_result = CompactionResult(
            summary="summary",
            tokens_before=10,
            tokens_after=5,
            first_kept_message_index=0,
            chunks_processed=1,
        )

        with patch.object(
            agent_loop,
            "compact_messages",
            new_callable=AsyncMock,
            return_value=compaction_result,
        ) as mock_compact:
            asyncio.run(agent._compact_in_place([], []))

        mock_compact.assert_awaited_once()
        assert mock_compact.await_args.kwargs["api_key"] == "run-key"
        assert mock_compact.await_args.kwargs["api_base"] == "http://127.0.0.1:4010/v1"
