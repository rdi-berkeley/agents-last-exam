import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ale_run.agents.ale_claw.harness.agent_loop import OpenClawComputerAgent
from ale_run.agents.ale_claw.harness.model.model_config import resolve_model
from ale_run.agents.ale_claw.harness.model.helper_runtime import call_helper_model


@pytest.mark.parametrize(
    ("primary", "summary", "forward"),
    [
        ("openai/gpt-5.4", "openai/gpt-5-mini", True),
        ("openai/gpt-5.4", "anthropic/claude-sonnet-4", False),
        ("openrouter/openai/gpt-5.4", "openrouter/anthropic/claude-sonnet-4", True),
        ("openrouter/openai/gpt-5.4", "openai/gpt-5-mini", False),
        ("gpt-5.4", "openai/gpt-5-mini", True),
        ("unknown-primary", "unknown-summary", False),
    ],
)
@pytest.mark.parametrize("runtime_override", [False, True])
def test_run_credentials_stay_with_summary_route(primary, summary, forward, runtime_override):
    agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
    agent.agent_config_info = SimpleNamespace(agent_class=OpenClawComputerAgent)
    agent.get_capabilities = MagicMock(return_value={"step"})
    agent._initialize_computers = AsyncMock()
    agent._process_input = MagicMock(return_value=[])
    agent._on_run_start = AsyncMock()
    agent.kwargs = {}
    agent.model = primary
    agent.summary_model = "ignored" if runtime_override else summary
    agent.summary_runtime = resolve_model(summary) if runtime_override else None
    agent.api_key = "synthetic-default"
    agent.api_base = "https://default.invalid/v1"
    asyncio.run(agent._run_setup([], False, "synthetic-run", "https://run.invalid/v1", {}))
    assert agent._helper_api_key == ("synthetic-run" if forward else None)
    assert agent._helper_api_base == ("https://run.invalid/v1" if forward else None)
    agent.api_key = None
    agent.api_base = None
    asyncio.run(agent._run_setup([], False, None, None, {}))
    assert agent._helper_api_key is None and agent._helper_api_base is None


@pytest.mark.parametrize(
    ("model", "purpose", "transport"),
    [
        ("openai/gpt-5.4", "memory_flush", "responses"),
        ("openai/gpt-5.4", "compaction", "chat"),
        ("anthropic/claude-sonnet-4", "memory_flush", "chat"),
        ("anthropic/claude-sonnet-4", "compaction", "chat"),
    ],
)
def test_helper_forwards_connection_to_both_transports(monkeypatch, model, purpose, transport):
    import litellm

    chat = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="summary", tool_calls=[]))]
        )
    )
    responses = AsyncMock(
        return_value=SimpleNamespace(
            model_dump=lambda: {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "summary"}]}
                ]
            }
        )
    )
    monkeypatch.setattr(litellm, "acompletion", chat)
    monkeypatch.setattr(litellm, "aresponses", responses)
    result = asyncio.run(
        call_helper_model(
            model,
            purpose=purpose,
            messages=[],
            max_tokens=10,
            temperature=1,
            api_key="synthetic-run",
            api_base="https://run.invalid/v1",
        )
    )
    invoked = responses if transport == "responses" else chat
    assert invoked.await_count == 1
    assert invoked.await_args.kwargs["api_key"] == "synthetic-run"
    assert invoked.await_args.kwargs["api_base"] == "https://run.invalid/v1"
    assert result.text == "summary"
