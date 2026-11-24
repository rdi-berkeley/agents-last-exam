"""Regression coverage for reasoning replay and generation settings."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from litellm.types.utils import ModelResponse

from ale_claw.harness.canonical.canonical import normalize_to_canonical, sanitize_items
from ale_claw.harness.context.replay import convert_to_responses_api_items
from ale_claw.harness.context.transcript import group_step_output
from ale_claw.harness.model.model_config import merge_helper_generation_kwargs, model_generation_kwargs, resolve_model
from ale_claw.harness.model.unified_loop import _convert_input_to_messages, _convert_response_to_output
from ale_claw.harness.subagent.subagent_registry import SubagentRegistry
from ale_claw.harness.subagent.subagent_session import GeneralSubagentSession


DETAILS = [
    {"type": "reasoning.text", "text": "think", "signature": "signed", "index": 0},
    {"type": "reasoning.encrypted", "data": "opaque", "index": 1},
]


def response(details=DETAILS, text="think"):
    return ModelResponse(choices=[{"message": {
        "role": "assistant", "content": "Using tools", "reasoning_content": text,
        "provider_specific_fields": {"reasoning_details": details},
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "first", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "second", "arguments": "{}"}},
        ],
    }}])


def assert_turn(messages):
    assistant = next(message for message in messages if message["role"] == "assistant")
    assert assistant["content"] == "Using tools"
    assert assistant["reasoning_content"] == "think"
    assert assistant["reasoning_details"] == DETAILS
    assert [call["id"] for call in assistant["tool_calls"]] == ["call_1", "call_2"]
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["call_1", "call_2"]


def test_reasoning_survives_live_transcript_replay_and_compaction():
    output = _convert_response_to_output(response())["output"]
    results = [
        {"type": "function_call_output", "call_id": "call_1", "output": "a"},
        {"type": "function_call_output", "call_id": "call_2", "output": "b"},
    ]
    assert_turn(_convert_input_to_messages([{"role": "user", "content": "go"}, *output, *results]))

    assistant, tool = group_step_output([*output, *results])
    history = [{"role": "user", "content": "go"}, {"role": "assistant", "content": assistant}, {"role": "tool", "content": tool}]
    assert_turn(_convert_input_to_messages(convert_to_responses_api_items(history)))
    for model in ("openrouter/deepseek/deepseek-v4.1-flash-1017", "openrouter/openai/gpt-5.4", "openrouter/anthropic/claude-sonnet-4.6"):
        compacted = sanitize_items(normalize_to_canonical(history), model=model)
        assert_turn(_convert_input_to_messages(compacted))


def test_encrypted_reasoning_without_plaintext_is_preserved():
    output = _convert_response_to_output(response(details=DETAILS, text=None))["output"]
    reasoning = next(item for item in output if item["type"] == "reasoning")
    assert reasoning["reasoning_details"] == DETAILS
    assistant, _ = group_step_output(output)
    assert assistant[0]["reasoning_details"] == DETAILS
    messages = _convert_input_to_messages(output)
    assert messages[0]["reasoning_details"] == DETAILS
    assert messages[0]["content"] == "Using tools"


def test_parallel_tool_results_stay_adjacent_before_screenshot():
    output = _convert_response_to_output(response())["output"]
    items = [
        {"role": "user", "content": "go"}, *output,
        {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]},
        {"type": "function_call_output", "call_id": "call_2", "output": "ok"},
    ]
    messages = _convert_input_to_messages(items)
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "tool", "user"]


def test_general_subagent_keeps_reasoning_and_generation(tmp_path):
    session = GeneralSubagentSession(
        run_id="worker", task="test", model="openrouter/deepseek/test", tools=[],
        registry=SubagentRegistry(), summary_model="openrouter/deepseek/test",
        parent_session_dir=tmp_path,
        generation_kwargs={"top_p": 0.95, "temperature": 0.7, "extra_body": {"provider": {"order": ["x"]}}},
    )
    message = response().choices[0].message
    from ale_claw.harness.model.reasoning import reasoning_fields
    session._append_assistant(message.content, message.tool_calls, reasoning=reasoning_fields(message))
    mock = SimpleNamespace(acompletion=AsyncMock(return_value=response()))
    asyncio.run(session._call_llm(mock, resolve_model("openrouter/deepseek/test")))
    call = mock.acompletion.call_args.kwargs
    assert call["top_p"] == 0.95 and call["temperature"] == 0.7
    assert call["extra_body"]["provider"]["order"] == ["x"]
    assert call["messages"][-1]["reasoning_details"] == DETAILS
    entries = session.session_mgr.load_history()
    block = entries[-1].data["message"]["content"][0]
    assert block["reasoning_details"] == DETAILS


def test_generation_options_respect_provider_and_owned_limits():
    configured = {"top_p": 0.95, "temperature": 1.0, "extra_body": {"provider": {"order": ["x"]}}, "max_tokens": 9, "truncation": "disabled"}
    same = model_generation_kwargs("openrouter/deepseek/test", "openrouter/other/test", configured)
    assert same == {"top_p": 0.95, "temperature": 1.0, "extra_body": configured["extra_body"]}
    other = model_generation_kwargs("openrouter/deepseek/test", "anthropic/claude-test", configured)
    assert other == {"top_p": 0.95, "temperature": 1.0}


def test_helper_thinking_overrides_do_not_erase_sampling_or_routing():
    options = {
        "top_p": .95,
        "reasoning": {"effort": "high"},
        "extra_body": {"provider": {"order": ["chosen"]}, "thinking": {"type": "enabled"}},
    }
    assert merge_helper_generation_kwargs(options, {"reasoning": {"effort": "low"}}) == options
    override = merge_helper_generation_kwargs(options, {}, override_thinking=True)
    assert override == {"top_p": .95, "extra_body": {"provider": {"order": ["chosen"]}}}
    override = merge_helper_generation_kwargs(options, {"reasoning": {"effort": "low"}}, override_thinking=True)
    assert override["reasoning"] == {"effort": "low"}
    assert "thinking" not in override["extra_body"]
    assert options["extra_body"]["thinking"] == {"type": "enabled"}


def test_tool_assembly_inherits_generation_options(tmp_path):
    from unittest.mock import MagicMock
    from ale_claw.harness.tools.tools import build_tools
    session = MagicMock()
    session._computer = object()
    options = {"top_p": 0.95, "extra_body": {"provider": {"order": ["chosen"]}}}
    tools = build_tools(
        session, MagicMock(), registry=SubagentRegistry(), parent_session_dir=tmp_path,
        default_model="openrouter/deepseek/main", summary_model="openrouter/deepseek/helper",
        gui_model="openrouter/deepseek/gui", generation_kwargs=options,
    )
    named = {getattr(tool, "name", ""): tool for tool in tools}
    assert named["analyze_image"].thinking_params == options
    assert named["delegate_general"]._generation_kwargs == options
    assert named["delegate_gui"]._generation_kwargs == options


def test_vision_default_off_removes_inherited_thinking(tmp_path):
    from unittest.mock import MagicMock
    from ale_claw.harness.tools.tools import build_tools
    session = MagicMock()
    session._computer = object()
    tools = build_tools(
        session, MagicMock(), registry=SubagentRegistry(), parent_session_dir=tmp_path,
        default_model="openrouter/deepseek/main", summary_model="openrouter/deepseek/helper",
        generation_kwargs={"top_p": .95, "reasoning": {"effort": "high"}},
        vision_thinking_params={},
    )
    named = {getattr(tool, "name", ""): tool for tool in tools}
    assert named["analyze_image"].thinking_params == {"top_p": .95}


def test_memory_flush_receives_generation_options(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from ale_claw.harness import agent_loop
    agent = agent_loop.OpenClawComputerAgent.__new__(agent_loop.OpenClawComputerAgent)
    agent.session_mgr = SimpleNamespace(_state=object(), transcript_path=tmp_path / "transcript")
    agent.overflow_cb = SimpleNamespace(current_tokens=100, context_window=1000, compaction_threshold_ratio=.8)
    agent.memory_store = MagicMock()
    agent.summary_model = "openrouter/deepseek/helper"
    agent.summary_runtime = None
    agent.thinking_config = None
    agent._helper_api_key = None
    agent._helper_api_base = None
    agent._helper_generation_kwargs = {"top_p": .95, "extra_body": {"provider": {"order": ["chosen"]}}}
    mock = AsyncMock()
    monkeypatch.setattr(agent_loop, "should_run_memory_flush", lambda *a, **k: True)
    monkeypatch.setattr(agent_loop, "run_memory_flush", mock)
    asyncio.run(agent._maybe_flush_memory())
    assert mock.call_args.kwargs["thinking_params"] == agent._helper_generation_kwargs


def test_subagent_compaction_uses_summary_generation_options(tmp_path, monkeypatch):
    from ale_claw.harness.subagent import subagent_session
    session = GeneralSubagentSession(
        run_id="worker", task="test", model="openrouter/deepseek/main", tools=[],
        registry=SubagentRegistry(), summary_model="openrouter/deepseek/helper",
        parent_session_dir=tmp_path, generation_kwargs={"top_p": .5},
        summary_generation_kwargs={"top_p": .95, "extra_body": {"provider": {"order": ["chosen"]}}},
    )
    mock = AsyncMock(return_value=SimpleNamespace(summary="summary", first_kept_message_index=0, tokens_before=1, tokens_after=1))
    monkeypatch.setattr(subagent_session, "compact_messages", mock)
    asyncio.run(session._compact_in_place())
    assert mock.call_args.kwargs["thinking_params"] == session._summary_generation_kwargs


def test_subagent_compaction_explicit_off_overrides_inherited_thinking(tmp_path, monkeypatch):
    from ale_claw.harness.subagent import subagent_session
    session = GeneralSubagentSession(
        run_id="worker", task="test", model="openrouter/deepseek/main", tools=[],
        registry=SubagentRegistry(), summary_model="openrouter/deepseek/helper",
        parent_session_dir=tmp_path, summary_thinking_params={}, summary_thinking_overridden=True,
        summary_generation_kwargs={"top_p": .95, "reasoning": {"effort": "high"}},
    )
    mock = AsyncMock(return_value=SimpleNamespace(summary="summary", first_kept_message_index=0, tokens_before=1, tokens_after=1))
    monkeypatch.setattr(subagent_session, "compact_messages", mock)
    asyncio.run(session._compact_in_place())
    assert mock.call_args.kwargs["thinking_params"] == {"top_p": .95}


def test_subagent_thinking_is_resolved_for_selected_models(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from ale_claw.harness.model.thinking import ThinkLevel, ThinkingConfig
    from ale_claw.harness.subagent import subagent_tools
    registry = SubagentRegistry()
    capture = AsyncMock()
    monkeypatch.setattr(subagent_tools, "run_general_subagent", capture)
    tool = subagent_tools.DelegateGeneralTool(
        registry=registry, tools=[], memory_store=MagicMock(),
        default_model="openrouter/deepseek/main", auxiliary_model="anthropic/claude-test",
        summary_model="anthropic/claude-test", parent_session_dir=tmp_path,
        thinking_config=ThinkingConfig(level=ThinkLevel.HIGH, compaction_level=ThinkLevel.LOW),
        generation_kwargs={"top_p": .95, "extra_body": {"provider": {"order": ["chosen"]}}},
    )
    async def scenario():
        result = tool.call({"task": "analyze", "model": "anthropic/claude-test"})
        await registry._tasks[result["run_id"]]
    asyncio.run(scenario())
    kwargs = capture.call_args.kwargs
    assert kwargs["thinking_params"]["thinking"]["budget_tokens"] == 16000
    assert kwargs["summary_thinking_params"]["thinking"]["budget_tokens"] == 5000
    assert kwargs["generation_kwargs"] == {"top_p": .95}
    assert kwargs["summary_generation_kwargs"] == {"top_p": .95}
