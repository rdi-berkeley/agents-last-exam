"""Regression coverage for session-backed GUI, lifecycle and trajectory handling."""

import asyncio
import base64
import json
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from ale_claw.harness.adapters.trajectory_saver import (
    OpenClawTrajectorySaverCallback,
)
from ale_claw.harness.agent_loop import OpenClawComputerAgent
from ale_claw.harness.context.transcript import group_step_output
from ale_claw.harness.model.model_config import model_credentials
from ale_claw.harness.model.unified_loop import _convert_response_to_output
from ale_claw.harness.subagent.subagent_gui import (
    _TranscriptWriter,
    _relay_until_done,
    run_gui_subagent,
)
from ale_claw.harness.subagent.subagent_registry import (
    SubagentRegistry,
    SubagentType,
    SubagentUsage,
)
from ale_claw.harness.subagent.subagent_tools import DelegateGUITool
from ale_claw.harness.tools.computer_handler import OpenClawComputerHandler
from ale_claw.harness.tools.fs_backends import FilesystemRegistry, VMBackend
from ale_claw.harness.tools.tools_fs import WriteFileTool
from ale_run.agents.ale_claw.transcript_to_trajectory import parse_transcripts_into
from ale_run.base_interface import TrajectoryBuilder


def test_gui_chord_sequence_and_image_coordinates(monkeypatch):
    handler = OpenClawComputerHandler(MagicMock())
    handler.interface = SimpleNamespace(hotkey=AsyncMock(), press_key=AsyncMock())
    handler.get_dimensions = AsyncMock(return_value=(1920, 1080))
    click = AsyncMock()
    monkeypatch.setattr("agent.computers.cua.cuaComputerHandler.click", click)

    async def exercise():
        assert await handler.get_model_dimensions() == (1200, 675)
        handler.set_screenshot_dimensions(1200, 675)
        await handler.click(600, 337)
        await handler.keypress(["ctrl", "a"])
        await handler.keypress(["up", "up"])

    asyncio.run(exercise())
    assert click.await_args.args[:2] == (960, 539)
    handler.interface.hotkey.assert_awaited_once_with("ctrl", "a")
    assert handler.interface.press_key.await_count == 2


@pytest.mark.parametrize(
    ("os_type", "expected"), [("windows", "windows"), ("linux", "linux"), ("macos", "mac")]
)
def test_computer_tool_reports_guest_os(os_type, expected):
    handler = OpenClawComputerHandler(SimpleNamespace(os_type=os_type))
    assert asyncio.run(handler.get_environment()) == expected


def _fake_agent(result):
    agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
    agent._run_setup = AsyncMock(return_value=([], {}, {}))
    agent._on_run_continue = AsyncMock(return_value=True)
    agent._on_llm_start = AsyncMock(side_effect=lambda values: values)
    agent._sanitize_runtime_messages = MagicMock(side_effect=lambda values: values)
    agent._maybe_flush_memory = AsyncMock()
    agent.agent_loop = SimpleNamespace(predict_step=AsyncMock(return_value=result))
    agent._on_llm_end = AsyncMock(side_effect=lambda values: values)
    agent._on_responses = AsyncMock()
    agent._on_api_start = AsyncMock()
    agent._on_api_end = AsyncMock()
    agent._on_usage = AsyncMock()
    agent._on_screenshot = AsyncMock()
    agent._sanitize_truncated_function_calls = MagicMock()
    agent._log_step_to_transcript = MagicMock()
    agent._log_assistant_text = MagicMock()
    agent._handle_item = AsyncMock(
        return_value=[{"type": "function_call_output", "call_id": "a", "output": "ok"}]
    )
    agent._log_partial_items_to_transcript = MagicMock()
    agent._drain_completions = MagicMock()
    agent._drain_post_delegation = MagicMock()
    agent._on_run_end = AsyncMock()
    agent.model = "openrouter/test"
    agent.tool_schemas = []
    agent.computer_handler = None
    agent.max_retries = 0
    agent.use_prompt_caching = False
    agent.session_mgr = MagicMock()
    return agent


def test_last_budget_turn_executes_and_finalizes():
    result = {
        "output": [{"type": "function_call", "name": "write", "call_id": "a", "arguments": "{}"}],
        "usage": {},
    }
    agent = _fake_agent(result)

    async def exercise():
        async for _ in agent.run("task", max_turns=1):
            agent._handle_item.assert_awaited_once()

    asyncio.run(exercise())
    agent._log_partial_items_to_transcript.assert_called_once()
    assert agent._on_run_end.await_args.args[0]["ale_claw_status"] == "failed"


def test_done_is_logged_and_callback_finalizes():
    result = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "DONE"}],
            }
        ],
        "usage": {},
    }
    agent = _fake_agent(result)

    async def exercise():
        return [item async for item in agent.run("task", max_turns=1)]

    assert len(asyncio.run(exercise())) == 1
    agent._log_step_to_transcript.assert_called_once()
    assert agent._on_run_end.await_args.args[0]["ale_claw_status"] == "completed"


def test_key_redacted_in_callback_artifacts(tmp_path):
    callback = OpenClawTrajectorySaverCallback(str(tmp_path))
    key = "secret-canary-example"

    async def exercise():
        await callback.on_run_start({"model": "test", "api_key": key}, [])
        await callback.on_api_start({"headers": {"X-API-Key": key}, "message": "error " + key})
        await callback.on_run_end({"ale_claw_status": "failed", "api_key": key}, [], [])

    asyncio.run(exercise())
    for path in tmp_path.rglob("*.json"):
        assert key not in path.read_text()
    metadata = json.loads(next(tmp_path.rglob("metadata.json")).read_text())
    assert metadata["status"] == "failed"


def test_credentials_do_not_cross_provider_boundary():
    assert model_credentials("openrouter/a", "openrouter/b", "key", "https://gateway") == {
        "api_key": "key",
        "api_base": "https://gateway",
    }
    assert model_credentials("openrouter/a", "anthropic/b", "key", "https://gateway") == {}


def test_screenshot_and_failure_make_it_into_trajectory(tmp_path):
    session = tmp_path / "openclaw_sessions" / "task"
    session.mkdir(parents=True)
    image = base64.b64encode(b"fake screenshot").decode()
    call = {"type": "function_call", "name": "computer", "call_id": "c1", "arguments": "{}"}
    result = {"type": "function_call_output", "call_id": "c1", "output": '{"error":"failed"}'}
    user_image = {
        "role": "user",
        "content": [{"type": "input_image", "image_url": f"data:image/png;base64,{image}"}],
    }
    assistant, _ = group_step_output([call], tmp_path / "trajectories")
    _, results = group_step_output([result, user_image], tmp_path / "trajectories")
    stamp = "2026-01-01T00:00:00Z"
    entries = [
        {"type": "message", "timestamp": stamp, "message": {"role": role, "content": content}}
        for role, content in (("assistant", assistant), ("tool", results))
    ]
    (session / "transcript.jsonl").write_text("\n".join(json.dumps(entry) for entry in entries))
    builder = TrajectoryBuilder(agent_name="ale_claw", task_path="demo", variant_index=0)
    parse_transcripts_into(tmp_path, builder)
    steps = builder.trajectory.steps
    assert steps[0].tool_calls[0].id == "c1"
    assert steps[1].timestamp == stamp
    observation = steps[1].observation.results[0]
    assert observation.tool_call_id == "c1" and observation.is_error
    assert observation.content[1].image.data == image


def test_gui_relay_consumes_output_after_last_call(tmp_path):
    transcript = _TranscriptWriter(
        tmp_path / "openclaw_sessions" / "task" / "subagents" / "sub1" / "transcript.jsonl"
    )
    completed = []

    class Agent:
        async def run(self, instruction, **kwargs):
            yield {
                "output": [
                    {
                        "type": "function_call",
                        "name": "computer",
                        "call_id": "c1",
                        "arguments": '{"action":"terminate"}',
                    }
                ]
            }
            completed.append(True)
            yield {"output": [{"type": "function_call_output", "call_id": "c1", "output": "ok"}]}

    asyncio.run(_relay_until_done(Agent(), "task", "sub1", 1, SubagentUsage(), transcript))
    assert completed == [True]
    entries = [json.loads(line) for line in transcript._path.read_text().splitlines()]
    assert [entry["message"]["role"] for entry in entries] == ["assistant", "tool"]


def test_gui_exhaustion_is_failure(tmp_path):
    transcript = _TranscriptWriter(
        tmp_path / "openclaw_sessions" / "task" / "subagents" / "sub1" / "transcript.jsonl"
    )

    class Agent:
        async def run(self, instruction, **kwargs):
            yield {
                "output": [
                    {
                        "type": "function_call",
                        "name": "computer",
                        "call_id": "c1",
                        "arguments": '{"action":"click","x":1,"y":2}',
                    }
                ]
            }

    with pytest.raises(RuntimeError, match="without terminate"):
        asyncio.run(_relay_until_done(Agent(), "task", "sub1", 1, SubagentUsage(), transcript))


def test_gui_subagent_dispatches_real_sdk_function_calls(tmp_path, monkeypatch):
    import litellm

    image = io.BytesIO()
    Image.new("RGB", (1920, 1080), "white").save(image, format="PNG")
    interface = SimpleNamespace(
        get_screen_size=AsyncMock(return_value={"width": 1920, "height": 1080}),
        screenshot=AsyncMock(return_value=image.getvalue()),
        left_click=AsyncMock(),
    )
    session = SimpleNamespace(computer=SimpleNamespace(interface=interface))
    requests = []
    actions = iter(
        [
            {"action": "screenshot"},
            {"action": "click", "x": 600, "y": 337},
            {"action": "terminate", "status": "success"},
        ]
    )

    async def completion(**kwargs):
        requests.append(kwargs)
        action = next(actions)
        return litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{len(requests)}",
                                "type": "function",
                                "function": {
                                    "name": "computer",
                                    "arguments": json.dumps(action),
                                },
                            }
                        ],
                    },
                }
            ],
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 20},
                "cost": 0.01,
            },
        )

    monkeypatch.setattr(litellm, "acompletion", completion)
    session_dir = tmp_path / "openclaw_sessions" / "task"
    registry = SubagentRegistry(persist_path=session_dir / "subagent-runs.jsonl")
    run = registry.register(type=SubagentType.GUI, task="click", model="openrouter/test")
    asyncio.run(
        run_gui_subagent(
            instruction="click",
            session=session,
            registry=registry,
            run_id=run.run_id,
            model="openrouter/test",
            max_steps=3,
            parent_session_dir=session_dir,
        )
    )

    interface.left_click.assert_awaited_once_with(960, 539)
    assert len(requests) == 3
    assert "1200x675" in requests[0]["tools"][0]["function"]["description"]
    assert any("data:image/" in json.dumps(request["messages"]) for request in requests[1:])
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "transcript.jsonl").write_text("")
    builder = TrajectoryBuilder(agent_name="ale_claw", task_path="demo", variant_index=0)
    parse_transcripts_into(tmp_path, builder)
    child = builder.trajectory.subagent_trajectories[0]
    calls = [call.id for step in child.steps for call in step.tool_calls]
    results = [
        result for step in child.steps if step.observation for result in step.observation.results
    ]
    assert calls == ["call-1", "call-2", "call-3"]
    assert [result.tool_call_id for result in results] == calls
    assert any(part.type == "image" for result in results for part in result.content)
    assert child.final_metrics.total_input_tokens == 300
    assert child.final_metrics.total_cache_read_tokens == 60
    assert child.final_metrics.total_cost_usd == pytest.approx(0.03)


def test_cancelled_gui_subagent_persists_usage(tmp_path, monkeypatch):
    registry = SubagentRegistry(persist_path=tmp_path / "subagent-runs.jsonl")
    run = registry.register(type=SubagentType.GUI, task="task", model="test")
    reached = asyncio.Event()

    class Agent:
        async def run(self, instruction, **kwargs):
            yield {"output": [], "usage": {"input_tokens": 12, "output_tokens": 3}}
            reached.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(
        "ale_claw.harness.subagent.subagent_gui._build_gui_agent",
        AsyncMock(return_value=Agent()),
    )

    async def exercise():
        task = asyncio.create_task(
            run_gui_subagent(
                instruction="task",
                session=object(),
                registry=registry,
                run_id=run.run_id,
                model="test",
                parent_session_dir=tmp_path,
            )
        )
        registry.attach_task(run.run_id, task)
        await reached.wait()
        await registry.close()

    asyncio.run(exercise())
    last_record = json.loads((tmp_path / "subagent-runs.jsonl").read_text().splitlines()[-1])
    assert last_record["status"] == "killed"
    assert last_record["usage"]["input_tokens"] == 12
    assert last_record["usage"]["missing_cost_calls"] == 1


def test_cancelling_remote_write_cancels_the_pending_rpc():
    cancelled = []
    writes = []

    async def mkdir(path):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(path)
            raise

    interface = SimpleNamespace(
        create_dir=mkdir,
        write_bytes=AsyncMock(side_effect=lambda *args, **kwargs: writes.append(args)),
    )
    registry = FilesystemRegistry()
    registry.register(VMBackend(interface))
    tool = WriteFileTool(registry)
    agent = _fake_agent({"output": [], "usage": {}})
    agent._get_tool = MagicMock(return_value=tool)
    agent._on_function_call_start = AsyncMock()
    agent._on_function_call_end = AsyncMock()

    async def exercise():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                agent._dispatch_function_call(
                    {
                        "name": "write",
                        "call_id": "1",
                        "arguments": json.dumps({"path": "/workspace/file", "content": "x"}),
                    },
                    None,
                ),
                timeout=0.05,
            )

    asyncio.run(exercise())
    assert cancelled == ["/workspace"]
    assert writes == []


def test_gui_failure_termination_is_not_reported_as_success(tmp_path):
    transcript = _TranscriptWriter(tmp_path / "subagents" / "sub1" / "transcript.jsonl")

    class Agent:
        async def run(self, instruction, **kwargs):
            yield {
                "output": [
                    {
                        "type": "function_call",
                        "name": "computer",
                        "call_id": "c1",
                        "arguments": '{"action":"terminate","status":"failure"}',
                    }
                ]
            }
            yield {
                "output": [
                    {
                        "type": "function_call_output",
                        "call_id": "c1",
                        "output": '{"terminated":true,"status":"failure"}',
                    }
                ]
            }

    with pytest.raises(RuntimeError, match="terminated with failure"):
        asyncio.run(_relay_until_done(Agent(), "task", "sub1", 1, SubagentUsage(), transcript))


def test_gui_subagent_owns_desktop_until_completion(tmp_path):
    registry = SubagentRegistry()
    run = registry.register(type=SubagentType.GUI, task="task", model="test")
    delegate = DelegateGUITool(registry=registry, session=object(), parent_session_dir=tmp_path)
    assert delegate.call({"instruction": "another task"})["status"] == "rejected"

    agent = _fake_agent({"output": [], "usage": {}})
    agent._registry = registry
    agent._on_function_call_start = AsyncMock()
    computer = SimpleNamespace(click=AsyncMock())
    item = {
        "type": "function_call",
        "name": "computer",
        "call_id": "1",
        "arguments": '{"action":"click","x":1,"y":2}',
    }
    output = asyncio.run(OpenClawComputerAgent._handle_item(agent, item, computer))
    assert "GUI subagent" in str(output)
    computer.click.assert_not_awaited()
    registry.kill(run.run_id)


def test_model_usage_survives_tool_failure():
    result = {
        "output": [{"type": "function_call", "name": "write", "call_id": "a", "arguments": "{}"}],
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }
    agent = _fake_agent(result)
    agent._handle_item = AsyncMock(side_effect=RuntimeError("tool failed"))

    async def exercise():
        with pytest.raises(RuntimeError, match="tool failed"):
            async for _ in agent.run("task", max_turns=1):
                pass

    asyncio.run(exercise())
    agent.session_mgr.update_tokens.assert_called_once_with(12, 3)
    agent.session_mgr.update_step_count.assert_called_once_with(1)
    assert agent._on_run_end.await_args.args[0]["ale_claw_status"] == "failed"


def test_unreported_cost_remains_unknown_in_main_transcript():
    message = SimpleNamespace(content="hello", tool_calls=None, reasoning_content=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage={"prompt_tokens": 12, "completion_tokens": 3},
        _hidden_params={},
    )
    result = _convert_response_to_output(response)
    assert "response_cost" not in result["usage"]
    agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
    agent.session_mgr = MagicMock()
    agent.trajectory_dir = None
    agent.resolved_model = None
    agent._log_step_to_transcript(result)
    assert agent.session_mgr.append_message.call_args.kwargs["usage"]["cost"] is None

    response.usage["cost"] = 0.0
    assert _convert_response_to_output(response)["usage"]["response_cost"] == 0.0
