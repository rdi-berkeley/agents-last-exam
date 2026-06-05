"""Tests for delegation tools (subagent_tools.py).

Covers:
  * DelegateGeneralTool: schema, accepted path (task scheduled + attached),
    rejected path (concurrency cap), constructor forwards parent_session_dir
    and memory_store.
  * DelegateGUITool: schema, complete path (synchronous summary return),
    error path (exception captured, registry already failed by run_gui_subagent).
  * SubagentsTool: list splits active/recent, kill cancels the asyncio.Task
    and transitions registry state, kill on unknown/terminal is safe.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from ale_run.agents.ale_claw.harness.subagent.subagent_registry import (
    SubagentLimitError,
    SubagentRegistry,
    SubagentStatus,
    SubagentType,
    SubagentUsage,
)
from ale_run.agents.ale_claw.harness.subagent.subagent_tools import (
    DELEGATE_GENERAL_DEFAULT_MAX_STEPS,
    DelegateGeneralTool,
    DelegateGUITool,
    SubagentsTool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


async def _await_gui_task(registry: SubagentRegistry, run_id: str) -> None:
    """Wait for the fire-and-forget _drive() task to finish."""
    task = registry._tasks.get(run_id)
    if task is not None:
        try:
            await task
        except Exception:
            pass


def _make_general_tool(
    registry: SubagentRegistry | None = None,
    memory_store=None,
    parent_session_dir: Path | None = None,
    default_model: str = "anthropic/claude-sonnet-4-20250514",
    summary_model: str = "anthropic/claude-haiku-4-5-20251001",
    thinking_params: dict | None = None,
    auxiliary_model: str | None = None,
) -> DelegateGeneralTool:
    return DelegateGeneralTool(
        registry=registry or SubagentRegistry(),
        tools=[],
        memory_store=memory_store or MagicMock(),
        default_model=default_model,
        summary_model=summary_model,
        parent_session_dir=parent_session_dir or Path("/tmp/fake-session"),
        thinking_params=thinking_params,
        auxiliary_model=auxiliary_model,
    )


def _make_gui_tool(
    registry: SubagentRegistry | None = None,
    session=None,
    parent_session_dir: Path | None = None,
    thinking_params: dict | None = None,
) -> DelegateGUITool:
    return DelegateGUITool(
        registry=registry or SubagentRegistry(),
        session=session if session is not None else MagicMock(),
        parent_session_dir=parent_session_dir or Path("/tmp/fake-session"),
        thinking_params=thinking_params,
    )


# ---------------------------------------------------------------------------
# DelegateGeneralTool
# ---------------------------------------------------------------------------


class TestDelegateGeneralTool:
    def test_name_is_registered(self):
        tool = _make_general_tool()
        assert tool.name == "delegate_general"

    def test_parameter_schema_requires_task(self):
        tool = _make_general_tool()
        schema = tool.parameters
        assert schema["type"] == "object"
        assert "task" in schema["properties"]
        assert schema["required"] == ["task"]
        assert "max_steps" in schema["properties"]
        assert "model" in schema["properties"]
        assert "label" in schema["properties"]

    def test_default_max_steps_is_50(self):
        assert DELEGATE_GENERAL_DEFAULT_MAX_STEPS == 50

    def test_accepted_path_schedules_and_attaches(self, tmp_path):
        registry = SubagentRegistry()

        async def _scenario():
            spawned: dict[str, object] = {}

            async def _fake_run_general(**kwargs):
                spawned["kwargs"] = kwargs
                await asyncio.sleep(0)

            tool = _make_general_tool(
                registry=registry,
                parent_session_dir=tmp_path,
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_general_subagent",
                side_effect=_fake_run_general,
            ):
                result = tool.call({"task": "plan next move", "label": "plan"})

            assert result["status"] == "accepted"
            run_id = result["run_id"]
            assert run_id.startswith("sub-")
            assert result["note"] == (
                "persistent session — result auto-announces when complete; do not poll"
            )

            # Task was created and attached to the registry.
            assert run_id in registry._tasks
            task = registry._tasks[run_id]
            assert isinstance(task, asyncio.Task)
            # Let the task run so fake_run_general captures kwargs.
            await task

            kwargs = spawned["kwargs"]
            assert kwargs["task"] == "plan next move"
            assert kwargs["parent_session_dir"] == tmp_path
            assert kwargs["summary_model"] == "anthropic/claude-haiku-4-5-20251001"
            assert kwargs["max_steps"] == 50
            assert kwargs["run_id"] == run_id

        _run(_scenario())

    def test_accepted_custom_max_steps_and_auxiliary_model(self, tmp_path):
        registry = SubagentRegistry()

        async def _scenario():
            spawned: dict[str, object] = {}

            async def _fake_run_general(**kwargs):
                spawned["kwargs"] = kwargs
                await asyncio.sleep(0)

            tool = _make_general_tool(
                registry=registry,
                parent_session_dir=tmp_path,
                default_model="openrouter/openai/gpt-5.4",
                auxiliary_model="openrouter/openai/gpt-5.4-mini",
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_general_subagent",
                side_effect=_fake_run_general,
            ):
                result = tool.call({
                    "task": "analyze",
                    "max_steps": 12,
                    "model": "openrouter/openai/gpt-5.4-mini",
                })
            assert result["status"] == "accepted"
            assert "model_warning" not in result
            task = registry._tasks[result["run_id"]]
            await task
            assert spawned["kwargs"]["max_steps"] == 12
            assert spawned["kwargs"]["model"] == "openrouter/openai/gpt-5.4-mini"

        _run(_scenario())

    def test_disallowed_model_falls_back_with_warning(self, tmp_path):
        """Hallucinated sibling IDs (e.g. gpt-5.1-mini) must fall back to default."""
        registry = SubagentRegistry()

        async def _scenario():
            spawned: dict[str, object] = {}

            async def _fake_run_general(**kwargs):
                spawned["kwargs"] = kwargs
                await asyncio.sleep(0)

            tool = _make_general_tool(
                registry=registry,
                parent_session_dir=tmp_path,
                default_model="openrouter/openai/gpt-5.4",
                auxiliary_model="openrouter/openai/gpt-5.4-mini",
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_general_subagent",
                side_effect=_fake_run_general,
            ):
                result = tool.call({
                    "task": "analyze",
                    "model": "openrouter/openai/gpt-5.1-mini",
                })
            assert result["status"] == "accepted"
            assert "model_warning" in result
            assert "gpt-5.1-mini" in result["model_warning"]
            task = registry._tasks[result["run_id"]]
            await task
            assert spawned["kwargs"]["model"] == "openrouter/openai/gpt-5.4"

        _run(_scenario())

    def test_model_param_schema_uses_enum_with_two_options(self, tmp_path):
        tool = _make_general_tool(
            parent_session_dir=tmp_path,
            default_model="openrouter/openai/gpt-5.4",
            auxiliary_model="openrouter/openai/gpt-5.4-mini",
        )
        model_schema = tool.parameters["properties"]["model"]
        assert model_schema["type"] == "string"
        assert model_schema["enum"] == [
            "openrouter/openai/gpt-5.4",
            "openrouter/openai/gpt-5.4-mini",
        ]
        assert "gpt-5.4-mini" in model_schema["description"]

    def test_model_param_schema_single_option_when_no_auxiliary_model(self, tmp_path):
        tool = _make_general_tool(
            parent_session_dir=tmp_path,
            default_model="openrouter/openai/gpt-5.4",
            auxiliary_model=None,
        )
        model_schema = tool.parameters["properties"]["model"]
        assert model_schema["enum"] == ["openrouter/openai/gpt-5.4"]

    def test_rejected_when_limit_reached(self, tmp_path):
        registry = SubagentRegistry(max_concurrent=1)

        async def _scenario():
            # Occupy the one slot.
            registry.register(type=SubagentType.GENERAL, task="existing")
            tool = _make_general_tool(
                registry=registry, parent_session_dir=tmp_path
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_general_subagent",
                side_effect=SubagentLimitError("unreachable"),
            ):
                result = tool.call({"task": "queued"})
            assert result == {
                "status": "rejected",
                "reason": "max concurrent subagents reached",
            }

        _run(_scenario())

    def test_constructor_stores_parent_session_dir_and_memory_store(
        self, tmp_path
    ):
        memory_store = MagicMock()
        tool = _make_general_tool(
            memory_store=memory_store, parent_session_dir=tmp_path
        )
        assert tool._parent_session_dir == tmp_path
        assert tool._memory_store is memory_store

    def test_empty_task_returns_error(self):
        tool = _make_general_tool()
        result = tool.call({"task": "   "})
        assert result["status"] == "error"

    def test_screenshot_paths_in_parameter_schema(self):
        tool = _make_general_tool()
        schema = tool.parameters
        assert "screenshot_paths" in schema["properties"]
        assert schema["properties"]["screenshot_paths"]["type"] == "array"
        assert schema["properties"]["screenshot_paths"]["items"]["type"] == "string"

    def test_screenshot_paths_forwarded_to_run_general(self, tmp_path):
        registry = SubagentRegistry()

        async def _scenario():
            spawned: dict[str, object] = {}

            async def _fake_run_general(**kwargs):
                spawned["kwargs"] = kwargs
                await asyncio.sleep(0)

            tool = _make_general_tool(
                registry=registry, parent_session_dir=tmp_path
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_general_subagent",
                side_effect=_fake_run_general,
            ):
                result = tool.call({
                    "task": "analyze",
                    "screenshot_paths": ["/tmp/a.png", "/tmp/b.png"],
                })

            assert result["status"] == "accepted"
            await registry._tasks[result["run_id"]]
            assert spawned["kwargs"]["initial_screenshot_paths"] == [
                "/tmp/a.png",
                "/tmp/b.png",
            ]

        _run(_scenario())

    def test_screenshot_paths_filters_non_strings(self, tmp_path):
        registry = SubagentRegistry()

        async def _scenario():
            spawned: dict[str, object] = {}

            async def _fake_run_general(**kwargs):
                spawned["kwargs"] = kwargs
                await asyncio.sleep(0)

            tool = _make_general_tool(
                registry=registry, parent_session_dir=tmp_path
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_general_subagent",
                side_effect=_fake_run_general,
            ):
                result = tool.call({
                    "task": "analyze",
                    "screenshot_paths": ["/tmp/a.png", "", 42, None, "/tmp/b.png"],
                })
            await registry._tasks[result["run_id"]]
            assert spawned["kwargs"]["initial_screenshot_paths"] == [
                "/tmp/a.png",
                "/tmp/b.png",
            ]

        _run(_scenario())

    def test_screenshot_paths_absent_forwards_none(self, tmp_path):
        registry = SubagentRegistry()

        async def _scenario():
            spawned: dict[str, object] = {}

            async def _fake_run_general(**kwargs):
                spawned["kwargs"] = kwargs
                await asyncio.sleep(0)

            tool = _make_general_tool(
                registry=registry, parent_session_dir=tmp_path
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_general_subagent",
                side_effect=_fake_run_general,
            ):
                result = tool.call({"task": "analyze"})
            await registry._tasks[result["run_id"]]
            paths = spawned["kwargs"]["initial_screenshot_paths"]
            # Helper accepts either None or an empty list as "no screenshots".
            assert not paths


# ---------------------------------------------------------------------------
# DelegateGUITool
# ---------------------------------------------------------------------------


class TestDelegateGUITool:
    def test_name_is_registered(self):
        tool = _make_gui_tool()
        assert tool.name == "delegate_gui"

    def test_parameter_schema_requires_instruction(self):
        tool = _make_gui_tool()
        schema = tool.parameters
        assert schema["required"] == ["instruction"]
        assert "max_steps" in schema["properties"]
        assert "model" in schema["properties"]

    def test_accepted_then_drain_yields_result(self, tmp_path):
        registry = SubagentRegistry()
        session = MagicMock()
        session.screenshot = AsyncMock(return_value=b"\x89PNG\r\nfake")

        async def _fake_relay(**kwargs):
            registry._runs[kwargs["run_id"]].usage = SubagentUsage(
                input_tokens=111, output_tokens=22
            )
            registry.complete(kwargs["run_id"], "opened notepad")
            return "opened notepad"

        async def _scenario():
            tool = _make_gui_tool(
                registry=registry, session=session, parent_session_dir=tmp_path
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_gui_subagent",
                side_effect=_fake_relay,
            ):
                result = tool.call({"instruction": "open notepad"})
                assert result["status"] == "accepted"
                assert result["run_id"].startswith("sub-")
                await _await_gui_task(registry, result["run_id"])
            # Drain the completion queue.
            completions = registry.drain_completions()
            assert len(completions) == 1
            assert completions[0].result_text == "opened notepad"
            assert completions[0].usage.input_tokens == 111

        _run(_scenario())

    def test_error_arrives_via_drain(self, tmp_path):
        registry = SubagentRegistry()
        session = MagicMock()

        async def _fake_relay(**kwargs):
            registry.fail(kwargs["run_id"], "click failed")
            raise RuntimeError("click failed")

        async def _scenario():
            tool = _make_gui_tool(
                registry=registry, session=session, parent_session_dir=tmp_path
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_gui_subagent",
                side_effect=_fake_relay,
            ):
                result = tool.call({"instruction": "break things"})
                assert result["status"] == "accepted"
                await _await_gui_task(registry, result["run_id"])
            completions = registry.drain_completions()
            assert len(completions) == 1
            assert completions[0].status == SubagentStatus.ERROR
            assert "click failed" in (completions[0].error_message or "")

        _run(_scenario())

    def test_empty_instruction_returns_error(self):
        tool = _make_gui_tool()
        assert tool.call({"instruction": ""})["status"] == "error"

    def test_constructor_stores_parent_session_dir(self, tmp_path):
        tool = _make_gui_tool(parent_session_dir=tmp_path)
        assert tool._parent_session_dir == tmp_path

    # ---- Post-delegation screenshot injection ----

    def test_post_delegation_message_enqueued_on_complete(self, tmp_path):
        registry = SubagentRegistry()
        session = MagicMock()
        session.screenshot = AsyncMock(return_value=b"\x89PNG\r\nfake-post-shot")

        async def _fake_relay(**kwargs):
            registry.complete(kwargs["run_id"], "done")
            return "done"

        async def _scenario():
            tool = _make_gui_tool(
                registry=registry, session=session, parent_session_dir=tmp_path
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_gui_subagent",
                side_effect=_fake_relay,
            ):
                result = tool.call({"instruction": "open notepad"})
                assert result["status"] == "accepted"
                run_id = result["run_id"]
                await _await_gui_task(registry, run_id)

            # Completion queue has the result.
            completions = registry.drain_completions()
            assert len(completions) == 1

            # Post-delegation queue has exactly one user message.
            msgs = registry.drain_post_delegation()
            assert len(msgs) == 1
            msg = msgs[0]
            assert msg["role"] == "user"
            assert isinstance(msg["content"], list)
            assert msg["content"][0] == {
                "type": "text",
                "text": "[VM state after GUI delegation]",
            }
            img_block = msg["content"][1]
            assert img_block["type"] == "image_url"
            assert img_block["image_url"]["url"].startswith("data:image/png;base64,")

            # PNG persisted under <parent>/subagents/<run_id>/post_delegation.png
            saved = tmp_path / "subagents" / run_id / "post_delegation.png"
            assert saved.exists()
            assert saved.read_bytes() == b"\x89PNG\r\nfake-post-shot"

        _run(_scenario())

    def test_post_delegation_skipped_on_relay_error(self, tmp_path):
        registry = SubagentRegistry()
        session = MagicMock()
        session.screenshot = AsyncMock(return_value=b"\x89PNG\r\nshould-not-be-used")

        async def _fake_relay(**kwargs):
            registry.fail(kwargs["run_id"], "click failed")
            raise RuntimeError("click failed")

        async def _scenario():
            tool = _make_gui_tool(
                registry=registry, session=session, parent_session_dir=tmp_path
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_gui_subagent",
                side_effect=_fake_relay,
            ):
                result = tool.call({"instruction": "bad action"})
                assert result["status"] == "accepted"
                await _await_gui_task(registry, result["run_id"])
            # Queue stays empty on error — VM state is unreliable.
            assert registry.drain_post_delegation() == []

        _run(_scenario())

    def test_post_delegation_skipped_when_screenshot_fails(self, tmp_path):
        registry = SubagentRegistry()
        session = MagicMock()
        session.screenshot = AsyncMock(side_effect=RuntimeError("no display"))

        async def _fake_relay(**kwargs):
            registry.complete(kwargs["run_id"], "done")
            return "done"

        async def _scenario():
            tool = _make_gui_tool(
                registry=registry, session=session, parent_session_dir=tmp_path
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_gui_subagent",
                side_effect=_fake_relay,
            ):
                result = tool.call({"instruction": "open notepad"})
                assert result["status"] == "accepted"
                await _await_gui_task(registry, result["run_id"])
            # Screenshot failure is non-fatal — completion still drained.
            completions = registry.drain_completions()
            assert len(completions) == 1
            assert registry.drain_post_delegation() == []

        _run(_scenario())

    def test_post_delegation_skipped_when_screenshot_empty(self, tmp_path):
        registry = SubagentRegistry()
        session = MagicMock()
        session.screenshot = AsyncMock(return_value=b"")

        async def _fake_relay(**kwargs):
            registry.complete(kwargs["run_id"], "done")
            return "done"

        async def _scenario():
            tool = _make_gui_tool(
                registry=registry, session=session, parent_session_dir=tmp_path
            )
            with patch(
                "ale_run.agents.ale_claw.harness.subagent.subagent_tools.run_gui_subagent",
                side_effect=_fake_relay,
            ):
                result = tool.call({"instruction": "open notepad"})
                assert result["status"] == "accepted"
                await _await_gui_task(registry, result["run_id"])
            assert len(registry.drain_completions()) == 1
            assert registry.drain_post_delegation() == []

        _run(_scenario())


# ---------------------------------------------------------------------------
# SubagentsTool
# ---------------------------------------------------------------------------


class TestSubagentsTool:
    def test_name_is_registered(self):
        tool = SubagentsTool(registry=SubagentRegistry())
        assert tool.name == "subagents"

    def test_list_splits_active_vs_recent(self):
        reg = SubagentRegistry()
        active_run = reg.register(type=SubagentType.GENERAL, task="active", label="a")
        reg.mark_running(active_run.run_id)

        done_run = reg.register(type=SubagentType.GENERAL, task="done", label="d")
        reg.complete(done_run.run_id, "ok")

        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "list"})
        assert result["status"] == "ok"
        assert {r["run_id"] for r in result["active"]} == {active_run.run_id}
        assert {r["run_id"] for r in result["recent"]} == {done_run.run_id}

    def test_default_action_is_list(self):
        tool = SubagentsTool(registry=SubagentRegistry())
        result = tool.call({})
        assert result["status"] == "ok"
        assert result["active"] == []
        assert result["recent"] == []

    def test_kill_cancels_task_and_transitions_state(self, tmp_path):
        async def _scenario():
            reg = SubagentRegistry()
            run = reg.register(type=SubagentType.GENERAL, task="runaway")

            async def _long_lived():
                await asyncio.sleep(10.0)

            task = asyncio.get_event_loop().create_task(_long_lived())
            reg.attach_task(run.run_id, task)

            tool = SubagentsTool(registry=reg)
            result = tool.call({"action": "kill", "target": run.run_id})
            assert result == {"status": "ok", "killed": run.run_id}
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert task.cancelled()
            assert reg.get_run(run.run_id).status == SubagentStatus.KILLED

        _run(_scenario())

    def test_kill_requires_target(self):
        tool = SubagentsTool(registry=SubagentRegistry())
        result = tool.call({"action": "kill"})
        assert result["status"] == "error"
        assert "target" in result["reason"]

    def test_kill_unknown_run(self):
        tool = SubagentsTool(registry=SubagentRegistry())
        result = tool.call({"action": "kill", "target": "sub-missing"})
        assert result == {"status": "error", "reason": "unknown run_id"}

    def test_kill_terminal_noop(self):
        reg = SubagentRegistry()
        run = reg.register(type=SubagentType.GENERAL, task="done")
        reg.complete(run.run_id, "ok")
        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "kill", "target": run.run_id})
        assert result == {"status": "noop", "reason": "already terminal"}
        assert reg.get_run(run.run_id).status == SubagentStatus.COMPLETE

    def test_unknown_action_returns_error(self):
        tool = SubagentsTool(registry=SubagentRegistry())
        result = tool.call({"action": "bogus"})
        assert result["status"] == "error"
        assert "bogus" in result["reason"]


# ---------------------------------------------------------------------------
# SubagentsTool — steer action
# ---------------------------------------------------------------------------


class TestSubagentsToolSteer:
    def _make_running(self, reg: SubagentRegistry, label: str = "") -> str:
        """Register + mark_running a general subagent, attach an inbox."""
        import asyncio as _aio
        run = reg.register(type=SubagentType.GENERAL, task="work", label=label)
        reg.mark_running(run.run_id)
        inbox: _aio.Queue[str] = _aio.Queue()
        reg.attach_inbox(run.run_id, inbox)
        return run.run_id

    def test_steer_by_exact_run_id(self):
        reg = SubagentRegistry()
        run_id = self._make_running(reg)
        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "steer", "target": run_id, "message": "change plan"})
        assert result == {"status": "ok", "steered": run_id}
        inbox = reg.get_inbox(run_id)
        assert inbox is not None
        assert inbox.get_nowait() == "change plan"

    def test_steer_by_label(self):
        reg = SubagentRegistry()
        run_id = self._make_running(reg, label="planner")
        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "steer", "target": "planner", "message": "hurry up"})
        assert result["status"] == "ok"
        assert result["steered"] == run_id

    def test_steer_by_label_case_insensitive(self):
        reg = SubagentRegistry()
        run_id = self._make_running(reg, label="Planner")
        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "steer", "target": "planner", "message": "msg"})
        assert result["status"] == "ok"
        assert result["steered"] == run_id

    def test_steer_by_run_id_prefix(self):
        reg = SubagentRegistry()
        run_id = self._make_running(reg)
        prefix = run_id[:8]
        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "steer", "target": prefix, "message": "msg"})
        assert result["status"] == "ok"
        assert result["steered"] == run_id

    def test_steer_last_target(self):
        reg = SubagentRegistry()
        _first = self._make_running(reg, label="first")
        second = self._make_running(reg, label="second")
        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "steer", "target": "last", "message": "msg"})
        assert result["status"] == "ok"
        assert result["steered"] == second

    def test_steer_missing_target(self):
        tool = SubagentsTool(registry=SubagentRegistry())
        result = tool.call({"action": "steer", "message": "hi"})
        assert result["status"] == "error"
        assert "target" in result["reason"]

    def test_steer_missing_message(self):
        reg = SubagentRegistry()
        run_id = self._make_running(reg)
        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "steer", "target": run_id})
        assert result["status"] == "error"
        assert "message" in result["reason"]

    def test_steer_empty_message(self):
        reg = SubagentRegistry()
        run_id = self._make_running(reg)
        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "steer", "target": run_id, "message": "   "})
        assert result["status"] == "error"
        assert "message" in result["reason"]

    def test_steer_message_too_long(self):
        from ale_run.agents.ale_claw.harness.subagent.subagent_tools import MAX_STEER_MESSAGE_CHARS
        reg = SubagentRegistry()
        run_id = self._make_running(reg)
        tool = SubagentsTool(registry=reg)
        long_msg = "x" * (MAX_STEER_MESSAGE_CHARS + 1)
        result = tool.call({"action": "steer", "target": run_id, "message": long_msg})
        assert result["status"] == "error"
        assert "too long" in result["reason"]

    def test_steer_unknown_target(self):
        tool = SubagentsTool(registry=SubagentRegistry())
        result = tool.call({"action": "steer", "target": "sub-nope", "message": "hi"})
        assert result["status"] == "error"
        assert "unknown target" in result["reason"]

    def test_steer_terminal_run(self):
        reg = SubagentRegistry()
        run = reg.register(type=SubagentType.GENERAL, task="done")
        reg.complete(run.run_id, "finished")
        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "steer", "target": run.run_id, "message": "hi"})
        assert result["status"] == "error"

    def test_steer_no_inbox(self):
        """Run is active but inbox was never attached (shouldn't happen in
        practice, but the code path should be safe)."""
        reg = SubagentRegistry()
        run = reg.register(type=SubagentType.GENERAL, task="work")
        reg.mark_running(run.run_id)
        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "steer", "target": run.run_id, "message": "hi"})
        assert result["status"] == "error"
        assert "no inbox" in result["reason"]

    def test_steer_schema_includes_steer_enum(self):
        tool = SubagentsTool(registry=SubagentRegistry())
        schema = tool.parameters
        actions = schema["properties"]["action"]["enum"]
        assert "steer" in actions
        assert "message" in schema["properties"]

    def test_steer_gui_subagent_by_run_id(self):
        """GUI subagents are steerable — message lands in inbox."""
        import asyncio as _aio
        reg = SubagentRegistry()
        run = reg.register(type=SubagentType.GUI, task="click button")
        reg.mark_running(run.run_id)
        inbox: _aio.Queue[str] = _aio.Queue()
        reg.attach_inbox(run.run_id, inbox)
        tool = SubagentsTool(registry=reg)
        result = tool.call({
            "action": "steer",
            "target": run.run_id,
            "message": "click the other button instead",
        })
        assert result == {"status": "ok", "steered": run.run_id}
        assert inbox.get_nowait() == "click the other button instead"

    def test_steer_last_resolves_gui_subagent(self):
        """'last' resolves to the most recent active run regardless of type."""
        import asyncio as _aio
        reg = SubagentRegistry()
        _gen = reg.register(type=SubagentType.GENERAL, task="plan")
        reg.mark_running(_gen.run_id)
        reg.attach_inbox(_gen.run_id, _aio.Queue())
        gui = reg.register(type=SubagentType.GUI, task="click")
        reg.mark_running(gui.run_id)
        inbox: _aio.Queue[str] = _aio.Queue()
        reg.attach_inbox(gui.run_id, inbox)
        tool = SubagentsTool(registry=reg)
        result = tool.call({"action": "steer", "target": "last", "message": "msg"})
        assert result["status"] == "ok"
        assert result["steered"] == gui.run_id
