from __future__ import annotations

import base64
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

from ale_run.agents.grok_build.config import GrokBuildConfig
from ale_run.agents.grok_build.deployer import (
    GrokBuildDeployer,
    _expected_version,
    _grok_cwd,
)
from ale_run.base_interface import AgentRunResult, TrajectoryBuilder
from ale_run.orchestration.factory import AGENT_REGISTRY


def _executor(
    tmp_path: Path,
    config: GrokBuildConfig,
    *,
    is_linux: bool = True,
) -> SimpleNamespace:
    sandbox = SimpleNamespace(
        is_linux=is_linux,
        mcp_server_dir="/opt/cua_mcp_server",
    )
    return SimpleNamespace(
        config=config,
        env={config.api_key_env: "test-key"},
        work_dir=str(tmp_path),
        sandbox=sandbox,
        cua_bridge_url=lambda: "http://127.0.0.1:5000",
    )


def _builder() -> TrajectoryBuilder:
    return TrajectoryBuilder(
        agent_name="grok-build",
        model="grok-4.5",
        task_path="demo/seecheck",
        variant_index=0,
    )


def test_grok_build_is_registered() -> None:
    assert AGENT_REGISTRY["grok_build"] is GrokBuildDeployer


def test_expected_version_parses_scoped_npm_spec() -> None:
    assert _expected_version("@xai-official/grok@0.2.112") == "0.2.112"
    assert _expected_version("@xai-official/grok") is None


def test_headless_plan_mode_is_disabled_by_default() -> None:
    assert GrokBuildConfig().plan_mode is False


def test_default_model_matches_official_cli_catalog() -> None:
    assert GrokBuildConfig().model == "grok-4.5"


def test_primary_session_exports_are_hot_artifacts() -> None:
    assert {
        "session_chat_history.jsonl",
        "session_updates.jsonl",
        "session_events.jsonl",
        "session_summary.json",
        "session_media.jsonl",
    }.issubset(GrokBuildDeployer.hot_artifacts)


def test_build_env_isolates_home_and_keeps_custom_key_ephemeral(tmp_path: Path) -> None:
    config = GrokBuildConfig(
        model="kimi-k2.5",
        api_key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.ai/v1",
        api_backend="chat_completions",
    )
    deployer = GrokBuildDeployer(_executor(tmp_path, config))

    env = deployer._build_env(config, work_dir=tmp_path)

    assert env["GROK_HOME"] == str(tmp_path / "grok-home")
    assert env["GROK_LOG_FILE"] == os.devnull
    assert env["ALE_GROK_BUILD_API_KEY"] == "test-key"
    assert env["GROK_DISABLE_AUTOUPDATER"] == "1"
    assert env["GROK_SANDBOX"] == "off"
    assert env["GROK_CURSOR_MCPS_ENABLED"] == "0"
    assert env["GROK_CLAUDE_MCPS_ENABLED"] == "0"
    assert "grok.log" not in GrokBuildDeployer.hot_artifacts


def test_custom_model_key_does_not_replace_xai_imagine_key(tmp_path: Path) -> None:
    config = GrokBuildConfig(
        model="kimi-k2.5",
        api_key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.ai/v1",
        api_backend="chat_completions",
    )
    executor = _executor(tmp_path, config)
    executor.env["XAI_API_KEY"] = "xai-imagine-key"
    deployer = GrokBuildDeployer(executor)

    env = deployer._build_env(config, work_dir=tmp_path)

    assert env["ALE_GROK_BUILD_API_KEY"] == "test-key"
    assert env["XAI_API_KEY"] == "xai-imagine-key"


def test_build_env_maps_direct_key_to_xai(tmp_path: Path) -> None:
    config = GrokBuildConfig()
    deployer = GrokBuildDeployer(_executor(tmp_path, config))

    env = deployer._build_env(config, work_dir=tmp_path)

    assert env["XAI_API_KEY"] == "test-key"
    assert "ALE_GROK_BUILD_API_KEY" not in env


def test_write_config_registers_cua_and_custom_model_without_secret(
    tmp_path: Path,
) -> None:
    config = GrokBuildConfig(
        model="kimi-k2.5",
        api_key="must-not-be-written",
        base_url="https://api.moonshot.ai/v1",
        api_backend="chat_completions",
        context_window=262144,
        max_completion_tokens=8192,
    )
    executor = _executor(tmp_path, config)
    deployer = GrokBuildDeployer(executor)
    deployer._node_path = "/usr/bin/node"
    (tmp_path / "grok-home").mkdir()

    deployer._write_config(config, work_dir=tmp_path)

    rendered = (tmp_path / "grok-home" / "config.toml").read_text(encoding="utf-8")
    assert '[model."ale-custom"]' in rendered
    assert 'model = "kimi-k2.5"' in rendered
    assert 'api_backend = "chat_completions"' in rendered
    assert 'env_key = "ALE_GROK_BUILD_API_KEY"' in rendered
    assert "context_window = 262144" in rendered
    assert "[mcp_servers.cua]" in rendered
    assert 'args = ["/opt/cua_mcp_server/src/index.js"]' in rendered
    assert 'CUA_SERVER_URL = "http://127.0.0.1:5000"' in rendered
    assert "must-not-be-written" not in rendered


def test_build_argv_uses_supported_headless_flags(tmp_path: Path) -> None:
    config = GrokBuildConfig(
        model="kimi-k2.5",
        base_url="https://api.moonshot.ai/v1",
        reasoning_effort="high",
        max_turns=42,
        disable_web_search=True,
        plan_mode=False,
    )
    deployer = GrokBuildDeployer(_executor(tmp_path, config))
    deployer._grok_path = "/opt/grok"
    prompt_file = tmp_path / "prompt.txt"

    argv = deployer._build_argv(
        config,
        grok_cwd=tmp_path,
        prompt_file=prompt_file,
    )

    assert argv[:3] == ["/opt/grok", "--prompt-file", str(prompt_file)]
    assert argv[argv.index("--cwd") + 1] == str(tmp_path)
    assert argv[argv.index("--model") + 1] == "ale-custom"
    assert argv[argv.index("--output-format") + 1] == "streaming-json"
    assert "--always-approve" in argv
    assert "--no-auto-update" in argv
    assert "--disable-web-search" in argv
    assert "--no-plan" in argv
    assert argv[argv.index("--disallowed-tools") + 1] == (
        "ask_user_question,enter_plan_mode,exit_plan_mode"
    )
    assert argv[argv.index("--reasoning-effort") + 1] == "high"
    assert argv[argv.index("--max-turns") + 1] == "42"


def test_windows_grok_cwd_is_short_stable_and_linux_uses_work_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    work_dir = tmp_path / ("long-run-id-" * 12)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    windows_cwd = _grok_cwd(work_dir, is_linux=False)

    assert windows_cwd == _grok_cwd(work_dir, is_linux=False)
    assert windows_cwd.parent == Path.home() / ".ale-grok-build" / "cwd"
    assert len(windows_cwd.name) == 12
    assert len(str(windows_cwd)) < len(str(work_dir))
    assert _grok_cwd(work_dir, is_linux=True) == work_dir


def test_parse_session_preserves_tools_images_and_authoritative_usage(
    tmp_path: Path,
) -> None:
    session_id = "019f9bd7-3d6a-7132-9b65-af4af83b7606"
    session_dir = tmp_path / "grok-home" / "sessions" / "%2Fwork" / session_id
    session_dir.mkdir(parents=True)
    image_data = base64.b64encode(b"png-bytes").decode("ascii")
    transcript_events = [
        {"type": "thought", "data": "ignored because chat history is richer"},
        {
            "type": "end",
            "stopReason": "EndTurn",
            "sessionId": session_id,
            "requestId": "request-1",
            "usage": {
                "input_tokens": 795,
                "cache_read_input_tokens": 33792,
                "output_tokens": 175,
                "reasoning_tokens": 121,
                "total_tokens": 34762,
            },
            "num_turns": 3,
            "modelUsage": {
                "kimi-k2.5": {
                    "inputTokens": 795,
                    "outputTokens": 175,
                    "cacheReadInputTokens": 33792,
                    "modelCalls": 3,
                }
            },
            "total_cost_usd": 0.0123,
        },
    ]
    (tmp_path / "transcript.jsonl").write_text(
        "\n".join(json.dumps(event) for event in transcript_events),
        encoding="utf-8",
    )
    chat_events = [
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Inspect the desktop."}],
        },
        {
            "type": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "use_tool_1",
                    "name": "use_tool",
                    "arguments": json.dumps(
                        {
                            "tool_name": "cua__screenshot",
                            "tool_input": {},
                        }
                    ),
                }
            ],
        },
        {
            "type": "tool_result",
            "tool_call_id": "use_tool_1",
            "content": f"Screenshot captured\ndata:image/png;base64,{image_data}",
        },
        {
            "type": "assistant",
            "content": "The code is visible.",
            "model_id": "kimi-k2.5",
        },
    ]
    (session_dir / "chat_history.jsonl").write_text(
        "\n".join(json.dumps(event) for event in chat_events),
        encoding="utf-8",
    )
    update_events = [
        {
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "use_tool_1",
                    "status": "completed",
                },
            },
        }
    ]
    (session_dir / "updates.jsonl").write_text(
        "\n".join(json.dumps(event) for event in update_events),
        encoding="utf-8",
    )
    builder = _builder()

    GrokBuildDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=GrokBuildConfig(),
        run_result=AgentRunResult(status="completed", exit_code=0),
        builder=builder,
    )
    trajectory = builder.finalize(reward=1.0)

    agent_step, environment_step, final_step = trajectory.steps
    assert agent_step.reasoning == "Inspect the desktop."
    assert agent_step.tool_calls[0].name == "cua__screenshot"
    assert agent_step.tool_calls[0].arguments == {}
    result = environment_step.observation.results[0]
    assert result.content[0].text == "Screenshot captured\n"
    assert result.content[1].image.type == "base64"
    assert result.content[1].image.media_type == "image/png"
    assert result.content[1].image.data == image_data
    assert final_step.message == "The code is visible."
    assert trajectory.final_metrics.total_input_tokens == 795
    assert trajectory.final_metrics.total_cache_read_tokens == 33792
    assert trajectory.final_metrics.total_output_tokens == 175
    assert trajectory.final_metrics.total_cost_usd == 0.0123
    assert trajectory.extra["grok_build"]["session_id"] == session_id
    assert trajectory.extra["grok_build"]["num_turns"] == 3


def test_parse_stream_fallback_and_error(tmp_path: Path) -> None:
    transcript_events = [
        {"type": "thought", "data": "Checking"},
        {"type": "thought", "data": " files."},
        {"type": "text", "data": "Done"},
        {"type": "error", "message": "late warning"},
    ]
    (tmp_path / "transcript.jsonl").write_text(
        "\n".join(json.dumps(event) for event in transcript_events),
        encoding="utf-8",
    )
    builder = _builder()

    GrokBuildDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=GrokBuildConfig(),
        run_result=AgentRunResult(status="failed", exit_code=1),
        builder=builder,
    )

    assert builder.trajectory.steps[0].reasoning == "Checking files."
    assert builder.trajectory.steps[0].message == "Done"
    assert builder.trajectory.steps[1].message == "late warning"
    assert builder.trajectory.steps[1].extra["grok_build_error"] is True


def test_parse_session_recovers_image_from_mcp_spill(tmp_path: Path) -> None:
    session_id = "session-with-spill"
    session_dir = tmp_path / "grok-home" / "sessions" / "%2Fwork" / session_id
    (session_dir / "mcp").mkdir(parents=True)
    image_data = base64.b64encode(b"spilled-png").decode("ascii")
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps({"type": "end", "sessionId": session_id}),
        encoding="utf-8",
    )
    (session_dir / "chat_history.jsonl").write_text(
        json.dumps(
            {
                "type": "tool_result",
                "tool_call_id": "use_tool_7",
                "content": "[MCP output truncated; full output written separately]",
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "mcp" / "use_tool_7.txt").write_text(
        f"Screenshot captured\ndata:image/png;base64,{image_data}",
        encoding="utf-8",
    )
    builder = _builder()

    GrokBuildDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=GrokBuildConfig(),
        run_result=AgentRunResult(status="completed", exit_code=0),
        builder=builder,
    )

    content = builder.trajectory.steps[0].observation.results[0].content
    assert content[0].text == "[MCP output truncated; full output written separately]"
    assert content[1].image.data == image_data


def test_exported_session_recovers_trajectory_when_session_tree_is_missing(
    tmp_path: Path,
) -> None:
    session_id = "session-export"
    session_dir = tmp_path / "grok-home" / "sessions" / "%2Fwork" / session_id
    (session_dir / "mcp").mkdir(parents=True)
    image_data = base64.b64encode(b"exported-png").decode("ascii")
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps(
            {
                "type": "end",
                "sessionId": session_id,
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 20,
                    "output_tokens": 5,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (session_dir / "chat_history.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "tool_calls": [
                            {
                                "id": "use_tool_9",
                                "name": "use_tool",
                                "arguments": json.dumps(
                                    {
                                        "tool_name": "cua__screenshot",
                                        "tool_input": {},
                                    }
                                ),
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_result",
                        "tool_call_id": "use_tool_9",
                        "content": "[MCP output written separately]",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (session_dir / "updates.jsonl").write_text("", encoding="utf-8")
    (session_dir / "events.jsonl").write_text(
        json.dumps({"type": "turn_ended", "ts": "2026-07-26T01:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    (session_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (session_dir / "mcp" / "use_tool_9.txt").write_text(
        f"Screenshot captured\ndata:image/png;base64,{image_data}",
        encoding="utf-8",
    )

    GrokBuildDeployer._export_session_artifacts(
        tmp_path,
        tmp_path / "transcript.jsonl",
    )
    shutil.rmtree(tmp_path / "grok-home")

    builder = _builder()
    GrokBuildDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=GrokBuildConfig(),
        run_result=AgentRunResult(status="completed", exit_code=0),
        builder=builder,
    )

    assert (tmp_path / "session_chat_history.jsonl").is_file()
    assert (tmp_path / "session_events.jsonl").is_file()
    assert (tmp_path / "session_media.jsonl").is_file()
    assert (tmp_path / "telemetry_summary.json").is_file()
    assert builder.trajectory.steps[0].tool_calls[0].name == "cua__screenshot"
    image = builder.trajectory.steps[1].observation.results[0].content[1].image
    assert image.data == image_data
    assert builder.trajectory.extra["grok_build"]["session_dir"] is None
    assert builder.trajectory.extra["grok_build"]["events_path"].endswith("session_events.jsonl")


def test_parse_missing_transcript_emits_system_step(tmp_path: Path) -> None:
    builder = _builder()

    GrokBuildDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=GrokBuildConfig(),
        run_result=AgentRunResult(status="failed", exit_code=1),
        builder=builder,
    )

    assert builder.trajectory.steps[0].source == "system"
    assert builder.trajectory.steps[0].extra["reason"] == "no_transcript"
