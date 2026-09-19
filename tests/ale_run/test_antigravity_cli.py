"""Focused regression tests for the Antigravity CLI integration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.agents.antigravity_cli.config import AntigravityCliConfig
from ale_run.agents.antigravity_cli.deployer import (
    AntigravityCliDeployer,
    _find_agy,
    _install_release,
    _official_release,
)
from ale_run.base_interface import AgentRunResult, TrajectoryBuilder
from ale_run.orchestration.experiment_spec import AgentSpec, RunUnit
from ale_run.orchestration.lifecycle import _build_run_meta, _trajectory_status


def _result() -> AgentRunResult:
    return AgentRunResult(
        status="completed",
        exit_code=0,
        transcript_path="transcript.jsonl",
        duration_s=1.25,
    )


def test_parse_json_envelope_records_response_and_usage(tmp_path: Path) -> None:
    envelope = {
        "conversation_id": "conv-123",
        "status": "SUCCESS",
        "response": "finished",
        "error": "",
        "duration_seconds": 1.25,
        "num_turns": 3,
        "usage": {
            "input_tokens": 120,
            "output_tokens": 30,
            "thinking_tokens": 7,
            "cache_read_tokens": 80,
            "total_tokens": 157,
        },
    }
    (tmp_path / "transcript.jsonl").write_text(json.dumps(envelope), encoding="utf-8")
    builder = TrajectoryBuilder(
        agent_name="antigravity-cli", task_path="demo/seecheck", variant_index=0
    )

    AntigravityCliDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=AntigravityCliConfig(),
        run_result=_result(),
        builder=builder,
    )

    step = builder.trajectory.steps[0]
    assert step.source == "agent"
    assert step.message == "finished"
    assert step.metrics is not None
    assert step.metrics.input_tokens == 120
    assert step.metrics.output_tokens == 30
    assert step.metrics.cache_read_tokens == 80
    assert step.metrics.duration_ms == 1250
    assert step.extra["thinking_tokens"] == 7
    extra = builder.trajectory.extra["antigravity_cli"]
    assert extra["conversation_id"] == "conv-123"
    assert extra["num_turns"] == 3

    metrics = builder.finalize(reward=1.0, status="completed").final_metrics
    assert metrics.total_input_tokens == 120
    assert metrics.total_output_tokens == 30
    assert metrics.total_cache_read_tokens == 80


def test_parse_json_error_envelope_adds_system_step(tmp_path: Path) -> None:
    envelope = {
        "status": "ERROR",
        "response": "",
        "error": "authentication failed",
        "usage": {},
    }
    (tmp_path / "transcript.jsonl").write_text(json.dumps(envelope), encoding="utf-8")
    builder = TrajectoryBuilder(
        agent_name="antigravity-cli", task_path="demo/seecheck", variant_index=0
    )

    AntigravityCliDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=AntigravityCliConfig(),
        run_result=AgentRunResult(status="failed", exit_code=1),
        builder=builder,
    )

    assert builder.trajectory.steps[0].source == "system"
    assert builder.trajectory.steps[0].message == "authentication failed"


def test_missing_transcript_still_builds_metrics_summary(tmp_path: Path) -> None:
    builder = TrajectoryBuilder(
        agent_name="antigravity-cli", task_path="demo/seecheck", variant_index=0
    )

    AntigravityCliDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=AntigravityCliConfig(),
        run_result=AgentRunResult(status="failed", exit_code=1),
        builder=builder,
    )

    summary = json.loads((tmp_path / "metrics_summary.json").read_text())
    assert summary["native_stream_present"] is False
    assert summary["native_log_present"] is False
    assert summary["native_log_bytes"] == 0
    assert summary["availability"]["model_generations_observed"] is False
    assert summary["availability"]["per_transport_attempt"]["observed"] is False
    assert summary["availability"]["tool_calls"] is False
    assert summary["availability"]["tool_latency"] is False
    assert summary["availability"]["multimodal_output_groups"] is False
    assert not (tmp_path / "telemetry.jsonl").exists()
    assert not (tmp_path / "otel_requests.jsonl").exists()


def test_parse_plain_text_fallback(tmp_path: Path) -> None:
    (tmp_path / "transcript.jsonl").write_text("first\n\nsecond\n", encoding="utf-8")
    builder = TrajectoryBuilder(
        agent_name="antigravity-cli", task_path="demo/seecheck", variant_index=0
    )

    AntigravityCliDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=AntigravityCliConfig(),
        run_result=_result(),
        builder=builder,
    )

    assert builder.trajectory.steps[0].message == "first\nsecond"


def test_parse_stream_json_builds_metrics_summary(tmp_path: Path) -> None:
    events = [
        {
            "event": "init",
            "init": {"model": "Gemini 3.7 Flash (High)"},
        },
        {
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv-stream",
                "step_index": 1,
                "state": "DONE",
                "step_type": "agent_response",
                "duration_seconds": 2.5,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "thinking_tokens": 8,
                    "cache_read_tokens": 60,
                    "total_tokens": 120,
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv-stream",
                "step_index": 2,
                "state": "DONE",
                "step_type": "tool",
                "tool_name": "call_mcp_tool__cua__screenshot",
                "duration_seconds": 0.25,
                "tool_info": {
                    "name": "call_mcp_tool__cua__screenshot",
                    "parameters": {},
                    "output": "captured",
                },
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv-stream",
                "step_index": 3,
                "state": "ACTIVE",
                "step_type": "agent_response",
                "text_delta": "all ",
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "conversation_id": "conv-stream",
                "step_index": 3,
                "state": "DONE",
                "step_type": "agent_response",
                "text_delta": "finished",
                "duration_seconds": 1.0,
                "usage": {
                    "input_tokens": 70,
                    "output_tokens": 10,
                    "thinking_tokens": 4,
                    "cache_read_tokens": 50,
                    "total_tokens": 80,
                },
            },
        },
        {
            "event": "result",
            "result": {
                "conversation_id": "conv-stream",
                "status": "SUCCESS",
                "response": "finished",
                "duration_seconds": 3.75,
                "num_turns": 1,
                "usage": {
                    "input_tokens": 170,
                    "output_tokens": 30,
                    "thinking_tokens": 12,
                    "cache_read_tokens": 110,
                    "total_tokens": 200,
                },
            },
        },
    ]
    (tmp_path / "transcript.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (tmp_path / "agy_cli.log").write_text(
        "ERROR: logging before google.Init: I0828 08:33:41.418031 462 "
        "http_helpers.go:246] URL: https://example/v1internal:streamGenerateContent?alt=sse "
        "Trace: 0x123 ResponseID: response-1\n",
        encoding="utf-8",
    )
    builder = TrajectoryBuilder(
        agent_name="antigravity-cli", task_path="demo/seecheck", variant_index=0
    )

    AntigravityCliDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=AntigravityCliConfig(),
        run_result=_result(),
        builder=builder,
    )

    summary = json.loads((tmp_path / "metrics_summary.json").read_text())
    assert summary["native_log_present"] is True
    assert summary["native_log_bytes"] > 0
    assert summary["api_call_count"] == 2
    assert summary["logical_model_generation_count"] == 2
    assert summary["api_calls"][0]["uncached_input_tokens"] == 100
    assert summary["api_calls"][0]["cached_input_tokens"] == 60
    assert summary["api_calls"][1]["output_tokens"] == 10
    assert summary["tool_call_count"] == 1
    assert summary["tool_latency_ms"] == 250
    assert summary["multimodal_output_group_count"] == 1
    assert summary["transport_attempts"][0]["response_id"] == "response-1"
    assert summary["availability"]["generation_usage_reconciles_with_result"] is True
    assert summary["availability"]["per_model_generation"]["cache_write_tokens"] is False
    assert summary["run"]["model"] == "Gemini 3.7 Flash (High)"
    assert summary["run"]["configured_model"] == "Gemini 3.7 Flash (High)"
    assert not (tmp_path / "telemetry.jsonl").exists()
    assert not (tmp_path / "otel_requests.jsonl").exists()

    metrics = builder.finalize(reward=1.0, status="completed").final_metrics
    assert metrics.total_input_tokens == 170
    assert metrics.total_output_tokens == 30
    assert metrics.total_cache_read_tokens == 110

    trajectory = builder.trajectory
    assert [step.source for step in trajectory.steps] == [
        "agent",
        "agent",
        "environment",
        "agent",
    ]
    tool_step = trajectory.steps[1]
    assert tool_step.tool_calls[0].id == "agy_step_2"
    assert tool_step.tool_calls[0].name == "call_mcp_tool__cua__screenshot"
    result_step = trajectory.steps[2]
    assert result_step.observation is not None
    assert result_step.observation.results[0].tool_call_id == "agy_step_2"
    assert result_step.observation.results[0].content[0].text == "captured"
    assert trajectory.steps[3].message == "all finished"


def test_incomplete_tool_is_retained_without_fabricated_result(tmp_path: Path) -> None:
    events = [
        {
            "event": "init",
            "init": {"model": "Observed Model"},
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 1,
                "state": "DONE",
                "step_type": "agent_response",
                "usage": {"input_tokens": 4, "output_tokens": 2, "cache_read_tokens": 1},
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 2,
                "state": "ACTIVE",
                "step_type": "tool",
                "tool_name": "run_command",
                "tool_info": {"parameters": {"CommandLine": "long-running-task"}},
            },
        },
        {
            "event": "result",
            "result": {
                "status": "ERROR",
                "error": "stream interrupted",
                "usage": {"input_tokens": 4, "output_tokens": 2, "cache_read_tokens": 1},
            },
        },
    ]
    (tmp_path / "transcript.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    builder = TrajectoryBuilder(
        agent_name="antigravity-cli", task_path="demo/interrupted", variant_index=0
    )

    AntigravityCliDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=AntigravityCliConfig(model="Configured Model"),
        run_result=AgentRunResult(status="failed", exit_code=1),
        builder=builder,
    )

    calls = [call for step in builder.trajectory.steps for call in step.tool_calls]
    results = [
        result
        for step in builder.trajectory.steps
        if step.observation is not None
        for result in step.observation.results
    ]
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert results == []
    tool_step = next(step for step in builder.trajectory.steps if step.tool_calls)
    assert tool_step.extra["incomplete"] is True

    summary = json.loads((tmp_path / "metrics_summary.json").read_text())
    assert summary["run"]["model"] == "Observed Model"
    assert summary["run"]["configured_model"] == "Configured Model"
    assert summary["tool_call_count"] == 1
    assert summary["tool_executions"][0]["state"] == "ACTIVE"
    assert summary["tool_executions"][0]["success"] is None
    assert summary["availability"]["tool_latency"] is False


def test_find_agy_prefers_managed_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    managed = tmp_path / ".local" / "bin" / "agy"
    managed.parent.mkdir(parents=True)
    managed.write_text("managed", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/agy")

    assert _find_agy(str(tmp_path), is_windows=False) == str(managed)


def test_custom_download_requires_checksum(tmp_path: Path) -> None:
    config = AntigravityCliConfig(download_url="https://example.invalid/agy.tar.gz")

    with pytest.raises(RuntimeError, match="download_sha512"):
        _install_release(config, str(tmp_path), is_windows=False)


def test_pinned_release_does_not_depend_on_latest_manifest() -> None:
    linux_url, linux_checksum = _official_release("1.1.25", is_windows=False)
    windows_url, windows_checksum = _official_release("1.1.25", is_windows=True)

    assert "/1.1.25-" in linux_url
    assert linux_url.endswith("/cli_linux_x64.tar.gz")
    assert len(linux_checksum) == 128
    assert "/1.1.25-" in windows_url
    assert windows_url.endswith("/cli_windows_x64.exe")
    assert len(windows_checksum) == 128


@pytest.mark.parametrize(
    ("agent_status", "eval_status", "expected"),
    [
        ("completed", "success", "completed"),
        ("completed", "failed", "failed"),
        ("failed", "success", "failed"),
        ("completed", "timeout", "timeout"),
        ("timeout", "success", "timeout"),
        ("timeout", "failed", "timeout"),
        ("failed", "timeout", "failed"),
    ],
)
def test_trajectory_status_covers_agent_and_evaluator(
    agent_status: str,
    eval_status: str,
    expected: str,
) -> None:
    assert _trajectory_status(agent_status, eval_status) == expected


def test_run_metadata_records_cli_version() -> None:
    agent = AgentSpec(
        id="antigravity-cli",
        class_="antigravity_cli",
        config={"model": "Gemini 3.7 Flash (High)", "cli_version": "1.1.25"},
    )
    unit = RunUnit(
        agent_id=agent.id,
        agent_spec=agent,
        task_path="demo/seecheck",
        variant_index=0,
    )

    metadata = _build_run_meta(
        run_id="run-1",
        unit=unit,
        config=AntigravityCliConfig(),
        executor_type="sandbox",
        status="completed",
        score=1.0,
        phase=None,
        error_obj=None,
        error_str=None,
        total_s=1.0,
        trajectory=None,
        category=None,
    )

    assert metadata["agent"]["version"] == "1.1.25"


@pytest.mark.asyncio
async def test_windows_prewarm_places_global_flag_before_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AntigravityCliConfig()
    executor = SimpleNamespace(config=config)
    deployer = AntigravityCliDeployer(executor)
    deployer._agy_path = r"C:\agy\agy.exe"
    deployer._gemini_dir = r"C:\Users\ale\.gemini"
    captured: dict[str, object] = {}
    monkeypatch.setenv("ANTIGRAVITY_OAUTH_TOKEN", "secret-token-json")
    monkeypatch.setenv("ANTIGRAVITY_OAUTH_TOKEN_PATH", "/host/secret/token")
    monkeypatch.setenv("ANTIGRAVITY_GOOGLE_ACCOUNTS", "secret-account-json")

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)
    await deployer._prewarm_agy()

    assert captured["argv"] == [
        r"C:\agy\agy.exe",
        r"--gemini_dir=C:\Users\ale\.gemini",
        "models",
    ]
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert "ANTIGRAVITY_OAUTH_TOKEN" not in child_env
    assert "ANTIGRAVITY_OAUTH_TOKEN_PATH" not in child_env
    assert "ANTIGRAVITY_GOOGLE_ACCOUNTS" not in child_env


@pytest.mark.asyncio
async def test_launch_passes_prompt_as_print_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AntigravityCliConfig(model="")
    sandbox = SimpleNamespace(task_data_root="", is_linux=True)
    executor = SimpleNamespace(
        work_dir=str(tmp_path),
        env={
            "ANTIGRAVITY_OAUTH_TOKEN": "secret-token-json",
            "ANTIGRAVITY_OAUTH_TOKEN_PATH": "/host/secret/token",
            "ANTIGRAVITY_GOOGLE_ACCOUNTS": "secret-account-json",
        },
        sandbox=sandbox,
        config=config,
    )
    deployer = AntigravityCliDeployer(executor)
    deployer._agy_path = "/fake/agy"
    deployer._gemini_dir = "/fake/.gemini"
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 123
        returncode = 0

        def poll(self) -> int:
            return 0

    def fake_spawn(argv: list[str], **kwargs: object) -> FakeProcess:
        captured["argv"] = argv
        captured.update(kwargs)
        transcript = kwargs["transcript_file"]
        assert isinstance(transcript, Path)
        transcript.write_text(
            json.dumps({"event": "result", "result": {"status": "SUCCESS"}}) + "\n",
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr("ale_run.agents.antigravity_cli.deployer._spawn_agy", fake_spawn)
    result = await deployer.launch("full task instruction")

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert argv[1] == "--print=full task instruction"
    assert "-p" not in argv
    assert f"--log-file={tmp_path / 'agy_cli.log'}" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert "ANTIGRAVITY_OAUTH_TOKEN" not in child_env
    assert "ANTIGRAVITY_OAUTH_TOKEN_PATH" not in child_env
    assert "ANTIGRAVITY_GOOGLE_ACCOUNTS" not in child_env
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_launch_treats_native_error_as_failed_even_with_zero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AntigravityCliConfig(model="")
    sandbox = SimpleNamespace(task_data_root="", is_linux=True)
    executor = SimpleNamespace(work_dir=str(tmp_path), env={}, sandbox=sandbox, config=config)
    deployer = AntigravityCliDeployer(executor)
    deployer._agy_path = "/fake/agy"
    deployer._gemini_dir = "/fake/.gemini"

    class FakeProcess:
        pid = 456
        returncode = 0

        def poll(self) -> int:
            return 0

    def fake_spawn(argv: list[str], **kwargs: object) -> FakeProcess:
        transcript = kwargs["transcript_file"]
        assert isinstance(transcript, Path)
        transcript.write_text(
            json.dumps(
                {
                    "event": "result",
                    "result": {
                        "status": "ERROR",
                        "error": "Individual quota reached",
                        "usage": {},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr("ale_run.agents.antigravity_cli.deployer._spawn_agy", fake_spawn)
    result = await deployer.launch("task")

    assert result.status == "failed"
    assert result.exit_code == 0
    assert result.error == "agy result status=ERROR: Individual quota reached"


@pytest.mark.asyncio
async def test_launch_rejects_zero_exit_without_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AntigravityCliConfig(model="")
    sandbox = SimpleNamespace(task_data_root="", is_linux=True)
    executor = SimpleNamespace(work_dir=str(tmp_path), env={}, sandbox=sandbox, config=config)
    deployer = AntigravityCliDeployer(executor)
    deployer._agy_path = "/fake/agy"
    deployer._gemini_dir = "/fake/.gemini"

    class FakeProcess:
        pid = 789
        returncode = 0

        def poll(self) -> int:
            return 0

    def fake_spawn(argv: list[str], **kwargs: object) -> FakeProcess:
        transcript = kwargs["transcript_file"]
        assert isinstance(transcript, Path)
        transcript.write_text(
            json.dumps({"event": "init", "init": {"model": "Model"}}) + "\n",
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr("ale_run.agents.antigravity_cli.deployer._spawn_agy", fake_spawn)
    result = await deployer.launch("task")

    assert result.status == "failed"
    assert result.exit_code == 0
    assert result.error == "agy exited without a terminal SUCCESS result"


def test_partial_stream_without_result_retains_response_and_tool(tmp_path: Path) -> None:
    events = [
        {"event": "init", "init": {"model": "Observed Model"}},
        {
            "event": "step_update",
            "step_update": {
                "step_index": 1,
                "state": "ACTIVE",
                "step_type": "agent_response",
                "text_delta": "partial response",
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 2,
                "state": "ACTIVE",
                "step_type": "tool",
                "tool_name": "run_command",
                "tool_info": {"parameters": {"CommandLine": "long-running-task"}},
            },
        },
    ]
    (tmp_path / "transcript.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    builder = TrajectoryBuilder(
        agent_name="antigravity-cli", task_path="demo/interrupted", variant_index=0
    )

    AntigravityCliDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=AntigravityCliConfig(),
        run_result=AgentRunResult(status="timeout", exit_code=None),
        builder=builder,
    )

    assert builder.trajectory.steps[0].message == "partial response"
    assert builder.trajectory.steps[0].extra["incomplete"] is True
    assert builder.trajectory.steps[1].tool_calls[0].name == "run_command"
    assert builder.trajectory.steps[1].extra["incomplete"] is True
    assert all(step.observation is None for step in builder.trajectory.steps)
