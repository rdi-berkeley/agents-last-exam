from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Self

import pytest

from ale_run.agents.deepseek_harness import driver
from ale_run.agents.deepseek_harness.config import DeepSeekHarnessConfig
from ale_run.agents.deepseek_harness.deployer import DeepSeekHarnessDeployer
from ale_run.base_interface import AgentRunResult, TrajectoryBuilder
from ale_run.orchestration.factory import AGENT_REGISTRY


def _executor(tmp_path: Path, config: DeepSeekHarnessConfig) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        env={config.api_key_env: "test-key"},
        work_dir=str(tmp_path),
        sandbox=SimpleNamespace(is_linux=True),
    )


def _builder() -> TrajectoryBuilder:
    return TrajectoryBuilder(
        agent_name="deepseek-harness",
        model="deepseek-v4-flash",
        task_path="demo/code-task",
        variant_index=0,
    )


def test_deepseek_harness_is_registered() -> None:
    assert AGENT_REGISTRY["deepseek_harness"] is DeepSeekHarnessDeployer


def test_config_defaults_match_bundled_runtime() -> None:
    config = DeepSeekHarnessConfig()

    assert config.model == "deepseek-v4-flash"
    assert config.provider == "deepseek-official"
    assert config.sdk_version == "0.1.0rc6"


def test_unattended_permissions_are_not_public_config_fields() -> None:
    config_fields = {field.name for field in fields(DeepSeekHarnessConfig)}

    assert config_fields.isdisjoint({"approval_policy", "permission_mode", "sandbox_mode", "yolo"})


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_config_rejects_nonpositive_max_tokens(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        DeepSeekHarnessConfig(max_tokens=max_tokens)


def test_build_env_isolates_runtime_and_maps_credentials(tmp_path: Path) -> None:
    config = DeepSeekHarnessConfig(
        api_key_env="CUSTOM_DEEPSEEK_KEY",
        base_url="https://gateway.example/v1",
        system_prompt="Work autonomously.",
    )
    executor = _executor(tmp_path, config)
    executor.env.update(
        {
            "DSH_CORDIS_CONFIG": "/untrusted/ambient.yml",
            "DSH_RUNTIME_MODE": "node",
        }
    )
    deployer = DeepSeekHarnessDeployer(executor)

    env = deployer._build_env(config, session_root=tmp_path / "sessions")

    assert env["DEEPSEEK_API_KEY"] == "test-key"
    assert env["DEEPSEEK_BASE_URL"] == "https://gateway.example/v1"
    assert env["DSH_SYSTEM_PROMPT"] == "Work autonomously."
    assert env["DSH_SESSION_ROOT"] == str(tmp_path / "sessions")
    assert env["DSH_TELEMETRY_DISABLED"] == "1"
    assert "DSH_CORDIS_CONFIG" not in env
    assert "DSH_RUNTIME_MODE" not in env


def test_build_env_requires_key_and_clears_ambient_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DeepSeekHarnessConfig(api_key_env="ABSENT_DEEPSEEK_KEY")
    executor = _executor(tmp_path, config)
    executor.env = {}
    deployer = DeepSeekHarnessDeployer(executor)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://ambient.invalid/v1")
    monkeypatch.delenv("ABSENT_DEEPSEEK_KEY", raising=False)

    with pytest.raises(RuntimeError, match="no API key configured"):
        deployer._build_env(config, session_root=tmp_path / "sessions")

    executor.env["ABSENT_DEEPSEEK_KEY"] = "key"
    env = deployer._build_env(config, session_root=tmp_path / "sessions")
    assert "DEEPSEEK_BASE_URL" not in env


def test_build_argv_uses_driver_without_putting_key_on_command_line(tmp_path: Path) -> None:
    config = DeepSeekHarnessConfig(api_key="secret", max_tokens=8192)

    argv = DeepSeekHarnessDeployer._build_argv(
        config,
        prompt_file=tmp_path / "prompt.txt",
        work_dir=tmp_path,
        session_root=tmp_path / "sessions",
    )

    assert argv[:3] == [
        sys.executable,
        "-m",
        "ale_run.agents.deepseek_harness.driver",
    ]
    assert argv[-2:] == ["--max-tokens", "8192"]
    assert "secret" not in argv


@pytest.mark.asyncio
async def test_install_bootstraps_pip_and_verifies_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DeepSeekHarnessConfig()
    deployer = DeepSeekHarnessDeployer(_executor(tmp_path, config))
    runtime = tmp_path / "dsh-jsonrpc-agent"
    runtime.write_text("runtime", encoding="utf-8")
    versions = iter([None, None, config.sdk_version, config.sdk_version])
    runtimes = iter([None, runtime])
    calls: list[list[str]] = []
    results = iter(
        [
            subprocess.CompletedProcess([], 1, "", "pip missing"),
            subprocess.CompletedProcess([], 0, "pip installed", ""),
            subprocess.CompletedProcess([], 0, "sdk installed", ""),
        ]
    )

    monkeypatch.setattr(
        "ale_run.agents.deepseek_harness.deployer._installed_version",
        lambda _distribution: next(versions),
    )
    monkeypatch.setattr(
        "ale_run.agents.deepseek_harness.deployer._bundled_runtime_path",
        lambda: next(runtimes),
    )

    def fake_run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return next(results)

    monkeypatch.setattr(subprocess, "run", fake_run)

    await deployer.install()

    assert calls[0][-2:] == ["pip", "--version"]
    assert calls[1][-2:] == ["ensurepip", "--upgrade"]
    assert calls[2][-1] == "deepseek-harness-sdk==0.1.0rc6"


@pytest.mark.asyncio
async def test_install_rejects_windows_before_package_install(tmp_path: Path) -> None:
    config = DeepSeekHarnessConfig()
    executor = _executor(tmp_path, config)
    executor.sandbox.is_linux = False

    with pytest.raises(RuntimeError, match="no Windows wheel"):
        await DeepSeekHarnessDeployer(executor).install()


def test_parse_canonical_session_events(tmp_path: Path) -> None:
    session_id = "session-root"
    records = [
        {
            "type": "notification",
            "method": "session.event",
            "params": {
                "sessionId": session_id,
                "event": {
                    "type": "assistant/message",
                    "seq": 10,
                    "data": {
                        "turn": 1,
                        "step": 1,
                        "message": {
                            "content": [
                                {"type": "reasoning", "text": "Inspect files."},
                                {
                                    "type": "tool-call",
                                    "id": "call-1",
                                    "name": "bash",
                                    "arguments": json.dumps({"command": "pwd"}),
                                },
                            ]
                        },
                        "usage": {
                            "inputTokens": 20,
                            "outputTokens": 5,
                            "cacheReadTokens": 3,
                            "cacheWriteTokens": 2,
                            "reasoningTokens": 4,
                        },
                    },
                },
            },
        },
        {
            "type": "notification",
            "method": "session.event",
            "params": {
                "sessionId": session_id,
                "event": {
                    "type": "tool/result",
                    "seq": 12,
                    "data": {
                        "turn": 1,
                        "step": 1,
                        "message": {
                            "content": [
                                {
                                    "type": "tool-result",
                                    "toolCallId": "call-1",
                                    "content": [{"type": "text", "text": "/workspace"}],
                                    "isError": False,
                                }
                            ]
                        },
                    },
                },
            },
        },
        {
            "type": "notification",
            "method": "subagent.started",
            "params": {
                "parentSessionId": session_id,
                "childSessionId": "session-child",
            },
        },
        {
            "type": "notification",
            "method": "session.event",
            "params": {
                "sessionId": "session-child",
                "event": {
                    "type": "assistant/message",
                    "data": {"content": [{"type": "text", "text": "child only"}]},
                },
            },
        },
        {
            "type": "notification",
            "method": "session.event",
            "params": {
                "sessionId": session_id,
                "event": {
                    "type": "assistant/message",
                    "seq": 20,
                    "data": {
                        "turn": 1,
                        "step": 2,
                        "message": {"content": [{"type": "text", "text": "Done."}]},
                        "usage": {"inputTokens": 30, "outputTokens": 6},
                    },
                },
            },
        },
        {
            "type": "result",
            "session_id": session_id,
            "final_response": "Done.",
            "finish_reason": "completed",
        },
    ]
    (tmp_path / "transcript.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    builder = _builder()

    DeepSeekHarnessDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=DeepSeekHarnessConfig(),
        run_result=AgentRunResult(
            status="completed",
            exit_code=0,
            stderr_path=str(tmp_path / "stderr.log"),
        ),
        builder=builder,
    )
    trajectory = builder.finalize(reward=1.0)

    tool_step, result_step, final_step = trajectory.steps
    assert tool_step.reasoning == "Inspect files."
    assert tool_step.tool_calls[0].name == "bash"
    assert tool_step.tool_calls[0].arguments == {"command": "pwd"}
    assert tool_step.metrics.input_tokens == 20
    assert tool_step.metrics.cache_read_tokens == 3
    assert tool_step.metrics.cache_creation_tokens == 2
    assert result_step.observation.results[0].content[0].text == "/workspace"
    assert final_step.message == "Done."
    assert trajectory.final_metrics.total_input_tokens == 50
    assert trajectory.final_metrics.total_output_tokens == 11
    assert trajectory.extra["deepseek_harness"]["session_id"] == session_id
    assert trajectory.extra["deepseek_harness"]["subagent_count"] == 1
    assert "child only" not in [step.message for step in trajectory.steps]


def test_parse_turn_error_and_missing_transcript(tmp_path: Path) -> None:
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps(
            {
                "type": "notification",
                "method": "session.event",
                "params": {
                    "sessionId": "failed-session",
                    "event": {
                        "type": "turn/end",
                        "seq": 4,
                        "data": {
                            "turn": 1,
                            "reason": {
                                "kind": "error",
                                "error": {
                                    "message": "credential rejected",
                                    "code": "INVALID_CREDENTIAL",
                                },
                            },
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    builder = _builder()

    DeepSeekHarnessDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=DeepSeekHarnessConfig(),
        run_result=AgentRunResult(status="failed", exit_code=1),
        builder=builder,
    )

    assert builder.trajectory.steps[0].message == "credential rejected"
    assert builder.trajectory.steps[0].extra["code"] == "INVALID_CREDENTIAL"

    (tmp_path / "transcript.jsonl").unlink()
    missing_builder = _builder()
    DeepSeekHarnessDeployer.parse_artifacts(
        work_dir=tmp_path,
        config=DeepSeekHarnessConfig(),
        run_result=AgentRunResult(status="failed", exit_code=1),
        builder=missing_builder,
    )
    assert missing_builder.trajectory.steps[0].extra["reason"] == "no_transcript"


def test_driver_streams_notifications_and_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("finish the task", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeHarness:
        def __init__(self, **kwargs: object) -> None:
            captured["kwargs"] = kwargs

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def run(self, prompt: str, *, on_notification: object) -> SimpleNamespace:
            captured["prompt"] = prompt
            on_notification(
                SimpleNamespace(
                    method="session.event",
                    payload={"sessionId": "s1", "event": {"type": "turn/start"}},
                )
            )
            return SimpleNamespace(
                session_id="s1",
                final_response="ok",
                finish_reason="completed",
            )

    module = ModuleType("deepseek_harness")
    module.DeepSeekHarness = FakeHarness
    monkeypatch.setitem(sys.modules, "deepseek_harness", module)

    exit_code = driver.main(
        [
            "--prompt-file",
            str(prompt_file),
            "--cwd",
            str(tmp_path),
            "--session-root",
            str(tmp_path / "sessions"),
            "--provider",
            "deepseek-official",
            "--model",
            "deepseek-v4-flash",
        ]
    )

    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert exit_code == 0
    assert captured["prompt"] == "finish the task"
    assert output[0]["method"] == "session.event"
    assert output[-1] == {
        "type": "result",
        "session_id": "s1",
        "final_response": "ok",
        "finish_reason": "completed",
    }


def test_signal_process_group_targets_driver_group(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)))

    DeepSeekHarnessDeployer._signal_process_group(
        SimpleNamespace(pid=1234),
        signal.SIGTERM,
    )

    assert calls == [(1234, signal.SIGTERM)]
