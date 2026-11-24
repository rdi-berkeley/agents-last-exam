"""The CLI and the ALE executor share the same CUA session runner."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from ale_claw import AleClawConfig, RunResult, cli, runner

from ale_run.agents.ale_claw.deployer import AleClawDeployer


@pytest.mark.parametrize(
    "options, endpoint",
    [
        ({}, "http://127.0.0.1:8000"),
        ({"sandbox_url": "http://desktop:5000", "os_type": "windows"}, "http://desktop:5000"),
    ],
)
async def test_runner_connects_to_selected_sandbox(monkeypatch, tmp_path, options, endpoint):
    session = AsyncMock()
    factory = Mock(return_value=session)
    execute = AsyncMock(return_value=RunResult("completed", 1, "transcript.jsonl"))
    monkeypatch.setattr(runner, "RemoteDesktopSession", factory)
    monkeypatch.setattr(runner, "_run_session", execute)
    config = AleClawConfig()
    result = await runner.run("goal", config=config, output_dir=tmp_path, **options)
    os_type = options.get("os_type", "linux")
    factory.assert_called_once_with(
        api_url=endpoint, os_type=os_type, ephemeral=False, headless=True
    )
    execute.assert_awaited_once_with("goal", config, session, os_type, tmp_path)
    session.close.assert_awaited_once()
    assert result is execute.return_value


async def test_missing_cua_stops_before_agent(monkeypatch):
    session = AsyncMock()
    session.check_status.return_value = False
    monkeypatch.setattr(runner, "RemoteDesktopSession", Mock(return_value=session))
    execute = AsyncMock()
    monkeypatch.setattr(runner, "_run_session", execute)
    with pytest.raises(RuntimeError, match="127.0.0.1:8000"):
        await runner.run("goal", config=AleClawConfig())
    execute.assert_not_awaited()
    session.close.assert_awaited_once()


async def test_runs_in_same_output_directory_start_independent_sessions(monkeypatch, tmp_path):
    prompts = []

    async def record_turn(self, prompt, **kwargs):
        prompts.append(prompt)
        self.session_mgr.append_message("user", prompt)
        yield {"output": [{"type": "message", "content": "DONE"}]}

    monkeypatch.setattr(runner, "RemoteDesktopSession", Mock(return_value=AsyncMock()))
    monkeypatch.setattr(runner.OpenClawComputerAgent, "run", record_turn)
    config = AleClawConfig(disable_main_computer=True)
    paths = []
    for prompt in ("First task", "Second task"):
        result = await runner.run(prompt, config=config, output_dir=tmp_path)
        assert result.status == "completed"
        paths.append(Path(result.transcript_path))

    assert prompts == ["First task", "Second task"]
    assert paths[0].parent != paths[1].parent
    for path, prompt in zip(paths, prompts, strict=True):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        contents = [e["message"]["content"] for e in events if e["type"] == "message"]
        assert contents == [[{"type": "text", "text": prompt}]]


def test_cli_overrides_config_and_returns_failure_status(monkeypatch, tmp_path, capsys):
    config = tmp_path / "settings.toml"
    config.write_text('sandbox_url="http://desktop:5000"\n[agent]\nmodel="old"\nmax_turns=2\n')
    async def run_with_progress(*args, **kwargs):
        print("Running the agent")
        return RunResult("failed", 1, None, "turn limit")

    execute = AsyncMock(side_effect=run_with_progress)
    monkeypatch.setattr(runner, "run", execute)
    monkeypatch.setattr(
        sys, "argv", ["ale-claw", "--config", str(config), "--model", "new", "goal"]
    )
    assert cli.main() == 1
    streams = capsys.readouterr()
    assert streams.err == "Running the agent\n"
    assert json.loads(streams.out)["error"] == "turn limit"
    execute.assert_awaited_once_with(
        "goal",
        config=AleClawConfig(model="new", max_turns=2),
        sandbox_url="http://desktop:5000",
    )


async def test_deployer_forwards_benchmark_environment(monkeypatch, tmp_path):
    executor = SimpleNamespace(
        config=AleClawConfig(),
        sandbox=SimpleNamespace(endpoint="http://desktop:5000", os="windows"),
        work_dir=str(tmp_path),
    )
    execute = AsyncMock(return_value=RunResult("failed", 2.5, "transcript.jsonl", "turn limit"))
    monkeypatch.setattr("ale_run.agents.ale_claw.deployer.run", execute)
    for key in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    agent = AleClawDeployer(executor)
    await agent.install()
    result = await agent.launch("goal")
    execute.assert_awaited_once_with(
        "goal",
        config=executor.config,
        sandbox_url=executor.sandbox.endpoint,
        os_type="windows",
        output_dir=str(tmp_path),
    )
    assert (result.status, result.duration_s, result.transcript_path, result.error) == (
        "failed",
        2.5,
        "transcript.jsonl",
        "turn limit",
    )
