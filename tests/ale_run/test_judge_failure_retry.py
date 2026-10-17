import json
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ale_run import cli
from ale_run.base_interface import AgentRunResult, GatherReport
from ale_run.orchestration import lifecycle
from ale_run.orchestration.experiment_spec import (
    AgentSpec,
    EnvironmentSpec,
    ExperimentSpec,
    OutputSpec,
    RunUnit,
)
from ale_run.orchestration.runner import Runner
from tasks.utils.evaluation import JudgeInfrastructureError


@dataclass
class JudgeTestConfig:
    model: str = "test-model"
    name: str = "judge-test"


@pytest.fixture
def episode(tmp_path, monkeypatch):
    sandbox = SimpleNamespace(id="fake", metadata={}, os="linux")
    environment = SimpleNamespace(
        sandbox=sandbox,
        session=None,
        current_phase=None,
        reset_async=AsyncMock(),
        close_async=AsyncMock(),
        reset_session=Mock(),
    )
    environment.set_phase = lambda phase: setattr(environment, "current_phase", phase)
    executor = SimpleNamespace(
        work_dir=str(tmp_path / "work"),
        run_deployer=AsyncMock(),
        stop_deployer=AsyncMock(),
        gather_dir=AsyncMock(return_value=GatherReport(transport="local")),
    )
    driver = SimpleNamespace(setup=AsyncMock(), close=AsyncMock(), evaluate=AsyncMock())
    deployer = SimpleNamespace(parse_artifacts=Mock())
    unit = RunUnit(
        "judge-test",
        AgentSpec("judge-test", "dummy", config={"model": "test-model"}, executor="local"),
        "demo/hello",
        0,
    )
    monkeypatch.setattr(lifecycle, "resolve_agent", lambda spec: (deployer, JudgeTestConfig))
    monkeypatch.setattr(lifecycle, "task_revision", lambda path: "test-revision")
    monkeypatch.setattr(cli, "task_revision", lambda path: "test-revision")
    monkeypatch.setattr(lifecycle, "ALEEnv", lambda **kwargs: environment)
    monkeypatch.setattr(
        lifecycle,
        "TaskLoader",
        lambda path: SimpleNamespace(
            load=lambda variant: {"description": "Test judge failure", "snapshot_name": "fake"}
        ),
    )
    monkeypatch.setattr(lifecycle, "TaskDriver", lambda **kwargs: driver)
    monkeypatch.setattr(lifecycle, "_build_executor", lambda **kwargs: executor)
    monkeypatch.setattr(lifecycle, "_stage_task_data", AsyncMock())
    monkeypatch.setattr(lifecycle, "pull_agent_output", AsyncMock())
    monkeypatch.setattr(lifecycle, "stage_reference", AsyncMock())
    monkeypatch.setattr(lifecycle, "install_signal_handlers", Mock())
    monkeypatch.setattr("ale_run.orchestration.runner.check_judges", AsyncMock())
    monkeypatch.setattr("ale_run.orchestration.host_preflight.check_host_dependencies", Mock())
    runner = Runner(ExperimentSpec(
        name="judge-failure-test",
        output=OutputSpec(root=str(tmp_path)),
        environment=EnvironmentSpec(),
        agents=[],
        tasks=[],
    ))
    runner._router = SimpleNamespace(provider_for=lambda snapshot: None)
    return SimpleNamespace(
        unit=unit, runner=runner, executor=executor, driver=driver, environment=environment
    )


@pytest.mark.parametrize("agent_status", ["completed", "timeout"])
@pytest.mark.parametrize("failure", ["exception", "error-result", "no-result"])
async def test_judge_failure_is_failed_unscored_and_not_resume_skippable(
    episode, agent_status, failure
):
    episode.executor.run_deployer.return_value = AgentRunResult(
        status=agent_status, error="agent budget exhausted" if agent_status == "timeout" else None
    )
    if failure == "exception":
        episode.driver.evaluate.side_effect = JudgeInfrastructureError("judge unavailable")
    else:
        episode.driver.evaluate.return_value = (
            {"score": 1.0, "error": "judge unavailable"} if failure == "error-result" else None
        )

    result = (await episode.runner.run([episode.unit]))[0]

    assert result.status == "failed"
    assert result.eval_status == "failed"
    assert result.score is None
    assert result.error != "agent budget exhausted"
    metadata = json.loads((result.run_dir / "run.json").read_text())
    evaluation = json.loads((result.run_dir / "eval_result.json").read_text())
    trajectory = json.loads((result.run_dir / "trajectory.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["score"] is None
    assert metadata["termination"]["phase"] == "evaluation"
    assert metadata["termination"]["error"] == evaluation["error"]
    assert evaluation["eval_status"] == "failed"
    assert evaluation["score"] is None
    assert trajectory["final_metrics"]["status"] == "failed"
    assert trajectory["final_metrics"]["reward"] is None
    assert not cli._unit_already_done(episode.unit, episode.runner.output_root)
    episode.environment.close_async.assert_awaited_once_with(mode="delete")


@pytest.mark.parametrize("agent_status", ["completed", "timeout"])
async def test_runner_retries_actual_lifecycle_after_judge_failure(episode, agent_status):
    episode.executor.run_deployer.return_value = AgentRunResult(status=agent_status)
    episode.driver.evaluate.side_effect = JudgeInfrastructureError("judge unavailable")

    result = (await episode.runner.run([episode.unit], max_attempts=2))[0]

    assert result.status == "failed"
    assert result.score is None
    assert episode.executor.run_deployer.await_count == 2
    assert episode.driver.evaluate.await_count == 2
    records = list(episode.runner.output_root.rglob("run.json"))
    assert len(records) == 2
    assert all(json.loads(path.read_text())["status"] == "failed" for path in records)
    assert not cli._unit_already_done(episode.unit, episode.runner.output_root)


@pytest.mark.parametrize("score", [0.0, 0.75, 1.0])
async def test_valid_scored_agent_timeout_remains_timeout_without_retry(episode, score):
    episode.executor.run_deployer.return_value = AgentRunResult(
        status="timeout", error="agent budget exhausted"
    )
    episode.driver.evaluate.return_value = {"score": score}

    result = (await episode.runner.run([episode.unit], max_attempts=2))[0]

    assert result.status == "timeout"
    assert result.eval_status == "success"
    assert result.score == score
    assert result.error == "agent budget exhausted"
    metadata = json.loads((result.run_dir / "run.json").read_text())
    trajectory = json.loads((result.run_dir / "trajectory.json").read_text())
    assert metadata["status"] == "timeout"
    assert metadata["score"] == score
    assert trajectory["final_metrics"]["status"] == "timeout"
    assert trajectory["final_metrics"]["reward"] == score
    assert episode.executor.run_deployer.await_count == 1
    assert cli._unit_already_done(episode.unit, episode.runner.output_root)


@pytest.mark.parametrize("agent_status", ["completed", "timeout"])
async def test_evaluator_timeout_semantics_are_unchanged(episode, agent_status):
    episode.executor.run_deployer.return_value = AgentRunResult(status=agent_status)
    episode.driver.evaluate.side_effect = TimeoutError()

    result = (await episode.runner.run([episode.unit], max_attempts=2))[0]

    assert result.status == "timeout"
    assert result.eval_status == "timeout"
    assert episode.executor.run_deployer.await_count == 1
    assert result.score is None
    assert not cli._unit_already_done(episode.unit, episode.runner.output_root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("eval_status", "failed"),
        ("error", "Judge HTTP 401"),
        ("score", None),
        ("score", True),
        ("score", 0.5),
    ],
)
async def test_resume_requires_consistent_successful_evaluation(episode, field, value):
    episode.executor.run_deployer.return_value = AgentRunResult(status="completed")
    episode.driver.evaluate.return_value = {"score": 1.0}
    result = (await episode.runner.run([episode.unit]))[0]
    assert cli._unit_already_done(episode.unit, episode.runner.output_root)
    path = result.run_dir / "eval_result.json"
    evaluation = json.loads(path.read_text())
    evaluation[field] = value
    path.write_text(json.dumps(evaluation))
    assert not cli._unit_already_done(episode.unit, episode.runner.output_root)


@pytest.mark.parametrize("score", [None, True, float("nan"), float("inf"), -0.1, 1.1])
async def test_resume_rejects_invalid_record_score(episode, score):
    episode.executor.run_deployer.return_value = AgentRunResult(status="completed")
    episode.driver.evaluate.return_value = {"score": 1.0}
    result = (await episode.runner.run([episode.unit]))[0]
    path = result.run_dir / "run.json"
    record = json.loads(path.read_text())
    record["score"] = score
    path.write_text(json.dumps(record))
    assert not cli._unit_already_done(episode.unit, episode.runner.output_root)
