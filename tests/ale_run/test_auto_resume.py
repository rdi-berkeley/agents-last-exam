from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from ale_run import cli
from ale_run.orchestration.config_loader import load_experiment
from ale_run.orchestration.experiment_spec import (
    AgentSpec,
    EnvironmentSpec,
    ExperimentSpec,
    OutputSpec,
    RunUnit,
    UnitResult,
)
from ale_run.orchestration.run_writer import RunWriter
from ale_run.orchestration.runner import Runner


def _spec(
    tmp_path: Path,
    *,
    auto_resume: bool = True,
    max_attempts: int = 3,
) -> ExperimentSpec:
    return ExperimentSpec(
        name="auto-resume-test",
        output=OutputSpec(root=str(tmp_path)),
        environment=EnvironmentSpec(),
        agents=[],
        tasks=[],
        auto_resume=auto_resume,
        max_attempts=max_attempts,
    )


def _unit() -> RunUnit:
    agent = AgentSpec(id="dummy", class_="dummy", config={"model": "test"})
    return RunUnit(
        agent_id=agent.id,
        agent_spec=agent,
        task_path="demo/hello",
        variant_index=0,
    )


def _write_experiment(tmp_path: Path, run_config: str = "") -> Path:
    (tmp_path / "agent.yaml").write_text(
        "harness: dummy\nmodel: test\n",
        encoding="utf-8",
    )
    (tmp_path / "environment.yaml").write_text(
        "provider: static\nendpoint: http://127.0.0.1:5000\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "name: auto-resume-test\n"
        "agent: agent.yaml\n"
        "environment: environment.yaml\n"
        "tasks:\n"
        "  - path: demo/hello\n"
        f"{run_config}",
        encoding="utf-8",
    )
    return experiment


def test_loader_enables_auto_resume_by_default(tmp_path: Path) -> None:
    spec = load_experiment(_write_experiment(tmp_path))

    assert spec.auto_resume is True
    assert spec.max_attempts == 3


def test_loader_accepts_auto_resume_config(tmp_path: Path) -> None:
    spec = load_experiment(
        _write_experiment(
            tmp_path,
            "auto_resume: false\nmax_attempts: 2\n",
        )
    )

    assert spec.auto_resume is False
    assert spec.max_attempts == 2


@pytest.mark.parametrize(
    ("run_config", "error_type", "match"),
    [
        ("auto_resume: 'yes'\n", TypeError, "auto_resume must be a boolean"),
        ("max_attempts: true\n", TypeError, "max_attempts must be an integer"),
        ("max_attempts: 4\n", ValueError, "max_attempts must be between 1 and 3"),
    ],
)
def test_loader_rejects_invalid_auto_resume_config(
    tmp_path: Path,
    run_config: str,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        load_experiment(_write_experiment(tmp_path, run_config))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("statuses", "expected_calls", "expected_status"),
    [
        (["failed", "failed", "completed"], 3, "completed"),
        (["failed", "failed", "failed", "completed"], 3, "failed"),
        (["timeout", "completed"], 1, "timeout"),
    ],
)
async def test_runner_auto_resume_retries_only_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[str],
    expected_calls: int,
    expected_status: str,
) -> None:
    calls = 0

    async def fake_run_one_unit(**kwargs) -> UnitResult:
        nonlocal calls
        status = statuses[calls]
        calls += 1
        return UnitResult(unit=kwargs["unit"], status=status)

    monkeypatch.setattr(
        "ale_run.orchestration.lifecycle.run_one_unit",
        fake_run_one_unit,
    )

    results = await Runner(_spec(tmp_path)).run([_unit()], max_attempts=3)

    assert calls == expected_calls
    assert len(results) == 1
    assert results[0].status == expected_status


@pytest.mark.asyncio
async def test_runner_default_does_not_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_run_one_unit(**kwargs) -> UnitResult:
        nonlocal calls
        calls += 1
        return UnitResult(unit=kwargs["unit"], status="failed")

    monkeypatch.setattr(
        "ale_run.orchestration.lifecycle.run_one_unit",
        fake_run_one_unit,
    )

    results = await Runner(_spec(tmp_path)).run([_unit()])

    assert calls == 1
    assert results[0].status == "failed"


def test_disable_resume_cli_flag_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = None

    async def fake_cmd_run(args) -> int:
        nonlocal captured
        captured = args
        return 0

    monkeypatch.setattr(cli, "_cmd_run", fake_cmd_run)

    assert cli.main(["run", "experiment.yaml", "--disable-resume"]) == 0
    assert captured is not None
    assert captured.disable_resume is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("yaml_enabled", "cli_disabled", "expected_attempts", "filters_prior"),
    [
        (True, False, 2, True),
        (False, False, 1, False),
        (True, True, 1, False),
    ],
)
async def test_auto_resume_config_and_cli_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    yaml_enabled: bool,
    cli_disabled: bool,
    expected_attempts: int,
    filters_prior: bool,
) -> None:
    unit = _unit()
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, spec: ExperimentSpec):
            self.output_root = tmp_path

        def enumerate_units(self) -> list[RunUnit]:
            return [unit]

        async def run(
            self,
            units: list[RunUnit],
            *,
            max_attempts: int,
        ) -> list[UnitResult]:
            captured["units"] = units
            captured["max_attempts"] = max_attempts
            return [UnitResult(unit=unit, status="completed")]

    def fake_filter_resume(
        units: list[RunUnit],
        output_root: Path,
    ) -> list[RunUnit]:
        captured["resume_output_root"] = output_root
        return units

    monkeypatch.setattr(
        cli,
        "load_experiment",
        lambda _: _spec(tmp_path, auto_resume=yaml_enabled, max_attempts=2),
    )
    monkeypatch.setattr(cli, "Runner", FakeRunner)
    monkeypatch.setattr(cli, "_filter_resume", fake_filter_resume)
    monkeypatch.setattr(cli, "_print_results_table", lambda _: None)

    exit_code = await cli._cmd_run(
        argparse.Namespace(
            spec_path=Path("experiment.yaml"),
            filter_agents=None,
            filter_tasks=None,
            disable_resume=cli_disabled,
            dry_run=False,
        )
    )

    assert exit_code == 0
    assert captured["units"] == [unit]
    assert captured["max_attempts"] == expected_attempts
    if filters_prior:
        assert captured["resume_output_root"] == tmp_path
    else:
        assert "resume_output_root" not in captured


def test_run_writer_allocates_unique_same_second_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ale_run.orchestration.run_writer.time.strftime", lambda *_: "fixed")

    first = RunWriter(
        output_root=tmp_path,
        agent_id="dummy",
        model="test",
        task_path="demo/hello",
        variant_index=0,
    )
    second = RunWriter(
        output_root=tmp_path,
        agent_id="dummy",
        model="test",
        task_path="demo/hello",
        variant_index=0,
    )
    first.close()
    second.close()

    assert first.run_dir.name == "fixed"
    assert second.run_dir.name == "fixed_01"
    assert first.run_id != second.run_id
