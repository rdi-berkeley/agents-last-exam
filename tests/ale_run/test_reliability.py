from __future__ import annotations

import json
from pathlib import Path

import pytest

from ale_run.orchestration.config_loader import _build_reliability
from ale_run.orchestration.experiment_spec import (
    AgentSpec,
    EnvironmentSpec,
    ExperimentSpec,
    OutputSpec,
    ReliabilitySpec,
    RunUnit,
    UnitResult,
)
from ale_run.orchestration.reliability import (
    TrialObservation,
    anytime_hoeffding_interval,
    pass_k_estimate,
    summarize_trials,
)
from ale_run.orchestration.run_writer import RunWriter
from ale_run.orchestration.runner import Runner


def _unit() -> RunUnit:
    agent = AgentSpec(id="dummy", class_="dummy", config={"model": "test"})
    return RunUnit(
        agent_id=agent.id,
        agent_spec=agent,
        task_path="demo/hello",
        variant_index=0,
    )


def _spec(tmp_path: Path, reliability: ReliabilitySpec) -> ExperimentSpec:
    return ExperimentSpec(
        name="reliability-test",
        output=OutputSpec(root=str(tmp_path)),
        environment=EnvironmentSpec(),
        agents=[],
        tasks=[],
        reliability=reliability,
        auto_resume=True,
    )


def test_reliability_loader_defaults_to_disabled() -> None:
    spec = _build_reliability(None)

    assert spec == ReliabilitySpec()
    assert spec.enabled is False


def test_reliability_loader_accepts_sequential_policy() -> None:
    spec = _build_reliability(
        {
            "min_trials": 3,
            "max_trials": 10,
            "confidence": 0.95,
            "target_half_width": 0.2,
            "pass_threshold": 0.8,
            "pass_k": [1, 2, 3],
        }
    )

    assert spec.enabled is True
    assert spec.min_trials == 3
    assert spec.max_trials == 10
    assert spec.target_half_width == 0.2
    assert spec.pass_k == (1, 2, 3)


@pytest.mark.parametrize(
    ("raw", "error_type", "match"),
    [
        ({"min_trials": 0}, ValueError, "min_trials"),
        ({"min_trials": 4, "max_trials": 3}, ValueError, "cannot exceed"),
        ({"confidence": 1.0}, ValueError, "confidence"),
        ({"target_half_width": 0.0}, ValueError, "target_half_width"),
        ({"pass_threshold": 1.1}, ValueError, "pass_threshold"),
        ({"pass_k": [1, 0]}, ValueError, "pass_k"),
    ],
)
def test_reliability_loader_rejects_invalid_policy(
    raw: dict,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        _build_reliability(raw)


def test_anytime_interval_is_bounded_and_nested_around_mean() -> None:
    scores = [0.2, 0.4, 0.8, 1.0]
    lo, hi, half_width = anytime_hoeffding_interval(scores, confidence=0.95)

    mean = sum(scores) / len(scores)
    assert 0.0 <= lo <= mean <= hi <= 1.0
    assert 0.0 < half_width <= 1.0


def test_pass_k_estimator_matches_combinatorial_definition() -> None:
    assert pass_k_estimate(successes=4, total=5, k=2) == pytest.approx(0.6)
    assert pass_k_estimate(successes=1, total=5, k=2) == 0.0
    assert pass_k_estimate(successes=1, total=1, k=2) is None


def test_summary_separates_timeout_from_scored_trials() -> None:
    summary = summarize_trials(
        [
            TrialObservation(0, "completed", 1.0),
            TrialObservation(1, "timeout", None),
            TrialObservation(2, "completed", 0.5),
        ],
        confidence=0.95,
        pass_threshold=0.8,
        pass_k=(1, 2),
    )

    assert summary["terminal_trials"] == 3
    assert summary["scored_trials"] == 2
    assert summary["non_scored_terminal_trials"] == 1
    assert summary["mean_score"] == pytest.approx(0.75)
    assert summary["completion_rate"] == pytest.approx(2 / 3)
    assert summary["pass_reliability"]["successes"] == 1
    assert summary["pass_reliability"]["trials"] == 3
    assert summary["pass_reliability"]["pass_rate"] == pytest.approx(1 / 3)
    assert summary["pass_reliability"]["pass_k"]["1"] == pytest.approx(1 / 3)


def test_run_writer_puts_explicit_trials_in_separate_directories(tmp_path: Path) -> None:
    writer = RunWriter(
        output_root=tmp_path,
        agent_id="dummy",
        model="test",
        task_path="demo/hello",
        variant_index=0,
        trial_index=2,
    )
    writer.close()

    assert writer.run_dir.parent.name == "trial_003"
    assert "__trial3__" in writer.run_id


@pytest.mark.asyncio
async def test_runner_executes_fixed_independent_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int | None] = []
    scores = [0.2, 0.4, 0.6]

    async def fake_run_one_unit(**kwargs) -> UnitResult:
        unit = kwargs["unit"]
        seen.append(unit.trial_index)
        score = scores[unit.trial_index]
        return UnitResult(
            unit=unit,
            status="completed",
            score=score,
            run_dir=tmp_path / f"trial-{unit.trial_index}",
        )

    monkeypatch.setattr(
        "ale_run.orchestration.lifecycle.run_one_unit",
        fake_run_one_unit,
    )

    runner = Runner(
        _spec(
            tmp_path,
            ReliabilitySpec(min_trials=3, max_trials=3),
        )
    )
    results = await runner.run([_unit()])

    assert seen == [0, 1, 2]
    assert [result.score for result in results] == scores

    summary_path = runner.output_root / "dummy" / "test" / "demo__hello" / "v0" / "reliability.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["terminal_trials"] == 3
    assert payload["scored_trials"] == 3
    assert payload["mean_score"] == pytest.approx(0.4)
    assert payload["stopping"]["stop_reason"] == "max_trials"


@pytest.mark.asyncio
async def test_runner_stops_after_requested_anytime_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int | None] = []

    async def fake_run_one_unit(**kwargs) -> UnitResult:
        unit = kwargs["unit"]
        seen.append(unit.trial_index)
        return UnitResult(unit=unit, status="completed", score=1.0)

    monkeypatch.setattr(
        "ale_run.orchestration.lifecycle.run_one_unit",
        fake_run_one_unit,
    )

    runner = Runner(
        _spec(
            tmp_path,
            ReliabilitySpec(
                min_trials=2,
                max_trials=5,
                target_half_width=1.0,
            ),
        )
    )
    await runner.run([_unit()])

    assert seen == [0, 1]


@pytest.mark.asyncio
async def test_runner_retries_infrastructure_failure_inside_same_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int | None] = []
    calls = 0

    async def fake_run_one_unit(**kwargs) -> UnitResult:
        nonlocal calls
        unit = kwargs["unit"]
        seen.append(unit.trial_index)
        calls += 1
        if calls == 1:
            return UnitResult(unit=unit, status="failed")
        return UnitResult(unit=unit, status="completed", score=0.5)

    monkeypatch.setattr(
        "ale_run.orchestration.lifecycle.run_one_unit",
        fake_run_one_unit,
    )

    runner = Runner(
        _spec(
            tmp_path,
            ReliabilitySpec(min_trials=2, max_trials=2),
        )
    )
    await runner.run([_unit()], max_attempts=2)

    assert seen == [0, 0, 1]


@pytest.mark.asyncio
async def test_runner_resumes_completed_trials_without_remeasuring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = Runner(
        _spec(
            tmp_path,
            ReliabilitySpec(min_trials=3, max_trials=3),
        )
    )
    variant_dir = runner.output_root / "dummy" / "test" / "demo__hello" / "v0"
    prior_dir = variant_dir / "trial_001" / "20260930_000000"
    prior_dir.mkdir(parents=True)
    (prior_dir / "run.json").write_text(
        json.dumps({"status": "completed", "score": 0.25}),
        encoding="utf-8",
    )

    seen: list[int | None] = []

    async def fake_run_one_unit(**kwargs) -> UnitResult:
        unit = kwargs["unit"]
        seen.append(unit.trial_index)
        return UnitResult(unit=unit, status="completed", score=0.75)

    monkeypatch.setattr(
        "ale_run.orchestration.lifecycle.run_one_unit",
        fake_run_one_unit,
    )

    await runner.run([_unit()])

    assert seen == [1, 2]
